//! GPU gate for the card kernels of EmbeddingGemma 2's global attention
//! layers, model-less at the model's shapes: 4 query heads over one
//! key/value head of 512 values, a NEOX rope over all 512 values at θ 1e6,
//! RMS eps 1e-6 and an attention scale of 1 (llama.cpp's
//! `f_attention_scale` for the architecture: the query norm stands in for
//! the scale). Inputs are seeded ([`activations`]); nothing is read from a
//! model file.
//!
//! 1. Rope (`rope_neox::head_norm_neox_append_512`), against the shared host
//!    transcription of its rule: the norm (each thread's two squares added in
//!    f64, the warp butterfly, the eight warp sums in warp order, `(sum / 512)
//!    as f32`), `(scale · gain) · x`, the turn of pairs `(i, i + 256)` by the
//!    table's row at the token's position, the query read from its own rows
//!    at head stride 512, the key turned in place, both planes' rows the f16
//!    (nearest even) of the turned key and of the value: all bit for bit, the
//!    planes' other rows untouched, and a rerun bit for bit. Runs use 4/1
//!    and 8/2 query/key heads and end at position `ctx - 1`. A position at the
//!    cache's height raises `cache_pos` with the launch's layer, leaves that
//!    token's heads NaN and appends nothing; every other token is the clean
//!    run's.
//! 2. Prefill flash (`flash_gqa_prefill_512::enqueue_p4`): each output within
//!    its first-order bound of the exact attention computed here in f64 on the
//!    same f16 keys and values (`flash_band`'s exact attention and prefill
//!    bound at a head of 512 and a key tile of 32). The launches: every row
//!    over the whole input (the encoder's counts, `T` ∈ [`ENC_T`]), an
//!    unpadded encoder at `T = ctx = 33`, counts
//!    `p0 + t + 1` ([`SEED_CAUSAL`], crossing the 32-key tile and the
//!    8-position block edges), counts scattered within one block
//!    ([`SCATTER_T`]), one encoder launch at scale [`SCALE_ALT`], and one
//!    causal launch of [`GROUP`] heads (two key heads, two packs of four
//!    query heads each), whose blocks map to every key head and pack; the
//!    output starts NaN, so a value no block writes is out of band. Every row
//!    is bit for bit the same row launched alone, NaN in the cache rows past
//!    the launch's largest count changes no bit, a rerun is bit for bit. Then
//!    a count of zero and one past the cache (`key_count`, NaN in those rows,
//!    the rest the clean run's), the captured launch (one node) the eager
//!    bits, groups that are not a multiple of four query heads a key head
//!    (3 over 1, 4 over 2) refused by the group's error, and windows at
//!    aligned offsets accepted while `q` at 4 bytes past 8 and `kc`/`vc` at 8
//!    bytes past 16 are refused by name.
//! 3. The lower-bound entry ([`FlashGqaPrefill512::enqueue_p4_lower`]): row
//!    `t` attends `[first[t], counts[t])`. A sweep whose row `r` attends
//!    `[r, 72 + r)` — every offset mod 32 within one launch, windows crossing
//!    tiles and blocks — at the model's shape and at [`GROUP`] heads, and a
//!    packed launch, the encoder's own shape, whose inputs of lengths
//!    [`PACKED_LENS`] sit at their packed offsets so every row of one input
//!    shares its window while blocks straddle inputs. Each row against the
//!    exact attention over its own window (the cache cut to it,
//!    `judge_prefill` at `first = lo`), each row bit for bit the same row
//!    launched alone, a rerun and the NaN-padded planes bit for bit, and one
//!    captured launch (one node) the eager bits. Then a first key at its
//!    count, one past it, a count of zero and one past the cache
//!    (`key_count`, NaN in those rows, the rest the clean run's) and a short
//!    first-key buffer refused by name.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_gemma_embedding2_head512", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::fault::{Fault, FaultSink, FaultSite, LAYER_NONE};
    use bloomery_gpu::flash_gqa_prefill::GqaPrefillArgs;
    use bloomery_gpu::flash_gqa_prefill_512::{FlashGqaPrefill512, KEY_TILE_512};
    use bloomery_gpu::rope_neox::{HEAD_512 as HEAD, RopeNeoxKernels};
    use bloomery_gpu::{Gpu, GpuError};
    use bloomery_gpu_gates::flash_band::judge_prefill;
    use bloomery_gpu_gates::prefill_gate::{
        HostCache, WindowCase, Windows, cache, window_refusals,
    };
    use bloomery_gpu_gates::rope_gate::{self, RopeIn, RopeOut, RopeShape};
    use bloomery_gpu_gates::{GateError, activations, bits_equal, checks_failed, verdict};
    use cuda_core::{CudaStream, DeviceBuffer};

    /// The model's global attention shape.
    const N_HEAD: usize = 4;
    const N_KV: usize = 1;
    /// The grouped launch's heads `(n_head, n_kv)`: two key heads, two packs
    /// of four query heads each.
    const GROUP: (usize, usize) = (16, 2);
    const THETA: f32 = 1.0e6;
    const EPS: f32 = 1.0e-6;
    const SCALE: f32 = 1.0;
    /// The scale of one more encoder launch: a kernel that ignores its
    /// `scale` argument passes at 1.
    const SCALE_ALT: f32 = 0.0625;
    const ALT_T: usize = 40;

    /// The rope check's tokens, first position and cache height.
    const ROPE_M: usize = 9;
    const ROPE_CTX: usize = 40;

    /// The encoder launches: `T` rows, each over all `T` keys.
    const ENC_T: [usize; 6] = [1, 8, 31, 33, 100, 512];
    /// The causal launches `(p0, T)`: row `t` over `p0 + t + 1` keys.
    const SEED_CAUSAL: [(usize, usize); 5] = [(0, 17), (31, 9), (32, 40), (1000, 9), (2047, 5)];
    /// The scattered launch's rows: counts `(37·t mod 97) + 1`.
    const SCATTER_T: usize = 24;
    /// The lower sweep's rows: row `r` attends `[r, SWEEP_REACH + r)` — every
    /// offset mod 32 in one launch, windows crossing tiles and blocks.
    const SWEEP_T: usize = 40;
    const SWEEP_REACH: usize = 72;
    /// The packed launch's input lengths, one window an input at its packed
    /// offset, so blocks straddle inputs as the encoder's packed pass's do.
    const PACKED_LENS: [usize; 9] = [2, 3, 4, 5, 6, 7, 8, 29, 33];
    /// Rows past the largest count in each cache, NaN in the padded copy.
    const PAD: usize = 40;
    /// The seeded query's scale: scores of a few tens, as a normed head's.
    const SEED_Q_SCALE: f32 = 2.0;
    /// The layer the fault checks label their launches with.
    const LAYER: usize = 11;

    // ------------------------------------------------------------ rope

    fn rope_run(
        kern: &RopeNeoxKernels,
        stream: &CudaStream,
        inp: &RopeIn,
        pos: &[u32],
        fault: FaultSink,
        (n_head, n_kv): (usize, usize),
    ) -> Result<RopeOut, GateError> {
        rope_gate::run(
            stream,
            inp,
            pos,
            RopeShape {
                head: HEAD,
                n_head,
                n_kv,
                ctx: ROPE_CTX,
                eps: EPS,
            },
            fault,
            |s, args| {
                // SAFETY: rope_gate::run checks that the positions are distinct.
                unsafe { kern.enqueue_head_norm_neox_append_512(s, args) }
            },
        )
    }

    fn rope_check(
        gpu: &Gpu,
        kern: &RopeNeoxKernels,
        heads: (usize, usize),
    ) -> Result<bool, GateError> {
        let (inp, pos) = rope_gate::seeded_inputs(HEAD, HEAD, THETA, (ROPE_CTX, ROPE_M), heads)?;
        let shape = RopeShape {
            head: HEAD,
            n_head: heads.0,
            n_kv: heads.1,
            ctx: ROPE_CTX,
            eps: EPS,
        };
        let exact = rope_gate::check::<HEAD, HEAD, HEAD>(gpu, &inp, &pos, shape, LAYER, |p, f| {
            rope_run(kern, gpu.stream(), &inp, p, f, heads)
        })?;
        let overflow =
            rope_gate::check_overflow::<HEAD>(gpu, &inp, &pos, shape, LAYER, |i, p, f| {
                rope_run(kern, gpu.stream(), i, p, f, heads)
            })?;
        Ok(exact && overflow)
    }

    // --------------------------------------------------------- prefill

    const WIDTH: usize = N_HEAD * HEAD;
    const WHAT: &str = "flash_gqa_prefill_512::enqueue_p4";

    /// One launch over `n_keys.len()` rows of `heads = (n_head, n_kv)` into
    /// an output that starts NaN, read back.
    #[allow(
        clippy::too_many_arguments,
        reason = "one launch's inputs, each a value the callers vary"
    )]
    fn run_pref(
        kp: &FlashGqaPrefill512,
        gpu: &Gpu,
        q: &[f32],
        n_keys: &[u32],
        (kc, vc): (&DeviceBuffer<u16>, &DeviceBuffer<u16>),
        ctx: usize,
        scale: f32,
        (n_head, n_kv): (usize, usize),
    ) -> Result<Vec<f32>, GateError> {
        let stream = gpu.stream();
        let qd = DeviceBuffer::from_host(stream, q)?;
        let t = n_keys.len();
        let nk = DeviceBuffer::from_host(stream, n_keys)?;
        let mut y = DeviceBuffer::from_host(stream, &vec![f32::NAN; t * n_head * HEAD])?;
        kp.enqueue_p4(
            stream,
            GqaPrefillArgs {
                q: &qd,
                kc,
                vc,
                n_keys: &nk,
                scale,
                n_head,
                n_kv,
                ctx,
                t,
                fault: gpu.unlabelled_sink(),
                y: &mut y,
            },
        )?;
        stream.synchronize()?;
        Ok(y.to_host_vec(stream)?)
    }

    /// One seeded launch: its name, its rows' counts, its scale and its
    /// heads `(n_head, n_kv)` and cache padding.
    type Seeded = (String, Vec<usize>, f32, (usize, usize), usize);

    /// The seeded launches: the encoder's counts, the causal and the
    /// scattered at the model's scale and shape, one encoder launch at
    /// [`SCALE_ALT`], and one causal launch of [`GROUP`] heads.
    fn seeded() -> Vec<Seeded> {
        let model = (N_HEAD, N_KV);
        let enc = ENC_T
            .iter()
            .map(move |&t| (format!("encoder T={t}"), vec![t; t], SCALE, model, PAD));
        let causal = SEED_CAUSAL.iter().map(move |&(p0, t)| {
            (
                format!("causal T={t} p0={p0}"),
                (0..t).map(|i| p0 + i + 1).collect(),
                SCALE,
                model,
                PAD,
            )
        });
        let scatter = std::iter::once((
            format!("scattered T={SCATTER_T}"),
            (0..SCATTER_T).map(|t| (37 * t) % 97 + 1).collect(),
            SCALE,
            model,
            PAD,
        ));
        let alt = std::iter::once((
            format!("encoder T={ALT_T} scale {SCALE_ALT}"),
            vec![ALT_T; ALT_T],
            SCALE_ALT,
            model,
            PAD,
        ));
        let (gh, gkv) = GROUP;
        let group = std::iter::once((
            format!("causal T=17 p0=40 heads {gh}/{gkv}"),
            (0..17).map(|i| 40 + i + 1).collect(),
            SCALE,
            GROUP,
            PAD,
        ));
        let tight = std::iter::once((
            "encoder T=33 unpadded".to_string(),
            vec![33; 33],
            SCALE,
            model,
            0,
        ));
        enc.chain(causal)
            .chain(scatter)
            .chain(alt)
            .chain(group)
            .chain(tight)
            .collect()
    }

    /// The prefill clause (module doc, 2) up to the window refusals. Returns
    /// the verdict and what [`misaligned`] reads, which the caller runs last.
    fn prefill_check(gpu: &Gpu, kp: &FlashGqaPrefill512) -> Result<(bool, Windows), GateError> {
        let stream = gpu.stream();
        let model = (N_HEAD, N_KV);
        let mut ok = true;
        let mut seed = 700u32;
        for (name, cu, scale, heads, pad) in seeded() {
            seed += 7;
            let (n_head, n_kv) = heads;
            let w = n_head * HEAD;
            let live = cu.iter().copied().max().unwrap_or(0);
            let ctx = live + pad;
            let t = cu.len();
            let c = cache(stream, HEAD, n_kv, ctx, live, seed)?;
            let q: Vec<f32> = activations(HEAD, t * n_head, seed + 3)
                .iter()
                .map(|v| v * SEED_Q_SCALE)
                .collect();
            let counts: Vec<u32> = cu
                .iter()
                .map(|&n| u32::try_from(n))
                .collect::<Result<_, _>>()?;
            let y = run_pref(kp, gpu, &q, &counts, (&c.kc, &c.vc), ctx, scale, heads)?;
            let y2 = run_pref(kp, gpu, &q, &counts, (&c.kc, &c.vc), ctx, scale, heads)?;
            let yn = run_pref(kp, gpu, &q, &counts, (&c.kn, &c.vn), ctx, scale, heads)?;
            let (rerun, nan_same) = (bits_equal(&y, &y2), bits_equal(&y, &yn));
            let (band, worst) =
                judge_prefill::<HEAD, KEY_TILE_512>(&q, &cu, &c.host, &y, scale, heads, 0);
            let mut differ = 0usize;
            for (r, &n) in counts.iter().enumerate() {
                let one = run_pref(
                    kp,
                    gpu,
                    &q[r * w..(r + 1) * w],
                    &[n],
                    (&c.kc, &c.vc),
                    ctx,
                    scale,
                    heads,
                )?;
                differ += usize::from(!bits_equal(&y[r * w..(r + 1) * w], &one));
            }
            let pass = rerun && nan_same && band && differ == 0;
            println!(
                "prefill {name} ctx={ctx}: measured/bound {worst:.3e} band={band} \
                 rows_alone_differing={differ} rerun={rerun} nan_padding_same={nan_same} {}",
                verdict(pass)
            );
            ok &= pass;
        }

        // A count of zero and one past the cache, on a 17-row launch at 63.
        let (t, p0) = (17usize, 63usize);
        let ctx = p0 + t + PAD;
        let w = WIDTH;
        let c = cache(stream, HEAD, N_KV, ctx, p0 + t, 91)?;
        let q = activations(HEAD, t * N_HEAD, 94);
        let clean: Vec<u32> = (0..t)
            .map(|i| u32::try_from(p0 + i + 1))
            .collect::<Result<_, _>>()?;
        let (bad_hi, bad_zero) = (3usize, 10usize);
        let mut bad = clean.clone();
        bad[bad_hi] = u32::try_from(ctx + 1)?;
        bad[bad_zero] = 0;
        let before = gpu.fault()?;
        let y = run_pref(kp, gpu, &q, &clean, (&c.kc, &c.vc), ctx, SCALE, model)?;
        let after_clean = gpu.fault()?;
        let yb = run_pref(kp, gpu, &q, &bad, (&c.kc, &c.vc), ctx, SCALE, model)?;
        let raised = gpu.take_fault()?;
        let mut others = true;
        let mut nan = true;
        for r in 0..t {
            let (a, b) = (&y[r * w..(r + 1) * w], &yb[r * w..(r + 1) * w]);
            if r == bad_hi || r == bad_zero {
                nan &= b.iter().all(|v| v.is_nan());
            } else {
                others &= bits_equal(a, b);
            }
        }
        let want = Fault::at(LAYER_NONE, FaultSite::KeyCount);
        let fault_ok =
            before.is_none() && after_clean.is_none() && raised == Some(want) && nan && others;
        println!(
            "prefill fault: counts {} (past ctx {ctx}) and 0 at rows {bad_hi}, {bad_zero}: word \
             {raised:?} (want {want:?}), those rows NaN {nan}, other rows bit-identical {others} {}",
            ctx + 1,
            verdict(fault_ok)
        );
        ok &= fault_ok;

        // The captured launch: one node, the eager bits.
        let qd = DeviceBuffer::from_host(stream, &q)?;
        let nk = DeviceBuffer::from_host(stream, &clean)?;
        let mut yg = DeviceBuffer::from_host(stream, &vec![f32::NAN; t * w])?;
        let graph = gpu.capture(|s| {
            kp.enqueue_p4(
                s,
                GqaPrefillArgs {
                    q: &qd,
                    kc: &c.kc,
                    vc: &c.vc,
                    n_keys: &nk,
                    scale: SCALE,
                    n_head: N_HEAD,
                    n_kv: N_KV,
                    ctx,
                    t,
                    fault: gpu.unlabelled_sink(),
                    y: &mut yg,
                },
            )
        })?;
        graph.launch(stream)?;
        stream.synchronize()?;
        let same = bits_equal(&yg.to_host_vec(stream)?, &y);
        let nodes = graph.node_count();
        let graph_ok = same && nodes == 1;
        println!(
            "prefill graph T={t} p0={p0}: eager_vs_graph_bit_identical={same} graph_nodes={nodes} \
             {}",
            verdict(graph_ok)
        );
        ok &= graph_ok;

        // Groups that are not a multiple of four query heads a key head, over
        // planes that hold every key head, so only the group refuses them.
        let planes2 = vec![0u16; 2 * ctx * HEAD];
        let (k2, v2) = (
            DeviceBuffer::from_host(stream, &planes2)?,
            DeviceBuffer::from_host(stream, &planes2)?,
        );
        let mut yr = DeviceBuffer::<f32>::zeroed(stream, t * w)?;
        for (n_head, n_kv) in [(N_HEAD - 1, N_KV), (4, 2)] {
            let refused = kp.enqueue_p4(
                stream,
                GqaPrefillArgs {
                    q: &qd,
                    kc: &k2,
                    vc: &v2,
                    n_keys: &nk,
                    scale: SCALE,
                    n_head,
                    n_kv,
                    ctx,
                    t,
                    fault: gpu.unlabelled_sink(),
                    y: &mut yr,
                },
            );
            let refuse_ok = matches!(
                refused,
                Err(GpuError::Shape {
                    what,
                    ref detail,
                }) if what == WHAT && detail.starts_with("the kernel packs")
            );
            println!(
                "prefill refusal n_head={n_head} over n_kv={n_kv}: {} {}",
                refused
                    .err()
                    .map_or("accepted".to_string(), |e| e.to_string()),
                verdict(refuse_ok)
            );
            ok &= refuse_ok;
        }
        let win = Windows {
            q,
            kb: c.kb,
            vb: c.vb,
            clean,
            y,
            ctx,
        };
        Ok((ok, win))
    }

    // ------------------------------------------------------------ lower

    /// One `enqueue_p4_lower` launch over `n_keys.len()` rows into an output
    /// that starts NaN, read back.
    #[allow(
        clippy::too_many_arguments,
        reason = "one launch's inputs, each a value the callers vary"
    )]
    fn run_lower(
        kp: &FlashGqaPrefill512,
        gpu: &Gpu,
        q: &[f32],
        first: &[u32],
        n_keys: &[u32],
        (kc, vc): (&DeviceBuffer<u16>, &DeviceBuffer<u16>),
        ctx: usize,
        scale: f32,
        (n_head, n_kv): (usize, usize),
    ) -> Result<Vec<f32>, GateError> {
        let stream = gpu.stream();
        let qd = DeviceBuffer::from_host(stream, q)?;
        let t = n_keys.len();
        let fd = DeviceBuffer::from_host(stream, first)?;
        let nd = DeviceBuffer::from_host(stream, n_keys)?;
        let mut y = DeviceBuffer::from_host(stream, &vec![f32::NAN; t * n_head * HEAD])?;
        kp.enqueue_p4_lower(
            stream,
            GqaPrefillArgs {
                q: &qd,
                kc,
                vc,
                n_keys: &nd,
                scale,
                n_head,
                n_kv,
                ctx,
                t,
                fault: gpu.unlabelled_sink(),
                y: &mut y,
            },
            &fd,
        )?;
        stream.synchronize()?;
        Ok(y.to_host_vec(stream)?)
    }

    /// The lower clause's seeded launches: a name, its rows' first keys and
    /// counts, its cache height, its scale, its heads and whether every row
    /// is also launched alone.
    type Lowered = (String, Vec<u32>, Vec<u32>, usize, f32, (usize, usize), bool);

    fn lowered() -> Result<Vec<Lowered>, GateError> {
        let mut sweep = Vec::with_capacity(SWEEP_T);
        for r in 0..SWEEP_T {
            sweep.push((u32::try_from(r)?, u32::try_from(SWEEP_REACH + r)?));
        }
        let (sweep_first, sweep_counts): (Vec<u32>, Vec<u32>) = sweep.iter().copied().unzip();
        let (mut first, mut counts) = (Vec::new(), Vec::new());
        for &len in &PACKED_LENS {
            let lo = u32::try_from(first.len())?;
            let end = lo + u32::try_from(len)?;
            for _ in 0..len {
                first.push(lo);
                counts.push(end);
            }
        }
        let packed_ctx = counts.iter().copied().max().unwrap_or(0) as usize + PAD;
        Ok(vec![
            (
                format!("sweep T={SWEEP_T} windows [r, {SWEEP_REACH}+r)"),
                sweep_first.clone(),
                sweep_counts.clone(),
                SWEEP_REACH + SWEEP_T + PAD,
                SCALE,
                (N_HEAD, N_KV),
                true,
            ),
            (
                format!("sweep T={SWEEP_T} heads {}/{}", GROUP.0, GROUP.1),
                sweep_first,
                sweep_counts,
                SWEEP_REACH + SWEEP_T + PAD,
                SCALE,
                GROUP,
                false,
            ),
            (
                format!("packed T={} lens {PACKED_LENS:?}", first.len()),
                first,
                counts,
                packed_ctx,
                SCALE,
                (N_HEAD, N_KV),
                true,
            ),
        ])
    }

    /// The lower-bound clause (module doc, 3).
    fn lower_check(gpu: &Gpu, kp: &FlashGqaPrefill512) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let mut ok = true;
        for (name, first, counts, ctx, scale, heads, alone) in lowered()? {
            let (n_head, n_kv) = heads;
            let w = n_head * HEAD;
            let t = first.len();
            let live = counts.iter().copied().max().unwrap_or(0) as usize;
            let c = cache(stream, HEAD, n_kv, ctx, live, 721)?;
            let q = activations(HEAD, t * n_head, 724);
            let y = run_lower(
                kp,
                gpu,
                &q,
                &first,
                &counts,
                (&c.kc, &c.vc),
                ctx,
                scale,
                heads,
            )?;
            let y2 = run_lower(
                kp,
                gpu,
                &q,
                &first,
                &counts,
                (&c.kc, &c.vc),
                ctx,
                scale,
                heads,
            )?;
            let yn = run_lower(
                kp,
                gpu,
                &q,
                &first,
                &counts,
                (&c.kn, &c.vn),
                ctx,
                scale,
                heads,
            )?;
            let (rerun, nan_same) = (bits_equal(&y, &y2), bits_equal(&y, &yn));
            let (mut band, mut worst) = (true, 0.0f64);
            for r in 0..t {
                let (lo, hi) = (first[r] as usize, counts[r] as usize);
                let cut = |p: &[f32]| {
                    p.chunks(ctx * HEAD)
                        .flat_map(|h| h[lo * HEAD..hi * HEAD].iter().copied())
                        .collect()
                };
                let host = HostCache {
                    kf: cut(&c.host.kf),
                    vf: cut(&c.host.vf),
                    ctx: hi - lo,
                };
                let (pass, ratio) = judge_prefill::<HEAD, KEY_TILE_512>(
                    &q[r * w..(r + 1) * w],
                    &[hi - lo],
                    &host,
                    &y[r * w..(r + 1) * w],
                    scale,
                    heads,
                    lo,
                );
                band &= pass;
                worst = worst.max(ratio);
            }
            let mut differ = 0usize;
            if alone {
                for r in 0..t {
                    let one = run_lower(
                        kp,
                        gpu,
                        &q[r * w..(r + 1) * w],
                        &[first[r]],
                        &[counts[r]],
                        (&c.kc, &c.vc),
                        ctx,
                        scale,
                        heads,
                    )?;
                    differ += usize::from(!bits_equal(&y[r * w..(r + 1) * w], &one));
                }
            }
            let pass = rerun && nan_same && band && differ == 0 && gpu.fault()?.is_none();
            println!(
                "lower {name} ctx={ctx}: measured/bound {worst:.3e} band={band} \
                 rows_alone_differing={differ} rerun={rerun} nan_padding_same={nan_same} {}",
                verdict(pass)
            );
            ok &= pass;
        }

        // First keys at and past their counts, and counts of zero and past
        // the cache, on a 17-row launch of windows at 63.
        let (t, p0) = (17usize, 63usize);
        let ctx = p0 + t + PAD;
        let w = WIDTH;
        let c = cache(stream, HEAD, N_KV, ctx, p0 + t, 731)?;
        let q = activations(HEAD, t * N_HEAD, 734);
        let first: Vec<u32> = (0..t)
            .map(|i| u32::try_from(p0 + i.saturating_sub(8)))
            .collect::<Result<_, _>>()?;
        let counts: Vec<u32> = (0..t)
            .map(|i| u32::try_from(p0 + i + 1))
            .collect::<Result<_, _>>()?;
        let y = run_lower(
            kp,
            gpu,
            &q,
            &first,
            &counts,
            (&c.kc, &c.vc),
            ctx,
            SCALE,
            (N_HEAD, N_KV),
        )?;
        let qd = DeviceBuffer::from_host(stream, &q)?;
        let fd = DeviceBuffer::from_host(stream, &first)?;
        let nd = DeviceBuffer::from_host(stream, &counts)?;
        let mut yd = DeviceBuffer::from_host(stream, &vec![f32::NAN; t * w])?;
        let launch =
            |first: &DeviceBuffer<u32>, counts: &DeviceBuffer<u32>, y: &mut DeviceBuffer<f32>| {
                kp.enqueue_p4_lower(
                    stream,
                    GqaPrefillArgs {
                        q: &qd,
                        kc: &c.kc,
                        vc: &c.vc,
                        n_keys: counts,
                        scale: SCALE,
                        n_head: N_HEAD,
                        n_kv: N_KV,
                        ctx,
                        t,
                        fault: gpu.unlabelled_sink(),
                        y,
                    },
                    first,
                )
            };
        let want = Fault::at(LAYER_NONE, FaultSite::KeyCount);
        for (row, lo, hi) in [
            (3, counts[3], counts[3]),
            (10, counts[10] + 1, counts[10]),
            (5, first[5], 0),
            (12, first[12], u32::try_from(ctx + 1)?),
        ] {
            let mut bad_first = first.clone();
            let mut bad_counts = counts.clone();
            bad_first[row] = lo;
            bad_counts[row] = hi;
            let bd = DeviceBuffer::from_host(stream, &bad_first)?;
            let bn = DeviceBuffer::from_host(stream, &bad_counts)?;
            yd.copy_from_host(stream, &y)?;
            launch(&bd, &bn, &mut yd)?;
            let out = yd.to_host_vec(stream)?;
            let fault = gpu.take_fault()?;
            let isolated = out
                .chunks(w)
                .zip(y.chunks(w))
                .enumerate()
                .all(|(r, (a, b))| {
                    if r == row {
                        a.iter().all(|v| v.is_nan())
                    } else {
                        bits_equal(a, b)
                    }
                });
            let pass = fault == Some(want) && isolated;
            println!(
                "lower refusal row={row} first={lo} count={hi}: word {fault:?} (want {want:?}), \
                 that row NaN and the rest bit-identical {isolated} {}",
                verdict(pass)
            );
            ok &= pass;
        }
        let short = DeviceBuffer::from_host(stream, &first[..t - 1])?;
        let refused = launch(&short, &nd, &mut yd);
        let named = matches!(refused, Err(GpuError::Shape { ref detail, .. }) if detail.starts_with("k_first.len()"));
        println!("lower short first-key buffer: {}", verdict(named));
        ok &= named;

        // The captured launch: one node, the eager bits.
        let graph = gpu.capture(|_| launch(&fd, &nd, &mut yd))?;
        yd.copy_from_host(stream, &vec![f32::NAN; t * w])?;
        graph.launch(stream)?;
        stream.synchronize()?;
        let same = bits_equal(&yd.to_host_vec(stream)?, &y);
        let nodes = graph.node_count();
        let graph_ok = same && nodes == 1;
        println!(
            "lower graph T={t} p0={p0}: eager_vs_graph_bit_identical={same} graph_nodes={nodes} {}",
            verdict(graph_ok)
        );
        ok &= graph_ok;
        Ok(ok)
    }

    fn misaligned(kp: &FlashGqaPrefill512, gpu: &Gpu, win: &Windows) -> Result<bool, GateError> {
        window_refusals(
            gpu,
            WHAT,
            "prefill",
            WindowCase {
                q: &win.q,
                planes: (&win.kb, &win.vb),
                counts: &win.clean,
                y: &win.y,
                heads: (N_HEAD, N_KV),
                ctx: win.ctx,
                scale: SCALE,
            },
            |s, args| kp.enqueue_p4(s, args),
        )
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let ctx = gpu.context();
        let rope = RopeNeoxKernels::load(ctx)?;
        let kp = FlashGqaPrefill512::load(ctx)?;
        println!(
            "gate_gemma_embedding2_head512: device {}: {N_HEAD}/{N_KV} heads of {HEAD}, rope over \
             {HEAD} at θ {THETA:e}, scale {SCALE:e}, key tile {KEY_TILE_512}",
            gpu.device_name()?
        );
        let mut ok = true;
        for heads in [(N_HEAD, N_KV), (8, 2)] {
            let r = rope_check(&gpu, &rope, heads)?;
            println!("rope-512 heads {}/{} {}", heads.0, heads.1, verdict(r));
            ok &= r;
        }
        let (p, win) = prefill_check(&gpu, &kp)?;
        println!("prefill flash 512 p4 {}", verdict(p));
        ok &= p;
        let l = lower_check(&gpu, &kp)?;
        println!("prefill flash 512 p4 lower {}", verdict(l));
        ok &= l;
        // Last: the window refusals, whose failure would be a sticky error.
        let m = misaligned(&kp, &gpu, &win)?;
        println!("prefill windows {}", verdict(m));
        ok &= m;
        println!("gate_gemma_embedding2_head512: {}", verdict(ok));
        if !ok {
            return Err(checks_failed());
        }
        Ok(())
    }
}
