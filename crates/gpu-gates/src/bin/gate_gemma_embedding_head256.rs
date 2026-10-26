//! Head-256 prefill with three query heads per key head, against the f64
//! attention and first-order band in `flash_band`, over whole inputs and
//! symmetric windows. Block, slice and key-tile tails run on two key heads too.
//!
//! The pack-of-two lower entry ([`FlashGqaPrefill::enqueue_256_p2_lower`]:
//! EmbeddingGemma 2's sliding layers, four query heads over two key heads)
//! at the encoder's packed-pass shape, which no earlier clause holds — every
//! row of an input sharing one window at the input's packed offset, blocks
//! straddling inputs. A sweep whose row `r` attends `[r, 72 + r)` — every
//! offset mod 64 (the key tile) and mod 32 (the block) in one launch — and
//! a packed launch whose inputs of lengths `[2, 3, 4, 5, 6, 7, 8, 29, 33]`
//! sit at their packed offsets. Each row against the exact attention over
//! its own window (the cache cut to it) and bit for bit its one-row launch,
//! a rerun and the NaN-padded planes the same bits; first keys at and past
//! their counts and counts of zero and past the cache refused with
//! `key_count` (that row NaN, the rest bit-identical), a short first-key
//! buffer by name, and one captured launch the eager bits.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_gemma_embedding_head256", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::Gpu;
    use bloomery_gpu::fault::{Fault, FaultSite, LAYER_NONE};
    use bloomery_gpu::flash_gqa_prefill::{FlashGqaPrefill, GqaPrefillArgs, HEAD_256, KEY_TILE};
    use bloomery_gpu_gates::flash_band::judge_prefill;
    use bloomery_gpu_gates::prefill_gate::{HostCache, cache};
    use bloomery_gpu_gates::{GateError, activations, bits_equal, checks_failed, verdict};
    use cuda_core::DeviceBuffer;

    // ------------------------------------------------- pack-of-two lower

    /// EmbeddingGemma 2's sliding layers: four query heads over two key
    /// heads, the group the encoder's packed pass launches the lower entry
    /// at.
    const N_HEAD_P2: usize = 4;
    const N_KV_P2: usize = 2;
    /// The grouped sweep's heads: two key heads, four query heads each.
    const GROUP_P2: (usize, usize) = (8, 2);
    /// The sweep's rows: row `r` attends `[r, SWEEP_REACH_P2 + r)` — every
    /// offset mod 64 (the key tile) and mod 32 (the block) in one launch,
    /// windows crossing tiles and blocks.
    const SWEEP_T_P2: usize = 72;
    const SWEEP_REACH_P2: usize = 72;
    /// The packed launch's input lengths: one window an input at its packed
    /// offset, every row of an input sharing it while blocks straddle
    /// inputs, as the encoder's packed pass's do.
    const PACKED_LENS_P2: [usize; 9] = [2, 3, 4, 5, 6, 7, 8, 29, 33];
    /// Rows past the largest count in each cache, NaN in the padded copy.
    const PAD_P2: usize = 40;

    /// One `enqueue_256_p2_lower` launch over `n_keys.len()` rows into an
    /// output that starts NaN, read back.
    #[allow(
        clippy::too_many_arguments,
        reason = "one launch's inputs, each a value the callers vary"
    )]
    fn run_p2_lower(
        kernel: &FlashGqaPrefill,
        gpu: &Gpu,
        q: &[f32],
        first: &[u32],
        n_keys: &[u32],
        (kc, vc): (&DeviceBuffer<u16>, &DeviceBuffer<u16>),
        ctx: usize,
        (n_head, n_kv): (usize, usize),
    ) -> Result<Vec<f32>, GateError> {
        let stream = gpu.stream();
        let qd = DeviceBuffer::from_host(stream, q)?;
        let t = n_keys.len();
        let fd = DeviceBuffer::from_host(stream, first)?;
        let nd = DeviceBuffer::from_host(stream, n_keys)?;
        let mut y = DeviceBuffer::from_host(stream, &vec![f32::NAN; t * n_head * HEAD_256])?;
        kernel.enqueue_256_p2_lower(
            stream,
            GqaPrefillArgs {
                q: &qd,
                kc,
                vc,
                n_keys: &nd,
                scale: 1.0,
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
    /// counts, its cache height and its heads `(n_head, n_kv)`; the last
    /// flag is whether every row is also launched alone.
    type LoweredP2 = (String, Vec<u32>, Vec<u32>, usize, (usize, usize), bool);

    fn lowered_p2() -> Result<Vec<LoweredP2>, GateError> {
        let mut sweep = Vec::with_capacity(SWEEP_T_P2);
        for r in 0..SWEEP_T_P2 {
            sweep.push((u32::try_from(r)?, u32::try_from(SWEEP_REACH_P2 + r)?));
        }
        let (sweep_first, sweep_counts): (Vec<u32>, Vec<u32>) = sweep.iter().copied().unzip();
        let (mut first, mut counts) = (Vec::new(), Vec::new());
        for &len in &PACKED_LENS_P2 {
            let (lo, end) = (
                u32::try_from(first.len())?,
                u32::try_from(first.len() + len)?,
            );
            for _ in 0..len {
                first.push(lo);
                counts.push(end);
            }
        }
        let packed_ctx = counts.iter().copied().max().unwrap_or(0) as usize + PAD_P2;
        Ok(vec![
            (
                format!("sweep T={SWEEP_T_P2} windows [r, {SWEEP_REACH_P2}+r)"),
                sweep_first.clone(),
                sweep_counts.clone(),
                SWEEP_REACH_P2 + SWEEP_T_P2 + PAD_P2,
                (N_HEAD_P2, N_KV_P2),
                true,
            ),
            (
                format!("sweep T={SWEEP_T_P2} heads {}/{}", GROUP_P2.0, GROUP_P2.1),
                sweep_first,
                sweep_counts,
                SWEEP_REACH_P2 + SWEEP_T_P2 + PAD_P2,
                GROUP_P2,
                false,
            ),
            (
                format!("packed T={} lens {PACKED_LENS_P2:?}", first.len()),
                first,
                counts,
                packed_ctx,
                (N_HEAD_P2, N_KV_P2),
                true,
            ),
        ])
    }

    /// The pack-of-two lower clause (the module doc's second paragraph).
    fn p2_lower_check(gpu: &Gpu, kernel: &FlashGqaPrefill) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let mut ok = true;
        for (name, first, counts, ctx, heads, alone) in lowered_p2()? {
            let (n_head, n_kv) = heads;
            let w = n_head * HEAD_256;
            let t = first.len();
            let live = counts.iter().copied().max().unwrap_or(0) as usize;
            let c = cache(stream, HEAD_256, n_kv, ctx, live, 741)?;
            let q = activations(HEAD_256, t * n_head, 744);
            let y = run_p2_lower(kernel, gpu, &q, &first, &counts, (&c.kc, &c.vc), ctx, heads)?;
            let y2 = run_p2_lower(kernel, gpu, &q, &first, &counts, (&c.kc, &c.vc), ctx, heads)?;
            let yn = run_p2_lower(kernel, gpu, &q, &first, &counts, (&c.kn, &c.vn), ctx, heads)?;
            let (rerun, nan_same) = (bits_equal(&y, &y2), bits_equal(&y, &yn));
            let (mut band, mut worst) = (true, 0.0f64);
            for r in 0..t {
                let (lo, hi) = (first[r] as usize, counts[r] as usize);
                let cut = |p: &[f32]| {
                    p.chunks(ctx * HEAD_256)
                        .flat_map(|h| h[lo * HEAD_256..hi * HEAD_256].iter().copied())
                        .collect()
                };
                let host = HostCache {
                    kf: cut(&c.host.kf),
                    vf: cut(&c.host.vf),
                    ctx: hi - lo,
                };
                let (pass, ratio) = judge_prefill::<HEAD_256, KEY_TILE>(
                    &q[r * w..(r + 1) * w],
                    &[hi - lo],
                    &host,
                    &y[r * w..(r + 1) * w],
                    1.0,
                    heads,
                    lo,
                );
                band &= pass;
                worst = worst.max(ratio);
            }
            let mut differ = 0usize;
            if alone {
                for r in 0..t {
                    let one = run_p2_lower(
                        kernel,
                        gpu,
                        &q[r * w..(r + 1) * w],
                        &[first[r]],
                        &[counts[r]],
                        (&c.kc, &c.vc),
                        ctx,
                        heads,
                    )?;
                    differ += usize::from(!bits_equal(&y[r * w..(r + 1) * w], &one));
                }
            }
            let pass = rerun && nan_same && band && differ == 0 && gpu.fault()?.is_none();
            println!(
                "p2 lower {name} ctx={ctx}: measured/bound {worst:.3e} band={band} \
                 rows_alone_differing={differ} rerun={rerun} nan_padding_same={nan_same} {}",
                verdict(pass)
            );
            ok &= pass;
        }

        // First keys at and past their counts, and counts of zero and past
        // the cache, on a 17-row launch of windows at 63.
        let (t, p0) = (17usize, 63usize);
        let ctx = p0 + t + PAD_P2;
        let w = N_HEAD_P2 * HEAD_256;
        let c = cache(stream, HEAD_256, N_KV_P2, ctx, p0 + t, 747)?;
        let q = activations(HEAD_256, t * N_HEAD_P2, 750);
        let first: Vec<u32> = (0..t)
            .map(|i| u32::try_from(p0 + i.saturating_sub(8)))
            .collect::<Result<_, _>>()?;
        let counts: Vec<u32> = (0..t)
            .map(|i| u32::try_from(p0 + i + 1))
            .collect::<Result<_, _>>()?;
        let y = run_p2_lower(
            kernel,
            gpu,
            &q,
            &first,
            &counts,
            (&c.kc, &c.vc),
            ctx,
            (N_HEAD_P2, N_KV_P2),
        )?;
        let qd = DeviceBuffer::from_host(stream, &q)?;
        let fd = DeviceBuffer::from_host(stream, &first)?;
        let nd = DeviceBuffer::from_host(stream, &counts)?;
        let mut yd = DeviceBuffer::from_host(stream, &vec![f32::NAN; t * w])?;
        let launch =
            |first: &DeviceBuffer<u32>, counts: &DeviceBuffer<u32>, y: &mut DeviceBuffer<f32>| {
                kernel.enqueue_256_p2_lower(
                    stream,
                    GqaPrefillArgs {
                        q: &qd,
                        kc: &c.kc,
                        vc: &c.vc,
                        n_keys: counts,
                        scale: 1.0,
                        n_head: N_HEAD_P2,
                        n_kv: N_KV_P2,
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
                "p2 lower refusal row={row} first={lo} count={hi}: word {fault:?} (want \
                 {want:?}), that row NaN and the rest bit-identical {isolated} {}",
                verdict(pass)
            );
            ok &= pass;
        }
        let short = DeviceBuffer::from_host(stream, &first[..t - 1])?;
        let refused = launch(&short, &nd, &mut yd);
        let named = matches!(refused, Err(bloomery_gpu::GpuError::Shape { ref detail, .. }) if detail.starts_with("k_first.len()"));
        println!("p2 lower short first-key buffer: {}", verdict(named));
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
            "p2 lower graph T={t} p0={p0}: eager_vs_graph_bit_identical={same} \
             graph_nodes={nodes} {}",
            verdict(graph_ok)
        );
        ok &= graph_ok;
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let gpu = Gpu::new()?;
        let kernel = FlashGqaPrefill::load(gpu.context())?;
        let stream = gpu.stream();
        println!(
            "gate_gemma_embedding_head256: device {}",
            gpu.device_name()?
        );
        let mut ok = true;
        let lower = p2_lower_check(&gpu, &kernel)?;
        println!("prefill flash 256 p2 lower {}", verdict(lower));
        ok &= lower;
        for n_kv in [1, 2] {
            let n_head = 3 * n_kv;
            let width = n_head * HEAD_256;
            for t in [1usize, 63, 65, 129] {
                let ctx = if t == 65 { t } else { t + KEY_TILE };
                let c = cache(stream, HEAD_256, n_kv, ctx, t, 700)?;
                let q = activations(HEAD_256, t * n_head, 703);
                let qd = DeviceBuffer::from_host(stream, &q)?;
                for reach in [t, 17, 64].into_iter().filter(|&r| r == t || r < t - 1) {
                    let first: Vec<u32> = (0..t)
                        .map(|i| u32::try_from(i.saturating_sub(reach)))
                        .collect::<Result<_, _>>()?;
                    let counts: Vec<u32> = (0..t)
                        .map(|i| u32::try_from((i + reach + 1).min(t)))
                        .collect::<Result<_, _>>()?;
                    let fd = DeviceBuffer::from_host(stream, &first)?;
                    let nd = DeviceBuffer::from_host(stream, &counts)?;
                    for scale in [1.0f32, 0.0625] {
                        let launch =
                            |kc: &DeviceBuffer<u16>,
                             vc: &DeviceBuffer<u16>,
                             nd: &DeviceBuffer<u32>,
                             fd: &DeviceBuffer<u32>,
                             out: &mut DeviceBuffer<f32>| {
                                kernel.enqueue_256_p1_lower(
                                    stream,
                                    GqaPrefillArgs {
                                        q: &qd,
                                        kc,
                                        vc,
                                        n_keys: nd,
                                        scale,
                                        n_head,
                                        n_kv,
                                        ctx,
                                        t,
                                        fault: gpu.unlabelled_sink(),
                                        y: out,
                                    },
                                    fd,
                                )
                            };
                        let mut yd = DeviceBuffer::from_host(stream, &vec![f32::NAN; t * width])?;
                        launch(&c.kc, &c.vc, &nd, &fd, &mut yd)?;
                        let y = yd.to_host_vec(stream)?;
                        launch(&c.kn, &c.vn, &nd, &fd, &mut yd)?;
                        let padded = yd.to_host_vec(stream)?;
                        launch(&c.kc, &c.vc, &nd, &fd, &mut yd)?;
                        let again = yd.to_host_vec(stream)?;
                        let mut band = true;
                        let mut worst = 0.0f64;
                        for r in 0..t {
                            let (lo, hi) = (first[r] as usize, counts[r] as usize);
                            let cut = |p: &[f32]| {
                                p.chunks(ctx * HEAD_256)
                                    .flat_map(|h| h[lo * HEAD_256..hi * HEAD_256].iter().copied())
                                    .collect()
                            };
                            let host = HostCache {
                                kf: cut(&c.host.kf),
                                vf: cut(&c.host.vf),
                                ctx: hi - lo,
                            };
                            let (pass, ratio) = judge_prefill::<HEAD_256, KEY_TILE>(
                                &q[r * width..(r + 1) * width],
                                &[hi - lo],
                                &host,
                                &y[r * width..(r + 1) * width],
                                scale,
                                (n_head, n_kv),
                                lo,
                            );
                            band &= pass;
                            worst = worst.max(ratio);
                        }
                        let graph = gpu.capture(|_| launch(&c.kc, &c.vc, &nd, &fd, &mut yd))?;
                        yd.copy_from_host(stream, &vec![f32::NAN; t * width])?;
                        graph.launch(stream)?;
                        let captured = yd.to_host_vec(stream)?;
                        let pass = band
                            && bits_equal(&y, &padded)
                            && bits_equal(&y, &again)
                            && bits_equal(&y, &captured)
                            && graph.node_count() == 1
                            && gpu.fault()?.is_none();
                        println!(
                            "prefill heads={n_head}/{n_kv} T={t} ctx={ctx} reach={reach} scale={scale}: measured/bound={worst:.3e} band={band} rerun={} nan_padding={} graph={} {}",
                            bits_equal(&y, &again),
                            bits_equal(&y, &padded),
                            bits_equal(&y, &captured),
                            verdict(pass)
                        );
                        ok &= pass;

                        if t == 65 && reach == 17 && scale == 1.0 {
                            let want = Some(Fault::at(LAYER_NONE, FaultSite::KeyCount));
                            for (row, lo, hi) in [
                                (3, counts[3], counts[3]),
                                (37, counts[37] + 1, counts[37]),
                                (3, first[3], 0),
                                (37, first[37], u32::try_from(ctx + 1)?),
                            ] {
                                let mut bad_first = first.clone();
                                let mut bad_counts = counts.clone();
                                bad_first[row] = lo;
                                bad_counts[row] = hi;
                                let bd = DeviceBuffer::from_host(stream, &bad_first)?;
                                let bn = DeviceBuffer::from_host(stream, &bad_counts)?;
                                yd.copy_from_host(stream, &y)?;
                                launch(&c.kc, &c.vc, &bn, &bd, &mut yd)?;
                                let out = yd.to_host_vec(stream)?;
                                let fault = gpu.take_fault()?;
                                let isolated =
                                    out.chunks(width).zip(y.chunks(width)).enumerate().all(
                                        |(r, (a, b))| {
                                            if r == row {
                                                a.iter().all(|v| v.is_nan())
                                            } else {
                                                bits_equal(a, b)
                                            }
                                        },
                                    );
                                let pass = fault == want && isolated;
                                println!(
                                    "prefill refusal row={row} first={lo} count={hi}: {}",
                                    verdict(pass)
                                );
                                ok &= pass;
                            }
                            let short = DeviceBuffer::from_host(stream, &first[..t - 1])?;
                            let refused = launch(&c.kc, &c.vc, &nd, &short, &mut yd);
                            let named = matches!(refused, Err(bloomery_gpu::GpuError::Shape { ref detail, .. }) if detail.starts_with("k_first.len()"));
                            println!("prefill short first-key buffer: {}", verdict(named));
                            ok &= named;
                        }
                    }
                }
            }
        }
        println!("gate_gemma_embedding_head256: {}", verdict(ok));
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
