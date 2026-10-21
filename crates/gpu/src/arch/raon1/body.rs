//! The Raon-1 `ChainBody` over one sequence on one card: the file's keys read
//! once through [`Hparams`], each layer's sites, router and stores resolved
//! once at load, the step's buffers allocated once, and the token and
//! position refreshed before every enqueue or replay.
//!
//! A step (llama.cpp `src/models/raon1.cpp`): the embedding row repeated into
//! the [`STREAMS`] residual streams; every layer then runs its two
//! sub-layers through Raon_mHC ([`super::mhc`]): the route of the streams,
//! their fold into the sub-layer's input, the sub-layer, the post mix of the
//! streams and its output. The attention sub-layer is an RDA layer
//! ([`super::rda`]) or a latent one ([`super::mla`]), each applying its own
//! `attn_norm`; the feed-forward one is `ffn_norm` then the dense or routed
//! SiTU block ([`super::ffn`]). The head reads the streams' mean.

use super::ffn::{self, Ffn, FfnKernels, FfnScratch, Glu, Routed};
use super::mhc::{HC_OUT, MhcKernels, STREAMS, SiteVecs};
use super::mla::{self, MlaBufs, MlaDims, MlaKernels, MlaSites};
use super::rda::{self, RdaSites};
use crate::arch::hybrid::{Acts, DecodeInput, GqaCx, Site};
use crate::elem::EmbedRowsArgs;
use crate::flash_gqa::FlashGqaKernels;
use crate::gemm::Gemm32Kernels;
use crate::head::Head;
use crate::kquant::Act;
use crate::linear::mixer_kda::{KdaBufs, KdaDims, KdaState};
use crate::linear::{HEAD, LinearKernels};
use crate::model::{ChainBody, GpuModel, NoHost, block_count};
use crate::site::{self, KGemvKernels, SiteTy};
use crate::weights::{DevWeight, Weights};
use crate::{DeviceTensor, Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::arch::Arch;
use model::arch::models::KdaGate;
use model::arch::raon1::hparams::{Hparams, Kind};
use model::arch::raon1::names::{self, Sub};
use model::placement::CardFormat;
use model::placement::kernels;
use std::ops::Range;

const WHAT: &str = "raon1::Body::load";
const STEP: &str = "raon1::Body::enqueue_chain";

/// The types a dense or shared-expert projection may have.
const FFN_TYPES: &[SiteTy] = &[
    SiteTy::Q4K,
    SiteTy::Q5K,
    SiteTy::Q6K,
    SiteTy::Q8_0,
    SiteTy::Q5_0,
];

/// A layer's attention sub-layer and its store.
enum Mixer {
    /// The recurrent state, its stamp and the conv ring of one lane.
    Rda {
        sites: Box<RdaSites>,
        state: DeviceBuffer<f32>,
        stamp: DeviceBuffer<u32>,
        ring: DeviceBuffer<f32>,
    },
    /// The latent cache, `[ctx][row]` f16.
    Mla {
        sites: Box<MlaSites>,
        cache: DeviceTensor<u16>,
    },
}

/// One layer: its attention sub-layer, its feed-forward block and the names
/// of its two hyper-connection sites' vectors (`fn`, `scale`, `base`).
struct Layer {
    mixer: Mixer,
    ffn: Ffn,
    hc: [[String; 3]; 2],
}

impl Layer {
    fn sites(&self) -> Vec<&Site> {
        let mut out: Vec<&Site> = match &self.mixer {
            Mixer::Rda { sites, .. } => sites.sites().to_vec(),
            Mixer::Mla { sites, .. } => sites.sites().to_vec(),
        };
        match &self.ffn {
            Ffn::Dense(g) => out.extend(g.sites()),
            Ffn::Routed(r) => out.extend(r.shared.sites()),
        }
        out
    }

    fn store_bytes(&self) -> usize {
        match &self.mixer {
            Mixer::Rda {
                state, stamp, ring, ..
            } => state.num_bytes() + stamp.num_bytes() + ring.num_bytes(),
            Mixer::Mla { cache, .. } => cache.buf().num_bytes(),
        }
    }
}

/// The kernels the step launches beyond the `Gpu`'s own.
struct Kernels {
    kgemv: KGemvKernels,
    g32: Gemm32Kernels,
    flash: FlashGqaKernels,
    linear: LinearKernels,
    hc: MhcKernels,
    ffn: FfnKernels,
    mla: MlaKernels,
}

impl Kernels {
    fn load(gpu: &Gpu, mla: &MlaDims) -> Result<Kernels, GpuError> {
        let ctx = gpu.context();
        Ok(Kernels {
            kgemv: KGemvKernels::load(gpu)?,
            g32: Gemm32Kernels::load(ctx)?,
            flash: FlashGqaKernels::load(ctx)?,
            linear: LinearKernels::load(ctx)?,
            hc: MhcKernels::load(ctx)?,
            ffn: FfnKernels::load(gpu)?,
            mla: MlaKernels::load(gpu, mla)?,
        })
    }
}

/// The step's one-token buffers.
struct Scratch {
    /// The streams the layer reads and the ones it writes, `STREAMS · n`.
    x: DeviceBuffer<f32>,
    x2: DeviceBuffer<f32>,
    /// The sub-layer's route ([`HC_OUT`] values).
    hc: DeviceBuffer<f32>,
    /// The fold, its norm and the sub-layer's output, `n` each.
    u: DeviceBuffer<f32>,
    un: DeviceBuffer<f32>,
    br: DeviceBuffer<f32>,
    /// The token once a stream: the embedding writes the streams in one
    /// launch.
    ids: DeviceBuffer<u32>,
    pos0: DeviceBuffer<u32>,
    /// The embedding's per-row position and live key count; row 0's are the
    /// step's.
    pos: DeviceBuffer<u32>,
    n_keys: DeviceBuffer<u32>,
    /// The recurrent stores' one lane word (lane 0).
    lane: DeviceBuffer<u32>,
    /// The RDA mixer's: qkv, fa, ga, beta_raw, write_raw, f, z, conv, beta,
    /// decay, o, gated.
    kda: [DeviceBuffer<f32>; 12],
    /// The latent layer's: qa, qr, kv, gate, q, qabs, att, av, gated,
    /// part_v, part_ms.
    mla: [DeviceBuffer<f32>; 11],
    ffn: FfnScratch,
}

impl Scratch {
    fn new(
        stream: &CudaStream,
        hp: &Hparams,
        (d, m): (&KdaDims, &MlaDims),
        ctx: usize,
    ) -> Result<Scratch, GpuError> {
        let z = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n.max(1));
        let u = |n: usize| DeviceBuffer::<u32>::zeroed(stream, n);
        let n = hp.n_embd;
        let lin = d.shape;
        let (channels, v, n_v) = (lin.channels(), lin.n_v * HEAD, lin.n_v);
        let routed_ff = hp.experts.ff;
        let shared_ff = hp.experts.shared_ff();
        let (part_v, part_ms) = mla::partials_lens(m, ctx);
        let (heads, row) = (m.heads, m.row());
        Ok(Scratch {
            x: z(STREAMS * n)?,
            x2: z(STREAMS * n)?,
            hc: z(HC_OUT)?,
            u: z(n)?,
            un: z(n)?,
            br: z(n)?,
            ids: u(STREAMS)?,
            pos0: u(1)?,
            pos: u(STREAMS)?,
            n_keys: u(STREAMS)?,
            lane: u(1)?,
            kda: [
                z(channels)?,
                z(hp.decay_rank)?,
                z(hp.decay_rank)?,
                z(n_v)?,
                z(n_v)?,
                z(v)?,
                z(v)?,
                z(channels)?,
                z(n_v)?,
                z(v)?,
                z(v)?,
                z(v)?,
            ],
            mla: [
                z(m.q_lora)?,
                z(m.q_lora)?,
                z(row)?,
                z(heads * m.v_head)?,
                z(heads * m.head_k())?,
                z(heads * row)?,
                z(heads * hp.mla.kv_lora)?,
                z(heads * m.v_head)?,
                z(heads * m.v_head)?,
                z(part_v)?,
                z(part_ms)?,
            ],
            ffn: FfnScratch::new(stream, n, hp.dense_ff.max(shared_ff), routed_ff)?,
        })
    }

    fn bytes(&self) -> usize {
        let f = [&self.x, &self.x2, &self.hc, &self.u, &self.un, &self.br]
            .into_iter()
            .chain(self.kda.iter())
            .chain(self.mla.iter())
            .map(DeviceBuffer::num_bytes)
            .sum::<usize>();
        let w = [&self.ids, &self.pos0, &self.pos, &self.n_keys, &self.lane]
            .into_iter()
            .map(DeviceBuffer::num_bytes)
            .sum::<usize>();
        f + w + self.ffn.bytes()
    }
}

/// The Raon-1 chain over one sequence on one card.
pub struct Body {
    hp: Hparams,
    kda: KdaDims,
    mla: MlaDims,
    act: Act,
    layers: Vec<Layer>,
    k: Kernels,
    acts: Acts,
    s: Scratch,
    /// The token once a stream, rewritten by every refresh.
    ids: [u32; STREAMS],
    ctx: usize,
}

impl GpuModel<Body> {
    /// The whole model of `file` resident on `gpu`, plus the output head:
    /// every tensor in its kernels' device format (the F16 ones widened to
    /// f32), the latent layers' caches of `ctx` rows, the RDA layers' stores
    /// and the step's buffers. The model takes the file and closes it once
    /// the weights are resident.
    pub fn open(gpu: Gpu, file: Split, ctx: usize) -> Result<GpuModel<Body>, GpuError> {
        let n_layers = block_count(&file, "raon1 GpuModel::open")?;
        GpuModel::load_blocks(gpu, &file, ctx, 0..n_layers, true, |gpu, w| {
            let hp = Hparams::read(&file).map_err(|e| GpuError::plan(WHAT, e))?;
            for l in (0..hp.n_layer).filter(|&l| hp.kind(l) == Kind::Rda) {
                rda::derive(gpu.stream(), w, l)?;
            }
            for l in (0..hp.n_layer).filter(|&l| !hp.dense(l)) {
                down_words(gpu.stream(), &file, w, &names::ffn_down_exps(l))?;
            }
            Body::load(gpu, &file, w, hp, ctx)
        })
    }
}

/// The routed down stack `name` in the layout the card leg reads it in
/// ([`LegStacks::of_resident`](crate::card_leg::LegStacks::of_resident)): a
/// type whose down entry reads the file's blocks as words, and which
/// [`Weights::load`] laid out otherwise (Q8_0's planes), is uploaded again
/// as words. Load-time only.
fn down_words(
    stream: &CudaStream,
    file: &Split,
    w: &mut Weights,
    name: &str,
) -> Result<(), GpuError> {
    let (_, t) = file
        .find(name)
        .ok_or_else(|| GpuError::tensor(WHAT, name, "a routed down stack of the file"))?;
    if kernels::card_down_runs(t.ty) && CardFormat::of(t.ty) != Some(CardFormat::KQuant) {
        w.relayout(stream, file, name, CardFormat::KQuant)?;
    }
    Ok(())
}

/// The site of `name` at `(rows, k)`, its type read from the file among
/// `allowed` and held to the resident weight.
fn typed(
    file: &Split,
    w: &Weights,
    name: String,
    (rows, k): (usize, usize),
    allowed: &[SiteTy],
) -> Result<Site, GpuError> {
    let ty = site::file_site(file, WHAT, &name, (rows, k), allowed)?;
    site::site(w, WHAT, &name, (rows, k), ty)?;
    Ok(Site::new(name, ty, k))
}

/// The resident f32 vector or matrix `name`, held to `len` values.
fn exact(w: &Weights, name: String, len: usize) -> Result<String, GpuError> {
    let got = w.f32_buf(WHAT, &name)?.len();
    if got != len {
        return Err(GpuError::shape(
            WHAT,
            format!("{name} holds {got} values; the step reads {len}"),
        ));
    }
    Ok(name)
}

impl Body {
    /// The last hyper-connection site's route as the step left it: the last
    /// layer's feed-forward site, `pre[4]`, `post[4]`, then `comb[16]` at
    /// `src · 4 + dst` ([`HC_OUT`] values). Synchronizes the stream.
    pub fn last_route(&self, stream: &CudaStream) -> Result<Vec<f32>, GpuError> {
        Ok(self.s.hc.to_host_vec(stream)?)
    }

    fn load(
        gpu: &Gpu,
        file: &Split,
        w: &Weights,
        hp: Hparams,
        ctx: usize,
    ) -> Result<Body, GpuError> {
        let stream = gpu.stream();
        let n = hp.n_embd;
        if hp.hc.streams != STREAMS {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "hyper_connection.count is {}; the card's mHC runs {STREAMS} streams",
                    hp.hc.streams
                ),
            ));
        }
        let KdaGate::Sigmoid { lower_bound } = hp.gate else {
            return Err(GpuError::shape(
                WHAT,
                "the RDA layers' decay is the softplus form; the card runs the bounded sigmoid \
                 (kda.gate_lower_bound)",
            ));
        };
        let e = hp.experts;
        if e.n_expert != ffn::sigmoid_128_6::N_EXPERT
            || e.n_used != ffn::sigmoid_128_6::N_USED
            || !e.weights_norm
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the router is {} experts, top {}, renormalized {}; the card's router is \
                     {} experts, top {}, renormalized",
                    e.n_expert,
                    e.n_used,
                    e.weights_norm,
                    ffn::sigmoid_128_6::N_EXPERT,
                    ffn::sigmoid_128_6::N_USED
                ),
            ));
        }
        let kda = rda::dims(n, hp.n_head, hp.rms_eps, lower_bound);
        kda.rule.check(WHAT)?;
        let mla = MlaDims {
            embd: n,
            heads: hp.n_head,
            q_lora: hp.mla.q_lora,
            nope: hp.mla.nope(),
            extra: hp.mla.pe,
            v_head: hp.mla.head_v,
            rms_eps: hp.rms_eps,
            latent_eps: hp.latent_eps,
        };
        if mla.row() != hp.mla.row() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the latent row is {} values in the file; the card's latent layer caches {}",
                    hp.mla.row(),
                    mla.row()
                ),
            ));
        }
        let act = Act::Situ {
            beta: hp.situ.gate_beta,
            linear_beta: hp.situ.value_beta,
        };
        act.code(WHAT)?;
        let glu = |l: usize, (gate, up, down): (String, String, String), ff: usize| {
            Ok::<Glu, GpuError>(Glu {
                gate: typed(file, w, gate, (ff, n), FFN_TYPES)?,
                up: typed(file, w, up, (ff, n), FFN_TYPES)?,
                down: typed(file, w, down, (n, ff), FFN_TYPES)?,
                ff,
            })
            .map_err(|err| GpuError::shape(WHAT, format!("layer {l}'s feed-forward block: {err}")))
        };
        let mut layers = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let mixer = match hp.kind(l) {
                Kind::Rda => {
                    let lin = kda.shape;
                    Mixer::Rda {
                        sites: Box::new(RdaSites::load(file, w, &kda, l)?),
                        state: DeviceBuffer::zeroed(stream, lin.state_len())?,
                        stamp: DeviceBuffer::from_host(stream, &[0u32])?,
                        ring: DeviceBuffer::zeroed(stream, lin.ring_len())?,
                    }
                }
                Kind::Mla => Mixer::Mla {
                    sites: Box::new(MlaSites::load(file, w, &mla, l)?),
                    cache: DeviceTensor::zeroed(stream, ctx, mla.row())?,
                },
            };
            let ffn = if hp.dense(l) {
                Ffn::Dense(glu(
                    l,
                    (names::ffn_gate(l), names::ffn_up(l), names::ffn_down(l)),
                    hp.dense_ff,
                )?)
            } else {
                Ffn::Routed(Routed {
                    router: exact(w, names::ffn_gate_inp(l), e.n_expert * n)?,
                    bias: exact(w, names::exp_probs_b(l), e.n_expert)?,
                    stacks: [
                        names::ffn_gate_exps(l),
                        names::ffn_up_exps(l),
                        names::ffn_down_exps(l),
                    ],
                    shared: glu(
                        l,
                        (
                            names::ffn_gate_shexp(l),
                            names::ffn_up_shexp(l),
                            names::ffn_down_shexp(l),
                        ),
                        e.shared_ff(),
                    )?,
                })
            };
            let hc = [Sub::Attn, Sub::Ffn].map(|sub| {
                [
                    names::hc_fn(l, sub),
                    names::hc_scale(l, sub),
                    names::hc_base(l, sub),
                ]
            });
            for [f, scale, base] in &hc {
                exact(w, f.clone(), hp.hc.mix_dim() * STREAMS * n)?;
                exact(w, scale.clone(), 4)?;
                exact(w, base.clone(), hp.hc.mix_dim())?;
            }
            exact(w, names::ffn_norm(l), n)?;
            layers.push(Layer { mixer, ffn, hc });
        }
        let acts = Acts::new(stream, layers.iter().flat_map(Layer::sites))?;
        let s = Scratch::new(stream, &hp, (&kda, &mla), ctx)?;
        Ok(Body {
            k: Kernels::load(gpu, &mla)?,
            kda,
            mla,
            act,
            layers,
            acts,
            s,
            ids: [0; STREAMS],
            ctx,
            hp,
        })
    }

    /// Device bytes the body holds past the weights and the stores.
    fn scratch_bytes(&self) -> usize {
        self.s.bytes() + self.acts.bytes() + self.k.kgemv.device_bytes()
    }

    /// The whole chain: the embedding into the streams, every layer, the
    /// streams' mean into the head's input, then `head`.
    fn enqueue(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        let stream = gpu.stream();
        let (n, hcf) = (self.hp.n_embd, (self.hp.hc.eps, self.hp.hc.magnitude));
        let cx = GqaCx {
            what: STEP,
            gpu,
            w,
            kgemv: &self.k.kgemv,
            g32: &self.k.g32,
            flash: &self.k.flash,
            mma: false,
            out_gate: None,
            hidden: n,
            eps: self.hp.rms_eps,
            n_kv: 1,
            ctx: self.ctx,
        };
        let Scratch {
            x,
            x2,
            hc,
            u,
            un,
            br,
            ids,
            pos0,
            pos,
            n_keys,
            lane,
            kda,
            mla: mla_b,
            ffn: ffn_s,
        } = &mut self.s;
        let embd = names::token_embd();
        let Some(DevWeight::KQuant { ty, w: table, .. }) = w.get(&embd) else {
            return Err(GpuError::tensor(STEP, &embd, "K-quant rows"));
        };
        gpu.elem().enqueue_embed_rows_kquant(
            stream,
            *ty,
            EmbedRowsArgs {
                w: table,
                ids,
                pos0,
                first: 0,
                y: x,
                pos,
                n_keys,
            },
        )?;
        let mhc = &self.k.hc;
        for (l, layer) in self.layers.iter_mut().enumerate() {
            let [attn_hc, ffn_hc] = &layer.hc;
            let vecs = |[f, scale, base]: &[String; 3]| -> Result<SiteVecs<'_>, GpuError> {
                Ok(SiteVecs {
                    w: w.f32_buf(STEP, f)?,
                    scale: w.f32_buf(STEP, scale)?,
                    base: w.f32_buf(STEP, base)?,
                })
            };
            mhc.enqueue_route(stream, &vecs(attn_hc)?, x, (n, 1), hcf, hc)?;
            mhc.enqueue_fold(stream, x, hc, (n, 1), u)?;
            match &mut layer.mixer {
                Mixer::Rda {
                    sites,
                    state,
                    stamp,
                    ring,
                } => {
                    let [
                        qkv,
                        fa,
                        ga,
                        beta_raw,
                        write_raw,
                        f,
                        z,
                        conv,
                        beta,
                        decay,
                        o,
                        gated,
                    ] = &mut *kda;
                    let st = KdaState {
                        state,
                        stamp,
                        ring,
                        lane,
                        lanes: 1,
                        each: false,
                        row: 0,
                    };
                    let bufs = KdaBufs {
                        x: u,
                        pos,
                        xn: un,
                        qkv,
                        fa,
                        ga,
                        beta_raw,
                        write_raw: Some(write_raw),
                        f,
                        z,
                        conv,
                        beta,
                        decay,
                        o,
                        gated,
                        out: br,
                    };
                    rda::step(
                        &cx,
                        &self.k.linear,
                        &mut self.acts,
                        sites,
                        (l, &self.kda),
                        st,
                        bufs,
                    )?;
                }
                Mixer::Mla { sites, cache } => {
                    let [qa, qr, kv, gate, q, qabs, att, av, gated, part_v, part_ms] = &mut *mla_b;
                    let bufs = MlaBufs {
                        x: u,
                        pos,
                        n_keys,
                        xn: un,
                        qa,
                        qr,
                        kv,
                        gate,
                        q,
                        qabs,
                        att,
                        av,
                        gated,
                        part_v,
                        part_ms,
                        out: br,
                        cache,
                    };
                    mla::step(
                        &cx,
                        &self.k.mla,
                        &mut self.acts,
                        (&**sites, &self.mla),
                        l,
                        bufs,
                    )?;
                }
            }
            mhc.enqueue_post(stream, (x, br, hc), (n, 1), x2)?;
            mhc.enqueue_route(stream, &vecs(ffn_hc)?, x2, (n, 1), hcf, hc)?;
            mhc.enqueue_fold(stream, x2, hc, (n, 1), u)?;
            let gain = w.f32_buf(STEP, &names::ffn_norm(l))?;
            gpu.elem()
                .enqueue_rms_norm(stream, u, gain, self.hp.rms_eps, n, 1, un)?;
            ffn::ffn(
                &cx,
                &self.k.ffn,
                &mut self.acts,
                (self.act, self.hp.experts.weights_scale),
                (l, &layer.ffn),
                un,
                ffn_s,
                br,
            )?;
            mhc.enqueue_post(stream, (x2, br, hc), (n, 1), x)?;
        }
        mhc.enqueue_mean(stream, x, (n, 1), head.input_mut())?;
        head.enqueue(gpu, w)
    }
}

impl ChainBody for Body {
    type Input = DecodeInput;
    type Host = NoHost;
    const UNHOSTED_POOL: bloomery_levers::UnhostedPool = bloomery_levers::UnhostedPool::NotCarried;

    fn arch() -> Arch {
        Arch::Raon1
    }

    /// The step at `pos`; a position past the caches is refused by name.
    fn decode_input(&mut self, token: u32, pos: u32) -> Result<DecodeInput, GpuError> {
        if pos as usize >= self.ctx {
            return Err(GpuError::shape(
                "raon1::Body::decode_input",
                format!("a step at position {pos} in caches of {}", self.ctx),
            ));
        }
        Ok(DecodeInput { token, pos })
    }

    /// The token once a stream and the position: two copies.
    fn refresh(&mut self, stream: &CudaStream, input: &DecodeInput) -> Result<(), GpuError> {
        self.ids = [input.token; STREAMS];
        self.s.ids.copy_from_host(stream, &self.ids)?;
        Ok(self.s.pos0.copy_from_host(stream, &[input.pos])?)
    }

    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        self.enqueue(gpu, w, head)
    }

    /// Every RDA state and conv ring back to zero in place (a captured chain
    /// keeps their addresses) with its lane at position 0. The latent caches
    /// need nothing: no row at or past the live count is read, and every row
    /// below it is written by its own step first.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        let stream = gpu.stream();
        for layer in &mut self.layers {
            if let Mixer::Rda {
                state, stamp, ring, ..
            } = &mut layer.mixer
            {
                state.zero_async(stream)?;
                ring.zero_async(stream)?;
                stamp.copy_from_host(stream, &[0u32])?;
            }
        }
        Ok(())
    }

    /// `attention.layer_norm_rms_epsilon`: every block norm's, the head's
    /// included.
    fn head_eps(&self) -> f32 {
        self.hp.rms_eps
    }

    fn resident_bytes(&self) -> usize {
        let stores: usize = self.layers.iter().map(Layer::store_bytes).sum();
        stores + self.scratch_bytes()
    }

    fn layers(&self) -> Range<usize> {
        0..self.layers.len()
    }

    fn host(&mut self) -> Option<&mut NoHost> {
        None
    }
}
