//! A Raon gated MLA layer on the card at one row: absorbed latent attention
//! with no positional encoding, its `rope.dimension_count` channels an
//! unrotated extra key part shared by the heads, and a sigmoid output gate
//! (llama.cpp `src/models/raon1.cpp` `build_mla_layer`, the HF
//! `Raon1MiniMLA`).
//!
//! The launches, in order, over common owners:
//! 1. `attn_norm` RMS of `x`;
//! 2. `attn_q_a` and `attn_kv_a_mqa` of the normed row (through one q8_1
//!    form when they are K-quants) and `attn_gate` (its own form);
//! 3. `attn_q_a_norm` RMS of the q_a rows (the latent ε);
//! 4. `attn_q_b`: each head's `[nope | extra]` query;
//!
//! then [`attend`], the core the kernel gate runs:
//! 5. the cache row at the position: the raw extra part, then the latent's
//!    RMS with `attn_kv_a_norm` (`latent::enqueue_latent_extra_append`), f16;
//! 6. `attn_k_b` per head: each head's nope query absorbed into the latent
//!    width, behind its extra part (`enqueue_q8_0_gemv_heads`);
//! 7. each head's extra query part copied in front (`enqueue_gather` over
//!    load-time tables);
//! 8. the latent attention over the cache rows `0 ..= pos`
//!    (`flash::enqueue_flash_latent_split`, one or two launches), the row
//!    `[extra | latent]` the key and its latent the value, at the scale
//!    `1/√(nope + extra)`;
//! 9. `attn_v_b` per head: each head's latent output to its values
//!    (`q6k_heads` for Q6_K, `enqueue_q8_0_gemv_heads` for Q8_0);
//! 10. `attn · σ(gate)` (`linear::out_gate`);
//!
//! and last, 11. `attn_output` into `out`.
//!
//! What the file forces here: the NoPE extra part and the output gate; the
//! widths are the header's ([`MlaDims::of`]).

use crate::arch::hybrid::{Acts, GqaCx, Site};
use crate::fault::FaultSink;
use crate::flash::{FlashGeom, FlashInputs, FlashSplitArgs, partials_ms_len, partials_v_len};
use crate::latent::{LATENT, LatentExtraAppendArgs, LatentKernels, Rows};
use crate::linear::out_gate::{OutGateArgs, OutGateKernels};
use crate::model::{Q8_0GemvHeadsArgs, StepKernels};
use crate::q6k_heads::{Q6kHeadsArgs, Q6kHeadsKernels};
use crate::site::{self, Form, SiteTy};
use crate::tensor::DeviceTensor;
use crate::weights::{DevWeight, Weights};
use crate::{Gpu, GpuError};
use cuda_core::DeviceBuffer;
use gguf::Split;
use gguf::quant::GgmlType;
use model::arch::raon1::names;

/// What the refusals of this file name.
const WHAT: &str = "raon1 mla";

/// Types a dense projection of the layer may have.
const DENSE: &[SiteTy] = &[
    SiteTy::Q3K,
    SiteTy::Q4K,
    SiteTy::Q5K,
    SiteTy::Q6K,
    SiteTy::Q8_0,
    SiteTy::F32,
];

/// The layer's widths, from the file's header.
#[derive(Clone, Copy, Debug)]
pub struct MlaDims {
    pub embd: usize,
    pub heads: usize,
    pub q_lora: usize,
    /// A query head's values before the absorption.
    pub nope: usize,
    /// The unrotated key values beside the latent, shared by the heads.
    pub extra: usize,
    pub v_head: usize,
    pub rms_eps: f32,
    /// The latent norms' ε (`attention.latent_norm_rms_epsilon`).
    pub latent_eps: f32,
}

impl MlaDims {
    /// The key width a score's scale is `1/√` of.
    #[must_use]
    pub fn head_k(&self) -> usize {
        self.nope + self.extra
    }

    /// The cache row: the extra part, then the latent.
    #[must_use]
    pub fn row(&self) -> usize {
        self.extra + LATENT
    }

    /// Refuse by name a shape no launch here runs.
    pub fn check(&self) -> Result<(), GpuError> {
        if self.v_head != crate::linear::HEAD {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the output gate's entry is 128 a head, got v_head {}",
                    self.v_head
                ),
            ));
        }
        if self.row() != 576 {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the latent attention's tensor-core pass reads 576-wide rows, got extra {} + \
                     latent {LATENT}",
                    self.extra
                ),
            ));
        }
        Ok(())
    }
}

/// How a layer's per-head value projection runs, by the file type of
/// `attn_v_b`.
enum ValueHeads {
    Q6K(String),
    Q8_0(String),
}

/// A resident per-head value projection, as [`attend`] reads it.
pub enum ValueW<'a> {
    /// Q6_K words, `heads·v_head` rows of `LATENT`.
    Q6K(&'a DeviceTensor<u32>),
    /// Q8_0 planes (`qs`, `d`).
    Q8_0(&'a DeviceTensor<u32>, &'a DeviceTensor<u16>),
}

/// One MLA layer's widths, projections and the names of its vectors.
pub(crate) struct MlaSites {
    q_a: Site,
    kv_a: Site,
    gate: Site,
    q_b: Site,
    out: Site,
    /// `attn_k_b`, Q8_0: each head's `LATENT` rows over its nope values.
    k_b: String,
    v_b: ValueHeads,
    /// attn_norm, attn_q_a_norm, attn_kv_a_norm.
    vecs: [String; 3],
}

impl MlaSites {
    /// Layer `l`'s sites at `d`'s widths, each type read from `file`'s
    /// header and checked against the resident weight. Load-time only.
    pub(crate) fn load(
        file: &Split,
        w: &Weights,
        d: &MlaDims,
        l: usize,
    ) -> Result<MlaSites, GpuError> {
        d.check()?;
        let typed = |name: String, (rows, k): (usize, usize)| -> Result<Site, GpuError> {
            let ty = site::file_site(file, WHAT, &name, (rows, k), DENSE)?;
            site::site(w, WHAT, &name, (rows, k), ty)?;
            Ok(Site::new(name, ty, k))
        };
        let vector = |name: String, len: usize| -> Result<String, GpuError> {
            let got = w.f32_buf(WHAT, &name)?.len();
            if got != len {
                return Err(GpuError::shape(
                    WHAT,
                    format!("{name} holds {got} values; the layer reads {len}"),
                ));
            }
            Ok(name)
        };
        let ty_of = |name: &str| file.find(name).map(|(_, t)| t.ty);
        let k_b = names::attn_k_b(l);
        if ty_of(&k_b) != Some(GgmlType::Q8_0) {
            return Err(GpuError::shape(
                WHAT,
                format!("{k_b} is {:?}; the per-head absorb runs Q8_0", ty_of(&k_b)),
            ));
        }
        let v_name = names::attn_v_b(l);
        let v_b = match ty_of(&v_name) {
            Some(GgmlType::Q6_K) => ValueHeads::Q6K(v_name),
            Some(GgmlType::Q8_0) => ValueHeads::Q8_0(v_name),
            other => {
                return Err(GpuError::shape(
                    WHAT,
                    format!("{v_name} is {other:?}; the per-head value runs Q6_K or Q8_0"),
                ));
            }
        };
        let (embd, hv) = (d.embd, d.heads * d.v_head);
        Ok(MlaSites {
            q_a: typed(names::attn_q_a(l), (d.q_lora, embd))?,
            kv_a: typed(names::attn_kv_a_mqa(l), (d.row(), embd))?,
            gate: typed(names::attn_gate(l), (hv, embd))?,
            q_b: typed(names::attn_q_b(l), (d.heads * d.head_k(), d.q_lora))?,
            out: typed(names::attn_output(l), (embd, hv))?,
            k_b,
            v_b,
            vecs: [
                vector(names::attn_norm(l), embd)?,
                vector(names::attn_q_a_norm(l), d.q_lora)?,
                vector(names::attn_kv_a_norm(l), LATENT)?,
            ],
        })
    }

    /// Every site the layer launches, for the load's [`Acts::new`].
    pub(crate) fn sites(&self) -> [&Site; 5] {
        [&self.q_a, &self.kv_a, &self.gate, &self.q_b, &self.out]
    }
}

/// The kernels every MLA layer launches beyond the `Gpu`'s own, and the
/// extra part's gather tables (`[h·head_k + nope + j] → [h·row + j]`).
pub struct MlaKernels {
    step: StepKernels,
    latent: LatentKernels,
    q6k: Q6kHeadsKernels,
    out_gate: OutGateKernels,
    src: DeviceBuffer<u32>,
    dst: DeviceBuffer<u32>,
}

impl MlaKernels {
    /// Load the modules and build the gather tables for `d`. Load-time only.
    pub fn load(gpu: &Gpu, d: &MlaDims) -> Result<MlaKernels, GpuError> {
        d.check()?;
        let at = |v: usize| {
            u32::try_from(v).map_err(|_| GpuError::shape(WHAT, "a gather index past u32"))
        };
        let mut src = Vec::with_capacity(d.heads * d.extra);
        let mut dst = Vec::with_capacity(d.heads * d.extra);
        for h in 0..d.heads {
            for j in 0..d.extra {
                src.push(at(h * d.head_k() + d.nope + j)?);
                dst.push(at(h * d.row() + j)?);
            }
        }
        let (ctx, stream) = (gpu.context(), gpu.stream());
        Ok(MlaKernels {
            step: StepKernels::load(ctx)?,
            latent: LatentKernels::load(ctx)?,
            q6k: Q6kHeadsKernels::load(ctx)?,
            out_gate: OutGateKernels::load(ctx)?,
            src: DeviceBuffer::from_host(stream, &src)?,
            dst: DeviceBuffer::from_host(stream, &dst)?,
        })
    }
}

/// The lengths of the attention's partials (`part_v`, `part_ms`) for one row
/// of `d.heads` heads over a cache of `ctx` rows.
#[must_use]
pub fn partials_lens(d: &MlaDims, ctx: usize) -> (usize, usize) {
    (partials_v_len(d.heads, ctx), partials_ms_len(d.heads, ctx))
}

/// The buffers one row of an MLA layer reads and writes: `x` the layer's
/// input, `pos` the row's position and `n_keys` its live key count (`pos +
/// 1`), the intermediates (each at least its width), the attention's
/// partials ([`partials_lens`]), `out` the update, and the layer's cache
/// (`[ctx][row]` f16).
pub(crate) struct MlaBufs<'a> {
    pub x: &'a DeviceBuffer<f32>,
    pub pos: &'a DeviceBuffer<u32>,
    pub n_keys: &'a DeviceBuffer<u32>,
    /// `[embd]`.
    pub xn: &'a mut DeviceBuffer<f32>,
    /// `[q_lora]` each.
    pub qa: &'a mut DeviceBuffer<f32>,
    pub qr: &'a mut DeviceBuffer<f32>,
    /// `[row]`: the latent, then the extra part.
    pub kv: &'a mut DeviceBuffer<f32>,
    /// `[heads·v_head]`.
    pub gate: &'a mut DeviceBuffer<f32>,
    /// `[heads·head_k]`.
    pub q: &'a mut DeviceBuffer<f32>,
    /// `[heads·row]`.
    pub qabs: &'a mut DeviceBuffer<f32>,
    /// `[heads·LATENT]`.
    pub att: &'a mut DeviceBuffer<f32>,
    /// `[heads·v_head]` each.
    pub av: &'a mut DeviceBuffer<f32>,
    pub gated: &'a mut DeviceBuffer<f32>,
    pub part_v: &'a mut DeviceBuffer<f32>,
    pub part_ms: &'a mut DeviceBuffer<f32>,
    /// `[embd]`.
    pub out: &'a mut DeviceBuffer<f32>,
    pub cache: &'a mut DeviceTensor<u16>,
}

/// [`attend`]'s own buffers: the absorbed queries (`[heads·row]`), the
/// latent outputs (`[heads·LATENT]`), the value heads' outputs
/// (`[heads·v_head]`) and the attention's partials ([`partials_lens`]).
pub struct AttnBufs<'a> {
    pub qabs: &'a mut DeviceBuffer<f32>,
    pub att: &'a mut DeviceBuffer<f32>,
    pub av: &'a mut DeviceBuffer<f32>,
    pub part_v: &'a mut DeviceBuffer<f32>,
    pub part_ms: &'a mut DeviceBuffer<f32>,
}

/// The Q8_0 planes of `name`, or a named refusal.
fn q8_planes<'w>(
    w: &'w Weights,
    name: &str,
) -> Result<(&'w DeviceTensor<u32>, &'w DeviceTensor<u16>), GpuError> {
    match w.get(name) {
        Some(DevWeight::Q8_0 { qs, d, .. }) => Ok((qs, d)),
        Some(_) => Err(GpuError::tensor(WHAT, name, "Q8_0 planes")),
        None => Err(GpuError::tensor(WHAT, name, "resident")),
    }
}

/// The resident value projection `v`, or a named refusal.
fn value_w<'w>(w: &'w Weights, v: &ValueHeads) -> Result<ValueW<'w>, GpuError> {
    match v {
        ValueHeads::Q6K(name) => match w.get(name) {
            Some(DevWeight::KQuant {
                ty: GgmlType::Q6_K,
                w: wv,
                ..
            }) => Ok(ValueW::Q6K(wv)),
            _ => Err(GpuError::tensor(WHAT, name, "resident Q6_K words")),
        },
        ValueHeads::Q8_0(name) => {
            let (qs, d) = q8_planes(w, name)?;
            Ok(ValueW::Q8_0(qs, d))
        }
    }
}

/// [`attend`]'s inputs: one row's projections, the layer's weights and the
/// row's position and live key count (`pos + 1`).
pub struct AttendIn<'a> {
    /// `[row]`: `attn_kv_a_mqa`'s output, the latent then the extra part.
    pub kv: &'a DeviceBuffer<f32>,
    /// `[heads·head_k]`: `attn_q_b`'s output, `[nope | extra]` a head.
    pub q: &'a DeviceBuffer<f32>,
    /// `[heads·v_head]`: `attn_gate`'s output.
    pub gate: &'a DeviceBuffer<f32>,
    /// `attn_kv_a_norm`, `[LATENT]`.
    pub kv_norm: &'a DeviceBuffer<f32>,
    /// `attn_k_b`'s Q8_0 planes.
    pub k_b: (&'a DeviceTensor<u32>, &'a DeviceTensor<u16>),
    pub v_b: ValueW<'a>,
    pub pos: &'a DeviceBuffer<u32>,
    pub n_keys: &'a DeviceBuffer<u32>,
    pub fault: FaultSink,
}

/// Steps 5 to 10 of the module doc for one row: the cache row at `pos`
/// written into `cache`, the gated heads into `y` (`[heads·v_head]`).
pub fn attend(
    gpu: &Gpu,
    k: &MlaKernels,
    d: &MlaDims,
    a: AttendIn<'_>,
    s: AttnBufs<'_>,
    cache: &mut DeviceTensor<u16>,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    k.latent.enqueue_latent_extra_append(
        stream,
        LatentExtraAppendArgs {
            rows: Rows {
                x: a.kv,
                stride: d.row(),
                m: 1,
            },
            off: 0,
            e_off: LATENT,
            extra: d.extra,
            gain: a.kv_norm,
            pos: a.pos,
            eps: d.latent_eps,
            fault: a.fault,
            cache: &mut *cache,
        },
    )?;
    let (qs, dd) = a.k_b;
    k.step.enqueue_q8_0_gemv_heads(
        stream,
        Q8_0GemvHeadsArgs {
            qs,
            d: dd,
            x: a.q,
            rows_per_head: LATENT,
            x_head_stride: d.head_k(),
            y_head_stride: d.row(),
            y_off: d.extra,
            y: &mut *s.qabs,
        },
    )?;
    k.step
        .enqueue_gather(stream, a.q, &k.src, &k.dst, d.heads * d.extra, s.qabs)?;
    gpu.flash().enqueue_flash_latent_split(
        stream,
        FlashSplitArgs {
            inputs: FlashInputs {
                q: s.qabs,
                kv: cache,
                n_keys_buf: a.n_keys,
                kq_scale: 1.0 / (d.head_k() as f32).sqrt(),
                geom: FlashGeom {
                    tokens: 1,
                    heads: d.heads,
                    rope_dims: d.extra,
                    latent_dims: LATENT,
                },
            },
            part_v: s.part_v,
            part_ms: s.part_ms,
            y: &mut *s.att,
        },
    )?;
    match a.v_b {
        ValueW::Q6K(wv) => k.q6k.enqueue_q6k_heads(
            stream,
            Q6kHeadsArgs {
                w: wv,
                x: s.att,
                k: LATENT,
                rows_per_head: d.v_head,
                x_head_stride: LATENT,
                y_head_stride: d.v_head,
                y_off: 0,
                y: &mut *s.av,
            },
        )?,
        ValueW::Q8_0(qs, dd) => k.step.enqueue_q8_0_gemv_heads(
            stream,
            Q8_0GemvHeadsArgs {
                qs,
                d: dd,
                x: s.att,
                rows_per_head: d.v_head,
                x_head_stride: LATENT,
                y_head_stride: d.v_head,
                y_off: 0,
                y: &mut *s.av,
            },
        )?,
    }
    k.out_gate.enqueue_out_gate(
        stream,
        OutGateArgs {
            attn: s.av,
            gate: a.gate,
            n_head: d.heads,
            m: 1,
            fault: a.fault,
            y,
        },
    )
}

/// Enqueue MLA layer `l` at one row on `b`'s buffers (module doc).
pub(crate) fn step(
    cx: &GqaCx<'_>,
    k: &MlaKernels,
    acts: &mut Acts,
    (sites, d): (&MlaSites, &MlaDims),
    l: usize,
    b: MlaBufs<'_>,
) -> Result<(), GpuError> {
    let (gpu, w) = (cx.gpu, cx.w);
    let stream = gpu.stream();
    let fault = gpu.layer_sink(l)?;
    let [norm, q_a_norm, kv_a_norm] = &sites.vecs;
    gpu.elem().enqueue_rms_norm(
        stream,
        b.x,
        w.f32_buf(WHAT, norm)?,
        d.rms_eps,
        d.embd,
        1,
        b.xn,
    )?;
    let q8 = |t: &Site| t.ty().reads() == Form::Q8x128;
    let input: [(&Site, &mut DeviceBuffer<f32>); 3] = [
        (&sites.q_a, &mut *b.qa),
        (&sites.kv_a, &mut *b.kv),
        (&sites.gate, &mut *b.gate),
    ];
    let mut rest = Vec::with_capacity(3);
    if input.iter().any(|(t, _)| q8(t)) {
        let act = acts.q8_1_at(WHAT, d.embd, 1)?;
        gpu.enqueue_quantize_q8_1_layer(b.xn, act, l)?;
        for (t, y) in input {
            if q8(t) {
                t.kgemv(cx, act, l, y)?;
            } else {
                rest.push((t, y));
            }
        }
    } else {
        rest.extend(input);
    }
    for (t, y) in rest {
        t.project(cx, acts, l, b.xn, y)?;
    }
    gpu.elem().enqueue_rms_norm(
        stream,
        b.qa,
        w.f32_buf(WHAT, q_a_norm)?,
        d.latent_eps,
        d.q_lora,
        1,
        b.qr,
    )?;
    sites.q_b.project(cx, acts, l, b.qr, b.q)?;
    attend(
        gpu,
        k,
        d,
        AttendIn {
            kv: b.kv,
            q: b.q,
            gate: b.gate,
            kv_norm: w.f32_buf(WHAT, kv_a_norm)?,
            k_b: q8_planes(w, &sites.k_b)?,
            v_b: value_w(w, &sites.v_b)?,
            pos: b.pos,
            n_keys: b.n_keys,
            fault,
        },
        AttnBufs {
            qabs: b.qabs,
            att: b.att,
            av: b.av,
            part_v: b.part_v,
            part_ms: b.part_ms,
        },
        b.cache,
        b.gated,
    )?;
    sites.out.project(cx, acts, l, b.gated, b.out)
}
