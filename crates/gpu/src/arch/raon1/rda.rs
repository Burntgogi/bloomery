//! A Raon Delta Attention layer on the card: the KDA mixer
//! ([`crate::linear::mixer_kda`], GLM-5.3-Flash's bounded-sigmoid decay and
//! `σ(b)` erase gate) with the one fact Raon adds, a write gate apart from
//! the erase gate (`σ(b + w) / σ(b)` on the values, [`crate::linear::write_gate`]),
//! over Raon's typed projections ([`RdaSites`]).
//!
//! What the file forces here, each read from its header at load:
//! - q, k, v and the output gate `g` are full-rank projections of the normed
//!   input (`attn_q`, `attn_k`, `attn_v`, `ssm_g`; Q6_K in the Q4_K_M file),
//!   read through one q8_1 form of it when they are K-quants; `g` writes the
//!   gate rows `z` directly (Raon has no low-rank gate);
//! - β's raw projection `ssm_beta`, the write projection `ssm_rda` and the
//!   forget projection's halves `ssm_f_a` (2560 → 128) and `ssm_f_b` (128 →
//!   heads·128), F16 in the file and widened to f32 at load;
//! - the output projection `attn_output` (Q8_0).
//!
//! The step is one row (`m` = 1); a prompt runs its rows as steps.

use crate::Gpu;
use crate::GpuError;
use crate::arch::hybrid::{Acts, GqaCx, Site};
use crate::linear::mixer_kda::{
    self, KdaBufs, KdaDecay, KdaDims, KdaIn, KdaProj, KdaRule, KdaState, KdaVecs,
};
use crate::linear::{CONV_TAPS, HEAD, KHeadMap, LinearKernels, LinearShape};
use crate::site::{self, Form, SiteTy};
use crate::tensor::WindowMut;
use crate::weights::Weights;
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::arch::raon1::names;

/// What the refusals of this file name.
const WHAT: &str = "raon1 rda";

/// Types a full-rank projection of the normed input may have.
const DENSE: &[SiteTy] = &[
    SiteTy::Q3K,
    SiteTy::Q4K,
    SiteTy::Q5K,
    SiteTy::Q6K,
    SiteTy::Q8_0,
    SiteTy::F32,
];

/// The joined conv taps of RDA layer `l`'s q, k and v, in the conv's channel
/// order: the one weight the mixer's conv reads, made at load ([`derive`]).
fn conv_joined(l: usize) -> String {
    format!("derived.blk.{l}.ssm_conv1d_qkv")
}

/// Join layer `l`'s three conv tap sets into the one vector the mixer reads.
/// Load-time only; the body calls it for every RDA layer before
/// [`RdaSites::load`].
pub(crate) fn derive(stream: &CudaStream, w: &mut Weights, l: usize) -> Result<(), GpuError> {
    let parts = ['q', 'k', 'v'].map(|p| names::ssm_conv1d(l, p));
    let parts: Vec<&str> = parts.iter().map(String::as_str).collect();
    w.join_rows(stream, &parts, conv_joined(l))
}

/// The mixer's widths and rule for Raon's RDA layers: `heads` query/key and
/// value heads of [`HEAD`], the RMS epsilon, and the decay's lower bound
/// (`kda.gate_lower_bound`, below 0).
pub(crate) fn dims(embd: usize, heads: usize, rms_eps: f32, lower_bound: f32) -> KdaDims {
    KdaDims {
        embd,
        shape: LinearShape {
            n_k: heads,
            n_v: heads,
            map: KHeadMap::Tiled,
        },
        rms_eps,
        rule: KdaRule {
            decay: KdaDecay::Sigmoid { lower_bound },
            beta_scale: 1.0,
            write_gate: true,
        },
    }
}

/// One RDA layer's typed projections and the names of its f32 vectors.
pub(crate) struct RdaSites {
    /// q, k and v, written into the joined rows' three windows.
    pub qkv: [Site; 3],
    /// The output gate's full-rank projection, written into `z`.
    pub g: Site,
    pub beta: Site,
    pub write: Site,
    pub f_a: Site,
    pub f_b: Site,
    pub out: Site,
    /// attn_norm, the joined conv taps, dt bias, `ssm_a`, the gate norm.
    pub vecs: [String; 5],
}

impl RdaSites {
    /// Layer `l`'s sites at `d`'s widths, each type read from `file`'s header
    /// and checked against the resident weight, and its vectors checked by
    /// length. Load-time only, after [`derive`].
    pub(crate) fn load(
        file: &Split,
        w: &Weights,
        d: &KdaDims,
        l: usize,
    ) -> Result<RdaSites, GpuError> {
        let embd = d.embd;
        let lin = d.shape;
        let (qk, v) = (lin.n_k * HEAD, lin.n_v * HEAD);
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
        let f_a = names::ssm_f_a(l);
        let low_rank = w.get(&f_a).map_or(0, |_| {
            file.find(&f_a)
                .and_then(|(_, t)| t.dims.get(1).copied())
                .and_then(|r| usize::try_from(r).ok())
                .unwrap_or(0)
        });
        if low_rank == 0 {
            return Err(GpuError::tensor(
                WHAT,
                f_a,
                "resident, with a positive rank",
            ));
        }
        Ok(RdaSites {
            qkv: [
                typed(names::attn_q(l), (qk, embd))?,
                typed(names::attn_k(l), (qk, embd))?,
                typed(names::attn_v(l), (v, embd))?,
            ],
            g: typed(names::ssm_g(l), (v, embd))?,
            beta: typed(names::ssm_beta(l), (lin.n_v, embd))?,
            write: typed(names::ssm_rda(l), (lin.n_v, embd))?,
            f_a: typed(names::ssm_f_a(l), (low_rank, embd))?,
            f_b: typed(names::ssm_f_b(l), (v, low_rank))?,
            out: typed(names::attn_output(l), (embd, v))?,
            vecs: [
                vector(names::attn_norm(l), embd)?,
                vector(conv_joined(l), lin.channels() * CONV_TAPS)?,
                vector(names::ssm_dt_bias(l), v)?,
                vector(names::ssm_a(l), lin.n_v)?,
                vector(names::ssm_norm(l), HEAD)?,
            ],
        })
    }

    /// Every site the layer launches, for the load's [`Acts::new`].
    pub(crate) fn sites(&self) -> [&Site; 9] {
        let [q, k, v] = &self.qkv;
        [
            q,
            k,
            v,
            &self.g,
            &self.beta,
            &self.write,
            &self.f_a,
            &self.f_b,
            &self.out,
        ]
    }

    /// The layer's resident f32 vectors, as the mixer reads them.
    pub(crate) fn vecs<'w>(&self, w: &'w Weights) -> Result<KdaVecs<'w>, GpuError> {
        let [norm, conv, dt_bias, a, gate_norm] = &self.vecs;
        Ok(KdaVecs {
            norm: w.f32_buf(WHAT, norm)?,
            conv: w.f32_buf(WHAT, conv)?,
            dt_bias: w.f32_buf(WHAT, dt_bias)?,
            a: w.f32_buf(WHAT, a)?,
            gate_norm: w.f32_buf(WHAT, gate_norm)?,
        })
    }
}

/// The step's projections of one RDA layer: its sites, the activation forms
/// they read and the launch context.
struct Step<'a> {
    cx: &'a GqaCx<'a>,
    sites: &'a RdaSites,
    shape: LinearShape,
    acts: &'a mut Acts,
}

/// The window of `y` input site `i` writes (in [`RdaSites::sites`] order):
/// q, k, v in the joined rows, then `z`, β's raw rows, the write gate's raw
/// rows and the forget projection's first half.
fn window<'b>(
    y: &'b mut KdaIn<'_>,
    i: usize,
    shape: LinearShape,
) -> Result<WindowMut<'b, f32>, GpuError> {
    let (qk, v) = (shape.n_k * HEAD, shape.n_v * HEAD);
    let whole = |b: &'b mut DeviceBuffer<f32>| {
        let len = b.len();
        WindowMut::of_mut(b, 0, len)
    };
    match i {
        0 => WindowMut::of_mut(&mut *y.qkv, 0, qk),
        1 => WindowMut::of_mut(&mut *y.qkv, 4 * qk, qk),
        2 => WindowMut::of_mut(&mut *y.qkv, 8 * qk, v),
        3 => whole(&mut *y.z),
        4 => whole(&mut *y.beta_raw),
        5 => whole(y.write_raw.as_deref_mut().ok_or(GpuError::state(
            WHAT,
            "the write projection's rows (the rule has a write gate)",
        ))?),
        6 => whole(&mut *y.fa),
        _ => Err(GpuError::state(WHAT, "an input site past the seventh")),
    }
}

impl KdaProj for Step<'_> {
    /// q, k, v, the gate rows `z`, β's and the write gate's raw rows and the
    /// forget projection's first half from the normed row — the K-quant sites
    /// through one q8_1 form of it, the others each through its own — then
    /// `f = f_b · fa`. `ga` is not written: Raon's gate is full-rank.
    fn input(
        &mut self,
        gpu: &Gpu,
        l: usize,
        m: usize,
        xn: &DeviceBuffer<f32>,
        mut y: KdaIn<'_>,
    ) -> Result<(), GpuError> {
        if m != 1 {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {l}: {m} rows through the step's gemvs; a prompt runs its rows as steps"
                ),
            ));
        }
        let (cx, s, shape) = (self.cx, self.sites, self.shape);
        let all = s.sites();
        let input = &all[..7];
        let q8 = |t: &Site| t.ty().reads() == Form::Q8x128;
        if input.iter().any(|t| q8(t)) {
            let act = self.acts.q8_1_at(WHAT, cx.hidden, 1)?;
            gpu.enqueue_quantize_q8_1_layer(xn, act, l)?;
            for (i, t) in input.iter().enumerate().filter(|(_, t)| q8(t)) {
                let mut win = window(&mut y, i, shape)?;
                t.kgemv(cx, act, l, &mut win)?;
            }
        }
        for (i, t) in input.iter().enumerate().filter(|(_, t)| !q8(t)) {
            let mut win = window(&mut y, i, shape)?;
            t.project(cx, self.acts, l, xn, &mut win)?;
        }
        s.f_b.project(cx, self.acts, l, y.fa, y.f)
    }

    /// `out = o_proj · gated`: the site quantizes `gated` into its own form.
    fn output(
        &mut self,
        _gpu: &Gpu,
        l: usize,
        m: usize,
        gated: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        if m != 1 {
            return Err(GpuError::shape(WHAT, format!("layer {l}: {m} rows")));
        }
        self.sites.out.project(self.cx, self.acts, l, gated, out)
    }
}

/// Enqueue RDA layer `l` at one row (module doc): `bufs.x` the layer's input
/// (the hyper-connections' fold), `bufs.out` the update the layer hands
/// back, the store `st` read and written in place. `d` is [`dims`]'s,
/// `bufs.write_raw` lent.
pub(crate) fn step(
    cx: &GqaCx<'_>,
    k: &LinearKernels,
    acts: &mut Acts,
    sites: &RdaSites,
    (l, d): (usize, &KdaDims),
    st: KdaState<'_>,
    bufs: KdaBufs<'_>,
) -> Result<(), GpuError> {
    let gpu = cx.gpu;
    let vecs = sites.vecs(cx.w)?;
    let mut proj = Step {
        cx,
        sites,
        shape: d.shape,
        acts,
    };
    mixer_kda::enqueue(gpu, gpu.stream(), k, &mut proj, (l, 1, d), &vecs, st, bufs)
}
