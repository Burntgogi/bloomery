//! Raon_mHC on the card: the four residual streams around every sub-layer,
//! mixed through the Helmert chart (llama.cpp `src/models/raon1.cpp`
//! `build_hc_pre` / `build_hc_post`, the HF `RaonMHC`).
//!
//! Layouts, every entry token-major: a token's streams are `4 · n` f32,
//! stream `i` at `i · n`, read flat the `K = 4n` values the selector takes.
//! A token's route is [`HC_OUT`] f32 — `pre[4]`, `post[4]`, then `comb[16]`
//! with `comb[j · 4 + i]` the weight of source stream `j` in stream `i`
//! (V4.1's HC_PRE layout, so the apply stages are the same rule).
//!
//! - [`raon_hc_kernels::raon_hc_route`], one block a token: the selector's
//!   [`HC_MIX`] dots of the widened f16 rows `hc_*_fn` with the token's
//!   streams and the streams' sum of squares, then on thread 0
//!   `mix = dot / √(ss/K + eps)`, `pre = σ(α_pre·mix + b)`,
//!   `post = mag·σ(α_post·mix + b)`, the chart `e = H z Hᵀ` of the
//!   `(n−1)²` coordinates `z`, `ρ = 1 + max(0, max(−e/B))`, `φ = e/ρ + B`
//!   and `comb = (φ − I)·σ(η) + I`, with `B = (I + 11ᵀ/n)/2` and `H` the
//!   Helmert basis.
//! - [`raon_hc_kernels::raon_hc_fold`]: the sub-layer's input `Σ_i pre_i ·
//!   s_i`, streams in order.
//! - [`raon_hc_kernels::raon_hc_post`]: stream `i` becomes `post_i · out +
//!   Σ_j comb[j][i] · s_j`, the branch first, then the sources in order (the
//!   reference's fused `dsv4_hc_post`).
//! - [`raon_hc_kernels::raon_hc_mean`]: the head's input, `((s_0 + s_1) +
//!   s_2) + s_3` then `· 1/4`.
//!
//! Every product and sum of the apply stages is rounded on its own, in the
//! reference's CPU ops' order. Raon-only today: V4.1's
//! `ds41_hc_fold` / `ds41_hc_post` / `ds41_hc_mean` are the same apply rule
//! in `bloomery-gpu-deepseek41`, which this crate cannot reach; lifting
//! them here makes the route the one per-model part.

use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{add_rn_f32, div_rn_f32, mul_rn_f32, sqrt_rn_f32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Residual streams (`hyper_connection.count`).
pub const STREAMS: usize = 4;
/// Selector rows a site: `pre`, `post` and the chart's `(n−1)²` coordinates.
pub const HC_MIX: usize = 2 * STREAMS + (STREAMS - 1) * (STREAMS - 1);
/// Route values a token: `pre`, `post`, `comb`.
pub const HC_OUT: usize = 2 * STREAMS + STREAMS * STREAMS;
const _: () = assert!(HC_MIX == 17 && HC_OUT == 24);
/// The most tokens one route launch takes (a block each).
pub const MAX_TOKENS: usize = 512;
/// Threads of a route block: eight warps.
const ROUTE_THREADS: u32 = 256;
const ROUTE_WARPS: usize = 8;
/// Threads of the element-wise entries.
const ELEM_THREADS: u32 = 256;

/// Helmert basis entry `(i, j)` of `n` streams: `1` for `i ≤ j`, `−(j+1)`
/// for `i = j + 1`, else 0, over `√((j+1)(j+2))` — the reference's f32
/// values (`raon1.cpp`'s graph constructor).
#[inline(always)]
fn helmert(i: usize, j: usize) -> f32 {
    let num = if i <= j {
        1.0
    } else if i == j + 1 {
        -((j + 1) as f32)
    } else {
        0.0
    };
    div_rn_f32(num, sqrt_rn_f32(((j + 1) * (j + 2)) as f32))
}

/// `1 / (1 + e^(0 − x))`, ggml's sigmoid with ik's exponential.
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    crate::linear::sigmoid(x)
}

#[allow(
    dead_code,
    reason = "#[cuda_module] derives an `_unchecked` launcher per entry that no caller uses"
)]
#[cuda_module]
mod raon_hc_kernels {
    use super::*;

    /// One token's route (module doc), block `t` token `t`: `w` the
    /// [`HC_MIX`] selector rows of `4n` f32, `scale` the four scales
    /// (`α_pre`, `α_post`, `α_z`, `η`), `base` the [`HC_MIX`] biases, `x` the
    /// tokens' streams; writes [`HC_OUT`] values a token into `hc`.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            w.len() >= 68 * n,
            scale.len() >= 4,
            base.len() >= 17,
            x.len() >= m * 4 * n,
            hc.len() >= m * 24
        )
    )]
    pub fn raon_hc_route(
        w: &[f32],
        scale: &[f32],
        base: &[f32],
        x: &[f32],
        n: u32,
        m: u32,
        eps: f32,
        magnitude: f32,
        mut hc: DisjointSlice<f32>,
    ) {
        static mut MIX: SharedArray<f32, HC_MIX> = SharedArray::UNINIT;
        static mut SS: SharedArray<f32, ROUTE_WARPS> = SharedArray::UNINIT;

        let t = thread::blockIdx_x() as usize;
        if t >= m as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let wi = tid / 32;
        let lane = warp::lane_id() as usize;
        let k = STREAMS * n as usize;
        let xt = t * k;
        // SAFETY: MIX and SS are this block's own shared allocations; the
        // raw form reaches each `static mut` without a reference.
        let (mix, ss) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut MIX),
                SharedArray::as_raw_mut_ptr(&raw mut SS),
            )
        };
        let mut sq = 0.0f32;
        let mut i = tid;
        while i < k {
            // SAFETY: xt + i < (t + 1)·k <= m·k <= x.len() by the contract.
            let v = unsafe { *x.get_unchecked(xt + i) };
            sq = v.mul_add(v, sq);
            i += ROUTE_THREADS as usize;
        }
        let sq = warp::reduce_sum_f32(sq);
        if lane == 0 {
            // SAFETY: wi < ROUTE_WARPS; lane 0 of each warp writes its slot.
            unsafe { *ss.add(wi) = sq };
        }
        let mut r = wi;
        while r < HC_MIX {
            let row = r * k;
            let mut acc = 0.0f32;
            let mut i = lane;
            while i < k {
                // SAFETY: row + i < 17·k <= w.len() and xt + i < x.len() by
                // the contract.
                let (a, b) = unsafe { (*w.get_unchecked(row + i), *x.get_unchecked(xt + i)) };
                acc = a.mul_add(b, acc);
                i += 32;
            }
            let acc = warp::reduce_sum_f32(acc);
            if lane == 0 {
                // SAFETY: r < HC_MIX; this warp alone owns row r.
                unsafe { *mix.add(r) = acc };
            }
            r += ROUTE_WARPS;
        }
        thread::sync_threads();
        if tid != 0 {
            return;
        }
        let mut sum = 0.0f32;
        let mut s = 0usize;
        while s < ROUTE_WARPS {
            // SAFETY: s < ROUTE_WARPS, written above, past the barrier.
            sum = add_rn_f32(sum, unsafe { *ss.add(s) });
            s += 1;
        }
        let rms = sqrt_rn_f32(add_rn_f32(div_rn_f32(sum, k as f32), eps));
        // SAFETY: scale.len() >= 4 and base.len() >= HC_MIX by the contract.
        let (a_pre, a_post, a_z, eta) = unsafe {
            (
                *scale.get_unchecked(0),
                *scale.get_unchecked(1),
                *scale.get_unchecked(2),
                *scale.get_unchecked(3),
            )
        };
        let affine = |r: usize, a: f32| -> f32 {
            // SAFETY: r < HC_MIX, written above, past the barrier; base.len()
            // >= HC_MIX by the contract.
            let (v, b) = unsafe { (*mix.add(r), *base.get_unchecked(r)) };
            add_rn_f32(mul_rn_f32(div_rn_f32(v, rms), a), b)
        };
        let o = t * HC_OUT;
        let mut i = 0usize;
        while i < STREAMS {
            let pre = sigmoid(affine(i, a_pre));
            let post = mul_rn_f32(sigmoid(affine(STREAMS + i, a_post)), magnitude);
            // SAFETY: o + 8 <= m·HC_OUT <= hc.len() by the contract; thread 0
            // of block t alone writes token t's values.
            unsafe {
                *hc.get_unchecked_mut(o + i) = pre;
                *hc.get_unchecked_mut(o + STREAMS + i) = post;
            }
            i += 1;
        }
        const NM: usize = STREAMS - 1;
        let mut z = [[0.0f32; NM]; NM];
        let mut a = 0usize;
        while a < NM {
            let mut b = 0usize;
            while b < NM {
                z[a][b] = affine(2 * STREAMS + a * NM + b, a_z);
                b += 1;
            }
            a += 1;
        }
        // hz[i][a] = Σ_b H[i][b] · z[a][b], then e[i][j] = Σ_a H[j][a] ·
        // hz[i][a]: the reference's two products in their order.
        let mut hz = [[0.0f32; NM]; STREAMS];
        let mut i = 0usize;
        while i < STREAMS {
            let mut a = 0usize;
            while a < NM {
                let mut acc = 0.0f32;
                let mut b = 0usize;
                while b < NM {
                    acc = add_rn_f32(acc, mul_rn_f32(helmert(i, b), z[a][b]));
                    b += 1;
                }
                hz[i][a] = acc;
                a += 1;
            }
            i += 1;
        }
        let mut e = [[0.0f32; STREAMS]; STREAMS];
        let mut worst = f32::NEG_INFINITY;
        let mut i = 0usize;
        while i < STREAMS {
            let mut j = 0usize;
            while j < STREAMS {
                let mut acc = 0.0f32;
                let mut a = 0usize;
                while a < NM {
                    acc = add_rn_f32(acc, mul_rn_f32(helmert(j, a), hz[i][a]));
                    a += 1;
                }
                e[i][j] = acc;
                let chart = chart_base(i, j);
                let ratio = 0.0 - div_rn_f32(acc, chart);
                if ratio > worst {
                    worst = ratio;
                }
                j += 1;
            }
            i += 1;
        }
        let rho = add_rn_f32(if worst > 0.0 { worst } else { 0.0 }, 1.0);
        let gamma = sigmoid(eta);
        let mut i = 0usize;
        while i < STREAMS {
            let mut j = 0usize;
            while j < STREAMS {
                let id = if i == j { 1.0 } else { 0.0 };
                let phi = add_rn_f32(div_rn_f32(e[i][j], rho), chart_base(i, j));
                let comb = add_rn_f32(mul_rn_f32(add_rn_f32(phi, 0.0 - id), gamma), id);
                // The chart's (i, j) is source i's weight in stream j: the
                // reference's `comb` at (dst j, src i), stored at src·4 + dst.
                // SAFETY: o + 2·STREAMS + i·STREAMS + j < (t + 1)·HC_OUT <=
                // hc.len() by the contract; thread 0 of block t alone writes
                // token t's values.
                unsafe { *hc.get_unchecked_mut(o + 2 * STREAMS + i * STREAMS + j) = comb };
                j += 1;
            }
            i += 1;
        }
    }

    /// `y[t·n + d] = Σ_i pre_i · s_i[d]` of token `t`, `i` ascending, each
    /// product and sum rounded: one thread a (token, value).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            x.len() >= m * 4 * n,
            hc.len() >= m * 24,
            y.len() >= m * n
        )
    )]
    pub fn raon_hc_fold(x: &[f32], hc: &[f32], n: u32, m: u32, mut y: DisjointSlice<f32>) {
        let g = thread::index_1d().get();
        let n = n as usize;
        if g >= m as usize * n {
            return;
        }
        let (t, d) = (g / n, g % n);
        let mut acc = 0.0f32;
        let mut i = 0usize;
        while i < STREAMS {
            // SAFETY: t < m, i < 4 and d < n bound both reads by the contract.
            let (s, p) = unsafe {
                (
                    *x.get_unchecked(t * STREAMS * n + i * n + d),
                    *hc.get_unchecked(t * HC_OUT + i),
                )
            };
            let term = mul_rn_f32(s, p);
            acc = if i == 0 { term } else { add_rn_f32(acc, term) };
            i += 1;
        }
        // SAFETY: g < m·n <= y.len(); thread g alone writes it.
        unsafe { *y.get_unchecked_mut(g) = acc };
    }

    /// `next[t·4n + i·n + d] = post_i · out[t·n + d] + Σ_j comb[j][i] ·
    /// res_j[d]`, the branch then the sources ascending, each product and
    /// sum rounded: one thread a (token, stream, value).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            res.len() >= m * 4 * n,
            out.len() >= m * n,
            hc.len() >= m * 24,
            next.len() >= m * 4 * n
        )
    )]
    pub fn raon_hc_post(
        res: &[f32],
        out: &[f32],
        hc: &[f32],
        n: u32,
        m: u32,
        mut next: DisjointSlice<f32>,
    ) {
        let g = thread::index_1d().get();
        let n = n as usize;
        if g >= m as usize * STREAMS * n {
            return;
        }
        let (t, r) = (g / (STREAMS * n), g % (STREAMS * n));
        let (i, d) = (r / n, r % n);
        let base = t * STREAMS * n;
        let o = t * HC_OUT;
        // SAFETY: t < m, d < n and i < 4 bound both reads by the contract.
        let (b, p) = unsafe {
            (
                *out.get_unchecked(t * n + d),
                *hc.get_unchecked(o + STREAMS + i),
            )
        };
        let mut acc = mul_rn_f32(b, p);
        let mut j = 0usize;
        while j < STREAMS {
            // SAFETY: as above, and j < 4.
            let (s, c) = unsafe {
                (
                    *res.get_unchecked(base + j * n + d),
                    *hc.get_unchecked(o + 2 * STREAMS + j * STREAMS + i),
                )
            };
            acc = add_rn_f32(acc, mul_rn_f32(s, c));
            j += 1;
        }
        // SAFETY: g < m·4n <= next.len(); thread g alone writes it.
        unsafe { *next.get_unchecked_mut(g) = acc };
    }

    /// `y[t·n + d] = (((s_0 + s_1) + s_2) + s_3) · 1/4` of token `t`: one
    /// thread a (token, value).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            x.len() >= m * 4 * n,
            y.len() >= m * n
        )
    )]
    pub fn raon_hc_mean(x: &[f32], n: u32, m: u32, mut y: DisjointSlice<f32>) {
        let g = thread::index_1d().get();
        let n = n as usize;
        if g >= m as usize * n {
            return;
        }
        let (t, d) = (g / n, g % n);
        let mut acc = 0.0f32;
        let mut i = 0usize;
        while i < STREAMS {
            // SAFETY: t < m, i < 4 and d < n bound the read by the contract.
            let s = unsafe { *x.get_unchecked(t * STREAMS * n + i * n + d) };
            acc = if i == 0 { s } else { add_rn_f32(acc, s) };
            i += 1;
        }
        // SAFETY: g < m·n <= y.len(); thread g alone writes it.
        unsafe { *y.get_unchecked_mut(g) = mul_rn_f32(acc, 0.25) };
    }
}

/// `B = (I + 11ᵀ/n)/2` at `(i, j)`: the reference's `scale_bias(I, 0.5,
/// 0.5/n)`.
#[inline(always)]
fn chart_base(i: usize, j: usize) -> f32 {
    let id = if i == j { 1.0 } else { 0.0 };
    add_rn_f32(mul_rn_f32(id, 0.5), 0.5 / STREAMS as f32)
}

/// One site's resident pieces: the widened selector rows, the scales and
/// the biases.
pub(crate) struct SiteVecs<'a> {
    pub(crate) w: &'a DeviceBuffer<f32>,
    pub(crate) scale: &'a DeviceBuffer<f32>,
    pub(crate) base: &'a DeviceBuffer<f32>,
}

/// The loaded module.
pub(crate) struct MhcKernels {
    module: raon_hc_kernels::LoadedModule,
}

impl MhcKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub(crate) fn load(ctx: &Arc<CudaContext>) -> Result<MhcKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launchers check its launch contracts.
        let module = unsafe { crate::shared_module!(raon_hc_kernels, ctx)? };
        Ok(MhcKernels { module })
    }

    /// The route of `m` tokens' streams `x` (`4n` a token) at site `v` into
    /// `hc` ([`HC_OUT`] a token). Asynchronous, allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "the site's three vectors, the streams, the widths and the file's two constants"
    )]
    pub(crate) fn enqueue_route(
        &self,
        stream: &CudaStream,
        v: &SiteVecs<'_>,
        x: &DeviceBuffer<f32>,
        (n, m): (usize, usize),
        (eps, magnitude): (f32, f32),
        hc: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "raon1 mhc::enqueue_route";
        let k = STREAMS * n;
        if m == 0
            || m > MAX_TOKENS
            || v.w.len() < HC_MIX * k
            || v.scale.len() < 4
            || v.base.len() < HC_MIX
            || x.len() < m * k
            || hc.len() < m * HC_OUT
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "m={m} (1..={MAX_TOKENS}), n={n}: w {} (want {}), scale {}, base {}, x {} \
                     (want {}), hc {} (want {})",
                    v.w.len(),
                    HC_MIX * k,
                    v.scale.len(),
                    v.base.len(),
                    x.len(),
                    m * k,
                    hc.len(),
                    m * HC_OUT
                ),
            ));
        }
        let mu = launch_u32(what, "m", m)?;
        let prep = self
            .module
            .prepare_raon_hc_route(LaunchConfig1D::new(mu, ROUTE_THREADS, 0))?;
        self.module.raon_hc_route(
            stream,
            &prep,
            v.w,
            v.scale,
            v.base,
            x,
            launch_u32(what, "n", n)?,
            mu,
            eps,
            magnitude,
            hc,
        )?;
        Ok(())
    }

    /// The fold of `m` tokens' streams `x` by their routes `hc` into `y`
    /// (`n` a token). Asynchronous, allocation-free, capturable.
    pub(crate) fn enqueue_fold(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        hc: &DeviceBuffer<f32>,
        (n, m): (usize, usize),
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "raon1 mhc::enqueue_fold";
        if m == 0 || x.len() < m * STREAMS * n || hc.len() < m * HC_OUT || y.len() < m * n {
            return Err(GpuError::shape(
                what,
                format!(
                    "m={m}, n={n}: x {}, hc {}, y {}",
                    x.len(),
                    hc.len(),
                    y.len()
                ),
            ));
        }
        let total = launch_u32(what, "m·n", m * n)?;
        let prep = self.module.prepare_raon_hc_fold(LaunchConfig1D::new(
            total.div_ceil(ELEM_THREADS),
            ELEM_THREADS,
            0,
        ))?;
        self.module.raon_hc_fold(
            stream,
            &prep,
            x,
            hc,
            launch_u32(what, "n", n)?,
            launch_u32(what, "m", m)?,
            y,
        )?;
        Ok(())
    }

    /// The streams after a sub-layer: `res` mixed by the routes `hc` plus
    /// the branch `out`, into `next`. Asynchronous, allocation-free,
    /// capturable.
    pub(crate) fn enqueue_post(
        &self,
        stream: &CudaStream,
        (res, out, hc): (&DeviceBuffer<f32>, &DeviceBuffer<f32>, &DeviceBuffer<f32>),
        (n, m): (usize, usize),
        next: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "raon1 mhc::enqueue_post";
        let k = STREAMS * n;
        if m == 0
            || res.len() < m * k
            || out.len() < m * n
            || hc.len() < m * HC_OUT
            || next.len() < m * k
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "m={m}, n={n}: res {}, out {}, hc {}, next {}",
                    res.len(),
                    out.len(),
                    hc.len(),
                    next.len()
                ),
            ));
        }
        let total = launch_u32(what, "m·4n", m * k)?;
        let prep = self.module.prepare_raon_hc_post(LaunchConfig1D::new(
            total.div_ceil(ELEM_THREADS),
            ELEM_THREADS,
            0,
        ))?;
        self.module.raon_hc_post(
            stream,
            &prep,
            res,
            out,
            hc,
            launch_u32(what, "n", n)?,
            launch_u32(what, "m", m)?,
            next,
        )?;
        Ok(())
    }

    /// The mean of `m` tokens' streams `x` into `y` (`n` a token).
    /// Asynchronous, allocation-free, capturable.
    pub(crate) fn enqueue_mean(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        (n, m): (usize, usize),
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "raon1 mhc::enqueue_mean";
        if m == 0 || x.len() < m * STREAMS * n || y.len() < m * n {
            return Err(GpuError::shape(
                what,
                format!("m={m}, n={n}: x {}, y {}", x.len(), y.len()),
            ));
        }
        let total = launch_u32(what, "m·n", m * n)?;
        let prep = self.module.prepare_raon_hc_mean(LaunchConfig1D::new(
            total.div_ceil(ELEM_THREADS),
            ELEM_THREADS,
            0,
        ))?;
        self.module.raon_hc_mean(
            stream,
            &prep,
            x,
            launch_u32(what, "n", n)?,
            launch_u32(what, "m", m)?,
            y,
        )?;
        Ok(())
    }
}
