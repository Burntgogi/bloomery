//! The write gate of a delta layer whose write and erase gates differ (Raon
//! Delta Attention): the erase gate is the delta rule's `β = sigmoid(b)`, the
//! write gate `sigmoid(b + w)`, and the rule realises their ratio as a scale
//! on the value rows the conv and prep left, before the delta step reads
//! them: `v ← v · sigmoid(b + w) / sigmoid(b)`, one scale per (token, value
//! head), `b` the β projection's raw rows and `w` the write projection's.
//!
//! Geometry: one thread per value of the value heads, [`WRITE_GATE_THREADS`]
//! a block, no shared memory, in place on the conv output. Value `j` of value
//! head `h` of token `t` sits at `t·C + 2·n_k·HEAD + h·HEAD + j` of the conv
//! output (`C = (2·n_k + n_v)·HEAD`, the layout of [`super::conv`]); `b` and
//! `w` are `[m][n_v]`.
//!
//! Numeric contract (the host rule [`write_gate_host`] is this): `e =`
//! [`softplus`]`(0 − b)`, `u =` [`softplus`]`(0 − (b + w))`, the scale
//! [`expf_ik`]`(e − u)`, and the value times the scale, each op rounded —
//! ik's `exp(softplus(−b) − softplus(−(b + w)))`, the log-sigmoid difference
//! of the reference model.
//!
//! No silent failure: a non-finite `b`, `w` or value, or a scaled value that
//! is not finite, raises [`FaultSite::LinearConv`] (the gate is part of the
//! prep), and that value is written NaN.

use super::{HEAD, LinearShape, expf_ik, softplus};
use crate::GpuError;
use crate::fault::{FaultSink, FaultSite};
use crate::launch_u32;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::mul_rn_f32;
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

const _: () = assert!(HEAD == 128);

/// Threads per block of the entry.
pub const WRITE_GATE_THREADS: u32 = 256;
// The entry's `launch_bounds` and `launch_contract` block are literals.
const _: () = assert!(WRITE_GATE_THREADS == 256);

/// The scale of one (token, value head): `expf_ik(softplus(−b) −
/// softplus(−(b + w)))`, each op rounded. The device and host rule share it,
/// so it uses plain operators only (no device intrinsic runs on the host).
#[inline(always)]
#[must_use]
pub fn write_scale(b: f32, w: f32) -> f32 {
    let e = softplus(-b);
    let u = softplus(-(b + w));
    expf_ik(e - u)
}

#[cuda_module]
mod write_gate_kernels {
    use super::*;

    /// `v ← v · write_scale(b, w)` over the value heads of `m` tokens of a
    /// conv output of `n_k` query/key and `n_v` value heads (module doc). A
    /// non-finite input or result raises [`FaultSite::LinearConv`] and the
    /// value is written NaN.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            b.len() >= n_v * m,
            w.len() >= n_v * m,
            y.len() >= 128 * (2 * n_k + n_v) * m
        )
    )]
    pub fn kda_write_gate(
        b: &[f32],
        w: &[f32],
        n_k: u32,
        n_v: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let row = HEAD * n_v as usize;
        if i >= row * m as usize {
            return;
        }
        let (t, r) = (i / row, i % row);
        let g = t * n_v as usize + r / HEAD;
        let at = t * HEAD * (2 * n_k as usize + n_v as usize) + 2 * n_k as usize * HEAD + r;
        // SAFETY: g < n_v·m <= b.len(), w.len(), and at < HEAD·(2·n_k + n_v)·m
        // <= y.len(), by the launch contract; one thread per value, so this
        // thread alone reads and writes `at`.
        let (bv, wv, v) = unsafe {
            (
                *b.get_unchecked(g),
                *w.get_unchecked(g),
                *y.get_unchecked_mut(at),
            )
        };
        let out = mul_rn_f32(v, write_scale(bv, wv));
        let ok = bv.is_finite() & wv.is_finite() & v.is_finite() & out.is_finite();
        if !ok {
            fault.raise(FaultSite::LinearConv);
        }
        // SAFETY: as above.
        unsafe { *y.get_unchecked_mut(at) = if ok { out } else { f32::NAN } };
    }
}

/// [`WriteGateKernels::enqueue_write_gate`]'s arguments: `m` tokens of the β
/// projection's raw rows `b` and the write projection's `w` (`[m][n_v]`
/// each), and the conv output `y` (`[m][C]`) whose value heads it scales in
/// place.
pub struct WriteGateArgs<'a> {
    pub b: &'a DeviceBuffer<f32>,
    pub w: &'a DeviceBuffer<f32>,
    pub shape: LinearShape,
    pub m: usize,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct WriteGateKernels {
    module: write_gate_kernels::LoadedModule,
}

impl WriteGateKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<WriteGateKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(write_gate_kernels, ctx)? };
        Ok(WriteGateKernels { module })
    }

    /// Enqueue the write gate over `args.m` tokens: `⌈m·n_v·HEAD / 256⌉`
    /// blocks of 256 threads. Asynchronous, allocation-free, capturable.
    pub fn enqueue_write_gate(
        &self,
        stream: &CudaStream,
        args: WriteGateArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_write_gate";
        let WriteGateArgs {
            b,
            w,
            shape,
            m,
            fault,
            y,
        } = args;
        shape.check(what)?;
        if m == 0 {
            return Err(GpuError::shape(what, "need m >= 1, got m=0".to_string()));
        }
        let lens = [
            ("b", b.len(), shape.n_v * m),
            ("w", w.len(), shape.n_v * m),
            ("y", y.len(), shape.channels() * m),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let n = shape.n_v * HEAD * m;
        let grid = launch_u32(what, "grid", n.div_ceil(WRITE_GATE_THREADS as usize))?;
        let n_k = launch_u32(what, "n_k", shape.n_k)?;
        let n_v = launch_u32(what, "n_v", shape.n_v)?;
        let m = launch_u32(what, "m", m)?;
        let prep =
            self.module
                .prepare_kda_write_gate(LaunchConfig1D::new(grid, WRITE_GATE_THREADS, 0))?;
        self.module
            .kda_write_gate(stream, &prep, b, w, n_k, n_v, m, fault, y)?;
        Ok(())
    }
}

/// The host rule of [`WriteGateKernels::enqueue_write_gate`]: the module
/// doc's numeric contract, op for op, on the conv output `y` of `m` tokens in
/// place; a value whose input or result is not finite is NaN.
///
/// # Panics
///
/// When `b` or `w` is shorter than `n_v·m` or `y` than `C·m`.
pub fn write_gate_host(b: &[f32], w: &[f32], shape: LinearShape, m: usize, y: &mut [f32]) {
    let (n_k, n_v) = (shape.n_k, shape.n_v);
    let ch = shape.channels();
    for t in 0..m {
        for h in 0..n_v {
            let (bv, wv) = (b[t * n_v + h], w[t * n_v + h]);
            let scale = write_scale(bv, wv);
            for j in 0..HEAD {
                let at = t * ch + 2 * n_k * HEAD + h * HEAD + j;
                let v = y[at];
                let out = v * scale;
                let ok = bv.is_finite() && wv.is_finite() && v.is_finite() && out.is_finite();
                y[at] = if ok { out } else { f32::NAN };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{write_gate_host, write_scale};
    use crate::linear::{HEAD, KHeadMap, LinearShape};

    /// The scale against `sigmoid(b + w) / sigmoid(b)` in f64, over the
    /// range a trained β and write projection reach: the two softplus
    /// polynomials, the exp and the four roundings stay under 4e-6 relative.
    #[test]
    fn write_scale_is_the_sigmoid_ratio() {
        let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
        for i in 0..4096u16 {
            let b = (f32::from(i) * 0.013).sin() * 12.0;
            let w = (f32::from(i) * 0.029).cos() * 12.0;
            let got = f64::from(write_scale(b, w));
            let want = sig(f64::from(b) + f64::from(w)) / sig(f64::from(b));
            assert!(
                (got - want).abs() <= 4e-6 * want,
                "b={b} w={w}: {got} vs {want}"
            );
        }
    }

    /// The host rule scales only the value heads, each by its own (token,
    /// head) scale, and writes NaN where an input is not finite.
    #[test]
    fn write_gate_scales_only_the_value_heads() {
        let shape = LinearShape {
            n_k: 2,
            n_v: 2,
            map: KHeadMap::Tiled,
        };
        let m = 2;
        let ch = shape.channels();
        let y0: Vec<f32> = (0..ch * m).map(|i| (i as f32 * 0.7).sin()).collect();
        let b = [0.5f32, -1.0, 2.0, f32::NAN];
        let w = [1.0f32, 0.25, -3.0, 0.0];
        let mut y = y0.clone();
        write_gate_host(&b, &w, shape, m, &mut y);
        for t in 0..m {
            for c in 0..ch {
                let at = t * ch + c;
                if c < 2 * shape.n_k * HEAD {
                    assert_eq!(y[at].to_bits(), y0[at].to_bits(), "t={t} c={c}");
                    continue;
                }
                let h = (c - 2 * shape.n_k * HEAD) / HEAD;
                let g = t * shape.n_v + h;
                if b[g].is_nan() {
                    assert!(y[at].is_nan(), "t={t} c={c}");
                } else {
                    let want = y0[at] * write_scale(b[g], w[g]);
                    assert_eq!(y[at].to_bits(), want.to_bits(), "t={t} c={c}");
                }
            }
        }
    }
}
