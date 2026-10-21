//! Q6_K gemv per head over f32 activations: the value projection of an
//! absorbed MLA layer whose `attn_v_b` stack is Q6_K (Raon-1's Q4_K_M file),
//! the Q6_K sibling of `model::StepKernels::enqueue_q8_0_gemv_heads`. Weight
//! row `r` belongs to head `h = r / rows_per_head` and dots that head's slice
//! `x[h·x_head_stride ..][..K]`; its output lands at `y[h·y_head_stride +
//! y_off + r mod rows_per_head]`. One token (`m = 1`).
//!
//! Geometry: one warp per weight row, eight rows per 256-thread block. A row
//! is `n_sb` super-blocks of 210 bytes (`ql[128]`, `qh[64]`, 16 signed
//! sub-block scales, the f16 `d`), read byte by byte from the row's words,
//! so any row start is in reach.
//!
//! Numeric contract (the host rule [`q6k_heads_host`] is this): lane `l`
//! walks super-blocks in order and, in each, the two halves `n` and the four
//! quarter values of the half — ggml's `dequantize_row_q6_K` order, value
//! `128n + l + 32j` with scale byte `8n + l/16 + 2j` — each weight `(d ·
//! sc) · (q − 32)` rounded at each product, added as `acc = fma(w, x,
//! acc)`; the 32 lanes then by the xor butterfly (16, 8, 4, 2, 1) in f32.
//! The activation stays f32 (ggml quantizes it to q8_K), so the result sits
//! within a band of ggml's, not on its bits.

use crate::cores::half_to_f32;
use crate::tensor::DeviceTensor;
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Bytes of a Q6_K super-block of 256 values.
const SB_BYTES: usize = 210;

/// Byte `b` of a little-endian word stream.
#[inline(always)]
fn byte(w: &[u32], b: usize) -> u32 {
    // SAFETY: the callers' byte offsets lie in the matrix's rows, which the
    // launch contract (device) or the slice length (host) covers.
    (unsafe { *w.get_unchecked(b >> 2) } >> ((b & 3) * 8)) & 0xff
}

/// Lane `l`'s partial of row `r` (module doc), the one order the device
/// and the host rule share.
#[inline(always)]
fn lane_partial(w: &[u32], x: &[f32], row: usize, x0: usize, n_sb: usize, l: usize) -> f32 {
    let mut acc = 0.0f32;
    let mut sb = 0usize;
    while sb < n_sb {
        let b0 = (row * n_sb + sb) * SB_BYTES;
        let d = half_to_f32((byte(w, b0 + 208) | (byte(w, b0 + 209) << 8)) as u16);
        let mut n = 0usize;
        while n < 2 {
            let ql = b0 + 64 * n;
            let qh = byte(w, b0 + 128 + 32 * n + l);
            let lo = byte(w, ql + l);
            let hi = byte(w, ql + l + 32);
            let q = [
                (lo & 0xf) | ((qh & 3) << 4),
                (hi & 0xf) | (((qh >> 2) & 3) << 4),
                (lo >> 4) | (((qh >> 4) & 3) << 4),
                (hi >> 4) | (((qh >> 6) & 3) << 4),
            ];
            let mut j = 0usize;
            while j < 4 {
                let sc = byte(w, b0 + 192 + 8 * n + l / 16 + 2 * j) as u8 as i8;
                let wv = (d * f32::from(sc)) * (q[j] as f32 - 32.0);
                // SAFETY: x0 + 256·sb + 128·n + l + 32·j < x0 + K, inside the
                // head's slice, which the launch contract (device) or the
                // slice length (host) covers.
                let xv = unsafe { *x.get_unchecked(x0 + 256 * sb + 128 * n + l + 32 * j) };
                acc = wv.mul_add(xv, acc);
                j += 1;
            }
            n += 1;
        }
        sb += 1;
    }
    acc
}

#[cuda_module]
mod q6k_heads_kernels {
    use super::*;

    /// The per-head Q6_K gemv (module doc): warp `r` of the launch computes
    /// weight row `r` of `n_rows`.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the flat arguments (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * w.len() >= n_rows * 210 * n_sb,
            n_rows == heads * rows_per_head,
            heads >= 1,
            x.len() >= (heads - 1) * x_head_stride + 256 * n_sb,
            y.len() >= (heads - 1) * y_head_stride + y_off + rows_per_head
        )
    )]
    pub fn q6k_heads_f32(
        w: &[u32],
        x: &[f32],
        n_sb: u32,
        rows_per_head: u32,
        x_head_stride: u32,
        y_head_stride: u32,
        y_off: u32,
        heads: u32,
        n_rows: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let r = thread::index_1d().get() / 32;
        if r >= n_rows as usize {
            return; // warp-uniform
        }
        let l = warp::lane_id() as usize;
        let rph = rows_per_head as usize;
        let h = r / rph;
        if h >= heads as usize {
            return; // warp-uniform; n_rows = heads · rows_per_head
        }
        let mut acc = lane_partial(w, x, r, h * x_head_stride as usize, n_sb as usize, l);
        acc += warp::shuffle_xor_f32(acc, 16);
        acc += warp::shuffle_xor_f32(acc, 8);
        acc += warp::shuffle_xor_f32(acc, 4);
        acc += warp::shuffle_xor_f32(acc, 2);
        acc += warp::shuffle_xor_f32(acc, 1);
        if l == 0 {
            // SAFETY: h < heads and r mod rph < rph, so the
            // index is inside the bound the launch contract gives y; one lane
            // of one warp writes it.
            unsafe {
                *y.get_unchecked_mut(h * y_head_stride as usize + y_off as usize + r % rph) = acc;
            }
        }
    }
}

/// [`Q6kHeadsKernels::enqueue_q6k_heads`]'s arguments: the Q6_K stack `w`
/// (`n_rows × 210·n_sb/4` words), the f32 activations `x`, the head layout
/// and the output.
pub struct Q6kHeadsArgs<'a> {
    pub w: &'a DeviceTensor<u32>,
    pub x: &'a DeviceBuffer<f32>,
    /// K, the values a row dots: `256·n_sb`.
    pub k: usize,
    pub rows_per_head: usize,
    pub x_head_stride: usize,
    pub y_head_stride: usize,
    pub y_off: usize,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct Q6kHeadsKernels {
    module: q6k_heads_kernels::LoadedModule,
}

impl Q6kHeadsKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Q6kHeadsKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(q6k_heads_kernels, ctx)? };
        Ok(Q6kHeadsKernels { module })
    }

    /// Enqueue the per-head Q6_K gemv (module doc): one warp a weight row.
    /// Refused by name: a K not a positive multiple of 256, a weight of other
    /// than `210·n_sb` bytes a row, a row count not a positive multiple of
    /// `rows_per_head`, an `x` or `y` shorter than the heads reach.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_q6k_heads(
        &self,
        stream: &CudaStream,
        a: Q6kHeadsArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_q6k_heads";
        let Q6kHeadsArgs {
            w,
            x,
            k,
            rows_per_head,
            x_head_stride,
            y_head_stride,
            y_off,
            y,
        } = a;
        if k == 0 || !k.is_multiple_of(256) {
            return Err(GpuError::shape(
                what,
                format!("K={k} is not a positive multiple of 256"),
            ));
        }
        let n_sb = k / 256;
        let n_rows = w.rows();
        if 4 * w.cols() != SB_BYTES * n_sb || 4 * w.buf().len() < n_rows * SB_BYTES * n_sb {
            return Err(GpuError::shape(
                what,
                format!(
                    "Q6_K rows of K={k} are {} bytes, the weight holds {} words a row over {} \
                     words",
                    SB_BYTES * n_sb,
                    w.cols(),
                    w.buf().len()
                ),
            ));
        }
        if rows_per_head == 0 || n_rows == 0 || !n_rows.is_multiple_of(rows_per_head) {
            return Err(GpuError::shape(
                what,
                format!(
                    "n_rows={n_rows} is not a positive multiple of rows_per_head={rows_per_head}"
                ),
            ));
        }
        let heads = n_rows / rows_per_head;
        let (need_x, need_y) = (
            (heads - 1) * x_head_stride + k,
            (heads - 1) * y_head_stride + y_off + rows_per_head,
        );
        if x.len() < need_x || y.len() < need_y {
            return Err(GpuError::shape(
                what,
                format!(
                    "x.len() {} (need {need_x}), y.len() {} (need {need_y})",
                    x.len(),
                    y.len()
                ),
            ));
        }
        let grid = launch_u32(what, "grid", n_rows.div_ceil(8))?;
        let prep = self
            .module
            .prepare_q6k_heads_f32(LaunchConfig1D::new(grid, 256, 0))?;
        self.module.q6k_heads_f32(
            stream,
            &prep,
            w.buf(),
            x,
            launch_u32(what, "n_sb", n_sb)?,
            launch_u32(what, "rows_per_head", rows_per_head)?,
            launch_u32(what, "x_head_stride", x_head_stride)?,
            launch_u32(what, "y_head_stride", y_head_stride)?,
            launch_u32(what, "y_off", y_off)?,
            launch_u32(what, "heads", heads)?,
            launch_u32(what, "n_rows", n_rows)?,
            y,
        )?;
        Ok(())
    }
}

/// The host rule of [`Q6kHeadsKernels::enqueue_q6k_heads`]: the module
/// doc's numeric contract, op for op, over the stack's words `w` of
/// `n_rows` rows into `y`.
///
/// # Panics
///
/// When `w`, `x` or `y` is shorter than the heads reach, or `k` is not a
/// multiple of 256.
#[allow(
    clippy::too_many_arguments,
    reason = "the launch's own arguments, as the host rule of one kernel entry"
)]
pub fn q6k_heads_host(
    w: &[u32],
    x: &[f32],
    k: usize,
    n_rows: usize,
    rows_per_head: usize,
    x_head_stride: usize,
    (y_head_stride, y_off): (usize, usize),
    y: &mut [f32],
) {
    assert!(k.is_multiple_of(256), "K={k}");
    let n_sb = k / 256;
    for r in 0..n_rows {
        let h = r / rows_per_head;
        let lanes: Vec<f32> = (0..32)
            .map(|l| lane_partial(w, x, r, h * x_head_stride, n_sb, l))
            .collect();
        y[h * y_head_stride + y_off + r % rows_per_head] = crate::linear::butterfly_f32(&lanes);
    }
}

#[cfg(test)]
mod tests {
    use super::{SB_BYTES, q6k_heads_host};
    use gguf::quant::{GgmlType, dequant_row};

    /// The host rule against ggml's own Q6_K dequantization dotted with the
    /// activations in f64: each head's rows read only its slice, and the
    /// f32 sums sit within 1e-5 relative of the f64 dot (K = 512).
    #[test]
    fn heads_rule_is_the_dequantized_dot_per_head() {
        let (k, rph, heads) = (512usize, 4usize, 3usize);
        let n_rows = rph * heads;
        let n_sb = k / 256;
        let mut seed = 0x9e37_79b9_u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let mut bytes = vec![0u8; n_rows * n_sb * SB_BYTES];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = next() as u8;
            // Keep each super-block's f16 `d` small and finite.
            let at = i % SB_BYTES;
            if at == 209 {
                *b = 0x1c;
            }
        }
        let words: Vec<u32> = bytes
            .chunks(4)
            .map(|c| {
                u32::from_le_bytes([c[0], c[1], *c.get(2).unwrap_or(&0), *c.get(3).unwrap_or(&0)])
            })
            .collect();
        let x: Vec<f32> = (0..heads * k).map(|i| ((i as f32) * 0.173).sin()).collect();
        let (y_stride, y_off) = (rph + 2, 1);
        let mut y = vec![f32::NAN; heads * y_stride];
        q6k_heads_host(&words, &x, k, n_rows, rph, k, (y_stride, y_off), &mut y);
        let row_bytes = n_sb * SB_BYTES;
        for r in 0..n_rows {
            let h = r / rph;
            let mut wv = vec![0.0f32; k];
            dequant_row(
                GgmlType::Q6_K,
                &bytes[r * row_bytes..(r + 1) * row_bytes],
                &mut wv,
            )
            .unwrap();
            let want: f64 = wv
                .iter()
                .zip(&x[h * k..(h + 1) * k])
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            let scale: f64 = wv
                .iter()
                .zip(&x[h * k..(h + 1) * k])
                .map(|(&a, &b)| (f64::from(a) * f64::from(b)).abs())
                .sum();
            let got = f64::from(y[h * y_stride + y_off + r % rph]);
            assert!(
                (got - want).abs() <= 1e-5 * scale,
                "row {r}: {got} vs {want}"
            );
        }
        assert!(y[0].is_nan(), "a slot before y_off is not written");
    }
}
