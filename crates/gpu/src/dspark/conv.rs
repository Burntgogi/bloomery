//! DFlash2's dynamic convolution over a block's rows (llama.cpp `src/models/dflash.cpp`,
//! `build_dflash2_conv`), the step a DFlash2 draft's layers run around both sub-layers. Row `t`,
//! channel `c` of side `s` is
//!
//! `y[t][c] = Σ_{j < min(kernel, m)} (coeff[t][s][j][c / group] + base[s][j][c]) · h[t − j][c]`
//!
//! over the block's `m` rows, a row before the block's first reading as zero. `coeff` is the
//! layer's projection of the sub-layer's normed input (`*_conv_proj`): `2 · kernel · n / group`
//! values a row in ggml's `[n / group, kernel, 2]` order, so side `s`, tap `j`, group `g` is value
//! `(s · kernel + j) · n / group + g`. `base` is the layer's `*_conv_base`, ggml's `[n, kernel, 2]`:
//! side `s`, tap `j`, channel `c` is value `(s · kernel + j) · n + c`. Side 0 mixes the sub-layer's
//! input before its projections, side 1 its output before the residual add, both from the one
//! projection of the input ([`Side`]).
//!
//! Numeric contract, the reference's op order ([`conv_host`] is the rule): the tap's weight
//! `coeff + base` rounded once, its product with the row rounded once, and the taps added in
//! ascending order from tap 0 (tap 0's product is the start, not `0 + product`). No fused
//! multiply-add. A tap past the block's first row multiplies its weight by zero, as the reference's
//! zero padding does.
//!
//! Geometry: one thread a value, [`CONV_THREADS`] a block, no shared memory.

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{add_rn_f32, mul_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use model::arch::dflash2::Conv;

use crate::{GpuError, launch_u32};

/// Threads per block of the entry.
pub const CONV_THREADS: u32 = 256;
// The entry's `launch_bounds` and `launch_contract` block are literals.
const _: () = assert!(CONV_THREADS == 256);

/// Which side of a sub-layer the convolution mixes (the module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// The sub-layer's normed input, before its projections.
    Input,
    /// The sub-layer's output, before the residual add.
    Output,
}

impl Side {
    /// The side's index in the coefficient and base layouts.
    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Side::Input => 0,
            Side::Output => 1,
        }
    }
}

#[cuda_module]
mod conv_kernels {
    use super::*;

    /// `y = conv(h)` over `m` rows of `n` channels on side `side` (module
    /// doc), `groups · group >= n` channels' coefficients a tap.
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
            kernel >= 1,
            group >= 1,
            side <= 1,
            groups * group >= n,
            h.len() >= m * n,
            coeff.len() >= m * groups * kernel * 2,
            base.len() >= n * kernel * 2,
            y.len() >= m * n
        )
    )]
    pub fn dflash_conv(
        h: &[f32],
        coeff: &[f32],
        base: &[f32],
        n: u32,
        m: u32,
        kernel: u32,
        group: u32,
        groups: u32,
        side: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let (n, m) = (n as usize, m as usize);
        if i >= n * m {
            return;
        }
        let t = i / n;
        let c = i - t * n;
        let (k, g, gs, s) = (
            kernel as usize,
            group as usize,
            groups as usize,
            side as usize,
        );
        let taps = if k < m { k } else { m };
        let row = (t * 2 + s) * k * gs;
        let mut acc = 0.0f32;
        let mut j = 0usize;
        while j < taps {
            // SAFETY: t < m, s <= 1, j < kernel and c / group < groups (c < n
            // <= groups · group), so the coefficient index is below m · groups
            // · kernel · 2 <= coeff.len(); the base index (s · kernel + j) · n
            // + c is below 2 · kernel · n <= base.len(); the row index (t − j)
            // · n + c, read only for j <= t, is below m · n <= h.len().
            let (cw, bw, v) = unsafe {
                (
                    *coeff.get_unchecked(row + j * gs + c / g),
                    *base.get_unchecked((s * k + j) * n + c),
                    if j <= t {
                        *h.get_unchecked((t - j) * n + c)
                    } else {
                        0.0
                    },
                )
            };
            let term = mul_rn_f32(add_rn_f32(cw, bw), v);
            acc = if j == 0 { term } else { add_rn_f32(acc, term) };
            j += 1;
        }
        // SAFETY: i < m · n <= y.len(); one thread per value.
        unsafe { *y.get_unchecked_mut(i) = acc };
    }
}

/// [`ConvKernels::enqueue_conv`]'s arguments: `m` rows of `n` channels in `h`
/// and out `y` (token-major), the rows' coefficients `coeff`
/// ([`Conv::coeffs`] a row) and the layer's `base` (`2 · kernel · n`).
pub struct ConvArgs<'a> {
    pub h: &'a DeviceBuffer<f32>,
    pub coeff: &'a DeviceBuffer<f32>,
    pub base: &'a DeviceBuffer<f32>,
    pub conv: Conv,
    pub n: usize,
    pub m: usize,
    pub side: Side,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct ConvKernels {
    module: conv_kernels::LoadedModule,
}

impl ConvKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<ConvKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(conv_kernels, ctx)? };
        Ok(ConvKernels { module })
    }

    /// Enqueue `y = conv(h)` on `args.side` (module doc): `⌈m·n / 256⌉`
    /// blocks of 256 threads. A kernel or group of 0, a group that does not
    /// divide `n`, an empty launch, and each short buffer are refused by
    /// name. Asynchronous, allocation-free, capturable.
    pub fn enqueue_conv(&self, stream: &CudaStream, args: ConvArgs<'_>) -> Result<(), GpuError> {
        let what = "dspark::conv";
        let ConvArgs {
            h,
            coeff,
            base,
            conv,
            n,
            m,
            side,
            y,
        } = args;
        check(conv, n, m, [h.len(), coeff.len(), base.len(), y.len()])?;
        let grid = launch_u32(what, "grid", (m * n).div_ceil(CONV_THREADS as usize))?;
        let n_u = launch_u32(what, "n", n)?;
        let m_u = launch_u32(what, "m", m)?;
        let kernel = launch_u32(what, "kernel", conv.kernel)?;
        let group = launch_u32(what, "group", conv.group)?;
        let groups = launch_u32(what, "groups", n / conv.group)?;
        let side = launch_u32(what, "side", side.index())?;
        let prep = self
            .module
            .prepare_dflash_conv(LaunchConfig1D::new(grid, CONV_THREADS, 0))?;
        self.module.dflash_conv(
            stream, &prep, h, coeff, base, n_u, m_u, kernel, group, groups, side, y,
        )?;
        Ok(())
    }
}

/// The launch's refusals, shared with [`conv_host`]: `lens` are `h`, `coeff`,
/// `base` and `y`'s lengths.
fn check(conv: Conv, n: usize, m: usize, lens: [usize; 4]) -> Result<(), GpuError> {
    let what = "dspark::conv";
    if conv.kernel == 0 || conv.group == 0 || n == 0 || m == 0 || !n.is_multiple_of(conv.group) {
        return Err(GpuError::shape(
            what,
            format!(
                "a kernel of {} and groups of {} over {m} rows of {n} channels: all nonzero, the \
                 group dividing the channels",
                conv.kernel, conv.group
            ),
        ));
    }
    let need = [m * n, m * conv.coeffs(n), 2 * conv.kernel * n, m * n];
    for ((name, got), need) in ["h", "coeff", "base", "y"].iter().zip(lens).zip(need) {
        if got < need {
            return Err(GpuError::shape(
                what,
                format!("{name} holds {got} values; {need} wanted"),
            ));
        }
    }
    Ok(())
}

/// The module doc's rule on the host, op for op: `m` rows of `n` channels of
/// `h` on `side`, refused as [`ConvKernels::enqueue_conv`] refuses.
pub fn conv_host(
    h: &[f32],
    coeff: &[f32],
    base: &[f32],
    conv: Conv,
    n: usize,
    m: usize,
    side: Side,
) -> Result<Vec<f32>, GpuError> {
    check(conv, n, m, [h.len(), coeff.len(), base.len(), m * n])?;
    let (k, g) = (conv.kernel, conv.group);
    let gs = n / g;
    let s = side.index();
    let taps = k.min(m);
    let mut y = vec![0.0f32; m * n];
    for (t, out) in y.chunks_exact_mut(n).enumerate() {
        let row = &coeff[(t * 2 + s) * k * gs..];
        for (c, o) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for j in 0..taps {
                let w = row[j * gs + c / g] + base[(s * k + j) * n + c];
                let v = if j <= t { h[(t - j) * n + c] } else { 0.0 };
                let term = w * v;
                acc = if j == 0 { term } else { acc + term };
            }
            *o = acc;
        }
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gpu;

    /// The host rule on two rows of four channels in groups of two, a kernel
    /// of two, worked by hand from the reference's views: row `t`'s
    /// coefficients run side, then tap, then group; the base runs side, then
    /// tap, then channel; tap 1 of row 0 reads the zero before the block.
    #[test]
    fn conv_reads_the_reference_layout() {
        let conv = Conv {
            kernel: 2,
            group: 2,
        };
        let (n, m) = (4, 2);
        let h = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        // Row t, side s, tap j, group g at (t·2 + s)·2·2 + j·2 + g: value
        // 1000·t + 100·s + 10·j + g.
        let coeff: Vec<f32> = (0..m)
            .flat_map(|t| {
                (0..2).flat_map(move |s| {
                    (0..2).flat_map(move |j| {
                        (0..2).map(move |g| (1000 * t + 100 * s + 10 * j + g) as f32)
                    })
                })
            })
            .collect();
        // Side s, tap j, channel c at (s·2 + j)·4 + c: value 0.5 · c + s + 0.25 · j.
        let base: Vec<f32> = (0..2)
            .flat_map(|s| {
                (0..2).flat_map(move |j| {
                    (0..4).map(move |c| 0.5 * c as f32 + s as f32 + 0.25 * j as f32)
                })
            })
            .collect();
        let y0 = conv_host(&h, &coeff, &base, conv, n, m, Side::Input).expect("side 0");
        // Row 0: tap 0 only (tap 1 reads the zero row): (coeff(0,0,0,c/2) + base(0,0,c)) · h[0][c].
        // Row 1: + (coeff(1,0,1,c/2) + base(0,1,c)) · h[0][c].
        let want0 = [
            (0.0 + 0.0) * 1.0,
            (0.0 + 0.5) * 2.0,
            (1.0 + 1.0) * 3.0,
            (1.0 + 1.5) * 4.0,
            (1000.0 + 0.0) * 10.0 + (1010.0 + 0.25) * 1.0,
            (1000.0 + 0.5) * 20.0 + (1010.0 + 0.75) * 2.0,
            (1001.0 + 1.0) * 30.0 + (1011.0 + 1.25) * 3.0,
            (1001.0 + 1.5) * 40.0 + (1011.0 + 1.75) * 4.0,
        ];
        assert_eq!(y0, want0);
        let y1 = conv_host(&h, &coeff, &base, conv, n, m, Side::Output).expect("side 1");
        let want1 = [
            (100.0 + 1.0) * 1.0,
            (100.0 + 1.5) * 2.0,
            (101.0 + 2.0) * 3.0,
            (101.0 + 2.5) * 4.0,
            (1100.0 + 1.0) * 10.0 + (1110.0 + 1.25) * 1.0,
            (1100.0 + 1.5) * 20.0 + (1110.0 + 1.75) * 2.0,
            (1101.0 + 2.0) * 30.0 + (1111.0 + 2.25) * 3.0,
            (1101.0 + 2.5) * 40.0 + (1111.0 + 2.75) * 4.0,
        ];
        assert_eq!(y1, want1);
        // One row runs tap 0 alone, the kernel cut to the block.
        let one = conv_host(&h[..4], &coeff[..8], &base, conv, n, 1, Side::Input).expect("m = 1");
        assert_eq!(one, want0[..4]);
        // A group that does not divide the channels, and a short base, are refused.
        let odd = Conv {
            kernel: 2,
            group: 3,
        };
        assert!(conv_host(&h, &coeff, &base, odd, n, m, Side::Input).is_err());
        assert!(conv_host(&h, &coeff, &base[..15], conv, n, m, Side::Input).is_err());
    }

    /// Seeded values in [−2, 2).
    fn seeded(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 8) as f32 / 4_194_304.0 - 2.0
            })
            .collect()
    }

    /// The card's convolution is the host rule bit for bit: Raon's draft's
    /// width, kernel and group over 1, 2, 5 and 8 rows on both sides, and a
    /// kernel of 3 over 2 rows (the taps cut to the block); an output tail
    /// past `m · n` stays untouched; each refusal of the host rule refuses
    /// the launch by name.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_conv_is_the_host_rule() {
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let stream = gpu.stream();
        let k = ConvKernels::load(gpu.context()).expect("the conv module");
        let n = 2560;
        let mut ok = true;
        let cases = [
            (
                Conv {
                    kernel: 2,
                    group: 16,
                },
                1,
            ),
            (
                Conv {
                    kernel: 2,
                    group: 16,
                },
                2,
            ),
            (
                Conv {
                    kernel: 2,
                    group: 16,
                },
                5,
            ),
            (
                Conv {
                    kernel: 2,
                    group: 16,
                },
                8,
            ),
            (
                Conv {
                    kernel: 3,
                    group: 32,
                },
                2,
            ),
        ];
        for (i, &(conv, m)) in cases.iter().enumerate() {
            let seed = 7 + 3 * i as u32;
            let h = seeded(m * n, seed);
            let coeff = seeded(m * conv.coeffs(n), seed + 1);
            let base = seeded(2 * conv.kernel * n, seed + 2);
            for side in [Side::Input, Side::Output] {
                const GUARD: f32 = 12_345.0;
                let want = conv_host(&h, &coeff, &base, conv, n, m, side).expect("host rule");
                let hd = DeviceBuffer::from_host(stream, &h).expect("h");
                let cd = DeviceBuffer::from_host(stream, &coeff).expect("coeff");
                let bd = DeviceBuffer::from_host(stream, &base).expect("base");
                let mut yd = DeviceBuffer::from_host(stream, &vec![GUARD; m * n + 7]).expect("y");
                k.enqueue_conv(
                    stream,
                    ConvArgs {
                        h: &hd,
                        coeff: &cd,
                        base: &bd,
                        conv,
                        n,
                        m,
                        side,
                        y: &mut yd,
                    },
                )
                .expect("conv");
                stream.synchronize().expect("sync");
                let got = yd.to_host_vec(stream).expect("y back");
                let same = got[..m * n]
                    .iter()
                    .zip(&want)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                let tail = got[m * n..].iter().all(|v| v.to_bits() == GUARD.to_bits());
                println!("conv {conv:?} m={m} {side:?}: bits {same} tail {tail}");
                ok &= same && tail;
            }
        }
        assert!(ok, "the card's convolution is the host rule bit for bit");

        let conv = Conv {
            kernel: 2,
            group: 16,
        };
        let h = DeviceBuffer::from_host(stream, &seeded(2 * n, 1)).expect("h");
        let c = DeviceBuffer::from_host(stream, &seeded(2 * conv.coeffs(n), 2)).expect("coeff");
        let b = DeviceBuffer::from_host(stream, &seeded(4 * n, 3)).expect("base");
        let short = DeviceBuffer::from_host(stream, &seeded(4 * n - 1, 3)).expect("short base");
        for (conv, m, base, what) in [
            (
                Conv {
                    kernel: 2,
                    group: 24,
                },
                2,
                &b,
                "a group that does not divide the channels",
            ),
            (conv, 0, &b, "an empty launch"),
            (conv, 2, &short, "a short base"),
            (conv, 3, &b, "rows past the inputs"),
        ] {
            let mut y = DeviceBuffer::<f32>::zeroed(stream, 3 * n).expect("y");
            let args = ConvArgs {
                h: &h,
                coeff: &c,
                base,
                conv,
                n,
                m,
                side: Side::Input,
                y: &mut y,
            };
            let e = k.enqueue_conv(stream, args).expect_err(what);
            assert!(e.to_string().contains("dspark::conv"), "{what}: {e}");
        }
    }
}
