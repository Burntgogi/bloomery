//! Original linear-attention arithmetic, with AMDGPU intrinsics only at the shim boundary.
use core::sync::atomic::{AtomicU32, Ordering};

#[inline(always)]
pub(super) unsafe fn mark(fault: *mut u32, tag: u32) {
    // SAFETY: every forward launch owns a live, aligned atomic fault word.
    unsafe {
        (&*fault.cast::<AtomicU32>()).fetch_min(tag, Ordering::Relaxed);
    }
}

#[inline(always)]
pub(super) unsafe fn checked(value: f32, fault: *mut u32, tag: u32) -> f32 {
    if !value.is_finite() {
        // SAFETY: fault is the launch's live, u32-aligned atomic word;
        // reporting a nonfinite value only performs an atomic minimum.
        unsafe {
            mark(fault, tag);
        }
    }
    value
}

#[inline(always)]
pub(super) fn fma(a: f32, b: f32, c: f32) -> f32 {
    core::intrinsics::fmaf32(a, b, c)
}

#[inline(always)]
pub(super) fn sqrt(x: f32) -> f32 {
    core::intrinsics::sqrtf32(x)
}

/// Op-for-op transcription of linear::expf_ik (including both exponent escapes).
#[inline(always)]
pub(super) fn exp(x: f32) -> f32 {
    let z = fma(x, f32::from_bits(0x3fb8_aa3b), f32::from_bits(0x4b40_0000));
    let n = z - f32::from_bits(0x4b40_0000);
    let b = fma(
        -n,
        f32::from_bits(0x35bf_be8e),
        fma(-n, f32::from_bits(0x3f31_7200), x),
    );
    let e = z.to_bits() << 23;
    let k = f32::from_bits(e.wrapping_add(0x3f80_0000));
    let u = b * b;
    let j = fma(
        fma(
            fma(f32::from_bits(0x3c07_2010), b, f32::from_bits(0x3d2b_9f17)),
            u,
            fma(f32::from_bits(0x3e2a_af33), b, f32::from_bits(0x3eff_fedb)),
        ),
        u,
        f32::from_bits(0x3f7f_fff6) * b,
    );
    if n.abs() > 126.0 {
        let g: u32 = if n <= 0.0 { 0x8200_0000 } else { 0 };
        let s1 = f32::from_bits(g.wrapping_add(0x7f00_0000));
        let s2 = f32::from_bits(e.wrapping_sub(g));
        if n.abs() > 192.0 {
            s1 * s1
        } else {
            fma(s2, j, s2) * s1
        }
    } else {
        fma(j, k, k)
    }
}

/// Cephes polynomial used by linear::softplus; its input is positive normal.
#[inline(always)]
fn log(x: f32) -> f32 {
    let bits = x.to_bits();
    let mut e = ((bits >> 23) & 255) as i32 - 126;
    let m = f32::from_bits((bits & 0x007f_ffff) | 0x3f00_0000);
    let low = m < f32::from_bits(0x3f35_04f3);
    e -= low as i32;
    let f = (if low { m + m } else { m }) - 1.0;
    let z = f * f;
    let mut p = f32::from_bits(0x3d90_21bb);
    for bits in [
        0xbdeb_d1b8,
        0x3def_251a,
        0xbdfe_5d4f,
        0x3e11_e9bf,
        0xbe2a_ae50,
        0x3e4c_ceac,
        0xbe7f_fffc,
        0x3eaa_aaaa,
    ] {
        p = fma(p, f, f32::from_bits(bits));
    }
    let fe = e as f32;
    let y = fma(z, -0.5, fma(fe, f32::from_bits(0xb95e_8083), p * f * z));
    fma(fe, f32::from_bits(0x3f31_8000), f + y)
}

#[inline(always)]
pub(super) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(0.0 - x))
}
#[inline(always)]
pub(super) fn silu(x: f32) -> f32 {
    x / (1.0 + exp(0.0 - x))
}
#[inline(always)]
pub(super) fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { log(1.0 + exp(x)) }
}

/// GGUF's IEEE F16 round-to-nearest-even, including gradual underflow.
#[inline(always)]
pub(super) fn half_rne(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let a = b & 0x7fff_ffff;
    if a > 0x7f80_0000 {
        return sign | 0x7e00;
    }
    let exp = ((a >> 23) as i32) - 127;
    let frac = a & 0x007f_ffff;
    if exp > 15 {
        return sign | 0x7c00;
    }
    if exp >= -14 {
        let v = (((exp + 15) as u32) << 23) | frac;
        return sign | ((v + 0x0fff + ((v >> 13) & 1)) >> 13) as u16;
    }
    if exp < -25 {
        return sign;
    }
    let shift = (-1 - exp) as u32;
    let v = 0x0080_0000 | frac;
    let midpoint = 1u32 << (shift - 1);
    let rem = v & ((1 << shift) - 1);
    let h = v >> shift;
    sign | (h + u32::from(rem > midpoint || (rem == midpoint && h & 1 == 1))) as u16
}
