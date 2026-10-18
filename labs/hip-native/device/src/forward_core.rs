//! Eager scalar whole-forward operators over packed GGUF rows.

use crate::forward_math::{checked, mark, silu, sqrt};
use core::sync::atomic::{AtomicU32, Ordering};

#[inline(always)]
fn row_bytes(width: usize, kind: u32) -> Option<usize> {
    match kind {
        0 if width > 0 => Some(width * 4),
        8 if width > 0 && width.is_multiple_of(32) => Some(width / 32 * 34),
        12 if width > 0 && width.is_multiple_of(256) => Some(width / 256 * 144),
        14 if width > 0 && width.is_multiple_of(256) => Some(width / 256 * 210),
        _ => None,
    }
}

#[inline(always)]
unsafe fn value(row: *const u8, i: usize, kind: u32) -> f32 {
    // SAFETY: the caller has checked kind/width and supplies a complete packed row.
    unsafe {
        match kind {
            0 => {
                let p = row.add(i * 4);
                f32::from_bits(u32::from_le_bytes([*p, *p.add(1), *p.add(2), *p.add(3)]))
            }
            8 => {
                let b = row.add(i / 32 * 34);
                let d = crate::half(u16::from_le_bytes([*b, *b.add(1)]));
                (*b.add(2 + i % 32) as i8 as f32) * d
            }
            12 => crate::q4(row.add(i / 256 * 144), i % 256),
            14 => {
                let b = row.add(i / 256 * 210);
                let j = i % 256;
                let half = j / 128;
                let part = j % 128 / 32;
                let lane = j % 32;
                let low = *b.add(half * 64 + (part % 2) * 32 + lane);
                let low = if part < 2 { low & 15 } else { low >> 4 };
                let high = (*b.add(128 + half * 32 + lane) >> (part * 2)) & 3;
                let code = i32::from(low | (high << 4)) - 32;
                let scale = *b.add(192 + half * 8 + part * 2 + lane / 16) as i8;
                let d = crate::half(u16::from_le_bytes([*b.add(208), *b.add(209)]));
                (d * scale as f32) * code as f32
            }
            _ => f32::NAN,
        }
    }
}

/// # Safety
/// Host validates token bounds and complete packed rows; out covers width and fault is aligned.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn bloomery_hip_embed(
    weights: *const u8,
    out: *mut f32,
    width: u32,
    token: u32,
    kind: u32,
    fault: *mut u32,
    tag: u32,
) {
    let i = crate::index();
    if i >= width as usize {
        return;
    }
    // SAFETY: host provides the validated row and unique output ownership.
    unsafe {
        let v = match row_bytes(width as usize, kind) {
            Some(bytes) => value(weights.add(token as usize * bytes), i, kind),
            None => f32::NAN,
        };
        *out.add(i) = checked(v, fault, tag);
    }
}

/// # Safety
/// Host validates packed matrix/x/output spans, distinct output, and aligned fault storage.
#[allow(
    clippy::too_many_arguments,
    reason = "kernel ABI: flat pointer and scalar arguments"
)]
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn bloomery_hip_gemv(
    weights: *const u8,
    x: *const f32,
    out: *mut f32,
    k: u32,
    rows: u32,
    kind: u32,
    fault: *mut u32,
    tag: u32,
) {
    let row = crate::index();
    if row >= rows as usize {
        return;
    }
    // SAFETY: dimensions and pointer spans are host-validated; one workitem owns each output row.
    unsafe {
        let Some(bytes) = row_bytes(k as usize, kind) else {
            mark(fault, tag);
            *out.add(row) = f32::NAN;
            return;
        };
        let weights = weights.add(row * bytes);
        let mut sum = 0.0_f32;
        let mut j = 0;
        while j < k as usize {
            let w = checked(value(weights, j, kind), fault, tag);
            let a = checked(*x.add(j), fault, tag);
            let product = checked(w * a, fault, tag);
            sum = checked(sum + product, fault, tag);
            j += 1;
        }
        *out.add(row) = sum;
    }
}

/// # Safety
/// Each of heads spans width values; gain spans width, output may equal x, and fault is aligned.
#[allow(
    clippy::too_many_arguments,
    reason = "kernel ABI: flat pointer and scalar arguments"
)]
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn bloomery_hip_norm(
    x: *const f32,
    gain: *const f32,
    out: *mut f32,
    width: u32,
    heads: u32,
    eps: f32,
    fault: *mut u32,
    tag: u32,
) {
    let head = crate::index();
    if head >= heads as usize {
        return;
    }
    // SAFETY: host validates contiguous head/gain spans; this workitem alone owns its output head.
    unsafe {
        if width == 0 {
            mark(fault, tag);
            return;
        }
        let base = head * width as usize;
        let mut sum = 0.0_f64;
        let mut j = 0;
        while j < width as usize {
            let v = checked(*x.add(base + j), fault, tag);
            sum += checked(v * v, fault, tag) as f64;
            j += 1;
        }
        // The reference rounds the squares, then the f64 mean, then each product separately.
        let mean = checked((sum / width as f64) as f32, fault, tag);
        let denominator = checked(mean + eps, fault, tag);
        let scale = if eps.is_finite() && eps > 0.0 && denominator.is_finite() && denominator > 0.0
        {
            checked(1.0 / sqrt(denominator), fault, tag)
        } else {
            mark(fault, tag);
            f32::NAN
        };
        j = 0;
        while j < width as usize {
            let weighted = checked(scale * checked(*gain.add(j), fault, tag), fault, tag);
            *out.add(base + j) = checked(weighted * *x.add(base + j), fault, tag);
            j += 1;
        }
    }
}

/// # Safety
/// All spans cover n values; each workitem owns its output and fault is aligned.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn bloomery_hip_glu(
    gate: *const f32,
    up: *const f32,
    out: *mut f32,
    n: u32,
    fault: *mut u32,
    tag: u32,
) {
    let i = crate::index();
    if i < n as usize {
        // SAFETY: host validates spans and output ownership is unique.
        unsafe {
            let g = checked(*gate.add(i), fault, tag);
            let u = checked(*up.add(i), fault, tag);
            let activated = checked(silu(g), fault, tag);
            *out.add(i) = checked(activated * u, fault, tag);
        }
    }
}

/// # Safety
/// x/y cover n values and fault is aligned; no other workitem writes y[i].
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn bloomery_hip_add(
    x: *const f32,
    y: *mut f32,
    n: u32,
    fault: *mut u32,
    tag: u32,
) {
    let i = crate::index();
    if i < n as usize {
        // SAFETY: host validates spans and output ownership is unique.
        unsafe {
            let a = checked(*x.add(i), fault, tag);
            let b = checked(*y.add(i), fault, tag);
            *y.add(i) = checked(a + b, fault, tag);
        }
    }
}

/// # Safety
/// logits covers n values, summary covers two u32, fault is aligned and all prior launches completed.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn bloomery_hip_argmax(
    logits: *const f32,
    summary: *mut u32,
    n: u32,
    fault: *mut u32,
    tag: u32,
) {
    if crate::index() != 0 {
        return;
    }
    // SAFETY: one workitem owns summary, and fault has atomic u32 storage initialized by the host.
    unsafe {
        let mut best = f32::NEG_INFINITY;
        let mut id = u32::MAX;
        let mut valid = n > 0;
        let mut j = 0;
        while j < n {
            let v = *logits.add(j as usize);
            if !v.is_finite() {
                mark(fault, tag);
                valid = false;
            } else if v > best {
                best = v;
                id = j;
            }
            j += 1;
        }
        if !valid {
            mark(fault, tag);
            id = u32::MAX;
        }
        *summary = id;
        *summary.add(1) = (&*fault.cast::<AtomicU32>()).load(Ordering::Relaxed);
    }
}
