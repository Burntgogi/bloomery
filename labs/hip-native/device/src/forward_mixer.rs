//! Eager one-token hybrid mixers. Host validates spans; each entry uses 64-thread blocks.
use super::forward_math::{checked, exp, fma, half_rne, mark, sigmoid, silu, softplus, sqrt};

unsafe extern "C" {
    #[link_name = "llvm.amdgcn.ds.bpermute"]
    fn bpermute(index: u32, word: u32) -> u32;
}

#[inline(always)]
fn reduce32(mut lanes: [f64; 32]) -> f64 {
    let mut d = 16;
    while d != 0 {
        for i in 0..d {
            lanes[i] += lanes[i + d];
        }
        d /= 2;
    }
    lanes[0]
}

/// linear::{conv,norm_gate}: four F32 squares per logical lane, then F64 butterfly.
#[inline(always)]
fn sum128(values: &[f32; 128]) -> f64 {
    let mut lanes = [0.0f64; 32];
    for l in 0..32 {
        for i in 0..4 {
            lanes[l] += (values[l + 32 * i] * values[l + 32 * i]) as f64;
        }
    }
    reduce32(lanes)
}

#[inline(always)]
unsafe fn input(p: *const f32, i: usize, fault: *mut u32, tag: u32) -> f32 {
    let x = unsafe { *p.add(i) };
    if x.is_finite() {
        x
    } else {
        unsafe {
            mark(fault, tag);
        }
        f32::NAN
    }
}

/// # Safety
/// x/out cover (2*n_k+n_v)*128, conv four weights/channel, ring eleven rows;
/// n_k positive divides n_v, eps positive finite; no alias except the owned ring.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn bloomery_hip_gdn_conv(
    x: *const f32,
    conv: *const f32,
    ring: *mut f32,
    out: *mut f32,
    n_k: u32,
    n_v: u32,
    pos: u32,
    eps: f32,
    fault: *mut u32,
    tag: u32,
) {
    let h = super::index();
    let heads = 2 * n_k as usize + n_v as usize;
    if h >= heads {
        return;
    }
    if n_k == 0 || n_v == 0 || !n_v.is_multiple_of(n_k) || !eps.is_finite() || eps <= 0.0 {
        unsafe {
            mark(fault, tag);
        }
        for i in 0..128 {
            unsafe {
                *out.add(h * 128 + i) = f32::NAN;
            }
        }
        return;
    }
    let channels = heads * 128;
    let mut values = [0.0f32; 128];
    for (i, value) in values.iter_mut().enumerate() {
        let c = h * 128 + i;
        let mut acc = 0.0;
        for j in 0..4 {
            let distance = 3 - j;
            let v = if distance == 0 {
                unsafe { input(x, c, fault, tag) }
            } else if (pos as usize) < distance {
                0.0
            } else {
                unsafe {
                    input(
                        ring,
                        ((pos as usize - distance) % 11) * channels + c,
                        fault,
                        tag,
                    )
                }
            };
            let w = unsafe { input(conv, c * 4 + j, fault, tag) };
            acc = if j == 0 { v * w } else { fma(v, w, acc) };
        }
        *value = unsafe { checked(silu(checked(acc, fault, tag)), fault, tag) };
        // Predecessors of this channel have all been read before its ring write.
        unsafe {
            *ring.add((pos as usize % 11) * channels + c) = input(x, c, fault, tag);
        }
    }
    if h < 2 * n_k as usize {
        let sum = sum128(&values) as f32;
        let scale = if sum.is_finite() {
            1.0 / sqrt(sum).max(eps)
        } else {
            unsafe {
                mark(fault, tag);
            }
            f32::NAN
        };
        for v in &mut values {
            *v *= scale;
            if h < n_k as usize {
                *v *= f32::from_bits(0x3db5_04f3);
            }
        }
    }
    for (i, &value) in values.iter().enumerate() {
        unsafe {
            *out.add(h * 128 + i) = checked(value, fault, tag);
        }
    }
}

/// # Safety
/// All arrays cover n_v values; outputs do not alias inputs, fault is one atomic word.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn bloomery_hip_gdn_decay(
    b_raw: *const f32,
    a_raw: *const f32,
    dt: *const f32,
    ssm_a: *const f32,
    beta: *mut f32,
    decay: *mut f32,
    n_v: u32,
    fault: *mut u32,
    tag: u32,
) {
    let h = super::index();
    if h >= n_v as usize {
        return;
    }
    unsafe {
        let b = input(b_raw, h, fault, tag);
        let a = input(a_raw, h, fault, tag);
        let bias = input(dt, h, fault, tag);
        let sa = input(ssm_a, h, fault, tag);
        let step = checked(a + bias, fault, tag);
        let rate = checked(softplus(step) * sa, fault, tag);
        // Nonfinite softplus inputs must not become a plausible polynomial result.
        *beta.add(h) = checked(sigmoid(b), fault, tag);
        *decay.add(h) = checked(
            if step.is_finite() && rate.is_finite() {
                exp(rate)
            } else {
                f32::NAN
            },
            fault,
            tag,
        );
    }
}

#[inline(always)]
fn lane_dot(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    let mut p = [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]];
    for i in 4..16 {
        p[i % 4] = fma(a[i], b[i], p[i % 4]);
    }
    (p[0] + p[1]) + (p[2] + p[3])
}

#[inline(always)]
fn column_sum(mut value: f32, lane: usize) -> f32 {
    for d in [4, 2, 1] {
        // Each group of eight is wholly active, inside a logical wave32.
        let other = unsafe { bpermute(((lane ^ d) * 4) as u32, value.to_bits()) };
        value += f32::from_bits(other);
    }
    value
}

/// # Safety
/// n_k positive divides n_v. qkv covers (2*n_k+n_v)*128, state n_v*128*128,
/// out n_v*128; workitems n_v*128*8, wave32. Each lane owns sixteen state keys.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn bloomery_hip_gdn_delta(
    qkv: *const f32,
    beta: *const f32,
    decay: *const f32,
    state: *mut f32,
    out: *mut f32,
    n_k: u32,
    n_v: u32,
    fault: *mut u32,
    tag: u32,
) {
    let t = super::index();
    if t >= n_v as usize * 128 * 8 {
        return;
    }
    if n_k == 0 || !n_v.is_multiple_of(n_k) {
        unsafe {
            mark(fault, tag);
            if t.is_multiple_of(8) {
                *out.add(t / 8) = f32::NAN;
            }
        }
        return;
    }
    let col = t / 8;
    let h = col / 128;
    let kh = h % n_k as usize;
    let j = t % 8;
    let mut s = [0.0f32; 16];
    let mut k = [0.0f32; 16];
    let mut q = [0.0f32; 16];
    unsafe {
        let dc = input(decay, h, fault, tag);
        for i in 0..16 {
            let key = 32 * (i / 4) + 4 * j + i % 4;
            s[i] = checked(input(state, col * 128 + key, fault, tag) * dc, fault, tag);
            k[i] = input(qkv, (n_k as usize + kh) * 128 + key, fault, tag);
            q[i] = input(qkv, kh * 128 + key, fault, tag);
        }
        let dot = column_sum(lane_dot(&s, &k), t % 32);
        let v = input(qkv, 2 * n_k as usize * 128 + col, fault, tag);
        let u = checked((v - dot) * input(beta, h, fault, tag), fault, tag);
        for i in 0..16 {
            let key = 32 * (i / 4) + 4 * j + i % 4;
            s[i] = checked(fma(k[i], u, s[i]), fault, tag);
            *state.add(col * 128 + key) = s[i];
        }
        let result = checked(column_sum(lane_dot(&s, &q), t % 32), fault, tag);
        if j == 0 {
            *out.add(col) = result;
        }
    }
}

/// # Safety
/// x,z,out cover n_v*128; gain covers128, eps finite positive. One owner/head.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn bloomery_hip_gdn_gate(
    x: *const f32,
    z: *const f32,
    gain: *const f32,
    out: *mut f32,
    n_v: u32,
    eps: f32,
    fault: *mut u32,
    tag: u32,
) {
    let h = super::index();
    if h >= n_v as usize {
        return;
    }
    let mut values = [0.0f32; 128];
    for (i, value) in values.iter_mut().enumerate() {
        *value = unsafe { input(x, h * 128 + i, fault, tag) };
    }
    let mean = (sum128(&values) / 128.0) as f32;
    let raw_scale = 1.0 / sqrt(mean + eps);
    let scale = if mean.is_finite()
        && eps.is_finite()
        && eps > 0.0
        && raw_scale.is_finite()
        && raw_scale > 0.0
    {
        raw_scale
    } else {
        unsafe {
            mark(fault, tag);
        }
        f32::NAN
    };
    for (i, &value) in values.iter().enumerate() {
        unsafe {
            let w = input(gain, i, fault, tag);
            let gate = silu(input(z, h * 128 + i, fault, tag));
            *out.add(h * 128 + i) = checked(((value * scale) * w) * gate, fault, tag);
        }
    }
}

/// Original Neox norm: paired F32 squares, warp F64 butterflies, warp-order sum.
#[inline(always)]
unsafe fn head_scale(x: *const f32, width: usize, eps: f32, fault: *mut u32, tag: u32) -> f32 {
    let mut total = 0.0f64;
    let half = width / 2;
    let mut base = 0;
    while base < half {
        let mut lanes = [0.0f64; 32];
        for (l, value) in lanes.iter_mut().enumerate() {
            if base + l < half {
                let a = unsafe { input(x, base + l, fault, tag) };
                let b = unsafe { input(x, base + l + half, fault, tag) };
                *value = (a * a) as f64 + (b * b) as f64;
            }
        }
        total += reduce32(lanes);
        base += 32;
    }
    let mean = (total / width as f64) as f32;
    let scale = 1.0 / sqrt(mean + eps);
    if !mean.is_finite() || !eps.is_finite() || eps <= 0.0 || !scale.is_finite() || scale <= 0.0 {
        unsafe {
            mark(fault, tag);
        }
        f32::NAN
    } else {
        scale
    }
}

#[inline(always)]
fn gqa_shape(qheads: u32, kvheads: u32, head_dim: u32, pos: u32, context: u32) -> bool {
    qheads > 0
        && kvheads > 0
        && qheads.is_multiple_of(kvheads)
        && head_dim > 0
        && head_dim <= 256
        && head_dim.is_multiple_of(2)
        && context > 0
        && context <= 64
        && pos < context
}

/// # Safety
/// Host validates the contract's Q/gate, gain, rope and head-major cache spans.
/// Only this position is written; no input aliases output. Workitems qheads+kvheads.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn bloomery_hip_gqa_prep(
    qgate: *const f32,
    k: *const f32,
    v: *const f32,
    qgain: *const f32,
    kgain: *const f32,
    rope: *const f32,
    qout: *mut f32,
    kcache: *mut u16,
    vcache: *mut u16,
    qheads: u32,
    kvheads: u32,
    head_dim: u32,
    rope_dim: u32,
    pos: u32,
    context: u32,
    eps: f32,
    fault: *mut u32,
    tag: u32,
) {
    let unit = super::index();
    if unit >= qheads as usize + kvheads as usize {
        return;
    }
    if !gqa_shape(qheads, kvheads, head_dim, pos, context)
        || rope_dim == 0
        || !rope_dim.is_multiple_of(2)
        || rope_dim > head_dim
    {
        unsafe {
            mark(fault, tag);
        }
        return;
    }
    let query = unit < qheads as usize;
    let h = if query { unit } else { unit - qheads as usize };
    let width = head_dim as usize;
    unsafe {
        let source = if query {
            qgate.add(h * 2 * width)
        } else {
            k.add(h * width)
        };
        let gain = if query { qgain } else { kgain };
        let scale = head_scale(source, width, eps, fault, tag);
        let cache_at = (h * context as usize + pos as usize) * width;
        let rot_half = rope_dim as usize / 2;
        for i in 0..width {
            let own = (scale * input(gain, i, fault, tag)) * input(source, i, fault, tag);
            let result = if i < rope_dim as usize {
                let pair = if i < rot_half { i } else { i - rot_half };
                let partner = if i < rot_half {
                    i + rot_half
                } else {
                    i - rot_half
                };
                let other =
                    (scale * input(gain, partner, fault, tag)) * input(source, partner, fault, tag);
                let c = input(rope, pos as usize * rope_dim as usize + pair, fault, tag);
                let s = input(
                    rope,
                    pos as usize * rope_dim as usize + rot_half + pair,
                    fault,
                    tag,
                );
                if i < rot_half {
                    fma(own, c, -(other * s))
                } else {
                    fma(other, s, own * c)
                }
            } else {
                own
            };
            let result = checked(result, fault, tag);
            if query {
                *qout.add(h * width + i) = result;
            } else {
                let kb = half_rne(result);
                let vb = half_rne(input(v, h * width + i, fault, tag));
                checked(super::half(kb), fault, tag);
                checked(super::half(vb), fault, tag);
                *kcache.add(cache_at + i) = kb;
                *vcache.add(cache_at + i) = vb;
            }
        }
    }
}

/// # Safety
/// Q/out cover qheads*head_dim, qgate qheads*2*head_dim, caches kvheads*context*head_dim.
/// All live cache rows were appended; rows after pos are never accessed. Workitems qheads.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn bloomery_hip_attention(
    q: *const f32,
    qgate: *const f32,
    kcache: *const u16,
    vcache: *const u16,
    out: *mut f32,
    qheads: u32,
    kvheads: u32,
    head_dim: u32,
    pos: u32,
    context: u32,
    fault: *mut u32,
    tag: u32,
) {
    let h = super::index();
    if h >= qheads as usize {
        return;
    }
    if !gqa_shape(qheads, kvheads, head_dim, pos, context) {
        unsafe {
            mark(fault, tag);
        }
        return;
    }
    let width = head_dim as usize;
    let kh = h / (qheads / kvheads) as usize;
    let mut weights = [0.0f32; 64];
    let scale = 1.0 / sqrt(width as f32);
    let mut maximum = f32::NEG_INFINITY;
    let mut valid = true;
    unsafe {
        for (t, weight) in weights.iter_mut().enumerate().take(pos as usize + 1) {
            // flash_gqa's four rotating pair partials and fixed final tree.
            let mut a = [0.0f32; 4];
            let at = (kh * context as usize + t) * width;
            for pair in 0..width / 2 {
                let i = 2 * pair;
                let k0 = checked(super::half(*kcache.add(at + i)), fault, tag);
                let k1 = checked(super::half(*kcache.add(at + i + 1)), fault, tag);
                a[pair % 4] = fma(input(q, h * width + i, fault, tag), k0, a[pair % 4]);
                a[pair % 4] = fma(input(q, h * width + i + 1, fault, tag), k1, a[pair % 4]);
            }
            let score = checked(((a[0] + a[1]) + (a[2] + a[3])) * scale, fault, tag);
            valid &= score.is_finite();
            *weight = score;
            if score > maximum {
                maximum = score;
            }
        }
        let mut sum = 0.0f32;
        for weight in weights.iter_mut().take(pos as usize + 1) {
            // Original attention uses exp2(LOG2_E*x); AMD replaces only its intrinsic.
            *weight = checked(
                core::intrinsics::exp2f32((*weight - maximum) * core::f32::consts::LOG2_E),
                fault,
                tag,
            );
            sum += *weight;
        }
        checked(sum, fault, tag);
        valid &= sum.is_finite() && sum > 0.0;
        if !valid {
            mark(fault, tag);
        }
        let inverse = if valid { 1.0 / sum } else { f32::NAN };
        for i in 0..width {
            let mut value = 0.0f32;
            for (t, &weight) in weights.iter().enumerate().take(pos as usize + 1) {
                let v = checked(
                    super::half(*vcache.add((kh * context as usize + t) * width + i)),
                    fault,
                    tag,
                );
                value = fma(weight, v, value);
            }
            let gate = input(qgate, h * 2 * width + width + i, fault, tag);
            // gated_quant uses route_core::sigmoid, whose exponential is natural exp,
            // unlike the bit-polynomial used by the GDN's own gate and decay.
            let gate_scale = 1.0 / (1.0 + core::intrinsics::expf32(0.0 - gate));
            *out.add(h * width + i) = checked((value * inverse) * gate_scale, fault, tag);
        }
    }
}
