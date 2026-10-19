//! Small independent CPU referees; never used by the production forward path.
use crate::{Arg, Context};
use gguf::quant::{f32_to_f16_bits, half_to_f32};
use std::ffi::c_void;

const HEAD: usize = 128;
const EPS: f32 = 1.0e-5;
const TAG: u32 = 0x8702;

fn signal(n: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 19) % 113) as f32 - 56.0) * scale)
        .collect()
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub(super) fn compare(name: &str, got: &[f32], want: &[f32]) -> Result<f64, String> {
    if got.len() != want.len() || got.is_empty() {
        return Err(format!("{name}: invalid oracle span"));
    }
    let mut error = 0.0f64;
    let mut energy = 0.0f64;
    let mut max_abs = 0.0f64;
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        if !a.is_finite() || !b.is_finite() {
            return Err(format!("{name}: nonfinite value {i}: gpu={a} oracle={b}"));
        }
        let delta = f64::from(a) - f64::from(b);
        error += delta * delta;
        energy += f64::from(b) * f64::from(b);
        max_abs = max_abs.max(delta.abs());
    }
    let nmse = if energy == 0.0 {
        if error == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        error / energy
    };
    if !nmse.is_finite() || nmse > 1.0e-6 {
        return Err(format!(
            "{name}: NMSE={nmse:.9e} max_abs={max_abs:.9e}, limit=1e-6"
        ));
    }
    println!(
        "{{\"operation\":\"{name}\",\"values\":{},\"nmse\":{nmse},\"max_abs\":{max_abs}}}",
        got.len()
    );
    Ok(nmse)
}

fn clear_fault(ctx: &Context, fault: *mut c_void) -> Result<(), String> {
    ctx.fill(fault, 255, 4)
}
pub(super) fn fault_is(
    ctx: &Context,
    fault: *mut c_void,
    want: u32,
    name: &str,
) -> Result<(), String> {
    let mut word = [0u32];
    ctx.download(fault, &mut word)?;
    if word[0] != want {
        return Err(format!("{name}: fault={:#x}, expected={want:#x}", word[0]));
    }
    Ok(())
}
fn refused(ctx: &Context, fault: *mut c_void, out: *mut c_void, name: &str) -> Result<(), String> {
    fault_is(ctx, fault, TAG, name)?;
    let mut value = [0.0f32];
    ctx.download(out, &mut value)?;
    if value[0].is_finite() {
        return Err(format!(
            "{name}: fault returned plausible output {}",
            value[0]
        ));
    }
    Ok(())
}

/// Original delta::column_dot_host's key walk and eight-lane butterfly.
fn column_dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    for (j, lane) in lanes.iter_mut().enumerate() {
        let mut p = [0.0f32; 4];
        for i in 0..16 {
            let key = 32 * (i / 4) + 4 * j + i % 4;
            p[i % 4] = if i < 4 {
                a[key] * b[key]
            } else {
                a[key].mul_add(b[key], p[i % 4])
            };
        }
        *lane = (p[0] + p[1]) + (p[2] + p[3]);
    }
    for d in [4, 2, 1] {
        let old = lanes;
        for j in 0..8 {
            lanes[j] += old[j ^ d];
        }
    }
    lanes[0]
}

fn conv_reference(
    x: &[f32],
    weights: &[f32],
    ring: &mut [f32],
    nk: usize,
    nv: usize,
    pos: usize,
) -> Vec<f32> {
    let channels = (2 * nk + nv) * HEAD;
    let mut out = vec![0.0; channels];
    for c in 0..channels {
        let mut acc = 0.0;
        for j in 0..4 {
            let d = 3 - j;
            let v = if d == 0 {
                x[c]
            } else if pos < d {
                0.0
            } else {
                ring[((pos - d) % 11) * channels + c]
            };
            acc = if j == 0 {
                v * weights[4 * c]
            } else {
                v.mul_add(weights[4 * c + j], acc)
            };
        }
        out[c] = silu(acc);
        ring[(pos % 11) * channels + c] = x[c];
    }
    for h in 0..2 * nk {
        let head = &mut out[h * HEAD..(h + 1) * HEAD];
        let sum: f64 = head.iter().map(|&v| (v * v) as f64).sum();
        let scale = 1.0 / (sum as f32).sqrt().max(EPS);
        for v in head {
            *v *= scale;
            if h < nk {
                *v *= f32::from_bits(0x3db5_04f3);
            }
        }
    }
    out
}
fn delta_reference(
    qkv: &[f32],
    beta: &[f32],
    decay: &[f32],
    state: &mut [f32],
    nk: usize,
    nv: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; nv * HEAD];
    for h in 0..nv {
        let kh = h % nk;
        let q = &qkv[kh * HEAD..(kh + 1) * HEAD];
        let k = &qkv[(nk + kh) * HEAD..(nk + kh + 1) * HEAD];
        for col in 0..HEAD {
            let s = &mut state[(h * HEAD + col) * HEAD..(h * HEAD + col + 1) * HEAD];
            for v in s.iter_mut() {
                *v *= decay[h];
            }
            let u = (qkv[2 * nk * HEAD + h * HEAD + col] - column_dot(s, k)) * beta[h];
            for i in 0..HEAD {
                s[i] = k[i].mul_add(u, s[i]);
            }
            out[h * HEAD + col] = column_dot(s, q);
        }
    }
    out
}
fn gate_reference(x: &[f32], z: &[f32], gain: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for (h, head) in x.as_chunks::<HEAD>().0.iter().enumerate() {
        let sum: f64 = head.iter().map(|&v| (v * v) as f64).sum();
        let scale = 1.0 / ((sum / HEAD as f64) as f32 + EPS).sqrt();
        for i in 0..HEAD {
            out[h * HEAD + i] = ((head[i] * scale) * gain[i]) * silu(z[h * HEAD + i]);
        }
    }
    out
}

fn gdn(ctx: &mut Context) -> Result<(), String> {
    // Unequal head counts expose tiled h%nk versus grouped h/(nv/nk).
    let (nk, nv) = (2usize, 4usize);
    let channels = (2 * nk + nv) * HEAD;
    let weights: Vec<_> = (0..channels)
        .flat_map(|c| [0.11 + (c % 3) as f32 * 0.01, -0.19, 0.31, 0.43])
        .collect();
    let gain: Vec<_> = (0..HEAD).map(|i| 0.7 + (i % 13) as f32 * 0.025).collect();
    let bias = signal(nv, 5, 0.006);
    let sa: Vec<_> = (0..nv).map(|h| -0.2 - h as f32 * 0.11).collect();
    let w = ctx.upload(&weights)?;
    let gain_ptr = ctx.upload(&gain)?;
    let bias_ptr = ctx.upload(&bias)?;
    let sa_ptr = ctx.upload(&sa)?;
    let x = ctx.upload(&vec![0.0f32; channels])?;
    let z = ctx.upload(&vec![0.0f32; nv * HEAD])?;
    let b = ctx.upload(&vec![0.0f32; nv])?;
    let a = ctx.upload(&vec![0.0f32; nv])?;
    let ring = ctx.upload(&vec![7.0f32; 11 * channels])?;
    let state = ctx.upload(&vec![0.0f32; nv * HEAD * HEAD])?;
    let qkv = ctx.alloc(channels * 4)?;
    let beta = ctx.alloc(nv * 4)?;
    let decay = ctx.alloc(nv * 4)?;
    let delta = ctx.alloc(nv * HEAD * 4)?;
    let gated = ctx.alloc(nv * HEAD * 4)?;
    let fault = ctx.upload(&[u32::MAX])?;
    let conv_fn = ctx.function("bloomery_hip_gdn_conv")?;
    let decay_fn = ctx.function("bloomery_hip_gdn_decay")?;
    let delta_fn = ctx.function("bloomery_hip_gdn_delta")?;
    let gate_fn = ctx.function("bloomery_hip_gdn_gate")?;
    let mut worst = [0.0f64; 5];
    let mut repeat = Vec::new();
    for pass in 0..2 {
        let mut host_ring = vec![if pass == 0 { 7.0 } else { 0.0 }; 11 * channels];
        let mut host_state = vec![0.0; nv * HEAD * HEAD];
        if pass != 0 {
            ctx.fill(ring, 0, 11 * channels * 4)?;
            ctx.fill(state, 0, nv * HEAD * HEAD * 4)?;
        }
        for pos in 0..15 {
            let xv = signal(channels, 9 + pos, 0.012);
            let zv = signal(nv * HEAD, 31 + pos, 0.019);
            let bv = signal(nv, 17 + pos, 0.021);
            let av = signal(nv, 23 + pos, 0.017);
            ctx.overwrite(x, &xv)?;
            ctx.overwrite(z, &zv)?;
            ctx.overwrite(b, &bv)?;
            ctx.overwrite(a, &av)?;
            ctx.call(
                conv_fn,
                2 * nk + nv,
                [
                    Arg::ptr(x),
                    Arg::ptr(w),
                    Arg::ptr(ring),
                    Arg::ptr(qkv),
                    Arg::u32(nk as u32),
                    Arg::u32(nv as u32),
                    Arg::u32(pos as u32),
                    Arg::f32(EPS),
                    Arg::ptr(fault),
                    Arg::u32(TAG),
                ],
            )?;
            ctx.call(
                decay_fn,
                nv,
                [
                    Arg::ptr(b),
                    Arg::ptr(a),
                    Arg::ptr(bias_ptr),
                    Arg::ptr(sa_ptr),
                    Arg::ptr(beta),
                    Arg::ptr(decay),
                    Arg::u32(nv as u32),
                    Arg::ptr(fault),
                    Arg::u32(TAG),
                ],
            )?;
            ctx.call(
                delta_fn,
                nv * HEAD * 8,
                [
                    Arg::ptr(qkv),
                    Arg::ptr(beta),
                    Arg::ptr(decay),
                    Arg::ptr(state),
                    Arg::ptr(delta),
                    Arg::u32(nk as u32),
                    Arg::u32(nv as u32),
                    Arg::ptr(fault),
                    Arg::u32(TAG),
                ],
            )?;
            ctx.call(
                gate_fn,
                nv,
                [
                    Arg::ptr(delta),
                    Arg::ptr(z),
                    Arg::ptr(gain_ptr),
                    Arg::ptr(gated),
                    Arg::u32(nv as u32),
                    Arg::f32(EPS),
                    Arg::ptr(fault),
                    Arg::u32(TAG),
                ],
            )?;
            fault_is(ctx, fault, u32::MAX, "gdn-finite")?;
            let expected_qkv = conv_reference(&xv, &weights, &mut host_ring, nk, nv, pos);
            let expected_beta: Vec<_> = bv.iter().map(|&v| sigmoid(v)).collect();
            let expected_decay: Vec<_> = av
                .iter()
                .enumerate()
                .map(|(h, &v)| {
                    let step = v + bias[h];
                    let sp = if step > 20.0 {
                        step
                    } else {
                        (1.0 + step.exp()).ln()
                    };
                    (sp * sa[h]).exp()
                })
                .collect();
            let expected_delta = delta_reference(
                &expected_qkv,
                &expected_beta,
                &expected_decay,
                &mut host_state,
                nk,
                nv,
            );
            let expected_gate = gate_reference(&expected_delta, &zv, &gain);
            let mut got_qkv = vec![0.0; channels];
            ctx.download(qkv, &mut got_qkv)?;
            let mut got_beta = vec![0.0; nv];
            ctx.download(beta, &mut got_beta)?;
            let mut got_decay = vec![0.0; nv];
            ctx.download(decay, &mut got_decay)?;
            let mut got_delta = vec![0.0; nv * HEAD];
            ctx.download(delta, &mut got_delta)?;
            let mut got_gate = vec![0.0; nv * HEAD];
            ctx.download(gated, &mut got_gate)?;
            let mut got_state = vec![0.0; host_state.len()];
            ctx.download(state, &mut got_state)?;
            let mut got_ring = vec![0.0; host_ring.len()];
            ctx.download(ring, &mut got_ring)?;
            for (i, (name, got, want)) in [
                (
                    "conv-oldest-first-ring-wrap",
                    got_qkv.as_slice(),
                    expected_qkv.as_slice(),
                ),
                ("delta-tiled-value-major", &got_delta, &expected_delta),
                ("delta-recurrent-state", &got_state, &host_state),
                ("gdn-gated-norm", &got_gate, &expected_gate),
                ("gdn-decay", &got_decay, &expected_decay),
            ]
            .into_iter()
            .enumerate()
            {
                worst[i] = worst[i].max(compare(
                    &format!("{name}/pass={pass}/pos={pos}"),
                    got,
                    want,
                )?);
            }
            compare("gdn-beta", &got_beta, &expected_beta)?;
            if got_ring != host_ring {
                return Err(format!(
                    "conv-input-ring/pass={pass}/pos={pos}: raw-input rows differ"
                ));
            }
            if pass == 0 {
                repeat.extend(got_gate.iter().map(|v| v.to_bits()));
            } else {
                let expected = &repeat[pos * nv * HEAD..(pos + 1) * nv * HEAD];
                if got_gate
                    .iter()
                    .map(|v| v.to_bits())
                    .ne(expected.iter().copied())
                {
                    return Err(format!("gdn-reset-repeat/pos={pos}: output bits differ"));
                }
            }
        }
    }
    println!(
        "forward-mixer GDN positions=15 passes=2 ring_rows=11 nk=2 nv=4 worst_nmse={worst:?} reset_repeat=bitwise"
    );
    // Explicit refusals, not merely observing that an invalid launch completed.
    clear_fault(ctx, fault)?;
    ctx.call(
        conv_fn,
        2 * nk + nv,
        [
            Arg::ptr(x),
            Arg::ptr(w),
            Arg::ptr(ring),
            Arg::ptr(qkv),
            Arg::u32(nk as u32),
            Arg::u32(nv as u32),
            Arg::u32(0),
            Arg::f32(0.0),
            Arg::ptr(fault),
            Arg::u32(TAG),
        ],
    )?;
    refused(ctx, fault, qkv, "conv-invalid-epsilon")?;
    clear_fault(ctx, fault)?;
    ctx.overwrite(b, &[f32::INFINITY, 0.2, 0.3, 0.4])?;
    ctx.call(
        decay_fn,
        nv,
        [
            Arg::ptr(b),
            Arg::ptr(a),
            Arg::ptr(bias_ptr),
            Arg::ptr(sa_ptr),
            Arg::ptr(beta),
            Arg::ptr(decay),
            Arg::u32(nv as u32),
            Arg::ptr(fault),
            Arg::u32(TAG),
        ],
    )?;
    refused(ctx, fault, beta, "decay-infinite-beta-input")?;
    clear_fault(ctx, fault)?;
    ctx.overwrite(qkv, &signal(channels, 2, 0.003))?;
    ctx.overwrite(beta, &[0.5f32; 4])?;
    ctx.overwrite(decay, &[0.8f32; 4])?;
    let mut bad_state = vec![0.0; nv * HEAD * HEAD];
    bad_state[0] = f32::NAN;
    ctx.overwrite(state, &bad_state)?;
    ctx.call(
        delta_fn,
        nv * HEAD * 8,
        [
            Arg::ptr(qkv),
            Arg::ptr(beta),
            Arg::ptr(decay),
            Arg::ptr(state),
            Arg::ptr(delta),
            Arg::u32(nk as u32),
            Arg::u32(nv as u32),
            Arg::ptr(fault),
            Arg::u32(TAG),
        ],
    )?;
    refused(ctx, fault, delta, "delta-nonfinite-state")?;
    clear_fault(ctx, fault)?;
    ctx.overwrite(delta, &vec![f32::MAX; nv * HEAD])?;
    ctx.call(
        gate_fn,
        nv,
        [
            Arg::ptr(delta),
            Arg::ptr(z),
            Arg::ptr(gain_ptr),
            Arg::ptr(gated),
            Arg::u32(nv as u32),
            Arg::f32(EPS),
            Arg::ptr(fault),
            Arg::u32(TAG),
        ],
    )?;
    refused(ctx, fault, gated, "gate-overflowed-square")?;
    Ok(())
}

fn normalized_rope(x: &[f32], gain: &[f32], row: &[f32]) -> Vec<f32> {
    let sum: f64 = x.iter().map(|&v| (v * v) as f64).sum();
    let scale = 1.0 / ((sum / x.len() as f64) as f32 + EPS).sqrt();
    let mut y: Vec<_> = x.iter().zip(gain).map(|(&v, &g)| (scale * g) * v).collect();
    let half = row.len() / 2;
    for i in 0..half {
        let a = y[i];
        let b = y[i + half];
        y[i] = a.mul_add(row[i], -(b * row[i + half]));
        y[i + half] = a.mul_add(row[i + half], b * row[i]);
    }
    y
}

struct AttentionGeometry {
    qh: usize,
    kh: usize,
    width: usize,
    context: usize,
}

fn attention_reference(
    q: &[f32],
    qgate: &[f32],
    k: &[u16],
    v: &[u16],
    geometry: &AttentionGeometry,
    pos: usize,
) -> Vec<f32> {
    let AttentionGeometry {
        qh,
        kh,
        width,
        context,
    } = *geometry;
    let mut out = vec![0.0; qh * width];
    for h in 0..qh {
        let kvh = h / (qh / kh);
        // Independent F64 score/softmax referee, using the actual F16 cache contract.
        let scores: Vec<f64> = (0..=pos)
            .map(|t| {
                let at = (kvh * context + t) * width;
                (0..width)
                    .map(|i| q[h * width + i] as f64 * half_to_f32(k[at + i]) as f64)
                    .sum::<f64>()
                    / (width as f64).sqrt()
            })
            .collect();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<_> = scores.iter().map(|&s| (s - max).exp()).collect();
        let sum: f64 = weights.iter().sum();
        for i in 0..width {
            let value: f64 = (0..=pos)
                .map(|t| weights[t] * half_to_f32(v[(kvh * context + t) * width + i]) as f64)
                .sum();
            out[h * width + i] = ((value / sum) as f32) * sigmoid(qgate[h * 2 * width + width + i]);
        }
    }
    out
}

fn gqa(ctx: &mut Context) -> Result<(), String> {
    // Exercise full production head width and 64 partial-Neox values, but only four heads.
    let (qh, kh, width, rot, context) = (4usize, 2usize, 256usize, 64usize, 64usize);
    let qgain: Vec<_> = (0..width).map(|i| 0.6 + (i % 17) as f32 * 0.035).collect();
    let kgain: Vec<_> = (0..width).map(|i| 0.8 + (i % 11) as f32 * 0.027).collect();
    let rope: Vec<_> = (0..context)
        .flat_map(|p| {
            let angles: Vec<_> = (0..rot / 2)
                .map(|i| p as f64 * 10000.0f64.powf(-((2 * i) as f64) / rot as f64))
                .collect();
            angles
                .iter()
                .map(|a| a.cos() as f32)
                .chain(angles.iter().map(|a| a.sin() as f32))
                .collect::<Vec<_>>()
        })
        .collect();
    let qg = ctx.upload(&vec![0.0f32; qh * 2 * width])?;
    let kp = ctx.upload(&vec![0.0f32; kh * width])?;
    let vp = ctx.upload(&vec![0.0f32; kh * width])?;
    let qgp = ctx.upload(&qgain)?;
    let kgp = ctx.upload(&kgain)?;
    let rp = ctx.upload(&rope)?;
    let qo = ctx.alloc(qh * width * 4)?;
    let out = ctx.alloc(qh * width * 4)?;
    let poison = vec![0x7e00u16; kh * context * width];
    let kc = ctx.upload(&poison)?;
    let vc = ctx.upload(&poison)?;
    let fault = ctx.upload(&[u32::MAX])?;
    let prep_fn = ctx.function("bloomery_hip_gqa_prep")?;
    let attn_fn = ctx.function("bloomery_hip_attention")?;
    let mut worst_q = 0.0f64;
    let mut worst_attention = 0.0f64;
    let mut repeat = Vec::new();
    for pass in 0..2 {
        let mut host_k = poison.clone();
        let mut host_v = poison.clone();
        if pass != 0 {
            ctx.overwrite(kc, &poison)?;
            ctx.overwrite(vc, &poison)?;
        }
        for pos in 0..context {
            let qgate = signal(qh * 2 * width, 51 + pos, 0.023);
            let k = signal(kh * width, 71 + pos, 0.017);
            let mut v = signal(kh * width, 91 + pos, 0.029);
            // RNE ties (including subnormal and signed zero) bypass norm in V.
            for (i, value) in [
                f32::from_bits(0x3f80_1000),
                f32::from_bits(0x3f80_3000),
                f32::from_bits(0xbf80_1000),
                f32::from_bits(0x3300_0000),
                f32::from_bits(0x33c0_0000),
                -0.0,
            ]
            .into_iter()
            .enumerate()
            {
                v[i] = value;
            }
            ctx.overwrite(qg, &qgate)?;
            ctx.overwrite(kp, &k)?;
            ctx.overwrite(vp, &v)?;
            ctx.call(
                prep_fn,
                qh + kh,
                [
                    Arg::ptr(qg),
                    Arg::ptr(kp),
                    Arg::ptr(vp),
                    Arg::ptr(qgp),
                    Arg::ptr(kgp),
                    Arg::ptr(rp),
                    Arg::ptr(qo),
                    Arg::ptr(kc),
                    Arg::ptr(vc),
                    Arg::u32(qh as u32),
                    Arg::u32(kh as u32),
                    Arg::u32(width as u32),
                    Arg::u32(rot as u32),
                    Arg::u32(pos as u32),
                    Arg::u32(context as u32),
                    Arg::f32(EPS),
                    Arg::ptr(fault),
                    Arg::u32(TAG),
                ],
            )?;
            ctx.call(
                attn_fn,
                qh,
                [
                    Arg::ptr(qo),
                    Arg::ptr(qg),
                    Arg::ptr(kc),
                    Arg::ptr(vc),
                    Arg::ptr(out),
                    Arg::u32(qh as u32),
                    Arg::u32(kh as u32),
                    Arg::u32(width as u32),
                    Arg::u32(pos as u32),
                    Arg::u32(context as u32),
                    Arg::ptr(fault),
                    Arg::u32(TAG),
                ],
            )?;
            fault_is(ctx, fault, u32::MAX, "gqa-causal-poisoned-future")?;
            let row = &rope[pos * rot..(pos + 1) * rot];
            let mut want_q = Vec::new();
            for h in 0..qh {
                want_q.extend(normalized_rope(
                    &qgate[h * 2 * width..h * 2 * width + width],
                    &qgain,
                    row,
                ));
            }
            for h in 0..kh {
                let key = normalized_rope(&k[h * width..(h + 1) * width], &kgain, row);
                for i in 0..width {
                    let at = (h * context + pos) * width + i;
                    host_k[at] = f32_to_f16_bits(key[i]);
                    host_v[at] = f32_to_f16_bits(v[h * width + i]);
                }
            }
            let mut got_q = vec![0.0; qh * width];
            ctx.download(qo, &mut got_q)?;
            worst_q = worst_q.max(compare(
                &format!("gqa-qnorm-partial-neox/pass={pass}/pos={pos}"),
                &got_q,
                &want_q,
            )?);
            let mut got_k = vec![0u16; host_k.len()];
            ctx.download(kc, &mut got_k)?;
            let mut got_v = vec![0u16; host_v.len()];
            ctx.download(vc, &mut got_v)?;
            if got_v != host_v {
                return Err(format!(
                    "gqa-f16-rne-v/pass={pass}/pos={pos}: cache bits differ"
                ));
            }
            if got_k != host_k {
                return Err(format!(
                    "gqa-f16-rne-k/pass={pass}/pos={pos}: norm/rope cache bits differ"
                ));
            }
            let want = attention_reference(
                &want_q,
                &qgate,
                &host_k,
                &host_v,
                &AttentionGeometry {
                    qh,
                    kh,
                    width,
                    context,
                },
                pos,
            );
            let mut got = vec![0.0; qh * width];
            ctx.download(out, &mut got)?;
            worst_attention = worst_attention.max(compare(
                &format!("gqa-grouped-gated-causal/pass={pass}/pos={pos}"),
                &got,
                &want,
            )?);
            if pass == 0 {
                repeat.extend(got.iter().map(|v| v.to_bits()));
            } else if got
                .iter()
                .map(|v| v.to_bits())
                .ne(repeat[pos * qh * width..(pos + 1) * qh * width]
                    .iter()
                    .copied())
            {
                return Err(format!("gqa-reset-repeat/pos={pos}: output bits differ"));
            }
        }
    }
    println!(
        "forward-mixer GQA qh=4 kvh=2 width=256 rope=64 positions={context} passes=2 q_nmse={worst_q:.9e} attention_nmse={worst_attention:.9e} f16_cache=bitwise reset_repeat=bitwise poisoned_future=unread"
    );
    clear_fault(ctx, fault)?;
    ctx.overwrite(qg, &vec![f32::MAX; qh * 2 * width])?;
    ctx.call(
        prep_fn,
        qh + kh,
        [
            Arg::ptr(qg),
            Arg::ptr(kp),
            Arg::ptr(vp),
            Arg::ptr(qgp),
            Arg::ptr(kgp),
            Arg::ptr(rp),
            Arg::ptr(qo),
            Arg::ptr(kc),
            Arg::ptr(vc),
            Arg::u32(qh as u32),
            Arg::u32(kh as u32),
            Arg::u32(width as u32),
            Arg::u32(rot as u32),
            Arg::u32(0),
            Arg::u32(context as u32),
            Arg::f32(EPS),
            Arg::ptr(fault),
            Arg::u32(TAG),
        ],
    )?;
    refused(ctx, fault, qo, "gqa-norm-overflowed-square")?;
    clear_fault(ctx, fault)?;
    ctx.overwrite(qg, &signal(qh * 2 * width, 2, 0.01))?;
    ctx.overwrite(qo, &signal(qh * width, 3, 0.01))?;
    ctx.overwrite(kc, &poison)?;
    ctx.call(
        attn_fn,
        qh,
        [
            Arg::ptr(qo),
            Arg::ptr(qg),
            Arg::ptr(kc),
            Arg::ptr(vc),
            Arg::ptr(out),
            Arg::u32(qh as u32),
            Arg::u32(kh as u32),
            Arg::u32(width as u32),
            Arg::u32(0),
            Arg::u32(context as u32),
            Arg::ptr(fault),
            Arg::u32(TAG),
        ],
    )?;
    refused(ctx, fault, out, "attention-live-nonfinite-key")?;
    clear_fault(ctx, fault)?;
    ctx.overwrite(kc, &vec![0u16; kh * context * width])?;
    let mut bad_gate = signal(qh * 2 * width, 2, 0.01);
    bad_gate[width] = f32::INFINITY;
    ctx.overwrite(qg, &bad_gate)?;
    ctx.call(
        attn_fn,
        qh,
        [
            Arg::ptr(qo),
            Arg::ptr(qg),
            Arg::ptr(kc),
            Arg::ptr(vc),
            Arg::ptr(out),
            Arg::u32(qh as u32),
            Arg::u32(kh as u32),
            Arg::u32(width as u32),
            Arg::u32(0),
            Arg::u32(context as u32),
            Arg::ptr(fault),
            Arg::u32(TAG),
        ],
    )?;
    refused(ctx, fault, out, "attention-infinite-gate")?;
    Ok(())
}

pub(super) fn run(ctx: &mut Context) -> Result<(), String> {
    crate::selected::wave_size(ctx)?;
    gdn(ctx)?;
    gqa(ctx)?;
    println!(
        "forward-mixer refusals=conv-eps,decay-beta,delta-state,gate-square,gqa-square,attention-key,attention-gate band_nmse<=1e-6"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::compare;

    #[test]
    fn comparison_enforces_oracle_span_finiteness_and_nmse_contract() {
        assert_eq!(compare("zero", &[0.0, -0.0], &[0.0, 0.0]).unwrap(), 0.0);
        // A zero-energy oracle requires exact agreement; a denominator floor
        // can conceal a nonzero F64 error.
        assert!(compare("zero-oracle", &[1.0e-20], &[0.0]).is_err());
        assert!(compare("tiny-oracle", &[2.0e-20], &[1.0e-20]).is_err());
        assert_eq!(compare("exact", &[f32::MAX], &[f32::MAX]).unwrap(), 0.0);
        assert_eq!(compare("at-limit", &[1001.0], &[1000.0]).unwrap(), 1.0e-6);
        assert!(compare("below-limit", &[1000.5], &[1000.0]).is_ok());
        assert!(compare("above-limit", &[1001.0001], &[1000.0]).is_err());
        assert!(compare("large-error", &[-f32::MAX], &[f32::MAX]).is_err());
        assert!(compare("empty", &[], &[]).is_err());
        assert!(compare("short-output", &[], &[0.0]).is_err());
        assert!(compare("long-output", &[0.0], &[]).is_err());
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(compare("nonfinite-output", &[value], &[1.0]).is_err());
            assert!(compare("nonfinite-oracle", &[1.0], &[value]).is_err());
        }
    }
}
