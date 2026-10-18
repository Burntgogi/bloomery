//! Bounded GPU operator evidence; CPU arithmetic here is an oracle, never a forward fallback.

use crate::{Arg, Context};
use std::ffi::c_void;

struct Kernels {
    embed: *mut c_void,
    gemv: *mut c_void,
    norm: *mut c_void,
    glu: *mut c_void,
    add: *mut c_void,
    argmax: *mut c_void,
}

fn compare(name: &str, got: &[f32], want: &[f32]) -> Result<(), String> {
    if got.len() != want.len() || got.is_empty() {
        return Err(format!("{name}: oracle/output shape mismatch"));
    }
    let mut error = 0.0_f64;
    let mut energy = 0.0_f64;
    let mut maximum = 0.0_f64;
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        if !a.is_finite() || !b.is_finite() {
            return Err(format!("{name}: nonfinite output/oracle at {i}: {a}/{b}"));
        }
        let delta = f64::from(a) - f64::from(b);
        error += delta * delta;
        energy += f64::from(b) * f64::from(b);
        maximum = maximum.max(delta.abs());
    }
    let nmse = error / energy.max(1e-30);
    if !nmse.is_finite() || nmse > 1e-6 {
        return Err(format!(
            "{name}: NMSE {nmse} exceeds fixed 1e-6, max_abs={maximum}"
        ));
    }
    println!(
        "{{\"operation\":\"{name}\",\"values\":{},\"nmse\":{nmse},\"max_abs\":{maximum}}}",
        got.len()
    );
    Ok(())
}

fn fault_is(ctx: &Context, fault: *mut c_void, expected: u32, name: &str) -> Result<(), String> {
    let mut got = [0_u32];
    ctx.download(fault, &mut got)?;
    if got[0] != expected {
        return Err(format!(
            "{name}: fault={:#x}, expected={expected:#x}",
            got[0]
        ));
    }
    Ok(())
}

fn fixture(kind: u32, width: usize, rows: usize) -> Vec<u8> {
    let scales = [0x1800_u16, 0x9800, 0x0001, 0x0000, 0x1400, 0x1c00];
    match kind {
        0 => (0..width * rows)
            .flat_map(|i| (((i * 17 % 127) as f32 - 63.0) * 0.0078125).to_le_bytes())
            .collect(),
        12 => {
            let mut out = crate::fixture(width, rows);
            for (i, block) in out.as_chunks_mut::<144>().0.iter_mut().enumerate() {
                block[..2].copy_from_slice(&scales[i % scales.len()].to_le_bytes());
                block[2..4].copy_from_slice(&scales[(i + 4) % scales.len()].to_le_bytes());
            }
            out
        }
        14 => {
            let mut out = vec![0_u8; width / 256 * rows * 210];
            for (i, block) in out.as_chunks_mut::<210>().0.iter_mut().enumerate() {
                for (j, byte) in block[..208].iter_mut().enumerate() {
                    *byte = (j * 37 + i * 73 + 129) as u8;
                }
                block[208..].copy_from_slice(&scales[i % scales.len()].to_le_bytes());
            }
            out
        }
        8 => {
            let mut out = vec![0_u8; width / 32 * rows * 34];
            for (i, block) in out.as_chunks_mut::<34>().0.iter_mut().enumerate() {
                block[..2].copy_from_slice(&scales[i % scales.len()].to_le_bytes());
                for (j, byte) in block[2..].iter_mut().enumerate() {
                    *byte = (j * 31 + i * 79 + 128) as u8;
                }
            }
            out
        }
        _ => unreachable!("probe fixture kind"),
    }
}

fn packed(ctx: &mut Context, kernels: &Kernels, fault: *mut c_void) -> Result<(), String> {
    for (kind, width, rows) in [(0, 67, 3), (12, 512, 3), (14, 512, 3), (8, 96, 3)] {
        let weights = fixture(kind, width, rows);
        let x: Vec<f32> = (0..width)
            .map(|i| ((i * 13 % 97) as f32 - 48.0) * 0.03125 + 0.00390625)
            .collect();
        let mut decoded = vec![0.0_f32; width * rows];
        let row_bytes = weights.len() / rows;
        for (row, out) in decoded.chunks_exact_mut(width).enumerate() {
            gguf::dequant_row(
                gguf::GgmlType::from_u32(kind),
                &weights[row * row_bytes..(row + 1) * row_bytes],
                out,
            )
            .map_err(|e| format!("core packed oracle kind={kind}: {e}"))?;
        }
        let dw = ctx.upload(&weights)?;
        let dx = ctx.upload(&x)?;
        let dy = ctx.alloc(width.max(rows) * 4)?;
        ctx.fill(fault, 255, 4)?;
        let want: Vec<f32> = decoded
            .chunks_exact(width)
            .map(|row| {
                row.iter()
                    .zip(&x)
                    .map(|(&w, &a)| f64::from(w) * f64::from(a))
                    .sum::<f64>() as f32
            })
            .collect();
        ctx.call(
            kernels.gemv,
            rows,
            [
                Arg::ptr(dw),
                Arg::ptr(dx),
                Arg::ptr(dy),
                Arg::u32(width as u32),
                Arg::u32(rows as u32),
                Arg::u32(kind),
                Arg::ptr(fault),
                Arg::u32(0x101),
            ],
        )?;
        let mut got = vec![0.0_f32; rows];
        ctx.download(dy, &mut got)?;
        fault_is(ctx, fault, u32::MAX, "core_gemv_finite")?;
        compare(&format!("core_gemv_kind_{kind}"), &got, &want)?;
        for token in [0, 1, rows - 1] {
            ctx.call(
                kernels.embed,
                width,
                [
                    Arg::ptr(dw),
                    Arg::ptr(dy),
                    Arg::u32(width as u32),
                    Arg::u32(token as u32),
                    Arg::u32(kind),
                    Arg::ptr(fault),
                    Arg::u32(0x102),
                ],
            )?;
            let mut got = vec![0.0_f32; width];
            ctx.download(dy, &mut got)?;
            let want = &decoded[token * width..(token + 1) * width];
            fault_is(ctx, fault, u32::MAX, "core_embed_finite")?;
            compare(&format!("core_embed_kind_{kind}_token_{token}"), &got, want)?;
            if let Some(i) = got
                .iter()
                .zip(want)
                .position(|(a, b)| a.to_bits() != b.to_bits())
            {
                return Err(format!(
                    "core_embed_kind_{kind}_token_{token}: decoded bits differ at {i}"
                ));
            }
        }
        // Nonfinite scale/payload must not disappear in a packed decoder or zero activation.
        let mut bad = weights;
        if kind == 0 {
            bad[..4].copy_from_slice(&f32::INFINITY.to_le_bytes());
        } else {
            let offset = if kind == 14 { 208 } else { 0 };
            bad[offset..offset + 2].copy_from_slice(&0x7c00_u16.to_le_bytes());
        }
        let dbad = ctx.upload(&bad)?;
        for (f, name) in [(kernels.embed, "embed"), (kernels.gemv, "gemv")] {
            ctx.fill(fault, 255, 4)?;
            if name == "embed" {
                ctx.call(
                    f,
                    width,
                    [
                        Arg::ptr(dbad),
                        Arg::ptr(dy),
                        Arg::u32(width as u32),
                        Arg::u32(0),
                        Arg::u32(kind),
                        Arg::ptr(fault),
                        Arg::u32(0x103),
                    ],
                )?;
            } else {
                ctx.call(
                    f,
                    rows,
                    [
                        Arg::ptr(dbad),
                        Arg::ptr(dx),
                        Arg::ptr(dy),
                        Arg::u32(width as u32),
                        Arg::u32(rows as u32),
                        Arg::u32(kind),
                        Arg::ptr(fault),
                        Arg::u32(0x103),
                    ],
                )?;
            }
            fault_is(
                ctx,
                fault,
                0x103,
                &format!("core_{name}_kind_{kind}_nonfinite"),
            )?;
            let mut got = [0.0_f32];
            ctx.download(dy, &mut got)?;
            if got[0].is_finite() {
                return Err(format!(
                    "core_{name}_kind_{kind}_nonfinite: plausible output"
                ));
            }
        }
    }
    Ok(())
}

fn elementwise(ctx: &mut Context, kernels: &Kernels, fault: *mut c_void) -> Result<(), String> {
    let (width, heads) = (67_usize, 3_usize);
    let eps = 1e-6_f32;
    let x: Vec<f32> = (0..width * heads)
        .map(|i| {
            if i / width == 1 {
                0.0
            } else {
                ((i * 7 % 53) as f32 - 26.0) * 0.0625
            }
        })
        .collect();
    let gain: Vec<f32> = (0..width).map(|i| 0.5 + (i % 17) as f32 * 0.0625).collect();
    let gate: Vec<f32> = (0..x.len())
        .map(|i| {
            let boundary = [-100.0, -20.0, -1.0, -0.0, 0.0, 1.0, 20.0, 100.0];
            if i < boundary.len() {
                boundary[i]
            } else {
                ((i * 11 % 89) as f32 - 44.0) * 0.125
            }
        })
        .collect();
    let dx = ctx.upload(&x)?;
    let dg = ctx.upload(&gain)?;
    let gate_ptr = ctx.upload(&gate)?;
    let dy = ctx.alloc(x.len() * 4)?;
    ctx.fill(fault, 255, 4)?;
    ctx.call(
        kernels.norm,
        heads,
        [
            Arg::ptr(dx),
            Arg::ptr(dg),
            Arg::ptr(dy),
            Arg::u32(width as u32),
            Arg::u32(heads as u32),
            Arg::f32(eps),
            Arg::ptr(fault),
            Arg::u32(0x110),
        ],
    )?;
    let want: Vec<f32> = x
        .chunks_exact(width)
        .flat_map(|head| {
            let mean = (qdot::sum_sq_f64(head) / width as f64) as f32;
            let scale = 1.0 / (mean + eps).sqrt();
            head.iter().zip(&gain).map(move |(&v, &g)| (scale * g) * v)
        })
        .collect();
    let mut got = vec![0.0_f32; x.len()];
    ctx.download(dy, &mut got)?;
    fault_is(ctx, fault, u32::MAX, "core_norm_finite")?;
    compare("core_grouped_norm", &got, &want)?;
    ctx.call(
        kernels.glu,
        x.len(),
        [
            Arg::ptr(gate_ptr),
            Arg::ptr(dx),
            Arg::ptr(dy),
            Arg::u32(x.len() as u32),
            Arg::ptr(fault),
            Arg::u32(0x111),
        ],
    )?;
    let want: Vec<f32> = gate
        .iter()
        .zip(&x)
        .map(|(&g, &u)| (g / (1.0 + qdot::expf_ik_scalar(0.0 - g))) * u)
        .collect();
    ctx.download(dy, &mut got)?;
    fault_is(ctx, fault, u32::MAX, "core_glu_finite")?;
    compare("core_glu", &got, &want)?;
    ctx.call(
        kernels.add,
        x.len(),
        [
            Arg::ptr(dx),
            Arg::ptr(dy),
            Arg::u32(x.len() as u32),
            Arg::ptr(fault),
            Arg::u32(0x112),
        ],
    )?;
    let want: Vec<f32> = want.iter().zip(&x).map(|(&a, &b)| a + b).collect();
    ctx.download(dy, &mut got)?;
    fault_is(ctx, fault, u32::MAX, "core_add_finite")?;
    compare("core_residual", &got, &want)?;
    // In-place normalization must reduce the original head before writing any element.
    let inplace = ctx.upload(&x)?;
    ctx.call(
        kernels.norm,
        heads,
        [
            Arg::ptr(inplace),
            Arg::ptr(dg),
            Arg::ptr(inplace),
            Arg::u32(width as u32),
            Arg::u32(heads as u32),
            Arg::f32(eps),
            Arg::ptr(fault),
            Arg::u32(0x113),
        ],
    )?;
    ctx.download(inplace, &mut got)?;
    let want: Vec<f32> = x
        .chunks_exact(width)
        .flat_map(|head| {
            let scale = 1.0 / ((qdot::sum_sq_f64(head) / width as f64) as f32 + eps).sqrt();
            head.iter().zip(&gain).map(move |(&v, &g)| (scale * g) * v)
        })
        .collect();
    compare("core_norm_in_place", &got, &want)?;
    fault_is(ctx, fault, u32::MAX, "core_norm_in_place")?;
    for (name, values, epsilon) in [
        ("invalid_epsilon", vec![1.0_f32, 2.0], -1.0_f32),
        ("nan_epsilon", vec![1.0_f32, 2.0], f32::NAN),
        ("nan_input", vec![f32::NAN, 2.0], eps),
        ("square_overflow", vec![f32::MAX, 2.0], eps),
    ] {
        let bad = ctx.upload(&values)?;
        ctx.fill(fault, 255, 4)?;
        ctx.call(
            kernels.norm,
            1,
            [
                Arg::ptr(bad),
                Arg::ptr(dg),
                Arg::ptr(dy),
                Arg::u32(2),
                Arg::u32(1),
                Arg::f32(epsilon),
                Arg::ptr(fault),
                Arg::u32(0x114),
            ],
        )?;
        fault_is(ctx, fault, 0x114, &format!("core_norm_{name}"))?;
        let mut out = [0.0_f32; 2];
        ctx.download(dy, &mut out)?;
        if out.iter().any(|v| v.is_finite()) {
            return Err(format!("core_norm_{name}: nonfinite reduction masked"));
        }
    }
    Ok(())
}

fn refusals(ctx: &mut Context, kernels: &Kernels, fault: *mut c_void) -> Result<(), String> {
    let x = ctx.upload(&[1.0_f32; 67])?;
    let bad_values: Vec<f32> = (0..67)
        .map(|i| if i % 2 == 0 { f32::NAN } else { f32::INFINITY })
        .collect();
    let bad = ctx.upload(&bad_values)?;
    let out = ctx.alloc(67 * 4)?;
    for kind in [99, 12, 14, 8] {
        // Width 67 is invalid for every packed type and 99 is unsupported.
        ctx.fill(fault, 255, 4)?;
        ctx.call(
            kernels.embed,
            67,
            [
                Arg::ptr(x),
                Arg::ptr(out),
                Arg::u32(67),
                Arg::u32(0),
                Arg::u32(kind),
                Arg::ptr(fault),
                Arg::u32(0x120),
            ],
        )?;
        fault_is(ctx, fault, 0x120, "core_embed_invalid_kind_or_width")?;
        let mut got = [0.0_f32; 67];
        ctx.download(out, &mut got)?;
        if got.iter().any(|v| !v.is_nan()) {
            return Err(format!(
                "core_embed_invalid_kind_or_width kind={kind}: not NaN"
            ));
        }
        ctx.fill(fault, 255, 4)?;
        ctx.call(
            kernels.gemv,
            1,
            [
                Arg::ptr(x),
                Arg::ptr(x),
                Arg::ptr(out),
                Arg::u32(67),
                Arg::u32(1),
                Arg::u32(kind),
                Arg::ptr(fault),
                Arg::u32(0x121),
            ],
        )?;
        fault_is(ctx, fault, 0x121, "core_gemv_invalid_kind_or_width")?;
        ctx.download(out, &mut got[..1])?;
        if !got[0].is_nan() {
            return Err(format!(
                "core_gemv_invalid_kind_or_width kind={kind}: not NaN"
            ));
        }
    }
    ctx.fill(fault, 255, 4)?;
    ctx.call(
        kernels.glu,
        67,
        [
            Arg::ptr(bad),
            Arg::ptr(x),
            Arg::ptr(out),
            Arg::u32(67),
            Arg::ptr(fault),
            Arg::u32(0x139),
        ],
    )?;
    fault_is(ctx, fault, 0x139, "core_glu_nonfinite")?;
    ctx.call(
        kernels.add,
        67,
        [
            Arg::ptr(bad),
            Arg::ptr(out),
            Arg::u32(67),
            Arg::ptr(fault),
            Arg::u32(0x107),
        ],
    )?;
    fault_is(ctx, fault, 0x107, "core_add_atomic_min")?;
    let mut got = [0.0_f32; 67];
    ctx.download(out, &mut got)?;
    if got.iter().any(|v| v.is_finite()) {
        return Err("core_glu_add_nonfinite: plausible output".into());
    }
    let summary = ctx.alloc(8)?;
    ctx.call(
        kernels.argmax,
        1,
        [
            Arg::ptr(x),
            Arg::ptr(summary),
            Arg::u32(67),
            Arg::ptr(fault),
            Arg::u32(0x140),
        ],
    )?;
    let mut words = [0_u32; 2];
    ctx.download(summary, &mut words)?;
    if words != [0, 0x107] {
        return Err(format!("core_argmax_prior_fault: summary={words:?}"));
    }
    for (name, logits, n, expected) in [
        ("tie", vec![-4.0_f32, 3.0, 3.0, 1.0], 4, 1),
        ("negative", vec![f32::MIN, -3.0, -3.0, -4.0], 4, 1),
        ("minimum", vec![f32::MIN, f32::MIN], 2, 0),
        ("tail", (0..67).map(|i| i as f32 - 100.0).collect(), 67, 66),
        (
            "cross_wave_tie",
            (0..67)
                .map(|i| if i == 63 || i == 66 { 3.0 } else { -1.0 })
                .collect(),
            67,
            63,
        ),
        ("nan", vec![1.0, f32::NAN, 2.0], 3, u32::MAX),
        (
            "positive_infinity",
            vec![1.0, f32::INFINITY, 2.0],
            3,
            u32::MAX,
        ),
        (
            "negative_infinity",
            vec![f32::NEG_INFINITY, 2.0],
            2,
            u32::MAX,
        ),
        ("empty", vec![1.0], 0, u32::MAX),
    ] {
        let logits = ctx.upload(&logits)?;
        ctx.fill(fault, 255, 4)?;
        ctx.call(
            kernels.argmax,
            1,
            [
                Arg::ptr(logits),
                Arg::ptr(summary),
                Arg::u32(n),
                Arg::ptr(fault),
                Arg::u32(0x122),
            ],
        )?;
        ctx.download(summary, &mut words)?;
        let expected_fault = if expected == u32::MAX {
            0x122
        } else {
            u32::MAX
        };
        if words != [expected, expected_fault] {
            return Err(format!(
                "core_argmax_{name}: summary={words:?}, expected={:?}",
                [expected, expected_fault]
            ));
        }
        fault_is(ctx, fault, expected_fault, &format!("core_argmax_{name}"))?;
    }
    // Finite operands overflowing the residual must fault, not just nonfinite input operands.
    let large = ctx.upload(&[f32::MAX])?;
    ctx.fill(fault, 255, 4)?;
    ctx.call(
        kernels.add,
        1,
        [
            Arg::ptr(large),
            Arg::ptr(large),
            Arg::u32(1),
            Arg::ptr(fault),
            Arg::u32(0x123),
        ],
    )?;
    fault_is(ctx, fault, 0x123, "core_add_finite_overflow")?;
    ctx.download(large, &mut got[..1])?;
    if got[0].is_finite() {
        return Err("core_add_finite_overflow: plausible output".into());
    }
    println!(
        "{{\"operation\":\"core_fault_and_argmax\",\"atomic_min\":true,\"tie_smallest\":true,\"nonfinite_refused\":true,\"invalid_formats_refused\":true}}"
    );
    Ok(())
}

pub(super) fn run(ctx: &mut Context) -> Result<(), String> {
    let kernels = Kernels {
        embed: ctx.function("bloomery_hip_embed")?,
        gemv: ctx.function("bloomery_hip_gemv")?,
        norm: ctx.function("bloomery_hip_norm")?,
        glu: ctx.function("bloomery_hip_glu")?,
        add: ctx.function("bloomery_hip_add")?,
        argmax: ctx.function("bloomery_hip_argmax")?,
    };
    let fault = ctx.upload(&[u32::MAX])?;
    packed(ctx, &kernels, fault)?;
    elementwise(ctx, &kernels, fault)?;
    refusals(ctx, &kernels, fault)
}
