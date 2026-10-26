//! Muse Spark's 256-expert top-8 sigmoid router (`sigmoid_256_8`),
//! model-free: the same planted case as Flint's gate (`gate_flint_router`)
//! at the other pick count, plus the bias clause that pins the selection to
//! the biased scores and the weights to the unbiased ones — Muse's
//! `exp_probs_b` contract.
//!
//! One token. The weight's column 0 holds the logits and the other columns
//! are 0; the activation is 1 at column 0, so each logit is one product.
//! Bias is 0 and the scale is 1 except in the bias clause. The logits fall
//! strictly through the first sixteen experts and sit at −20 after that, so
//! the eighth and the ninth do not tie.
//!
//! The slot count checked here is 8, not the instance's `N_USED`. The ids
//! equal the host top 8 of the kernel's scores plus bias (larger value, and
//! a tie to the larger index) and the host sigmoid ranking of the planted
//! logits. The weights are bit-identical to the host renorm of those scores
//! (f64 slot sum, narrowed, `renorm_divisor`, times the scale) and differ
//! from the first 8 weights of a top-9 renorm. With a nonzero bias (−30 on
//! expert 0, +30 on expert 20) the ids follow the biased scores while the
//! weights stay the renorm of the unbiased ones: expert 20 enters at the top
//! carrying sigmoid(−20), not its 30-biased score. A NaN logit raises
//! `Router`; a clean launch raises nothing.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_muse_router", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::route_body::sigmoid_256_8;
    use bloomery_gpu::route_core::{renorm_divisor, sigmoid};
    use bloomery_gpu::{DeviceTensor, Fault, FaultSite, Gpu, LAYER_NONE};
    use bloomery_gpu_gates::{GateError, bits_equal, checks_failed, verdict};
    use cuda_core::DeviceBuffer;

    /// Experts, the pick this gate names, and the activation width. The pick
    /// is not read from the instance.
    const EXPERTS: usize = 256;
    const USED: usize = 8;
    const TOP9: usize = 9;
    const K: usize = 32;
    const SCALE: f32 = 1.0;

    fn logit(expert: usize) -> f32 {
        if expert < 16 {
            20.0 - expert as f32
        } else {
            -20.0
        }
    }

    fn weight_matrix() -> Vec<f32> {
        let mut w = vec![0.0f32; EXPERTS * K];
        for e in 0..EXPERTS {
            w[e * K] = logit(e);
        }
        w
    }

    /// The top `k` of `key`, larger value first, a tie to the larger index.
    fn top_k(key: &[f32], k: usize) -> Vec<u32> {
        let mut idx: Vec<u32> = (0..key.len() as u32).collect();
        idx.sort_by(|&a, &b| key[b as usize].total_cmp(&key[a as usize]).then(b.cmp(&a)));
        idx.truncate(k);
        idx
    }

    /// Unbiased scores at `ids`, summed in slot order in f64, narrowed, guarded, scaled.
    fn weights_of(scores: &[f32], ids: &[u32], scale: f32) -> Vec<f32> {
        let g: Vec<f32> = ids.iter().map(|&i| scores[i as usize]).collect();
        let div = renorm_divisor(g.iter().fold(0.0f64, |s, &v| s + f64::from(v)) as f32);
        g.into_iter().map(|v| v / div * scale).collect()
    }

    fn clause(name: &str, pass: bool, detail: &str) -> bool {
        println!("{name} {detail} {}", verdict(pass));
        pass
    }

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let gpu = Gpu::new()?;
        println!("gate_muse_router: device {}", gpu.device_name()?);
        let s = gpu.stream();
        let rk = sigmoid_256_8::RouterKernels::load(gpu.context())?;
        let planted: Vec<f32> = (0..EXPERTS).map(logit).collect();
        let by_logit = top_k(
            &planted.iter().copied().map(sigmoid).collect::<Vec<_>>(),
            USED,
        );
        let w_host = weight_matrix();
        let x_host = {
            let mut x = vec![0.0f32; K];
            x[0] = 1.0;
            x
        };
        let bias_host = vec![0.0f32; EXPERTS];
        let w = DeviceTensor::upload(s, &w_host, EXPERTS, K)?;
        let x = DeviceBuffer::from_host(s, &x_host)?;
        let bias = DeviceBuffer::from_host(s, &bias_host)?;
        let mut out = sigmoid_256_8::RouterOut::new(s)?;
        gpu.take_fault()?;
        rk.enqueue_router(s, &w, &x, &bias, SCALE, &mut out, gpu.unlabelled_sink())?;
        s.synchronize()?;
        let scores = out.probs.to_host_vec(s)?;
        let ids = out.ids.to_host_vec(s)?;
        let weights = out.weights.to_host_vec(s)?;
        let clean = gpu.take_fault()?;

        let by_score = top_k(
            &scores
                .iter()
                .zip(&bias_host)
                .map(|(&p, &b)| p + b)
                .collect::<Vec<_>>(),
            USED,
        );
        let len_ok = ids.len() == USED && weights.len() == USED;
        let ids_ok = len_ok && ids == by_score && ids == by_logit;
        let expect = weights_of(&scores, &by_score, SCALE);
        let w_ok = len_ok && bits_equal(&weights, &expect);
        let by9 = top_k(
            &scores
                .iter()
                .zip(&bias_host)
                .map(|(&p, &b)| p + b)
                .collect::<Vec<_>>(),
            TOP9,
        );
        let w9 = weights_of(&scores, &by9, SCALE);
        let differ =
            weights.len() == USED && w9.len() == TOP9 && !bits_equal(&weights, &w9[..USED]);

        let mut ok = clause(
            "router ids top 8",
            ids_ok && clean.is_none(),
            &format!(
                "len={} ids={ids:?} by_score={by_score:?} by_logit={by_logit:?} fault={clean:?}",
                ids.len()
            ),
        );
        ok &= clause(
            "router weights renorm of the top 8",
            w_ok && clean.is_none(),
            &format!("bit_identical={w_ok} slots={}", weights.len()),
        );
        ok &= clause(
            "router weights differ from a top-9 renorm",
            differ && clean.is_none(),
            &format!("differ={differ}"),
        );

        // The bias steers the selection only: expert 20 enters at the top by
        // its biased score while its weight stays its unbiased sigmoid(−20),
        // and expert 0 drops out. A biased-score renorm would give the top
        // slot a weight near 30/37 instead.
        let mut biased = vec![0.0f32; EXPERTS];
        biased[0] = -30.0;
        biased[20] = 30.0;
        let mut bias_out = sigmoid_256_8::RouterOut::new(s)?;
        gpu.take_fault()?;
        let bias_dev = DeviceBuffer::from_host(s, &biased)?;
        rk.enqueue_router(
            s,
            &w,
            &x,
            &bias_dev,
            SCALE,
            &mut bias_out,
            gpu.unlabelled_sink(),
        )?;
        s.synchronize()?;
        let b_scores = bias_out.probs.to_host_vec(s)?;
        let b_ids = bias_out.ids.to_host_vec(s)?;
        let b_weights = bias_out.weights.to_host_vec(s)?;
        let b_clean = gpu.take_fault()?;
        let b_by_score = top_k(
            &b_scores
                .iter()
                .zip(&biased)
                .map(|(&p, &b)| p + b)
                .collect::<Vec<_>>(),
            USED,
        );
        let b_ids_ok = b_ids == b_by_score && b_by_score.first() == Some(&20);
        let b_expect = weights_of(&b_scores, &b_by_score, SCALE);
        let b_w_ok = b_weights.len() == USED && bits_equal(&b_weights, &b_expect);
        ok &= clause(
            "router bias steers the selection only",
            b_ids_ok && b_w_ok && b_clean.is_none(),
            &format!(
                "ids={b_ids:?} by_score={b_by_score:?} top_weight={:e} fault={b_clean:?}",
                b_weights[0]
            ),
        );

        let mut nan_w = w_host;
        nan_w[40 * K] = f32::NAN;
        let nan_w = DeviceTensor::upload(s, &nan_w, EXPERTS, K)?;
        let mut nan_out = sigmoid_256_8::RouterOut::new(s)?;
        gpu.take_fault()?;
        rk.enqueue_router(
            s,
            &nan_w,
            &x,
            &bias,
            SCALE,
            &mut nan_out,
            gpu.unlabelled_sink(),
        )?;
        s.synchronize()?;
        let nan_fault = gpu.take_fault()?;
        let want = Some(Fault::at(LAYER_NONE, FaultSite::Router));
        ok &= clause(
            "router NaN logit",
            nan_fault == want,
            &format!("fault={nan_fault:?} want={want:?}"),
        );
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
