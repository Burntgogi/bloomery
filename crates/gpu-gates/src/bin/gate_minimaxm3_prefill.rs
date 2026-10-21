//! The MiniMax-M3 prefill gate: the prompt batch (`bloomery_gpu_minimaxm3::prefill`: up to `T_MAX` rows a unit, every
//! layer over the whole unit, the index keys of the unit's rows before any row is scored, the block-max selection over
//! each row's own live count, the routed block over the unit's slots) is the one-token step run per id to within the
//! family's own band, in the logits row and in every layer's output, and the stores it leaves are the ones the next step
//! reads; the prompt group changes no bit.
//!
//! The batch keeps its own arithmetic (the wide GEMM's sum order, where the step's sites read one column a gemv), so it
//! is the steps' within a band, not bit for bit. The band is the family's own: the forced arm's per-layer error over the
//! layers its prefix covers ([`seat::prefill_ceil`], the error model of the end-to-end gate's forced arm), and no ceiling
//! over the layers from that prefix on: a top-4 near tie can flip between the batch's GEMM sum order and the step's gemv,
//! and a flip is a discrete exchange of experts, not a rounding of the activations.
//!
//! The prompt lengths straddle the unit: `P = 1` (a unit of one row, the step's shape), `7` (rows short of any alignment),
//! `T_MAX` (one whole unit) and `T_MAX + 1` (a whole unit and a unit of one row: the K and V planes and the index keys
//! cross the cut between two units). The ids are the fixed walk of `e2e::walk`, none of the oracle's: the clauses compare
//! the engine's two paths to each other and need no reference.
//!
//! The clauses (all self-consistency) are the common ones ([`prefill_clauses`]): `prefill_near_steps`,
//! `prefill_then_step_near_steps`, `prefill_group_is_group_one` and `fault_word_clean`, and the family's own,
//! `prefill_batch_forms`, which runs first and on no file: the batch's wide pools quantize the input of every projection
//! type the loader accepts (each type alone at both widths, then a mix), so a file whose projections are K-quants
//! prefills as the released file's Q8_0 ones do.
//!
//! The family has no fixture file, so the fixture tier is refused by name before anything loads; the real tier's file is
//! the one the end-to-end gate reads.

#[cfg(feature = "minimaxm3")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_minimaxm3_prefill", gate::run())
}

#[cfg(feature = "minimaxm3")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "minimaxm3")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "minimaxm3")]
#[allow(
    dead_code,
    reason = "the engine serves several gates; each takes the calls it reads"
)]
#[path = "shared/lcpp_e2e.rs"]
mod lcpp_e2e;

#[cfg(feature = "minimaxm3")]
#[allow(
    dead_code,
    reason = "the prefill clauses serve several gates; each takes the helpers they read"
)]
#[path = "shared/prefill_clauses.rs"]
mod prefill_clauses;

#[cfg(feature = "minimaxm3")]
#[allow(
    dead_code,
    reason = "the seat serves two gates; each takes the calls it reads"
)]
#[path = "shared/minimaxm3_seat.rs"]
mod seat;

#[cfg(feature = "minimaxm3")]
mod gate {
    use crate::lcpp_e2e::{arm_arg, model_path};
    use crate::prefill_clauses::Case;
    use crate::seat;
    use bloomery_gpu::Gpu;
    use bloomery_gpu::arch::hybrid::Site;
    use bloomery_gpu::flash_gqa::HEAD;
    use bloomery_gpu::gemm::Gemm32Kernels;
    use bloomery_gpu::site::{SiteTy, WideActs, WideKernels};
    use bloomery_gpu_gates::tier::{self, Tag, Tier};
    use bloomery_gpu_gates::{GateError, checks_failed, verdict};
    use bloomery_gpu_minimaxm3::attn::MsaShape;
    use bloomery_gpu_minimaxm3::body::{Dims, INDEX, PROJ};
    use bloomery_gpu_minimaxm3::prefill::{Batch, T_MAX};
    use cuda_core::DeviceBuffer;
    use refset::arch::minimaxm3::ARCH;

    const NAME: &str = "gate_minimaxm3_prefill";

    /// The recipe the file is the one of.
    const RECIPE: &str = "gate-gpu-minimaxm3-prefill";

    /// The prompt lengths: one row, rows short of an alignment, one whole unit, a unit and a row.
    const LENGTHS: [usize; 4] = [1, 7, T_MAX, T_MAX + 1];

    /// Positions of the cache beyond the longest prompt and the step after it.
    const CTX_SPARE: usize = 8;

    /// The synthetic batch's widths: the model width, and the output projection's (the query heads of [`HEAD`] values).
    /// Both are whole blocks of the q8_1 form's 128 values and of the q8 form's 64-value step.
    const EMBD: usize = 3072;
    const QUERY_HEADS: usize = 64;

    /// The clause `prefill_batch_forms`: whether a batch cut for the sites it is told of quantizes the input of each of
    /// them. Each type the loader accepts is run alone at both widths (the projections' input at the model width, the
    /// output projection's at the query width), then a mix of the Q8_0 and K-quant types at both. A pool that holds no
    /// form a site's type reads refuses the quantize by name; each refusal is a line and fails the clause.
    fn batch_forms() -> Result<bool, GateError> {
        let gpu = Gpu::new()?;
        let stream = gpu.stream();
        let d = Dims {
            embd: EMBD,
            vocab: 1,
            heads: QUERY_HEADS,
            kv_heads: 4,
            rms_eps: 1.0e-6,
            rope_base: 5.0e6,
            ctx: 256,
        };
        let q_width = QUERY_HEADS * HEAD;
        let shape = MsaShape::new(16, 1)?;
        let g32 = Gemm32Kernels::load(gpu.context())?;
        let x = DeviceBuffer::<f32>::zeroed(stream, T_MAX * q_width)?;
        let accepted = INDEX
            .iter()
            .chain(PROJ.iter().filter(|t| !INDEX.contains(t)))
            .map(|&t| (format!("{t:?}"), vec![t], vec![t]));
        let mixed = (
            "mixed".to_string(),
            vec![SiteTy::Q8_0, SiteTy::Q4K, SiteTy::F32],
            vec![SiteTy::Q6K, SiteTy::Q8_0, SiteTy::Q5K],
        );
        let mut ok = true;
        for (label, at_embd, at_query) in accepted.chain([mixed]) {
            let sites: Vec<Site> = at_embd
                .iter()
                .map(|&t| Site::new(format!("{label}.embd"), t, EMBD))
                .chain(
                    at_query
                        .iter()
                        .map(|&t| Site::new(format!("{label}.query"), t, q_width)),
                )
                .collect();
            let refs: Vec<&Site> = sites.iter().collect();
            let mut batch = Batch::new(&gpu, (&d, &shape), T_MAX, &refs)?;
            let wk = WideKernels {
                gemm: &batch.gemm,
                g32: Some(&g32),
            };
            let mut quantize = |pool: &mut WideActs, tys: &[SiteTy]| -> Result<String, GateError> {
                Ok(
                    match pool.quantize(&gpu, &wk, gpu.layer_sink(0)?, &x, T_MAX, tys) {
                        Ok(()) => "ok".to_string(),
                        Err(e) => {
                            ok = false;
                            e.to_string()
                        }
                    },
                )
            };
            let model = quantize(&mut batch.wide, &at_embd)?;
            let output = quantize(&mut batch.arm, &at_query)?;
            println!(
                "prefill_batch_forms: {label}: the model width's pool {model}; the output width's pool {output}"
            );
        }
        stream.synchronize()?;
        println!(
            "prefill_batch_forms: every type the loader accepts, alone and mixed {}",
            verdict(ok)
        );
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let tier_now = Tier::from_env()?;
        if tier_now == Tier::Fixture {
            return Err(format!(
                "{NAME}: the family has no fixture file: the gate runs in the real tier (`just {RECIPE}`)"
            )
            .into());
        }
        // Before the file loads: a pool the batch was cut wrong for fails in seconds, not after the load.
        if tier::run_clause("prefill_batch_forms", Tag::SelfConsistency)? && !batch_forms()? {
            println!("clause prefill_batch_forms FAIL");
            println!("{}", tier::tally_line());
            return Err(checks_failed());
        }
        let path = model_path(tier_now, RECIPE, ARCH)?;
        let ctx = T_MAX + 1 + CTX_SPARE;
        let t = std::time::Instant::now();
        let arm = arm_arg()?;
        let mut seat = seat::open(&path, ctx, arm)?;
        let dims = seat.dims();
        println!(
            "load arm={} layers={} embd={} vocab={} ctx={ctx} t_max={T_MAX} in {:.1} s (runtime value)",
            arm.word(),
            dims.layers,
            dims.embd,
            dims.vocab,
            t.elapsed().as_secs_f64()
        );
        let case = Case {
            lengths: LENGTHS.to_vec(),
            tap_ceil: seat::prefill_ceil,
            tap_ceils: seat::prefill_ceils,
            group: 2,
            // A top-4 near tie can flip between the batch's GEMM sum order and the step's gemv, and
            // a flip is a discrete exchange of experts: the comparison is deferred, as the
            // end-to-end gate defers its top-logprobs clause.
            logit_bound: None,
            vocab: dims.vocab,
            embd: dims.embd,
            tier: tier_now,
        };
        crate::prefill_clauses::run(NAME, &case, seat.as_mut())
    }
}
