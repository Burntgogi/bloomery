//! Contributor-owned raw-token numerical harness, not a serving engine or backend API.
mod fixture;
mod plan;
mod runner;

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

use crate::Context;
use plan::Plan;
use runner::Model;

const PROBE_BUDGET: usize = 16 << 20;
const USAGE: &str = "--inspect MODEL [CONTEXT]\n  --forward DEVICE_ELF MODEL IDS_FILE OUTPUT_DIR [GREEDY_STEPS [CONTEXT]]\n  --forward-probe DEVICE_ELF OUTPUT_DIR";

fn number<T: std::str::FromStr>(value: &str, name: &str) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("invalid {name}: {value}"))
}

fn ids(text: &str) -> Result<Vec<u32>, String> {
    let values = text
        .lines()
        .map(|line| number(line, "token id"))
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() {
        return Err("token id file is empty".into());
    }
    Ok(values)
}

fn inputs(values: &[u32], steps: usize, plan: &Plan) -> Result<(), String> {
    if values.is_empty() || values.iter().any(|&id| id >= plan.spec.vocab) {
        return Err("prompt ids must be nonempty and inside the vocabulary".into());
    }
    let consumed = values
        .len()
        .checked_add(steps.saturating_sub(1))
        .ok_or("token count overflow")?;
    if consumed > plan.context as usize {
        return Err(format!(
            "prompt plus generated inputs needs {consumed} positions, context is {}",
            plan.context
        ));
    }
    Ok(())
}

fn new_dir(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::create_dir(path).map_err(|e| format!("new evidence directory {}: {e}", path.display()))
}

fn json_file(path: &Path, value: &serde_json::Value) -> Result<(), String> {
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    serde_json::to_writer_pretty(file, value).map_err(|e| e.to_string())
}

fn id_file(path: &Path, values: &[u32]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    for id in values {
        writeln!(file, "{id}").map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[derive(PartialEq)]
struct Run {
    inputs: Vec<u32>,
    predictions: Vec<u32>,
    logit_sha256: Vec<String>,
    greedy: Vec<u32>,
}

fn row(
    model: &mut Model<'_, '_>,
    input: u32,
    logits: &mut [f32],
    out: &Path,
    run: &mut Run,
) -> Result<u32, String> {
    let next = model.step(input)?;
    model.logits(logits)?;
    // SAFETY: f32 has no padding; these initialized values remain live through the write/hash.
    let bytes = unsafe {
        std::slice::from_raw_parts(logits.as_ptr().cast::<u8>(), std::mem::size_of_val(logits))
    };
    let path = out.join(format!("logits-{:04}.f32", run.inputs.len()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    file.write_all(bytes).map_err(|e| e.to_string())?;
    run.inputs.push(input);
    run.predictions.push(next);
    run.logit_sha256.push(gguf::digest::sha256_hex(bytes));
    Ok(next)
}

fn generate(
    model: &mut Model<'_, '_>,
    prompt: &[u32],
    steps: usize,
    vocab: usize,
    out: &Path,
) -> Result<Run, String> {
    new_dir(out)?;
    model.reset()?;
    let rows = prompt.len() + steps.saturating_sub(1);
    let mut run = Run {
        inputs: Vec::with_capacity(rows),
        predictions: Vec::with_capacity(rows),
        logit_sha256: Vec::with_capacity(rows),
        greedy: Vec::with_capacity(steps),
    };
    let mut logits = vec![0.0; vocab];
    let mut next = 0;
    for &input in prompt {
        next = row(model, input, &mut logits, out, &mut run)?;
    }
    for step in 0..steps {
        run.greedy.push(next);
        if step + 1 < steps {
            next = row(model, next, &mut logits, out, &mut run)?;
        }
    }
    Ok(run)
}

fn execute(
    model: &mut Model<'_, '_>,
    plan: &Plan,
    prompt: &[u32],
    steps: usize,
    out: &Path,
) -> Result<Run, String> {
    inputs(prompt, steps, plan)?;
    let first = generate(
        model,
        prompt,
        steps,
        plan.spec.vocab as usize,
        &out.join("run-0"),
    )?;
    let second = generate(
        model,
        prompt,
        steps,
        plan.spec.vocab as usize,
        &out.join("run-1"),
    )?;
    if first != second {
        return Err("reset/repeat changed forced logits, predictions, or greedy ids".into());
    }
    let (free, total) = model.memory()?;
    let manifest = serde_json::json!({"model":plan.identity(),"prompt_ids":prompt,"consumed_ids":first.inputs,
        "predictions":first.predictions,"greedy_ids":first.greedy,"logit_sha256":first.logit_sha256,
        "reset_repeat_runs":2,"reset_repeat_bitwise_equal":true,"all_observed_outputs_finite":true,
        "native_math":"Rust AMDGPU/HIP","cpu_compute_fallback":false,
        "hip_free_bytes_after_run":free,"hip_total_bytes":total,
        "logit_file_format":"little-endian F32; one vocabulary row per consumed id",
        "same_file_llama_HIP_acceptance":"not evaluated by this writer"});
    json_file(&out.join("forward.json"), &manifest)?;
    id_file(&out.join("prompt.ids"), prompt)?;
    id_file(&out.join("forced.ids"), &first.greedy)?;
    println!(
        "{}",
        serde_json::json!({"operation":"forward_hybrid","layers":plan.layers.len(),
        "consumed_tokens":first.inputs.len(),"greedy_ids":first.greedy,"reset_repeat_bitwise_equal":true})
    );
    Ok(first)
}

pub(super) fn run(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("--inspect") if (2..=3).contains(&args.len()) => {
            let context = args.get(2).map_or(Ok(64), |s| number(s, "context"))?;
            let plan = Plan::open(Path::new(&args[1]), context)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&plan.identity()).map_err(|e| e.to_string())?
            );
            Ok(())
        }
        Some("--forward") if (5..=7).contains(&args.len()) => {
            let steps = args.get(5).map_or(Ok(8), |s| number(s, "greedy steps"))?;
            let context = args.get(6).map_or(Ok(64), |s| number(s, "context"))?;
            let plan = Plan::open(Path::new(&args[2]), context)?;
            let prompt = ids(&fs::read_to_string(&args[3]).map_err(|e| e.to_string())?)?;
            inputs(&prompt, steps, &plan)?;
            let mut context = Context::open(&args[1])?;
            let mut model = Model::load(&mut context, &plan)?;
            let out = Path::new(&args[4]);
            new_dir(out)?;
            execute(&mut model, &plan, &prompt, steps, out)?;
            Ok(())
        }
        Some("--forward-probe") if args.len() == 3 => {
            let out = Path::new(&args[2]);
            new_dir(out)?;
            let path = out.join("smoke.gguf");
            fixture::Fixture::new().write(&path)?;
            let plan = Plan::open(&path, 16)?;
            let mut context = Context::open(&args[1])?;
            let (free, _) = context.memory()?;
            plan::fit(PROBE_BUDGET, free)?;
            context.allocation_limit = PROBE_BUDGET;
            crate::forward_core_probe::run(&mut context)?;
            crate::forward_mixer_probe::run(&mut context)?;
            let mut model = Model::load(&mut context, &plan)?;
            runner::probe_contract(&mut model)?;
            execute(&mut model, &plan, &[3, 17, 41, 9, 23], 8, out)?;
            println!(
                "{}",
                serde_json::json!({"operation":"forward_probe_complete","managed_allocation_budget_bytes":PROBE_BUDGET,
                "actual_managed_allocation_bytes":context.allocated_bytes,"real_model_acceptance":false})
            );
            Ok(())
        }
        _ => Err(format!("usage: bloomery-hip-spike {USAGE}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_ids_refuse_empty_negative_fractional_and_overflow() {
        assert_eq!(
            ids("760\n6511\n314\n9338\n369\n").unwrap(),
            vec![760, 6511, 314, 9338, 369]
        );
        for bad in ["", "\n", "1\n\n2", "-1", "2.0", "4294967296"] {
            assert!(ids(bad).is_err());
        }
    }

    #[test]
    fn admission_and_generation_bounds_precede_device_load() {
        let dir = std::env::temp_dir().join(format!(
            "bloomery-hip-plan-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&dir).unwrap();
        let good = dir.join("good.gguf");
        fixture::Fixture::new().write(&good).unwrap();
        let plan = Plan::open(&good, 16).unwrap();
        assert!(inputs(&[0, 257], 15, &plan).is_ok());
        assert!(inputs(&[0, 257], 16, &plan).is_err());
        assert!(inputs(&[258], 0, &plan).is_err());
        assert!(inputs(&[], 8, &plan).is_err());
        assert!(Plan::open(&good, 0).is_err());
        assert!(Plan::open(&good, 65).is_err());
        let bad = dir.join("bad.gguf");
        let mut fixture = fixture::Fixture::new();
        let (_, eps) = fixture
            .kvs
            .iter_mut()
            .find(|(key, _)| key == "qwen35.attention.layer_norm_rms_epsilon")
            .unwrap();
        *eps = gguf::Value::F32(f32::NAN);
        fixture.write(&bad).unwrap();
        assert!(Plan::open(&bad, 16).is_err());
        let bad_axes = dir.join("bad-vector-axes.gguf");
        let mut fixture = fixture::Fixture::new();
        let (tensor, _) = fixture
            .tensors
            .iter_mut()
            .find(|(tensor, _)| tensor.name == model::arch::qwen35moe::names::ssm_a(0))
            .unwrap();
        tensor.dims = vec![1, 2];
        fixture.write(&bad_axes).unwrap();
        assert!(Plan::open(&bad_axes, 16).is_err());
        fs::remove_dir_all(dir).unwrap();
    }
}
