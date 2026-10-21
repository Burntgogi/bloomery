//! raon1 metadata gate: what the header reader `arch::raon1` resolves from the Raon-1-mini-preview
//! file, what the coverage check lists for it — the parts no program in this tree runs yet, the
//! work queue of the rounds that build one — and the stores its layers hold. One contract, one
//! test a file; each prints what it compared, then fails with the whole list of what differs.
//!
//! `hw_`: needs the file on the box (`just gate-raon1-meta`). Headers only: seconds.

#[path = "common/spec_view.rs"]
mod spec_view;

use std::collections::BTreeMap;
use std::fmt::Write as _;

use gguf::Split;

/// The Q4_K_M file (KRAFTON's conversion, one shard): Q4_K gate and up expert stacks, Q8_0 and Q5_0
/// down stacks, Q6_K and Q8_0 attention, F16 and F32 for the small mixers' tensors.
const Q4KM: &str = "/models/Raon-1-mini-preview/raon-1-mini-preview-Q4_K_M.gguf";

// PIN(2026-10-21): the file's description as `arch::raon1::spec` reads it, line by line (the key
// values checked against the Q4_K_M header dump, 25 layers; `attention.head_count_kv` is 1 at
// layers 3, 7, 11, 15, 19, 23 and 24 and 0 elsewhere, `leading_dense_block_count` 1). The delta
// layers have 40 heads of 128, the latent layers 40 query heads over a 512-value latent with 64
// unrotated key values (`attention.key_length` 576 = 512 + 64) absorbing 192-wide queries into
// 128-wide values. The chat surface declares the family's parsers: its tool-call markup
// (`ToolFormat::QwenXml`) and its think span.
const VIEW: &[&str] = &[
    "arch Raon1",
    "hidden 2560 vocab 259840 ctx_train 262144",
    "rms_eps bits 0x3727c5ac",
    "layers 25 mtp 0",
    "hc Some(HcSpec { streams: 4, kind: Chart { eps: 1e-5, magnitude: 2.0, collapse: Mean } })",
    "engram None",
    "chat pre raon template bytes 6801 tools Some(QwenXml) reasoning Some(ThinkSpan { open: \"<think>\", close: \"</think>\" })",
    "[0] delta Rda { gate: Sigmoid { lower_bound: -5.0 } } k 40 v 40 d 128 conv 4 || dense 12032 situ 4/25 || hc",
    "[1-2,4-6,8-10,12-14,16-18,20-22] delta Rda { gate: Sigmoid { lower_bound: -5.0 } } k 40 v 40 d 128 conv 4 || moe 128/6 ff 1088 situ 4/25 Sigmoid bias true norm true x1 hash false shared 1088 situ 4/25 gate false || hc",
    "[3,7,11,15,19,23-24] latent h 40 q 576 kv 512 Absorbed { qk: 192, v: 128 } rope - qhn false out Plain win None sinks false pe 64 gate Proj || moe 128/6 ff 1088 situ 4/25 Sigmoid bias true norm true x1 hash false shared 1088 situ 4/25 gate false || hc",
];

// PIN(2026-10-21): the file carries every key the reader reads, so it takes no default.
const DEFAULTS: &[&str] = &[];

// PIN(2026-10-21): the keys the reader does not read: listed, not refused. The file's sampling
// defaults.
const UNREAD: &[&str] = &[
    "general.sampling.temp",
    "general.sampling.top_k",
    "general.sampling.top_p",
];

// PIN(2026-10-21): every tensor of the Q4_K_M file by role and type (731 in all) [derived] from the
// header dump's per-name table: attention 358 = norms, conv taps and delta parameters in f32 147
// (25 `attn_norm`, 18 each of `ssm_a`, `ssm_dt`, `ssm_norm`, 54 conv, 14 latent norms), f16 72 (18
// each of `ssm_beta`, `ssm_f_a`, `ssm_f_b`, `ssm_rda`), q6_K 93 (54 of `attn_q/k/v`, 18 `ssm_g`, 21
// latent), q8_0 46 (25 `attn_output`, 21 latent); the hyper-connections f16 50 and f32 100; the
// dense block q4_K 2 and q6_K 1; the routed stacks q4_K 48 (gate and up), q8_0 12 and q5_0 12
// (half the down stacks each); the shared expert q6_K 48 and q8_0 24.
const TENSORS: &[&str] = &[
    "attn f16 72",
    "attn f32 147",
    "attn q6_K 93",
    "attn q8_0 46",
    "dense_ffn q4_K 2",
    "dense_ffn q6_K 1",
    "ffn_norm f32 25",
    "hc f16 50",
    "hc f32 100",
    "head f32 1",
    "head q6_K 1",
    "routed q4_K 48",
    "routed q5_0 12",
    "routed q8_0 12",
    "router f32 48",
    "shexp q6_K 48",
    "shexp q8_0 24",
    "token_embd q4_K 1",
];

// PIN(2026-10-21): the coverage check's list with no program for the family: the program itself and
// each part of the description or of the file no program's table runs [derived] from the layer
// lines above: the delta layers' rule (erase and write gates apart, the full-width output gate), the
// dense block's and the experts' `situ`, the 128-expert sigmoid router with its bias, the latent
// layers' absorbed 192/128 attention with its 64 unrotated key values and its output gate, the
// four-stream chart mix, and the tensor types no program pins: f16 mixers and mixes (every layer
// holds one), and the routed down stacks in q8_0 (12 layers) and q5_0 (12). The recurrent state and
// the mixed trunk are covered by the Qwen3.5 body's rows; the pre-tokenizer `raon` and the
// tool-call parser by the tokenizer's table and the spec's declared markup.
const COVERAGE: &[&str] = &[
    "delta rule RDA: d 128, 40 heads, conv 4, erase and write gates apart, full-width output gate, sigmoid gate bounded below by -5: 0-2,4-6,8-10,12-14,16-18,20-22",
    "dense situ layer: ff 12032: 0",
    "chart-constrained hyper-connections of 4 streams, collapse Mean: 0-24",
    "router: sigmoid, 128 experts, top 6, with a selection bias: 1-24",
    "routed experts of situ: 1-24",
    "shared expert of situ: 1-24",
    "latent attention 512 x rope 0, absorbed qk 192 / v 128: 3,7,11,15,19,23-24",
    "latent attention with 64 unrotated key values beside the latent, shared by every head: 3,7,11,15,19,23-24",
    "latent attention output gate: sigmoid of its own projection: 3,7,11,15,19,23-24",
    "f16 weights: 0-24",
    "f16 mixer matrices (attention, Mamba-2 or delta rule): 0-2,4-6,8-10,12-14,16-18,20-22",
    "q8_0 routed experts: 1-2,5,8,11,14,17,20-24",
    "q5_0 routed experts: 3-4,6-7,9-10,12-13,15-16,18-19",
    "a layer program for raon1",
];

#[test]
#[ignore = "needs the Raon-1-mini-preview Q4_K_M file on the box"]
fn hw_raon1_spec_q4_k_m() {
    let mut o = String::new();
    let mut b = Vec::new();
    let split = Split::open(Q4KM).unwrap_or_else(|e| panic!("open {Q4KM}: {e}"));
    let read = model::arch::spec(&split).unwrap_or_else(|e| panic!("spec of {Q4KM}: {e}"));
    let _ = writeln!(o, "file {Q4KM}");
    spec_view::compare(
        &mut o,
        &mut b,
        "description",
        &spec_view::view(&read.spec),
        VIEW,
    );
    spec_view::compare(&mut o, &mut b, "defaults taken", &read.defaults, DEFAULTS);
    let unread = model::arch::raon1::hparams::unread_keys(&split);
    spec_view::compare(&mut o, &mut b, "keys not read", &unread, UNREAD);
    let mut by_role: BTreeMap<(String, String), usize> = BTreeMap::new();
    for t in &read.tensors.tensors {
        *by_role
            .entry((t.role.to_string(), t.ty.to_string()))
            .or_default() += 1;
    }
    let got: Vec<String> = by_role
        .iter()
        .map(|((role, ty), n)| format!("{role} {ty} {n}"))
        .collect();
    let _ = writeln!(o, "tensors {}", read.tensors.tensors.len());
    spec_view::compare(&mut o, &mut b, "tensors by role and type", &got, TENSORS);
    let list = model::arch::coverage::check(&read.spec, &read.tensors);
    spec_view::compare(
        &mut o,
        &mut b,
        "coverage",
        &spec_view::items(&list),
        COVERAGE,
    );
    println!("{o}");
    assert!(
        b.is_empty(),
        "{} pin(s) differ:\n  {}",
        b.len(),
        b.join("\n  ")
    );
}

// PIN(2026-10-21): the stores of the file's layers [derived] from the description alone: a delta
// layer 40 heads x 128 x 128 f32 (2,621,440 B) + the conv ring 11 rows x 3 x 40 x 128 f32 (675,840
// B) + one lane's u32 stamp = 3,297,284 B whatever the context, 18 layers; a latent layer a row of
// (512 + 64) f16 values = 1,152 B a position, 7 layers (8,064 B a position, the model's 7.9 KiB a
// token); no rope rows (NoPE). Mutants: the stamp or the pass rows of the ring left out, the
// unrotated values left out of the row, a window on a latent layer.
#[test]
#[ignore = "needs the Raon-1-mini-preview Q4_K_M file on the box"]
fn hw_raon1_stores() {
    use model::arch::raon1::place::PlanInputs;
    use model::placement::KvBytes;

    const CTX: u64 = 32_768;
    const DELTA_LAYERS: u64 = 18;
    const DELTA_STATE: u64 = 2_621_440 + 675_840 + 4;
    const LATENT_POSITION: u64 = 7 * (512 + 64) * 2;

    let mut o = String::new();
    let mut b = Vec::new();
    let split = Split::open(Q4KM).unwrap_or_else(|e| panic!("open {Q4KM}: {e}"));
    let inputs =
        PlanInputs::describe(&split).unwrap_or_else(|e| panic!("plan inputs of {Q4KM}: {e}"));
    let layers = inputs.spec.layers.len();
    let _ = writeln!(o, "file {Q4KM}, {layers} layers, ctx {CTX}");
    row(
        &mut o,
        &mut b,
        "kv bytes a position over the trunk",
        inputs.kv.bytes(0..layers, 1),
        DELTA_LAYERS * DELTA_STATE + LATENT_POSITION,
    );
    row(
        &mut o,
        &mut b,
        "kv bytes at the context",
        inputs.kv.bytes(0..layers, CTX),
        DELTA_LAYERS * DELTA_STATE + LATENT_POSITION * CTX,
    );
    row(
        &mut o,
        &mut b,
        "bytes a position of the latent layers",
        LATENT_POSITION,
        8_064,
    );
    row(
        &mut o,
        &mut b,
        "stage rows bytes",
        inputs.kv.stage_position_bytes(CTX),
        0,
    );
    println!("{o}");
    assert!(
        b.is_empty(),
        "{} pin(s) differ:\n  {}",
        b.len(),
        b.join("\n  ")
    );
}

/// One compared value: a line of the printed table, and an entry of `bad` when it differs.
fn row<T: PartialEq + std::fmt::Debug>(
    out: &mut String,
    bad: &mut Vec<String>,
    what: &str,
    got: T,
    want: T,
) {
    let mark = if got == want { "ok " } else { "BAD" };
    let _ = writeln!(out, "  {mark} {what:<38} {got:?}");
    if got != want {
        bad.push(format!("{what}: got {got:?}, want {want:?}"));
    }
}
