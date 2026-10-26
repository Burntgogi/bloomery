# MiniMax-M3.1-Flash: the port map (issue #698)

Snapshot: worktree `bloomery-mm31design` at the lead's integration head. This is the **design
round**: one file, no code, no box time, no measured claim. MiniMax-M3.1-Flash stays unsupported
until the phase gates below pass.

**Premise correction.** The brief said MiniMax-M3 is a model side only: no card body, no sparse
attention kernel, no refset family, no oracle dump. That is behind the tree. The family row is
`tools/families.tsv:20` (`minimax-m3`, F). The model side is `crates/model/src/arch/minimaxm3/`.
The card body is `crates/gpu-minimaxm3/` (MSA kernels in `src/attn.rs`). The app session is
`crates/app/src/arch/minimaxm3/`. The serve seat is
`crates/gpu-gates/src/bin/shared/serve_seats/minimaxm3.rs`. The refset family is
`crates/refset/src/arch/minimaxm3/mod.rs`: real-file dumps of
`/models/MiniMax-M3-UD-IQ4_XS/...-00001-of-00006.gguf` from mainline llama.cpp
`cd0a5b6a1+libllama-d846809bb9cd`. What that family does not have is a fixture tier
(`mod.rs:6`: "No fixture file exists yet"). What M3 never reached, and what M3.1 Flash adds, is
named per row in §4.

Sources, all already on disk. Weights were not opened.

- Configs and indexes: `~/.cache/bloomery/mm31/` — `config.json` (M3.1 Flash, rev
  `76faee1b4ac0f412e547781172e94548a6526a88`), `m3-config.json`, `idx.txt` (74 name patterns),
  `model.safetensors.index.json` (89,632 tensors, `metadata.total_size` 250,103,411,200),
  `dspark_config.json`, `dspark_model.safetensors.index.json` (97 tensors, `total_size`
  2,301,852,674), `docs_README-minimax-m3.1-flash.md`. The index maps a name to a shard only.
  It has no shapes. Parameter products below are the config times the dims llama.cpp creates
  (`src/models/minimax-m3.cpp:55-87`), checked against `total_size`.
- Transformers v5.12.1 modeling: `~/.cache/bloomery/mm31/modeling_minimax_m3_vl.py`.
  `configuration_minimax_m3_vl.py` is not in that cache, so the function that turns
  `sparse_attention_freq` / `moe_layer_freq` into `layer_types` / `mlp_layer_types` was not
  read. The JSON arrays and the index name classification are the evidence.
- llama.cpp, `/Users/hckim/repo/llama.cpp-upstream` (tree dated 2026-10-07 in the brief):
  graph `src/models/minimax-m3.cpp`, keys `src/llama-arch.cpp`, KV filter `src/llama-model.cpp`,
  converter `conversion/minimax.py` and `conversion/base.py`. `convert_hf_to_gguf.py` no longer
  holds the class. `sliding_window_pattern` is not in `src/llama-hparams.h`; it is
  `llama-arch.cpp:280`, loaded by `llama_model_base::load_swa_pattern` (`llama-model.cpp:3464`).
- Cost model: `~/repo/rig-log/docs/bloomery/plan.md` 「모델」 and 「흐름 모델」, `facts.md`.
  Card bytes: `crates/placement/src/placement/workstation.rs`.

## 1. File facts (every new byte figure is [derived])

`config.json`: 60 layers, hidden 6144, GQA 64/4, head 128, vocab 200,064, context 1,048,576,
`rope_theta` 10,000,000, `partial_rotary_factor` 0.5 (rotated dims 64), `rms_norm_eps` 1e-6,
`hidden_act` `swigluoai` (`swiglu_alpha` 1.702, `swiglu_limit` 7.0), `use_qk_norm` true,
`scoring_func` `sigmoid`, `use_routing_bias` true, `routed_scaling_factor` 2.0, 128 experts,
top 4, expert intermediate 3072, shared intermediate 3072, dense intermediate 12,288.
`num_mtp_modules` is absent (M3's config has `num_mtp_modules` 7 at `m3-config.json:97`).
`moe_layer_freq` is three 0s then 57 ones (`config.json:34`), the same prefix as M3
(`m3-config.json:34`). `sparse_attention_config.sparse_attention_freq` and
`sparse_disable_index_value` are 1 on all 60 layers (`config.json:109`, `:171`). M3's
`sparse_attention_freq` is 0 on layers 0–2 (`m3-config.json:173`). MSA geometry is unchanged:
4 index heads, dim 128, top-16 blocks, block 128, local 1 (`config.json:102-108`).

The index name classification agrees with those arrays, which is the check the missing
configuration class cannot give. Text layers 0–59 are present. `mlp.*` tensors exist on
layers {0, 1, 2} only. Routed-expert tensors exist on layers 3–59 (57 layers). Indexer
tensors (`index_q_proj`, `index_k_proj`, both norms) exist on all 60 layers. No layer has
both `mlp.*` and routed experts. No `mlp` layer lacks an indexer. No MoE layer lacks an
indexer. So layers 0–2 are MiniMax Sparse Attention over a dense `swigluoai` FFN. M3's
reader cannot say that: `Kind` is one axis (`hparams.rs:24-31`).

Quant, from `quantization_config` (`config.json:282-356`) and the name patterns in `idx.txt`.
Non-expert matrices that carry `weight_scale_inv` are MXFP8, block `[1, 32]`: attention
q/k/v/o, indexer q/k projections, the three dense-FFN matrices, the three shared-expert
matrices. Routed `w1`/`w2`/`w3` are NVFP4 packed (`weight_packed`, `weight_scale`,
`weight_global_scale`, `input_global_scale`, `group_size` 16). No scale sidecar, so stored
full width: every norm (including indexer q/k norms), `e_score_correction_bias`, every MoE
`gate`, `embed_tokens`, `lm_head`. `ignored_layers` also names `multi_modal_projector`,
`patch_merge_mlp`, `vision_tower`. Vision `image_size` is 672 (M3 is 2016,
`m3-config.json:246`); the tower is CLIP-shaped, hidden 1280, 16 heads, 32 layers,
intermediate 5120 (`config.json:239-244`).

**Parameter products** [derived: the products above]. One layer's attention projections
106,954,752; with the four norms (12,544) the attention block is 106,967,296. Indexer
projections 3,932,160, plus two norms of 128. Dense FFN 226,492,416 (3 × 6144 × 12288).
One expert triple 56,623,104 (3 × 6144 × 3072); both 6144 and 3072 divide 256, so an IQ4_XS
block is whole. ×128 experts = 7,247,757,312 per MoE layer; ×57 = 413,122,166,784 routed
parameters. Shared expert per MoE layer equals one expert triple. Router per MoE layer
786,560 (6144×128 plus a bias of 128). Text parameters, embed and `lm_head` both counted:
426,186,370,176. Active routed parameters (4 of 128): 12,910,067,712. The brief's active
set (those, plus the shared expert on 57 layers, plus the dense FFN on 3 layers):
16,817,061,888. Adding attention, indexer projections, norms, the router and the head:
24,745,077,888.

**Checkpoint bytes** [derived]. MXFP8 counted as 1 + 4/32 bytes per value (fp8 code, f32
scale per 32) on every `weight_scale_inv` matrix: 10,560,208,896 values → 11,880,235,008 B.
NVFP4 counted as 0.5 + 1/16 (4-bit code, one byte scale per 16) on the routed parameters,
plus 57×128×3×2×4 B of f32 global scales: 232,381,393,920 B. Norms, router, bias, embed and
`lm_head` as bf16: 5,007,988,992 B. Text sum 249,269,617,920 B. `metadata.total_size` is
250,103,411,200. The residual is 833,793,280 B (0.834 GB). That residual is the vision
tower, the projector and the patch-merge MLP: those names are in the index and in
`ignored_layers`, and they are the tensors the text sum leaves out. The index has no shapes,
so the split inside the 0.834 GB is not forced. An f32 NVFP4 scale (4 bytes per 16) adds
77.46 GB and overshoots `total_size`; that scale width is ruled out. Storing the small text
tensors as f32 instead of bf16 moves the residual to 0.743 GB and does not identify it.

**Shipped text GGUF** [derived], at the types in §6. Routed experts IQ4_XS (136/256).
MXFP8 projections (attention, indexer q/k, dense FFN, shared expert) Q8_0 (34/32). Norms,
router and bias F32, Gemma +1 baked into the norm weights at conversion
(`conversion/minimax.py:200-203`). Embed and `lm_head` BF16.

| Class | Bytes | Notes |
|---|---:|---|
| Routed experts, 57 × 128 triples | 219,471,151,104 = 219.471 GB | 3,850,371,072 B per MoE layer (3.850 GB) |
| Q8_0 projections | 11,220,221,952 = 11.220 GB | 10,560,208,896 values × 34/32 |
| F32 norms, router, bias | 182,432,256 | 774,144 norm values + 44,833,920 router values |
| BF16 embed + `lm_head` | 4,916,772,864 = 4.917 GB | untied; one embed row is 12,288 B |
| **Text tensor payload** | **235,790,578,176 = 235.791 GB** | GGUF alignment and metadata are not in this figure |

A full bf16 text file is 852.373 GB; bf16 experts alone are 826.244 GB. Neither fits the
drives in §6.

**Active bytes of one decode token** [derived, same types]. Routed 4×57 = 6,858,473,472 B
(6.858 GB). Shared expert ×57 at Q8_0 = 3,429,236,736 B. Dense FFN ×3 = 721,944,576 B.
Attention + indexer projections ×60 = 7,069,040,640 B. Norms + router = 182,432,256 B.
Head = 2,458,386,432 B. One embed row = 12,288 B. Always-on (everything except the routed
experts and the embed row) = 13,861,040,640 B (13.861 GB). Always-on plus routed =
20,719,514,112 B (20.720 GB). The brief's phrase (routed + shared + dense FFN only) =
11,009,654,784 B (11.010 GB).

**KV at ctx 32,768, one sequence, f16** [derived]. Per position per layer, K+V =
4×128×2×2 = 2,048 B. Index key = 128×2 = 256 B on every MSA layer. The M3 cache test
(`place_hybrid.rs:805-835`) pins 60×2048×32768 = 4,026,531,840 B of K/V and 57×256×32768
= 478,150,656 B of index keys. M3.1 adds index keys on layers 0–2: +3×256×32768 =
25,165,824 B. Index total 503,316,480 B. KV + index = 4,529,848,320 B (4.530 GB). One
shared rope table of 64 f32 (256 B/position) and one shared sink row of 128 f16
(256 B/position) stay as that test counts them; the selector bit decides who reads the
sink, and the table is still one.

## 2. What is the same and what moved

Same as the M3 body already runs, and the M3.1 config does not change the geometry:

- GQA 64/4, head 128, per-head q/k RMS, partial NeoX over 64 (`modeling_minimax_m3_vl.py:414-431`;
  our kernels assert it, `attn.rs:87`).
- `swigluoai` at alpha 1.702 and limit 7.0, the constants `hparams.rs:19-22`.
- Sigmoid noaux router, bias steers selection only, weights renormalized, scale 2.0.
  The converter writes `expert_weights_norm` true and `expert_weights_scale`
  (`conversion/minimax.py:180-182`). The shared expert is "ungated" in the sense of no extra
  sigmoid on the router weight; the gate projection is still there (`ffn.rs:7-8`).
- MSA geometry 4×128, block 128, top 16, local 1, no index value
  (`sparse_disable_index_value` all 1s).
- Gemma RMS: the weight is stored as the raw `w` and applied as `1+w`. The converter bakes
  `+1` (`conversion/minimax.py:200-203`). Our kernels run plain RMS on the stored weight
  (`attn.rs` module note: weights already include Gemma's +1; the graph comment is
  `minimax-m3.cpp:348`).
- MTP. `refuse_mtp` (`hparams.rs:270`) already refuses a non-zero `nextn_predict_layers`, a
  `nextn.*` tensor, or a block past `block_count`. M3.1's absence of MTP is that contract.
  No new MTP work.
- `rope.freq_base` is a file key (`hparams.rs:73-74`). The synthetic tests use 5e6; `real()`
  sets dims 64 and leaves the test file's base at 5e6 (`hparams.rs` around the real-file
  overwrite). M3.1 writes 1e7 into the same key.

What moved, each tied to a file:

1. Attention kind and FFN kind come apart. M3 couples them: layers `i < leading_dense_block_count`
   are full attention and a dense FFN; the rest are MSA and routed experts
   (`hparams.rs:204-209`, `minimax-m3.cpp:65-87` and `:334`). M3.1 keeps
   `leading_dense_block_count` 3 and puts an indexer on layers 0–2.
2. `rope_theta` 1e7 against M3's 5e6 (`m3-config.json:21`). A different value of a key we
   already read.
3. No MTP modules. Already refused.
4. The chat template. Vendor README points at FlagOS model cards for the template, multimodal
   inputs, DSpark and `reasoning_effort`. No M3.1 `chat_template.jinja` or `tokenizer_config.json`
   is in `~/.cache/bloomery/mm31/`. The M3 template's generation tags are `<mm:think>` /
   `</mm:think>` (`models/templates/MiniMax-M3.jinja:7-8`), which `spec.rs:28-30` hardcodes as
   `ReasoningFormat::ThinkSpan`. The M3 system fallback names the model "MiniMax-M3" and the
   cutoff "January 2026" (`MiniMax-M3.jinja:69`). The M3.1 delta against those bytes is
   **[unverified]** until the template file is in the cache. Do not invent the strings.
5. Vision `image_size` 672, and a separate DSpark draft (§9). Both are later phases.

**Two block-score rules, and the oracle is llama.cpp.** The graph keeps per-head top-k:
unscaled `mul_mat`, `pool_2d` MAX over the position axis with kernel `(blk, 1)` so heads stay,
local bias added, `ggml_top_k` per head, then a per-head gather (`minimax-m3.cpp:388-447`).
Transformers max-reduces over the keys in the block **and** over the index heads, then one
shared top-k of shape `[B, S_q, topk]`, with local blocks forced to `+inf`
(`modeling_minimax_m3_vl.py:579-593`). Our card kernel matches llama.cpp and `runtime::msa`:
per index head, unscaled scores, local forced by `runtime::msa::FORCE`, a tie to the lower
block (`attn.rs:18-21`). A dump compared with the transformers forward will disagree on which
blocks a head keeps. The oracle patch does not change our kernel to the pooled rule.

MSA also falls back. If flash attention is off, or the KV cache is unified with
`n_seq_max > 1`, the graph disables MSA and runs dense attention, with a warning that the
output is degraded (`minimax-m3.cpp:233-250`). Oracle dumps have to be taken with flash
attention on and either one sequence or a non-unified cache. Our body has no such fallback;
a comparison against the degraded path is not an MSA comparison.

## 3. The numerical contract (one M3.1 layer)

The order is the graph's. Layers 0–2 take the dense-FFN arm at `minimax-m3.cpp:550` and, once
the pattern bit is set, the sparse arm at `:334`. Layers 3–59 are the M3 sparse layer the
body already runs.

1. Pre-norm RMS on the residual (`attn_norm`, eps 1e-6, weight already `1+w`).
2. QKV projections. Per-head q and k RMS, then partial NeoX on the first 64 values, base the
   file's `rope.freq_base` (1e7 for this model).
3. Full attention (`build_attn`, scale `1/√128`) when the layer's attention kind is full.
   When it is MSA, the indexer chain in §8, then flash over the gathered blocks. K and V are
   still written for every layer.
4. Output projection, plain residual.
5. Post-attention RMS (`ffn_norm`).
6. Dense layer: `swigluoai` gate·up·down at intermediate 12,288. MoE layer: sigmoid noaux
   router (bias on the choice, unbiased scores renormalized, scale 2.0), top 4 of 128
   `swigluoai` experts at intermediate 3072, plus the shared expert at the same width.
7. Head: output RMS, then `lm_head`. No logit softcap in this graph.

## 4. The common-owner table

Every row of `docs/contrib/new-model.md` ("the common-owner table") plus the concepts the
round names. **Attaches** names the work. A row left for a later phase names the cut. "Later"
is not a value by itself.

M3 already covers the rows marked **already attached**. M3 never reached: the dense-FFN × MSA
combination (rows Reader, MSA, dense FFN, as a pair), the MXFP8/NVFP4 streaming convert
(GGUF route, §6), this file's rope base as a value rather than a key, the M3.1 chat-template
bytes, vision, and MiniMax's DSpark. The fixture tier is still absent.

| Row | Decision | Owner (`path:line`) | What M3.1 Flash adds | Size |
|---|---|---|---|---|
| Reader | attaches | `arch/minimaxm3/{hparams,roles,spec,names,host}.rs`; `Kind` at `hparams.rs:26` | two axes per layer (§5). No new stem (`names.rs:4-9`) | M |
| Tokenizer | already attached | `tokenizer.ggml.pre` `minimax-m2` (`spec.rs` test at `:137`); GPT-2 BPE | vocab size matches 200,064. A different pre-tokenizer is **[unverified]** until a tokenizer file is read | — |
| Chat template | attaches, cut on the bytes | `spec.rs:28-30` `ThinkSpan`; template bytes `MiniMax-M3.jinja:7-8` and `:69` | pin the M3.1 jinja when it is on disk. Until then the delta is **[unverified]** and the M3 strings stay | S |
| Reply parser | does not attach | `spec.rs:6-7`: the chat surface carries no tool-call parser | nothing in the cache describes a new parser. Adding one from a README pointer would be invented copy | — |
| GQA 64/4, head 128, q/k norm, partial NeoX 64 | already attached | `spec.rs:69-82` (`qk_norm` true, `RopeMode::Neox`, `window` none); kernels `attn.rs:57-65` | `rope.freq_base` 1e7, the existing key | — |
| MSA (indexer 4×128, block 128, top 16, local 1, KV side) | already attached for the kernel; attaches for layers 0–2 | `gpu-minimaxm3/src/attn.rs:1-21`; rule `runtime/src/msa.rs:182`; index bytes `place_hybrid.rs:543` | the same chain on a layer whose FFN is dense. Geometry is const-asserted (`attn.rs:87`); a file with another MSA width does not attach | S |
| Dense `swigluoai` FFN | already attached on the lead layers | `shared_routed::dense` (`ffn.rs:16`, `shared_routed.rs:851`) | runs on an MSA layer, which today's `Kind::Dense` forbids | S |
| Sigmoid noaux router, bias, scale 2.0 | already attached | `noaux::Experts` (`hparams.rs:78`); card router `sigmoid_128_4` (`ffn.rs:12`) | the same 128/4 instance. Do not add a MiniMax arm to the router | — |
| `swigluoai` experts + shared expert | already attached | host `host.rs:37` `build_oai`; card `shared_routed` (`ffn.rs:1-8`); alpha/limit `kquant/act.rs:50-52` | expert count and width unchanged. The type we ship is §6 | — |
| Gemma norm | already attached | bake `conversion/minimax.py:200-203`; plain RMS on the baked weight | the M3.1 convert bakes `+1` the same way | — |
| Kernel table / card leg | already attached for IQ4_XS | `card_gate_up_runs` / `card_q8act_down_runs` (`kernels.rs:203-216`, `:277-278`); `MiniMaxM3Body` uses `common_leg_gate_up` / `common_leg_down` (`accepts.rs:465-467`). `common_leg_gate_up(NVFP4)` is `None` (`accepts.rs:793`) | ship IQ4_XS (§6). Native NVFP4 is a common-leg gap, not a fourth leg | — |
| Host tier | already attached for IQ4_XS | qdot IQ4_XS × Q8_K (`qdot/src/lib.rs:29`, `:8955`). NVFP4 `host_act` is `None` (`qtype.rs:139`) | the IQ4_XS `(a, c)` fit at 6144×3072 is **[unverified]** (§7) | — |
| Residency | does not attach | `plan_host_routed` (`placement.rs:2293`) keeps every routed stack on the host and refuses an expert-tier card by name (`HostRoutedTier`) | M3.1 does not change that. Hot-expert residency is a separate owner (`TierBody::residency_parts`); this file's experts are the cold bulk | — |
| Expert tier | does not attach in the prediction | `body.rs:12` refuses a plan of other than one card with no expert tier; `one_card_run` at `:462` | the 3090's spare bytes are a phase-(c) question only if that refusal is lifted by the common tier glue, not by a MiniMax planner | — |
| KV cache | already attached | `place_hybrid.rs:407` keys index bytes on the selector bit. The M3 test hardcodes 57 selecting layers (`:806`) | M3.1 sets the bit on layers 0–2. +25,165,824 B at ctx 32,768. The layout function already counts a selector; the test comment is the M3 file | S |
| Prompt batch | already attached | `crates/app/src/arch/minimaxm3/`; walk `crates/runtime/src/prompt.rs` | the dense-FFN MSA layer uses the dense arm inside the same walk. No per-model loop | S |
| Resident slots | already attached | the seat parks more than the one sequence the app session runs | no new slot rule | — |
| Serve seat | already attached | `serve_seats/minimaxm3.rs` (named by `families.tsv:20`); `bloomery_serve` model word `minimaxm3` | phase (c) opens it on the M3.1 file. Same seat module; the file's pattern is data | S |
| Timing | does not attach a new driver | lease runners in `tools/ref/` | a same-lease A/B is phase (c), through the seat. This round times nothing | — |
| Placement plan | already attached | `plan_host_routed` / `plan_routed` (`placement.rs:2293`, `:2240`); card bytes `workstation.rs:71-85` | §7 is the paper split. No model-owned planner | S |
| DSpark draft | attaches in phase (e), a new family | shape `runtime/src/dspark.rs:13-37` can record the geometry; the body `model/src/arch/dspark.rs:1` is V4.1 (`dflash`, hyper-connections, MLA, MXFP4). `speculative.rs:670` already accepts `DraftSpec::Block` | a new reader that fills `DsparkShape`, and a new draft body. Not a flag on the V4.1 owner | L for the body, S for the shape |
| Vision | attaches in phase (d), a new encoder | converter `MiniMaxM3VisionModel` (`conversion/minimax.py:208-281`) keeps `vision_tower` / `multi_modal_projector` / `patch_merge_mlp`, projector `MINIMAXM3`, GELU, spatial merge 2, and asserts ViT `d_head == 80` (`:276`). 1280/16 = 80, so the assert holds. No MiniMax vision owner under `crates/gpu-vision/` | `image_size` 672. Weights are unquantized (`ignored_layers`). Tensor inventory inside the 0.834 GB residual is **[unverified]** (no shapes in the index) | L |

## 5. The reader change

> The reader landed with the key `minimax-m3.attention.sparse_layers` (a bool array), not `attention.indexer.types`:
> llama.cpp spells that key for a different fact (whether a DSA layer computes its own indexer, `is_indexer_full_impl`).

Decouple attention kind from FFN kind. FFN kind stays `leading_dense_block_count`: layer `i`
is a dense FFN iff `i < dense_lead` (3 on both released files). Attention kind is a per-layer
array.

**The key.** Reuse `LLM_KV_ATTENTION_INDEXER_TYPES`, spelled `%s.attention.indexer.types`
(`llama-arch.cpp:302`). Under architecture `minimax-m3` the key is
`minimax-m3.attention.indexer.types`, a uint32 array of length `block_count`. Value 1 means
the layer has the four indexer tensors and runs MSA. Value 0 means full attention and no
indexer tensors. The architecture prefix makes this a different key from
`glm-dsa.attention.indexer.types`. GLM already loads it with `get_key_or_arr(..., required
= false)` after filling a default (`glm-dsa.cpp:59-66`). That is the load shape.

`sliding_window_pattern` (`llama-arch.cpp:280`, `load_swa_pattern` at `llama-model.cpp:3464`)
is the other precedent: present array wins, absent array falls back. It is the wrong meaning
(SWA versus full). Prefer the indexer-types key. A new
`minimax-m3.attention.indexer.pattern` is the fallback only if reusing the GLM key's meaning
("this layer has an indexer") is rejected; the arch prefix already keeps the bytes apart.

**Absent key.** Reconstruct today's coupling: layer `i` is MSA iff `i >= dense_lead`. An M3
file, which has no such key, reads as it does now. Record the default the way an absent
`leading_dense_block_count` is recorded (`hparams.rs:159-164`).

**Present key.** Attention kind is the array. FFN kind stays `dense_lead`. M3.1 writes 60
ones and `leading_dense_block_count` 3, so layers 0–2 carry indexer tensors and
`ffn_gate` / `ffn_up` / `ffn_down`, and layers 3–59 carry indexer tensors and the routed
block. A 0 on a layer past `dense_lead`, or a 1 whose indexer tensors are missing, is a
named error. The array length must equal `block_count`.

| File:line | Today | After |
|---|---|---|
| `hparams.rs:24-31` `Kind` | `Dense` = full attention + dense FFN; `Sparse` = indexer + routed experts | `AttnKind::{Full, Msa}` × `FfnKind::{Dense, Moe}`. `kind()` (`:204-209`) splits into the two |
| `hparams.rs:97-118` `READ` | no per-layer attention array | add `attention.indexer.types`, optional |
| `hparams.rs:159-172` | `dense_lead` from `leading_dense_block_count`, absent → 0 | unchanged, and it still decides `FfnKind` only |
| `hparams.rs:221-264` `tensors_agree` / `layer_dims` | one kind's stems; indexer xor dense FFN | indexer stems follow `AttnKind`; dense or routed stems follow `FfnKind`. A layer may carry both |
| `hparams.rs:270` `refuse_mtp` | refuses MTP by name | unchanged |
| `roles.rs:34-38` `stems_of` | `Dense` → attention + dense FFN; `Sparse` → attention + routed + indexer | attention always; indexer iff MSA; dense FFN or routed by `FfnKind` |
| `roles.rs:95` | `blk.2.indexer.q_proj.weight` has no role | still none on an M3 file (no key, layer 2 is full). A header with the key and `pattern[2]=1` accepts it. The M3 assertion stays |
| `spec.rs:54-97` | `Selector::BlockMax` and `Ffn::Moe` share `Kind::Sparse` | the selector follows `AttnKind`; `Ffn::Dense` or `Ffn::Moe` follows `FfnKind`. `Residual::Plain`, `qk_norm`, `out_gate` none, `window` none, `mtp` empty stay |
| `names.rs:4-9` | re-exports the indexer stems | no new stem |
| `host.rs:29` | refuses `Kind::Dense` for a routed stack | refuses `FfnKind::Dense`. A dense-FFN MSA layer still has no host routed stack |
| `gpu-minimaxm3/src/body.rs:476` | `store(..., hp.kind(l) == Kind::Sparse)` | the indexer store follows `AttnKind`. The dense FFN arm (`shared_routed::dense`) follows `FfnKind` |
| `place_hybrid.rs:805-814` | the test builds 57 selecting layers and pins 478,150,656 index bytes at 32,768 | that assertion is the M3 file and stays. M3.1 is a second case: 60 selectors, 503,316,480 index bytes. The comment stops saying "57" as if it were the architecture |

`place.rs:91` asserts `kind(2) == Sparse` and `kind(1) == Dense` on the synthetic header,
which has no pattern key. It stays a statement about that header.

FAIL-first for the split is a synthetic header with `leading_dense_block_count` 3 and
`attention.indexer.types` all ones, carrying `blk.0.indexer.q_proj.weight` and
`blk.0.ffn_gate.weight`. Today's `tensors_agree` refuses the indexer on layer 0. After the
change it reads, and an M3 header with the key omitted still classifies layer 0 as full
attention plus a dense FFN.

## 6. The GGUF and the oracle

> Superseded: bloomery reads the checkpoint folder in its stored formats, with no GGUF and no requantization, and the
> oracle is transformers on the folder. See [safetensors-native.md](safetensors-native.md).

**What the stock converter does.** `MiniMaxM3Model.set_gguf_parameters`
(`conversion/minimax.py:177-198`) counts `leading_dense_block_count` as the prefix of
`moe_layer_freq` that is 0. On this config that is still 3. It writes the indexer scalars
from `sparse_attention_config` and does not write a per-layer array. `tensor_force_quant`
forces any name containing `.indexer.` to F32 (`:172-174`). `modify_tensors` bakes Gemma +1
into every `norm.weight` (`:200-203`) and merges `w1`/`w2`/`w3` only when the unquantized
`.weight` tensors exist (`:140-159`).

It does not dequant this checkpoint. `prepare_tensors` reads `quant_algo`, `quant_method`
and `format` (`base.py:960-965`). This file has `quant_method` `"mxfp8"`, `moe_quant_algo`
`"NVFP4"` and `moe_quant_format` `"nvfp4-pack-quantized"`. It has no `quant_algo`, no
`format`, no `config_groups`, no `quantized_layers`. `_is_nvfp4` becomes true only for
`quant_algo in {NVFP4, W4A16_NVFP4}`, or `quant_method == "compressed-tensors"` with an
NVFP4 format, or a `quantized_layers` entry whose `quant_algo` ends in `NVFP4`
(`base.py:985-1000`). None of those fire. `dequant_model` then raises
`NotImplementedError` on `quant_method` `"mxfp8"` (`base.py:579-580`). The fp8 dequant path
reads `weight_scale_inv` only for `quant_method == "fp8"`.

Even a forced NVFP4 path would miss these experts. `_generate_nvfp4_tensors` matches
`experts.<id>.(gate_proj|up_proj|down_proj).weight` (`base.py:896`) and flushes them as
`model.layers.<bid>.mlp.experts.<proj>.weight` (`:944`). MiniMax names the experts `w1`,
`w2`, `w3` under `block_sparse_moe`. ggml's NVFP4, our `GgmlType::NVFP4`, is a superblock
of 64 values, 36 bytes (`qtype.rs:137-140`). The checkpoint packs groups of 16.

**Our types against that layout** (`qtype.rs`).

| Type | Block | Host matmul | Card, the common leg |
|---|---|---|---|
| `NVFP4` | 64 values, 36 B | none (`host_act: None`) | planes leg only: `card_planes_runs` (`kernels.rs:287-289`), down activation `Q8Blocks32`. `common_leg_gate_up(NVFP4)` is `None` (`accepts.rs:793`) |
| `NVFP4_ROWS` | 16 values, 9 B, ModelOpt group-16 | none, and `dequant_row` does not decode it | `Card::NONE` (`qtype.rs:162-167`) |
| `MXFP8` | 32 values, 33 B, bloomery-own | none | dense `Mxfp8` only; no routed, no gate·up (`qtype.rs:160-161`) |
| `IQ4_XS` | 256 values, 136 B (0.53125 B/value) | Q8_K (`qdot/src/lib.rs:29`) | `card_gate_up_runs` and `card_q8act_down_runs` both admit it |
| `Q4_K` | 256 values, 144 B (0.5625) | Q8_2X4 | both common-leg arms |
| `MXFP4` | 32 values, 17 B (0.53125) | Q8_2X4 | both common-leg arms |
| `Q8_0` | 32 values, 34 B (1.0625) | Q8_2X4 | dense and the shared/dense block (`block.rs:41`). Not in `card_gate_up_runs` (the comment at `kernels.rs:196-200` leaves it out on purpose) |

**Ship routed experts as IQ4_XS.** The common leg and the host dot both take it, and the
live M3 file the gates already run is UD-IQ4_XS (`refset/.../minimaxm3/mod.rs:17-19`).
Q4_K also takes both, at 144/256 instead of 136/256, and would add about 12.9 GB to the
expert payload [derived: 413,122,166,784 × (144−136)/256]. MXFP4 is the same density and is
admitted, and it is not the type the family's dumps are pinned to. Native ggml NVFP4 hits
two gaps at once: no host matmul, and the MiniMax body does not use the planes leg. Closing
that is a common lift (a host dot, and teaching the common leg a `Q8Blocks32` down or calling
the planes leg), not a MiniMax copy of the leg. `NVFP4_ROWS` matches the group-16 packing
and has no card and no row dequant.

Requant is lossy against the checkpoint's NVFP4. There is no imatrix until one is built.
Say so on the file.

**Ship the MXFP8 projections as Q8_0**, not as our MXFP8 and not as the converter's F32
indexer force. Q8_0 is what the released M3 projections use (`body.rs:275-277` `PROJ`, and
`INDEX` at `:285` which also allows F32 and the K-quants). The Q8_0 shared/dense pair runs
SwiGLU-OAI: `ds41_shexp_gate_up` takes `act <= 2` and calls `apply_checked`
(`experts.rs:184`, `:216`), and `Act::code` accepts alpha 1.702 (`kquant/act.rs:77-82`).
The sentence at `shared_routed.rs:33-34` that says a Q8_0 SwiGLU-OAI expert is refused is
stale against that kernel. ggml's oracle can score Q8_0; it has no MXFP8. Norms and the
router stay F32, with Gemma +1 baked. Embed and `lm_head` stay BF16: the embed is a host
row (`body.rs:10`) and BF16 dequant is on (`qtype.rs:128`); the head has a BF16 entry
(`head.rs:4`).

Text tensor payload at those types: **235,790,578,176 B (235.791 GB)**. Experts are
219,471,151,104 B of that.

**Disk, each step.** The stage drive holds the 250.103 GB download at `/home/user/stage/mm31`
and has about 80 GB free after it. `/models` has about 150 GB free after the shipped file.
A bf16 copy of the experts (826 GB) or of the text (852 GB) fits on neither.

1. Leave the 48 shards where they land. Do not stage a second copy.
2. Stream. Dequant one tensor to f32, requant, append to the GGUF on `/models`, drop the
   f32. Peak scratch is one expert triple in f32, 226,492,416 B (226.5 MB), or one MoE
   layer of experts in f32 if the writer stacks a layer before flushing,
   28,991,029,248 B (28.991 GB). Both sit inside the 80 GB. The largest MXFP8 matrix in f32
   is a dense gate/up/down, 301,989,888 B.
3. Write the text GGUF (235.791 GB of tensors) to `/models`. The stage disk never holds it.
   Vision stays out of this file; its checkpoint residual is 0.834 GB and is phase (d).
4. The oracle is a local llama.cpp patch (§6, below) run against that GGUF. Phase (a) does
   not store full-model activation dumps. Phase (b)'s dumps are of the small fixture.

**The oracle patch is local.** llama.cpp bans agent-written PRs. The patch teaches the
loader and the graph the per-layer bit. Our streaming convert produces the GGUF; the stock
converter is not the path.

1. No new key enum. Call `get_key_or_arr(LLM_KV_ATTENTION_INDEXER_TYPES, ..., required = false)`
   from the MiniMax loader. The spelling is already `llama-arch.cpp:302`.
2. `minimax-m3.cpp` `load_arch_hparams` (`:15`, where `n_layer_dense_lead` is read): default
   `is_indexer_full_impl[i] = (i >= n_layer_dense_lead)`, then the optional array. Absent key
   → today's M3 behavior.
3. `load_arch_tensors` (`:65-88`): split the `if`. Dense FFN tensors still when
   `i < n_layer_dense_lead`. Indexer tensors when the layer bit is set, including
   `i < n_layer_dense_lead`. Routed and shared tensors stay in the else of the dense-lead test.
4. Graph `is_sparse` (`:334`): `msa_enabled && pattern[il]`, not `il >= n_layer_dense_lead`.
5. Graph FFN branch (`:550`): stays on `n_layer_dense_lead`. No change.
6. `llama-model.cpp:2393-2397`: the KV index filter is `il >= n_layer_dense_lead`. It has to
   become the per-layer bit, or layers 0–2 get no index-key cache and the gather reads an
   empty plane.
7. Do not patch `conversion/minimax.py` for quant. `base.py:580` and the NVFP4 gate are why
   the convert is ours. If a later round does write the array from the stock converter, it
   writes `sparse_attention_freq`. Our reader accepts both the omission (M3) and the all-ones
   array (M3.1).

## 7. Placement on the maintainer's box

Paper only. Cards from `workstation.rs` [the file's own measured tag]: A6000 total
49,140 MiB, driver reserve 548 MiB, usable 50,952,404,992 B (50.952 GB). RTX 3090 total
24,576 MiB, driver reserve 400 MiB, usable 25,350,373,376 B (25.350 GB). Per card the plan
also holds `CONTEXT` 512 MiB + `SCRATCH` 64 MiB + `MARGIN` 1 GiB = 1,677,721,600 B
(`workstation.rs:104-112`). Room after that overhead: A6000 49.275 GB, 3090 23.673 GB,
both 72.947 GB.

Host: `HOST_USABLE` 270,071,001,088 B (`workstation.rs:121`), which is 251.52 GiB. That is
the brief's "251 GB". `OS_OTHER` is 6,694,629,376 B. `PLAN_ROOM` = usable − OS =
263,376,371,712 B (263.376 GB), the room `HostNeed::check` holds a plan to. `facts.md`:
Threadripper PRO 5975WX, Zen 3, AVX2, no AVX-512; STREAM read 147.7 GB/s; the fitted host
DRAM the plan uses is 135–137 GB/s (this note uses 137, as muse-spark-attach.md §5 did).
A6000 H2D pinned 26.28 GB/s. Both cards PCIe 4.0 ×16, no P2P.

Non-routed text bytes (Q8_0 projections + F32 small + BF16 embed and head) = 16,319,427,072 B
(16.319 GB). Plus KV+index at 32,768 = 4,529,848,320 B. Together 20,849,275,392 B (20.849 GB).
That fits on the A6000 alone after overhead, with about 28.4 GB spare [derived:
49.275 − 20.849]. It also fits on the 3090, with about 2.8 GB spare, which is too tight to
be the stage card once scratch and a capture move. Routed experts at 219.471 GB do not fit
in 72.9 GB of card room.

`plan_host_routed` (`placement.rs:2293`) puts every routed stack on the host and refuses an
expert-tier card by name. The MiniMax body agrees (`body.rs:12`). The conservative paper
split, the one the prediction uses:

- Stage card = A6000. It holds the non-routed weights, the KV and the index keys, the head.
  The embed row is a host lookup copied in (`body.rs:10`).
- Routed experts, 219.471 GB, stay on the host. `PLAN_ROOM` after them is 43,905,220,608 B
  (43.905 GB) [derived: 263,376,371,712 − 219,471,151,104]. Decode activations are a few
  rows of 6144. Under that constant the expert set fits and the NVMe tier stays empty.
  The planner reads `host_room` on the day, not this constant. If that reading is under
  219.471 GB, the overflow is the NVMe tier. This note does not claim a fresh
  `MemAvailable`.
- The 3090 is idle in this split. Putting experts on it means lifting `body.rs:12` through
  the common expert-tier glue. Spare A6000 bytes (~28.4 GB, about 7 MoE layers at 3.850 GB)
  are the same kind of lever, and `plan_routed` (`placement.rs:2240`) already fills spare
  card VRAM with experts the leg runs. Neither lever is in the tok/s below.

**Decode, one token** [derived]. Inputs: active routed bytes 6,858,473,472; host 137 GB/s
(STREAM 147.7 as the read ceiling); card large-launch bandwidth 567–701 GB/s (`plan.md`
「모델」, A6000); host union ceiling 1.31 TMAC/s (`plan.md` 「흐름 모델」); bridge
36.0 + 119.5·k_host µs for k_host 2..6, protocol floor 21–36 µs, not extrapolated to a
large k.

| Resource | Work | Time |
|---|---|---|
| Host DRAM | read 6.858 GB of IQ4_XS experts | 50.06 ms at 137 GB/s; 46.44 ms at 147.7 |
| Host MAC | 12.910 GMAC (4×57 triples) at 1.31 TMAC/s | 9.86 ms |
| Card SMs | read 13.861 GB always-on at 567–701 GB/s | 24.45–19.77 ms |
| Card, KV depth 6 | 60×(2048+256)×6 = 829,440 B | 0.001 ms at 567 GB/s |
| Card, KV depth 4096 | 60×2304×4096 = 566,231,040 B | 1.00–0.81 ms |
| PCIe | activations only: 57 layers × 2 × 6144 × 4 = 2,801,664 B at 26.28 GB/s | 0.11 ms, plus the protocol floor |

The host is bandwidth-bound: 50 ms against 9.86 ms of MAC. The card's 20–24 ms runs in the
host's shadow once the shared expert and the card layers overlap the host experts, which is
the shadow `shared_routed` already schedules (`shared_routed.rs:15-20`). Wall = max(host,
card) = the host read. Launch tax is `N_node × c_node` with `c_node` 0.852 µs
(`plan.md`). This round did not count the M3 graph's nodes. A thousand nodes is under 1 ms
and does not move a 50 ms wall.

**Predicted decode: 20.0 tok/s at 137 GB/s, 21.5 tok/s at the STREAM ceiling 147.7 GB/s**
[derived: 1000/50.06 and 1000/46.44]. The bound is host DRAM on the active expert set.
Depth 6 and depth 4096 are the same prediction: the extra KV read is about 1 ms on the
card, under the host term. The IQ4_XS rate that would make the MAC term the bound is not
this number; the MAC term is already under the read. Moving experts onto spare VRAM would
shrink the host bytes and can move the bound. The body today will not do that, so the
prediction keeps every routed expert on the host.

**Prefill at P = 512** [derived]. A prompt wakes experts by the union
`t(m) = max(W, a + c·m)` (`plan.md` 「흐름 모델」). One IQ4_XS expert is 30,081,024 B, so
W = 219.6 µs at 137 GB/s. A MoE layer's 128 experts, each read once, are 3.850 GB and
28.10 ms of DRAM; ×57 layers = 1.602 s. That is the floor once every expert is touched.

Uniform routing gives m̄ = 512×4/128 = 16 tokens per expert. MAC per token per expert =
56,623,104. At 1.31 TMAC/s that is 43.2 µs per token, so 16 tokens are 692 µs against W of
220 µs: past a handful of tokens the expert is MAC-bound, not DRAM-bound. Per MoE layer,
128×16×56,623,104 / 1.31e12 = 88.5 ms. ×57 = 5.05 s. Card weight reads of the 13.861 GB
always-on set, once, are the same 20–24 ms as decode and sit under that. 512/5.05 s ≈
101 tok/s if the 1.31 TMAC/s ceiling holds and attention does not climb out of the host's
shadow.

The `(a, c)` pair in `plan.md` was fitted on a different quant and a different expert
shape. It is not reused here. The 1.31 TMAC/s figure is the ceiling that pair was checked
against, and applying it to IQ4_XS at 6144×3072 is **[unverified]**. If the IQ4_XS kernel
misses that rate, P = 512 slows in proportion. The prediction is about **100 tok/s**,
host-MAC-bound, at that unverified rate. It is not a lease number.

## 8. MSA on our card

The kernel family already exists. M3.1 does not add a width, a block, or a top-k. It runs
the same chain on layers 0–2, whose FFN is the dense block.

The graph's decode steps (`minimax-m3.cpp:343-447`) and the owner each one already has:

1. Indexer q and k projections from the normed residual. Owner: the body's gemv sites
   (`body.rs` `INDEX` at `:285`). Shape: q is 6144×512 (4×128), k is 6144×128. One k head,
   not one k per query head.
2. Per-head RMS on the index q (weight `indexer.q_norm`, length 128) and on the index k,
   then partial NeoX over the first 64, then append the k to the layer's f16 index-key
   plane. Owner: `rope_neox::head_norm_neox_append_128_partial` via `enqueue_index`
   (`attn.rs:6-10`). The head kernel also appends a value row; the index has none, so that
   row goes to a plane nothing reads. `sparse_disable_index_value` is 1 on every layer.
3. Block scores: unscaled dots, the block's score is the max over its visible positions,
   per index head. Owner: `msa_kernels::msa_score` (`attn.rs:11-12`). Not `qsa_kernels`
   (`crates/gpu/src/qsa.rs:1-9`), which scores pooled tokens of 4 for Qwen3.8 / GLM-5.3.
4. Top-k with the local block forced. Owner: `msa_kernels::msa_select` (`attn.rs:13-14`),
   rule `runtime::msa` (`msa.rs:182`: top_k 16, block 128, local 1; `FORCE` on the local
   blocks; tie to the lower block). Not `qsa_topk` and not the DSpark selector
   (`crates/gpu/src/dspark/selector.rs`).
5. Gather K and V for the blocks that head kept, then attention at scale `1/√128` over
   group 16, head 128. Owner: `flash_gqa::enqueue_pass_p16_blocks` (`attn.rs:15-16`). The
   head-128 path is the one this body already launches.

Transformers inserts a head reduction at step 3 (`modeling_minimax_m3_vl.py:580`) and then
one top-k shared by every head. We do not. The oracle comparison uses the per-head rule.

What M3.1 adds is the call on a layer that also runs `shared_routed::dense`. The store flag
at `body.rs:476` is the line that has to follow `AttnKind` instead of `Kind::Sparse`.

## 9. The DSpark draft

`dspark_config.json`: architecture `DSparkMiniMaxDraftModel`, `model_type` `minimax_m3_vl`.
Five layers, every one `sliding_attention` with window 4096 (`:113-120`). Dense FFN of width
12,288 (`dspark_use_dense_ffn` true, `dspark_ffn_hidden_size` 12288). `sparse_attention_config`
is null. Block size 7 (`:76`). Mask token and noise token 200,058 (`:80`, `:93`). Target
layers `[3, 17, 31, 45, 59]` (`:83-89`). Markov head, rank 256, type `vanilla`, aux gemm
bf16 (`:77-79`). `enable_confidence_head` and `confidence_head_with_markov` true (`:122-123`).
`hybrid_context_fusion` false. Hidden 6144, the target's GQA and rope, vocab 200,064.
`num_nextn_predict_layers` 0.

The draft index (`dspark_model.safetensors.index.json`) holds 97 tensors and
`total_size` 2,301,852,674 B. The names collapse to: `confidence_head.proj`, `fc.weight`,
`final_norm`, `hidden_norm`, five decoder layers (input and post norms, mlp gate/up/down
plus `weight_scale_inv`, self-attn q/k/v/o plus `weight_scale_inv`, q_norm, k_norm), and
`markov_head.markov_w1` / `markov_w2`. No routed experts, no indexer. The modeling file has
no DSpark class. The draft json's quant block lists MXFP8 and NVFP4 fields and ignores the
confidence, fc and Markov heads; the index itself shows dense-FFN MXFP8 names and no NVFP4
names. Row 0's drafting convention (`DsparkShape.row0_drafts`) is not in the config.
**[unverified]** until the modeling class or a graph says which row emits an id.

`DsparkShape` (`runtime/src/dspark.rs:13-37`) can record this geometry: block 7, taps those
five layers, hidden 6144, vocab 200,064, mask id 200,058, window 4096,
`HeadKind::Markov { rank: 256, confidence: true }`. `speculative.rs:670` already builds
`Program::Block` from `DraftSpec::Block`. The card body that runs a `DsparkShape` today is
the V4.1 draft (`model/src/arch/dspark.rs:1-8`: hyper-connections, MLA, MXFP4 routed
experts). A GQA layer with a dense `swigluoai` and a sliding window does not run there.
Phase (e) is a new reader that fills `DsparkShape`, plus a new draft body. It is not a
branch on the V4.1 reader. Size L for the body, S for recording the shape. It is last.

## 10. Phases

Each phase is one reviewable PR with the gate named here.

**(a) Reader split, tokenizer, template, our GGUF, the oracle patch.** The change table in
§5, the streaming convert in §6, and the local llama.cpp patch as a list of edits a
maintainer applies (not a PR to llama.cpp). Tokenizer: keep `minimax-m2` unless a tokenizer
file says otherwise. Template: pin the M3.1 jinja when its bytes are in the cache; until
then record the delta against `MiniMax-M3.jinja:69` as unverified and do not invent a
string. Gate: the existing hparams, roles and spec tests, plus the synthetic header of §5
(dense lead 3, indexer pattern all ones, indexer and dense FFN both on layer 0). That
header is refused today. M3 headers with the key absent stay on the old classification.
No card, no box.

**(b) Text body on a fixture, against the patched llama.cpp.** A small requantized GGUF:
few layers, few experts, and at least one dense-FFN MSA layer and one MoE MSA layer, at the
shipped types (IQ4_XS experts, Q8_0 projections). The reference dumps are a fixture tier on
`crates/refset/src/arch/minimaxm3/`, which has none today (`mod.rs:6`). The oracle is the
patched graph with flash attention on and a single sequence. Dumps the M3 sets already
teach the shape of: per-layer outputs, router ids, block scores and the selection (prompt E
is the one that drops a block), top logits, greedy tokens. The new set has to cover layer 0
as MSA-plus-dense, which no current set can, because no current file has that layer.
Gate: the fixture family check, and `gate-gpu-minimaxm3-e2e` on the fixture. FAIL-first is
the fixture header refused by the pre-split reader, then the body store refusing an indexer
on a dense-FFN layer.

**(c) Placement and serving on the real file.** The 235.791 GB text GGUF from §6, the split
in §7, the existing seat. Gate: the serve seat opens the file, one sequence, ctx 32,768,
and the plan's bytes match §7's paper (experts on the host, non-routed and KV on the A6000).
A same-lease decode A/B is how a later change to this path is judged; this phase's first
number is the one the lease prints, and §7's 20 tok/s is the prediction it is compared
with, not a pin. The 3090-as-expert-tier question is a separate PR inside this phase only
if the common tier glue is the change; it is not required to serve.

**(d) Vision.** The encoder `conversion/minimax.py:208-281` already describes, at
`image_size` 672, weights unquantized. New owner under the vision crate. Gate: a fixture
image through the projector against the patched llama.cpp mmproj, once that side exists.
The 0.834 GB residual is the byte budget until shapes are read.

**(e) DSpark, last.** New draft family filling `DsparkShape`. Gate: a block of 7 against
the vendor forward once the modeling class is in tree, or against a local oracle graph.
Not a flag on `model/src/arch/dspark.rs`.

## 11. Risks, and the lift instead of a branch

- A `Kind` match that grows a third variant "MSA plus dense FFN" inside `shared_routed` or
  `sigmoid_128_4` is the wrong cut. Those owners already run a dense block and a routed
  block. The new meaning is which layer is which, and it belongs in this model's spec.
- Shipping NVFP4 by copying GLM's planes leg into `gpu-minimaxm3` is a fourth expert leg.
  The lift, if a later round wants the native type, is a host qdot for NVFP4 and a common
  leg that can launch a `Q8Blocks32` down (or call `card_planes_runs`) for every family
  that already uses `common_leg_*`. `accepts.rs:793` is the line that says the common leg
  does not.
- `plan_host_routed` should not grow a MiniMax exception so the 3090 can hold experts.
  The body refusal at `body.rs:12` is the same decision. The lift is the existing
  expert-tier glue, and only when phase (c) asks for it. The prediction in §7 does not
  assume it.
- The converter's F32 force on `.indexer.` (`conversion/minimax.py:172-174`) is not our
  contract. `INDEX` already accepts Q8_0 (`body.rs:285`). Forcing F32 would add
  693,043,200 B (0.693 GB) over Q8_0 [derived: 3,932,160 × 60 × (4 − 34/32)] and would not match
  the M3 file the gates run.
- The transformers head-pooled top-k is a real numerical disagreement (§2). Papering over
  it inside `msa_select` would move every M3 dump. The oracle stays llama.cpp.
- `place.rs:94-96` still expects `PlanInputs::read` to report `swigluoai` as
  `Unimplemented`, while a card body now exists. That is a coverage-gate reconciliation,
  not an M3.1 feature. Left for the reader PR to look at, not to weaken.
- `shared_routed.rs:33-34` says a Q8_0 SwiGLU-OAI shared expert is refused at the first
  launch. `experts.rs:184` and `kquant/act.rs:77` accept that rule. The comment is stale.
  Not edited here.
- MSA geometry is a const assertion (`attn.rs:87`). M3.1 matches it. A checkpoint with a
  different index width, block or head count does not attach to this kernel; it needs a
  new instance. Nothing in this config asks for one.
