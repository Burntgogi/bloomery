# Muse Spark Open: the port map (issue #684)

Snapshot: worktree `bloomery-musedesign` at `56dbde997` (integration head). This is the **design
round**: one file, no code, no box time, no measured claim. Muse Spark (`muse-spark`) is
unsupported until the phase gates below pass.

Sources, all already on disk:

- The oracle: mainline llama.cpp PR 30616, merge `ba3c05f71da8` (2026-10-15). Whole diff:
  `/Users/hckim/.cache/bloomery/muse/pr30616.diff`. The graph is `src/models/muse-spark.cpp` (new,
  238 lines); hparams in `src/llama-model.cpp`; the chat parser `common/parsers/muse-spark.cpp`;
  the template `models/templates/muse-spark.jinja`; vision reuses Muse Glimmer's encoder; audio is
  `tools/mtmd/models/muse-spark-audio.cpp`. Line cites below are `pr30616.diff:NNN`, the diff's own
  line numbers.
- The specimen: unsloth/Muse-Spark-Open-816B-A42B-GGUF rev `c00670568b6c`, UD-IQ2_XXS (6 shards) +
  `mmproj-Q8_0.gguf`, landing on the box (`muse-dl.sh` moves it to
  `/models/Muse-Spark-Open-816B-A42B-UD-IQ2_XXS`). Header: `~/.cache/bloomery/muse/header.txt`
  (shard 1, metadata); every tensor: `~/.cache/bloomery/muse/tensors.txt` (1,429 rows).

## 1. File facts (every byte figure [derived] from the shard headers by the §9 script)

`general.architecture = muse-spark`; 62 layers, all MoE, no dense layer, no draft/MTP tensors;
hidden 8192; 128 query heads over 16 KV heads, head 64 (group 8); vocab 202,048 (gpt2 BPE, pre
`gpt-4o`, 439,802 merges); context 1,048,576. Parameters 819,991,575,808 ≈ 820 G under the "816B"
name; active ≈ 44.5 G ("A42B") — both [derived].

| Class | Bytes | Notes |
|---|---:|---|
| Routed experts (`ffn_{gate,up,down}_exps`) | 205,957,103,616 = 191.81 GiB | 62 × 3,168 MiB; 256 experts × 3 × 4096×4096, IQ2_XXS (2.0625 bpw) |
| Dense, per-layer × 62 | 17,309,463,552 = 16.12 GiB | 279,184,896 B/layer: q·o Q6_K (55.05 MB each), attn_gate Q8_0 (71.30 MB), k·v Q8_0 (8.91 MB each), latent down·up Q8_0 (35.65 MB each), router F32 (8.39 MB), norms/α/β ≈ 52 KB |
| `output` Q6_K / `token_embd` Q4_K | 1,357,762,560 / 931,037,184 | untied; embd row lookup per token ≈ 4.6 KB |
| **Total encoded** | **225,555,399,680 = 225.56 GB** | matches the file claim; type census F32 807 · Q8_0 310 · IQ2_XXS 186 · Q6_K 125 · Q4_K 1 |

**Window pattern, read from the array, not the shorthand.** `sliding_window_pattern` is
`[T,F,T,T]` repeating: **46 SWA layers** (window 2048) and **16 full layers**, layer 0 sliding,
layer 1 full. A "3:1" rule alone cannot decide the 62 = 15×4+2 tail (flint-attach.md §2 made the
same point). RoPE base 500000 (`rope.freq_base_swa`) applies **only on the SWA layers**; the
converter asserts `layer_rope_theta ∈ {0, θ}` and the graph gates rope on `is_swa(il)`
(pr30616.diff:311, pr30616.diff:1382-1398) — the 16 full layers are **NoPE**. The plain
`rope.freq_base = 500000` key the header also carries is not what the graph reads.
Rope type is NORM — interleaved pairs, the converter un-permutes HF's rotate-half layout
(pr30616.diff:1146-1157, llama-model.cpp rope-type group with CHATGLM/GRANITE).

Control ids: BOS 200000 (added), EOS 200001, `<|eot|>` 200008 (added to the EOG set by name,
pr30616.diff:1244), pad 200018. The specimen's embedded chat template differs from the PR's
`models/templates/muse-spark.jinja`: effort map `{low 8, medium 32, high 256, xhigh 512}`, default
`xhigh`, versus the PR's `{minimal 1, low 16, medium 64, high 256}`, default none. The embedded
bytes are the specimen's contract; keep them and record the divergence.

## 2. The numerical contract (muse-spark.cpp, pr30616.diff:1308-1560)

One layer, in the graph's order — every deviation from the plain pre-norm body is named:

1. `inpL` enters through a **weightless RMS norm on the embedding row** (eps 1e-5,
   pr30616.diff:1364). New: no current body normalises the embedding.
2. `attn_norm` RMS (weight, eps 1e-5). QKV projections, plus `attn_gate` as a projection of the
   same normed input.
3. **Q/K norms are conversion-synthesised constants**: `attn_q_norm` = the MetaP `qk_scale_factor`
   × ones, `attn_k_norm` = ones (pr30616.diff:338-344) — a per-head RMS norm over head_dim 64
   before rope, weight constant across heads and layers. The GGUF carries the [64] tensors; the
   scale itself is folded, not a header key.
4. Rope (NormTail pairs, base 500000) on SWA layers only; NoPE elsewhere. Attention at
   `kq_scale = 1/√64` over the iswa KV (`build_attn_inp_kv_iswa`, one cache for the mixed
   window/full layers).
5. Attention output: **per-head weightless RMS norm** (eps 1e-5), then × `sigmoid(attn_gate(x))`,
   then `attn_output` (pr30616.diff:1418-1428).
6. **Post-attention weightless RMS norm at `post_norm_epsilon` = 1e-8**, then the residual:
   `ffn_inp = inpSA ⊙ attn_resid_alpha + cur ⊙ attn_resid_beta`, per-channel [8192] F32 vectors.
   α² + β² = 1 per channel, resolved at conversion from the residual gate
   (`beta = sigmoid(w/T)`, `alpha = sqrt(clamp(sigmoid(−w)·(1+beta), 1e-3))`,
   pr30616.diff:370-384).
7. `ffn_norm` RMS (weight, 1e-5). `ffn_latent_down` 8192→4096 (Q8_0), `ffn_pre_exp_norm` RMS
   (weight, 4096-wide, 1e-5), experts in the latent, then `ffn_latent_up` 4096→8192.
8. **The router reads the full 8192 stream, not the latent**: `router_logits = ffn_gate_inp · cur`
   computed outside `build_moe_ffn` and passed in (pr30616.diff:1690-1707). Gating func **2 =
   sigmoid**; `exp_probs_b` (F32 [256], in the file) steers selection only; top-8 of the biased
   scores, weights from the **unbiased** scores, renormalised (`expert_weights_norm = true`),
   scale 1.0. The MetaP `output_multiplier` is **folded into the F32 router weights at
   conversion** (pr30616.diff:363-367) — the file's `ffn_gate_inp` is pre-scaled.
9. Experts: SiLU-gated (`LLM_FFN_SILU`) gate·up·down, 4096×4096 each, in the latent. The
   aggregated expert output is scaled by `ffn_post_exp_norm` (weight) **then** RMS-normalised at
   1e-8, before the router weights apply (`build_moe_ffn`'s new `post_exp_norm` arm,
   pr30616.diff:1067-1073). Block norms hold deltas folded +1 at conversion
   (pr30616.diff:264-270).
10. `ffn_post_norm` (`post_ffw_norm`): scale by weight **then** RMS-normalise at 1e-8
    (pr30616.diff:1478-1482), then the α/β residual as in 6.
11. Head: `output_norm` RMS (weight, 1e-5) → `output` → × `logit_scale` 0.1767766922712326 →
    softcap: `x/20 → tanh → ×20` (`final_logit_softcapping` 20).

## 3. The common-owner table

Every row of `docs/contrib/new-model.md` §"the common-owner table" plus the round's rows.
**Attaches** is a design commitment naming its prerequisites.

| Row | Decision | Owner (`path:line`) | What Muse adds | Size |
|---|---|---|---|---|
| Reader | attaches (new family) | common readers `crates/model/src/arch/mod.rs` (`per_layer_count`, `meta_*`, `classify_*`, `Stem`/`stems.rs`); new `arch/musespark/{hparams,names,roles,spec,place}.rs`, `Arch::MuseSpark`, coverage rows | 23 roles, 13 hparams keys, the 62-bool pattern array read and validated (never reconstructed), `program_of = None` until the body | M |
| Tokenizer | attaches | `Pre::MiniMaxM2` **is GPT4O's one regex** (`crates/tokenizer/src/pretok.rs:528-529`); the name `gpt-4o` is refused today (`pretok.rs:910-915`, the spec's premise) | a name row aliasing the existing splitter + id oracle vs mainline `llama-tokenize` | S |
| Chat template | attaches | `crates/jinja/src/lib.rs` (namespace, macro, `.items()`, `tojson`, `trim`/`lower`/`join`/`first`, `default(x,true)`, `raise_exception` — flint-attach.md §7's inventory) | the file's embedded template byte-pinned; 2-arg `replace` (flint's gap, check whether it landed); `strftime_now is defined` guard falls to none | S–M |
| Reply / tool parser | attaches (new variant) | `ToolFormat` `crates/models/src/lib.rs:773`, parsers `crates/serve/src/` | recipient protocol (`to=self|commentary|<tool>|user`, `<|eom|>`/`<|eot|>`) + ATEM markup; reasoning = the `to=self` span | M |
| GQA attention | attaches | spec `Gqa` `crates/models/src/lib.rs:230-253` (heads/kv/head_dim/value_dim/`qk_norm`/`out_gate`/`window`), `OutGate::Proj` :259-266, `RopeMode::NormTail` :513-522 | the instance: 128/16/64, qk_norm, out_gate Proj, per-layer window 2048/None, NoPE full layers. Kernel gaps in §6 | M |
| Latent MoE | attaches | `Moe.latent` `crates/models/src/lib.rs:547-575` (added for nemotron 3 super); body owner `crates/gpu/src/arch/nemotron_h_moe/moe.rs:1-24`; `Act::SwiGlu` :600 | `pre_exp_norm`/`post_exp_norm` fields (new meaning, new case); the gated (SiLU) expert against nemotron's gate-less ReluSqr | M |
| Router | attaches | `Router`/`Score::Sigmoid` `crates/models/src/lib.rs:568-599`; the shape owner `crates/gpu/src/arch/nemotron_h_moe/router.rs:303-333` (`enqueue_rows`, instances e240/e304) | an **e256** instance; selection-only bias and unbiased-weight renorm already the owner's contract | S |
| Residual α/β | attaches (new case) | `Residual` `crates/models/src/lib.rs:644` (`Plain | Hc` only today) | `Residual::AlphaBeta`-style case: per-channel α·stream + β·branch, weightless post-norm 1e-8 on both sublayer outputs — a new meaning, so a new case, not a flag on `Plain` | S–M |
| Logit scale + softcap | attaches (new epilogue) | head `crates/gpu/src/head.rs`; elementwise `crates/gpu/src/elem.rs` (has `gelu_tanh`, no plain `tanh`) | ×0.17677669 then /20→tanh→×20 on 202,048 logits | S |
| Card leg (experts) | attaches after one entry | `Card` columns `crates/gguf/src/card.rs:401-423`; IQ2_XXS prompt GEMM ✓ (`gemm_iq2xxs`, PIN 2026-10-21, `crates/placement/src/kernels.rs:592-610`); per-stack decode gemv ✓ `iq2_xxs_gemv_sel` (`crates/gpu/src/iq_sel.rs:32`, `quant_sel.rs:995`) | **no gate·up arm for IQ2_XXS** (`Card::gate_up = None`, `qtype.rs:105-106`): two `_sel` launches + elementwise SiLU·mul first, a fused entry when a round pays for it | S–M |
| Host tier (experts) | already attached | qdot IQ2_XXS × Q8_K AVX2 (`crates/qdot/src/lib.rs:18`); union rows `crates/model/src/ops.rs:3320,6413`; tier batch port `crates/gpu/src/arch/nemotron_h_moe/prefill.rs` (`PASS_ROWS_MAX`) | nothing; the `a`/`c` union fit for the 4096² IQ2_XXS geometry is unmeasured (§5 uses the MAC-scaled band) | — |
| Residency / expert tier | attaches (later) | `TierBody::residency_parts` `crates/gpu/src/host/served.rs`; glue like `crates/gpu-glm5next/src/swap.rs` | only after the decode path exists; §5 prices the spare VRAM | M |
| KV cache (mixed window/full) | already attached | `crates/runtime/src/stores.rs:47-48` (`window_rows`/`ring_rows`, the `UNION_MAX_COLS` append margin); flint-attach.md §6 ran the same owner | ring rows min(C, 2560) at window 2048; full planes on 16 layers | — |
| Prompt batch | attaches | `crates/runtime/src/prompt.rs`; nemotron's placed prompt through the tier batch port | the body's hooks; no per-model loop | S |
| Serve seat | attaches (new module) | `serve_seats/mod.rs:1-16`; donors `mimo2.rs:1-6` (host-routed open), `ds41.rs` (two-card chain, records); shared `hybrid.rs`, `rounds.rs`, the ctx ladder | §4 | S |
| Draft | does not attach | — | **no draft in the file**: 1,429 tensors, none `nextn`/MTP/draft; the seat serves no draft and says so | — |
| Placement sizing | attaches | `plan_routed`/`plan_host_routed` `crates/placement/src/placement.rs:2240,2293`; `PlanInputs::plan_with_slots`; pattern donor `crates/model/src/arch/nemotron_h_moe/place.rs:1-13` | `place.rs`: KV bytes per layer, the card rule (§5) | S |
| Vision encoder | attaches (new family) | `crates/gpu-vision/` common parts (encoder, chain, aligner) | Muse Glimmer's encoder (`MUSE_GLIMMER` projector, pixel-shuffle merge 2, erf-GELU 3-linear MLP, pr30616.diff:1826); media ids `<|patch|>`/`<|video|>`; mmproj tensor inventory [unverified — header not read yet] | M–L |
| Audio encoder | attaches (new) | none in the tree | causal depthwise-conv pyramid over the **raw waveform** (no spectrogram): per stage fold `patch` samples → project → `depth` residual blocks (conv + RMS + GELU-erf MLP), samples-per-token = Π patch_sizes, 24 kHz mono, `<|soft_audio_start|>`/`<|soft_audio_end|>` (pr30616.diff:1875-1941) | L |

## 4. The seat

A new seat module `serve_seats/musespark.rs`, one file per model as the registry's rule, opened
`bloomery-serve --model musespark`. It does **not** join an existing seat file: the architecture
facts that force its own are (a) the iswa 46/16 KV keep rule, (b) the recipient-protocol chat
surface with no think-tag span, (c) no draft, and (d) the latent-MoE load records. Everything the
seat *calls* is shared: mimo2's host-routed open over `PlanInputs::plan_with_slots`, ds41's
two-card stage records, `hybrid.rs`'s open/load record, `rounds.rs`, the shared `--ctx` ladder.
Registration touches `serve_seats/mod.rs`, the `Model` enum, `bloomery_serve.rs:206-226`, the
`tools/hf-arch-check.py` mirror and the README Use row (new-model.md step 7).

## 5. Placement on the maintainer's box — on paper, nothing measured

Machine (rig-log `facts.md`): Threadripper PRO 5975WX, Zen 3, 32 cores, AVX2 only, host DRAM read
135–137 GB/s fitted (STREAM 147.7 best-of-5); 251 GB host DRAM; RTX 3090 24 GB + RTX A6000 48 GB,
both PCIe 4.0 ×16, no P2P; A6000 H2D pinned 26.28 GB/s. Card bandwidth for large launches
567–701 GB/s, `c_node` 0.852 µs, bridge protocol floor 21–36 µs/layer (the per-expert slope
119.5 µs × k_host was fitted at k ≤ 6 and must not be extrapolated to 8 × 62 — batch per layer
instead).

**Bytes at ctx 32768 [derived].** Weights: routed 191.81 GiB, dense 16.12 GiB, output 1.26 GiB,
embd 0.87 GiB. KV (f16, 1024 K + 1024 V per token per layer): 16 full × 32768 × 4096 B +
46 SWA × 2560 × 4096 B = **2.63 GB**. One token's active set: 8 × 62 × 12.375 MiB =
**6.44 GB** of experts, plus 18.67 GB of dense+head the cards read.

**The split the planner makes.** `plan_host_routed` (all routed stacks on the host, stages split
the dense layers across both cards; it refuses tier cards by name,
`placement.rs:1201-1210`): A6000 and 3090 take ~41/21 layers of dense+KV (12.9 / 6.6 GB), every
expert on the host (191.81 GiB ≈ 206 GB into 251 GB DRAM; `host_room`'s MemAvailable check is the
binding witness, ~40 GB headroom, one Muse per box). **NVMe holds nothing at this quant** — the
weights fit in DRAM; a higher-bit variant (IQ4_XS ≈ 460 GB) is what would need the NVMe tier.
The follow-up plan (`plan_routed` + the card rule) fills the spare VRAM with experts: ~30 GB on
the A6000 + ~14 GB on the 3090 ≈ 3,530 of 15,872 triples ≈ 22 % of expert bytes, then residency
promotes hot triples on top — both plans exist in the owner today.

**Decode timeline, one token [derived].**

| Resource | busy | basis |
|---|---:|---|
| host DRAM | **47.0 ms** | 6.44 GB / 137 GB/s (43.6 ms at 147.7) |
| host CPU | 19.1 ms | 25.0 GMAC at 1.31 TMAC/s — under the DRAM shadow |
| card SM (both cards) | 26.7–32.9 ms | 18.67 GB at 567–701 GB/s, split by stages |
| PCIe | < 0.2 ms | activations only, ~2–4 MB/token |
| bridge | 1.3–2.2 ms | 21–36 µs × 62, bulk per-layer exchange |
| launches | ~1.4 ms | ~1,600 nodes × `c_node` |

Wall = the resources' **max** once the host expert window and the card layers overlap
(the ds41overlap lever): ≈ 47–50 ms → **decode ≈ 20–21 tok/s @ n=1, depth 6, host DRAM bound**
[derived]. Serial first landing (sum): ≈ 78 ms → ≈ 13 tok/s. Depth 4096 adds 654 MB of KV read
(+1.0 ms): ≈ 20 tok/s overlapped. With the 22 % card-expert fill: host term 5.02 GB → 36.6 ms →
**≈ 27 tok/s** ceiling [derived]. The bound resource is **host DRAM read bandwidth** in every
variant; no card-side lever moves it.

**Prefill, P = 512 [derived].** Per layer all 256 experts wake (4,096 slots, m̄ = 16). Host union
`t(m) = max(W, a + c·m)` with W = 12.98 MB / 137 GB/s = 94.7 µs; `a` 28.1 µs (h1fold fit);
`c` = 27.02 µs × (50.33/35.39 MMAC per column) = 38.4 µs at the Q3_K fit, × an IQ2_XXS unpack
penalty of 1.0–1.5 [unverified — no union bench has fitted this type and geometry]:
t = 636–957 µs, × 256 experts = **163–245 ms/layer**, × 62 = 10.1–15.2 s →
**pp512 ≈ 34–51 tok/s**, host-union bound. The ladder's ceilings, same terms as V4.1's: pure
PCIe streaming of the layer's experts 3.09 GiB / 26.28 GB/s = 120.7 ms/layer → 7.48 s → 68
tok/s (card GEMM 412 GOP/layer at 47 TOPS = 8.8 ms, under the pipe); a host/stream split lands
in between, but P = 512 carries the split penalty the ds41 ladder measured, so the first landing
targets the union number and the levers (SIMD union, cardtile, G-scheduler at P ≥ 1024) are the
same ones, already priced in plan.md 「흐름 모델」.

## 6. Quant coverage and the kernel gaps

From `QROWS` (`crates/gguf/src/qtype.rs`) and `crates/placement/src/kernels.rs`:

| Type | Where in Muse | Card dense | Card head/embed | Card routed (gemm / `_sel`) | Host (qdot/union) |
|---|---|---|---|---|---|
| IQ2_XXS | routed experts | — | — | ✓ `gemm_iq2xxs` / ✓ `iq2_xxs_gemv_sel` — **no gate·up arm** | ✓ / ✓ (fit unmeasured) |
| Q6_K | attn_q, attn_output, output | ✓ Q6k site | ✓ head Q6k | — | ✓ |
| Q8_0 | k, v, attn_gate, latent down/up | ✓ Q8_0Planes | ✓ | — | ✓ |
| Q4_K | token_embd | ✓ | ✓ embed Q4k | — | ✓ |
| F32 | router, norms, α/β, exp_probs_b | ✓ F32 site (`qtype.rs:67-68`) | — | router rows ✓ (nemotron owner); **no F32 GEMM column** — the prompt router runs through the router owner's `enqueue_rows` | ✓ |

Named gaps, each an add-class change with its own `ptx-scan`/FAIL-first proof:

1. **head-64 flash pass**: `flash_gqa.rs:187` has HEAD 128/192/256 only; Muse is head 64 at
   GROUP 8 (the base group exists). M.
2. **Gated IQ2_XXS expert arm**: no common gate·up entry for any i-quant (`card.rs:37-40` lists
   the family-own exceptions; nemotron's experts are gate-less). Two `_sel` launches + SiLU·mul
   composes today; a fused entry is the lift. S–M.
3. **The fused instance** "per-head q/k norm (constant weight) + NormTail rope + NoPE append":
   the pieces exist (NormTail runs in ds41/mimo2, head-norm-fused appends in `rope_neox.rs:41-55`,
   NoPE GQA in nemotron_h) — bind one correct pass, the flint G1/G2 lesson.
4. **Plain tanh elementwise** for the softcap epilogue (`elem.rs` has only `gelu_tanh`). S.
5. **Router e256** instance. S.

## 7. Phases (each a reviewable PR with its gate)

- **(a) Reader + tokenizer + template.** The family module, refusals, the pattern array
  validated, `program_of = None` (`Need::Program`), `Pre` name row for `gpt-4o`, template bytes
  from the file, the jinja gaps, the `ToolFormat` variant + parser with the mock engine. Gates:
  `bash tools/gate.sh --release -p bloomery-model --lib -- arch::musespark --nocapture` on
  synthetic headers (FAIL-first per refusal), tokenizer unit tests + the id oracle against
  mainline's `llama-tokenize` (B), template render compared to Jinja2 on roles/tools/recipients.
  Verify alone (L/A).
- **(b) The text body on fixture bits against llama.cpp.** Fixture: a requantized mini-Muse —
  e.g. hidden 1024, heads 16/2 × 64, 4 layers keeping `[T,F,T,T]` so both window kinds and the
  transitions survive, latent 512, 16 experts of 256-wide IQ2_XXS, every (role, type) pair of the
  real file preserved — through the fixture tier (`FixtureSpec`, `fixture fetch` regenerates from
  source headers). Oracle: the refset family needs an ik build with PR 30616 cherry-picked
  (`tools/ref/build-ik-musespark.sh` + the dated PIN, MiMo's precedent, new-model.md step 3); if
  ik diverges too far, mainline at `ba3c05f71da8` dumps and the reader pins that instead. Gates:
  `dump-ref-musespark` (B), `gate-musespark-e2e` teacher-forcing at the SWA/full transition, the
  latent pair, the α/β residual and the softcapped head; the host module tests of new-model.md
  (checked sizes, zero extents, NaN refusal, FMA order) for every new host piece. The card body
  composes the shared kernels — a new kernel is its own round (§6 gaps 1–4 land here or before).
- **(c) Placement + serving at the real file.** `place.rs` over `plan_host_routed` first
  (§5), the app session, the seat, `hw_` real-file gates (header meta, load, e2e), the serve
  gate shaped on `gate-gpu-glm5next-serve`, `just affected` rows, the ptx-shapes/unsafe/
  families bookkeeping. Then the prediction-card timing: decode d6/1024/4096 and pp512/pp4096 in
  one lease against the §5 bands — a miss names the wrong term, per AGENTS.md.
- **(d) Vision.** The mmproj reader (header [unverified] until read), the Glimmer encoder over
  `gpu-vision`'s common parts, media-id splicing, gate vs the mtmd dumps. M.
- **(e) Audio.** The conv pyramid encoder + raw-waveform preprocessor + `<|audio|>` splicing;
  no prior audio encoder exists, so this is the one genuinely new subsystem. L.

## 8. Risks: where a per-model branch would grow, and the common lift instead

- **The latent MoE is nemotron's, minus the gate.** Reusing `nemotron_h_moe/moe.rs` by copying it
  and swapping ReluSqr for SiLU is the branch. Lift instead: the expert inner activation is
  already `Moe.act` (`Act::SwiGlu` vs `Act::ReluSqr`) — the body dispatches on the spec's `Act`,
  and the pre/post-exp norms become `Moe` fields both families read (nemotron's file simply has
  none and refuses, or the fields are `Option` with a named refusal — the reader decides once).
- **The α/β residual.** Tempting to fold α/β into the neighbouring norm kernels (a per-model
  epilogue). Lift: a `Residual` case the reader states and one elementwise owner runs
  (two muls + add over 8192 values, ~launch-floor cost); the weightless 1e-8 post-norms belong to
  the same case, not to each sublayer's copy.
- **The output gate + per-head o-norm.** `OutGate::Proj` already names the gate; the per-head
  weightless o-norm before it is new — put it beside `qk_norm` in `Gqa` (a bool the pass binds),
  not inside one family's attention copy.
- **The softcap.** Gemma-style tanh capping on the head: one common epilogue entry (scale → tanh
  → scale), not a Muse sampler branch; the sampler must never see it.
- **NoPE + q/k norm.** The flint G1 gap recurs (a need hidden when rope is off). Fix it in
  `models::needs` once, for both families.
- **The bridge at 8 experts × 62 layers.** The measured per-expert handoff slope does not
  extrapolate (§5); the tier's bulk per-layer port is the design's stated path — a decode that
  regresses to per-expert transfers will sit at ~16 tok/s and look like a DRAM problem it is not.
- **Router weight pre-scaling.** The file's `ffn_gate_inp` already carries the output multiplier;
  a body that multiplies again (or a fixture regenerated with unscaled weights) diverges from the
  oracle by a constant per logit. The fixture generator and the reader must state which convention
  the bytes hold.

## 9. Evidence for this design

Host-only, this round: the shard headers were parsed from `~/.cache/bloomery/muse/tensors.txt`
(1,429 rows) and `header.txt`; every byte figure in §1/§5 comes from
`python3 -I /tmp/muse-design/bytes.py <tensors.txt>` (GGUF block sizes 66/256 B IQ2_XXS,
210/256 Q6_K, 34/32 Q8_0, 144/256 Q4_K, 4 B F32 — the same formulas `QROWS` carries), whose
total 225,555,399,680 B matches the file claim; the pattern array counted from the header (46/16);
the oracle diff read end to end. No GPU, no model load, no timing: nothing here is measured. The
mmproj header was not read; its rows are [unverified]. Premises corrected against the round spec:
rope is SWA-only (full layers NoPE), gating func 2 is sigmoid with the multiplier folded into the
router weights, k_norm is synthesised ones and q_norm a constant scale, the pattern is 46/16 not
47/15, the pre-tokenizer regex already runs as `MiniMaxM2`, and `exp_probs_b` is selection-only.
