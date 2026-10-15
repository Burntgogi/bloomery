# Bloomery native contribution implementation plan

Requested sequence: initial PRs, original HIP kernel, A6000 dense validation,
AMD full forward, then same-hardware serving performance. Reference repository:
https://github.com/midagedev/bloomery. Snapshot base: 9fc2cc0d67f9a5b1d71e10663ba43873431c25e5.

## 1. Submit the two reviewed initial changes

- Rebase the excluded HIP lab and dense qwen35 frontend separately onto current
  main. Preserve every upstream gate selector, `/props` placement, stage RoPE
  accounting and the normal decision/model selection flow.
- Verify the exact clean HIP head with the six existing GPU cases, binary and
  tree hashes, host/device clippy, root format/comments, and root manifest
  comparison. The initial lab does not claim original selected GEMV or serving.
- Run full contributor T0, affected-gate selection, host gpu-gates/model tests,
  and static check/clippy for all modified CUDA frontend consumers. Do not
  describe these as CUDA code generation or runtime evidence.
- Publish source branches and two implementation PRs using the existing issue
  scope (#87/#88). Attach exact-head evidence and report PR links to the issues.
- Completion: two reviewable PR URLs with current source/evidence and explicit
  remaining NVIDIA validation. GitHub OAuth workflow permission is needed for
  upstream's existing workflows to reach the older fork.

## 2. Port the original selected Q4_K operation through a HIP shim

- Stack a separate lab-only branch on the initial HIP branch.
- Port `q4k_sel::q4k_gemv_sel` and the `cores::q4k_row_dot_1col` arithmetic.
  Use original Q8 activation planes, selected expert addressing, 256 threads /
  eight logical warp32 rows, XOR reduction, float operation order, and faults.
- Reuse `qdot::card_q8_1_scalar` for host activation preparation and
  `qdot::card_q4k_dot_row_scalar` as the oracle. Host Q8 preparation is explicit;
  device quantization and full model execution are separate milestones.
- Validate bitwise outputs over tail/pair boundaries, rpe 7/8/9, duplicate and
  shuffled selection IDs, distinct activation columns, repeated launches,
  HOST sentinels, invalid IDs, and the unchanged K6144 Q4 fixture.
- Keep old NMSE/finite checks unchanged; add typed bitwise and fault contracts.
  Commit first, run clean-head verify, attach new evidence. No timing under
  the currently resident Strata service.
- Completion: original arithmetic's HIP selected GEMV passes all named clauses
  on gfx1100. This is still an operator result, not a model backend.

## 3. Add and run the dense Qwen validation lane

- Pin the full 16,810,714,336-byte lmstudio Q4_K_M model, revision
  5a7da681f60570ab5b439a587e912d2e5eddb582, SHA256
  e00082f779fa385cee8c68a3ec8833a75778cc87272240b942f74e0b8243e520.
- Add a dense generative reference family; existing Clef dense and routed
  Qwen3/Qwen3.6 references cannot be renamed to this model.
- Extend independent reference output to forced-token raw logits and validate
  model/reference/binary identities before comparisons. Existing CLI --logits
  hashes are insufficient for numerical error comparisons.
- Add a dense clause to common serving checks: CLI/HTTP last-step at short and
  4K prompts, eager/graph, forced IDs, sampling, reset, checkpoints/prefix reuse,
  resident slots and `/props` placement. Keep image/draft clauses separate.
- Preserve the real missing tool-parser admission error. No IQ/MTP expansion.
- Maintainer runs on A6000 48GB after 0.3.3, per #88. Local host has no NVIDIA
  card or configured maintainer workstation access; do not claim its gates.
- Completion: independently produced exact-file reference and NVIDIA gate logs
  accepted in the dense PR. Host preparation alone does not complete this step.

## 4. Establish AMD full-forward equivalence, then a common backend boundary

- Use a smaller ordinary generative model with verified metadata, supported
  tensor types and enough VRAM headroom for weights, states and scratch.
- Inventory the actual dependency closure before coding: Q4/Q6/F32 linear,
  embeddings, normalization, attention/RoPE/KV, recurrent GDN where applicable,
  head/logits, stream/buffer lifetime, faults and reset. A single FFN is not a
  transformer forward.
- Port dependency operators in the excluded lab and compare intermediate
  results with original rules, then full forced-token logits and greedy IDs
  against llama.cpp HIP on the same file. Preserve allocation-free steady work.
- Only from that working closure derive the minimal common resource/runtime
  interface: Session currently owns concrete CUDA GpuModel and cannot accept
  a HIP stream by a command-line switch. Retain one model/session/seat schedule;
  do not introduce a libllama-backed replacement model loop.
- Land boundary/integration changes separately with unchanged CUDA gates and
  AMD full-forward evidence. Resident slots/checkpoints/cancellation must be
  covered on the new backend too.
- Completion: actual AMD model execution through the shared native serving path,
  with model identity, logits/token checks and original CUDA regression proof.

## 5. Measure and improve serving only after stage 4

- Reserve a quiet GPU; record both commits, binary hashes, exact model bits,
  device/driver, flags, placement, KV types, context and concurrency.
- Compare Bloomery HIP, original llama.cpp HIP and Strata separately on the
  same GPU/file/session; report unsupported configurations as unavailable.
- Measure prompt lengths 512/4096, decode depth 6/1024/4096, concurrency 1 and
  then multi-slot, TTFT, TPOT, throughput, latency distribution and peak memory.
- Use repository timing lease/prediction cards or the contributor evidence
  contract, with warm-up and interleaved arms. Functional checks under a busy
  Strata process cannot supply speed claims.
- Derive one bottleneck change at a time from launches/bytes/resource timeline.
  Apply only measured improvements, rerun correctness and same-lease A/B,
  preserve CUDA behavior and publish machine-specific evidence.
- Completion: a working common native AMD service and reproducible matched
  baseline/improved tables. No forecast is reported as a measurement.

Stage transitions are evidence-gated. Independent implementation/preparation
continues when publication is pending, but later model/performance gates cannot
be marked passed until their prerequisites actually execute.
