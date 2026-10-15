# Native Bloomery roadmap: stages 3–5 implementation audit

Audit date: 2026-10-15. Read-only source/resource inspection; no model inference,
service restart, GPU timing, source edit, or external publication was performed
for this audit. The plan below is proposed implementation, not completed support.

Upstream inspected: `9fc2cc0d67f9a5b1d71e10663ba43873431c25e5`.
Dense contribution worktree: `/home/jutai05/bloomery-native-contributions-20261013/dense-pr`.
Its concurrent rebase completed during this audit at
`00a54d4b60ff6831d283d81c6bb5e6c95c71ab70`; obtain the final tested HEAD from the
stage-1 owner before creating any evidence manifest. Root `AGENTS.md` was read.

## Dependencies and deliverables

| Work package | Can be prepared on this AMD host | Required execution evidence |
| --- | --- | --- |
| 3a: exact-file reference profile, writer, readers and dense gate | Yes: source, host refusal checks, static build, run manifest | A6000/reference CUDA runs |
| 3b: dense HTTP/CLI, slots and checkpoints | Yes: extend common test helpers and static-check | Native NVIDIA binaries and model |
| 4a: original kernel and remaining op contracts | Yes: small HIP tests if current free VRAM suffices | AMD finite/numerical/fault evidence |
| 4b: smallest whole forward | Model inventory and design now | Free AMD VRAM, all needed ops, real model file |
| 4c: same-engine backend boundary | Concrete proposal after 4b | CUDA preservation plus AMD Session gates |
| 5: serving A/B and measured optimization | Case manifest, dry-run/checker, prediction card | Native HIP serving, quiet GPU lease, matching reference support |

Logical order is stage 1 PR publication → stage 2 original HIP operator → stage
3 A6000 validation support → stage 4 whole forward and common-engine integration
→ stage 5 performance work. Stage 3 cannot be called executed while the maintainer
GPU lane is pending. Independent host preparation may proceed without pretending
that dependency has passed.

## Stage 3: reproducible A6000 handoff

### Freeze the identity

Use this exact complete model, with no conversion or replacement:

- HF repository `lmstudio-community/Qwen3.8-27B-GGUF`.
- Revision `5a7da681f60570ab5b439a587e912d2e5eddb582`.
- File `Qwen3.8-27B-Q4_K_M.gguf`, 16,810,714,336 bytes.
- SHA-256 `e00082f779fa385cee8c68a3ec8833a75778cc87272240b942f74e0b8243e520`.
- Local complete file:
  `/home/jutai05/llama-stack/models/qwen38-native-q4km/lmstudio-community/Qwen3.8-27B-Q4_K_M.gguf`.
- Actual architecture `qwen35`; 65 declared layers, 64 trunk layers; embedding
  width 5120; dense FFN 17408; 24 query heads, 4 KV heads, head width 256;
  vocabulary 248320. F32 360, Q4_K 439, Q6_K 67; 866 tensors total.
- Fifteen next-token tensors are unused. MTP is off. Host coverage still returns
  nonzero for the missing template tool-call parser; plain-text GPU checks do not
  erase or reclassify that failure.

The maintainer accepted this target and offered A6000 validation after 0.3.3:
https://github.com/midagedev/bloomery/issues/88#issuecomment-6181414720 .

### Existing tools and the exact missing pieces

1. `crates/refset/src/arch/qwen35/mod.rs` currently pins the hidden-state family to
   `/models/clef-27b/clef-27b-Q4_K_M.gguf` and mainline build `53ed051ce`.
   `gate_clef_hidden` consumes that family. The target above is a different file.
   Setting `BLOOMERY_REF_MODEL` alone does not make this a valid new-file gate.
2. `tools/ref/hidden_ref.cpp` writes `result_norm` for all prompt positions,
   not teacher-forced next-token logit rows. `hidden.sh` selects fixed existing
   set names and rejects binary/ids provenance mismatches. Preserve those checks.
3. `generate_qwen3moe` already supports `BLOOMERY_FORCE_IDS=<file>:<offset>`,
   `--mode eager|graph`, `--prefill`, `--last-step`, and `--logits`. Its current
   `--logits` only prints the last row's length, argmax, margin and FNV64. That can
   detect an eager/graph mismatch but cannot compute cross-engine numeric error.
4. `generate.rs::Diag` already has a full-row writer used elsewhere. Reuse its
   storage/record owner if extending diagnostics; its `TopRows` currently assumes
   the selected token is argmax, so a forced-ID run needs a deliberate compatible
   route instead of bypassing that invariant.
5. `gate_qwen3_serve` has `qwen3moe`/`qwen35moe` assumptions in arena accounting,
   cache planes, checkpoint clauses and mandatory image clauses. Passing a dense
   file to this binary as it stands is not a dense gate. Its 12 GiB forced-host
   expert placement clause is also inappropriate for a dense 27B model.

### Implementation package 3a: file-specific numeric contract

- Add a distinct Qwen3.8 dense-generation reference family under the existing
  `refset::arch::qwen35` owner. Use distinct set names; never overwrite the Clef
  sets. Record model full path plus SHA, the reference commit/build, architecture,
  token-id file SHA, prompt length, forced continuation SHA, schedule and complete
  trailer. Use the same metadata in the writer, reader and evidence manifest.
- Add a writer mode/helper for independent mainline llama.cpp logit rows after
  a fixed prompt and forced continuation. Keep it in `tools/ref/` and the normal
  reference build/run recipes. Obtain `llama_get_logits_ith` from the same pinned
  file; store raw little-endian f32 rows plus a shape/provenance manifest. Retain
  hidden-state output when diagnosing layer accumulation; it is a separate data
  kind, not a proxy for logits.
- Add a native GPU gate that uses the existing shared Qwen load/planner and
  `Session<Body35>`/`GpuModel`, takes the same prompt and forced ids, and reads each
  corresponding row through the existing logits API. Check finite values,
  dimensions, row count and reference identity before comparing. Compare eager
  and graph using identical schedule, rows, KV type and file. For numerical
  comparison against mainline, derive the error band from activation formats and
  sum order and record the independent CPU/CUDA reference spread. Do not assume
  arbitrary cross-engine logits or every free-running token must be bit equal.
- Use a small fixed prompt, a 64-token prompt and a 4096-token prompt, with a
  modest fixed continuation (for example 32 steps). The context must cover prompt
  plus continuation; do not set context 4096 for a 4096+32 workload.
- Add host FAIL-first checks for wrong file digest, wrong architecture/build,
  truncated rows, wrong token list, absent completion trailer and NaNs. Add the
  new family/recipe to the existing family registry and gate-path selectors.

### Implementation package 3b: same-seat serving contract

- Extract only the existing HTTP/CLI assertions that really are shared from
  `gate_qwen3_serve` into one helper owner. Add a dense profile naming its shape,
  reference family and plain-text capabilities. Do not create a parallel server
  or repeat its lifecycle, sampling, prefix-cache or scheduler implementation.
- Validate raw `/completion` token ids and chat-template-produced ids against
  native CLI `--last-step`, at short and 4K prompts. Validate greedy and seeded
  sampling, request reset, recurrent checkpoint restore, repeated/prefix-edited
  prompts and resident slots. Together-versus-alone requests must preserve ids
  and positions. Keep tool calls/media/draft explicitly outside this first lane.
- Preserve `/props` placement from latest main and the plan's RoPE and resident
  slot memory terms. Dense weights cannot be offloaded through a routed-expert
  clause: an impossible dense placement should fail by name before upload.
- Run all affected existing routed and decision gates, or provide the repository
  proof required for narrowing. No lint/tolerance/coverage weakening.

### Handoff commands that exist now

The following are concrete preflight/build/smoke commands, not the missing numeric
or HTTP correctness gates. Run the executable commands only on the A6000 lane:

```sh
# In the pinned contribution checkout and its pinned toolchain environment:
just affected origin/main
just contrib-check --tier t0
cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates \
  --features glm5next,clef,vision,mimo2 --release \
  --bin bloomery-serve --bin generate_qwen3moe

# Replace MODEL with the exact SHA-checked file; prompt.ids is a recorded,
# nonempty comma-separated list of valid ids for the file's own vocabulary.
MODEL=/absolute/path/Qwen3.8-27B-Q4_K_M.gguf
PROMPT_IDS=100000,1001,1002,1003,1004,1005,1006,1007,1008
BLOOMERY_REF_MODEL="$MODEL" BLOOMERY_DRAFT=off \
  target/release/generate_qwen3moe --tokens "$PROMPT_IDS" -n 32 \
  --ctx 4352 --mode eager --last-step --logits
BLOOMERY_REF_MODEL="$MODEL" BLOOMERY_DRAFT=off \
  target/release/generate_qwen3moe --tokens "$PROMPT_IDS" -n 32 \
  --ctx 4352 --mode graph --last-step --logits
```

For the actual handoff, put execution under the existing bounded GPU gate runner
and replace this synthetic smoke prompt with the pinned shared token fixture.
New gate names above are design deliverables, not commands already present.
The final handoff manifest must carry source HEAD/tree, build commands and binary
digests, A6000/driver line, exact model and reference identity, all exit codes and
raw output. A static result is not an A6000 result.

## Stage 4: AMD full forward, then the common engine

### Local candidate inventory (header-only inspection)

These are complete allocated files. No model forward or weight upload was run.

| File | Bytes | Architecture and tensors | Fit for first AMD Qwen forward |
| --- | ---: | --- | --- |
| Newly acquired `bloomery-native-model-candidates/notschmee/Qwen3.5-0.8B-GGUF/1870603ac8ab59c7ddb9d94cccf3b297e379538f/Qwen3.5-0.8B-Q4_K_M.gguf` | 590,880,672 | `qwen35`, 24 layers, width1024, heads8/2; Q4_K162/Q6_K24/Q8_0 1/F32 133 | Full SHA verified; host coverage has only missing tool parser. Q8_0 embedding/tied head is an explicit additional port dependency. See `small-model/README.md`. |
| `llama-stack/models/clef-flash/Cloudflare_clef-flash-Q4_K_L.gguf` | 6,202,025,280 | `qwen35`, 32 layers, width4096, heads16/4; Q4_K98/Q6_K46/Q8_0 53/Q5_K5/F32 225 | Same native Body35 family, physically smaller; requires Q5_K and Q8_0 too, plus complete coverage check and exact digest. Not the three-type initial target. |
| `clef-tactical/models/Clef-Flash-Q4_K_M.gguf` | 6,486,448,192 | `clef`, decision type `clef`, 32 layers; Q8_0 211/Q4_K97/Q6_K1/F32 277/BF16 1 | Decision backbone layout; unsuitable as a plain-generation shortcut. |
| `d1-tactical/models/d1-3B-Q4_K_M.gguf` | 1,674,456,672 | `lfm2`, decision type `lfm2-d1`, 30 layers, width2048; Q4_K148/Q6_K19/F32 99 | No native Bloomery lfm2 architecture in the inspected dispatch; not a smaller Qwen. |

The two local Flash-Next IQ3_S/IQ3_XXS shard sets are larger `qwen4exp` targets
with extra kernel/format/host-placement requirements. Header-only sparse GGUFs
and partial downloads in the earlier evidence directory are not model candidates.
If reducing the first port to Q4_K/Q6_K/F32 is necessary, acquire a genuinely
smaller plain `qwen35` generation file only after inspecting every tensor type and
the model metadata. Pin its upstream revision/size/full SHA before claiming it is
the new target. The follow-up screened three0.8B files and found no exact
three-type target; it acquired the notschmee candidate above after explicitly
accepting its one Q8_0 embedding/tied-head dependency. This does not replace the
initial dense27B contribution target or complete the AMD whole-forward stage.

### Current resource blockers

At the audit reading, `/sys/class/drm/card0/device/mem_info_vram_total` was
17,163,091,968 bytes and `mem_info_vram_used` was 17,044,054,016 bytes: only
119,037,952 bytes (about 113.5 MiB) unoccupied. The existing user service was
Strata PID818278 with wrapper PID818272, serving Flash-Next on port8080, using
the IQ3_XXS pack and native GGUF shards. This was observed, not changed. Current
usage can vary and must be read again before any future allocation.

No `/dev/nvidia*` node exists; `/dev/dri/card0` and `renderD128` exist. Thus this
host cannot execute the offered A6000 lane, and its currently occupied AMD card
cannot load even the 6.20 GB candidate. A model run needs a user-approved service
maintenance window or a separate card. Do not stop the current service merely
because operator tests succeeded. No serving timing is valid while that service
competes for the card.

### Why the original-kernel port is necessary but insufficient

`app::Session<B>` owns concrete `GpuModel<B>` (`crates/app/src/lib.rs`).
`ChainBody::refresh` takes `CudaStream`; `enqueue_chain` takes concrete `Gpu`,
`Weights` and `Head` (`crates/gpu/src/model.rs`). `Gpu` owns CUDA modules,
`DeviceBuffer`s and a CUDA stream (`crates/gpu/src/lib.rs`). `tensor.rs`,
`graph.rs`, `weights.rs`, the arena and layer stores are also concrete CUDA types.
`Body35` dispatches both gated attention and gated delta recurrence through its
existing `Program` and `runtime::sched::walk`. `SeatEngine` is generic over its
seat, but replacing just its seat with a separate inference library would not
preserve this model/session engine.

The `model::arch::qwen35moe` CPU module supplies hparams/spec/roles/placement and
host expert helpers. It has no whole Qwen `forward.rs`; only the older deepseek2
architecture exposes the inspected CPU `forward`/`step`. A whole-model CPU
fallback is not an existing Qwen integration solution.

### Incremental implementation and exit gates

1. Complete stage 2's source-faithful selected Q4_K kernel using packed Q8
   activation planes, the same selected-id layout, logical warp32 semantics,
   reduction order and fault behavior. Record exactly which original entries were
   ported. F32 scalar GEMV is a separate reference, never evidence of that port.
2. Enumerate every entry the chosen dense one-row Body35 plan reaches. Port in
   operator groups with independent tests: embedding and head types; RMS/Q8
   quantization and dense FFN; gated-QKV/RoPE/KV write/read/GQA; GDN convolution,
   state update, gating and output; residual and finite/fault propagation.
   Add Q5_K/Q8_0 only if the chosen exact file needs them. Keep steady buffers
   resident and exclude graph, MTP, vision and multi-slot claims initially.
3. Demonstrate the complete one-row forward in the isolated native lab, using
   Bloomery's existing metadata, quantization contract and layer order. Use the
   exact same GGUF/id stream with llama.cpp HIP, record per-layer diagnostics and
   full logits, then short greedy output. Derive numeric bands; token divergence
   is a finding to inspect at its forced-prefix row, not permission to silently
   substitute a reference forward. A lab full forward is still not Session/HTTP
   integration. Do not publish its throughput as native Bloomery serving.
4. With a real second resource implementation, propose the smallest common
   backend cut to the maintainer: compile-time context/stream/buffer/event/copy/
   graph/fault ownership plus operator providers below the current model/session
   owners. Preserve one Body35 layer program, GpuModel state machine, Session
   reset/position/checkpoint logic, and SeatEngine scheduling. Avoid a giant
   trait branch per model or a second handwritten serving loop. A blanket
   CUDA-to-HIP type rename is not sufficient for CUDA graph/memory semantics.
5. Lift shared resource ownership incrementally with CUDA as the default.
   Preserve drop ordering (graphs before buffers), thread affinity, asynchronous
   streams, poison/reset, plan byte accounting and kernel faults. CUDA proofs
   need unchanged PTX where applicable and the repository's affected gates;
   HIP needs actual eager execution. Name/refuse unsupported HIP graph or slot
   capabilities instead of emulating their success. Preserve the API owners and
   add HIP graph/checkpoint/slots only as their real tests become possible.
6. Attach the native HIP body through the same `Session` and `SeatEngine`, then
   repeat stage3-style CLI/HTTP/reset/prefix/concurrent-slot checks on AMD. Only
   this step establishes native AMD Bloomery serving. Operator lab completion or
   a generic HTTP Engine implementation alone does not establish it.

## Stage 5: serving measurements and improvement

Stage 5 depends on stage4 native serving and a quiet card. Build a case manifest
before running: exact source+binary identities, GPU+driver, full GGUF SHA, token
fixture SHA, context/KV precision, batch/ubatch, placement, concurrency, sampler,
draft setting, warmup policy, timed repetitions and expected improvement.

Baseline cases: prefill P512 and P4096; decode depths6/1024/4096 with equal counted
tokens; HTTP concurrency1/2/4; fresh and prefix-reused prompts separately. Record
TTFT, inter-token latency, end-to-end latency, decode/aggregate throughput, peak
VRAM/host memory and errors. Free-running output length/EOS differences cannot
silently alter compared work. Keep functional output checks outside the timed
region, retaining the same file and token/sampling policy.

Use existing `tools/ref/` runners/lease infrastructure. Add HIP card discovery,
idle/clock/power witness and native-serving arms in the common owners; do not
time by manually issuing requests without provenance. Keep separate per-machine
tables. Rotate arm order and include an A/A ruler for small claims. Fail closed
when file identity, backend or placement differs. A runner may support a dry-run
manifest now, but measured rows remain absent until the prerequisites pass.

The current local Strata README and active process target Qwen3.8-Flash-Next,
not this dense27B file. Therefore the first native dense/small-model comparison
can be against llama.cpp HIP; a Strata ratio is only valid after both engines
actually support the same architecture and exact weight bits. Existing
Flash-Next IQ3_XXS Strata measurements cannot be placed in the dense27B table.

Choose an optimization only after the baseline identifies a real term: launch
submission cost → graph replay; repeated activation conversion → preserve the
existing fused Q8 path; narrow prefill → original wide operator path; memory
traffic → measured layout/coalescing; serving overhead → existing common prompt
and slot scheduler. Write the resource timeline and derive a predicted band
before timing. For each accepted change, retain correctness gates and same-lease
A/B evidence, then report the actual machine-specific improvement and interval.
