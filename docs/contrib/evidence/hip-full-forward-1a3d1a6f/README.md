# Native HIP full-forward evidence — 1a3d1a6f

This is the selected public evidence snapshot for the native HIP dense Qwen3.5 full-forward contribution. It is attached separately from the source branch. Run/preparation-time publication flags inside reports describe their capture time, not the live state of GitHub publication.

## Verified identities

- Source HEAD: `1a3d1a6f2f34cc808f4e1ab9fba83d8d23e5e004`.
- Repository tree: `84fbace83f79e05ffcc0c31d9ce5830c494f601e`.
- Lab tree: `660113afa5a1564dc44d27aebac6e3ccb1b4d40e`.
- Upstream main base: `11d8a7d37a7648f3d5fc058f293947638313b2c4`.
- Source: [verified fork commit](https://github.com/Burntgogi/bloomery/tree/1a3d1a6f2f34cc808f4e1ab9fba83d8d23e5e004).
- Model: `notschmee/Qwen3.5-0.8B-GGUF`, revision `1870603ac8ab59c7ddb9d94cccf3b297e379538f`, `Qwen3.5-0.8B-Q4_K_M.gguf`, 590880672 bytes; SHA-256 `be5b3cb8c2f3ebd791349c9c775ea38950f23a2b9e792ca2f0411597dd172988`.
- Reference llama.cpp: `fb4b2737a808a3fb7c2117a498f43815dc9be53e`, clean source and normally rebuilt HIP libraries.

This snapshot follows the three required corrections to PR #476 at `374fff16`: shared finite/F64 NMSE and fault gates, Context-owned typed transfers, and local device unsafe invariants. The zero-oracle denominator-floor error was reproduced with an isolated copy of the historical comparator (expected refusal assertion failed); the current shared-comparator regression passed. All canonical and model checks were rerun at the new clean source HEAD above. Fixed input text/IDs and their original tokenizer-attribution record are retained unchanged. No older model run substitutes for current-HEAD evidence. The nonblocking synchronization optimization is not implemented.

## Results and limits

RX 7900 GRE / gfx1100 / wave32, HIP runtime and driver API `70253210`. Canonical verification passed all eight command groups and 18 GPU cases. Both fixed natural-language cases passed independent greedy8, native reference-prefix forced argmax, complete finite vocabulary rows, and two actual in-process reset runs with bit-identical per-engine results. The reference observed all 24 layers and the output head on HIP; CPU math was limited to standard input embedding. Context64, ubatch1, F16 KV and the production 256MiB reserve were retained.

| Fixed input | Greedy / forced agreement | Maximum row relative L2, diagnostic only |
|---|---|---:|
| The capital of France is | 8/8 and 8/8 | 0.027311885469503642 |
| 대한민국의 수도는 | 8/8 and 8/8 | 0.0226808442104578 |

Each compared row contains all 248320 vocabulary values. Cross-engine logits are not bit-identical. These are two fixed raw prompts, not general English/Korean quality, serving/chat/tools, performance, CUDA parity, other-GPU/IGPU support, or full CI acceptance. Memory preflight is not an OS allocation cap or a measured peak. No resident service was stopped or reconfigured.

## Contents

- `public-summary.json`: curated identity, results, environment and limits; not a verifier output.
- `reports/canonical.redacted.json`: complete canonical report with declared path substitutions.
- `reports/{english,korean,small}.redacted.json`: complete current-HEAD comparison reports.
- `reports/reference-build.redacted.json`: reference source/binary/library proof.
- `reports/{scope,issue479,reproduction,comparison-vs-parent,patch-reconstruction,commands}.redacted.json`: scope, Q4 correction/reproduction, parent output-digest comparison, exact-tree patch proof and execution transcript.
- `inputs/`: byte-identical fixed input text/IDs and a path-redacted tokenizer-attribution record.
- `logs/`: selected canonical/static/reference/comparison/refset/CPU logs and real-case final reference/native logs.
- `hip-full-forward-current.patch`: byte-identical current-base-to-HEAD source patch.
- `artifact-ledger.json`: original/public byte counts, SHA-256 and transformation counts for every original-derived file.
- `SHA256SUMS`: public-copy hashes for every other text file in this evidence snapshot; excludes itself.

The source patch/bundle can reconstruct the verified source, but model weights, extracted tensor bytes, raw `.f32` logits, target/runtime binaries, Git bundles and credentials are not included here. Their identities may appear in reports. Unmodified complete original bundles remain retained locally.

## Path and digest disclosure

Known machine-specific prefixes in JSON string keys/values and logs are replaced by symbolic labels such as `$SOURCE_ROOT`, `$EVIDENCE_ROOT`, `$MODEL_ROOT`, `$LLAMA_SOURCE`, `$LLAMA_BUILD`, `$ROCM_SDK`, `$TOOLING_ROOT`, `$FIXTURE_ROOT`, `$PREVIOUS_EVIDENCE`, `$PREPUBLICATION_EVIDENCE`, `$ORIGINAL_SOURCE_ROOT` and `$HOME`. These are disclosure aliases, not executable environment assignments. Generic `/tmp/hip-*` paths in the unchanged source patch are public documentation examples.

The `.redacted.json` and `.redacted.log` files are explicitly named derivatives, **not byte-identical originals or canonical-admission inputs**. JSON processing changes matching prefixes only inside serialized string tokens; all non-path bytes, numerical spellings, results, statuses and identity hashes remain unchanged. Inputs and the source patch are copied unchanged.

Nested raw/binary/model/source/log digests and byte counts always describe retained original artifacts. They are not silently recomputed to describe redacted public files. A report's original log SHA therefore need not match its disclosed redacted log. Use `artifact-ledger.json` and `SHA256SUMS` to authenticate public copies; the ledger also identifies original file hashes without disclosing private absolute paths.

This is a publication record, not a self-contained runnable comparison bundle. Independent reruns need the pinned sources/model, appropriate toolchains/hardware, new output directories, matching binaries and fresh clean-HEAD canonical evidence. Source README commands and the historical command transcript document the execution interface.

Keep evidence outside the verified source branch. Any source commit or rebase requires fresh exact-head verification. The source PR remains related to #87 rather than closing it; #248 and #252 are already merged. The separate evidence commit must never be described as the source commit that was run.

## Q4 zero-energy refusal correction

This source additionally addresses #479. Relative to `f71e148395ab810aa40c577861ff7a3fd277873c`, only the lab README, host main and mixer comparator changed. One `checked_nmse` owner serves Q4, dense FFN and core/mixer oracles. The Q4 dot products stay F64; kernels, launches, transfers and synchronization are unchanged.

The existing Q4 gate was exercised on an actual RX 7900 GRE with a generated zero-filled 576-byte fixture (K256, rows4). An isolated device mutant storing `sum + 1e-20_f32` incorrectly passed at the parent source: NMSE=3.999999746124184e-10 for cols1 and 1.2799999187597388e-08 for cols32. The fixed host refuses that device with exit1/NMSE infinity. A second isolated device mutant adds the error only for cols32; the fixed host accepts the exact-zero cols1 case and then refuses cols32. The unmodified device passes both zero-output cases with NMSE0/exit0. The production-source device ELF is unchanged.

`reports/reproduction.redacted.json` records the observed baseline, and `reports/issue479.redacted.json` records both refusal paths and the legitimate-output control. `reports/comparison-vs-parent.redacted.json` compares complete output hashes and metrics to the parent. `issue479-fix.patch` is the incremental source patch. The two `mutant-*.patch` files are throwaway verification mutations only, not changes included in the source PR or canonical acceptance. The new permanent Q4 regression failed before normalization was corrected and passes afterward for zero/tiny oracle energy and cols1/32.

To reproduce, generate the packed zero fixture with Python's `Path('/tmp/q4-zero.bin').write_bytes(bytes(576))`; build the lab host/device using the source README, and run `"$BIN" "$DEVICE" /tmp/q4-zero.bin 256 4`. For negative controls, create isolated worktrees at the recorded parent source, apply each corresponding mutant patch, build only their device crate with the documented AMDGPU flags and separate target directories, then supply those mutant ELFs to the fixed host. Never reuse mutant artifacts as canonical or model-comparison evidence. The parent host should exit0 for the mutant; the fixed host should exit1 at the intended case. Fixed real-model prompt IDs, outputs and all native/reference full-row hashes remain unchanged in fresh exact-HEAD runs.
