# Native HIP full-forward evidence — f71e1483

This is the selected public evidence snapshot for the native HIP dense Qwen3.5 full-forward contribution. It is attached separately from the source branch. Run/preparation-time publication flags inside reports describe their capture time, not the live state of GitHub publication.

## Verified identities

- Source HEAD: `f71e148395ab810aa40c577861ff7a3fd277873c`.
- Repository tree: `ea2a9bb28ae919deb5ef6af1840d55401169b301`.
- Lab tree: `4d22acaf7d6dad0dc393edb21686ff8e146953b1`.
- Upstream main base: `11d8a7d37a7648f3d5fc058f293947638313b2c4`.
- Source: [verified fork commit](https://github.com/Burntgogi/bloomery/tree/f71e148395ab810aa40c577861ff7a3fd277873c).
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
- `reports/{scope,review,comparison-vs-reviewed,patch-reconstruction,commands}.redacted.json`: scope, required review corrections, pre-fix output-digest comparison, exact-tree patch proof and execution transcript.
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
