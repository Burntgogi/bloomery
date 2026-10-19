# Native Rust HIP feasibility lab

This isolated lab compiles Rust kernels for `amdgcn-amd-amdhsa`, loads their ELF through HIP, and runs operator probes and an eager raw-token dense Qwen3.5 hybrid forward. It calls no libllama, ggml model forward, or llama-server process. Both host and device code are Rust. The nested workspace is excluded from the default CUDA workspace.

The validation target is one RX 7900 GRE (`gfx1100`), Rust nightly 2026-08-28, its LLVM backend and the local HIP runtime. Rust's AMDGPU target is tier 3: this is a feasibility result, not a supported release backend. The original CUDA model chains, seats, kernels, scheduling and MTP remain unchanged.

## What runs

- Device-buffer allocation and synchronized HIP transfers, plus a Rust GPU copy kernel with exact roundtrip equality.
- A scalar Q4_K × F32 matrix kernel on synthetic shapes and real `blk.1.ssm_out.weight` rows from Qwen3.8-27B IQ3_S. Matrix rows use GGUF Q4_K's 144 bytes per 256 weights. Scales and fused dequantization follow `crates/gguf/src/quant.rs`.
- A 256-wide synthetic dense FFN unit: RMSNorm, separate gate/up projections, SwiGLU, down projection and residual. Its intermediate values stay on the GPU. It does not include attention, GDN state, routed experts or a complete transformer block.
- The original selected Q4_K down row walk with Q8 activation planes, 256-thread blocks and logical warp32 reduction. Host-side `qdot::card_q8_1_scalar` prepares the inputs; device quantization is a later step.
- A single-lane dense Qwen3.5 chain: on-device packed embedding/projections/head, GDN conv and recurrence, gated GQA with partial Neox RoPE and F16 KV, dense SwiGLU FFNs, residuals and finite argmax. The common `model::arch::spec` reader supplies layer kinds, dimensions and tensor roles. This numerical harness is not a second serving engine.
- Operator repeats use already allocated buffers. The reported zero count covers allocations explicitly issued by the selected-probe host, not internal driver allocations or diagnostic file output.

The scalar probes retain fixed NMSE <= 1e-6 and finite outputs. The selected Q4_K port requires bitwise equality with the existing `qdot::card_q4k_dot_row_scalar` oracle for finite activations, NaN propagation for refused activation scales, and exact fault words, masks and skipped-slot sentinels. This is host-oracle/HIP equality, not a measured CUDA/HIP or model-token comparison. No throughput is claimed.

## Verify

The lab owner is [Burntgogi](https://github.com/Burntgogi). Install the repository's pinned
nightly with rust-src, rustfmt and clippy, Python 3.11 or later, and a HIP runtime exposing
`libamdhip64.so`. The verified target is `gfx1100`; no other card is claimed.

From a clean, committed checkout:

```sh
export HIP_LIB_DIR=/opt/rocm/lib
export LD_LIBRARY_PATH="$HIP_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
labs/hip-native/verify.sh /tmp/hip-evidence
# Optional real Q4_K rows: exact packed bytes, K and row count.
labs/hip-native/verify.sh /tmp/hip-real-evidence \
  --q4-rows /path/to/q4_k_rows.bin --k 6144 --rows 64
```

The output directory must be new and outside the checkout. An SDK outside `/opt/rocm` may
need additional runtime library directories. `verify.sh` checks formatting, host tests,
host and device clippy with warnings denied, builds both release binaries, and runs the
numerical cases under a 900-second bound per command. It compiles the standalone Rust
AMDGPU device with `cargo -Zbuild-std=core`, never the existing CUDA device crates.

`result.json` schema 3 records the clean HEAD commit, whole-repository and lab Git tree hashes,
release binary SHA-256 hashes, Rust/LLVM version, target CPU, actual device name, HIP runtime
and driver versions, fixture hash when present, command logs, operator cases and bounded hybrid evidence.
A failed check returns nonzero and writes a failed result. Attach the JSON from the PR's exact head; a new
push requires a new run. Logs and generated artifacts stay out of the source tree.

The default run has three synthetic matrix shapes, one dense FFN case and eleven selected
Q4_K geometries. The optional fixture adds matrix-vector, 32-column and selected-Q4_K cases;
it must contain a positive multiple of four rows for the four-expert view. Use packed rows of a Q4_K tensor obtained
through the existing GGUF reader (`gguf::Split`/`TensorInfo`), with no conversion of its
weight bits. Record the original file SHA-256, tensor name and row range alongside the
result; do not upload model weights in the PR. Case results establish numerical operator
checks only, with no timing claim.
The verifier then runs `--forward-probe` in a separate process. Its `forward` result independently
checks both repeats' complete logit files, their digests, finiteness and argmax, rather than trusting
the writer's success flags. `same_file_real_model_llama_HIP` remains false: passing the bounded
synthetic hybrid does not close the pinned real-model milestone.

Scalar kernels use 64-thread workgroups. The selected port uses 256 threads (eight logical
32-lane rows), and a one-thread probe checks the compiled wave size before any selected launch.
Kernel signatures and argument storage must remain identical. Context/module/stream/buffer handles stay on their
owning thread, and synchronization precedes destruction.

## Eager dense Qwen3.5 forward

`--inspect` opens GGUF metadata through the common reader without initializing HIP or loading
weights on a GPU. Admission checks exact tensor axes and formats; unsupported mixers, routed FFNs,
extra sites, non-plain residuals, scaled/windowed attention and contexts outside 1–64 are refused.
Supported matrix storage is F32, Q4_K, Q6_K and Q8_0; gains, conv weights and GDN parameters are F32.
An absent `output.weight` aliases the embedding allocation, not a second copy.

All token-dependent numerical work runs on the device. RoPE constants, weight names, function
handles, dimensions and arena addresses are resolved at load. GDN uses head128, conv4, the original
11-row input ring, tiled key heads and value-major recurrent state. GQA reads interleaved per-head
Q/gates, applies Q/K norms and partial Neox rotation, appends round-to-nearest-even F16 KV, and attends
only live causal rows. Compiled wave32 is checked before the eight-lane delta reduction.

The eager GEMV lane dequantizes original GGUF bytes on the GPU and reads **F32 activations**. It is
not the selected kernel's Q8 activation arithmetic, and its scalar softmax is not CUDA flash's
segmented reduction. Preserve this distinction when interpreting logit differences. No throughput,
CUDA/HIP bit parity, production-backend support, or serving admission is claimed.

The model preallocates weights and one compact lab arena; `step` resolves no names and allocates no
buffers. Fit admission retains the production owner's `FIT_RESERVE`, extracted by `host/build.rs`
without linking CUDA. The reported managed-byte estimate excludes driver and kernel scratch.
The bounded probe additionally caps explicit managed allocations at 16 MiB. Neither estimate is a
measurement of peak VRAM residency or permission to interrupt another GPU service.

A failed launch or device fault poisons the model before accepting its token. Further steps and
logit reads are refused until `reset`, which clears recurrent state, conv rings, KV and the sticky
fault. Invalid token IDs and exhausted context are refused before mutation. The probe exercises
these transitions by inducing a real nonfinite gain, restoring it, and checking bitwise recovery.
Independent operator oracles retain NMSE <= 1e-6; GDN crosses ring wrap and GQA covers all 64 positions.
Core and mixer probes share one finite-value, F64 NMSE and fault-checking implementation.
Zero-energy oracles require exact agreement; nonzero energy is never floored before normalization.
Typed host/device transfers share `Context::upload`, `download` and `overwrite`, retaining the
existing synchronous copy and launch boundaries.

For development builds from this directory, with the same runtime setup as above:

```sh
unset CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_TARGET
HOST=x86_64-unknown-linux-gnu
RUSTFLAGS='-C target-cpu=native' \
  cargo build --release --target "$HOST" -p bloomery-hip-spike
RUSTFLAGS='-C target-cpu=gfx1100 -C panic=abort' \
  cargo build -Zbuild-std=core --release --target amdgcn-amd-amdhsa \
  -p bloomery-hip-spike-device
BIN="target/$HOST/release/bloomery-hip-spike"
DEVICE=target/amdgcn-amd-amdhsa/release/bloomery_hip_spike_device.elf
"$BIN" --inspect /path/to/Qwen3.5-0.8B-Q4_K_M.gguf 64
"$BIN" --forward-probe "$DEVICE" /tmp/hip-bounded-forward
# Only in an authorized GPU-memory window; output directories must be new.
"$BIN" --forward "$DEVICE" /path/to/Qwen3.5-0.8B-Q4_K_M.gguf \
  /path/to/prompt.ids /tmp/hip-real-forward 8 64
```

`prompt.ids` holds one unsigned token ID per line: no tokenizer, BOS insertion or chat template
runs here. Zero greedy steps feeds only the supplied IDs. For positive steps the first generated
ID is the last prompt row's argmax, and the final generated ID is not fed back; consumed positions
are `prompt.len() + max(steps - 1, 0)`.

`forward.json` records model file hashes, the resolved allocation estimate, input/predicted/greedy
IDs and both repeats' logit digests. `run-{0,1}/logits-NNNN.f32` is a full little-endian F32 vocabulary
row after each consumed token, including prompt rows. A reference whose row zero follows the whole
prompt aligns with native row `prompt.len() - 1`. Diagnostic host transfers and file/hash work are
outside `Model::step`; they are not a performance path.

The fixed acceptance file remains `notschmee/Qwen3.5-0.8B-GGUF` revision
`1870603ac8ab59c7ddb9d94cccf3b297e379538f`, `Qwen3.5-0.8B-Q4_K_M.gguf`,
590,880,672 bytes, SHA-256
`be5b3cb8c2f3ebd791349c9c775ea38950f23a2b9e792ca2f0411597dd172988`.
Header admission is not a real-model GPU execution. Completion still requires an authorized memory
window, refreshed llama.cpp HIP source/build attribution and observed HIP placement, identical raw
English/Korean IDs, context64/ubatch1/F16 KV, eight equal greedy tokens and two identical reset runs.
Report full-logit error diagnostically; do not invent a whole-model accuracy band or substitute CPU
reference/synthetic-model success. Keep chat/tool admission separate. Only then discuss the common
backend boundary with the maintainers.

## Same-small-GGUF HIP comparison

`compare-hip.py` is a separate, fail-closed comparison controller. It requires a clean,
committed, unchanged source HEAD and a **new canonical `verify.py` result from that exact
HEAD**, not historical evidence. Both native executable and device ELF must still reside
under this lab's `target` directory and match the canonical SHA-256 records. It does not
build binaries or silently refresh stale evidence.

The default target accepts only canonical `forward/smoke.gguf`: the deterministic two-layer
GDN+GQA fixture, 685,984 bytes, vocabulary258, SHA-256
`420f680e27e9ec75c7c29d39298574f7fd2aff8808e9b2566f3915664016ed06`.
The root reference target `hip-lab-hybrid` attributes this fixture to
`midagedev/bloomery`, revision `03372d339f99996ec5103484f6e917412191d661`, file
`smoke.gguf`. No real0.8B model is loaded by this workflow.

From the repository root, with the HIP runtime environment configured as in **Verify**:

```sh
# LLAMA_TREE is a clean pinned llama.cpp checkout; LLAMA_HIP_BUILD is its
# matching CMake build directory configured with GGML_HIP=ON.
python3 tools/ref/dense-logits.py build \
  --llama-tree "$LLAMA_TREE" --llama-build "$LLAMA_HIP_BUILD" \
  --out labs/hip-native/target/dense_logits_ref
labs/hip-native/verify.sh /tmp/hip-current-canonical
python3 labs/hip-native/compare-hip.py /tmp/hip-small-comparison \
  --reference labs/hip-native/target/dense_logits_ref \
  --native-evidence /tmp/hip-current-canonical/result.json

# Consumer-side contracts only; no GPU initialization or model execution:
python3 labs/hip-native/compare-hip.py --self-test
```

All evidence directories must be new and outside the checkout. The reference build record
must bind the current C++ harness, clean llama.cpp source, HIP-enabled build, binary and
resolved shared libraries. Cached-library smoke attribution is not accepted. Inherited
HIP/ROCR visibility is preserved; the reference explicitly selects the first visible ROCm
device and requires observed computation for both trunk layers and the output head on
that device. CPU math-node counts are retained honestly: the standard reference input
embedding leg may remain on CPU. This is not a CPU-reference substitution or a claim that
every reference node runs on HIP.

Captures use the current forced-logits v2 manifest, including its `schedule` header
and checked tensor sum. HIP diagnostics additionally bind the clean lab source HEAD
and retain the full llama.cpp commit hash. They are not registered `logits-qwen38d`
reference sets; that family's CPU/CUDA build identity and acceptance remain unchanged.

Inputs stay fixed: raw prompt `[3,17,41,9,23]`, eight greedy IDs, context64, ubatch1,
F16 KV and step prefill. The controller derives the reference continuation independently
with eight HIP prefix captures; sentinel forced ID0 is not consumed. It then captures all
eight reference rows in one final forced run and checks its argmax against those independently
derived IDs. Native HIP runs both its own greedy8 and a zero-greedy-step forward consuming
the original prompt followed by the first seven reference IDs, using the exact same GGUF.
Reference row0 aligns with native row4, not native row0 or5.

Every reference capture and both native runs perform two real in-process reset passes.
The controller reads the complete raw F32 files, checks dimensions, finiteness, independent
argmax, file/model/source/binary identities and reset bit equality, and retains each
reference capture's placement, memory and build attribution. Native processes and the
reference's actual GPU child use the root driver's bounded cleanup, with a 900-second
bound per launch. No service is stopped or reconfigured.

`comparison.json` schema1 has `status=pass|mismatch|error`; only `pass` exits zero.
Successful and token-mismatch comparisons contain all eight aligned-row diagnostics:
max absolute error, reference-normalized relativeL2 (null for a zero reference norm with
nonzero error), actual argmax IDs and diagnostic bit equality. There is **no invented
numeric tolerance or full-logit parity gate**. Passing requires finite complete rows,
reset equality, unchanged identities and agreement of the independently derived reference,
final reference, native greedy and native reference-forced token IDs.
`first-divergence.json` preserves the first differing token's prefixes, complete rows and
artifact offsets/paths. Errors preserve incomplete bundles under `.staging`, native row
files and command logs where available; they never publish a fake complete comparison.
The report distinguishes equivalent driver CLI arguments from actual child argv recorded
by the imported reference runner.

The reference's default16MiB tensor-payload bound is not a total VRAM cap. Both lanes retain
the current production `FIT_RESERVE`; the reference additionally checks selected-device free
memory before model/context/decode stages using its recorded conservative context allowance.
Driver allocations, fragmentation and unmodeled backend workspace remain outside that
estimate. These are stage preflight checks, not an OS-enforced cap or measured peak.
Insufficient headroom is a nonzero refusal, never permission to evict another service.
`real_model_HIP_acceptance` is always false for the small target. The opt-in pinned
real-model workflow below uses the same controller and checks, not a synthetic substitute.
Neither workflow grants serving, performance or CUDA/HIP bit-parity acceptance.

## Same-real-model HIP comparison

After obtaining an authorized memory window, use the pinned 0.8B GGUF from **Eager dense
Qwen3.5 forward** with `--target qwen35-0.8b-candidate --model ... --prompt ...`.
The model must match its complete file size and SHA-256. Native metadata must describe
24 layers (18 GDN, 6 GQA); reference callbacks must observe all24 layers and the output
head on HIP. The small canonical probe is still required from the current clean HEAD
to bind the native executables and their operator checks.

Prepare separate English and Korean raw-ID files with this exact model's tokenizer.
Record the original UTF-8 bytes, tokenizer source/binary identities, BOS and special-token
settings alongside the IDs. Do not apply a chat template or alter inputs after seeing a
token mismatch. Each prompt plus eight continuation IDs must fit context64.

```sh
python3 labs/hip-native/compare-hip.py /tmp/hip-real-english \
  --reference labs/hip-native/target/dense_logits_ref \
  --native-evidence /tmp/hip-current-canonical/result.json \
  --target qwen35-0.8b-candidate --model /absolute/path/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt /absolute/path/english.ids
python3 labs/hip-native/compare-hip.py /tmp/hip-real-korean \
  --reference labs/hip-native/target/dense_logits_ref \
  --native-evidence /tmp/hip-current-canonical/result.json \
  --target qwen35-0.8b-candidate --model /absolute/path/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt /absolute/path/korean.ids
```

The reference tensor-payload limit is the pinned real file's size rather than the small
target's16MiB default. This permits that file only; it does not reduce `FIT_RESERVE` or
skip selected-device memory preflight. Context64, ubatch1, F16 KV, independent greedy8,
complete finite vocabulary rows, actual context reset2 and same-prefix forced diagnostics
are unchanged. Reference row0 aligns with native row `prompt.len()-1`, including prompts
whose lengths differ from the small fixture.

For the real target, a passing report sets `real_model_HIP_acceptance=true` **for that
single raw prompt only**. The English/Korean milestone requires both reports to pass at
the same clean source and binary identities. A mismatch remains nonzero, preserves the
first differing full vocabulary row and never changes inputs or invents a logit-error
tolerance. These are correctness diagnostics, not chat/tool admission, release-backend
support or measured performance.

## Original selected Q4_K port

The arithmetic baseline is Bloomery `9fc2cc0d67f9a5b1d71e10663ba43873431c25e5`;
the whole-file drift pin is reviewed against main `6d4113c1`.
`crates/gpu/src/q4k_sel.rs::q4k_gemv_sel` owns selection and launch geometry;
`crates/gpu/src/cores.rs::q4k_row_dot_1col` owns the arithmetic. `device/build.rs`
extracts nine original core function bodies verbatim, retaining the two-iteration
accumulation rule. All nine bodies and the selected entry are unchanged from the
arithmetic baseline; the intervening core additions are unrelated IQ lookup helpers.
A pinned whole-file FNV-1a fingerprint makes unsupported source drift fail at build
time. The fingerprint is a drift check; verification records SHA-256 for
`cores.rs`, the selected entry, the existing host oracle, and the generated Rust source.
The generated source lives in `target/amdgcn-amd-amdhsa/release/build/.../out/q4k_core.rs`
and is copied to the evidence directory for inspection.

The AMD shim supplies signed non-saturating dot4, the 16/8/4/2/1 XOR reduction,
explicit rounded add/multiply/FMA and fault atomics. One coefficient is spelled explicitly:
`cdb = fma(8*d, sc, -(dmin*mi))`, the CUDA contraction already specified by the existing
`qdot` host mirror. The ABI wrapper replaces CUDA slices and launch annotations with
host-validated pointers; the selected row/slot/id relation is unchanged. No CUDA source
or normal engine owner is modified.

For `n_sb = K/256`, a column holds `256*ceil(n_sb/4)` packed Q8 code words, `8*n_sb` signed sums,
and `2*n_sb` scales. `host/src/selected.rs` packs the oracle's codes through the original
`q4_slot` permutation. Its unused tail words are poisoned. Selected slot `s` always uses
activation column `s`, including when two slots select the same expert. A HOST id
(`u32::MAX`) skips its output without fault. Other ids outside the expert stack leave the
output untouched and atomically set the earliest-layer/site word and the layer's ExpertId
mask. Faults remain sticky across launches.

The eleven synthetic K values are 256, 512, 768, 1024, 1280, 1792, 2048, 2304, 2816,
6144 and 20480; rows per expert rotate through 7/8/9. Each geometry runs nine launches:
three repeats, new ids, three invalid-id launches at layers 9/3/9, an isolated HOST skip,
and a refused-scale/zero/subnormal-column case. This covers partial super-block groups, paired
accumulations, incomplete row blocks, duplicate ids, changed ids, unused activation
columns, exact fault masks and NaN propagation. It does not exercise graph capture.
The real fixture's rows are viewed as four experts solely to test the selected operator;
this does not make the source tensor or model routed.

The selected experiment's host quantizer owns its non-finite refusal. The down kernel propagates
its NaN scale without claiming a GPU quantizer's QuantColumn fault. On-device Q8 activation
quantization and graph replay remain unimplemented; the eager forward above deliberately uses
F32 activations instead. Neither lane is connected to `Gpu`, `GpuModel`, an app session or
`bloomery-serve`; identical-file real-model llama.cpp HIP acceptance remains outstanding.

The verifier rejects an inherited `CARGO_BUILD_TARGET` and explicitly selects the
`rustc -vV` host triple for host tests, lint and build. Its executable comes from
that triple's output directory, so a Cargo config target cannot select an older
`target/release` binary.
