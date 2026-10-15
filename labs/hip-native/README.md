# Native Rust HIP feasibility lab

This isolated lab compiles Rust kernels for `amdgcn-amd-amdhsa`, loads their ELF through HIP, and checks them against Bloomery's scalar GGUF dequantizer with FP64 accumulation. It calls no libllama, ggml model forward, or llama-server process. Both host and device code are Rust. The nested workspace is excluded from the default CUDA workspace.

The validation target is one RX 7900 GRE (`gfx1100`), Rust nightly 2026-08-28, its LLVM backend and the local HIP runtime. Rust's AMDGPU target is tier 3: this is a feasibility result, not a supported release backend. The original CUDA model chains, seats, kernels, scheduling and MTP remain unchanged.

## What runs

- Device-buffer allocation and synchronized HIP transfers, plus a Rust GPU copy kernel with exact roundtrip equality.
- A scalar Q4_K × F32 matrix kernel on synthetic shapes and real `blk.1.ssm_out.weight` rows from Qwen3.8-27B IQ3_S. Matrix rows use GGUF Q4_K's 144 bytes per 256 weights. Scales and fused dequantization follow `crates/gguf/src/quant.rs`.
- A 256-wide synthetic dense FFN unit: RMSNorm, separate gate/up projections, SwiGLU, down projection and residual. Its intermediate values stay on the GPU. It does not include attention, GDN state, routed experts or a complete transformer block.
- Three repeated launches use already allocated buffers. The reported zero count covers allocations explicitly issued by this host, not internal allocations by the driver.

The fixed numerical acceptance is NMSE <= 1e-6 and finite outputs. Activation columns are F32: the original CUDA engine's Q8 activation scratch and quantized-dot order have not been ported. These checks do not establish CUDA/HIP bitwise equality or model-token equivalence. No throughput is claimed for the scalar kernels.

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

`result.json` records the clean HEAD commit, whole-repository and lab Git tree hashes,
release binary SHA-256 hashes, Rust/LLVM version, target CPU, actual device name, HIP runtime
and driver versions, fixture hash when present, command logs and each case. A failed check
returns nonzero and writes a failed result. Attach the JSON from the PR's exact head; a new
push requires a new run. Logs and generated artifacts stay out of the source tree.

The default run has three synthetic matrix shapes and one dense FFN case. The optional
fixture adds matrix-vector and 32-column cases. Use packed rows of a Q4_K tensor obtained
through the existing GGUF reader (`gguf::Split`/`TensorInfo`), with no conversion of its
weight bits. Record the original file SHA-256, tensor name and row range alongside the
result; do not upload model weights in the PR. Case results establish numerical operator
checks only, with no timing claim.

Every kernel uses a 64-thread workgroup, matching its host launch. Kernel signatures and
argument storage must remain identical. Context/module/stream/buffer handles stay on their
owning thread, and synchronization precedes destruction.

## Next native work

This lab is not connected to `Gpu`, `GpuModel`, an app session or `bloomery-serve`. The next
experiment is the existing Q4_K selected GEMV through a thin HIP shim, preserving its Q8
activation layout, logical warp operations, launch shape and sum order. The scalar F32
kernel here is not that port. Then gate one complete model forward against llama.cpp HIP
on identical model bits before designing a common backend boundary with the maintainers.
Attention/GDN, state reset/checkpoints, routed experts and media remain separate work.

The verifier rejects an inherited `CARGO_BUILD_TARGET` and explicitly selects the
`rustc -vV` host triple for host tests, lint and build. Its executable comes from
that triple's output directory, so a Cargo config target cannot select an older
`target/release` binary.
