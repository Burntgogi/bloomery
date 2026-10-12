# Native Rust HIP feasibility lab

This isolated lab compiles Rust kernels for `amdgcn-amd-amdhsa`, loads their ELF through HIP, and checks them against Bloomery's scalar GGUF dequantizer with FP64 accumulation. It calls no libllama, ggml model forward, or llama-server process. Both host and device code are Rust. The nested workspace is excluded from the default CUDA workspace.

The validation target is one RX 7900 GRE (`gfx1100`), Rust nightly 2026-08-28, its LLVM backend and the local HIP runtime. Rust's AMDGPU target is tier 3: this is a feasibility result, not a supported release backend. The original CUDA model chains, seats, kernels, scheduling and MTP remain unchanged.

## What runs

- Device-buffer allocation and synchronized HIP transfers, plus a Rust GPU copy kernel with exact roundtrip equality.
- A scalar Q4_K × F32 matrix kernel on synthetic shapes and real `blk.1.ssm_out.weight` rows from Qwen3.8-27B IQ3_S. Matrix rows use GGUF Q4_K's 144 bytes per 256 weights. Scales and fused dequantization follow `crates/gguf/src/quant.rs`.
- A 256-wide synthetic dense FFN unit: RMSNorm, separate gate/up projections, SwiGLU, down projection and residual. Its intermediate values stay on the GPU. It does not include attention, GDN state, routed experts or a complete transformer block.
- Three repeated launches use already allocated buffers. The reported zero count covers allocations explicitly issued by this host, not internal allocations by the driver.

The fixed numerical acceptance is NMSE <= 1e-6 and finite outputs. Activation columns are F32: the original CUDA engine's Q8 activation scratch and quantized-dot order have not been ported. These checks do not establish CUDA/HIP bitwise equality or model-token equivalence. No throughput is claimed for the scalar kernels.

## Build and run

Install the repository's pinned nightly and `rust-src`. No cuda-oxide invocation is used for this AMD target; none of the existing CUDA device crates is built here.

```sh
cd labs/hip-native
RUSTFLAGS='-C target-cpu=gfx1100 -C panic=abort' \
  cargo -Zbuild-std=core build --release --target amdgcn-amd-amdhsa \
  -p bloomery-hip-spike-device

export HIP_LIB_DIR=/opt/rocm/lib
export LD_LIBRARY_PATH="$HIP_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
RUSTFLAGS='-C target-cpu=native' cargo build --release -p bloomery-hip-spike
RUSTFLAGS='-C target-cpu=native' cargo test -p bloomery-hip-spike
RUSTFLAGS='-C target-cpu=native' cargo clippy --no-deps \
  -p bloomery-hip-spike --all-targets -- -D warnings

target/release/bloomery-hip-spike \
  target/amdgcn-amd-amdhsa/release/bloomery_hip_spike_device.elf
# Optional real Q4_K rows: exact packed bytes, K and row count.
target/release/bloomery-hip-spike \
  target/amdgcn-amd-amdhsa/release/bloomery_hip_spike_device.elf \
  /path/to/q4_k_rows.bin 6144 64
```

An SDK shipped outside `/opt/rocm` may need additional library directories. Each kernel uses a 64-thread workgroup, matching the host launch. Kernel signatures and argument storage must remain identical. All context/module/stream/buffer handles stay on their owning thread, and synchronization precedes destruction.

## Scope for an upstream design discussion

The experiment demonstrates a Rust device-code route that preserves the project's language choice. It does not yet connect AMD buffers to `Gpu`, `GpuModel`, an app session or `bloomery-serve`. A native backend needs a common contract for buffer windows, streams/events, kernel launches and capture, while retaining the existing model forward and expert placement owners. Compile-time selection should keep vendor dispatch out of the hot per-operator path.

Next operator work should match the original activation layouts and cover GDN/QSA, routed experts, PLE and state reset/checkpoint semantics. The CUDA path must retain its existing gates; AMD numerical and end-to-end gates must use AMD hardware. Wider hardware coverage, graph capture and performance tuning follow correctness. The whole-model libllama adapter is not part of this native backend proposal.
