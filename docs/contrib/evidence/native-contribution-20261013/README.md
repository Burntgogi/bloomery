# Contributor evidence: native AMD feasibility, routing costs and dense Qwen admission

These are recorded contributor results, with the exact source commits named in verification.json. They are not measurements of native AMD Bloomery serving or generated tokens from the dense CUDA draft. The code snapshots live on the contributor fork:

- [Routing byte-cost analyzer and report](https://github.com/Burntgogi/bloomery/tree/b948a27d5e76ed8852aa4b3bfb3445ae58f6a69a) (based on PR85, now merged).
- [Rust HIP lab](https://github.com/Burntgogi/bloomery/tree/3a17df85726f5cff6ee9428f1d93920478c355fc/labs/hip-native).
- [Dense Qwen frontend draft](https://github.com/Burntgogi/bloomery/tree/a98f4ee8e617277e21274572ef22ce134aa20a04).

## Contents

- flash-*.json: per-layer GGUF gate/up/down payload sizes, CPU input/prefill prediction results under assumed placements, and separate serial O_DIRECT read calibration. The profile excludes scales, alignment, repacking and runtime padding. The IO result includes temporary aligned-buffer creation, bypasses page cache and does not reset SSD/controller caches. No overlap or native inference latency is measured.
- hip-result.json: five Q4_K matmul cases (three synthetic, two using 64 real projection rows) and one synthetic 256-wide FFN unit. F32 activations, fixed NMSE <= 1e-6. This does not reproduce the CUDA Q8 activation contract. No model token or throughput claim.
- qwen27-*.json / qwen27-admission.log: native metadata/coverage on Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf at the recorded base: 64 trunk layers, 15 unused next-token tensors, 314 coverage entries (313 format/role entries plus one optional parser). Coverage entries are not a count of distinct missing kernels.
- verification.json and build/test logs: scope, toolchain, source commits and checks. The initial full recipe run found a source-mirror incompatibility that was corrected, plus a chain-only mock observing two launches where one was expected. The final run passed 542/542 batch self-tests and the full checker, without changing that runner or any tolerance.

Personal directory prefixes are replaced with `<local>`. The HIP result omits operational service names, and log endings are normalized to one newline. Numerical values, tensor metadata and binary/fixture hashes are unchanged. sha256.json records both original-file and published-file hashes. Raw model weights, extracted weight binaries and compiled device/host binaries are not included.

The source branches preserve the tested commits, rather than pretending to be rebased or tested against the newer main. The dense draft still needs rebase, a cargo-oxide binary build, and NVIDIA model/token gates before a merge-ready PR.
