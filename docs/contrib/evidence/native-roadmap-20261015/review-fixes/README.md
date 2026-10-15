# Review corrections and current verification

Initial HIP PR248: inherited CARGO_BUILD_TARGET is rejected before builds; host
checks/build/output use the explicit rustc host triple, including when Cargo
config contains a target. The redundant PhantomData/Rc marker was removed;
raw pointers still make Context neither Send nor Sync. Six GPU cases/seven
commands pass on the corrected clean head.

Negative controls are intentionally red: target override returns failure with
zero build commands; removing the K-multiple guard makes the newK257 fixture
fail; a separate local committed mutant doubles Q4 output and the GPU comparison
fails at NMSE0.999999753388892. The latter is not part of the contribution branch;
its one-line patch/result/source identity are preserved. Normal source stays green.
The original-Q4 follow-up retains18 passing GPU cases after inheriting these fixes.

Dense PR251: the shared model_file::decision_type reader rejects present non-string
metadata for both CLI and HTTP frontend before model loading. A minimal GGUF
header regression covers absent/string/empty/numeric markers. Restoring the old
permissive lookup makes it fail. CLI load options now reuse serve_open35.
The isolated host library117 tests and static clippy pass; model/parser numerical
bands and actual NVIDIA runtime remain unchanged/unverified. Parent full T0 was82
units; CI reruns it on the new push. Format/comments are checked on the fix.

Diagnostic draft254 inherits the strict reader/shared load owner. Its isolated
host library119 tests, static clippy and both Python self-tests pass. The earlier
CPU-only reference smoke remains valid historical data from unchanged helper/C++
bytes at9c5b3a5f, not a new CUDA or model accuracy result. Current source bundles
preserve all four corrected source heads; no weights or executables are included.
