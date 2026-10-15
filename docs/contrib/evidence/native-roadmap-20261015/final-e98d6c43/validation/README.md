# Dense Qwen diagnostic preparation

The clean source head is in source-identity.json. This is a follow-up to the
native dense frontend PR, not GPU serving or numerical acceptance evidence.

- Initial frontend: full T0 82 units and its host/static checks passed at its
  own head. Here the five changed T0 units passed, including both new Python
  self-tests, format, comments and recipe checks.
- A fresh separate build directory passed 118 gpu-gates host library tests
  (3 ignored) including both new row-writer tests, and warnings-denied static
  clippy for all touched frontend consumers. No CUDA executable was built here.
- A shared pre-existing Cargo target had returned an older116-test executable
  without the new module. That cached output is NOT used as follow-up evidence.
  The isolated test/clippy logs here were compiled from this worktree.
- The independent C++ harness compiled with warnings denied; six compiled host
  refusal/input contracts passed. Its build record identifies clean official
  llama.cpp fb4b2737, successful incremental library build and actual library hashes.
- Two clean CPU-only0.8B captures each contain3x248320 finite f32 rows. They use
  different original-llama prefill schedules, so the unequal bits/relative errors
  are diagnostics only. They are not Bloomery comparisons or an accuracy band.
- The supplied dense-inputs fixtures were tokenized from the exact full27B file
  using Bloomery's pure tokenizer, with explicit no-BOS/no-special parsing flags.
  Model, corpus, tokenizer binary and64/4096-token fixture digests are recorded.

Raw logit outputs are diagnostic data, not model weights. No weights, sparse
model headers, reference executables or shared libraries are included. A6000
native/reference logits, eager/graph and actual CLI/HTTP captures remain pending,
along with a registered dense reference family and measured acceptance bands.
