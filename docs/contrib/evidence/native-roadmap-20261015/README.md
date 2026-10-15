# Native contribution roadmap and refreshed evidence

Base: `9fc2cc0d67f9a5b1d71e10663ba43873431c25e5`.

The HIP initial lab is commit `e203eb9ce3f45ff3222a301efa1a304ddf69341b`;
the dense frontend attach is `00a54d4b60ff6831d283d81c6bb5e6c95c71ab70`.
This evidence commit is not either implementation head.

Both bundles preserve their source commits and require the base above. In a
checkout containing that base, `git bundle verify /path/source.bundle` then
`git fetch /path/source.bundle HEAD:refs/heads/local-review` restores the source.
No model weights are included.

Initial HIP: seven command checks and six GPU cases passed. Root fmt/comments
passed; parsed root Cargo.toml equals the base except for the lab exclusion.
Dense: T0 82 units passed, 116 gpu-gates host tests and 342 model host tests
passed, static checks and warnings-denied clippy passed. Admission still exits
1 for the missing tool parser; no CUDA runtime validation is claimed.

The gate list uses `affected --no-box`: the local static dependency walk only,
without unavailable maintainer remote dep-info. It is not a narrowed GPU list.

`PLAN.md` gives ordered implementation and evidence criteria. `stages3-5-audit.md`
records the exact missing runtime/reference work and current hardware limits.
The original-kernel HIP experiment is a separate, later source change; these
initial six GPU cases do not establish its correctness or native AMD serving.
