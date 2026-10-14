# Native contribution source and validation handoff

Two implementation heads are prepared against `midagedev/bloomery` main
`ae09980c1754dfb01fbeff6f642b9c0c53ecf30f`:

| Request | Source head | Evidence |
|---|---|---|
| #87 standalone HIP lab | `f7cb7d92951a02aae985269f8d3da9d30d86673a` | [exact-head result](hip/result.json), [source patch](hip/source.patch) |
| #88 dense Qwen frontend/admission | `b511fb44d47ebb9823a36073c6bd93d0be916cc0` | [checks](dense/checks.json), [source patch](dense/source.patch), [model identity](dense/model-identity.json) |

The HIP lab passed host/device builds and lint, its host contract test and six
GPU cases on the RX 7900 GRE. It remains outside the CUDA workspace and does not
provide whole-model AMD serving. The scalar F32 operator is not the original
CUDA selected GEMV/Q8 activation port.

The dense attach retains the existing Qwen seat, native Body35, app Session,
common context/load planner, resident slots and checkpoints. It has host/static
evidence only. The exact Q4_K_M file's full SHA-256 was checked against HF; all
its tensor types are Q4_K/Q6_K/F32. Full coverage still returns nonzero for the
missing optional tool-call parser. The A6000 reference/forced-ID and HTTP-vs-CLI
lane remains to be run by the maintainer, as agreed in #88. No native serving
or speed result is claimed.

The final-head full T0 run passed: `DONE tier=t0 total=81 red=0 wall=253s`.
Both frontends and their shared planner gate callers pass static check and clippy
with warnings denied. The two host libraries pass 112 and 333 tests respectively.
The latest-main rebase preserved the frontend patch (`rebase-range-diff.txt`).
The HIP lab was rerun after that rebase, and all six GPU cases passed again.
Earlier T0 logs are kept as history: a missing derived gate row was corrected,
and an unmodified mock-server test's fixed artifact path was connected to the
configured cache through an ignored symlink. Those were resolved before the
final full T0 run; the complete coverage tool-parser refusal remains unresolved.

## Restore the exact source

Each `source.bundle` contains its implementation head, not the artifact branch's
head. It requires the base commit above, contains no model weights, and passes
`git bundle verify`. In an upstream checkout containing that base, download the
bundle and run:

```sh
git bundle verify /path/to/source.bundle
git fetch /path/to/source.bundle HEAD:refs/heads/contributor-review
git worktree add /path/to/review contributor-review
```

Use a different local branch/worktree for the other bundle. `source.patch` is
also readable/applicable, but `git am` can change the commit identity. The bundle
preserves the exact source commit and tree hashes recorded in the results.

## Publication status

Implementation PR publication was rejected by GitHub: the CLI OAuth token lacks
`workflow`, and syncing the old fork to current upstream also transfers upstream
workflow changes. No workflow edit is part of either implementation. This
artifact branch adds only source bundles, patches and evidence on top of an
already-published fork history; it is not a substitute implementation PR. A GitHub
authentication refresh is required before pushing the implementation branches.
