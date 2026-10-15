# Final source snapshot on 5e837723

The root status.json identifies each clean source head. Both initial contributions
were rebased again after the new encoder metadata landed. The dense gate-map
conflict keeps every upstream selector and the four added helper consumers.

HIP initial: seven command checks and six GPU cases passed. HIP selected follow-up:
seven checks and eighteen GPU cases passed, including twelve selected cases /108
launches with existing qdot oracle bit equality, exact fault/mask/sentinel behavior,
zero/subnormal inputs and NaN propagation. Activation quantization is still host
side. Neither result is native AMD model serving or a CUDA/HIP runtime comparison.

Dense: T0 total82 red0 wall264s; gpu-gates host116 and model host350 tests pass;
static check and warnings-denied clippy pass. Exact-file full admission still exits1
for the missing tool parser. NVIDIA runtime remains pending.

Root fmt/comments and parsed Cargo manifest comparison pass (only lab exclusion).
All changes in the selected follow-up relative to initial HIP are within the lab.

Bundles preserve each source head and require upstream5e837723. No weights are
included. The small-model identity next door records a separately downloaded and
SHA-verified0.8B candidate; its GPU forward has not run. Its host binary attribution
is the earlier9fc2cc0d base, explicitly preserved in that result.
