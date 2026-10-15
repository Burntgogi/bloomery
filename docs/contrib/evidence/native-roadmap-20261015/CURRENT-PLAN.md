# Ordered continuation after GitHub authentication

Reference repository: midagedev/bloomery. Source baseline: e98d6c43.

1. Initial changes are submitted: [HIP lab #248](https://github.com/midagedev/bloomery/pull/248)
   and [native dense frontend #251](https://github.com/midagedev/bloomery/pull/251).
   Handle review findings; the dense attach still needs the agreed NVIDIA evidence.
2. The original selected Q4_K HIP port is [draft #252](https://github.com/midagedev/bloomery/pull/252),
   dependent on #248. Seven checks/18GPU cases pass; activation quantization is
   host side and whole-model equivalence remains unverified.
3. [Diagnostic preparation #254](https://github.com/midagedev/bloomery/pull/254)
   depends on #251. Fresh host119 tests, static clippy and five changed T0 units
   pass. Execute the supplied64/4096-ID fixtures on the maintainer A6000 after
   release0.3.3: independent reference, native forced rows, eager/graph, CLI/HTTP,
   reset/prefix and resident slots. Register the resulting dense reference family
   and derive numerical acceptance bands from that evidence. CPU0.8B reference
   smoke is not an A6000 or Bloomery pass. No local NVIDIA card/access is configured.
4. Implement the nine concrete slices in amd-forward-closure.md for the verified
   0.8B candidate, including GPU Q8 preparation, Q6/Q8 weights, GDN/attention,
   buffers and faults, then full forward and shared native resource boundaries.
   The card currently has only119037952 free bytes, insufficient for whole model
   loading. Plan a GPU window when the actual forward executable is ready; the
   existing Strata service has not been stopped. Session/SeatEngine integration
   requires both AMD whole-forward and preservation of the CUDA behavior.
5. Once native AMD serving works, run matched same-file/same-card llama.cpp HIP
   and Strata comparisons with the project's timing lease and prediction cards.
   Measure prompt512/4096, decode depths6/1024/4096 and concurrency, then implement
   one derived bottleneck change and rerun correctness/A-B. No speed result exists
   from the current operator/CPU diagnostic checks.

Evidence: final-e98d6c43/, dense-inputs/, status.json and amd-forward-closure.md.
Actual progress is recorded separately from these remaining runtime prerequisites.

Review corrections have been pushed to all four PRs. Initial HIP rejects target
overrides and uses explicit host artifact paths; the redundant thread marker is
removed. Dense generation uses one strict decision metadata reader and the common
Open35 configuration. Negative controls prove the metadata/divisibility refusals
and the actual GPU numerical check fail on their intended mutations. The current
initial dense/diagnostic host libraries pass117/119 tests. CI/review reruns are
pending; no approval or NVIDIA execution is asserted.

The0.8B pilot is not the original27B target. Before attempting27B on a16GiB AMD
card, use the real native planner's weights/state/KV/scratch budget; file size
alone does not establish fit. Additional format or dense placement work, if the
budget refuses it, remains separate from the initial three-type frontend attach.
