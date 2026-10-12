# Rebase, SSD worker reads and multi-row expert unions

Source repository: https://github.com/midagedev/bloomery, upstream main
`eb186d979602a3e150967541ffb86bfea157866b`. Source refs and verified fork heads are in source-links.json
and source-verification.json. The measured probe snapshot is recorded in nvread/metadata.json.
Earlier reported commits are retained; these are follow-up branches.

- [SSD report](nvread-report.md): 12 complete direct-only runs, 72 native records, three rotated
  worker-count rounds on WD PC SN740 NVMe with the actual IQ3_XXS split file. Native records, disk
  counter reconciliation and concurrent Strata CPU/IO observations are in nvread/. The failed default
  cache-mode attempt remains in nvread/failed-cache-mode/.
- [Expert union report](union-report.md): byte-weighted k1/k2/k4/k8 union counts from the existing
  8192-row CPU input traces, with reset boundaries and assumed card prefixes. Trace hashes match
  the prior evidence; per-layer payload accounting is separate from physical reads.
- Rebase checks in rebase/: model/shape/dispatch/decision and probe host tests, HIP host test and
  warnings-denied clippy, HIP device build, CUDA frontend static check and clippy. Both full recipe
  checks finish green. The first analyzer check's unchanged batch mock duplicate-call failure is
  preserved in analyzer-recipes-first.log; the rerun changes no clause, band or runner.

The existing GPU model, app/session, CPU model and HTTP implementation sources are unchanged.
No native serving throughput is measured here. The dense mixed-IQ file still has 314 admission items;
NVIDIA device execution and a post-rebase HIP numerical GPU rerun remain unverified.

JSON numeric values are preserved. Local path prefixes are normalized to `<local>`, line-end space
is removed, and full process stat strings are reduced to CPU/start ticks in the public copies.
sha256.json records original and public hashes. nvread-sweep.py is the exact measured helper script;
the reports provide the native build and lease commands. Full model hashes are the previous recorded
identities, not a new 76 GB hash pass. Weight payloads, binaries and corpus text are not included.

The IO rows carry concurrent CPU load and substantial between-round scatter. Use them as a
calibration under these conditions; a quiet repeat and native arena/verify measurements precede a
serving default decision. The existing default page-cache probe and its mincore guards remain intact.

These source and evidence branches are on the contributor fork. This follow-up creates no new
upstream pull request or issue comment.
