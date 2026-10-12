# Contributor SSD worker-count calibration

Measured with Bloomery's host-only `probe_nvread` on WD PC SN740 SDDPTQD-1T00, NVMe; model filesystem ext4, i5-12600K and 64 GiB RAM. The source snapshot is `ff39a36874728fa3c5c22691baa33f2d60fb9e54`; the binary SHA256 is `dcba9df1cad3369fb252afbd95557d6fef2b7b6afa443cbb19502e8e69cfda1a`. No model forward or GPU work is timed.

## Measurement conditions

Explicit `--direct-only`: each batch reads its expert parts once through the existing worker pool and aligned `DirectFile` O_DIRECT path. It neither populates nor evicts the page cache. This mode emits directN rows and, for each range batch, a contiguous drive reference after directN. The default cache/direct1/directN/drive experiment and its mincore gates are retained.

The initial default-mode attempt ended with the existing mincore residency refusal after k2, and its partial rows are invalid as a complete calibration. No other process mapping this IQ3_XXS directory was found by the subsequent maps inspection; the error alone does not establish the cause. The failure is retained in failed-cache-mode.

Three rounds use seeds 11071, 11072, 11073; thread orders 1/2/4/8, 2/4/8/1, 4/8/1/2. Each k1/k2/k4/k8 case has 32 counted batches and two discarded warmups, drawing one expert from each of k distinct layers, three weight parts per expert. Range [480,512) has four counted batches and two warmups from distinct layers. The same seed yields identical draws across worker counts within a round. Buffers are allocated and primed before the timed arms. Rates count useful payload bytes, in decimal GB/s. Table cells are medians of the three per-round batch medians, not a confidence interval.

Workers bound the number of blocking requests in flight; actual NVMe hardware queue depth is not measured. k1 has only three parts, so eight workers cannot keep eight requests outstanding. SSD/controller caches are not reset. The range drive arm rereads the bytes after directN and is a warmed reference, not a cold-drive maximum.

**Concurrent Strata CPU work is present.** Its observed interval CPU use spans roughly 199–1092% of one logical core. Device/probe byte reconciliation excludes large foreign reads but does not exclude CPU scheduling or shared memory pressure. These rows are a contributor calibration under recorded load, not a quiet-drive throughput claim or a serving speedup.

## O_DIRECT results

| expert batch | threads1 GB/s | threads2 | threads4 | threads8 |
|---|---:|---:|---:|---:|
| k1 | 1.314 | 1.692 | 2.435 | 2.364 |
| k2 | 1.380 | 2.183 | 2.574 | 2.776 |
| k4 | 1.339 | 2.127 | 2.711 | 3.361 |
| k8 | 1.275 | 2.275 | 3.188 | 3.645 |
| range | 2.113 | 2.569 | 3.658 | 4.131 |

For k8, the per-round eight-worker medians are 3.703, 3.645 and 2.567 GB/s, versus 1.275, 1.111 and 1.333 GB/s on one worker. The substantial between-round spread is retained. The observation supports checking the paged engine's IO concurrency; it does not choose a default worker count. The next performance decision needs a quiet repeat and native arena/verify measurements.

All 12 complete direct-only runs reconcile. Summed probe requests are 18773016576 B and the device reports 18778923008 B. Individual device/probe ratios span 1.000018..1.001079. Counters include warmups and the range reference reads. They count block-device requests, not NAND traffic.

The model is the recorded Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS qwen4exp split set. Its raw GGUF layout differs from the maintainer UD-Q4_K_XL layout and from a native packed arena. Existing recorded full model hashes are supplied separately; the entire 76 GB set was not rehashed by this calibration.

## Reproduction

Build the host-only probe with the repository pinned nightly and native host CPU flags, then use the existing lease-hold runner in contributor local mode:

```sh
RUSTFLAGS='-C target-cpu=native' cargo build --locked --release \
  -p bloomery-gpu-gates --bin probe_nvread
mkdir -p "$CONTRIBUTOR_LOCK_DIR"
BLOOMERY_LOCK_DIR="$CONTRIBUTOR_LOCK_DIR" BLOOMERY_REF_MODEL="$MODEL" \
  bash tools/ref/lease-hold.sh --card docs/cards/nvread-contributor.card -- \
  timeout --kill-after=10 180 target/release/probe_nvread \
  --direct-only --ks 1,2,4,8 --threads 8 --batches 32 --warmup 2 \
  --range-from 480 --range-batches 4 --seed 11071 --lease
```

Repeat the recorded worker orders and seeds. The contributor sweep preserves native records through tools/bloomery/records.py, disk counters and foreign process CPU/IO before and after each child. Keep the binary/model identities, the lease log, and the raw rows with any summary.
