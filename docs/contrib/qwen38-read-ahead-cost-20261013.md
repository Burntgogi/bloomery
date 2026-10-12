# Qwen Flash Next read ahead payload cost

On PR66's CPU teacher-forced input traces, predicting the next layer's experts reaches about 69% recall at n=32, but starting a fixed number of reads at every position substantially increases NVMe payload traffic under every tested placement. This supports measuring wasted bytes before attaching that predictor to a serving engine. It is not a native Bloomery decode measurement or a measured inference speedup.

The analysis extends PR85's existing `router-coverage.py predict` with an explicit layer payload profile and JSON results. It retains the predictor, learn/held split, card/pool replay and recall definitions. The new counters use the same hits and candidate counts, weighted by each layer's bytes per expert. A supplied profile must name its layout and give a positive integer for every predicted layer.

## Conditions

The model is Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS (`qwen4exp`, 48 layers, 512 experts, top-10). Prose and code each contain 8192 positions, collected as CPU input evaluation with a context reset every 2048 tokens. The first 4096 positions learn the predictor and seed residency; the remaining 4096 are held out. These are the repeated, byte-verified traces of PR66, not new generated-answer traffic.

The three placements are assumptions: id-prefix card64 plus a pool128 LRU, id-prefix card128 plus a pool256 LRU, and learn-ranked card128 plus pool256 LRU. No claim is made that a production plan selects those counts. All pairs of consecutive layers, 1..47, are included.

Payload sizes come from the real GGUF's gate/up/down expert weight tensors: sum their sizes divided by 512 for each layer. They exclude scale tensors, sector alignment, runtime repacking and padding. They are not native arena bytes or measured physical reads. A native arena layout can be counted by supplying its own profile.

## Results

Recall@n across all expert selections, percent:

| corpus | n4 | n8 | n16 | n32 |
|---|---:|---:|---:|---:|
| prose | 22.01 | 36.36 | 53.30 | 69.77 |
| code | 22.46 | 36.51 | 52.98 | 69.31 |

The following traffic ratios are **derived** from held-out routing and the explicit payload profile. A useful read replaces a demand read. Every unused read adds bytes. Prefetch admission, queueing, cache pollution and compute overlap are not simulated.

| corpus | card / pool | n4 total/demand | n32 total/demand | n4 ahead bytes used % | n32 ahead bytes used % |
|---|---|---:|---:|---:|---:|
| prose | p64 / lru128 | 2.314 | 13.226 | 19.08 | 5.85 |
| code | p64 / lru128 | 2.778 | 16.687 | 13.11 | 4.17 |
| prose | p128 / lru256 | 6.100 | 44.252 | 7.42 | 1.86 |
| code | p128 / lru256 | 7.445 | 54.393 | 4.71 | 1.33 |
| prose | n128 / lru256 | 13.496 | 103.580 | 3.29 | 0.76 |
| code | n128 / lru256 | 13.568 | 103.436 | 2.50 | 0.66 |

The ratio grows when the pool already serves most demand: many positions start unused reads despite having no NVMe miss. High overall recall therefore does not establish that fixed-n read-ahead pays. These results reject adopting this fixed-n policy on this evidence. A budgeted, selective policy still needs a separate experiment, including native decode traces and the available lookahead time.

## File read calibration

Separately, 96 seed-11 expert samples read all three real GGUF weight slices with Linux O_DIRECT and aligned `preadv`, serial queue depth 1. On i5-12600K, 64 GiB RAM and the local model filesystem, payload was 167372800 bytes, aligned physical requests 168501248 bytes. The measured aggregate rate was 1.006 GB/s; median per-expert time 1.707 ms, p90 2.008 ms. Times include temporary aligned-buffer creation. Page cache is bypassed; SSD/controller caches are not reset.

This calibrates serial file reads only. It does not measure native arena loading, engine waiting, overlapped IO, or a maximum SSD rate. No inference latency is calculated from it. Running GPU services were stopped during this calibration and restored afterwards.

## Reproduction and checks

Use the PR66 trace collection and its model hashes in `router-locality-20261012.md`. PR85 base is `f110d061dfe49d1cef1b882c12d8795164a8a055`. The payload sidecar has this form, with real positive bytes for every predicted layer:

```json
{"source":"the exact expert layout being counted","bytes_by_layer":{"1":1868800}}
```

```sh
python3 tools/ref/router-coverage.py predict "$TRACE" \
  --n 4,8,16,32 --tiers p64,lru128 --tiers p128,lru256 --tiers n128,lru256 \
  --expert-bytes "$EXPERT_PAYLOAD_JSON" --json "$RESULT_JSON"
python3 tools/ref/router-coverage.py --self-test
```

The self-test retains PR85's recall and tier known answers, and adds hand-computed payload counters: demand80/ahead80/useful40/wasted40/total120 for n1, demand80/ahead160/useful80/wasted80/total160 for n2. It refuses missing layer sizes, nonpositive values, boolean values, noncanonical layer keys and an unnamed profile. Raw JSON results, payload tensor offsets, O_DIRECT samples, source commands and service restoration records remain in the contributor evidence bundle.
