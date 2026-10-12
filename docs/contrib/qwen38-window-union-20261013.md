# Qwen Flash Next multi-row expert union

These are derived payload counts from the existing PR66 CPU teacher-forced input/prefill traces. They are not native Bloomery decode, speculative draft traces, measured file IO, or tokens/s.

Each corpus has 8192 rows over 48 layers, top-10 of 512 experts, reset every 2048 rows. Every overlapping window wholly inside a reset chunk is included: k1=8192, k2=8188, k4=8180, k8=8164. Placements keep an assumed id prefix of 0, 64, 128 or 256 experts on the card per layer. RAM cache admission and eviction are not simulated.

Bytes use the declared raw GGUF gate/up/down payload profile, with each layer weighted by its own expert size. They exclude scales, native arena layout/padding and sector alignment. One union member is counted once per layer/window; the denominator counts a copy per host selection in each of the window's rows. The existing trace hashes match the previously recorded corpus evidence.

| corpus | assumed card prefix | k2 union/repeated bytes | k4 | k8 | k4 bytes-only break-even accepted tokens/pass |
|---|---:|---:|---:|---:|---:|
| prose | 0 | 0.8316 | 0.6936 | 0.5709 | 2.774 |
| prose | 64 | 0.8320 | 0.6944 | 0.5720 | 2.777 |
| prose | 128 | 0.8337 | 0.6969 | 0.5750 | 2.788 |
| prose | 256 | 0.8331 | 0.6960 | 0.5740 | 2.784 |
| code | 0 | 0.8273 | 0.6730 | 0.5375 | 2.692 |
| code | 64 | 0.8271 | 0.6728 | 0.5374 | 2.691 |
| code | 128 | 0.8264 | 0.6719 | 0.5367 | 2.688 |
| code | 256 | 0.8264 | 0.6722 | 0.5378 | 2.689 |

At k4 the derived host-payload reduction is about 30.3–30.6% for prose and 32.7–32.8% for code. A four-row pass still reads roughly 2.77–2.79 single-row payload equivalents for prose and 2.69 for code. A speculative pass would need roughly that many accepted/emitted tokens just to break even on the bytes term under the same residency assumptions. This is not a total-latency break-even: compute, rejected draft routing, fixed weights, cache behavior and queue overlap are absent.

The requested k4 union term is now quantified for these input traces. An MTP default decision still requires native verify rows, real placement/arena misses and the same-binary MTP/off experiment. Worker-count IO measurements and these same-layer union counts describe different request sets, so they are not multiplied into a predicted inference rate.

Reproduce from the step1-main tree:

```sh
python3 tools/ref/window-union.py "$TRACE" --n-l 0-47:64 --k 1,2,4,8 \
  --ranges 0-7,8-15,16-23,24-31,32-39,40-47 \
  --payload "$EXPERT_PAYLOAD_JSON" --json "$RESULT_JSON"
python3 tools/ref/window-union.py --self-test
```

Known-answer checks cover unequal layer payloads (union48/repeated88/saved40), k1 identity, missing layer sizes, zero host demand, empty windows and strict JSON nulls. Existing text output without --payload is byte-identical to upstream main for the real prose trace.
