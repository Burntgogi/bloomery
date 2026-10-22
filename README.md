<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.svg">
    <img src="docs/assets/banner.svg" width="720" alt="bloomery">
  </picture>
</p>

<p align="center">
  <b>A fast inference engine for mixture-of-experts models, built for offloading.</b><br>
  Written in Rust from the HTTP server down to the CUDA kernels. A drop-in
  <a href="https://github.com/ggml-org/llama.cpp/blob/master/tools/server">llama-server</a>.
</p>

- **MoE models larger than the card, and larger than the host's RAM.** Routed experts live on the GPU and in host
  RAM; the engine counts its own routing and moves the experts it calls most onto the card while it runs. On one
  RTX 3090 24 GB beside a 256 GB host, DeepSeek-V4.1-Flash `Q3_K_M` (347 GB) serves two requests at 33.5 tok/s in
  total and GLM-5.3-Flash at 25.3 (0.3.0). When the RAM cannot hold the host's experts either, Qwen3.8, GLM-5.3-Flash and
  DeepSeek-V4.1-Flash page the rest from the model file on NVMe.
- **A prompt cache that survives other requests.** The Qwen3.8 and GLM-5.3-Flash seats keep saved prompt states in
  host RAM: a request that shares a prefix with an earlier one restores it, even after another request used the
  slot ([recorded on tape](https://tape.midagedev.com/r/u8fnay7pn4djqmy6vvaq)).
- **Several requests, one pass.** With `--parallel 2` two busy streams run through the model together. With the MTP
  draft off, that gave 25–37 % more tokens a second in total than one stream (0.2.1,
  [measured on the A6000](https://github.com/midagedev/rig-log/blob/main/log/2026-10-06.md#rel021-slots)). The draft
  is on by default and already fills the pass's rows: in 0.2.10's release check (A6000), two streams of Qwen3.8
  (`--place a`) and GLM-5.3 `UD-IQ4_XS` gave about the total of one stream.
- **llama-server compatible.** The same HTTP API (OpenAI and Anthropic), GGUF files and flag spellings: your client
  works unchanged.
- **Rust all the way down.** Every CUDA kernel is written in Rust and compiled with
  [cuda-oxide](https://github.com/NVIDIA/cuda-rust).

```sh
brew install midagedev/tap/bloomery        # Linux x86-64, NVIDIA sm_86+
bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M --port 8080
```

## Status (0.3.8)

0.3.8 runs more of the Qwen3 MoE experts on the card: Qwen3.6 and Qwen3-30B `Q8_0`, `Q6_K`, `IQ4_XS` and `IQ3_S` stacks
that stayed on the host now take the card leg. On one RTX 3090 at its 250 W cap, at the serving defaults (ctx 262,144),
Qwen3.6-35B-A3B `Q8_0` decodes at 89.60 tok/s (68.73 before). Raon-1-mini's prompt runs in wide units: pp512 2,507.5
tok/s (312.4 before). Nemotron-3.5-Lightning runs on the `nemotron` seat. Every row is in the
[release notes](https://github.com/midagedev/bloomery/releases/tag/v0.3.8).

| What runs | State |
|---|---|
| NVIDIA cards | Compute capability 8.6 or newer, driver R580 or newer (CUDA 13); an older driver is refused at the start. Run here: RTX 3090, RTX A6000, RTX 3060 (WSL2). Run by users: RTX 5090 ([#2](https://github.com/midagedev/bloomery/issues/2)), DGX Spark (GB10, from source: [`docs/contrib/dgx-spark.md`](docs/contrib/dgx-spark.md)). Ada (sm_89): no report yet |
| Host CPU | x86-64 with AVX2 (x86-64-v3), AMD or Intel. The release binaries are x86-64 only |
| OS | Linux x86-64 with glibc 2.34 or newer; Windows through WSL2. macOS (a Metal backend is in progress, [#593](https://github.com/midagedev/bloomery/issues/593)) and AMD GPUs are not supported yet |

### Known limits (0.3.8)

- DeepSeek V4.1 on two cards (`--place bp`) is not faster than one card (`--place a`). Unset `--place` picks `a`.
- The `nemotron`, `raon1`, `solar2` and `minimaxm3` seats serve one slot and keep no saved prompt states; `--parallel` above 1 is refused by name.
- `--place bp` is refused by name on the `nemotron` seat, Qwen3.6 and Qwen3-30B; they run one card plus the host tier.
- Nemotron-3.5-Lightning `UD-Q4_K_XL` is a placed load: its whole-card expert body is IQ2_XXS, and the other expert types stay on the host.
- IQ2_S and IQ1_M routed experts do not load.
- GLM-5.3 holds at most 16,384 positions a slot; a `--ctx-size` whose slot share is larger is refused at the start.

Models and what each needs: [`docs/models.md`](docs/models.md). Flags, seats and every limit:
[`docs/serving.md`](docs/serving.md#limits). To try it and judge it fairly, against llama-server too:
[`docs/evaluating.md`](docs/evaluating.md).

## Install

One binary, no CUDA toolkit and no Rust toolchain: Linux x86-64 with AVX2, glibc 2.34+, an NVIDIA GPU of compute
capability 8.6 or newer and driver R580 or newer. On Windows, run it inside WSL2 (`wsl --install`, reboot, then the same commands).

```sh
# Homebrew (Linux x86-64)
brew install midagedev/tap/bloomery

# or the installer: latest release, sha256-checked, symlink into ~/.local/bin
curl -fsSL https://raw.githubusercontent.com/midagedev/bloomery/main/tools/release/install.sh | sh

# or Docker (needs the host's NVIDIA Container Toolkit; model downloads live in the bloomery-cache volume)
docker run --gpus all -p 8080:8080 -v bloomery-cache:/root/.cache/bloomery \
  ghcr.io/midagedev/bloomery --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M

# or the plain tarball
tar -xzf bloomery-0.3.8-linux-x86_64-cuda-sm86.tar.gz && cd bloomery-0.3.8-linux-x86_64-cuda-sm86
bin/bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M --port 8080
```

The tarballs are on the [releases page](https://github.com/midagedev/bloomery/releases)
([0.3.8](https://github.com/midagedev/bloomery/releases/tag/v0.3.8)). From source: [`docs/BUILD.md`](docs/BUILD.md).

## Use

One binary, ten seats; the file's architecture picks the seat, and `--place` can stay unset on every model.

| Seat | Serves | One command |
|---|---|---|
| `ds41` | [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | `bloomery-serve --hf vcruz305/DeepSeek-V4.1-Flash-GGUF:Q3_K_M` |
| `glm` | [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | `bloomery-serve --hf unsloth/GLM-5.3-Flash-GGUF:UD-Q4_K_XL` |
| `qwen38` | [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | `bloomery-serve --hf unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL` |
| `qwen3` | [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (images with `--mmproj`), [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507), [MiniCPM-V 4.7 35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf) (images with `--mmproj` since 0.3.3), [Sage-A1.5](https://huggingface.co/InternScience/Sage-A1.5) (images with `--mmproj`) | `bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M` |
| `mimo2` | [MiMo-V2.6-Flash](https://huggingface.co/ggml-org/MiMo-V2.6-Flash-MOPD-GGUF) | `bloomery-serve --hf ggml-org/MiMo-V2.6-Flash-MOPD-GGUF:MXFP4` |
| `decide` | [Clef-Flash](https://huggingface.co/Cloudflare/clef-flash) and [lev](https://huggingface.co/ggml-org/lev-GGUF) (decision models) | `bloomery-serve --hf bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M` |

Four more seats serve a file on disk:
`nemotron` (Nemotron 3 Nano 4B, Nemotron 3.5 Bolt 74B-A9B, Nemotron-3.5-Lightning 30B-A3B), `raon1`
(Raon-1-mini-preview, with its DFlash2 draft file beside it), `minimaxm3` (MiniMax-M3) and `solar2` (Solar-Open2-250B
on two cards, `--place bp`). For example: `bloomery-serve -m PATH/raon-1-mini-preview-Q4_K_M.gguf`.
The table above lists the seats with a checked `--hf` row. The qwen38 seat also serves NVIDIA's NVFP4 checkpoint
folder as downloaded, with no conversion, on a card that holds every routed expert such as a DGX Spark's GB10:
`bloomery-serve -m PATH/Qwen3.8-Flash-Next-NVFP4` or `bloomery-serve --hf nvidia/Qwen3.8-Flash-Next-NVFP4`
([`docs/serving.md`](docs/serving.md#the-seats)).

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64}'
```

Claude Code: `ANTHROPIC_BASE_URL=http://localhost:8080`. The API, the flags, concurrent requests, drafts and
adaptive residency: [`docs/serving.md`](docs/serving.md). What each model needs in GPU, RAM and disk:
[`docs/models.md`](docs/models.md). In short, Qwen3-30B and Qwen3.6 run on a 24 GB card (or 12–16 GB, split onto
the CPU by itself); Qwen3.8 wants 128 GB of RAM, GLM-5.3 and V4.1 256 GB, and less RAM runs with the overflow
paged from NVMe.

## Numbers

**0.3.1's release check: one request at the serving defaults** (every lever unset, no `--parallel`,
`--ctx-size 16384`), greedy, thinking off, RTX A6000 (+ RTX 3090 for `bp`), one load per row, the release
candidate's `bloomery-serve`. Prompt sizes are the server's prompt tokens. The `a` row of Qwen3.8 is 0.3.0's check;
the Bolt row is 0.3.4's.

| Model, place | Long prompt: tokens, prefill tok/s, first token | Decode after it | Short prompt decode |
|---|---|---|---|
| Qwen3.8 `UD-Q4_K_XL`, `a` (0.3.0) | 5,673, 1,693, 3.38 s | 75.1 tok/s | 86.3 tok/s |
| Qwen3.8 `UD-Q4_K_XL`, `bp` | 5,673, 1,707, 3.35 s | 84.3 tok/s | 94.0 tok/s |
| Qwen3.6-35B-A3B `UD-Q4_K_XL`, unset (picks `a`) | 5,673, 7,083, 0.81 s | 115.0 tok/s | 117.5 tok/s |
| GLM-5.3-Flash `UD-IQ4_XS`, unset (picks `bp`) | 5,548, 172, 32.3 s | 35.1 tok/s | 35.0 tok/s |
| DeepSeek-V4.1-Flash `Q3_K_M`, unset (picks `a`) | 5,642, 403, 14.0 s | 41.9 tok/s | 42.7 tok/s |
| MiMo-V2.6-Flash `MXFP4`, unset (picks `a`) | 5,620, 104, 53.9 s | 34.8 tok/s | 36.1 tok/s |
| Nemotron 3.5 Bolt 74B-A9B `24GB`, unset (picks `a`) (0.3.4) | 5,680, 2,716, 2.10 s | 87.5 tok/s | 88.7 tok/s |

**Two requests at once** (`--parallel 2`, both busy) after a 512-token prompt, decode tok/s of both together, and
one request's decode at depth 4096. Measured on 0.3.0 (2026-10-12), on a 32-core AVX2 host with 8 DDR4 channels and
256 GB of RAM; the 3090 at a 250 W cap; the MTP draft off. Conditions and the earlier rows:
[`docs/performance.md`](docs/performance.md#conditions-of-the-numbers) and
[rig-log's bench page](https://github.com/midagedev/rig-log/blob/main/docs/bloomery-bench.md).

**One RTX 3090 24 GB**

| Model | Where it runs | Two requests, tok/s in total | Decode at depth 4096 | Prompt tok/s (P = 512 / 4096) |
|---|---|---|---|---|
| Qwen3-30B-A3B `Q4_K_M` | the whole model on the card | 252.5 | 164.1 | 6,496 / 6,801 |
| Qwen3.6-35B-A3B `Q4_K_M` | the whole model on the card | 257.3 | 181.8 | 5,766 / 6,685 |
| Qwen3.8-Flash-Next `UD-Q4_K_XL` | card + CPU | 64.3 ¹ | 51.5 ¹ | 502 / 664 ¹ |
| GLM-5.3-Flash `UD-Q4_K_XL` | card + CPU | 25.3 ² | 19.5 | 176 / 205 |
| DeepSeek-V4.1-Flash `Q3_K_M` | card + CPU | 33.5 | 26.9 | 195 / 376 |
| Nemotron 3.5 Bolt 74B-A9B `24GB` (0.3.4) | the whole model on the card | — ⁷ | 86.9 | 1,773 / 2,047 |

**RTX A6000 48 GB + RTX 3090 24 GB (`--place bp`)**

| Model | Two requests, tok/s in total | Decode at depth 4096 | Prompt tok/s (P = 512 / 4096) |
|---|---|---|---|
| DeepSeek-V4.1-Flash `Q3_K_M` | 56.1 | 39.0 | 233 / 398 |
| GLM-5.3-Flash `UD-Q4_K_XL` | 34.0 | 22.7 | 218 / 262 |
| Qwen3.8-Flash-Next `UD-Q4_K_XL` | — ³ | 90.5 | — / 1,777 |

¹ with `BLOOMERY_RESIDENCY=mid-p32-s1` · ² 29.1 with `BLOOMERY_RESIDENCY=mid-p0-s1` · ³ 0.3.0's measurement: two requests ran in
turn on an expert-tier load. 0.3.1 runs them in one pass; not re-measured here.
⁷ the nemotron seat serves one slot. Bolt's rows are 0.3.4's sittings (n=4, prompt at e92fa8bde, decode at 44495f0ba);
on one RTX A6000 the same sittings read 87.4 tok/s at depth 4096 and 2,143 / 2,772 prompt tok/s.

**One RTX 3060 12 GB** — one request at the serving defaults (`--ctx-size 16384`, every lever unset), 0.3.0's
release tarball, an i7-12700F desktop with 32 GB of RAM under WSL2, greedy, thinking off.

| Model | Long prompt: tokens, prefill tok/s, first token | Decode after it | Short prompt decode |
|---|---|---|---|
| Qwen3-30B-A3B `Q4_K_M`, card + CPU | 5,619, 103 ⁵, 54.4 s | 38.8 tok/s | 40.4 tok/s |
| Qwen3.8-Flash-Next `UD-Q4_K_XL`, experts paged from NVMe ⁶ | 5,673, 84.9, 66.8 s | 4.5 tok/s | 5.4 tok/s |

⁵ 0.3.0's measurement: the prompt ran one row at a time on this load. 0.3.1 runs it in wide units; not re-measured here.
⁶ 0.3.1's release candidate with no flags at all: the plan chose one slot of 28,881 positions and pages 75 GB of
experts from NVMe through a 19 GB RAM arena; a paged plan runs with the MTP draft off. 0.3.0 refused this load at
the defaults.

**Contributors' cards** — one request at the serving defaults, measured by
[@juliankang4](https://github.com/juliankang4) on their own hardware at 0.3.0, greedy, thinking off, 256 tokens.
Each row is their measurement on that card; the conditions, logs and tapes are in
[#46](https://github.com/midagedev/bloomery/issues/46).

| Card | Model | ~4K prompt: prefill tok/s, first token | Decode after it | Short prompt decode |
|---|---|---|---|---|
| RTX 5090 32 GB | Qwen3.6-35B-A3B `Q4_K_M`, whole on the card | 14,937, 276 ms | 345.2 tok/s | 356.1 tok/s |
| RTX 5090 32 GB | Qwen3.8-Flash-Next `UD-Q4_K_XL` ⁴ | 447, 9.12 s | 54.5 tok/s | 47.7 tok/s |
| DGX Spark (GB10) | Qwen3.6-35B-A3B `Q4_K_M`, whole | 4,576, 899 ms | 89.8 tok/s | 93.5 tok/s |
| DGX Spark (GB10) | Qwen3.8-Flash-Next `UD-Q4_K_XL`, whole, MTP draft on | 1,038, 4.11 s | 41.6 tok/s | 41.5 tok/s |

⁴ with `BLOOMERY_HOST_ROOM=56G` on 64 GB of RAM: 0.3.0 refuses this load at the defaults.

## Why it is fast

Each point has a same-engine measurement in [`docs/performance.md`](docs/performance.md#why-it-is-fast) or a
release's check:

- **Experts move to where they are used**: V4.1 decode 29.48 → 43.86 tok/s with adaptive residency.
- **The card and the CPU work on a prompt at once** (V4.1): two batches in flight, 1.20× at P = 4096.
- **A prompt's CPU experts move to the card** (Qwen3.8): each layer seats the experts the prompt uses most, and
  they stay for the answer; P = 4096 from 733 to 1,258 tok/s on the A6000.
- **Qwen prompts as int8 tensor-core GEMMs**: P = 4096 from 357 to 2,615 tok/s, then a prefill flash kernel.
- **MiMo's hot experts on the card**: decode after a 5,620-token prompt 19.1 → 34.0 tok/s (0.2.10 → 0.3.0 release
  checks, A6000).
- **CPU experts at memory speed**: AVX2 kernels on the stored quantization (Q4_K's switch to AVX-512 VNNI on a CPU
  that has it), grouped dispatch on a pinned pool.
- **One CUDA graph a decode step**, with the CPU experts inside it, and speculative decoding (MTP, DSpark).

Output is checked layer by layer against ik_llama.cpp (V4.1: PPL 2.2401 against 2.2378, same top token 97.46 %),
and every gate is shown to fail on its defect before the fix lands ([how](docs/performance.md#how-it-is-verified)).

## Contributing

Issues and pull requests are welcome, in Korean or English: [`CONTRIBUTING.md`](CONTRIBUTING.md)
([한국어](CONTRIBUTING.ko.md)). Two tracks we hope for are a new model architecture
([`docs/contrib/new-model.md`](docs/contrib/new-model.md)) and the NVIDIA DGX Spark, which runs since 0.3.0 and
serves one model across several Sparks since 0.3.4 ([`docs/contrib/dgx-spark.md`](docs/contrib/dgx-spark.md)). Starter work carries the
[good first issue](https://github.com/midagedev/bloomery/labels/good%20first%20issue) label; questions go to
[Discussions](https://github.com/midagedev/bloomery/discussions). Review looks first at code shape: one owner for
shared logic, no second copy, no silent failure.

## Docs

| Page | What it covers |
|---|---|
| [`docs/evaluating.md`](docs/evaluating.md) | A checklist to run it and compare it fairly; hardware reports |
| [`docs/serving.md`](docs/serving.md) | llama-server compatibility, flags, seats, concurrent requests, limits |
| [`docs/models.md`](docs/models.md) | Recommended files and what each model needs |
| [`docs/performance.md`](docs/performance.md) | Number conditions, why it is fast, verification, recorded sessions |
| [`docs/BUILD.md`](docs/BUILD.md), [`docs/cuda-oxide.md`](docs/cuda-oxide.md) | Building from source; the kernels in Rust |
| [`docs/upstream/`](docs/upstream/) | The toolchain-defect ledger; our upstream pull-request drafts are in [rig-log](https://github.com/midagedev/rig-log/tree/main/docs/bloomery/upstream) |
| [`docs/README.md`](docs/README.md) | Every other page and the design records |

Measurements and command lines: [rig-log](https://github.com/midagedev/rig-log) (Korean). Working contract:
[`AGENTS.md`](AGENTS.md), [`CONTRIBUTING.md`](CONTRIBUTING.md). AI-assisted pull requests are welcome, and pull
requests for hardware we do not have most of all.

## References and credits

[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) and ggml are the numerical reference: the kernels
were written by reading ggml's k-quant code and ik_llama.cpp's `iqk_mul_mat`, and accuracy is defined against
their output. [exllamav3](https://github.com/turboderp-org/exllamav3) and
[mistral.rs](https://github.com/EricLBuehler/mistral.rs) are design references, cited where used. All four are
MIT. The GPU kernels compile with [cuda-oxide](https://github.com/NVIDIA/cuda-rust) (Apache-2.0).

AI assistants helped write the code and the documentation. Every number here was measured by the runners in
`tools/ref/`.

## License

MIT, see [`LICENSE`](LICENSE). `crates/oxide-ice-unroll` carries NVIDIA's Apache-2.0 headers. Third-party
notices are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
