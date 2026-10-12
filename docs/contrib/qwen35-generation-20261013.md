# Dense Qwen generation through the native Qwen seat

The initial contributor draft was against `midagedev/bloomery` main at
`1cba970783dc97f735d4320d35f4cbae2ce5e935`; the follow-up rebases it onto
`eb186d979602a3e150967541ffb86bfea157866b` (verification below). It connects dense `qwen35`
files to the existing CUDA Body35 generation path. The contributor machine
has an RX 7900 GRE (gfx1100), not an NVIDIA GPU. Host checks passed; a CUDA
binary build, device loading, generated tokens and serving performance are
not verified here. This draft does not make the native server run on AMD.

## Change

The dense model reader and Body35 already exist. The generator and Qwen
serving seat previously admitted only the routed families. The generator
now accepts `qwen35`, and an explicit `bloomery-serve --model qwen3` can send
it through the same seat as `qwen35moe`.

The seat retains `SeatEngine`, `app::Session<Body35>`, the existing prompt
schedule, resident slots, checkpoint-based prefix reuse, placement and
whole-fit checks. No libllama/ggml inference dependency is added. Device
kernels, the app session, model readers and the HTTP server implementation
are unchanged.

The decode flash follows the existing `models::shape::GQA` table. The 27B
file has 24 query heads and 4 KV heads at width 256: its group of six selects
the pair kernel, whose pass is scalar only. `mma` is therefore false. The
existing group-of-eight and quad cases keep `mma` true. Load, placement and
listening records report the file's architecture instead of labeling a
dense file `qwen35moe`.

An explicit generative word is required for the dense file. With no word,
the existing decision-head selection remains: a decision model's HF card
can name its head, an in-file decision type owns its seat, and a plain
backbone with no head remains a named refusal. `--head` beside a generative
word remains refused. Clef/other decision layouts are not admitted as
generative architectures. The dense target does not gain MTP drafting.
The HF architecture checker mirrors the added name and the preserved
no-word decision rule, so its source-consistency check covers the new route.

## Actual file admission

The host-only `qwen35_admission` example reads the native Hparams, ModelSpec,
tensor roles and coverage table without opening a GPU:

```sh
RUSTFLAGS='-C target-cpu=native' cargo run -p bloomery-gpu-gates \
  --example qwen35_admission -- /path/to/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
```

The local file is 12,120,016,960 bytes with 866 tensors. Native metadata
reading reports:

```text
metadata arch=qwen35 layers=65 trunk=64 embedding=5120 heads=24 kv_heads=4 head=256 vocab=248320 experts=0 flash_mma=false
coverage unsupported=314 unused_tensors=15
```

The next-token layer is excluded from the trunk, and its 15 tensors are
unused. The 314 coverage entries are 313 tensor-format/role entries and one
missing tool-call parser for the template. They are coverage entries, not
314 separately measured runtime failures. The probe checks optional chat
features as well as the native text body.

For example, `token_embd.weight` is IQ2_S, `blk.0.attn_qkv.weight` IQ4_XS,
and dense FFN matrices include IQ2_XS/IQ2_XXS/IQ3_S. The file also carries
BF16, IQ1_M, IQ3_XXS and Q2_K sites absent from this body's admitted types.
Its Q4_K output head and successful metadata reading do not make the
remaining matrices runnable. The admission command ends nonzero, as
expected. No format gate or coverage rule was relaxed.

The tensor-format/role entries group as follows: IQ3_S 98, IQ4_XS 65,
IQ3_XXS 60, BF16 48, IQ2_S 15, Q2_K 13, IQ2_XS 8, IQ2_XXS 5 and IQ1_M 1.
These are coverage-entry counts, not raw tensor counts. Existing `gpu::iq`
code has plane/row cores for IQ2_XS, IQ3_XXS, IQ4_XS and Q2_K, making reuse
worth investigating before new kernels. IQ3_S in that module has a host
layout rule only. A kernel elsewhere does not admit it at a Body35 site:
the site's activation quantization, gains/gates, accumulation and wide
prompt path must all match and be gated separately.

The first device gate should use a file whose needed sites have admitted
formats (start with Q4_K matrices and F32 gains, checking the actual header),
on an NVIDIA card that holds the whole dense model and its stores. An
arbitrary file advertised as Q4_K_M is not automatically admitted. Supporting
the current mixed-IQ file requires separate per-site kernel/format work;
requantization would change the weight bits and needs its own quality check.

## Verification performed

- `cargo test -p bloomery-gpu-gates --lib`: 99 passed, 3 ignored. This
  includes two new dispatch/flash tests. No GPU feature was enabled.
- `cargo test -p bloomery-model --lib`: 252 passed, 4 ignored.
- `cargo test -p bloomery-serve --lib decide::tests`: 13 passed.
- Static `cargo check` and `cargo clippy --no-deps` of `bloomery-serve` and
  `generate_qwen3moe` with `glm5next,clef,vision,mimo2`: passed, no clippy
  warnings. The admission example was included in clippy.
- Formatting, comment, architecture, load ownership and recipe checks are
  recorded alongside the contributor evidence. The shape table's gate
  trigger row gains the three library-test gates reached by the new tests;
  no exception removes a reached gate.

Static checks follow the repository's host checking approach. They use the
pinned nightly, CUDA 13.3 headers and libclang 18, and build no executable
device bundle. They are not a `cargo oxide` build or a GPU gate. CPU flags
were `-C target-cpu=native` for the contributor's Intel host.

## NVIDIA validation still required

Build the native binaries with `cargo oxide`, never plain `cargo build`:

```sh
cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates \
  --features glm5next,clef,vision,mimo2 --release \
  --bin bloomery-serve --bin generate_qwen3moe
```

Run the generated affected gate list on the maintainer workstation before
landing. The existing Qwen3/Qwen3.6 and decision serving gates must remain
green. A separate dense-27B device clause is still needed: the current
`gate_qwen3_serve` assumes the routed files and their reference sets and
cannot be relabeled as a dense-model gate.

For the dense clause, first compare eager and graph logits at forced IDs
against an independent reference on identical admitted GGUF bits. Then
check HTTP versus the native CLI's `--last-step` path at short and 4K
prompts, sampling, reset, checkpoint reuse and simultaneous slots. Keep
MTP and media out of the first text clause; verify them separately if added.
Only after correctness passes should the repository's timing runners
measure native serving against llama.cpp and Strata under matched model
bits, placement, context and request concurrency.

## Rebase verification

The follow-up rebases this draft onto upstream main `eb186d979602a3e150967541ffb86bfea157866b`.
The Qwen seat retains main's common `SERVE_RULE`, arm screening and load phase timing. Its load
record still uses the file's actual architecture. The gate-path merge keeps main's added consumers
and the draft's shape test consumers. No kernel admission guard or numerical band was relaxed.

Host checks pass: 254 model tests, 106 gpu-gates library tests, five probe helper tests, 22 model-shape
tests and 56 decision tests. The two native frontends and admission example pass static check and
clippy with warnings denied. The real mixed-IQ 27B file still produces the expected admission
refusal: 314 coverage items and 15 unused MTP tensors. This is host/static evidence; CUDA codegen,
device loading, generated tokens and NVIDIA serving gates remain unverified.
