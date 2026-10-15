# Small native AMD forward candidate: Qwen3.5-0.8B

This is preparation for a future native Bloomery HIP whole-forward gate. No GPU
model load, generated token, serving test or timing was run. Existing services
were not changed. The initial dense27B contribution target is unchanged.

## Exact candidate acquired and verified

- Publisher: `notschmee/Qwen3.5-0.8B-GGUF`, card base model `Qwen/Qwen3.5-0.8B`.
- Revision: `1870603ac8ab59c7ddb9d94cccf3b297e379538f`.
- [Immutable primary file page](https://huggingface.co/notschmee/Qwen3.5-0.8B-GGUF/blob/1870603ac8ab59c7ddb9d94cccf3b297e379538f/Qwen3.5-0.8B-Q4_K_M.gguf).
- Full file: `Qwen3.5-0.8B-Q4_K_M.gguf`, 590,880,672 bytes.
- Full SHA-256: `be5b3cb8c2f3ebd791349c9c775ea38950f23a2b9e792ca2f0411597dd172988`.
  The computed digest matches the saved HF LFS metadata.
- Local complete file:
  `/home/jutai05/bloomery-native-model-candidates/notschmee/Qwen3.5-0.8B-GGUF/1870603ac8ab59c7ddb9d94cccf3b297e379538f/Qwen3.5-0.8B-Q4_K_M.gguf`.

`candidate-identity.json` records the complete-file verification and source URLs.
The file is neither requantized nor converted. Full file download was approved
after the extra Q8_0 requirement and clean shape/role admission were identified.

## Actual model contract

The GGUF architecture is `qwen35`, with no decision keys and no next-token/MTP
layers. Its 24 layers are all trunk layers; width1024, FFN3584, query heads8,
KV heads2, head width256, vocabulary248320, routed experts0. The GDN state size
is128, group count16, inner size2048, time-step rank16 and convolution width4;
every fourth layer is full attention. The native shape selector returns
`flash_mma=true` for this group-of-four geometry.

Actual 320-tensor inventory: F32 133, Q4_K 162, Q6_K 24, Q8_0 1.
The sole Q8_0 tensor is `token_embd.weight`, shape `[1024,248320]`.
There is no separate `output.weight`; the output head is tied to this embedding.
Consequently a native HIP implementation needs both Q8_0 row embedding lookup and
the Q8_0 output-head projection, in addition to Q4_K/Q6_K/F32 operators. This is
an explicit port dependency, not evidence that the current HIP lab can run it.

## Host verification

The existing `qwen35_admission` example was run first against a clearly named
sparse header diagnostic, then against the SHA-verified complete file. Both
returned exit1 and the same result:

```text
tensor_types={"F32": 133, "Q4_K": 162, "Q6_K": 24, "Q8_0": 1}
metadata arch=qwen35 layers=24 trunk=24 embedding=1024 heads=8 kv_heads=2 head=256 vocab=248320 experts=0 flash_mma=true
coverage unsupported=1 unused_tensors=0
unsupported layer=None a tool-call parser for this template
```

No unsupported shape or tensor-format/role entry was reported. Full coverage
still fails on the template's tool-call parser, which is preserved. This is
host metadata/coverage evidence, not device load or token correctness.

The binary source was the prepared dense contribution at
`00a54d4b60ff6831d283d81c6bb5e6c95c71ab70`, based on
`9fc2cc0d67f9a5b1d71e10663ba43873431c25e5`. The root agent subsequently rebased
that worktree; this result is deliberately attributed to its actual built
source. Binary SHA-256 before and after the complete-file run was
`af4b515c2a5eb31a91a78133980bf14f1fb261bc0faf906419f2380c9f7cb186`.
See `admission-binary-identity.json`, `candidate-full-admission.json` and
`candidate-full-admission.log` for the command and actual exit code.

## Bounded candidate screening

Three Q4_K_M headers were inspected at immutable revisions. All have `qwen35`
generation metadata with no decision keys; none contains only the three types
Q4_K/Q6_K/F32:

| Publisher and revision | Bytes | Actual inventory | Result |
| --- | ---: | --- | --- |
| [bartowski](https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF/blob/f36b1ea49a332ede8fe5f389bbf5b3575ef71f48/Qwen_Qwen3.5-0.8B-Q4_K_M.gguf), `f36b1ea49a332ede8fe5f389bbf5b3575ef71f48` | 579,615,840 | F32 176/Q6_K35/Q4_K95/Q8_0 17/Q5_K12 | Header only; also has one MTP layer |
| [notschmee](https://huggingface.co/notschmee/Qwen3.5-0.8B-GGUF/blob/1870603ac8ab59c7ddb9d94cccf3b297e379538f/Qwen3.5-0.8B-Q4_K_M.gguf), `1870603ac8ab59c7ddb9d94cccf3b297e379538f` | 590,880,672 | F32 133/Q4_K162/Q6_K24/Q8_0 1 | Complete candidate, digest verified |
| [lmstudio-community](https://huggingface.co/lmstudio-community/Qwen3.5-0.8B-GGUF/blob/26bab2c9369648924251c0ebb3dae012f5147707/Qwen3.5-0.8B-Q4_K_M.gguf), `26bab2c9369648924251c0ebb3dae012f5147707` | 527,502,816 | F32 133/Q6_K17/Q4_K152/Q5_K18 | Header only |

API metadata was also saved for ggml-org (no file ending `Q4_K_M.gguf` at the
queried revision) and unsloth; their weights/headers were not inspected. The
search stopped after the one additional established-publisher header above.

`inspect-candidates.py` requests at most32MiB per header and requires HTTP206.
The `.inventory.json` files record header digest, range, exact source and every
tensor shape/type. `.header-only.bin` and `.header-only.gguf` are diagnostics;
the latter are sparse files containing only the fetched prefix, with the rest
holes. They are outside model auto-discovery directories and must never be
used as a real model or uploaded as source/evidence. These prefix files may
contain some weight bytes after the header; publish only the JSON/log/Markdown
evidence if a later report is authorized.

## Next execution conditions

This candidate reduces the future full-forward model footprint from27B to0.8B,
but it does not remove the missing attention/GDN/head/backend work. The earlier
AMD resource reading had only about113.5MiB free with the user's Strata service
running. No GPU allocation was attempted. A full forward needs the required
native HIP operators plus a maintenance window or another card. Stage4 remains
pending; same-engine serving and stage5 performance measurements remain pending.
