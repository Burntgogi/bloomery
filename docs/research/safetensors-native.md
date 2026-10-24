# Safetensors native: one folder reader, vendor formats in place, and the MiniMax-M3.1-Flash track cut

Snapshot: worktree `bloomery-stdesign` at the lead's integration head `3c87b1dbe`. This is a **design round**: one file,
no code, no box time. The only bytes read from the real checkpoint are its 48 shard headers and a few scalars of
layer 3, fetched by HTTP Range (§9). Every byte figure below is [derived] from those headers unless it cites a file
of this tree or a rig-log record.

**The decision this design serves** (user, 2026-10-23): load the vendor's safetensors as shipped. No GGUF conversion,
no requantization. MXFP8 dense weights and NVFP4 routed experts reach the kernels bit for bit. The loader is a common
owner the next safetensors-first model attaches to with a map, not a MiniMax special case.

**What this supersedes.** `docs/research/minimax-m31-attach.md` (round `mm31design`, worktree
`bloomery-mm31design`) §6 plans a streaming convert to an IQ4_XS / Q8_0 GGUF and a local llama.cpp patch as the
oracle. The decision above retires both: no GGUF is written (§1), the experts stay NVFP4 (§3), and the oracle is the
vendor's own modeling code (§4). Its §5 reader split, §7 placement and §8 MSA-on-card stand and are referenced, not
repeated. Two premise corrections carried from there: MiniMax-M3 already has a card body (`crates/gpu-minimaxm3`),
seat, refset family and five gates; and the reader split is already written in worktree `bloomery-mm31reader`
(uncommitted, round `mm31reader`, grok), with the key `minimax-m3.attention.sparse_layers` rather than §5's
`attention.indexer.types`. This design writes that key.

Sources read: the tree; `~/.cache/bloomery/mm31/` (config, index, modeling file, download script); transformers
v5.12.1 (`quantizers/auto.py:96,124`, `quantizer_finegrained_fp8.py:207-275`, `integrations/finegrained_fp8.py:1042-1065`,
`conversion_mapping.py:458-560`, `models/minimax_m3_vl/configuration_minimax_m3_vl.py:125-155`,
`integrations/msa_attention.py`); compressed-tensors main (`compressors/nvfp4/helpers.py:367-415`,
`compressors/.../base.py:101-135`, `entrypoints/convert/converters/modelopt_nvfp4.py:87-121`); llama.cpp-upstream
`36a73916e` (`conversion/minimax.py:124-207`, `conversion/base.py:825ff, :958-1042`).

## 0. Paper decomposition and resource timeline

The design question has three costs: load time, decode time per token, and the work of the change. The terms:

| Term | Value | Source |
|---|---|---|
| Text bytes in the folder | 248,369,266,176 B (vision 1,734,145,024 B is in the same shards, never read) | §9 census |
| Routed expert bytes (NVFP4, codes + scales + scalars) | 232,381,393,920 B | §9 |
| Non-routed text bytes (MXFP8, BF16, F32) | 15,987,872,256 B | §9 |
| One expert, three matrices, one contiguous run | 31,850,496 B | §9 (128 of 128 checked on layer 3) |
| Active routed bytes per token (4 × 57 × 31,850,496) | 7,261,913,088 B | [derived] |
| Same at IQ4_XS (mm31design §7) | 6,858,473,472 B; native is +5.9 % | mm31design §7 |
| Host DRAM read | 135–137 GB/s fitted, STREAM 147.7 GB/s | rig-log `facts.md` |
| Host union MAC ceiling | 1.31 TMAC/s (fitted on another quant; [unverified] for NVFP4) | rig-log `plan.md` 「흐름 모델」 |
| NVMe read by page faults, one reader | 1.33 GB/s (474.18 GB in 356 s) | rig-log 09-23#v41-oracle-dump |
| NVMe device-class sequential | ~7.4 GB/s [derived there, not measured] | rig-log `plan-ledger.md` B3 |
| A6000 H2D | 26.28 GB/s pinned, 21.16 GB/s pageable | rig-log `facts.md` |
| Host plan room | 263,376,371,712 B | `workstation.rs` via mm31design §7 |

**Decode, one token, all experts on the host** [derived]. Host DRAM reads 7.262 GB: 53.0 ms at 137 GB/s, 49.2 ms at
147.7. Host MAC 12.910 GMAC: 9.86 ms at 1.31 TMAC/s, under the read. Card SMs read the always-on set (15.988 GB minus
the 2.458 GB embed, a host lookup) = 13.530 GB: 19.3–23.9 ms at 567–701 GB/s, in the host's shadow
(`shared_routed` overlaps it, mm31design §7). PCIe carries activations only, ~0.1 ms. **Wall = max = host DRAM:
18.9 tok/s at 137 GB/s, 20.3 at 147.7.** Spare card room holding experts lowers the host bytes; mm31design §7 keeps
it out of the prediction and so does this note.

The format choice does not move this term except by the byte ratio. Planar or interleaved, the host kernel reads the
same 0.5625 B per weight; its extra unpack instructions (§3) sit under a DRAM-bound row walk ("instruction count on a
kernel near the bandwidth roof: the model says 0", AGENTS.md change classes). So **the format decision is decided by
load time and memory, not decode**.

**Load, the real file, one start** [derived]. Per resource, with the plan of mm31design §7 (non-routed + KV on the
A6000, every routed expert on the host):

| Resource | In place (this design) | Repack at load to an anonymous copy |
|---|---|---|
| NVMe, cold | 248.4 GB once: 187 s at the measured 1.33 GB/s fault rate; ~34 s at ~7.4 GB/s with a parallel walk [unverified] | the same read, every start (below) |
| NVMe, warm restart | card bytes only if the page cache dropped them (`card_dontneed`); host experts 0 B | 232 GB again: the anonymous copy (232 GB) and the page cache (232 GB) do not both fit in 251 GiB, so every start is cold |
| Host CPU | normalize of card-bound bytes only (16 GB MXFP8, plus any card experts); host experts 0 B | + a rewrite of 232 GB (read + write = 464 GB of DRAM traffic, ≥ 3.4 s at 137 GB/s, more at a nibble-shuffle rate) |
| Host DRAM held | 232 GB of file pages, locked in place (`HostSet`) | 232 GB anonymous, and the page cache thrown out under it |
| PCIe | 16 GB + KV setup: ~0.8 s at 21 GB/s | the same |

Wall is NVMe-bound cold in both designs; warm, in-place pays the card bytes only (seconds) and the repack pays a cold
start every time (≥ 34 s, 187 s at the measured fault rate). That decides it: **host experts are read in place, in
the checkpoint's own layout.**

**The flow question first.** A unit of work here is one MoE layer of one token. The host and the card already
overlap (card attention and shared expert under the host's routed walk). Bulk: one walk per layer over the four
experts' runs, each run 31.85 MB contiguous, three planes in one run, so the prefetcher sees one stream per matrix
and no gather. SIMD: the host kernel is the MXFP4 kernel's twin with the e2m1 table (§3). Nothing in this design
shortens a term under another's shadow.

## 1. The reader: a virtual GGUF over the folder

**Recommendation: (a), the virtual GGUF, lifted into one common owner.** Every family reader in
`crates/model/src/arch/*` reads GGUF names and keys through `gguf::Split`. A reader trait both formats implement (b)
would make every family reader speak two vocabularies, or grow a translation layer in each family: the per-model
branch AGENTS.md forbids. The tree already took road (a) for Qwen3.8: `Split::header_only`
(`crates/gguf/src/lib.rs:611`) holds a synthetic header with every tensor and no bytes, and `model::part::Source`
(`crates/model/src/part.rs:127`) with `model_hf::part::Checkpoint` reads each card tensor from the folder, normalized
(`hf::modelopt::normalize`). llama.cpp's converter is the same map written in Python, so each map row has a line in
`conversion/minimax.py` to be checked against. What (a) lacks today is one thing: a tensor that stays where the
folder stores it and is read there by the host (the embed rows, the router, every host expert). §1.2 adds it.

### 1.1 The common owner

`crates/model-hf/src/folder.rs` (new) owns everything about a folder that is not a model fact. It is
`crates/model-hf/src/arch/qwen35moe/folder.rs` (593 lines) with its Qwen types taken out: `read` (`:86`), `open`
(`:140`), `plan_tensors` (`:267`), `served_as_hf` (`:363`), `size_of` (`:440`). The model facts move behind one
trait, which each architecture's map implements:

```rust
/// A checkpoint folder's model, as `folder::read` takes it.
pub trait FolderMap: Sized {
    /// `config.json`'s `model_type` values (the top level's, else `text_config`'s) this map reads.
    const MODEL_TYPES: &'static [&'static str];
    /// The config read and checked: geometry, and every field the map does not take refused by name.
    fn parse(config: &Json) -> Result<Self, Error>;
    /// One folder tensor's fate: a header tensor it feeds (`Mapped`, its sources, steps and place),
    /// or `Skip` with the reason (vision, a draft), never a silent drop.
    fn map(&self, name: &str, dtype: Dtype, shape: &[usize]) -> Result<Mapping, Error>;
    /// The header's model keys, from the config and the planned tensors.
    fn header(&self, planned: &[(&str, &[u64])], folder: &Folder) -> Result<Vec<(String, Value)>, Error>;
    /// The tokenizer's pre-tokenizer row and special ids (§2).
    fn vocab(&self) -> &VocabSpec;
}
```

`folder::read(dir)` reads `config.json`, picks the map by `model_type` (a type no map names is refused by name),
opens the shards (`safetensors::Folder`, which already maps every shard and answers `file_range`), plans the
tensors, builds the keys (the map's plus `hf::vocab::keys`), and checks the tokenizer against Hugging Face's.
`folder::open(dir)` returns the `Opened` it returns today (header, part source, stored rows, tokenizer), with the
header now a **located** split (§1.2). `crates/gpu-gates/src/bin/shared/serve_seats/mod.rs:96-98` stops hard-wiring
`qwen38_folder::open` for every directory: a directory goes to `folder::open`, and the seat is chosen from the
header's `general.architecture` like a GGUF's.

### 1.2 Located tensors: the host reads the folder in place

`Split::located(path, layout, files, places)` beside `Split::header_only` in `crates/gguf/src/lib.rs`. Its shards are
the folder's files, mapped as GGUF shards are; its header is the synthetic one. Each tensor carries a `Place`:

| Place | Meaning | `Split::data` | MiniMax tensors |
|---|---|---|---|
| `File { shard, range }` | the declared type's bytes, at that range of that shard | the mapped bytes | `token_embd` (BF16), `output` (BF16), `ffn_gate_inp` (F32), `exp_probs_b` (F32) |
| `Stack { shard, at: [u64; n_expert], per, layout }` | a stack whose matrix `e` is `per` bytes at `at[e]`, in `layout` | refused (a stack is read by matrix) | `ffn_{gate,up,down}_exps` (NVFP4_ROWS, `RowLayout::Planes`) |
| `Made` | bytes a step makes (a cast, a +1, a normalize) | refused by name; the part source makes them | norms (+1, F32), MXFP8 projections, the stacks' `.scale` |

Because the shards are the folder's files, `HostSet` (`crates/model/src/placement/host_lock.rs:73`) locks host
segments as page ranges of those files with no change: its `HostFile::Shard(s)` is shard `s` of the split, now a
`.safetensors` file. `ExpertStack` (`crates/model/src/ops.rs:1237`) gains `located(info, span, at)`: matrix `e` is
`span[at[e]..at[e] + per]`, where today it is the `e`-th equal run of one span. The NVMe tier (`StackTier`) is
unchanged: an arena slot holds a matrix's `per` bytes in the same layout. The embed table (`EmbedTable` over
`Split::data`) and the router read their `File` bytes unchanged. The precedent for a table read where the folder
stores it is `model::part::StoredRows` (`part.rs:120`, Qwen3.8's PLE table); `Place` generalizes it to every host
tensor.

What stays refused: `Split::data` on a `Made` tensor, a stack whose experts span two shards (none does here: every
layer lives in one shard, §9), and residency over a folder until phase 2 (`body38.rs:1306` `refuse_folder` is the
precedent; §6 track `stcardraw`).

### 1.3 The MiniMax map

`crates/model-hf/src/arch/minimaxm3.rs` (new), the rows of `conversion/minimax.py` `MiniMaxM3Model` (`:169-207`):

| Folder (prefix `language_model.model.`) | Header tensor | Place / step |
|---|---|---|
| `embed_tokens.weight` BF16 [200064, 6144] | `token_embd.weight` BF16 | File |
| `language_model.lm_head.weight` BF16 | `output.weight` BF16 | File |
| `norm.weight`, `layers.L.{input,post_attention}_layernorm.weight`, `self_attn.{q,k,index_q,index_k}_norm.weight` BF16 | `output_norm`, `blk.L.attn_norm`, `ffn_norm`, `attn_q_norm`, `attn_k_norm`, indexer norms, F32 | Made: f32 + 1.0 (the Gemma norm; `minimax.py:200-203` adds 1 to every `norm.weight`) |
| `self_attn.{q,k,v,o}_proj.weight` F8_E4M3 + `.weight_scale_inv` U8 [N, K/32] | `attn_{q,k,v,output}.weight` MXFP8 | Made: `Format::Mxfp8` normalize (phase 2: File planes, §3.3) |
| `self_attn.index_{q,k}_proj.*` | the indexer tensors the M3 reader names (`names.rs`) | Made: MXFP8 (llama.cpp forces F32 here, `tensor_force_quant`; we keep the vendor's MXFP8) |
| `mlp.{gate,up,down}_proj.*` (layers 0–2) | `ffn_{gate,up,down}.weight` MXFP8 | Made: MXFP8 |
| `block_sparse_moe.shared_experts.*` | `ffn_{gate,up,down}_shexp.weight` MXFP8 | Made: MXFP8 |
| `block_sparse_moe.gate.weight` F32 [128, 6144] | `ffn_gate_inp.weight` F32 | File |
| `block_sparse_moe.e_score_correction_bias` F32 [128] | `exp_probs_b.bias` F32 | File |
| `experts.E.{w1,w3,w2}.weight_packed` U8 + `.weight_scale` F8_E4M3 | `ffn_{gate,up,down}_exps.weight` NVFP4_ROWS [k, n, 128] | Stack, `Planes`, `at[e]` from the shard header |
| `experts.E.wN.weight_global_scale` F32 [] | `ffn_*_exps.scale` F32 [128] = 1 / g | Made (read whole, 175,104 B of scalars a file) |
| `experts.E.wN.input_global_scale` F32 [] | none | read, checked finite and > 0, counted in the load line, unused (§3.4) |
| `patch_merge_mlp.*`, vision tower, `dspark/` | none | Skip("vision"), Skip("draft"): named in `--plan`, never silent |

Keys from `config.json` `text_config` (the rows `MiniMaxM3Model.set_gguf_parameters` writes, `minimax.py:177-198`):
`general.architecture = minimax-m3`, block_count 60, embedding_length 6144, feed_forward_length 12288 (dense),
expert_feed_forward_length 3072, expert_shared_feed_forward_length 3072, expert_count 128, expert_used_count 4,
expert_shared_count 1, expert_gating_func sigmoid, expert_weights_scale 2.0, expert_weights_norm true,
attention.head_count 64, head_count_kv 4, key/value_length 128, layer_norm_rms_epsilon 1e-6, rope.freq_base 1e7,
rope.dimension_count 64, `leading_dense_block_count` = the zero prefix of `moe_layer_freq` (3), the indexer keys
(index heads 4, index dim 128, top-k 16, block 128, local 1), and `minimax-m3.attention.sparse_layers` =
`sparse_attention_freq` (60 ones; the `mm31reader` key), swiglu alpha 1.702 and limit 7.0. A config value the
graph does not implement is refused by name (`hidden_act` other than `swigluoai`, `use_gemma_norm` false,
`attention_output_gate` true, `qk_norm_type` other than `per_head`, `use_routing_bias` false, `scoring_func` other
than `sigmoid`, `sparse_score_type` other than `max`, `sparse_init_block` other than 0, `sparse_disable_index_value`
not all 1, `tie_word_embeddings` true, any MTP field), and so is a field the map does not know, so a sibling
checkpoint with a moved fact fails at open, not in a gate. The map is picked by the top-level `model_type`
(`minimax_m3_vl`).

### 1.4 Identity, fixture tier, fetch

- **Refset identity** is the first shard's full path, as for a split GGUF: `Split::located`'s `path` is
  `<dir>/model-00001-of-00048.safetensors` (the index's first file by name). A set dumped from the folder names that
  path in `# model`; a set from the M3 GGUF is refused against it by the family check.
- **Fixture tier.** `tools/fixture-dirs.tsv` gains `minimax-m3 → minimax-m31`; `crates/refset/src/fixture.rs`
  `first_shard_in` (`:315`) accepts a directory holding `model.safetensors.index.json` and returns its first shard.
  The `# fixture` line reads the `bloomery.fixture.*` keys, which the cutter (§5) writes into the first shard's
  `__metadata__` and `Split::located` surfaces as header keys.
- **`--hf MiniMaxAI/MiniMax-M3.1-Flash`** already fetches a folder: `hf::checkpoint_files` (`hf/src/lib.rs:409`)
  takes the top-level index, every `.safetensors`, `.json` and `.jinja`, and `Client::resolve_drafts`
  (`hf/src/fetch.rs:627`) returns the folder when the repo has no GGUF set. `dspark/` is not fetched (a subfolder),
  and the vision bytes ride in the text shards (1.73 GB, unavoidable). The only change is downstream: the folder it
  returns now opens through `folder::open` for any mapped architecture.

### 1.5 Change table

| File:line today | Today | After |
|---|---|---|
| `crates/model-hf/src/arch/qwen35moe/folder.rs:86,140,267,363,440` | the folder reader, typed on Qwen's `Checkpoint` | moves to `crates/model-hf/src/folder.rs` over `FolderMap`; Qwen keeps its map (`qwen35moe.rs` `parse :299`, `map :628`, `header :677`) as `impl FolderMap` |
| `crates/gguf/src/lib.rs:611` `Split::header_only` | a header with no bytes | beside it `Split::located` (§1.2); `header_only` stays for the `prepare` path |
| `crates/gpu-gates/src/bin/shared/serve_seats/mod.rs:96-98` | every directory is Qwen3.8 | `folder::open`, seat by `general.architecture` |
| `crates/model/src/ops.rs:1124` `RowLayout {Rows, R8}` | two layouts | + `Planes`: a matrix's codes for every row, then its block scales |
| `crates/model/src/ops.rs:1237` `ExpertStack` | matrix `e` = `e`-th equal run | + `located(info, span, at)` |
| `crates/model/src/moe.rs:2853` `stacks_of` | one run per stack | reads the stack's `Place` |
| `crates/hf/src/quant_config.rs:305-318` | `quant_method` None/modelopt/fp8/compressed-tensors; `mxfp8` refused | + `"mxfp8"`: MXFP8 on every weight not in `ignored_layers`, and `moe_quant_algo NVFP4` + `moe_quant_format nvfp4-pack-quantized` → `Format::Nvfp4Ct` on the experts |
| `crates/hf/src/quant_config.rs:47` `Format` | NVFP4 ModelOpt and csf scales | + `Nvfp4Ct`: `weight_packed`, `weight_scale`, `weight_global_scale` a divisor, `input_global_scale` |
| `crates/hf/src/modelopt.rs:117` `type_of` / `normalize` | ModelOpt names, `weight_scale_2` a multiplier | the `Nvfp4Ct` arm shares the nibble move; tensor scale = 1 / g |
| `crates/hf/src/vocab.rs:483, :709` | `tokenizer.ggml.pre` is always `qwen35` | the map's `VocabSpec` names the row (§2) |
| `crates/gguf/src/qtype.rs:167` NVFP4_ROWS | `host_act None`, `Card::NONE` | `host_act Q8_2X4`, NVFP4's card columns (§3.2) |
| `crates/gguf/src/quant.rs:263` `reads_scale` | NVFP4_ROWS: one scale a tensor | one a tensor, or one an expert of a stack (as NVFP4) |
| `crates/qdot/src/lib.rs:8246` | MXFP4 row dot and tile | + the NVFP4 planes twin (§3.2) |
| `crates/placement/src/accepts.rs:512-524` | the common leg (mimo2, solar2, minimax-m3) admits `card_gate_up_runs` / `card_q8act_down_runs` | + the planes arm (`kernels.rs:287` `card_planes_runs`) for a body whose leg buffers hold the 32-value down form |
| `crates/gpu-minimaxm3/src/body.rs:452` `load` | GGUF bytes (`load_placed_with`) | `CardBytes::Parts` for a folder (`gpu/src/model.rs:2105`, `load_placed_from :789`); leg buffers in both down forms |
| `crates/refset/src/fixture.rs:315`, `tools/fixture-dirs.tsv` | GGUF first shards | + a folder's first shard; the `minimax-m3` row |

## 2. Tokenizer and chat

The folder's `tokenizer.json`, `tokenizer_config.json` and `generation_config.json` are the same git blobs as
MiniMax-M3's (`tokenizer.json` oid `9065e14e…`, 9,731,500 B). BPE, 200,000 vocabulary entries plus 61 added tokens
(ids 200000–200060, 54 special), 199,744 merges, an NFC normalizer, a `Split` pre-tokenizer with the GPT-4o-style
pattern then `ByteLevel`: our `Pre::MiniMaxM2` (`minimax-m2`). bos 200019 `]~b]`, eos 200020 `[e~[`, pad `]!p~[`.

- **`hf::vocab`** already turns `tokenizer.json` into GGUF's `tokenizer.*` keys, with NFC handled, but writes
  `tokenizer.ggml.pre = qwen35` unconditionally (`vocab.rs:483`) and checks for it (`:709`). The map's `VocabSpec`
  names the pre row; `vocab` checks the `Split` pattern in `tokenizer.json` against that row's pattern byte for byte
  and refuses any other by name. eos ids come from `generation_config.json`.
- **Chat.** `chat_template.jinja` (11,633 B) differs from M3's (11,756 B): `current_date` / `current_location`
  variables, `<model_id>8bb8149fe</model_id>`, `reasoning_effort` / `<effort>`, a mid-conversation
  `system`/`developer` turn rendered as `developer`, adaptive thinking always (no `thinking_mode`). Its filters and
  calls are the set M3's uses (`tojson`, `strip`, `split`, `items`, `namespace`, macros, `raise_exception`; a diff of
  the two sets is empty), so `crates/jinja` renders it if it renders M3's. The header carries the text as
  `tokenizer.chat_template`; `chat_of` (`crates/model/src/arch/mod.rs:801`) reads it as today. The reasoning span
  stays `<mm:think>` … `</mm:think>`.
- **Byte-identity plan.** Three checks, cheapest first:
  1. the keys `hf::vocab` derives from the folder equal the M3 GGUF header's `tokenizer.*` keys (tokens, types,
     merges, pre, special ids), compared value by value; the GGUF header is 0.01 % of its first shard and can be
     fetched by Range for a Mac test;
  2. the folder's tokenizer encodes the texts behind `tools/ref/data/minimaxm3/{a,b,c}.ids` to those ids exactly
     (they are llama-tokenize's on the same blob), and Hugging Face `tokenizers` agrees on the same texts;
  3. the template renders byte-identical to `jinja2` / `apply_chat_template` (transformers 5.12.1) over a table of
     conversations: plain, system, developer mid-turn, tools, reasoning effort, a date and location set.

## 3. Native formats and the kernels

### 3.1 Byte layouts

| Source tensor kind | Bytes | Value |
|---|---|---|
| NVFP4 `weight_packed` U8 [n, k/2] | byte `i` of a row: value `2i` in the low nibble, `2i + 1` in the high (`pack_fp4_to_uint8`, `helpers.py:367-415`) | e2m1 × `weight_scale` / `weight_global_scale` |
| NVFP4 `weight_scale` F8_E4M3 [n, k/16] | one byte a 16 values, sign bit 0 (the checker refuses bit 7 and 0x7F) | UE4M3 |
| NVFP4 `weight_global_scale` F32 [] | one a matrix; w1 = w3 for every expert of layer 3, 8,548–16,887 (w1/w3) and 4,248–18,229 (w2) | a **divisor** (`modelopt_nvfp4.py:87-121`: "1 / weight_scale_2 → weight_global_scale") |
| NVFP4 `input_global_scale` F32 [] | 1.0 for every expert of layer 3 | the activation's FP4 tensor scale in W4A4 |
| MXFP8 `weight` F8_E4M3 [n, k] | one byte a value | e4m3 × 2^(s − 127) |
| MXFP8 `weight_scale_inv` U8 [n, k/32] | one E8M0 byte a 32 values (a multiplier despite the name, `Fp8DecodeScale`) | |
| BF16 embed, head, norms; F32 router and bias | as declared | |

Within a shard the data is sorted by dtype width, then name. So a layer's F32 scalars come first, then BF16, then the
1-byte tensors, and one expert's six 1-byte planes form one run of 31,850,496 B:
`[w1 codes 9,437,184 | w1 scales 1,179,648 | w2 codes 9,437,184 | w2 scales 1,179,648 | w3 codes | w3 scales]`.
Each matrix is its codes then its scales: a matrix in `RowLayout::Planes`. Experts lie in string order
(0, 1, 10, 100, …), so `at[e]` is a table read from the header, not a stride. Data starts are 8-byte aligned only
(mod 64: 39 shards at 32, 2 at 24, 6 at 8, 1 at 40), so a kernel loads unaligned (it does today: `_mm_loadu_si128`).

### 3.2 Kernel sites

| Site | Tensor | Format | Entry today | Gap |
|---|---|---|---|---|
| card dense, decode | attn q/k/v/o, indexer q/k, shared, dense FFN | MXFP8 → `Mxfp8Planes` | `mxfp8_gemv` / `mxfp8_gemv_mcol` (`site.rs`) | none |
| card wide, prompt | same | MXFP8 | `site::gemm` → `mxfp8_tile_gemm` (`gemm/kernels32.rs:1106`); `site::gated` with `Act::SwigluOai` | the MiniMax body's MXFP8 shared/dense arm (track `mm31body`) |
| card router, head, embed | F32, BF16 | as stored | F32 dense, BF16 head/embed columns (`qtype.rs:67, :128`) | none |
| card routed experts | NVFP4 → `Nvfp4Planes` | card-held experts | `nvfp4_gate_up_sel` (act a launch argument), `nvfp4_gemv_sel32` (down over `Q8Blocks32`), `gemm_nvfp4p` | MiniMax's leg buffers hold `Q8Act` downs only (`kernels.rs:277` `card_q8act_down_runs`); the NVFP4 down reads `Q8Blocks32` |
| host routed experts | NVFP4_ROWS, `Planes`, in place | every host expert | none | the AVX2 kernel below |

**The host kernel** is a twin of `dot_mxfp4_q82x4_avx2` (`qdot/src/lib.rs:8246`), against the same `Q8_2_X4`
activation form, so no new activation form. They differ in four places:

1. **codes**: 16 code bytes → `lo = b & 0xF`, `hi = b >> 4`, then `unpacklo/hi_epi8(lo, hi)` puts 32 values in
   natural order (pair order needs two unpacks where MXFP4's split order needs none); a `pshufb` of the signed
   doubled e2m1 table {0, 1, 2, 3, 4, 6, 8, 12, 0, −1, …, −12} (MXFP4's own values);
2. **sign**: `maddubs(sign(w, w), sign(a, w))` instead of MXFP4's offset table and per-32 minimum, because NVFP4's
   scale changes every 16 values and `Q8_2_X4` carries one activation sum per 32. A pair peaks at 2 × 12 × 127 =
   3,048: no i16 saturation;
3. **scales**: two UE4M3 bytes per 32 values; f32 from `cvtepu8_epi32`, `<< 20`, cast, × 2^120 (exact for normals and
   subnormals with FTZ/DAZ off); after `madd` with ones, i32 lanes 0–3 are the first 16 values and 4–7 the next, so
   one 8-lane scale vector multiplies one `cvtepi32_ps`;
4. **tensor scale**: 1 / g, once per row at the end (the epilogue), where MXFP4 has none.

The row walk reads two forward streams (codes, scales), each prefetcher-friendly. A column tile (`<const C>`, as
`dot_mxfp4_q82x4_tile_avx2`, `:8371`) serves the prompt path. Integer partial sums are exact; the f32 order across
groups is the kernel's, pinned by its gate ("this order is the gate").

**qtype rows.** No new type id. The type says how a row encodes values; the layout says which bytes a row is
(`ops.rs:1121-1123`). NVFP4_ROWS (2002, `qtype.rs:167`) is already ModelOpt's encoding: groups of 16, UE4M3, pair
order, a tensor scale. It gains `host_act: Q8_2X4` and the card columns NVFP4 (40) has (`:139-140`:
`Nvfp4Planes Nvfp4Planes Nvfp4 (Nvfp4, Q8Blocks32) Nvfp4 Nvfp4`), so `card_planes_runs` admits it and the card loads
it through the part source. The checkpoint's matrix-planar arrangement is `RowLayout::Planes`, not a type. MXFP8
(2001) needs no row change for the critical path.

### 3.3 Repack or planar: the decision

| Where | Decision | Load cost | Kernel cost |
|---|---|---|---|
| host experts (≈ 180–232 GB) | **read in place**, `Planes`, pair order | 0 B rewritten; HostSet locks file pages | +2 unpacks per 32 values, under the DRAM roof (§0) |
| card experts (spare room, 0–~45 GB) | phase 1: the existing CPU nibble move into `Nvfp4Planes` (`Nvfp4Ct` shares ModelOpt's); phase 2: raw H2D of the file bytes + a card permute kernel (pair → GGUF nibble order) | phase 1: one CPU pass over card-bound bytes [rate unverified]; phase 2: 2 × bytes at ~700 GB/s on the card, ≈ 0.13 s for 45 GB | none: the four card entries keep GGUF order |
| card MXFP8 (16 GB) | phase 1: the existing `Format::Mxfp8` normalize (planes → 33-byte blocks → planes again); phase 2: File planes, raw H2D (`Mxfp8Planes` is exactly the two checkpoint tensors) | phase 1: two CPU passes over 16 GB; phase 2: 0 | none |

Why not flip the card's nibble order to pair order instead of a permute kernel: the card dot reads four values per
32-bit word against four activation bytes; pair order would cost two byte permutes per eight values in
`gemm_nvfp4p`, a prompt GEMM that is not at the bandwidth roof [unverified: no ncu row]. The permute kernel costs
nothing at decode and keeps the Qwen3.8 NVFP4 gates bit-identical by construction.

### 3.4 What `input_global_scale` is, and why it is ignored

In compressed-tensors' W4A4 scheme the activation is quantized at run time to FP4 with 16-value FP8 block scales, and
`input_global_scale` is that activation's tensor scale (it keeps the block scales inside E4M3's range). The weight's
dequantization does not read it. bloomery never quantizes an activation to FP4 (sm_86 has no FP4 tensor cores): the
card and host quantize activations to q8 blocks with their own dynamic scales (W4A8). So the scale has no role in
any of our arithmetic, for any value it holds. It is still read: a non-finite or non-positive value is refused by
name (a corrupt file), and the load line counts its distinct values. On this file it is 1.0 for every expert of
layer 3, which says the vendor did not calibrate activations for W4A4. The vendor's README describes the experts as
NVFP4 QAT weights. vLLM on sm_86 serves NVFP4 as W4A16 [unverified: not read in its source].

The oracle (§4) dequantizes as compressed-tensors does: `e2m1 × (scale / g)`. The engine computes
`(e2m1 × scale) × f32(1/g)`. Those differ by one rounding of `1/g` and the order: the float-precision class, judged
by σ (`exact_ref --act ours` simulates our rule), not a bug.

## 4. The oracle without llama.cpp

llama.cpp's converter cannot read this file (`quant_method: mxfp8` raises `NotImplementedError`), and transformers
5.12.1 cannot load its experts (`conversion_mapping.py:458-560` expects `experts.*.w1.weight`). So the oracle is the
vendor's modeling code with our own dequantization, f32 throughout, attention `eager` (the MSA hub kernel is SM100
only; the eager path builds the block mask from the indexer's indices, `msa_attention.py`).

**The dumper** `tools/ref/mm31/dump_tf.py` (precedents: `tools/ref/vision/dump_vision.py`,
`tools/ref/glm5next-hf-map.py`):

- **Dequant** in its own 40 lines: MXFP8 `e4m3 × 2^(s − 127)`; NVFP4 `e2m1 × (e4m3 / g)`, both exact in f32 but for
  the one division. Its self-test compares them with compressed-tensors' `decompress` and transformers'
  `Fp8DecodeScale` on a few tensors of the fixture: equal bit for bit, or the dumper refuses to run.
- **Modules**: `MiniMaxM3VLDecoderLayer` (`modeling_minimax_m3_vl.py:632`) built from `text_config`, one layer at a
  time, its parameters filled with the dequantized f32 weights (one MoE layer is 29 GB of f32 experts, then freed);
  `MiniMaxM3VLRMSNorm` (`:137`) is the Gemma form, so the dumper feeds the norms as stored (no +1; the +1 is the
  GGUF convention our header applies).
- **Layer-major streaming**: every prompt's hidden rows (4,308 tokens × 6144 × 4 B = 106 MB) advance through layer
  `L` before layer `L + 1` is read, so each layer's bytes are read once for all prompts. Each layer's span is read
  sequentially with `posix_fadvise(WILLNEED)` on the next layer, not by page faults (the 1.33 GB/s fault rate is the
  cost to avoid).
- **Taps**: forward hooks mapped to the M3 family's node names (`tools/ref/minimaxm3-dump.sh`): `attn_norm`,
  `Qcur_normed`, `Kcur_normed`, `Qcur`, `Kcur`, `kqv_out`, `ffn_inp`, `ffn_norm`, `ffn_moe_probs`, `ffn_moe_topk`
  (i32), `ffn_moe_weights_scaled`, `ffn_moe_out`, `ffn_shexp`, `ffn_out`, `l_out`, `result_norm`, plus `topk.tsv` and
  `greedy.tsv`. The MSA taps (`msa_bs`, `msa_bsf`, `msa_block_mask`, `msa_fattn`) need their HF counterparts read
  against llama.cpp's graph first [unverified: the correspondence of block-score layouts is not read]; a tap with no
  counterpart is a coverage change with a dated reason, not a silent gap.
- **Prompts**: `tools/ref/data/minimaxm3/{a..e}.ids` as they are (the same tokenizer, §2). e (2,560 ids) is the only
  one past 16 × 128 positions, so the only one where the top-16 drops a block.
- **Format**: the dump_ref set the M3 family reads (`MANIFEST.tsv`, one f32 file a node and occurrence, an i32 twin
  and `int` row for an integer node, `topk.tsv`, `greedy.tsv`), written with the column names `crates/refset` reads.

**Refset family** `tf-minimaxm31` in `crates/refset/src/arch/minimaxm31/mod.rs` (new): sets
`ref_minimaxm31_{a..e}`, `# arch minimax-m3`, `# model` the folder's first shard
(`/models/MiniMax-M3.1-Flash/model-00001-of-00048.safetensors` once the folder moves out of
`/home/user/stage/mm31`: a lead decision, before any dump), `# build`
`transformers-5.12.1+torch-<v>+dump_tf-<md5 12>` through `Build::Is`, and a fixture tier on the cut (§5).

**Venv** `/home/user/venvs/tf512` [proposal]: Python ≥ 3.11, `torch` (CPU wheel), `transformers==5.12.1`,
`compressed-tensors` (self-test only), `safetensors`, `numpy`, `tokenizers`, `jinja2`; pinned in
`tools/ref/mm31/requirements.txt`.

**Wall** [derived, CPU only, the box's 32 cores]:

| Run | Terms | Estimate |
|---|---|---|
| fixture, all sets | 7 GB read; 3 layers; routed ff 512 | < 2 min |
| real, teacher-forced a–e (4,308 tokens) | NVMe 248 GB (34 s at 7.4 GB/s sequential … 187 s by faults); routed 55.6 TMAC + non-routed ~45 TMAC at 1.0–1.5 TFLOP/s sgemm: 135–200 s; dequant 413 G values at 2–5 G/s: 80–200 s | 6–10 min |
| real, greedy a–c, 32 tokens each | 96 tokens × (7.26 GB of active experts from the page cache + 12.9 G values dequantized + Python per layer) ≈ 3–6 s a token | 5–10 min |

Not a timed run, so no lease; but it fills the page cache and the DRAM bus, so it goes through `tools/box.sh` and waits
on a sitting like any box job.

## 5. The fixture: a cut of the real bits

A real-bits cut, no requantization: the bytes the gates read are the vendor's, and the oracle (§4) runs on the cut
unchanged. `crates/model-hf` writes it (a `cut` subcommand of the `fixture` bin, `bin/fixture.rs`, beside `fetch`),
from the local folder or by HTTP Range through the hf client (the `fetch` precedent).

| Choice | Value | Why |
|---|---|---|
| layers | 0, 3, 59 → fixture layers 0, 1, 2 | 0 is MSA + dense FFN (M3.1's new pair); 3 the first MoE; 59 the last, a second MoE layer for the residual chain |
| hidden, heads, head dim, vocab | kept (6144, 64/4, 128, 200064) | the MXFP8 and indexer kernels run at their real shapes; the tokenizer and prompt ids stay valid |
| experts | all 128, top-4 | the router instance (sigmoid over 128, bias, top-4) is the real one |
| routed ff | 3072 → 512 | w1/w3 keep rows 0..511 (contiguous); w2 keeps K columns 0..511 of codes (256 bytes a row) and scales (32 a row). Legal: SwiGLU is elementwise over ff, 512 is whole 16- and 32-value groups, 128-value `Q8_2_X4` groups and the card's 32-value down form |
| shared and dense FFN | kept whole | their MXFP8 sites run at the real K |
| scalars | `weight_global_scale`, `input_global_scale` copied | the cut changes no value |
| config | `num_hidden_layers 3`, `moe_layer_freq [0,1,1]`, `sparse_attention_freq [1,1,1]`, `intermediate_size 512`, quantization block unchanged | |
| identity | `bloomery.fixture.{source,layers,ff,rev}` in shard 1's `__metadata__` | the `# fixture` line (§1.4) |

Size [derived]: embed + head 4.917 GB; layer 0 ≈ 0.35 GB; each MoE layer ≈ 0.86 GB (experts 128 × 5.31 MB = 0.68 GB);
total ≈ 7.0 GB, one or two shards. The cutter's own gate: each kept tensor's bytes equal its source range (a hash per
tensor), and the cut opens through `folder::open` with the MiniMax map.

## 6. The track cut

Tracks run in parallel worktrees. **No two tracks own one file.** A file a later track needs from an earlier one is
read-only to it; it codes against the stub interface named here.

**Shared registration files.** `tools/gate-paths.tsv`, `tools/gate-class.tsv`, `tools/families.tsv`, `justfile`,
`crates/gpu-gates/Cargo.toml`, the workspace `Cargo.toml`: **the lead applies them last**, from the rows each
report lists. `crates/gguf/src/qtype.rs`: track `nvfp4host` (one row). `tools/fixture-dirs.tsv`: track `mm31fixture`.
`crates/refset/src/arch/mod.rs` (the family table): track `mm31oracle`. `crates/model-hf/src/arch/mod.rs` and
`crates/model-hf/src/lib.rs`: track `stfolder`; `mm31map` adds its `mod` line after `stfolder` lands, or the lead
does.

| Track | Owns | Needs (stub) | Gate, FAIL-first | Size | Backend |
|---|---|---|---|---|---|
| `mm31reader` (in flight) | `crates/model/src/arch/minimaxm3/*`, `gpu-minimaxm3/src/body.rs` lines it forced, `serve_seats/minimaxm3.rs` | — | its round's tests; lead lands it first | M | grok (done) |
| `stfolder` | `crates/model-hf/src/folder.rs` (new), `arch/qwen35moe/folder.rs`, `arch/qwen35moe.rs` (the `impl FolderMap`), `arch/mod.rs`, `lib.rs`, `crates/gguf/src/lib.rs` (`Split::located`, `Place`), `serve_seats/mod.rs`, `serve_seats/qwen38_folder.rs` | — | Qwen3.8 folder gates bit-identical (a move); a pure test: a two-shard synthetic folder (`safetensors::write`) opens located, `data()` of a File tensor is the file's bytes, of a Made tensor a named refusal (FAIL-first: today `data()` on a header-only split ends `OutOfBounds` for every tensor, `gguf/src/lib.rs:485`) | L | Sonnet |
| `stquant` | `crates/hf/src/quant_config.rs`, `crates/hf/src/modelopt.rs` | — | pure: the M3.1 `quantization_config` parses to MXFP8 + `Nvfp4Ct` (FAIL-first: refused today at `:313-318`); `Nvfp4Ct` normalize of synthetic experts equals ModelOpt's with `weight_scale_2 = 1/g`, bit for bit; NaN, 0 or negative `input_global_scale` refused | M | Sonnet |
| `stvocab` | `crates/hf/src/vocab.rs` | `VocabSpec` (a struct this track defines; `stfolder` re-exports it) | §2 checks 1 and 2 (FAIL-first: the derived pre is `qwen35`) | S | Sonnet |
| `nvfp4host` (worktree exists, clean) | `crates/qdot/src/lib.rs` (the twin, its tile, its emulation), `crates/gguf/src/qtype.rs` (NVFP4_ROWS row), `crates/gguf/src/quant.rs` (`reads_scale`) | — | `gate-qdot`: AVX2 = emulation bit for bit, and within band of the f64 decode (`quant::Nvfp4Rows`) at k 512, 3072, 6144; FAIL-first: NVFP4_ROWS has `host_act None` today, and a mutant reading GGUF nibble order goes red | M | Opus (kernel) |
| `sthost` | `crates/model/src/ops.rs` (`RowLayout::Planes`, `ExpertStack::located`, the dispatch arm), `crates/model/src/moe.rs`, `crates/model/src/placement/host_lock.rs` if touched | `Split::located` + `Place` (from `stfolder`'s doc comment); the kernel's name and signature (`nvfp4host`; the emulation until then) | pure: a located stack at scattered offsets gives the union call the same bits as the same experts in one contiguous `Planes` run (FAIL-first: `ExpertStack::of` reads the wrong matrix for `at` out of order); box: `gate-alloc` unchanged | M | Sonnet |
| `mm31map` | `crates/model-hf/src/arch/minimaxm3.rs` (new) | `FolderMap` (signature in §1.1), `Format::Nvfp4Ct` (`stquant`) | pure: a synthetic two-layer MiniMax folder maps to a header `minimaxm3::Hparams` reads and whose every tensor has a role (FAIL-first: no map for `model_type minimax_m3_vl`); box: the real folder opens header-only, text bytes = 248,369,266,176 | M | Sonnet |
| `mm31fixture` | `crates/model-hf/src/bin/fixture.rs` (`cut`), a new `crates/model-hf/src/cut.rs`, `crates/refset/src/fixture.rs`, `tools/fixture-dirs.tsv` | — | `fixture cut` + `verify` on the box: a hash per kept tensor equals its source range; FAIL-first: a cut whose w2 K is not a multiple of 32 is refused by name | M | Sonnet |
| `mm31oracle` | `tools/ref/mm31/{dump_tf.py,requirements.txt}`, `tools/ref/mm31-dump.sh`, `crates/refset/src/arch/minimaxm31/mod.rs` (new), `crates/refset/src/arch/mod.rs` | the fixture (`mm31fixture`) for the fixture sets; the real folder at `/models` (lead) | the dequant self-test (§4); `refset-check` reads the sets; FAIL-first: a set whose `# model` names another folder is refused | M | Sonnet |
| `mm31body` | `crates/gpu-minimaxm3/src/{body,ffn,step,prefill}.rs`, `crates/placement/src/accepts.rs` | all of the above landed; reads `mm31reader`'s split | `gate-gpu-minimaxm3-e2e` on the fixture against `tf-minimaxm31` (FAIL-first: today the folder is refused at `serve_seats/mod.rs:97`; then a mutant that drops the `+1` on the norms goes red) | L | Opus |
| `mm31serve` | `serve_seats/minimaxm3.rs`, `crates/app/src/arch/minimaxm3/*`, the chat render test | `mm31body`, `sthost` | `gate-gpu-minimaxm3-serve` on the fixture; §2 check 3; the real-file pass: the seat opens the folder, the plan prints §0's split, one request answers | M | Sonnet |
| `stcardraw` (phase 2, off the path) | `crates/gpu/src/parts.rs`, a new permute kernel file in `crates/gpu/src`, `crates/gpu/src/arch/qwen3moe/body38.rs` (`refuse_folder`) | `stfolder` | Qwen3.8 NVFP4 gates bit-identical with raw loads; residency over a folder | M | Opus |

**Critical path** to "bloomery-serve answers on the real file": `mm31reader` (land) → `stfolder` → `mm31map` →
`mm31body` → `mm31serve`. In parallel from the start: `stquant`, `stvocab`, `nvfp4host`, `mm31fixture`,
`mm31oracle` (code and fixture self-test). `sthost` starts on stubs and joins at `mm31serve`: the fixture e2e can
run every expert on the card (2 MoE layers × 0.68 GB), so the host path is not on the body's path, only on the real
file's. Box work in order: the fixture cut and its oracle sets, `gate-qdot`, the fixture e2e, the real-folder oracle
(after the move to `/models`), the real serve pass. The timed decode row comes after, against §0's 18.9–20.3 tok/s.

## 7. What the next safetensors model costs

| Piece | Qwen3.8 (today) | MiniMax-M3.1 (this) | The next one |
|---|---|---|---|
| folder open, plan, header, tokenizer check | its own `folder.rs` | common `folder.rs` | common |
| names and keys | `impl FolderMap` | `impl FolderMap` | `impl FolderMap` |
| formats | `quant_config` rows | + `mxfp8` method, `Nvfp4Ct` | a new row only for a new encoding |
| host in place | refused | `Place`, `ExpertStack::located`, the NVFP4 planes kernel | common |
| card raw loads, residency over a folder | refused | phase 2 | common |

## 8. Risks

- **The MSA taps' correspondence** (§4) is the oracle's one unread term; a reading round on `msa_attention.py` against
  `src/models/minimax-m3.cpp` closes it before the dumper writes those taps.
- **Host room.** 263.376 − 232.381 = 30.995 GB beside all host experts; the embed lookup (2.458 GB) and the union
  scratch fit, but a plan with no card experts has ~28 GB of slack, not mm31design's 43.9. The planner reads the day's
  `host_room`; an overflow goes to the NVMe tier, which reads the same `Planes` matrices.
- **Per-model drift.** If `mm31body` grows a folder branch, the lift was cut wrong: the body loads `CardBytes::Parts`
  as Qwen3.8 does, and every folder fact lives in `folder.rs` or the map.
- **NVFP4 host MAC rate.** 1.31 TMAC/s was fitted on another quant; prefill (mm31design §7, ~100 tok/s at P = 512) moves
  with the new kernel's rate [unverified].

## 9. Evidence

- **Census.** Every shard header fetched by Range (`bytes=0-7`, then the header length): 48 headers, 89,632 tensors;
  data bytes summed by dtype: F32 185,531,904; BF16 6,646,445,056; F8_E4M3 36,380,344,320; U8 206,891,089,920; total
  250,103,411,200 = the index's `metadata.total_size`. By kind: routed experts 232,381,393,920 (codes 206,561,083,392,
  scales 25,820,135,424, scalars 175,104); attention 6,617,825,280; indexer 243,333,120; dense FFN 700,710,912; shared
  3,328,376,832; router 179,335,680; norms 1,517,568; embed and head 2,458,386,432 each; vision 1,734,145,024.
- **Layout.** Each layer in one shard (shard 1: embed, head, layers 0, 1, 2, 14; shards 4–12 two layers each; shard 48
  layer 59 and vision). Layer 3's 128 experts: six planes contiguous per expert, offsets from the header.
- **Scalars.** Layer 3 (shard 2), by Range: `weight_global_scale` w1 = w3 per expert; `input_global_scale` 1.0 for all
  384; `e_score_correction_bias` 16.16–16.47.
- **Not measured here:** NVMe rate (1.33 GB/s is the 09-23 V4.1 dump's), the NVFP4 host kernel's MAC rate, the CPU
  normalize rate, the oracle's wall. Each is marked where it is used.
