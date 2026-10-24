# Request-driven residency as one general feature: census, replay simulator, the common cut

A design note (2026-10-23; Mac only, no build, no box, no code change). It extends `tierengine.md` §3(a) (heat in one
slot book) and the gap its S6 row left ("the step port's tally is off without a residency machine and never sees a
prompt"), and it builds on `warmprof.md` (per-stream heat, `SwapMachine::open`, the keep rule; rounds W1–W5) without
repeating it. Tags: `[measured, source]`, a code constant or behaviour with `path:line` (read at the worktree's base
`dbe7a9ed`), `[derived]` (arithmetic on cited terms), `[assumption]`. All paths are relative to the repository root
unless they name rig-log.

## 0. Result

1. **Census.** Eight `TierBody` implementations; seven run the one common machine (`runtime::swaprule` +
   `gpu::host::swap`) and one (Nemotron-H MoE) runs none. The decode heat is common. The **prompt heat is not**: only
   V4.1 and the Qwen3 bodies move experts in a prompt call, through two different drivers, and they end the call by
   **opposite rules**: V4.1 always keeps the prompt's picks for the decode, Qwen3.8 hands them back whenever the rule
   holds a decode history (every request after a fresh server's first). GLM, MiMo, MiniMax-M3 and Solar-Open2 see no
   prompt heat at all. The load-time profile (`BLOOMERY_WARM_PROFILE`) is common code with one caller,
   `generate_ds41`; no serve seat reads it.
2. **Pick log.** It exists for one-row steps: `gpu::host::route_trace` writes every position's routed ids per layer
   and the slot each ran in (card, tier, host), with a `call` row per prompt call and a `contexts.tsv` join per request
   from the chat driver. It refuses prompt batches, multi-row passes and a moving residency, and three serve seats wire
   it with three copies of the header builder (GLM's, MiMo's and Solar-Open2's seats do not). The smallest common hook
   is one call in `HostTier::serve_key` (every family's batch service) that adds a prompt batch's ids to per-call
   count blocks, plus one seat-side builder; §3.3.
3. **Simulator.** Most of it exists: `tools/ref/router-residency.py` (2,428 lines) replays the rule, Belady, LRU,
   prompt openings and timed windows over router sets. What it lacks is the engine's own policy as a twin (counts
   carried across requests, prompt rows uncounted, call picks kept or returned), a pure prompt-heat arm, a rolling
   window, a profile split, tiers past the card, and a tok/s price. `tools/flow/resplay.py` adds exactly those over
   `rr`'s library (the way `tools/flow/q38_step.py:237` already imports it); §4.
4. **The decisive number** is Δ_pr = h(prompt heat at the end of prefill, frozen) − h(rolling decode heat carried from
   the previous request), mean over the first 256 decode steps of held-out requests, at the card budget, per family and
   prompt stratum, with the companion h(prompt heat then the rule) − max of the two. It settles the keep-rule conflict
   of item 1 and whether the four tier families need prompt heat. Expected [derived, §5.2]: prompt heat then the rule
   wins on prompts of 512 ids and more by +5…+12 points; under 128 ids the rolling heat wins or ties. Box: 27–42
   minutes of functional runs in a hold, no lease (41–67 if no Qwen3.8 chat trace exists on the box); V4.1's short-chat
   answer needs none, a Mac replay of the existing D2 trace (§5.3).
5. **Prediction.** V4.1 (A6000, plan (a), prose P 512) has a measured step line, fitted here from the residency clip's
   282 steps: **t = 38.94 − 25.99·h + 0.099·F ms** (h the card share of routed slots, F flips a token; ±0.39, ±0.84,
   ±0.023 at 95 %, residual SD 1.47 ms) [derived from measured]. It reproduces the clip's three arms within 0.45 ms. For
   Muse Spark Open (bp shape, §6.5) the same structure gives **11.3–14.1 tok/s at h = 0, 11.9–15.1 at the A6000
   pool's static 11 %, 14.7–20.0 at h = 0.5, 16.7–23.9 at h = 0.7** [derived; the spread is t_serial's band and the
   two readings of §1.2], not the ~20 tok/s at h = 0 that `muse-spark-attach.md` §5 derives: one row's decode does not
   overlap a layer's attention with its host experts (§1.3).
6. **Order** (§6.4): R1 resplay (Mac) → R2 the decisive replay and collection (box functional) → R3 the heat owner as
   a move → R4 the trace hook → R5 the policy flip with its A/B → R6 glue removal → R7 Muse. The hook moves after the
   decisive number because the decisive number needs no engine change: the step feed's traces carry the same routes
   (V4.1's batch is the steps' state bit for bit with the prompt streaming off).

## 1. Paper decomposition and resource timeline

### 1.1 The terms

Per decode token (one row, one pass): N_s routed slots (layers × top-k), h the share the card serves (the stage card
and any tier card), B_e an expert's bytes, c_h the host cost of a slot, c_c the card cost of a slot, t_serial the
token's work that no host slot overlaps (each layer's front and back: norms, attention, the router, the handoff, the
combine; the head; launches), F flips a token, c_f a flip's cost on the step. Per prompt call: P ids, the units of U
rows the walk runs, the picks a call admits, and their bytes over the link.

| Term | V4.1 (A6000, plan (a)) | Qwen3.8 (A6000, plan (a)) | Muse Spark Open (box, bp shape) |
|---|---|---|---|
| layers × top-k = N_s | 40 × 6 = 240 [rig-log `tools/clip/residency/data.json` meta] | 48 × 10 = 480 (`q38_step.py:163`: host slots = 48 layers × 10 − card slots; `rr` FAMILY's 43 eligible layers predate the plan holding all 48) | 62 × 8 = 496 (muse §1) |
| B_e | 16,773,120 B (data.json) | 3,072,000 B (`router-residency.py:162-167`) | 12,976,128 B (3 × 4096² IQ2_XXS) |
| card slots a layer | 70 of 384: 40 pinned, 29 churn, 1 spare (data.json) | ~299 of 512 (FAMILY plan_total 12,841 over 43 layers, ctx 4,096) | A6000 ~28 moving + 3090 ~26 static of 256 [derived, §6.5] |
| c_h − κc_c (measured slope a slot) | **108.3 µs** joint fit, 92–108 band [derived from measured, §1.2] | `q38_step.py` (calibrated model) | 83.8 µs (V4.1's, × B_e ratio 0.774) under (B); c_h 105.5 µs, c_c 21.6 µs under (A) (§7) |
| c_f a flip | **0.099 ms** [derived from measured, §1.2] | — | 0.077 ms (× 0.774) [derived] |
| PCIe H2D | 26.28 GB/s pinned (plan.md 「비용 모델」) | same | same |
| host loss beside a staging copy | 0.24 GB/s of host read per 1 GB/s copied [measured, rig-log `log/2026-09-28.md` 스테이징 표] | same | same |

### 1.2 The measured V4.1 line

The residency clip (rig-log `log/2026-09-30.md#residency-clip`; data in rig-log `tools/clip/residency/data.json`:
A6000, plan (a), prose P 512, 96 decode steps per arm, sitting `resvideo-v41`) recorded each step's host slots and ms
for three arms. Least squares over the 282 steps after each arm's first [derived from measured]:

    t = 38.94 (±0.39) − 25.99 (±0.84)·h + 0.0993 (±0.0226)·F_land   ms     (95 %; residual SD 1.47 ms, n 282)

with h = 1 − host_slots/240 and F_land the flips landing at that step (the boundary steps every 4 rows carry them).

| Arm | mean h | flips a token | predicted ms (tok/s) | measured mean ms (tok/s) | runner tok/s |
|---|---:|---:|---:|---:|---:|
| residency off | 0.1746 | 0 | 34.40 (29.07) | 34.08 (29.34) | 29.48 |
| rule (`mid`) | 0.4698 | 5.56 | 27.29 (36.65) | 27.74 (36.05) | 35.70 |
| rule + prompt streaming | 0.6333 | 4.32 | 22.91 (43.64) | 22.79 (43.87) | 43.86 |

Readings. The same three-term fit inside sub-ranges gives slopes −21.1 ± 4.0 (h < 0.45), −23.6 ± 2.0 (h ≥ 0.45) and
−22.9 ± 4.0 (h ≥ 0.6) [derived]: no shallower high range, so no card-shadow kink below h 0.75 and the host leg
dominates every layer over the measured range. The within-range slopes sit ~10–15 % under the joint −26.0, so the
joint ±0.84 understates the slope's uncertainty; −22…−26 ms per unit of h (92–108 µs a slot) is the honest band, and
the tables below use the joint fit. A flip costs about one host slot for one step (0.099 ms ≈ 0.92 × 108.3 µs), which confirms the "c 1" the
09-28 replay assumed (rig-log `log/2026-09-28.md#residency-replay`). Two readings of the slope fit the line and the
data cannot separate them: **(A)** the card's experts add to the step (sum): c_c = B_e / 600 GB/s = 28 µs, so c_h =
136 µs, a host read of 123 GB/s inside plan.md's 121–137 band, and t_serial = 6.2 ms; **(B)** the card's experts hide
under the host leg (the per-layer max of `q38_step.py`'s header): c_h = 108 µs, an effective 155 GB/s, above STREAM's
147.7 GB/s triad but under the read-only peak, and t_serial = 12.9 ms. Both predict the same V4.1 step; they differ
for Muse (§7), and the first Muse measurement names which one held.

### 1.3 Resource timeline per decode token

| Resource | V4.1 at h 0.17 (off) | V4.1 at h 0.63 (stream) | Muse at h 0.11 (bp, static) | Muse at h 0.6 |
|---|---:|---:|---:|---:|
| host DRAM (expert reads) | 26.2 ms (3.32 GB / 127) | 11.6 ms (1.48 GB) | 45.3 ms (5.73 GB / 127) | 20.3 ms |
| host CPU (qdot) | under the DRAM shadow (V4.1 b5prof) | same | ~17 ms, under the DRAM shadow (muse §5) | ~8 ms |
| card SM | t_serial 6–13 ms + card experts | same | 26.7–32.9 ms dense (18.67 GB at 567–701 GB/s) + 1.2 ms experts | same + 6.4 ms experts |
| PCIe | 0 (no flips) | 4.32 flips × 16.8 MB = 72 MB → 2.8 ms of the link, async | 0 | ≤ 6 flips × 13.0 MB = 78 MB → 3.0 ms, async |
| NVMe | 0 (victims 0 in the clip) | 0 | 0 (206 GB of experts fit in 251 GB) | 0 |
| **wall** | 34.1 ms measured | 22.8 ms measured | 66.3–84.3 ms [derived, §7] | 45.9–63.9 ms |

**Sum or max.** Per layer the engine runs front + max(shadow, go → host service → signal) + back, and the layers add
(`tools/flow/q38_step.py:9-22`, read from `program38.rs` and `host/step.rs`; `BLOOMERY_HYBRID_OVERLAP` is retired,
`crates/levers/src/registry.rs:1427`). Only the card's experts (and a shared expert where the model has one) sit in the
shadow; attention, the latent projections and the head are front and back. So the token's wall is t_serial + the host
leg while the host dominates each layer's max. The V4.1 numbers confirm it: at h 0.17 the host leg alone is 24–26 ms
and the step 34.1 ms, so ~9 ms do not overlap it [derived]. Muse has no shared expert (muse §1: q·o, attn_gate, k·v,
latent down·up, router per layer), so its shadow holds only the card experts (8h × 22 µs a layer) against a host leg of
8(1−h) × 102 µs: the host dominates until h ≈ 0.85 [derived], and the sum reading holds over every h a residency
reaches. Qwen3.8 is the counter-case: at P 4096 the prompt's picks took the host slots from 180 to 77 a step and the
gain a slot halved, "the decode floor is now on the card side" [measured, rig-log `log/2026-09-30.md#q38seed-ab`].

**Flow levers before a term.** Asynchrony: flips already stage on the lane's threads and copy on their own stream
(`crates/gpu/src/host/swap.rs:33-45`); their PCIe time is under the step's shadow and worth 0; what they cost is the
boundary step's staging wait and the host's lost read (0.24 × 72 MB / 23 ms ≈ 0.75 GB/s of 127 → 0.6 % of the host
leg [derived]), which the fitted c_f carries. Bulk: one copy per expert part per flip is the unit already; a prompt
call's picks land per layer in batches (`swap.rs` call mode). SIMD: the host leg is qdot over the union, out of scope
here. SIMT: n/a. **The lever this design is about is the share of slots on the card: h**, worth 108 µs a slot on V4.1
and 81–84 µs on Muse, linear while the host dominates.

**The busiest unit and the step.** At V4.1 h 0.17 the host DRAM demand is 26 ms of a 34 ms step (77 %); at h 0.63,
11.6 of 22.8 ms (51 %). The rest is the serial card work, not latency the warps fail to hide, and the measured slope
equals the host leg's per-slot demand within reading (A)/(B). So a lever on h predicts its full per-slot value until
the card's shadow takes over (Qwen3.8 near h 0.8; Muse near 0.85).

## 2. Census of today's residency

### 2.1 The common owners

| Owner | Where | What it holds |
|---|---|---|
| the rule | `crates/runtime/src/swaprule.rs` | `SwapParams` (`:43`), `mid` (`:67`: every 4 kept rows, cap 24, margin 3, min count 2, decay 0.9); `SwapRule` (`:522`) with one heat vector per stream (`:544`, `with_streams` `:875`, `stream_open` `:929`, no engine caller); `observe` `:1193`, `end_pass` `:1235`, `end_pass_by` `:1248`, `plan` `:1336`; `open` `:1465` (a whole prompt's counts relayout, **no production caller**); `call_pick` `:1540` (a call's per-layer pick from its unit's counts; "the rule does not move"); `settle` `:1601`; `has_history` `:1652` ("a prompt call's rows are not counted, and its picks move no count"); `usage` `:1107` (what the warm profile saves) |
| the machine | `crates/gpu/src/host/swap.rs` | `Residency` `:218`; `SwapSource` `:305`; `Tally` `:398` (`note` `:444`, fed at `host/step.rs:1090` for a one-row step and `:1260` for a multi-row pass); `SwapMachine` `:1735`: `end_pass` `:2308`, `boundary` `:2438`, `reseed` `:3008`, call mode `begin_call` `:3197`, `call_pick` `:3319`, `has_history` `:3557`, `end_call` `:3579`, `return_layer` `:3754`. It moves the stage card's set only: tier card entries are "no part of the machine" (`:22-27`) |
| the file source and glue | `crates/gpu/src/host/swap_source.rs` | `FileStacks`, `LIVE_DELAY = 2` (`:154`), `DEADLINE` 30 s (`:178`), `StackFacts`, `PlanStacks`, `hosted_stacks`, `ResidencySpec` (`:361`: lever, pool, delay, deadline, top_k, max_rows, stacks), `FileSwap` (`:531`, the one `SwapSource`), `ResidencyGlue` (`:937`) |
| the body seam | `crates/gpu/src/host/served.rs` | `TierBody::residency_parts`; `prompt_pass` (`:199`: a prompt call is one pass, 0 rows kept, "the batch service notes no id"); `refuse_steps_feed` (`:215`) |
| the shell | `crates/gpu/src/host/shell.rs:115` | `TierShell::residency_parts`, always `Some`: GLM, MiMo, MiniMax-M3, Solar-Open2 |
| the prompt-call driver | `crates/gpu/src/host/prompt_stream.rs:47` | `PromptStream` ("one owner of a streaming prompt call's host streaming, any MoE family"): `begin` `:185`; users: Qwen3 bodies only (`body35.rs`, `body38.rs`, `placed.rs`, `wide38.rs`) |
| the host tier's call API | `crates/gpu/src/host/mod.rs` | `call_begin` `:1280`, `call_history` `:1573`, `call_end` `:1580`, `swap_reset` `:1758`, `route_prompt` `:1926`, `refuse_traced_batch` `:1935`, `routed_ids` `:2216`, `serve_key` `:2236` |
| the NVMe tier's heat | `crates/gpu/src/host/nvtier.rs` | the books' `counts` (`:1766`; prompt and step columns both), `HEAT_HALF_LIFE` 32 columns (`:1799`), evict by `tick` or `heat` (`BLOOMERY_NVTIER_EVICT`) |
| the warm profile | `crates/runtime/src/warmprof.rs` (`WarmSection` `:26`, `WarmProfile` `:291`), `crates/app/src/warm.rs` (`WarmFile` `:20`, `load` `:96` reseeds, `save` `:132`) | decode counts only (`SwapRule::usage`); one caller: `crates/gpu-gates/src/bin/generate_ds41.rs:1114` |
| the plan-side resolution | `crates/levers/src/lib.rs` | three copies: `ResidencyWhy` `:1297` / `residency_unset` `:1421` / `residency_at_plan` `:1731` (V4.1 and the tiered seats), `Residency38Why` `:1798` / `:1926` / `:1960` (Qwen3.8), `GlmWhy` `:2310` / `:2430` / `:2479` (GLM); gate side `crates/gpu-gates/src/residency.rs`, `residency38.rs` |
| the lever rows | `crates/levers/src/registry.rs` | `RESIDENCY` `:754` (unset `mid-p0-s1` on a serving place), `HOSTSTREAM` `:879` (V4.1 only), `XSTREAM` `:902` (Qwen3.8 only), `WARM_PROFILE` `:619`, `ROUTE_TRACE` `:641`, `CARD_BUDGET` `:261`, `NVTIER_BYTES` `:300`, `NVTIER_EVICT` `:356`, `HOST_POPULATE` `:679`, `HOST_LOCK` `:691` |

In serving the residency is never reset per request: the serve loop's `Engine::reset` drops the cache only
(`crates/gpu-gates/src/bind.rs:3113`, `Cmd::Reset` `:2321`); the map returns to the seed only on
`POST /residency/reset` (`crates/serve/src/api.rs:3294`) or `Session::clear` on one slot (`crates/app/src/lib.rs:450`).
So the counts a request's decode starts from are the last request's, decayed: a fact every replay twin needs.

### 2.2 Per family

| Family (TierBody) | Residency | Heat that feeds it | Promote cadence | What bounds promotion bytes | Per-model code not forced by an architecture fact |
|---|---|---|---|---|---|
| V4.1 (`crates/gpu-deepseek41/src/body.rs:2802`, `residency_parts` `:2813`) | yes, `mid-p0-s1` unset on `a`/`bp`; off on two-stage plans (`ResidencyWhy::FixedPlace`) | decode tally (rule); prompt call picks per group when P ≥ `STREAM_MIN_P` 128 (`body/prefill.rs:155`), floor `STREAM_FLOOR` 32 a batch (`:162`); warm profile in `generate_ds41` only | rule every 4 kept rows; call picks every unit of a call ≥ 128 ids | cap 24 flips a plan (6 a token); a call's pool (29 churn a layer: 480 experts, 8.05 GB at P 512 [measured, data.json `call_stream_end`]) | its own call driver (`prefill.rs` `begin_call` `:1788`, `call_pick_routed` `:2766`, `stream_end` `:1862`) beside the common `PromptStream`; the keep rule (`stream_end(.., done.is_ok())`, `:759`: always keep); `STREAM_FLOOR` copy; `LIVE_DELAY = 4` (`crates/gpu-deepseek41/src/swap.rs:53`, a derivation the common `2` also makes); the seat's trace builder (`serve_seats/ds41.rs:800-868`); `ResidencyWhy` resolution in its seat. Forced: the r8 Q3_K convert and churn hosting (`swap.rs`) |
| Qwen3.8 (`body38.rs:4900`, `:4911`) | yes; `XSTREAM` split/admit | decode tally; call picks per ubatch (`PromptStream`; floor `STREAM_FLOOR` 32 defined at `wide38.rs:2386`, re-exported at `body38.rs:2247`; "a prompt call of fewer ids opens no call") | rule every 4 kept rows; call picks every ubatch of a call ≥ 32 ids | rule cap; call picks; the split ring | the keep rule in the body: `stream_end(r.is_ok() && !history)` (`body38.rs:3852-3869`: return the picks once the rule holds a history); its own `Residency38Why` resolution; `route_trace38.rs` (`crates/gpu-gates/src/bin/shared/route_trace38.rs:58`) |
| Qwen3.5/3.6 (`qwen3moe/placed.rs:1676`, `:1689`; body `body35.rs`) | yes | decode tally; call picks (the same floor, re-exported at `body35.rs:1816`; history read `:2476-2481`) | as Qwen3.8 | as Qwen3.8 | the same keep rule, a second copy |
| GLM-5.3-Flash (`crates/gpu-glm5next/src/body.rs:2541`, `:2552`) | yes (`TierShell`) | decode tally only; a prompt call is one pass with 0 rows kept (`served.rs:199`) | rule only | rule cap | `GlmWhy` resolution; the seat leaves `ROUTE_TRACE` out (`serve_seats/glm.rs:313`). Forced: `swap.rs` (57 lines: dense blocks, the NextN layer, as `StackFacts`) |
| MiMo-V2.6 (`crates/gpu-mimo2/src/body.rs:1243`, `:1254`) | yes (`TierShell`, tiered seat) | decode tally only | rule | rule cap | none beyond `swap.rs`'s facts (55 lines) |
| MiniMax-M3 (`crates/gpu-minimaxm3/src/body.rs:835`, `:846`) | yes (`TierShell`, tiered seat) | decode tally only | rule | rule cap | its trace builder (`shared/minimaxm3_seat.rs:165-199`) |
| Solar-Open2 (`crates/gpu-solar2/src/body.rs:1344`, `:1355`) | yes (`TierShell`, hybrid seat) | decode tally only | rule | rule cap | none; no trace |
| Nemotron-H MoE (`crates/gpu/src/arch/nemotron_h_moe/placed.rs:312`, `:324`) | **no**: `residency_parts` → `None` | — | — | — | the plan's map stays; attaching it is the shell |

Two families encode opposite answers to the question this design asks. Qwen3.8's comment says why it returns the
picks: "the decode has been routing experts of its own, and the call's picks … put the pool back where they found it
rather than leave the decode to churn back at the flip cap" (`body38.rs:3853-3859`). V4.1 keeps them on every call.
Neither choice was replayed on chat with a history; `warmprof.md` §8 Q4 names the same case and expects up to +10.6 %
[assumption there].

### 2.3 Today's engine policy, as the twin a replay must reproduce

| Family | Seed at load | Prompt call | Call end | Decode | Across requests |
|---|---|---|---|---|---|
| V4.1 | id prefix (or `WARM_PROFILE` in `generate_ds41`) | P ≥ 128: per group, per layer, pick experts with count ≥ 32·group into the churn pool | keep always | rule `mid`, counts not seeded by the prompt | carried (decayed) |
| Qwen3.8, 3.5/3.6 | id prefix | P ≥ 32: per ubatch pick (floor 32), split ring under `split` | keep without history, return with | rule `mid`, unseeded | carried |
| GLM, MiMo, MiniMax-M3, Solar-Open2 | id prefix | none | — | rule `mid` | carried |

`tools/ref/router-residency.py gen` (`:1403`) has none of these as an arm: its (b) resets the rule to the seed at every
request, its (c) seeds the rule's counts with the prompt's decayed counts (`window_arm` `:698-746`, `counts0=dec`),
which the engine does not do, and its `steady` counts every prompt position as a decode row, which the engine refuses
(`served.rs:215`). The replays of rig-log `log/2026-09-28.md#residency-replay` are therefore bounds, not the engine.

## 3. Heat sources and the pick log

### 3.1 What exists

| Source | Owner | Phase | Reaches |
|---|---|---|---|
| decode tally | `Tally::note` (`swap.rs:444`) from the step port (`host/step.rs:1090`, `:1260`) | decode rows only; off without a machine (`Tally::off`) | `SwapRule` counts via `end_pass` |
| prompt call counts | each call's unit counts in `SwapMachine::call_pick` (`swap.rs:3319`) | prompt, V4.1 and Qwen3 only | the call's flips; never the rule's counts (`swaprule.rs:1652` doc) |
| NVMe books | `nvtier.rs:1766` | prompt and step columns | the arena's eviction under `NVTIER_EVICT=heat` |
| warm profile | `app::warm::WarmFile` | decode usage | the seed at load (`reseed`), `generate_ds41` only |
| records | `crates/gpu-gates/src/record.rs`: `TIER_COST` `:97` (hits before/after), `NVTIER` `:860`, `residency_pass`/`residency_lever`/`RESIDENCY_UNSET` `:1191`/`residency_host`/`residency_reset`, `WARM_LOAD` `:1260`/`warm_save`, `call_stream`/`CALL_STREAM_END` `:1327`, `STAT_STEP` `:1491` / `STAT_STEP_HOST` `:1555` (`host_slots` a step under `BLOOMERY_STEP_STATS=1`) | per pass / per call / per step | the online h a step (host slots), flips, call picks |

The online hit is readable today from `stat step` (host slots a step) and the clip's arms used exactly that. What no
record carries is *which* ids a position routed, so no online run can be replayed against another policy.

### 3.2 The pick log: `RouteTrace`

`crates/gpu/src/host/route_trace.rs` (909 lines) writes `tools/ref/router_trace.cpp`'s layout: `MANIFEST.tsv` (header,
a `layer` row per layer, a `call` row per prompt call, `# positions`, `# complete`), `topk-<layer>.u16` (positions ×
n_used ids) and `slots-<layer>.u8` (`C` stage card, `T` tier card, `H` host). The step port records each layer's
service in order (`record`, `:345`) and appends the row after the signal (`write_row`, `:404`); `prompt` (`:280`) marks a
call. Phase is derivable (positions inside a call row are prompt, the rest decode) and the request id comes from the
chat driver's `contexts.tsv` (`tools/ref/route-trace-chat.py`, read by `tools/bloomery/route_trace.py`). So the log the
spec asks for exists for the step feed. Its gaps:

| Gap | Where refused | Effect |
|---|---|---|
| G1 prompt batches | `HostTier::serve_key` → `refuse_traced_batch` (`mod.rs:1935`) | a trace needs `BLOOMERY_PREFILL=steps`: prompts run at the step's rate (V4.1 18.7 ms a position, Qwen3.8 30–101 ms [measured, rig-log `plan-triage.md` "What q38rtrace left"]) |
| G2 passes of several rows or columns | `record` (`route_trace.rs:345`, `Chain::Step` only), `host/step.rs:1221` | no trace under a draft (MTP, DSpark) or `--parallel` |
| G3 a moving residency | the seats (`route_trace38.rs` `residency_refusal`; `ds41.rs:840`) and `ResidencyWhy::RouteTrace` | slot kinds never record the online machine's moves, so an online run's h cannot be compared with its replay on the same routes |
| G4 phase and request | — | derivable (call rows, contexts.tsv); no gap |
| G5 seat wiring | three header builders (`serve_seats/ds41.rs:848`, `shared/route_trace38.rs:58`, `shared/minimaxm3_seat.rs:174`) plus four in bins and gates; none in `serve_seats/glm.rs`, `mimo2.rs`, `solar2.rs` | no chat trace of GLM, MiMo or Solar-Open2 |

### 3.3 The smallest common hook (R4)

One owner each, every family:

1. **Prompt batches as count blocks** (closes G1 for heat; positions inside a batch call are not needed by any policy
   in §4). `HostTier::serve_key` (`mod.rs:2236`, the batch service every family's walk calls) replaces
   `refuse_traced_batch` with `self.note_batch(key)`, which reads the batch's routed ids (the port already holds them,
   `BatchPort::routed_ids`, `batch.rs:493`) and adds them to:
   - `RouteTrace::note_batch(&mut self, key: BatchKey, ids: &[u32], n_used: usize) -> Result<(), GpuError>`: per
     (call, set, layer) one row of `n_expert` u32 counts in `prompt-<layer>.u32`, and a `batch` manifest row
     `call set layer rows routed` (`routed` the rows the layer routed: V4.1's CED layers skip positions no later reader
     needs, so it can be below `rows`). Refused by name: a batch outside a call, a layer out of the call's set order.
     The call row gains `feed batch|steps`; a batch call adds no positions to `topk-*`.
   - `HeatBook::note_batch` (§6.1), when a call is open: the same counts for the policy. One walk of the ids feeds both.
2. **Residency beside the trace** (closes G3). The header gains `residency <word>` (a new `RESERVED` key) and the seats
   stop refusing a moving residency; `ResidencyWhy::RouteTrace` goes (the routing does not depend on the map: the clip's
   three arms emitted identical tokens [measured, rig-log `log/2026-09-30.md#residency-clip`]). The slot files then hold
   the online kinds, so h(online) and h(replay) are read off one trace.
3. **One seat builder** (closes G5): `crates/gpu-gates/src/bin/shared/route_trace_seat.rs` with
   `pub fn open(levers: &Levers, facts: TraceFacts<'_>) -> Result<Option<RouteTrace>, GateError>` where
   `TraceFacts { model: &Path, arch: &str, build: &str, n_expert, n_used, first_layer, n_layer, place: &str, extra:
   Vec<(String, String)> }`; it owns the refusals that remain (a draft, more than one slot, `--prefill steps` no longer
   required) and is called from `serve_seats/ds41.rs`, `qwen38.rs`, `glm.rs`, `tiered.rs` (MiMo, MiniMax-M3) and
   `hybrid.rs` (Solar-Open2). `route_trace38.rs` and `minimaxm3_seat.rs:165-199` go. `tools/ref/route-trace-chat.py`'s
   `SERVERS` (`:66`) gains the generic `bloomery-serve` (its `listening` record), so one driver traces every family.
4. **Readers.** `tools/ref/router-coverage.py` gains `read_batch_counts(set, layer)`; `tools/bloomery/route_trace.py`
   carries the `batch` rows into its contexts.

G2 stays refused by name: a verify pass is the draft's question, and `--parallel` traces wait for the per-stream heat
(`warmprof.md` W2). Cost: one counting walk per batch layer, P × n_used adds (V4.1 P 4096: 4096 × 6 × 40 ≈ 1 M adds,
under a millisecond on the serving thread [derived]); zero with no trace and no open call.

## 4. The replay simulator: `tools/flow/resplay.py`

### 4.1 Where and what it reuses

`tools/flow/resplay.py` (Python 3 + numpy, the flow models' home), with its constants in
`tools/flow/resplay-families.tsv` (one row per constant, columns as `tools/flow/q38-constants.tsv`: name, value, lo,
hi, unit, kind measured|derived|assumed, conditions, source, anchor, note). It imports
`tools/ref/router-residency.py` as `rr` the way `tools/flow/q38_step.py:237-246` does and uses its readers (`rr.Set`,
`router-coverage.py`'s `read_manifest`/`read_topk`, `route_trace.py`'s contexts), its rule (`rr.Rule`, `rr.RULES`,
pinned to `swaprule.rs` by `rr fixture` and `crates/runtime/tests/data/swaprule-*-fixture.json`), `rr.replay` (`:471`:
flip semantics, spares, live delay, a copy link, `counts0`, `passes` with 0 kept rows), `rr.opening` (`:662`),
`rr.belady` (`:597`), `rr.static_hits`, `rr.spread`. One change to `rr`: `Replay` (`:466`) returns its final state
(`resident`, `counts`, pending flips, `since`) and `replay(…, state0=…)` continues from one, so requests chain in serve
order. For Qwen3.8 the price is `q38_step.step(p, depth, hit=h)` (`q38_step.py:454`), its calibrated model; no second
Qwen3.8 cost model.

### 4.2 Inputs

A router set (an `rr` set or an engine route trace), its `contexts.tsv` when it has one (request, prompt length P,
`split` learn|held, genre), and per family a budget word per tier: `card=plan` (the plan's n_l by `rr.spread`, the
`card_experts` the plan record prints) or `card=nN`, `pool=K` (the churn pool a layer, the machine's moving set),
`pinned=P`, `tier=…` (a tier card's static set), `ram=G` (the arena's bytes under an NVMe plan), `nvme` (the rest).
A corpus set with no contexts is cut into synthetic requests of P prompt + N decode positions, P ∈ {64, 512, 4096}.

### 4.3 Policies (CLI words)

| Word | Meaning | Engine twin of |
|---|---|---|
| `static` | the seed's set, never moved; seed = id prefix or a profile | residency off |
| `prompt-freeze[:all\|last=N][:m=M]` | at the end of each prompt the whole prompt's (or its last N positions') counts relayout the moving pool by `SwapRule::open`'s order (`rr.opening`), at most M flips; no rule after | `SwapRule::open` (no caller today) |
| `prompt-call[:min_p=128][:floor=32][:unit=U]` | per unit of U rows of the prompt, `call_pick` from the unit's counts into the pool (floor × group), kept | V4.1 `HOSTSTREAM`, Qwen3.8 `admit` |
| `rule[:mid\|knee\|strata][:carry=serve\|request]` | the decode rule; `carry=serve` keeps the map and decayed counts across requests in trace order with prompt rows uncounted (`passes` with 0 kept); `carry=request` resets to the seed at every request (`rr gen` (b)) | the decode rule as served / as timed |
| `rolling:W` | exact sliding window: at each boundary the moving pool is the top of the last W decode rows' counts, flips capped by `--cap` a boundary; no hysteresis | the "last 31 → 71 % of next 32" statistic; `mid` is an exponential window of 40 rows (decay 0.9 every 4: Σ weights = 4 / 0.1, half-life 26 rows [derived]) |
| `prompt+rule[:seed=none\|decayed][:end=keep\|return\|open]` | a prompt policy, then the rule; `seed=decayed` folds the prompt's decayed counts into the rule's counts (`rr` window arm), `none` is the engine; `end=return` hands the picks back when the rule has a history | V4.1 (`prompt-call`, `end=keep`, `seed=none`), Qwen3.8 (`end=return` with history) |
| `profile[:learn]` | the seed at load is the ranking of the learn split's counts (decode, or decode + prompt as the 09-28 user decision says: "디코드와 프리필 모두"), scored on held requests only; composes with every policy above | `WARM_PROFILE` |
| `engine:<family>` | the row of §2.3 | today's default |
| `belady`, `oracle-static` | MIN with bypass (the ceiling of any demand policy); the request's own best static set (the ceiling of any frozen one) | — |

### 4.4 Tiers and pacing

- **Card.** A flip decided at boundary b is live at max(b + delay, the link's ready time): `rr.Link` over PCIe at
  `pcie_gbps` (26.28) with the staging window share the machine allows; delay from the family row (`LIVE_DELAY`).
  A prompt call's picks land on the same link; a pick whose copy ends after the call's last unit is live late (counted
  `late`, and its slot serves as host until then). `prompt-freeze` copies run behind the call's last layers: per layer,
  its counts are final after its last unit, so its flips start then; the copies not landed by decode step 0 are late.
- **Host RAM and NVMe** (paged plans only). The arena holds `ram` bytes of experts; a slot off the card and off the arena
  is an NVMe read at `nvme_ms` a part (the row's constant, 2.51 GB/s at QD 1, ~5 GB/s at QD 3 [measured, tierengine
  §3(b)]); arena eviction `tick` or `heat` (`HEAT_HALF_LIFE` 32). The decisive number needs only the card.
- **Price of a move.** c_f a flip on the step (§1.2), plus the host's lost read 0.24 × (copy GB/s) while a copy runs
  beside the host leg, plus for prompt moves the call's wall if the link time exceeds it (V4.1 P 512: 480 picks, 8.05 GB,
  0.31 s against a 2.7 s prompt [derived from measured: `call_stream_end` 480 admitted, pp 190.5 tok/s]; the pp moved
  −1.3 %).

### 4.5 Cost model per family

    t(h_c, h_n, F) = t_serial + N_s · [(1 − h_c − h_n)·c_h + h_c·c_c·κ + h_n·c_n] + F·c_f     (ms a token)

κ = 1 under reading (A) and 0 under (B) (§1.2); c_n an NVMe slot's cost. V4.1 uses the fitted line itself (t_serial +
N_s·c_h = 38.94, N_s(c_h − κc_c) = 25.99, c_f 0.0993); Qwen3.8 calls `q38_step.step(…, hit=h)`; GLM and Muse rows carry
physical constants marked `derived`/`assumed` until two arms of their own anchor them (GLM: 39 eligible layers × 8,
B_e 15,204,352 (`router-residency.py` FAMILY glm)). A tok/s is printed with its conditions: card, plan, depth, P.

### 4.6 Output

Per (family, stratum, policy) one row: `h` over decode positions [P, P + 256) of the scored requests, `h16` (first 16
steps), `h_tail` (steps ≥ 64), flips a token, MB a token, prompt GB a request and its link seconds, late landings,
predicted ms and tok/s (both readings where they differ), and the paired difference against `engine:<family>` with a
95 % interval over requests (t over the scored requests; the replay is exact, the scatter is the requests). Subcommands:

    tools/flow/resplay.py replay <family> --data DIR <set>... [--split held] --budget card=plan[,pool=K][,ram=G]
                          --policies W1,W2,... [--steps 256] [--json PATH]
    tools/flow/resplay.py predict <family> --hit H [--flips F] [--prompt-gb G] [--depth D]
    tools/flow/resplay.py decide <json>          # §5.4's table, one line a family
    tools/flow/resplay.py --self-test

### 4.7 Self-test (each case with its FAIL-first mutation)

| Case | Asserts | FAIL-first |
|---|---|---|
| `static_count` | a 2-layer hand trace's static h equals the hand count | seed slice off by one → red |
| `freeze_is_open` | `prompt-freeze` makes `SwapRule::open`'s flips on `rr.make_open_fixture`'s trace (`:967`) | victims by count descending → red |
| `rule_twin` | `rule:mid:carry=request` reproduces every `swaprule-*-fixture.json` flip | decay before the plan → red |
| `carry_serve` | two requests: under `carry=serve` request 2's step 0 sees request 1's map; under `carry=request` the seed; prompt rows add no count | count the prompt's rows → red |
| `call_end` | `end=return` restores the pre-call map exactly when the rule has a history, `end=keep` never | return without history → red |
| `rolling_window` | `rolling:2` on a hand trace gives the hand sets | window W+1 → red |
| `profile_split` | `profile` learns on learn rows only | held rows in the ranking → red |
| `belady_bound` | every policy ≤ `belady` on 20 LCG traces | Belady without bypass → red |
| `pacing_late` | a 1 GB/s link and 16.8 MB experts: an admit serves from the hand-computed step | live at the decision → red |
| `cost_v41_clip` | `predict v41` at the clip's (h, F) of §1.2 gives each arm's measured mean within 0.6 ms | slope in ms per slot instead of per share → red |
| `q38_delegate` | `predict q38` equals `q38_step.step` at the same h | an own formula → red |
| `refusals` | a set without `# complete`, a split not learn/held, a budget over n_expert, an unknown word: each refused by name | — |
| `decide_table` | synthetic deltas reach each branch of §5.4 | — |

Registered in `tools/check-recipes.sh`'s self-test list (`:367-417`, beside `tools/flow/q38_step.py --self-test` `:381`).

## 5. The measurement plan

### 5.1 The decisive number

Per family f ∈ {V4.1, Qwen3.8, GLM-5.3-Flash} and stratum s ∈ {chat-short (P < 128), chat (128 ≤ P < 1024), long
(P ≥ 2048): prose, code, Korean}, over held requests, decode positions [P, P + 256), at the plan's card budget:

    Δ_pr = h(prompt-freeze:all) − h(rule:mid:carry=serve)           the one decisive number
    Δ_c  = h(prompt+rule:seed=decayed:end=keep) − max(both)          its companion
    Δ_e  = h(best) − h(engine:f)                                     what the policy step buys, in tok/s via §4.5

### 5.2 What is already measured, and the expected values

- V4.1, prose P 512, rule from the seed (the clip): first 16 steps h 0.227 (rule) vs 0.568 (prompt picks + rule),
  last 16 steps 0.629 vs 0.705; tg at P 4096: 29.12 / 31.71 / 36.90 tok/s (off / rule / stream) [measured, data.json
  `tg_tps`]. The prompt's heat wins early and still leads at step 96.
- V4.1 replays (09-23 corpus sets, 70–71 card experts a layer): id prefix 18.7 / 19.4 / 17.9 %, rule per request
  (d 8) 39.9 %, rule continuous 77.8 / 81.0 / 79.5 %, Belady 90.9 / 91.5 / 91.1 % (prose / code / korean)
  [measured replay, rig-log `log/2026-09-28.md#residency-replay`]; ideaverify: static cross 39–50 %, in-domain
  64–77 %, adaptive 78–81 %; GLM 49.1 vs 63.4 % (rig-log `plan-triage.md:1262`).
- Qwen3.8, A6000 plan (a), prose: keeping the prompt's picks cut host slots a step 181 → 136 (P 512) and 180 → 77
  (P 4096); decode with MTP 82.48 → 91.24 and 70.91 → 79.33 tok/s [measured, rig-log `log/2026-09-30.md#q38seed-ab`],
  a fresh server (no history), so the keep branch both families share.
- Chat transfer: a corpus-ranked expert list puts 26 % of chat picks in its coldest eighth; the top experts of the last 31 decode
  steps catch 71 % of the next 32 (the round spec, 2026-09-28; source not found in the tree or rig-log, cited as quoted).

Expected, written before any run [derived from the above]: long and chat strata, V4.1: h(prompt-freeze) 0.55–0.65,
h(rule carry=serve) 0.55–0.70, h(prompt+rule) 0.68–0.78, so Δ_pr −10…+5 and **Δ_c +5…+12**; chat-short: the prompt
holds 30–127 ids (V4.1 opens no call under 128; Qwen3.8's call opens at 32 but few experts reach a count of 32 in so
few rows), so **Δ_pr ≤ −10, Δ_c < 3**. Qwen3.8 alike
(llama.cpp #27861's locality: per-layer LRU-64 ~67 %, LRU-128 ~81 % (rig-log `plan-triage.md:2371`)). Expected
outcome: branch (iii) of §5.4.

Measured by R1's replay [measured replay, `tools/flow/resplay.py` (R1), 2026-10-23, Mac CPU; budget `card=plan`
(V4.1 70–71 card experts a layer, Qwen3.8 261–262, GLM 67), 256 decode steps, the sets M0 copied to the Mac's
`~/data/bloomery-router`; h in % of card-eligible slots over [P, P + 256)]. Corpus requests are contiguous slices of
one document a set (`--prompt 64,512,4096`, every request scored), so serve order carries the previous slice's topic:
favourable to every word that carries state. `d2-chat` is the only mixed-topic set (the engine's own routes, held half).

| Set (requests) | Stratum | engine | static | prompt-freeze | rule serve | rule request | rolling:32 | p+r keep | p+r open | Belady | oracle-static |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V4.1 `d2-chat` held (24, P 16–92) | chat-short | 75.3 | 18.0 | 59.0 | 75.3 | 62.8 | 68.2 | 75.7 | 74.8 | 93.4 | 89.8 |
| V4.1 prose+code+korean (468) | chat-short (P 64) | 78.4 | 18.7 | 72.0 | 78.4 | 59.5 | 69.2 | 78.7 | 77.7 | 91.7 | 86.8 |
| V4.1 prose+code+korean (195) | chat (P 512) | 77.7 | 18.6 | 76.4 | 77.0 | 59.6 | 68.7 | 78.5 | 79.4 | 91.7 | 86.8 |
| V4.1 prose+code+korean (33) | long (P 4096) | 75.6 | 18.5 | 76.3 | 74.2 | 60.2 | 69.7 | 76.8 | 79.6 | 91.9 | 87.2 |
| Qwen3.8 `q38-prose-trace-09300532` (8 calls) | chat (P 512) | 88.4 | 53.1 | 93.3 | 86.8 | 71.8 | 80.6 | 90.2 | 94.6 | 99.5 | 99.4 |
| GLM `glm5next-prose` (156) | chat-short (P 64) | 62.3 | 23.1 | 55.1 | 62.3 | 49.1 | 54.5 | 62.5 | 62.4 | 83.9 | 71.8 |
| GLM `glm5next-prose` (65) | chat (P 512) | 60.3 | 23.1 | 57.7 | 60.3 | 49.0 | 53.2 | 62.4 | 63.0 | 83.9 | 71.7 |
| GLM `glm5next-prose` (11) | long (P 4096) | 56.2 | 23.2 | 56.4 | 56.2 | 48.8 | 51.8 | 59.6 | 62.4 | 83.8 | 71.4 |

(p+r keep / open: `prompt+rule:seed=decayed:end=keep|open`; rule serve / request: `rule:mid:carry=serve|request`.)
The twin's check: V4.1 static on `d2-chat` 18.0 % against the trace's own slot kinds 17.97 % on the same rows.

- **chat-short as expected**: Δ_pr −16.4 ± 2.7, Δ_c +0.4 ± 0.2 (`d2-chat`). The freeze words lose partly to the link:
  626 of their 836 prompt moves land after the decode starts (a 16–92-id call is too short to cover the copies).
- **chat and long missed the expected Δ_c +5…+12**: Δ_c(keep) is +1.5 ± 0.2 / +0.6 ± 1.3 (V4.1 corpus chat / long),
  −3.1 ± 3.0 (Qwen3.8), +2.1 ± 0.5 / +3.2 ± 3.1 (GLM). The wrong term is the baseline: h(rule carry=serve) is 74–77 %
  on V4.1, not 0.55–0.70, because serving never resets the map (the 09-28 "rule continuous 77.8–81.0 %" already said
  so), and prompt-freeze reaches 76 % at P 4096, not 0.55–0.65. Belady leaves 12–16 points above the best word.
- **Open beats Keep from P 512 up**: `end=open` (the call's top-n opening, then the rule) is +4.0 ± 0.9 over the engine
  on V4.1 long (line 20.60 → 19.53 ms, +5.5 % tok/s by §4.5), +6.2 ± 4.0 on Qwen3.8 P 512 (16.57 → 16.50 ms: the card
  term is in the step's shadow past the kink), +6.2 ± 3.3 on GLM long (reading (A) 38.32 → 36.41 ms, assumed
  constants). Keep is +1.2, +1.8, +3.4 there. Δ_c read with Open is in §5.4's rows below.
- **carry=request costs 12–19 points everywhere**: a per-request reset is what serving must not do.

### 5.3 Runs, sized

| # | What | Where | Minutes [derived] | Needed when |
|---|---|---|---:|---|
| M0 | list and copy the existing sets to the Mac: V4.1 corpus sets prose/code/korean (09-23), V4.1 D2 chat trace (48 prompts × 256, `routetrace` round, contexts.tsv learn/held), the Qwen3.8 `q38rtrace` traces (10-09), `glm5next-prose` | `BLOOMERY_BOX_READONLY=1 tools/box.sh 'ls -la $BLOOMERY_DATA/router …'`, then an rsync from the Mac | ~5, no card | always; which sets exist is [assumption] until listed |
| M1 | the replay of every M0 set through §4.3's words | Mac (`resplay.py replay`) | ~10 of CPU | always; gives V4.1 chat-short and corpus strata, GLM prose |
| M2a | Qwen3.8 chat: `just route-trace-chat-q38 OUT` (48 prompts, step feed, residency off) | box, A6000, inside a hold, no lease | 8–19 + ~6 build (justfile `:3488` header: 800 positions in 55.7 s, 30–70 ms a position) | only if M0 finds no Qwen3.8 chat trace |
| M2b | V4.1 long: `just route-trace-chat OUT tools/ref/data/resgen-long.tsv` (12 prompts × ~3,000 ids + 256) | box, A6000, hold | 39 k positions × 18.7 ms = 12.2 + load ~2 + build ≈ 15–17 | always (no long chat-shaped V4.1 trace exists) |
| M2c | Qwen3.8 long: `just route-trace-chat-q38 OUT tools/ref/data/resgen-long-q38.tsv` (6 of the 12 rows) | box, A6000, hold | 19.5 k × 30–70 ms = 10–23 + load 2 | always |
| M2d | GLM long (optional): `generate_glm5next --prefill steps` with `BLOOMERY_ROUTE_TRACE` per prompt | box | 6 × (3,256 × ~40 ms + a load) ≈ 13 + 6 loads | only if V4.1 and Qwen3.8 disagree; else after R4 at batch speed |

`resgen-long.tsv` (R2 authors it): 12 rows of `route-trace-chat.py`'s columns, 4 English prose (repository docs), 4 code
(Rust sources of the tree), 4 Korean (rig-log `log/*.md`), ~3,000 ids each, learn/held alternating; one TSV line a
prompt (the driver joins lines, rig-log `plan-triage.md` "What q38rtrace left"). The recipes are lead-only and take
neither the card gate lock nor the load lock: the lead runs them or gives the round the go, inside a hold, one at a
time; each runs under the driver's own bound (`--bound`, 1800 s). **Box total: 27–42 minutes if a Qwen3.8 chat trace
exists, 41–67 if not; 0 for the V4.1 chat-short answer.** No run takes the lease, so none needs a card file; the
expected values of §5.2 are the prediction each run is read against.

M3 (after R5, not part of the decisive number): the winning policy's same-lease A/B per family (§6.4 R5), and one
online trace with residency on (R4's G3 lift) to check h(online) against h(replay) on the same routes.

### 5.4 The decision table

Thresholds are on the lower bound of the 95 % interval of the paired difference over held requests; 3 points of h on
V4.1 are 0.78 ms of a ~23 ms step (+3.4 %, outside the ruler's ±1.0 % at four rounds).

| Outcome | Rule | Next action |
|---|---|---|
| (i) Δ_c ≥ 3 in every stratum, V4.1 and Qwen3.8 | prompt heat then the rule, everywhere | R5 sets `CallEnd::Keep` (or `Open`) with `seed=decayed` for every family; Qwen3.8's history return goes; the four tier families gain call heat through `prompt_pass` |
| (ii) Δ_pr ≤ −3 everywhere and Δ_c < 3 | rolling heat suffices | R5 sets `CallEnd::Return` when the rule holds a history for V4.1 too (Qwen3.8's rule wins); no call heat for the tier families; the call picks stay a prefill speed lever only |
| (iii) mixed: Δ_c ≥ 3 for P ≥ P\*, < 3 below | prompt heat above a length | R5 sets the policy of (i) for calls of P ≥ P\* and (ii) below; P\* is the replay's crossover, one constant in the family row, replacing `STREAM_MIN_P` |
| profile: h(profile + best) − h(best) ≥ 5 over the first 32 steps, chat-short | the warm profile pays where prompts are short | R6 wires `WARM_PROFILE` into every seat |
| undecided (an interval spans the threshold) | — | more requests in that stratum (M2 with 12 more rows), not a timed run |

Rows the Mac replays settle [measured replay, `tools/flow/resplay.py decide <json>` and `decide --companion
prompt+rule:seed=decayed:end=open <json>` over §5.2's runs, 2026-10-23]:

| Family, set | Δ_c by Keep | Δ_c by Open | Row |
|---|---|---|---|
| V4.1 `d2-chat` (24) | chat-short: Δ_pr −16.4 ± 2.7, Δ_c +0.4 ± 0.2 | Δ_c −0.6 ± 1.1 | (ii) below P 128 (the D2 prompts are all short); **(i) is out for V4.1** (it needs Δ_c ≥ 3 in every stratum); profile: h32 +0.2 ± 0.3 → **does not pay on short chat** (R6 not opened by this) |
| V4.1 corpus (468 / 195 / 33) | chat-short −6.4 ± 0.4, +0.3 ± 0.0; chat −0.6 ± 0.6, +1.5 ± 0.2; long +2.1 ± 1.9, +0.6 ± 1.3 | −0.7 ± 0.2; +2.4 ± 0.4; +3.4 ± 1.0 | Keep: neither (i) nor (ii) (Δ_pr not ≤ −3 from P 512). Open: undecided in long |
| Qwen3.8 prose P 512 (8) | chat: Δ_pr +6.5 ± 5.7, Δ_c −3.1 ± 3.0 | +1.3 ± 0.9 | neither (i) nor (ii); prompt-freeze ≥ the rule there, so (ii) is out at P 512 |
| GLM `glm5next-prose` (156 / 65 / 11) | chat-short −7.2 ± 0.7, +0.3 ± 0.1; chat −2.6 ± 1.4, +2.1 ± 0.5; long +0.2 ± 4.2, +3.2 ± 3.1 | +0.1 ± 0.3; +2.7 ± 0.7; +6.0 ± 2.9 | Keep: undecided in long; Open: undecided in chat, ≥ 3 in long |

So far: (i) is out (V4.1 chat-short), (ii) is out from P 512 on V4.1 (Δ_pr −0.6 ± 0.6, +2.1 ± 1.9) and Qwen3.8
(+6.5 ± 5.7; GLM chat's −2.6 ± 1.4 spans the bar), and what is left is (iii) with `Open` above a P\* between 512 and 4096, or neither. The deciding stratum is long on
chat-shaped traffic, read with Open as the companion: M2b (V4.1) and M2c (Qwen3.8), whose expected Δ_c(Open) is now
+2…+5 [derived from the corpus rows above, which favour the carried state], not +5…+12.

The chat-shaped traces decide it [measured replay over M2 route traces, 2026-10-23; `resgen-long-v41` 12 requests,
P 2,857–5,003, 52,938 positions; `resgen-long-q38` 6 requests; `q38-chat-d2` 48 requests (24 held); h over 256 steps,
the paired difference against the engine with its 95 % interval]:

| Family, stratum | engine h | static | prompt-freeze | freeze last 512 | prompt-call | +rule Open | +rule decayed Keep | rule serve | Belady |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V4.1 long (12) | 74.2 | −57.1 ± 2.5 | −22.2 ± 4.5 | −26.7 ± 4.6 | −33.0 ± 3.4 | −4.0 ± 2.6 | −0.3 ± 0.6 | −0.5 ± 1.6 | +16.8 ± 2.4 |
| Qwen3.8 long (6) | 92.5 | −41.1 ± 2.8 | −10.3 ± 5.3 | −14.1 ± 5.3 | −10.1 ± 4.7 | −2.6 ± 3.1 | −2.1 ± 2.8 | −7.2 ± 8.6 | +6.8 ± 2.1 |
| Qwen3.8 chat-short (24) | 91.8 | — | −9.9 ± 1.3 | — | −39.5 ± 1.1 | −0.6 ± 0.8 | +0.1 ± 0.3 | −0.0 ± 0.0 | +7.7 ± 0.9 |

**Verdict: (ii).** On chat-shaped requests the prompt's heat is a worse predictor of the reply's routes than the
carried rule in every stratum and both families (Δ_pr −10 … −22, Δ_c(Open) −2.6 … −4.0, Δ_c(Keep) ≤ +0.1); the last
ubatch alone is worse still (an instruction-plus-document prompt's tail does not stand for the reply). The corpus
rows' Open gain (+3.4 V4.1, +6.0 GLM) came from cutting one document into requests, which favours any carried state.
What is left is the Belady gap (V4.1 +16.8, Qwen3.8 +6.8), which no count-based policy here closes: it needs a
predictor of the next steps' routes, not another heat source. Next actions: R3 (the move to the common owner) and R7
(Muse) stay; R5 reduces to `CallEnd::Return` with a history for V4.1 (the call picks a prefill speed lever only, the
pp effect checked in its A/B); no call heat for the tier families; R6's profile is not opened.

## 6. The common cut

### 6.1 The target owner

It extends `tierengine.md` §3(a) and closes S6's "two owners stay" for the prompt half: **one count source for both
phases in the host tier, and one policy object in `runtime`**. The single-book identity of §3(a) (the NVMe books and
the rule reading one count) stays S6's step; this design gives the prompt its count source first.

- `crates/runtime/src/heat.rs` (pure, new):

      pub enum CallEnd { Return, Keep, Open { m: usize } }
      pub enum Seed { None, Decayed }
      pub struct CallPolicy { pub min_p: usize, pub floor: u32, pub fresh: CallEnd, pub with_history: CallEnd, pub seed: Seed }
      pub struct HeatPolicy { pub call: Option<CallPolicy>, pub rule: Option<SwapParams>, pub profile: bool }
      impl HeatPolicy { pub fn engine(family: Family) -> Self }   // §2.3's rows as data
      pub const CALL_FLOOR: u32 = 32;                               // the one floor (three copies today)

  and on `SwapRule`: `pub fn fold_call(&mut self, counts: &[u32], rows: u64) -> Result<(), SwapRuleError>`, the prompt's
  counts decayed as `rows` kept rows at the rule's cadence (the exact form `rr.opening` uses, `router-residency.py:
  662-697`, pinned by a fixture), used only under `Seed::Decayed`.
- `crates/gpu/src/host/heat.rs` (new, in the host tier): `HeatBook` with `begin(call)`, `note_batch(key, ids)` (from
  `serve_key`), `note_step` (the tally's walk, unchanged), `end(policy) -> CallOutcome`; it applies `CallEnd` through
  the machine: `Return` → `SwapMachine::end_call(.., kept = false)`, `Keep` → `end_call(.., true)`, `Open { m }` →
  `SwapMachine::open` (`warmprof.md` W2's machine form of `SwapRule::open`, issuing per layer at that layer's last
  unit), and `Seed::Decayed` → `fold_call`.
- `PromptStream` stays the one in-call pick driver and **V4.1's driver moves onto it** (`prefill.rs` `begin_call`,
  `call_pick_routed`, `stream_end` → `PromptStream::begin`/its pick/its end), so the call's begin and end have one
  owner and `HOSTSTREAM` becomes `XSTREAM`'s `admit` word on V4.1 (the lever row retires by name).
- `prompt_pass` (`served.rs:199`) opens and ends a `HeatBook` call for every `TierShell` family: their prompt heat with
  no body code.
- One plan-side resolution: `bloomery_levers::heat_at_plan(family, place, plan facts, levers) -> HeatPick` with one
  `HeatWhy`, replacing `ResidencyWhy`, `Residency38Why` and `GlmWhy` (and the gate-side `residency.rs`/`residency38.rs`
  pair). Per-family text stays in the family row, not in the enum.

### 6.2 The interface a model provides

| Item | Type | Exists today as |
|---|---|---|
| router pick hook | none: the batch port's ids (`serve_key`) and the step port's handoff page (`step.rs:1090`) are common | — |
| expert bytes and parts | `StackFacts` / `PlanStacks::of` / `hosted_stacks` (`swap_source.rs`) | each family's `swap.rs` (architecture facts: dense blocks, NextN layers, the r8 convert) |
| tier slots | the plan (`card_experts`, n_l a layer, tier entries); the machine reads the slot map | `ResidencySpec` built in each body (`ds41 body.rs:2387`, `body35.rs:1456`, `body38.rs:1601`, `glm5next body.rs:1356`/`:1468`, `mimo2 body.rs:627`, `minimaxm3 body.rs:426`, `solar2 body.rs:672`) |
| prompt facts | `PromptFacts { unit_rows: usize, min_p: usize }` (new; `min_p` = P\* of §5.4) | `STREAM_MIN_P` (V4.1), the ubatch (Qwen3) |
| top-k, max rows | `ResidencySpec.top_k`, `.max_rows` | as now |

### 6.3 What leaves per-model code

V4.1's call driver and keep rule (`prefill.rs:155-162`, `:759`, `:1788-1870`, `:2766`), Qwen3's keep rules
(`body38.rs:3852-3869`, `body35.rs:2476-2490`), the two floor definitions (`prefill.rs:162`, `wide38.rs:2386`),
the three Why enums and their unset/at_plan functions, the per-seat trace builders, the per-seat `WARM_PROFILE` absence
(one call in the common seat path), and V4.1's `LIVE_DELAY = 4` (one common derivation of the delay from cap × B_e,
the lane's staging rate and the step; V4.1 keeps 4 only if that derivation gives 4).

### 6.4 Steps, each one reviewable PR

| Step | Content | Class, proof | Gates (FAIL-first) | Box | Size (context) |
|---|---|---|---|---|---|
| **R1** | `tools/flow/resplay.py`, `resplay-families.tsv`, `rr.Replay` state out/in, self-test (§4.7), check-recipes row | tooling, no engine file | the self-test with its 13 cases, each mutation red first | none | M (~250–350k) |
| **R2** | M0–M2 (§5.3), `tools/ref/data/resgen-long.tsv`, the replays, §5.4's verdict appended to this doc | measurement | — | 0–67 min functional, lead's go | S (~150k) |
| **R3** | `runtime::heat` (types, `HeatPolicy::engine`, `CALL_FLOOR`), `host::heat::HeatBook`, `serve_key` counting into it, V4.1's driver onto `PromptStream`, today's policies as data | **move**: every family's picks and flips identical; `just canon-check BASE` on the scan set | `gate-gpu-ds41-callstream`, `gate-gpu-qwen38-residency`, `gate-gpu-xstream`, `gate-gpu-swap`, `gate-runtime`; FAIL-first: Qwen3.8's `with_history` set to `Keep` → its history clause red | landing batch (fixture tier) | M (~300k) |
| **R4** | the trace hook (§3.3): `note_batch`, `residency` header, `route_trace_seat.rs`, the GLM/MiMo/Solar-Open2 seats, the driver's generic server, the readers | add on the trace path; host only, ptx-scan identical | `gate-gpu-lib` (`route_trace.rs` tests: a batch call's counts equal the steps' counts on one prompt), the trace clauses of `gate-gpu-qwen4exp-e2e` and `gate-gpu-glm5next-e2e`; FAIL-first: a batch layer out of order is accepted → red | landing batch | M (~300k) |
| **R5** | the policy of §5.4's outcome for every family; `prompt_pass` call heat for the tier families if (i)/(iii) | behaviour, host dispatch path | the R3 gates re-pinned with dated reasons; a resplay twin clause: the online trace's h within ±3 points of `resplay engine:f` on its own routes | same-lease A/B per family that changes (V4.1 and Qwen3.8 at P 512 and 4096, the effect predicted by `resplay predict`) | M (~250k) |
| **R6** | `heat_at_plan` + `HeatWhy` (three enums go), `WARM_PROFILE` through `app::warm` in every seat (`tiered.rs`, `hybrid.rs`, `ds41.rs`, `qwen38.rs`, `glm.rs`), prompt counts in the profile, the common `LIVE_DELAY` derivation | move for the resolution (`gate-levers` table identical), add for the profile | `gate-levers`, `records-refresh`, each seat's serve gate | landing batch | M (~250k) |
| **R7** | Muse's attachment (§6.5) | add | Muse's family gates + the residency clauses with a Muse caller | its own | S (~150k) |

R3 and R4 both touch `crates/gpu/src/host/mod.rs` (`serve_key`), so they run in order, one owner. Of `warmprof.md`'s
rounds W1 is in (`SwapRule::with_streams` `:875`, no engine caller) and W2 is not (no `SwapMachine::open`; the
machine's only relayout is the reset's, `HostTier::swap_relayout`, `mod.rs:1776`). `CallEnd::Open` needs W2's
`SwapMachine::open`: R3 carries `Open` as data with no caller, and R5 uses it only after W2 lands, or takes W2's
machine half into its own scope (one owner of `SwapMachine`'s call end either way). R1 and R2 touch no engine file.

### 6.5 Muse Spark Open: the attachment checklist

1. **Plan shape.** `plan_host_routed` (muse §5) holds no card expert, so the unset word resolves `off`
   (`NoCardExperts`), and a two-stage plan resolves `FixedPlace`; the machine moves only a stage card's set
   (`swap.rs:22-27`). Residency needs the bp shape: the A6000 one stage with every dense layer and the KV (18.67 GB +
   2.63 GB at ctx 32768 + workspace ≈ 23 GB [derived]) and **~22.5 GB of moving experts (~1,734, 28 a layer, 10.9 %)**,
   the 3090 a tier card with **~21 GB static (~1,618, 26 a layer, 10.2 %)** [derived from 12,976,128 B an expert]. The
   machine moving the tier card too is a separate extension, priced by `resplay` (h with 28 vs 54 moving a layer)
   before anyone opens it.
2. **Card expert kernel.** The card's routed experts need an IQ2_XXS gate·up arm; muse §6 lists `_sel` present and the
   gate·up arm missing. Residency attaches only after it (muse phases a–c).
3. **`ResidencySpec`** through `PlanStacks::of` with `StackFacts { what: "muse::swap::stacks", names: routed_stacks,
   check: None, empty: false (62 MoE layers), map_layers: None, unrouted: [] }`; no convert (file bytes, no r8);
   `top_k` 8; `max_rows` the seat's; `LIVE_DELAY` the common 2 (a plan's cap of 24 flips × 13.0 MB = 311 MB is 11.8 ms
   of the link against two passes of ~55–80 ms [derived]); `DEADLINE` common; churn `Hosted` (the host holds every
   expert anyway, so the pool costs no extra RAM).
4. **Body seam.** `TierShell` (`shell.rs:115`), no `residency_parts` of its own.
5. **Seat.** `tiered.rs` (`residency_pick` `:180`, `settle_pick` `:212`); after R6, `heat_at_plan`'s family row only.
6. **Prompt heat.** Through `prompt_pass` and `HeatBook` (R3/R5), with `PromptFacts { unit_rows, min_p: P* }` from
   the family row; no body code.
7. **Trace.** `route_trace_seat::open` (R4); `route-trace-chat` with `bloomery-serve --model <muse file>` gives its first
   chat trace, and `resplay-families.tsv` gains its row (256 experts, top-8, 12,976,128 B, layers 0–61, the bp budgets).
8. **Gates.** `crates/gpu-gates/src/bin/shared/residency_clauses.rs` with a Muse caller, as GLM's
   `gate_glm5next_residency` does.

## 7. Prediction

### 7.1 The formula

    tok/s = 1000 / [ t_serial + N_s·(1 − h)·c_h + κ·N_s·h·c_c + F·c_f ]        while the host leg dominates each layer

with h the card share (stage + tier), F flips a token, κ ∈ {1 (A), 0 (B)}; past the kink (the card's shadow above the
host leg in a layer) the slot's value falls toward 0.

### 7.2 V4.1 (the check)

`t = 38.94 − 25.99·h + 0.0993·F` (A6000, plan (a), P 512, depth ≤ 608). At the clip's arms it predicts 29.07 / 36.65 /
43.64 tok/s against 29.34 / 36.05 / 43.87 measured (mean step). For the policies of §4.3: h 0.75, F 4 → 19.85 ms, 50.4
tok/s; Belady's h 0.91 at F 4 → 15.7 ms, 63.7 tok/s (a ceiling; Belady's flips are free) [derived]. The first R5 A/B
on V4.1 checks the line: the online arm's mean step within ±0.6 ms of the line at its own (h, F) from `stat step` and
its flips; a miss names t_serial (depth, P 4096: off 34.34 ms vs the line's 34.40 at h 0.17 is the depth term's
first reading [measured, data.json `tg_tps` p4096]) or c_f.

### 7.3 Muse Spark Open (bp shape, n = 1, depth ≤ 4096) [derived]

t_serial = 29.3–36.5 ms (card dense 26.7–32.9 + bridge 1.3–2.2 + launches 1.4, muse §5); N_s = 496; reading (A)
c_h = 105.5 µs (123 GB/s, V4.1's (A) rate), c_c = 21.6 µs; reading (B) c_h − 0 = 83.8 µs (V4.1's slope per byte).
F ≤ 6 a token adds ≤ 0.46 ms.

| h | (A), t_serial 29.3 | (A), 36.5 | (B), 29.3 | (B), 36.5 |
|---:|---:|---:|---:|---:|
| 0 (residency off, `plan_host_routed`) | 81.6 ms, 12.3 tok/s | 88.8 ms, 11.3 | 70.9 ms, 14.1 | 78.1 ms, 12.8 |
| 0.109 (A6000 pool, static) | 77.1, 13.0 | 84.3, 11.9 | 66.3, 15.1 | 73.5, 13.6 |
| 0.21 (both cards, static) | 72.9, 13.7 | 80.1, 12.5 | 62.1, 16.1 | 69.3, 14.4 |
| 0.5 | 60.8, 16.4 | 68.0, 14.7 | 50.1, 20.0 | 57.3, 17.5 |
| 0.7 | 52.5, 19.0 | 59.7, 16.7 | 41.8, 23.9 | 49.0, 20.4 |
| 0.85 (near the kink) | 46.3, 21.6 | 53.5, 18.7 | 35.5, 28.1 | 42.7, 23.4 |

Which h Muse reaches is the replay's answer on its own trace (§6.5 item 7); V4.1's chat-free corpus replays put a
moving pool of 18 % a layer at 0.78–0.81 continuous, and Muse's 28 moving + 26 static a layer of 256 (21 %) is
[assumption] comparable, so 0.5–0.75 on long prompts is the working band.

### 7.4 Premise corrections

- `muse-spark-attach.md` §5: "Wall = the resources' max once the host expert window and the card layers overlap (the
  ds41overlap lever) ≈ 20–21 tok/s". The decode chain does not overlap a layer's front and back with its host leg (§1.3;
  the `ds41overlap` design in rig-log `plan-triage.md:1168` is a prompt window), and `BLOOMERY_HYBRID_OVERLAP` is retired.
  At h = 0 the derived decode is 11.3–14.1 tok/s; ~20 tok/s needs h ≈ 0.5–0.7. Its "≈ 27 tok/s with the 22 % fill" is
  12.5–16.1 tok/s by the same correction.
- `muse-spark-attach.md:108` names "glue like `crates/gpu-glm5next/src/swap.rs`"; that file is only `StackFacts` over a
  common helper, and §6.5 item 3 needs no file of Muse's own beyond its facts.
- `router-residency.py gen`'s arms (b), (c) and `steady` are not the engine (§2.3); the 39.9 % and 77.8–81.0 % rows
  bound the engine's served h from below and above, they do not estimate it.

## 8. Open questions (each with the run or replay that settles it)

| Question | Settled by | Expected |
|---|---|---|
| the decisive Δ_pr, Δ_c per family and stratum | R2 (§5) | branch (iii), P\* in 128–512 [derived, §5.2] |
| reading (A) or (B) | the first Muse decode at two h (static off/on), or a V4.1 arm at h > 0.8 | — |
| batch counts vs step counts on one prompt (prefill kernels near router ties) | R4's `gate-gpu-lib` clause | ≥ 99 % of cells equal (`warmprof.md` §8 Q3's expectation) |
| the profile's value on short chat | R2's `profile` arms on D2's learn/held split | under 5 points [assumption: the rule recovers within ~40 rows] |
| a tier-card machine for Muse | resplay with 28 vs 54 moving a layer | Δh ≥ 5 points to open it |

## 9. Prior work

The designs followed and where ik, exllamav3 and mistral.rs diverge are `tierengine.md` §9 (Mixtral offloading, Fiddler,
MoE-Infinity; none of the three reference engines moves experts at run time). The rule is Strata's evict-then-admit
(`swaprule.rs`; `router-residency.py` RULES `strata`). Prompt heat at the end of prefill as the decode's seed, with the
engine's own keep rule replayed against the rolling rule on held-out chat, is this design's; no outside source was read
for it.
