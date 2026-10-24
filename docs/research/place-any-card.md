# Placement on any card set — every byte a load puts on a device is in that device's plan

A design (round placegen, 2026-10-23). It changes no engine code. The evidence, family by family with `path:line`, is
[`placement-census.md`](placement-census.md); this file cites it as "census §n". Numbers marked [derived] are arithmetic
over pins and measured constants named beside them; nothing here was run on a card.

The ask (the user, 2026-10-23): two-card handling and one-card draft handling are thin; make placement general, judge
the draft plan, and improve placement seriously for the long term.

## 0. The draft plan, judged

The draft (Korean, this file's previous text) holds in its core and is kept:
- **The invariant** (§1), and its consequence that experts pushed out by a reserve go to the host as the tail of the
  same prefix, so a bit change is the host-kernel/card-kernel difference, not a new value.
- **No role enum.** A device's job follows from its budget; the stage is device 0.
- **One function owns reserves**, taking a device index, a name and bytes.
- **The index runs over the whole device list.** Today's `TierDraft.on` is a tier-list index, so 0 means both "the
  first tier" and, with no tier, "off the plan" (`workstation.rs:495`, `generate.rs:764`); that double meaning is what
  refuses a one-card draft plan (`plan_a_draft` → `DraftOffPlan`, `workstation.rs:300`, `placement.rs:1213`).
- **`a` / `gate` / `bp` are picks** (`crates/placement/src/placement/devices.rs:239 ALIASES`), not machines.
- **The default moves only on a prediction**, and layer cut, tp, adaptive residency and draft kernels stay out.

What it got wrong or missed:
1. **Its stage 1 contradicts itself.** It promises that the A6000's expert count does not move (draft line 42) and also
   closes the `--place a` path that hides the draft on the 3090 (line 54); its default rule (line 55) covers only `bp`
   and one card. V4.1's unset default on two cards is `a` (`shared/ds41_place.rs:30-48`, break-even `None`), so as
   written the draft would land on the A6000 and the default would move by ~515 experts. §3 resolves it: the 3090
   joins the plan as a reserve-only member, and the A6000 keeps 2,668.
2. **"No draft-only mode" (line 34) needs one refinement.** Today the planner refuses a tier with nothing to hold
   (`IdleTier`, `placement.rs:2425-2431`) and has no cap on a card's experts. A reserve-only member is not a mode; it is
   a device whose expert cap the chooser sets to 0 (§2.4).
3. **It misses the larger defect: the scalar reserve taxes every card.** `plan_routed_reserving`'s unnamed `reserve`
   comes off every tier's budget too (`placement.rs:2268,2341-2479`), and on a pool it is counted once per card
   (`:2472-2477`). GLM's front, NextN and beside bytes, and Solar's prompt reserve, are paid again on the 3090 (census
   §2; Solar ≈ 63 experts [derived]).
4. **It misses the off-plan seats.** An image tower or embedding model takes the smallest card *outside* the plan with
   reserve 0 (`crates/gpu-gates/src/vision_seat.rs:659-690`, `side_seat.rs:102-114`). Under `a` on this machine that is
   the 3090, the DSpark draft's card too, and nothing adds the two.
5. **It misses one card.** With no tier, the unset draft card is the visible device of the fewest bytes
   (`generate.rs:2537-2548`): under `gate`, or on any one-card machine, that is the stage itself, unreserved; an 8.6 GB
   draft then has only the plan's 1 GiB margin (`workstation.rs:112`) [derived].

The lead's review points, checked against the code:
- Right: the invariant, the double meaning of `TierDraft.on`, the code facts (`generate.rs:756,764,851`,
  `ds41_dspark.rs:62,76`, `qwen35moe/place.rs:1255`, `beside.rs:41`), the contradiction (item 1), the name collision
  (`beside::reserve_on` puts reserves into a machine; `PlanInputs::reserve_on`, `place_hybrid.rs:236`, computes
  bytes), and the post-plan hole (`workstation.rs:264 placed_on` has no term for Muse's prompt pass,
  `musespark/body.rs:854`).
- **Overturned: solar2 does not refuse a tier card.** Its description carries a tier reserve (`solar2/place.rs:190-194`),
  its test plans `bp` (`:244-290`), its seat serves `--place bp`; only its unset rule is untiered. nemotron_h_moe, not
  on the lead's list, does refuse (`hybrid.rs:564`). The refusing set is mimo2, minimaxm3, raon1, nemotron_h_moe (missing
  pieces), nemotron_h (an architecture fact), and musespark until muse2card lands (census §1).
- **Overturned: tier legs are not per-model adapters.** `bin/shared/routed_legs/*` are the adapters of the plan-legality
  gate `gate_routed_legs` (`gate_routed_legs.rs:13-17`). The tier leg is already one common owner: `LegTier`
  (`crates/gpu/src/host/leg_tier.rs`) over the common card leg (`crates/gpu/src/card_leg.rs`), each family giving a
  ~80-line `LegFamily` adapter (Qwen3, Qwen35 `tier_leg.rs:124,187`; GLM `gpu-glm5next/src/tier.rs:39`; Solar
  `gpu-solar2/src/tier.rs:41`). Two families keep their own: V4.1's `Ds41Tier` and Qwen3.8's `Tier38`.
- **Refined: choosers are already one function.** `Place::choose` (`generate.rs:590`) takes a family's `TierRule`;
  what is per family is the break-even number and its card. muse2card lifts the hook into the common `Tiered` seat.

## 1. The invariant

The bytes a load puts on a device leave that device's budget before any expert is placed there. The device a resident
lives on and the device its reserve is taken from are the same device. No load puts bytes on a device outside the plan.
No draft, buffer or side model loads without a reserve.

Experts a reserve pushes out go to the host as the tail of the same prefix; the cards keep the front of the prefix
they held before.

## 2. One owner per concept

### 2.1 Devices

**The plan's input is one resolved device list**, census-backed (`devices.rs:105 spec_of_device`, `:409 resolve`).
`--place` is a list of picks (`a`, `gate`, `bp` and `<stage>[+<card>…]` resolve through `ALIASES` and `word_picks`,
`devices.rs:239,301`); unset, the chooser picks (§2.4). The serving path sees only the resolved list, never `plan_a`,
`plan_gate` or `plan_bp`; those stay as names of `plan_tiers` over this workstation's cards for tests and the README's
gate plan.

**Roles follow from the budget, by position and cap:**
- device 0 is the stage: every layer, the head, KV, the stage's named reserves, then experts in what is left;
- every later member holds its named reserves, then the next ids of each layer up to its **expert cap**: `None` fills
  the budget (a tier, today's behaviour), `Some(0)` holds reserves only (a reserve-only member: a draft's or an
  encoder's home);
- a member with neither experts nor reserves is idle and refused (`IdleTier`, kept);
- the host and the NVMe tier take the rest, as today.

A reserve-only member is not a tier: the body attaches tier service (`TierCard`, the leg) only to members holding
experts, and `refuse_tier_count` (`crates/gpu/src/host/mod.rs:160-175`) counts those.

**Index type.** `placement::Device` (`placement.rs:498`) already is "where a segment lives", with `Card(i)` over
`Machine::all_cards`. A reserve is a segment, so it takes `Device` as is: `Card(i)` over the whole list (0 = stage) or
`Host`; `Nvme` and `Unused` are refused by name. No new index type.

**How it extends** (each already decided where cited; this design adds only the reserve's device):
- **N cards.** `plan_tiers` already takes a tier list (`workstation.rs:509`) and the slot map names 8 tiers
  (`crates/gpu/src/host/slots.rs:17`); the runtime serves one (`SERVED_TIERS = 1`, `host/mod.rs:160`, "N >= 2 lands
  with V4.1's `_tiers` handoff"). The chooser offers each prefix of the census by size up to the served count.
- **Unified memory (GB10).** One `Pool` device per node (`docs/research/tierengine.md` §3(l): "per device `{kind: Card
  | Host | Pool, capacity, bandwidth, reclaim order}`, on GB10 one `Pool` device"); residency on GB10 per §8. A named
  reserve has one device, so the per-card double count of the scalar on a pool (`placement.rs:2472-2477`) cannot be
  written.
- **Multi-node.** A per-node engine over a static placement split (tierengine §3(k)): each node plans its own device
  list; a reserve names a device of its node. This design does not touch the split.
- **Placement is the load-time solver; the tier engine is the runtime** (tierengine §3(l)). Adaptive residency moves
  slots inside the budgets this plan sets (`docs/research/residency-general.md`); a budget a reserve shrinks is a
  smaller slot count to it, not a new rule.

### 2.2 Reserves

**One owner**: `Reserve { on: Device, name: &'static str, bytes: u64 }`, entered by `Machine::reserve(Reserve) ->
Result<(), PlacementError>` (refusing a device past the list — today's `DraftOffPlan` generalised to
`ReserveOffPlan { on, devices }` — and `Nvme`/`Unused`), stored where today's named reserves live (`Card::reserves`,
`Host::reserves`, `placement.rs:228,309`) and taken out by `set_aside_bytes` (`:255`).

Everything that puts bytes on a device goes through it:

| Today | Becomes |
|---|---|
| the scalar `reserve` of `plan_routed_reserving` / `plan_host_routed_reserving` / `nodes::loaded_routed` (`placement.rs:2268,2308`; Fill's `reserve`) | removed; the callers enter named stage reserves |
| `TierDraft { on: tier index }` and the draft argument of `tier_card` (`workstation.rs:185,495`) | `Reserve { on: Card(i over all cards), name: DRAFT_RESERVE, .. }` |
| `Place::machine`'s refusal of draft bytes with no tier (`generate.rs:749-759`) and `draft_card` = first tier (`:851`) | gone; the chooser names the draft's device (§2.4) |
| Qwen3.8's two MTP paths (scalar on one card, named `MTP_RESERVE` on a tier machine: `qwen35moe/place.rs:744-800,1255`) | one named stage reserve |
| GLM's front, NextN card bytes + `NEXTN_ARENA_BYTES`, beside (`glm5next/place.rs:343,670,736,1044`) | named stage reserves |
| `place_hybrid`'s `PlanInputs::reserve_on(&Machine) -> u64` (`place_hybrid.rs:236`) | `PlanInputs::reserves(&Machine) -> Result<Vec<Reserve>, PlaceError>` (stage), keeping the `PlaceError::Tier` refusal of a body with no tier leg |
| `beside::reserve_on(&mut Machine, Beside)` (`beside.rs:41`) | `beside::enter(&mut Machine, Beside)` entering typed reserves (the name collision goes) |
| off-plan encoder / embedding seats, reserve 0 (`vision_seat.rs:659`, `side_seat.rs:104`) | a reserve on a plan member (a reserve-only member when no experts are chosen there) |
| post-plan buffers: Muse's prompt pass, MiMo2's prompt batch, Nemotron-H-MoE's prefill, Qwen3.6's DSpark from free room (census §2) | a named stage reserve with one byte formula in the family's `place.rs`, the body holding its allocation to it as `refuse_scratch_past` (`crates/gpu/src/model.rs:2221`) holds scratch; `largest_fit` (`fit.rs:27`) then picks rows inside the reserve, not inside the free room |

`FIT_RESERVE` (256 MiB, `fit.rs:19`) stays for what it is: slack inside a planned reserve, never a plan term.

**The record.** The `plan` record (`crates/gpu-gates/src/record.rs:3392`) prints, per device in list order, its role,
its expert count and `reserves=<name>:<bytes>,…`; a reserve entered and not printed fails the record's test. Its kind
change runs `just records-refresh`.

### 2.3 Tier leg

**One leg**: `LegTier` over the common card leg. What a family gives is data and one adapter:
- the expert shape: its input width (the trunk's `n_embd` or a latent, as Muse and Nemotron-H-MoE), `expert_ff` per
  layer, `n_used`;
- the quant binding per stack (gate·up, down), already data in `accepts.rs` (`leg_arms(Program)`, `routed_rule`:
  `DS41_LEG`, `BOLT_LEG`, `MUSESPARK_LEG`, `planes_leg_*`, `card_gate_up_runs` / `card_down_runs`,
  `resident_down_runs`);
- the activation rule (gated SiLU today; an ungated relu² arm for Nemotron-H-MoE is new data, not a new leg);
- what the stage does with a tier row (the post-norm and scale Muse applies, `musespark/body/tier.rs` in muse2card) —
  the one part that stays the model's, because an architecture fact forces it;
- the plan's side: `tier_batch` through `workstation::tier_leg_batch_bytes` (`workstation.rs:465`) and the stage's tier
  places as a named stage reserve (`tier_places_bytes`).

What each refusing family needs (census §3):

| Family | Needs | Cause class |
|---|---|---|
| musespark | nothing past muse2card | — |
| mimo2 | `LegFamily`, tier places, `tier_batch`, `TIER_RULE` | missing pieces |
| minimaxm3 | the same, with the NVFP4 planes binding on the tier | missing pieces |
| raon1 | the same, plus a Q6_K down arm on the card leg | missing pieces + binding |
| nemotron_h_moe | the same, plus an ungated relu² arm; drop `hybrid.rs:564`'s refusal for this body | missing pieces + binding |
| nemotron_h | none: no routed experts (`hybrid.rs:480`) | architecture fact |
| qwen3moe / qwen35moe seat | a pass of several slots beside a tier (`serve_seats/qwen3.rs:1014-1019`); plan and leg exist | runtime piece |

`Ds41Tier` and `Tier38` either converge on `LegTier` or name, beside the code, the measured speed gain that keeps them
(AGENTS.md "Common code first"). The census found no such note.

### 2.4 Chooser

**One function**: `Place::choose(flag, census, TierRule, tier_of)` (`generate.rs:590`); `choose_untiered` (`:565`) is
`choose` with `TierRule::UNTIERED`, as muse2card's `Tiered` seat already does. Every seat calls it; a family supplies a
`TierRule` and its card.

**The tier decision, from the cost model.** Decode is host-bound (the card waits on the host leg). One more expert on a
card saves, per token [derived, uniform routing over ids; a hot-ordered prefix makes the next ids hotter, so this is
the floor of the saving]:

    s = (n_used / n_expert) × B_host_expert / BW_host,  BW_host = 135..137 GB/s (rig-log plan.md, cost model)

and a tier costs a fixed `c` a pass (V4.1's `c_two` 1.9..2.5 ms over plan (a), `docs/cards/glmbp-ab.card:28`; Qwen3.8's
bp step +70 µs outside the layers, `docs/cards/q38bpbug-ab.card:86-87`). The break-even is `E* = ⌈c / s⌉`; the tier is
kept when its plan holds at least `E*`. A family's card derives its own `E*` over its grid (GLM 526, Qwen3.8 1,152 are
this, done in full); a sitting's measured verdict overrides the formula, and the gap between them is that card's
finding.

| Family | s per expert [derived] | E* at c = 1.9..2.5 ms | Tier holds (this machine) | Unset |
|---|---|---|---|---|
| V4.1 (6/384, 16.9 MB) | 1.94 µs | 980..1,290 | 1,392 (no draft), 881 (draft) | `a` — measured: bp not faster (README); the formula reads +0.2..+0.8 ms, inside or near the ruler |
| Muse (8/256, 12.98 MB, muse2card's card) | 2.96 µs | 642..845 (24 at c = 70 µs) | ~1,820 (muse2card's card) | `bp` either way |

**The draft's home, the same table.** A draft (or any resident) of `D` bytes goes on the plan member where it displaces
the fewest host-ms: `cost(dev) = ⌈D / B_card_expert⌉ × s` on a member that holds experts, 0 on a reserve-only member,
which the chooser may add from the census when a card is in view and unplanned. The stage is the last resort; it is
the only candidate on one card.

## 3. Two cards at `a` with DSpark — resolved, with the counts

V4.1 on this machine (pins, `crates/model/tests/placement.rs`): plan (a) holds 2,668 experts on the A6000 at
16,773,120 B each; the draft holds 8,629,780,480 B (the `bp` tier's reserve with the draft, 8,782,291,976, less the
tier batch's 152,511,496).

| Option | A6000 experts | Host ms a token [derived] | Verdict |
|---|---|---|---|
| (i) draft on the A6000 stage | 2,668 − ⌈8,629,780,480 / 16,773,120⌉ ≈ 2,153 (−515) | ≥ +1.0 (515 × 1.94 µs), ~4 % of a ~23.8 ms step | rejected |
| (ii) the 3090 joins the plan as a reserve-only member holding the draft | 2,668 (unchanged) | 0 | **chosen** |
| (iii) `bp` with the draft (tier 881) | stage as `bp` | tier saves 881 × 1.94 = 1.71 ms against c = 1.9..2.5: −0.2..−0.8 | rejected while `E*` > 881 |

(ii) keeps the draft plan's promise (no A6000 count moves) and closes the hidden path: the same device as today, now
in the plan, its reserve checked against what loads (`open_dspark`'s resident-over-reserve check runs on every
placement), and its sum with an image tower on the same card held by the plan. `BLOOMERY_DSPARK_CARD` may name only a
plan member (or a census card the chooser can add as one); the draft's pass runs where every measured DSpark row ran
it, so its time does not move.

**One card**, DSpark asked: the stage is the only home. An A6000 alone holds ~2,153 experts instead of 2,668; a 3090
alone (`gate`, 1,146) holds ~631 [derived, the same 515]. That is a new plan where today's has no room for the draft.
DSpark stays opt-in (`BLOOMERY_DRAFT=dspark`), so no unset default moves.

## 4. What stays out, and why

- **Layer cut (`plan_b`, `CUT`) and tp.** They split capacity, not experts by budget; one is a gate-only staging plan
  (`gate_load_v41.rs:248`), the other GLM's own machine. A reserve on them goes through the same owner; their rules do
  not change.
- **Adaptive residency's rules.** It moves slots inside the budgets this plan sets (tierengine §3(l)); this design
  only sizes the load-time prefixes.
- **The multi-node split.** Static per load, decided in tierengine §3(k).
- **Draft kernels, acceptance, quantisation, the card table** (the census already gives totals).
- **NVMe and pool for the `place_hybrid` families** (`PlaceError::NoNvmeTier`): a residency/tier-engine item, not a
  placement one.

## 5. Stages

Each stage is one round. "Reach" is `python3 tools/recipes.py reach --brief` (build-free) over the stage's main files;
a landing list is still derived by `just affected` on the finished diff. No stage blocks 0.3.9.

**S0 — muse2card (in flight, 0.3.9).** Absorbed as is: it uses `place_hybrid`'s tier path, the common card leg's
IQ2_XXS binding, `tier_leg_batch_bytes`, one `pass_rows_max` formula, and lifts `TIER_RULE` and `machine` into the
common `Tiered` seat. Two notes, neither blocking: (a) the scalar charges its tier places on the 3090 as well, ≤ 0.2 MB,
under one 12.98 MB expert [derived], so S1 is not on two-card Muse's path; (b) `break_even: Some(1)` reads "no fixed
cost"; the formula gives 642..845 at V4.1's `c_two` (24 at Qwen3.8's 70 µs). The default on this machine is `bp`
either way; on a tier card holding fewer than ~640..845 experts the two differ. Write the derived `E*` into its card and
rule if it costs one constant; otherwise S4 does it. The prompt pass's ~0.6 GB stays post-plan until S2.

**S1 — the reserve owner** (after the 0.3.9 cut; one train, fixture tier). Files: `crates/placement/src/placement.rs`
(`Reserve`, `Machine::reserve`, `ReserveOffPlan`, expert cap, scalar removed), `placement/workstation.rs` (`tier_card`,
`TierDraft` → `Reserve`, `plan_a_draft` → `plan_tiers` with a device index), `crates/model/src/arch/place_hybrid.rs`
(`reserves`), `glm5next/place.rs`, `qwen35moe/place.rs` (one MTP path), `deepseek41/place.rs`, `crates/gpu-gates/src/
generate.rs` (`machine`, `draft_card`, `draft_spec`), `bin/shared/ds41_dspark.rs`, `ds41_draft.rs`, `beside.rs`
(`enter`), `record.rs` (`plan`), `bin/gate_deepseek41_dsloop.rs`, the pins in `crates/model/tests/placement.rs`,
`glm5next_meta.rs`, `solar_open2_meta.rs`, `qwen4exp_meta.rs`; the lever text of `BLOOMERY_DSPARK_CARD`
(`crates/levers/src/registry.rs`), `docs/serving.md`, `docs/HARDWARE.md`.
FAIL-first (each red on base, green after):
1. A 1 GiB stage reserve on `plan_bp(layers, None, batch)` leaves `tier_n_l` identical; on base the tier loses
   ⌈2^30 / 16,773,120⌉ = 65 experts [derived].
2. On a pool machine (`workstation.rs:675-718` helpers) a reserve R lowers the pool's expert bytes by R, not R × cards.
3. `--place a` on the two-card census with the draft: the 3090 is a reserve-only member with `DSpark draft`
   8,629,780,480 B and 0 experts; the A6000 holds 2,668, prefix byte-identical to plan (a). Base refuses
   (`generate.rs:756`).
4. One card with the draft: the stage carries it, experts = plan (a) − 515 (± 1); `ReserveOffPlan` only past the list.
5. `BLOOMERY_DSPARK_CARD` naming a device outside the plan is refused by name (base accepts it under `a`).
6. The `plan` record prints every device's reserves and experts.
7. GLM and Solar `bp`: the tier gains ⌊(stage reserves) / B_card_expert⌋ (± 1) — Solar ≈ 63 [derived]; the GLM pin band
   (`gate_glm5next_twocard.rs:236`) re-pinned with a `PIN(date)` line and that derivation.
8. Qwen3.8 one card with MTP: the named path gives the same plan as the scalar did (a move).
Proof class: no kernel moves (`just canon-check BASE` identical); byte formulas move, so the whole gate list runs.
Reach: `placement.rs` and `workstation.rs` 205 of 210 gates; `place_hybrid.rs` 67, `generate.rs` 69, `record.rs` 70,
`glm5next/place.rs` 59, `beside.rs` 37. Default moves: GLM and Solar `bp` tier +k experts (decode +k × s, a fraction of
a ms [derived]); V4.1 at `a` with DSpark: 0 experts move, the 3090 enters the plan; V4.1 on one card with DSpark: −515
on the stage (a plan where today has none). Records: `just records-refresh`.

**S2 — post-plan buffers into reserves** (after S1). Files: `crates/model/src/arch/{musespark,mimo2,nemotron_h_moe}/
place.rs` (one byte formula each), the bodies (`crates/gpu/src/arch/musespark/body/prefill.rs`,
`crates/gpu-mimo2/src/prefill.rs`, `crates/gpu/src/arch/nemotron_h_moe/body.rs`), a `refuse_reserve_past` beside
`refuse_scratch_past` (`crates/gpu/src/model.rs:2221`); Qwen3.6's whole-card DSpark fit counts the draft
(`serve_seats/qwen3.rs:686`); the solar2 and minimaxm3 seat docs reconciled with their descriptions. FAIL-first: the
body's made bytes past its reserve are refused by name; a plan whose stage has room only for the plan without the
buffer is refused at plan time (today at the allocation or by a smaller pass). Default moves: Muse's stage −⌈0.6 GB /
12.98 MB⌉ ≈ 47 experts, +0.14 ms a token on 64..73 ms (−0.2 %) [derived from the lead's 0.6 GB]; pass rows become a
plan output. Reach: Muse's `prefill.rs` 19 gates; the family gates of the three.

**S3 — off-plan seats into the plan** (after S1; disjoint from S2). Files: `crates/gpu-gates/src/vision_seat.rs`
(`seat_of`), `side_seat.rs`, `beside.rs`. FAIL-first: under `a` on two cards the tower's card is a plan member with its
reserve, and a draft + tower that do not fit the 3090 together are refused by the plan (base: never summed). Default:
same devices, now printed.

**S4 — one chooser everywhere** (after S0 lands). Files: `generate.rs` (`choose_untiered` as `choose` with
`UNTIERED`), `serve_seats/{hybrid,nemotron,raon1,qwen3}.rs`, each family's `TIER_RULE` with its card's `E*`, and the
draft's home by §2.4's table. Default moves only where a card predicts one: Solar's unset becomes `bp` only if its card
derives `E*` under its tier count (Solar's `n_expert` and `c` decide; not computed here); Muse's rule takes its derived
`E*` (no move on this machine).

**S5 — tier legs for the refusing families**, one round each, by demand: minimaxm3 (the 0.3.9 headline model, natural
next), raon1 (Q6_K down arm), mimo2, nemotron_h_moe (ungated arm); the qwen3 seat's several-slot pass beside a tier;
`Ds41Tier`/`Tier38` converge or name their reason. Each: the adapter, tier places, `tier_batch`, `TIER_RULE`, a two-card
arm of the family gate, its decode card.

**S6 — N ≥ 2 tier cards**: the runtime's `SERVED_TIERS` with V4.1's `_tiers` handoff; the planner and chooser already
take N after S1/S4.

Order: S0 → 0.3.9 cut (by 10-26) → S1 → S2 ∥ S3 → S4 → S5 (per model) → S6. S1 touches 205 gates, so it rides one
fixture-tier train with whatever else touches `crates/placement` after the cut, never an express.
