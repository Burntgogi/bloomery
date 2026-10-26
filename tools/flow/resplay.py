#!/usr/bin/env python3
"""Request-driven residency replayed over router sets: the engine's own policy as a twin, prompt heat, a rolling
window, a profile split and tiers past the card, each priced in ms a token by its family's cost model.

    tools/flow/resplay.py replay <family> --data DIR <set>... [--split held|learn] [--prompt P,...] [--steps 256]
                          [--budget card=plan|nN[,pool=K][,pinned=P][,tier=N][,ram=G[:tick|heat]]]
                          [--policies W1,W2,...] [--json PATH]
    tools/flow/resplay.py predict <family> --hit H [--flips F] [--nvme H_N] [--depth D]
    tools/flow/resplay.py decide <json> [--companion WORD]
    tools/flow/resplay.py --self-test

The design is docs/research/residency-general.md §4; its words are the policies below. Families: v41, q38, glm
(replay and predict) and muse (predict only: no router set exists). The rule, the opening, Belady, the static count,
the set and trace readers are tools/ref/router-residency.py's (`rr`, pinned to crates/runtime/src/swaprule.rs by its
fixtures); this tool chains rr's replays request by request (rr.replay's state0) and adds what rr has no form of: the
prompt call's pick (SwapRule::call_pick), the call's end, the rolling window, the profile seed, the tiers and the
prices. Qwen3.8's price is tools/flow/q38_step.py's calibrated step; the constants no other owner holds are rows of
tools/flow/resplay-families.tsv.

A chain is one serve order over a set: a route trace's requests from its contexts.tsv (learn/held split, genre) or
its call rows; a corpus set (no call rows) cut into requests of P prompt + --steps decode positions for each P of
--prompt (64,512,4096 by default; the cut ignores the set's chunks). A request's prompt call is its call rows (one
pass, no row kept: crates/gpu/src/host/served.rs prompt_pass); its decode steps are the positions after it, one pass a
row, each kept. h is the share of a scored decode row's routed slots (the family's card-eligible layers) on a card
(the stage card, or a tier card's static set) when its pass starts; a request is scored over its first --steps decode
rows, h16 over the first 16, h_tail over the rows from 64 on.

Policies (--policies, comma separated; `engine` is always run, the paired differences are against it):
    static                       the seed's set at the card's slots, never moved (residency off)
    prompt-freeze[:last=N][:m=M] at each call's end SwapRule::open's relayout (rr.opening) of the current set by the
                                 prompt's counts (its last N rows'), at most M flips (all by default); no rule after
    prompt-call[:min_p=X][:floor=Y]   per group of the call, SwapRule::call_pick from its first batch's counts at
                                 floor Y a batch of the group; kept; no rule after (the family's batch and group)
    rule[:mid|knee|strata][:carry=serve|request]   the decode rule; serve keeps the map and the decayed counts from
                                 request to request in trace order, request resets both to the seed at every request
    rolling:W[:every=N]          at every N-th kept row (the rule's 4) the moving set heads for the top of the last W
                                 decode rows' counts: rr's pairing at margin 1, min count 1, no decay, cap and spares
    prompt+rule[:mid|knee|strata][:seed=none|decayed][:end=keep|return|open][:min_p=X]   the call's picks (end keep or
                                 return) or SwapRule::open at the call's end (end open), then the rule (carry serve);
                                 return hands the picks back when the rule holds a decode history; seed=decayed folds the
                                 prompt's counts, decayed at the rule's cadence (rr.opening's form), into the rule's
    engine                       the family's row of the design's §2.3 table: v41 prompt+rule:end=keep:min_p=128, q38
                                 prompt+rule:end=return:min_p=32, glm rule:mid:carry=serve
    belady, oracle-static        ceilings at the card's slots over each scored window: MIN with bypass from the window's
                                 own best start (its first-used ids), and the window's own top-n static set
    any word @profile[:decode|both]   the seed at load is the ranking of the learn requests' decode (and prompt) counts;
                                 scored on held requests only

Moving policies hold the card's slots less the spares live; static and the ceilings hold every slot. A flip planned at
boundary b is live from pass b + the family's live delay. A prompt move's copy is issued at its layer's place in its
group's walk (the call's wall from the family's prompt rate) and copies serially over PCIe at pcie_gbps; one that
lands after the call is live from the first decode step after it (a late landing; its victim serves until then).

Output: per (stratum, policy) a markdown row: requests, h, h16, h_tail (percent), flips and MB a token, prompt moves,
GB and link seconds a request, late landings a request, NVMe share, ms and tok/s by the family's cost model (both
readings where they differ), and h minus engine's h over the paired requests with its 95 % interval (points).
Strata: chat-short P < 128, chat 128..1023, p1k 1024..2047, long >= 2048, each also by genre. `decide` prints the
design's §5.4 row per family from a replay's --json: thresholds on the 95 % interval's bound; `--companion WORD` reads
Δ_c by another replayed word (§5.1's is prompt+rule:seed=decayed:end=keep; §5.4 (i) names Keep or Open).

Refused by name (exit 2): a family not in the table, a set without its `# complete` trailer, a split other than learn
or held, a split or @profile on a set with no contexts.tsv, a budget over n_expert or under the spares, an unknown
policy word or budget key, ram= with a family that prices no NVMe. Exit 64: usage.

Python 3 + numpy (through rr).
"""

import argparse
import heapq
import importlib.util
import json
import math
import os
import sys
import tempfile
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
FAMILIES_TSV = os.path.join(HERE, "resplay-families.tsv")
REF = os.path.join(HERE, "..", "ref")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rr = _load("rr", os.path.join(REF, "router-residency.py"))
np = rr.np
q38 = _load("q38_step", os.path.join(HERE, "q38_step.py"))
T975 = _load("tdist", os.path.join(REF, "tdist.py")).t975
rt = rr.rt

FAMILIES = ("v41", "q38", "glm", "muse")
REPLAYED = ("v41", "q38", "glm")
DEFAULT_PROMPTS = (64, 512, 4096)
DEFAULT_STEPS = 256
BANDS = (("chat-short", 0, 128), ("chat", 128, 1024), ("p1k", 1024, 2048), ("long", 2048, 1 << 40))
Q38_MACHINE = "box-a6000"  # the q38 price: plan (a)'s A6000 at its server configuration, the draft off
DECIDE_POINTS = 3.0  # §5.4: a stratum's difference is decided at 3 points of h
PROFILE_POINTS = 5.0  # §5.4: the profile row's bar over the first 32 steps


class Refused(Exception):
    pass


# ============================================================================ the family table


class Family:
    """One family's replay and price facts: rr.FAMILY's (v41, glm) or q38_step's (q38), and the rows of
    resplay-families.tsv."""

    def __init__(self, name, consts):
        if name not in FAMILIES:
            raise Refused(f"no family {name!r}: {', '.join(FAMILIES)}")
        self.name = name
        self.c = consts

        def row(k):
            key = f"{name}.{k}"
            if key not in consts:
                raise Refused(f"resplay-families.tsv: no row {key}")
            return consts[key].value

        self.row = row
        self.pcie_gbps = consts["pcie_gbps"].value
        if name == "muse":
            self.slots_token = int(row("slots_token"))
            self.B = row("expert_b")
            return
        if name == "q38":
            self.p = q38_params()
            qc = q38.load_constants()
            self.E, self.K = int(qc["n_expert"].value), int(qc["top_k"].value)
            self.B = qc["expert_b_q4"].value
            # every layer, not rr.FAMILY's 43: the plan holds all 48 since q38hol (q38_step.replay_main's set)
            self.eligible = list(range(int(qc["n_layer"].value)))
            card = int(q38.const_of(qc, Q38_MACHINE, "card_experts").value)
            spread = rr.spread(card, self.eligible)
            self.n_card = [spread[l] for l in self.eligible]
            self.step_ms = rr.family("q38")["step_ms"]
            self.open_m = rr.family("q38")["open_m"]
            self.batch = int(qc["ubatch_max"].value)
            self.slots_token = len(self.eligible) * self.K
        else:
            F = rr.family(name)
            self.E, self.K, self.B = F["n_expert"], F["n_used"], F["expert_bytes"]
            self.eligible = list(F["eligible"])
            self.n_card = rr.n_plan(name)
            self.step_ms, self.open_m = F["step_ms"], F["open_m"]
            self.batch = int(row("call_batch"))
            self.slots_token = int(row("slots_token"))
        self.L = len(self.eligible)
        self.live_delay = int(row("live_delay"))
        self.spares = int(row("spares"))
        self.pinned = int(row("pinned"))
        self.engine_call = row("engine_call") >= 0.5
        self.call_min_p = int(row("call_min_p"))
        self.call_floor = int(row("call_floor"))
        self.group = int(row("call_group"))
        self.end_return = row("end_return") >= 0.5

    def pp(self, P):
        """Prompt tok/s at P: the call's wall for link pacing."""
        if self.name == "v41":
            a, b = self.row("pp512"), self.row("pp4096")
            x = min(1.0, max(0.0, math.log(max(P, 1) / 512) / math.log(8)))
            return a * (b / a) ** x
        if self.name == "q38":
            return q38.prompt(self.p, max(P, 1))[0]
        return self.row("pp")

    def h_cost(self, h):
        """A replay's h (eligible slots) as the cost model's share of the token's slots."""
        return h * self.L * self.K / self.slots_token

    def cost(self, h, flips, nvme=0.0, depth=512):
        """ms a token by reading: {'line': ms} where both readings give one step, else {'A': ms, 'B': ms}. h and nvme
        are shares of the token's slots (h_cost), flips a token."""
        if self.name == "v41":
            slope = self.row("line_slope")
            ms = self.row("line_t0") - slope * h + self.row("line_flip") * flips
            if nvme:
                ms += nvme * self.slots_token * (self.B / (self.c["nvme_gbps"].value * 1e6) - slope / self.slots_token)
            return {"line": ms}
        if self.name == "q38":
            if nvme:
                raise Refused("q38's price has no NVMe term: drop ram= or price it by hand")
            p = dict(self.p, flips_tok=flips)
            return {"model": q38.step(p, depth, hit=h)[0]}
        N = self.slots_token
        cn = self.B / (self.c["nvme_gbps"].value * 1e6) if nvme else 0.0
        if self.name == "glm":
            tA, tB, cf = self.row("t_serial_A"), self.row("t_serial_B"), self.row("c_f")
        else:
            tA = tB = self.row("t_serial")
            cf = self.row("c_f")
        chA, ccA, chB = self.row("c_h_A"), self.row("c_c_A"), self.row("c_h_B")
        host = 1.0 - h - nvme
        return {"A": tA + N * (host * chA + h * ccA + nvme * cn) + flips * cf,
                "B": tB + N * (host * chB + nvme * cn) + flips * cf}


_Q38P = None


def q38_params():
    """q38_step's calibrated A6000 terms with the draft off (a plain decode step), computed once."""
    global _Q38P
    if _Q38P is None:
        _Q38P, _ = q38.calibrate(q38.load_constants(), Q38_MACHINE, ["mtp=0"])
    return _Q38P


def load_families(path=FAMILIES_TSV):
    try:
        return q38.load_constants(path)
    except q38.Refused as e:
        raise Refused(str(e)) from None


# ============================================================================ budgets and seeds


class Tiers:
    """The budget word: the stage card's slots a layer, its pinned seed ids, a tier card's static set, the arena."""

    def __init__(self, fam, word="card=plan"):
        self.word = word
        n_card, self.pool, self.pinned, self.tier, self.ram_gb, self.evict = None, None, fam.pinned, 0, None, "tick"
        for part in word.split(","):
            k, eq, v = part.partition("=")
            if not eq:
                raise Refused(f"--budget {part!r}: KEY=VALUE")
            try:
                if k == "card":
                    if v == "plan":
                        n_card = list(fam.n_card)
                    elif v.startswith("n"):
                        n_card = [int(v[1:])] * fam.L
                    else:
                        raise ValueError
                elif k == "pool":
                    self.pool = int(v)
                elif k == "pinned":
                    self.pinned = int(v)
                elif k == "tier":
                    self.tier = int(v)
                elif k == "ram":
                    g, _, ev = v.partition(":")
                    self.ram_gb = float(g)
                    self.evict = ev or "tick"
                    if self.evict not in ("tick", "heat"):
                        raise Refused(f"--budget ram={v}: eviction tick or heat")
                else:
                    raise Refused(f"--budget: no key {k!r} (card, pool, pinned, tier, ram)")
            except ValueError:
                raise Refused(f"--budget {part!r}: not a number") from None
        if n_card is None:
            raise Refused(f"--budget {word!r}: card=plan or card=nN")
        if any(n < fam.spares + 1 or n + self.tier > fam.E for n in n_card):
            raise Refused(f"--budget {word!r}: card slots {min(n_card)}..{max(n_card)} plus tier {self.tier} must lie "
                          f"in {fam.spares + 1}..{fam.E} (n_expert {fam.E}, spares {fam.spares})")
        self.n_card = n_card
        self.n_live = [n - fam.spares for n in n_card]
        if self.pool is not None:
            if not 0 <= self.pool <= min(self.n_live):
                raise Refused(f"--budget pool={self.pool}: 0..{min(self.n_live)} (the live slots)")
            self.pin_n = [n - self.pool for n in self.n_live]
        else:
            if not 0 <= self.pinned <= min(self.n_live):
                raise Refused(f"--budget pinned={self.pinned}: 0..{min(self.n_live)}")
            self.pin_n = [self.pinned] * fam.L
        self.ram_experts = None if self.ram_gb is None else int(self.ram_gb * 1e9 // fam.B)

    def masks(self, lists, E, moving):
        """(resident, pinned or None, tier or None) [L, E] bool from the seed lists."""
        n = self.n_live if moving else self.n_card
        res = rr.seed_resident(lists, n, E)
        pin = tier = None
        if moving and any(self.pin_n):
            pin = np.zeros_like(res)
            for i, ids in enumerate(lists):
                pin[i, np.asarray(ids[: self.pin_n[i]], dtype=np.int64)] = True
        if self.tier:
            tier = np.zeros_like(res)
            for i, ids in enumerate(lists):
                tier[i, np.asarray(ids[self.n_card[i]: self.n_card[i] + self.tier], dtype=np.int64)] = True
        return res, pin, tier


def prefix_lists(L, E):
    return [np.arange(E) for _ in range(L)]


def profile_lists(chain, L, E, what):
    """The warm profile's seed: per layer the learn requests' decode (and with `both` prompt) counts, ranked."""
    counts = np.zeros((L, E), dtype=np.int64)
    learned = 0
    for q in chain.reqs:
        if q.split != "learn":
            continue
        learned += 1
        t0 = q.t0 if what == "both" else q.t0 + q.pc
        rows = chain.X[t0:q.end]
        for i in range(L):
            counts[i] += np.bincount(rows[:, i].ravel(), minlength=E)
    if not learned:
        raise Refused(f"{chain.name}: @profile learns on learn requests and the chain has none")
    return [rr.rank(c) for c in counts]


# ============================================================================ chains


class Req:
    __slots__ = ("idx", "t0", "pc", "end", "P", "split", "genre")

    def __init__(self, idx, t0, pc, end, P, split=None, genre=""):
        self.idx, self.t0, self.pc, self.end, self.P, self.split, self.genre = idx, t0, pc, end, P, split, genre


class Chain:
    """One serve order: X [T, L, K] over the family's eligible layers and its requests in order."""

    def __init__(self, name, X, reqs, shape):
        self.name, self.X, self.reqs, self.shape = name, X, reqs, shape


def read_set(fam, path, prompts, steps):
    """The chains a set gives: one from a route trace (contexts.tsv or call rows), one per P from a corpus set."""
    try:
        s = rr.Set(path)
    except (rr.rc.SetError, rr.ToolError) as e:
        raise Refused(str(e)) from None
    if (s.E, s.K) != (fam.E, fam.K):
        raise Refused(f"{path}: {s.E} experts x top-{s.K}, family {fam.name} is {fam.E} x top-{fam.K}")
    missing = [l for l in fam.eligible if l not in s.layers]
    if missing:
        raise Refused(f"{path}: family {fam.name}'s eligible layers {missing} are not in the set")
    X = s.stack(fam.eligible)
    try:
        rows, _ = rt.read_contexts(path)
        calls = None if rows is not None or "engine" not in s.meta["header"] else rt.read_calls(path)[0]
    except (rt.TraceError, OSError) as e:
        raise Refused(str(e)) from None
    if rows is not None:
        reqs = [Req(r["request"], r["first"], r["prompt_call"], r["end"], r["prompt"], r["split"], r["genre"])
                for r in rows]
        return [Chain(s.name, X, reqs, "contexts.tsv")]
    if calls is not None:
        reqs = [Req(i, c["first"], c["prompt_call"], c["end"], c["prompt_call"], None, s.name)
                for i, c in enumerate(calls)]
        return [Chain(s.name, X, reqs, "call rows")]
    out = []
    for P in prompts:
        R = P + steps
        if s.T < R:
            raise Refused(f"{path}: {s.T} positions, one request of P {P} + {steps} is {R}")
        reqs = [Req(j, j * R, P, (j + 1) * R, P, None, s.name) for j in range(s.T // R)]
        out.append(Chain(f"{s.name}@P{P}", X, reqs, f"cut {P}+{steps}"))
    return out


# ============================================================================ the call's pick and the opening


def call_pick(live, off, counts, cap, floor):
    """SwapRule::call_pick at one layer (crates/runtime/src/swaprule.rs): `live` the pool (live residents, neither
    pinned nor in a flip in flight), `off` the ids off the card that may enter; the pool and the off ids counted at
    least `floor` rank together by (count descending, id ascending), the first min(cap, pool) are wanted, and the
    wanted off ids enter hottest first, each in place of the coldest pool resident left out (count ascending, id
    ascending), while the entering count is higher. Returns [(in, out)]."""
    ids = np.nonzero(live | (off & (counts >= floor)))[0]
    ranked = ids[np.lexsort((ids, -counts[ids]))]
    pool = int(live.sum())
    w = min(cap, pool, len(ranked))
    wanted, left = ranked[:w], ranked[w:]
    victims = left[live[left]]
    victims = victims[np.lexsort((victims, counts[victims]))]
    admits = wanted[~live[wanted]]
    out = []
    for a, v in zip(admits.tolist(), victims.tolist()):
        if counts[a] <= counts[v]:
            break
        out.append((a, v))
    return out


def ranked_lists(resident, lists):
    """Per layer the current card set first, then the rest, each in seed-rank order: rr.opening reads its first
    n ids as the card set and the order as the seed rank its ties go by (SwapRule::open's rank)."""
    out = []
    for i, ids in enumerate(lists):
        ids = np.asarray(ids)
        on = resident[i, ids]
        out.append(np.concatenate([ids[on], ids[~on]]))
    return out


def open_pairs(Xp, resident, lists, E, rule, m, last):
    """SwapRule::open over prompt Xp from the current set (no flip in flight, no pinned or away id: rr.opening has
    none): [(layer, in, out)] largest gain first, at most m, and the prompt's decayed counts."""
    n_l = resident.sum(axis=1).tolist()
    _, dec, pairs = rr.opening(Xp, ranked_lists(resident, lists), n_l, E, m, rule, last)
    return [(i, int(a), int(b)) for _, i, a, b in pairs], dec


def settle(st):
    """Land every flip in flight now (the relayout at a call's end runs on a quiet set)."""
    for _, _, i, a, v in st.heap:
        st.resident[i, a], st.resident[i, v] = True, False
    st.heap = []
    st.pend_in[:] = False
    st.pend_out[:] = False


def prompt_decayed(Xp, L, E, rule):
    """The prompt's counts decayed at the rule's cadence as rows kept (rr.opening's form)."""
    return rr.opening(Xp, prefix_lists(L, E), [0] * L, E, 0, rule)[1]


# ============================================================================ policies


class Policy:
    def __init__(self, word, fam):
        self.word = word
        self.kind = None
        self.rule = None
        self.carry = "serve"
        self.call = None  # (min_p, floor) when the call picks
        self.end = None  # keep | return | open
        self.fold = False
        self.last = None
        self.m = None
        self.window = None
        self.every = None
        self.profile = None
        base, at, prof = word.partition("@")
        if at:
            what = prof.split(":", 1)[1] if ":" in prof else "decode"
            if prof.split(":", 1)[0] != "profile" or what not in ("decode", "both"):
                raise Refused(f"policy {word!r}: @profile[:decode|both]")
            self.profile = what
        parts = base.split(":")
        head, opts = parts[0], parts[1:]
        if head == "engine":
            if opts and opts != [fam.name]:
                raise Refused(f"policy {word!r}: the replay's family is {fam.name}")
            if fam.engine_call:
                head, opts = "prompt+rule", ["end=return" if fam.end_return else "end=keep", f"min_p={fam.call_min_p}"]
            else:
                head, opts = "rule", ["mid", "carry=serve"]
        rule_name = "mid"
        kv = {}
        for o in opts:
            if o in rr.RULES:
                rule_name = o
            elif "=" in o:
                k, v = o.split("=", 1)
                kv[k] = v
            elif head == "rolling" and o.isdigit():
                kv["W"] = o
            else:
                raise Refused(f"policy {word!r}: no option {o!r}")
        allowed = {"static": set(), "belady": set(), "oracle-static": set(), "prompt-freeze": {"last", "m"},
                   "prompt-call": {"min_p", "floor"}, "rule": {"carry"}, "rolling": {"W", "every"},
                   "prompt+rule": {"seed", "end", "min_p", "floor", "m"}}
        if head not in allowed:
            raise Refused(f"policy {word!r}: no word {head!r} ({', '.join(allowed)}, engine)")
        extra = set(kv) - allowed[head]
        if extra:
            raise Refused(f"policy {word!r}: {head} takes no {', '.join(sorted(extra))}")
        try:
            min_p = int(kv.get("min_p", 1))
            floor = int(kv.get("floor", fam.call_floor))
            self.last = int(kv["last"]) if "last" in kv else None
            self.m = None if kv.get("m", "all") == "all" else int(kv["m"])
        except ValueError:
            raise Refused(f"policy {word!r}: a count that is not a number") from None
        self.kind = head
        if head in ("rule", "prompt+rule"):
            self.rule = rr.Rule(rule_name)
        if head == "rule":
            self.carry = kv.get("carry", "serve")
            if self.carry not in ("serve", "request"):
                raise Refused(f"policy {word!r}: carry serve or request")
        if head == "prompt-call":
            self.call, self.end = (min_p, floor), "keep"
        if head == "prompt+rule":
            self.end = kv.get("end", "keep")
            seed = kv.get("seed", "none")
            if self.end not in ("keep", "return", "open") or seed not in ("none", "decayed"):
                raise Refused(f"policy {word!r}: end keep|return|open, seed none|decayed")
            self.fold = seed == "decayed"
            self.call = None if self.end == "open" else (min_p, floor)
        if head == "prompt-freeze":
            self.end = "open"
        if head == "rolling":
            try:
                self.window = int(kv["W"])
                self.every = int(kv.get("every", rr.RULES["mid"]["every"]))
            except (KeyError, ValueError):
                raise Refused(f"policy {word!r}: rolling:W with W a count") from None
            if self.window < 1 or self.every < 1:
                raise Refused(f"policy {word!r}: W and every >= 1")
            m = rr.RULES["mid"]
            self.rule = rr.Rule("mid", every=self.every, margin=1.0, min_count=1.0, decay=1.0, cap=m["cap"])
        self.moving = head != "static"


def parse_policies(text, fam):
    words = [w for w in (text or "").split(",") if w]
    if "engine" not in words and f"engine:{fam.name}" not in words:
        words = ["engine"] + words
    return [Policy(w, fam) for w in words]


# ============================================================================ the replay of one chain


class Run:
    """A policy's state along one chain: rr's replay state, the decode history, the rolling window, the arena."""

    def __init__(self, pol, fam, tiers, lists):
        self.pol, self.fam, self.tiers, self.lists = pol, fam, tiers, lists
        self.res0, self.pin, self.tier = tiers.masks(lists, fam.E, pol.moving)
        self.n_l = tiers.n_live if pol.moving else tiers.n_card
        self.reset()
        self.win = np.zeros((fam.L, fam.E), dtype=np.int64)
        self.win_rows = deque()
        self.arena = Arena(tiers.ram_experts, tiers.evict, fam.c["heat_half_life"].value) \
            if tiers.ram_experts is not None else None

    def reset(self):
        L, E = self.fam.L, self.fam.E
        self.st = rr.ReplayState(self.res0.copy(), np.zeros((L, E)), np.zeros((L, E), dtype=bool),
                                 np.zeros((L, E), dtype=bool), [], 0, 0, 0)
        self.history = False

    def replay(self, X, passes=None, record=None, rule=None):
        r = rr.replay(X, None, self.n_l, self.fam.E, rule, sem="flip", spares=self.fam.spares,
                      d=self.fam.live_delay, state0=self.st, passes=passes, record=record, pinned=self.pin,
                      away=self.tier)
        self.st = r.state
        return r


def request(run, X, q, steps):
    """One request along `run`: the prompt pass, the prompt's moves at their pacing, the decode. Returns the request's
    measures (per_row over its decode rows, flips, moves, late landings, link ms, NVMe share)."""
    pol, fam = run.pol, run.fam
    L, E, K = fam.L, fam.E, fam.K
    if pol.kind == "rule" and pol.carry == "request":
        run.reset()
    Xp, Xd = X[q.t0:q.t0 + q.pc], X[q.t0 + q.pc:q.end]
    if q.pc:
        prec = np.zeros(Xp.shape, dtype=bool) if run.arena is not None else None
        run.replay(Xp, passes=[(0, q.pc, 0)], record=prec, rule=pol.rule)
        if run.arena is not None:
            run.arena.walk(Xp, prec | tier_hits(run.tier, Xp))
    moves, gb, link_ms = prompt_moves(run, Xp, q)
    late = land_moves(run, moves, q.pc / fam.pp(q.P) * 1e3)
    if pol.fold and q.pc:
        every = pol.rule.every
        run.st.counts = run.st.counts * pol.rule.decay ** (q.pc // every) + prompt_decayed(Xp, L, E, pol.rule)
    rec = np.zeros(Xd.shape, dtype=bool)
    flips = 0
    if pol.kind == "rolling":
        flips = rolling_decode(run, Xd, rec)
    else:
        flips = run.replay(Xd, record=rec, rule=pol.rule).swaps
    if pol.rule is not None and len(Xd):
        run.history = True
    hit = rec | tier_hits(run.tier, Xd)
    nvme = 0.0
    if run.arena is not None:
        nv = run.arena.walk(Xd, hit)
        nvme = float(nv[:steps].mean()) if len(nv) else 0.0
    return dict(per_row=hit.mean(axis=(1, 2)), flips=flips, rows=len(Xd), moves=len(moves), gb=gb, late=late,
                link_ms=link_ms, nvme=nvme)


def tier_hits(tier, X):
    if tier is None:
        return np.zeros(X.shape, dtype=bool)
    return tier[np.arange(X.shape[1])[:, None], X]


def prompt_moves(run, Xp, q):
    """The prompt's moves the policy keeps for the decode [(landed ms after the call's start, layer, in, out)], the GB
    the link carries (a returned call carries its picks and their way back) and its busy ms. Each copy is issued at
    its layer's place in its group's walk: a pick in the layer's shadow, the relayout after the layer's last unit."""
    pol, fam = run.pol, run.fam
    if pol.end is None or not q.pc:
        return [], 0.0, 0.0
    L, E = fam.L, fam.E
    st = run.st
    pp = fam.pp(q.P)
    groups = call_groups(q.pc, fam.batch, fam.group)
    copy_ms = fam.B / (fam.pcie_gbps * 1e6)
    if pol.call is not None and q.pc >= pol.call[0]:
        busy = st.pend_out | st.pend_in
        if run.pin is not None:
            busy = busy | run.pin
        off_ok = ~busy if run.tier is None else ~busy & ~run.tier
        res = st.resident.copy()
        start = res.copy()
        picks = []
        for g0, b1, g1, nb in groups:
            g_ms, t_ms = (g1 - g0) / pp * 1e3, g0 / pp * 1e3
            rows = Xp[g0:b1]
            for i in range(L):
                c = np.bincount(rows[:, i].ravel(), minlength=E)
                for a, v in call_pick(res[i] & ~busy[i], ~res[i] & off_ok[i], c, 1 << 30, pol.call[1] * nb):
                    res[i, a], res[i, v] = True, False
                    picks.append((t_ms + i / L * g_ms, i, a, v))
        ends = copy_ends([t for t, _, _, _ in picks], copy_ms)
        if pol.end == "return" and run.history:
            return [], 2 * len(picks) * fam.B / 1e9, 2 * len(picks) * copy_ms
        # what the call admitted and kept, each landed at its last pick's copy, against what left the start set
        landed = {(i, a): e for (_, i, a, _), e in zip(picks, ends)}
        net = []
        for i in range(L):
            ins = sorted((landed[(i, a)], a) for a in np.nonzero(res[i] & ~start[i])[0].tolist())
            outs = np.nonzero(start[i] & ~res[i])[0].tolist()
            net += [(e, i, a, v) for (e, a), v in zip(ins, outs)]
        return net, len(picks) * fam.B / 1e9, len(picks) * copy_ms
    if pol.end == "open":
        settle(st)
        g0, _, g1, _ = groups[-1]
        g_ms, t_ms = (g1 - g0) / pp * 1e3, g0 / pp * 1e3
        pairs, _ = open_pairs(Xp, st.resident, run.lists, E, pol.rule or rr.Rule("mid"), pol.m, pol.last)
        ends = copy_ends([t_ms + (i + 1) / L * g_ms for i, _, _ in pairs], copy_ms)
        return [(e, i, a, v) for e, (i, a, v) in zip(ends, pairs)], len(pairs) * fam.B / 1e9, len(pairs) * copy_ms
    return [], 0.0, 0.0


def copy_ends(issues, copy_ms):
    """Each copy's end (ms) on one serial link, copies taken in issue order."""
    ends = [0.0] * len(issues)
    free = 0.0
    for j in sorted(range(len(issues)), key=lambda j: issues[j]):
        free = max(free, issues[j]) + copy_ms
        ends[j] = free
    return ends


def call_groups(pc, batch, group):
    """The call's groups [(first row, its first batch's end, end, batches)]: batches of `batch` rows, `group` a group."""
    out = []
    span = batch * group
    for g0 in range(0, pc, span):
        g1 = min(pc, g0 + span)
        out.append((g0, min(g1, g0 + batch), g1, -(-(g1 - g0) // batch)))
    return out


def land_moves(run, moves, call_ms):
    """Apply the kept moves: one landed by the call's end (call_ms) is live at the first decode step, a later one from
    the first step that starts after it (pending in rr's state, its victim live until then). Returns the late ones."""
    fam, st = run.fam, run.st
    late = 0
    for end, i, a, v in moves:
        if end <= call_ms:
            st.resident[i, a], st.resident[i, v] = True, False
            continue
        late += 1
        k = math.ceil((end - call_ms) / fam.step_ms - 1e-9)
        st.pend_in[i, a] = True
        st.pend_out[i, v] = True
        st.heap.append((st.passes + k, st.seq, i, a, v))
        st.seq += 1
    return late


def rolling_decode(run, Xd, rec):
    """rolling:W: each decode pass replayed with the rule's counts set to the window's last W - 1 rows, so the plan
    after the pass pairs over the last W (rr's pairing, margin 1, no decay)."""
    pol, fam = run.pol, run.fam
    ar = np.arange(fam.L)[:, None]
    flips = 0
    for k in range(len(Xd)):
        if len(run.win_rows) == pol.window:
            run.win[ar, run.win_rows.popleft()] -= 1
        run.st.counts = run.win.astype(float)
        flips += run.replay(Xd[k:k + 1], record=rec[k:k + 1], rule=pol.rule).swaps
        run.win[ar, Xd[k]] += 1
        run.win_rows.append(Xd[k])
    return flips


class Arena:
    """The host's arena of off-card experts under an NVMe plan: `cap` experts, evicted by tick (least recent) or by
    heat (a count halved every `half` rows). walk() returns per row the share of slots read from NVMe."""

    def __init__(self, cap, evict, half):
        self.cap, self.evict, self.half = cap, evict, half
        self.tick = 0
        self.last = {}  # (layer, id) -> (heat, tick)
        self.heap = []

    def _key(self, e):
        h, t = self.last[e]
        return math.log2(h) + t / self.half if self.evict == "heat" else float(t)

    def walk(self, X, hit):
        out = np.zeros(X.shape[0])
        T, L, K = X.shape
        for r in range(T):
            self.tick += 1
            miss = 0
            for i in range(L):
                for k in range(K):
                    if hit[r, i, k]:
                        continue
                    e = (i, int(X[r, i, k]))
                    if e in self.last:
                        h, t = self.last[e]
                        self.last[e] = (h * 2.0 ** (-(self.tick - t) / self.half) + 1.0, self.tick)
                    else:
                        miss += 1
                        self.last[e] = (1.0, self.tick)
                        while len(self.last) > self.cap:
                            k0, e0, t0 = heapq.heappop(self.heap)
                            if e0 in self.last and self.last[e0][1] == t0:
                                del self.last[e0]
                    heapq.heappush(self.heap, (self._key(e), e, self.tick))
            out[r] = miss / (L * K)
        return out


def ceilings(fam, tiers, Xw):
    """(belady h, belady h16, oracle per row) over a scored window at the card's slots."""
    L, E = fam.L, fam.E
    first = []
    for i in range(L):
        seen = list(dict.fromkeys(Xw[:, i].ravel().tolist()))
        used = set(seen)
        first.append(np.asarray(seen + [e for e in range(E) if e not in used]))
    n = tiers.n_card
    bh, _ = rr.belady(Xw, first, n)
    b16, _ = rr.belady(Xw[:16], first, n)
    lists = [rr.rank(np.bincount(Xw[:, i].ravel(), minlength=E)) for i in range(L)]
    orc = rr.replay(Xw, rr.seed_resident(lists, n, E), n, E).per_row
    return bh.sum() / Xw.size, b16.sum() / Xw[:16].size, orc


def replay_chain(chain, fam, tiers, pols, steps, split):
    """Every policy over one chain: per policy the scored requests' measures."""
    scored = [q for q in chain.reqs if (split is None or q.split == split) and q.end - (q.t0 + q.pc) > 0]
    if not scored:
        raise Refused(f"{chain.name}: no request of split {split} with a decode row")
    keys = {id(q) for q in scored}
    out = {}
    for pol in pols:
        if pol.end == "open" and (tiers.tier or any(tiers.pin_n)):
            raise Refused(f"policy {pol.word!r} with pinned or tier ids: rr.opening (SwapRule::open's twin) holds none")
        rows = []
        if pol.kind in ("belady", "oracle-static"):
            for q in scored:
                d0 = q.t0 + q.pc
                Xw = chain.X[d0:min(q.end, d0 + steps)]
                bh, b16, orc = ceilings(fam, tiers, Xw)
                pr = orc if pol.kind == "oracle-static" else None
                rows.append(measures(q, chain.name, pr, steps, h=bh if pr is None else None,
                                     h16=b16 if pr is None else None))
            out[pol.word] = rows
            continue
        if pol.profile is not None:
            if split != "held":
                raise Refused(f"policy {pol.word!r}: a profile scores held requests only (--split held)")
            lists = profile_lists(chain, fam.L, fam.E, pol.profile)
        else:
            lists = prefix_lists(fam.L, fam.E)
        run = Run(pol, fam, tiers, lists)
        for q in chain.reqs:
            m = request(run, chain.X, q, steps)
            if id(q) in keys:
                rows.append(measures(q, chain.name, m["per_row"], steps, m=m))
        out[pol.word] = rows
    return out


def measures(q, chain, per_row, steps, m=None, h=None, h16=None):
    r = dict(chain=chain, req=q.idx, P=q.P, genre=q.genre, split=q.split)
    if per_row is not None:
        w = per_row[:steps]
        r.update(h=float(w.mean()), h16=float(w[:16].mean()), h32=float(w[:32].mean()),
                 h_tail=float(w[64:].mean()) if len(w) > 64 else None)
    else:
        r.update(h=float(h), h16=float(h16), h32=None, h_tail=None)
    if m is not None:
        r.update(flips_tok=m["flips"] / max(1, m["rows"]), moves=m["moves"], gb=m["gb"], late=m["late"],
                 link_s=m["link_ms"] / 1e3, nvme=m["nvme"])
    return r


# ============================================================================ tables


def band_of(P):
    for name, lo, hi in BANDS:
        if lo <= P < hi:
            return name
    raise Refused(f"a prompt of {P} ids")


def interval(d):
    """Mean and 95 % half-width of paired differences (t over the requests)."""
    n = len(d)
    if n < 2:
        return (float(np.mean(d)) if n else float("nan")), float("inf")
    return float(np.mean(d)), T975(n - 1) * float(np.std(d, ddof=1)) / math.sqrt(n)


def strata(results):
    """{stratum: {policy: [rows]}}: the band of each request's P, and the band by genre."""
    out = {}
    for pol, rows in results.items():
        for r in rows:
            b = band_of(r["P"])
            for key in (b, f"{b}:{r['genre']}"):
                out.setdefault(key, {}).setdefault(pol, []).append(r)
    return out


def summary(fam, rows, base, depth):
    """One stratum's row for a policy: means, its price, and the paired difference against `base` (engine)."""
    def mean(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return float(np.mean(v)) if v else None
    s = dict(n=len(rows), h=mean("h"), h16=mean("h16"), h_tail=mean("h_tail"), flips_tok=mean("flips_tok"),
             moves=mean("moves"), gb=mean("gb"), late=mean("late"), link_s=mean("link_s"), nvme=mean("nvme"))
    if s["flips_tok"] is not None:
        try:
            s["ms"] = fam.cost(fam.h_cost(s["h"]), s["flips_tok"], fam.h_cost(s["nvme"] or 0.0), depth)
        except Refused as e:
            s["ms"] = {"refused": str(e)}
    if base is not None:
        pair = {(r["chain"], r["req"]): r["h"] for r in base}
        d = [100.0 * (r["h"] - pair[(r["chain"], r["req"])]) for r in rows]
        s["d"], s["ci"] = interval(d)
    return s


def fmt(v, f="{:.1f}", scale=1.0):
    return "—" if v is None else f.format(v * scale)


def table(fam, res, steps, out=print):
    st = strata(res)
    out("| stratum | policy | n | h % | h16 % | h_tail % | flips/tok | MB/tok | moves/req | GB/req | link s/req | "
        "late/req | NVMe % | ms | tok/s | Δh vs engine (pts, 95 %) |")
    out("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    summ = {}
    for key in sorted(st, key=lambda k: ([b[0] for b in BANDS].index(k.split(":")[0]), k)):
        byp = st[key]
        base = byp.get("engine")
        depth = int(np.mean([r["P"] for r in base or next(iter(byp.values()))])) + steps // 2
        summ[key] = {}
        for pol, rows in byp.items():
            s = summary(fam, rows, base, depth)
            summ[key][pol] = s
            ms = s.get("ms") or {}
            mtxt = "/".join(f"{k} {v:.2f}" for k, v in ms.items() if k != "refused") if "refused" not in ms else "—"
            ttxt = "/".join(f"{1e3 / v:.1f}" for k, v in ms.items() if k != "refused") if "refused" not in ms else "—"
            dtxt = "—" if pol == "engine" or "d" not in s else f"{s['d']:+.1f} ± {s['ci']:.1f}"
            mb = None if s["flips_tok"] is None else s["flips_tok"] * fam.B / 1e6
            out(f"| {key} | {pol} | {s['n']} | {fmt(s['h'], scale=100)} | {fmt(s['h16'], scale=100)} | "
                f"{fmt(s['h_tail'], scale=100)} | {fmt(s['flips_tok'], '{:.2f}')} | {fmt(mb, '{:.0f}')} | "
                f"{fmt(s['moves'], '{:.0f}')} | {fmt(s['gb'], '{:.2f}')} | {fmt(s['link_s'], '{:.3f}')} | "
                f"{fmt(s['late'], '{:.1f}')} | {fmt(s['nvme'], scale=100)} | {mtxt or '—'} | {ttxt or '—'} | {dtxt} |")
    return summ


# ============================================================================ decide (§5.4)

DPR = ("prompt-freeze", "rule:mid:carry=serve")
DC = "prompt+rule:seed=decayed:end=keep"


def deltas(rows_by_pol, dc=DC):
    """Δ_pr and Δ_c (points: mean, half-width) over a stratum's paired requests, or None when a word is missing; dc
    is the companion word Δ_c reads (§5.1's is DC)."""
    fz, rl, pc = rows_by_pol.get(DPR[0]), rows_by_pol.get(DPR[1]), rows_by_pol.get(dc)
    if not (fz and rl and pc):
        return None

    def paired(a, b):
        pb = {(r["chain"], r["req"]): r["h"] for r in b}
        return interval([100.0 * (r["h"] - pb[(r["chain"], r["req"])]) for r in a])
    best = fz if np.mean([r["h"] for r in fz]) >= np.mean([r["h"] for r in rl]) else rl
    return dict(pr=paired(fz, rl), c=paired(pc, best), n=len(fz))


def profile_delta(rows_by_pol):
    """h32(best@profile) - h32(best) over chat-short's paired requests, the best by h among words with a twin."""
    twins = [p for p in rows_by_pol if "@profile" in p and p.split("@")[0] in rows_by_pol]
    if not twins:
        return None
    best = max(twins, key=lambda p: np.mean([r["h"] for r in rows_by_pol[p.split("@")[0]]]))
    a, b = rows_by_pol[best], rows_by_pol[best.split("@")[0]]
    pb = {(r["chain"], r["req"]): r["h32"] for r in b}
    return best, interval([100.0 * (r["h32"] - pb[(r["chain"], r["req"])]) for r in a])


def side(m_hw, bar):
    """'ge' when the interval lies at or above bar, 'lt' below it, 'span' across it."""
    m, hw = m_hw
    if m - hw >= bar:
        return "ge"
    if m + hw < bar:
        return "lt"
    return "span"


def decide_family(bands):
    """bands: [(band, lo P, deltas)] in P order (bands with no data left out). The §5.4 branch and its reason."""
    if not bands:
        return "no stratum carries prompt-freeze, rule:mid:carry=serve and prompt+rule:seed=decayed:end=keep"
    c = [(b, lo, side(d["c"], DECIDE_POINTS)) for b, lo, d in bands]
    pr = [(b, lo, side(d["pr"], -DECIDE_POINTS + 1e-12)) for b, lo, d in bands]
    if any(s == "span" for _, _, s in c):
        spans = [b for b, _, s in c if s == "span"]
        return f"undecided: Δ_c's interval spans 3 points in {', '.join(spans)} -> more requests there (M2 + 12 rows)"
    if all(s == "ge" for _, _, s in c):
        return "(i) Δ_c >= 3 in every stratum -> prompt heat then the rule, everywhere (CallEnd::Keep, seed decayed)"
    if all(s == "lt" for _, _, s in c):
        if all(s == "lt" for _, _, s in pr):
            return "(ii) Δ_pr <= -3 and Δ_c < 3 everywhere -> rolling heat suffices (CallEnd::Return with a history)"
        spans = [b for b, _, s in pr if s != "lt"]
        return (f"Δ_c < 3 everywhere, Δ_pr not <= -3 in {', '.join(spans)}: neither (i) nor (ii); the call picks stay "
                "a prefill lever, the decode keeps the rule")
    first = next(lo for _, lo, s in c if s == "ge")
    if all(s == "ge" for _, lo, s in c if lo >= first) and all(s == "lt" for _, lo, s in c if lo < first):
        return f"(iii) Δ_c >= 3 from P* = {first} up, < 3 below -> the policy of (i) at P >= {first}, (ii) below"
    return "mixed out of P order: Δ_c >= 3 in " + ", ".join(b for b, _, s in c if s == "ge") + " only"


def decide(doc, out=print, dc=DC):
    if not any(dc in st[b] for st in doc["strata"].values() for b in st):
        raise Refused(f"--companion {dc!r}: no stratum carries it")
    for fam_name, st in doc["strata"].items():
        bands = []
        for name, lo, _ in BANDS:
            if name in st:
                d = deltas(st[name], dc)
                if d is not None:
                    bands.append((name, lo, d))
        text = decide_family(bands)
        nums = "; ".join(f"{b}: Δ_pr {d['pr'][0]:+.1f} ± {d['pr'][1]:.1f}, Δ_c {d['c'][0]:+.1f} ± {d['c'][1]:.1f} "
                         f"(n {d['n']})" for b, _, d in bands)
        head = fam_name if dc == DC else f"{fam_name} (Δ_c by {dc})"
        out(f"{head}: {text}" + (f"  [{nums}]" if nums else ""))
        if "chat-short" in st:
            pd = profile_delta(st["chat-short"])
            if pd is not None:
                best, (m, hw) = pd
                verdict = {"ge": "the warm profile pays on short chat -> R6 wires WARM_PROFILE into every seat",
                           "lt": "the profile does not pay on short chat", "span": "undecided"}[side((m, hw),
                                                                                                    PROFILE_POINTS)]
                out(f"{fam_name} profile: {best} h32 {m:+.1f} ± {hw:.1f} points -> {verdict}")


# ============================================================================ commands


def cmd_replay(a, out=print):
    consts = load_families()
    fam = Family(a.family, consts)
    if fam.name not in REPLAYED:
        raise Refused(f"family {fam.name} has no router set: predict only")
    if a.split is not None and a.split not in rt.SPLITS:
        raise Refused(f"--split {a.split!r}: learn or held")
    tiers = Tiers(fam, a.budget)
    if tiers.ram_experts is not None and fam.name == "q38":
        raise Refused("ram=: q38's price has no NVMe term")
    pols = parse_policies(a.policies, fam)
    try:
        prompts = [int(p) for p in a.prompt.split(",")] if a.prompt else list(DEFAULT_PROMPTS)
    except ValueError:
        raise Refused(f"--prompt {a.prompt!r}: P,P,...") from None
    chains = []
    for name in a.sets:
        try:
            path = rr.set_dir(a.data, name)
        except rr.ToolError as e:
            raise Refused(str(e)) from None
        for ch in read_set(fam, path, prompts, a.steps):
            if a.split is not None and all(q.split is None for q in ch.reqs):
                raise Refused(f"{ch.name}: no contexts.tsv, so no learn/held split for --split {a.split}")
            chains.append(ch)
    res = {}
    for ch in chains:
        for pol, rows in replay_chain(ch, fam, tiers, pols, a.steps, a.split).items():
            res.setdefault(pol, []).extend(rows)
    out(f"# resplay {fam.name}: data={a.data} sets={','.join(a.sets)} chains="
        + ", ".join(f"{c.name} ({len(c.reqs)} requests, {c.shape})" for c in chains))
    out(f"# budget {tiers.word}: card {min(tiers.n_card)}..{max(tiers.n_card)} a layer on {fam.L} layers, live "
        f"{min(tiers.n_live)}..{max(tiers.n_live)} under residency, pinned {max(tiers.pin_n)}, tier {tiers.tier}, "
        f"ram {tiers.ram_gb} GB {tiers.evict}; spares {fam.spares}, live delay {fam.live_delay}, steps {a.steps}, "
        f"split {a.split or 'none (every request scored)'}")
    out(f"# price: {price_conditions(fam)}")
    summ = table(fam, res, a.steps, out)
    if a.json:
        doc = dict(family=fam.name, budget=tiers.word, steps=a.steps, split=a.split, sets=a.sets,
                   strata={fam.name: strata(res)}, summary={fam.name: summ})
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1, default=float)
        out(f"# json {a.json}")
    return res


def price_conditions(fam):
    if fam.name == "v41":
        return ("the measured line t = t0 - slope h + flip F (A6000, plan (a), P 512, depth <= 608), h the card "
                f"share of {fam.slots_token} slots")
    if fam.name == "q38":
        return f"q38_step.step on {Q38_MACHINE}, the draft off, depth = the stratum's mean P + steps/2"
    return "physical constants, readings A (card experts add) and B (they hide), assumed t_serial"


def cmd_predict(a, out=print):
    fam = Family(a.family, load_families())
    if not 0.0 <= a.hit <= 1.0 or not 0.0 <= a.nvme <= 1.0 - a.hit:
        raise Refused(f"--hit {a.hit} and --nvme {a.nvme}: shares in [0, 1] summing to at most 1")
    ms = fam.cost(a.hit, a.flips, a.nvme, a.depth)
    for k, v in ms.items():
        out(f"{fam.name} {k}: {v:.2f} ms, {1e3 / v:.2f} tok/s @ h {a.hit:.4f} of {fam.slots_token} slots, flips "
            f"{a.flips:g} a token, NVMe {a.nvme:g}, depth {a.depth}")
    return ms


def cmd_decide(a, out=print):
    try:
        with open(a.json, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        raise Refused(f"{a.json}: {e}") from None
    decide(doc, out, a.companion)


# ============================================================================ self-test


def _fam(L=1, E=4, K=1, n=1, d=1, spares=1, step_ms=10.0, pp=None, B=16_800_000, gbps=26.28, floor=32, batch=512,
         group=1, open_m=None):
    """A hand family for the self-test: L layers of E experts, top-K, n card slots a layer."""
    f = Family.__new__(Family)
    f.name, f.c = "hand", {"heat_half_life": type("C", (), {"value": 32.0})()}
    f.E, f.K, f.B, f.L, f.eligible = E, K, B, L, list(range(L))
    f.n_card, f.live_delay, f.spares, f.pinned, f.step_ms, f.open_m = [n] * L, d, spares, 0, step_ms, open_m
    f.pcie_gbps, f.call_floor, f.batch, f.group, f.slots_token = gbps, floor, batch, group, L * K
    f.engine_call, f.call_min_p, f.end_return = False, 1, False
    f.pp = (lambda P: 1e9) if pp is None else pp
    return f


def _chain(rows, reqs):
    """rows: per position, per layer a tuple of ids; reqs: (t0, pc, end, split)."""
    X = np.asarray(rows, dtype=np.int64)
    return Chain("hand", X, [Req(j, t0, pc, end, pc, sp, "g") for j, (t0, pc, end, sp) in enumerate(reqs)], "hand")


def _run(fam, word, chain, budget=None, lists=None, steps=256):
    pol = Policy(word, fam)
    tiers = Tiers(fam, budget or f"card=n{fam.n_card[0]}")
    run = Run(pol, fam, tiers, lists or prefix_lists(fam.L, fam.E))
    return run, [request(run, chain.X, q, steps) for q in chain.reqs]


def case_static_count():
    # 2 layers, E 4, top-1, 2 slots, seed prefix {0, 1}: layer 0 routes 0 1 2 3 (2 hits), layer 1 1 1 3 0 (3 hits)
    fam = _fam(L=2, n=2)
    ch = _chain([[(0,), (1,)], [(1,), (1,)], [(2,), (3,)], [(3,), (0,)]], [(0, 0, 4, None)])
    _, ms = _run(fam, "static", ch, "card=n2")
    assert abs(ms[0]["per_row"].sum() - 2.5) < 1e-12, ms[0]["per_row"]  # (2 + 3) / 2 slots a row


def case_freeze_is_open():
    path = os.path.join(HERE, "..", "..", "crates", "runtime", "tests", "data", "swaprule-open-fixture.json")
    with open(path, encoding="utf-8") as f:
        fx = json.load(f)
    p = fx["params"]
    E, L, n = p["n_expert"], len(fx["seed"]), p["capacity"]
    lists = [np.asarray(s) for s in fx["seed"]]
    Xp = np.asarray([[fx["prompt"][l][t] for l in range(L)] for t in range(p["prompt_rows"])], dtype=np.int64)
    res = rr.seed_resident(lists, n, E)
    for m, key in ((p["m"], "flips"), (None, "flips_all")):
        pairs, _ = open_pairs(Xp, res, lists, E, rr.Rule("mid"), m, None)
        want = [(f["layer"], f["in"], f["out"]) for f in fx[key]]
        assert pairs == want, (key, pairs[:4], want[:4])
    # the relayout is a fixed point: from the set it leaves, the same counts pair nothing
    for i, a, b in open_pairs(Xp, res, lists, E, rr.Rule("mid"), None, None)[0]:
        res[i, a], res[i, b] = True, False
    assert open_pairs(Xp, res, lists, E, rr.Rule("mid"), None, None)[0] == []
    # through the word: a request's prompt-freeze moves the card set by the fixture's first m flips
    fam = _fam(L=L, E=E, K=p["top_k"], n=n[0] + 1, pp=lambda P: 1.0)  # a 48 s call: every copy lands in it
    Xd = Xp[:1]
    ch = Chain("open", np.concatenate([Xp, Xd]), [Req(0, 0, len(Xp), len(Xp) + 1, len(Xp))], "fixture")
    run, _ = _run(fam, f"prompt-freeze:m={p['m']}", ch, lists=lists)
    moved = {(i, int(a)) for i in range(L) for a in np.nonzero(run.st.resident[i] & ~rr.seed_resident(lists, n, E)[i])[0]}
    assert moved == {(f["layer"], f["in"]) for f in fx["flips"]}, sorted(moved)[:4]


def case_rule_twin():
    # every swaprule fixture's flips, by `rule:mid:carry=request` over its passes, run as two chained replays
    data = os.path.join(HERE, "..", "..", "crates", "runtime", "tests", "data")
    names = [n for n in sorted(os.listdir(data)) if n.startswith("swaprule-") and n.endswith(".json") and "open" not in n]
    assert len(names) >= 3, names
    for name in names:
        with open(os.path.join(data, name), encoding="utf-8") as f:
            top = json.load(f)
        for fx in (top, top["cap_case"]):
            p = fx["params"]
            L, E = len(fx["trace"]), p["n_expert"]
            X, passes = [], []
            for q in range(len(fx["kept"])):
                r0 = len(X)
                X += [[fx["trace"][l][q][r] for l in range(L)] for r in range(len(fx["trace"][0][q]))]
                passes.append((r0, len(X), fx["kept"][q]))
            X = np.asarray(X, dtype=np.int64)
            pol = Policy("rule:mid:carry=request", _fam())
            rule = rr.Rule(pol.rule.name, cap=p["cap"])
            res0 = np.zeros((L, E), dtype=bool)
            for l, s in enumerate(fx["seed"]):
                res0[l, s] = True
            pin = away = None
            if "pinned" in p:
                pin = np.zeros_like(res0)
                for l, s in enumerate(fx["seed"]):
                    pin[l, s[:p["pinned"][l]]] = True
            if "away" in p:
                away = np.zeros_like(res0)
                for l, ids in enumerate(p["away"]):
                    away[l, ids] = True
            flips = []
            st = rr.ReplayState(res0, np.zeros((L, E)), np.zeros_like(res0), np.zeros_like(res0), [], 0, 0, 0)
            cut = len(passes) // 2
            for part in (passes[:cut], passes[cut:]):
                base = part[0][0]
                r = rr.replay(X[base:part[-1][1]], None, p["capacity"], E, rule, sem="flip", spares=p["spares"],
                              d=p["d"], link=rr.Budget(None), state0=st, pinned=pin, away=away,
                              passes=[(a - base, b - base, k) for a, b, k in part],
                              on_flip=lambda b, i, o, n, live: flips.append(
                                  {"boundary": b, "layer": i, "out": o, "in": n, "live_at": live}))
                st = r.state
            assert flips == fx["flips"], (name, len(flips), len(fx["flips"]))


def case_carry_serve():
    # E 4, top-1, 2 slots less 1 spare: 1 live, seed {0}. Request 0 decodes 1 x8: the rule admits 1 (live at 5).
    # Request 1's prompt routes 3 x20 (no count), its decode starts on 1 under serve, on 0 under request.
    fam = _fam(n=2, d=1)
    rows = [[(2,)]] * 2 + [[(1,)]] * 8 + [[(3,)]] * 20 + [[(1,)]] * 2
    ch = _chain(rows, [(0, 2, 10, None), (10, 20, 32, None)])
    run, ms = _run(fam, "rule:mid:carry=serve", ch)
    assert ms[1]["per_row"][0] == 1.0, ms[1]["per_row"]
    assert run.st.counts[0, 3] == 0.0, run.st.counts
    _, ms = _run(fam, "rule:mid:carry=request", ch)
    assert ms[1]["per_row"][0] == 0.0, ms[1]["per_row"]


def case_call_end():
    # E 8, top-2, 3 live: the prompt routes 5 and 6 on every row (count 40 >= floor 32); a decode history -> return
    fam = _fam(E=8, K=2, n=4, d=1)
    rows = [[(1, 2)]] * 4 + [[(5, 6)]] * 40 + [[(1, 2)]] * 4 + [[(5, 6)]] * 40 + [[(5, 6)]] * 2
    ch = _chain(rows, [(0, 0, 4, None), (4, 40, 48, None), (48, 40, 90, None)])
    for end, kept_at_1 in (("return", False), ("keep", True)):
        run = Run(Policy(f"prompt+rule:end={end}", fam), fam, Tiers(fam, "card=n4"), prefix_lists(1, 8))
        request(run, ch.X, ch.reqs[0], 256)  # a decode history first
        before = run.st.resident.copy()
        request(run, ch.X, ch.reqs[1], 256)
        assert bool(run.st.resident[0, 5]) == kept_at_1, (end, np.nonzero(run.st.resident[0])[0])
        if end == "return":
            # the call ended where it started: the map is the one before the call, the rule's own flips aside
            assert (run.st.resident[0, 5:7] == before[0, 5:7]).all(), (before, run.st.resident)
    # without a history the return word keeps the picks (a fresh server)
    run = Run(Policy("prompt+rule:end=return", fam), fam, Tiers(fam, "card=n4"), prefix_lists(1, 8))
    request(run, ch.X, ch.reqs[2], 256)
    assert run.st.resident[0, 5] and run.st.resident[0, 6], np.nonzero(run.st.resident[0])[0]


def _rolling_sets(W):
    fam = _fam(n=2, d=1)
    rows = [[(1,)], [(1,)], [(2,)], [(1,)], [(2,)], [(2,)], [(3,)], [(3,)]]
    ch = _chain(rows, [(0, 0, 8, None)])
    run = Run(Policy(f"rolling:{W}:every=1", fam), fam, Tiers(fam, "card=n2"), prefix_lists(1, 4))
    sets = []
    rec = np.zeros(ch.X.shape, dtype=bool)
    for k in range(8):
        rolling_decode(run, ch.X[k:k + 1], rec[k:k + 1])
        sets.append(int(np.nonzero(run.st.resident[0])[0][0]))
    return sets


def case_rolling_window():
    # 1 live slot, seed {0}, a plan after every row, a flip live one pass later (it lands before the next plan).
    # W 2: row 0 admits 1; the windows (1 2), (2 1), (1 2) tie; (2 2) after row 5 admits 2, on the card from row 7;
    # (3 3) admits 3 after row 7. A window of 3 holds (2 1 2) after row 4 and admits 2 a row earlier.
    assert _rolling_sets(2) == [0, 1, 1, 1, 1, 1, 2, 2], _rolling_sets(2)


def case_profile_split():
    # learn requests decode 1 (8 rows), the held one 2 (12 rows): the profile's top is 1
    rows = [[(1,)]] * 4 + [[(2,)]] * 12 + [[(1,)]] * 4
    ch = _chain(rows, [(0, 0, 4, "learn"), (4, 0, 16, "held"), (16, 0, 20, "learn")])
    lists = profile_lists(ch, 1, 4, "decode")
    assert int(lists[0][0]) == 1, lists[0]


def _lcg_chain(g, T=160, E=16, K=2, L=2, drift=True):
    """Two requests of 20 prompt + 60 decode rows, 3 in 4 ids from a hot five, which moves every 40 rows (drift)
    or never."""
    rows = []
    hot = [g.below(E) for _ in range(5)]
    for t in range(T):
        if drift and t % 40 == 0:
            hot = [g.below(E) for _ in range(5)]
        layer = []
        for _ in range(L):
            row = []
            while len(row) < K:
                e = hot[g.below(5)] if g.below(4) < 3 else g.below(E)
                if e not in row:
                    row.append(e)
            layer.append(tuple(row))
        rows.append(layer)
    return _chain(rows, [(0, 20, 80, None), (80, 20, 160, None)])


def case_belady_bound():
    words = ["static", "prompt-freeze", "prompt-call:floor=2", "rule:mid:carry=serve", "rolling:8",
             "prompt+rule:seed=decayed:end=keep", "prompt+rule:end=open", "oracle-static"]
    # stationary traces too: there the window's own static set holds the hot five, which MIN beats only by bypass
    for s in range(20):
        g = rr.Lcg(s + 1)
        ch = _lcg_chain(g, drift=s % 2 == 0)
        fam = _fam(L=2, E=16, K=2, n=5, floor=2, batch=10, group=1)
        tiers = Tiers(fam, "card=n5")
        res = replay_chain(ch, fam, tiers, [Policy("belady", fam)] + [Policy(w, fam) for w in words], 256, None)
        for q in range(2):
            b = res["belady"][q]["h"]
            for w in words:
                assert res[w][q]["h"] <= b + 1e-12, (s, q, w, res[w][q]["h"], b)


def case_pacing_late():
    # one layer, 2 slots less 1 spare, seed {0}; the prompt routes 1 x8 at the call's end; a 1 GB/s link, 16.8 MB
    # experts (16.8 ms a copy), a 10 ms step, a call of 8 ids at 1e9 tok/s (it ends at once): the freeze's flip of 1
    # is issued at the call's end, lands 16.8 ms later and serves from decode step ceil(1.68) = 2.
    fam = _fam(n=2, d=1, gbps=1.0, B=16_800_000, step_ms=10.0)
    rows = [[(1,)]] * 8 + [[(1,)]] * 4
    ch = _chain(rows, [(0, 8, 12, None)])
    run, ms = _run(fam, "prompt-freeze", ch)
    assert ms[0]["per_row"].tolist() == [0.0, 0.0, 1.0, 1.0] and ms[0]["late"] == 1, ms[0]


def case_cost_v41_clip():
    fam = Family("v41", load_families())
    for h, F, want in ((0.1746, 0.0, 34.08), (0.4698, 5.56, 27.74), (0.6333, 4.32, 22.79)):
        got = fam.cost(h, F)["line"]
        assert abs(got - want) < 0.6, (h, F, got, want)
    # the replay's h of the 228 eligible slots is the line's of 240
    assert abs(fam.h_cost(1.0) - 228 / 240) < 1e-12, fam.h_cost(1.0)


def case_q38_delegate():
    fam = Family("q38", load_families())
    p = dict(q38_params(), flips_tok=0.87)
    for h in (0.5, 0.9):
        assert fam.cost(h, 0.87, depth=640)["model"] == q38.step(p, 640, hit=h)[0], h


def case_call_pick():
    # swaprule.rs call_pick: pool {0, 1, 2} counts 5 0 9; off 3 (40), 4 (33), 5 (31 < floor 32): wanted by count
    # 3 4 2 (cap = pool 3); 3 and 4 enter for the coldest left out, 1 (0) then 0 (5)
    live = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    c = np.array([5, 0, 9, 40, 33, 31])
    assert call_pick(live, ~live, c, 1 << 30, 32) == [(3, 1), (4, 0)], call_pick(live, ~live, c, 1 << 30, 32)
    assert call_pick(live, ~live, c, 2, 32) == [(3, 1), (4, 0)]
    assert call_pick(live, ~live, c, 1 << 30, 41) == []


def case_refusals():
    fam = Family("v41", load_families())
    with tempfile.TemporaryDirectory() as root:
        d = rr.q38_engine_set(root, "torn", {l: [[0] * 6] * 4 for l in range(40)}, complete=False, n_expert=384)
        for fn, needle in ((lambda: read_set(fam, d, [8], 4), "# complete"),
                           (lambda: Tiers(fam, "card=n400"), "n_expert 384"),
                           (lambda: Tiers(fam, "card=n70,disk=1"), "no key 'disk'"),
                           (lambda: Policy("prompt-thaw", fam), "no word 'prompt-thaw'"),
                           (lambda: Policy("rule:carry=never", fam), "carry serve or request"),
                           (lambda: Family("mixtral", {}), "no family 'mixtral'"),
                           (lambda: cmd_replay(argparse.Namespace(family="v41", split="test", budget="card=plan",
                                                                  policies="", prompt=None, sets=[], data=root,
                                                                  steps=4, json=None), out=lambda s: None),
                            "learn or held")):
            try:
                fn()
            except Refused as e:
                assert needle in str(e), (needle, str(e))
            else:
                raise AssertionError(f"not refused: {needle}")


def case_decide_table():
    def band(pr, c):
        return dict(pr=pr, c=c, n=24)
    lines = {
        "(i)": [("chat-short", 0, band((-5, 1), (4, 0.5))), ("chat", 128, band((1, 1), (6, 1)))],
        "(ii)": [("chat-short", 0, band((-12, 1), (0, 1))), ("chat", 128, band((-8, 2), (1, 1)))],
        "(iii)": [("chat-short", 0, band((-12, 1), (0, 1))), ("chat", 128, band((-2, 1), (5, 1))),
                  ("long", 2048, band((2, 1), (9, 2)))],
        "undecided": [("chat-short", 0, band((-12, 1), (2.5, 1)))],
    }
    for want, bands in lines.items():
        got = decide_family(bands)
        assert got.startswith(want), (want, got)
    assert "P* = 128" in decide_family(lines["(iii)"])
    out = []
    rows = {"x": [dict(chain="c", req=0, h=0.5, h32=0.5)], "x@profile": [dict(chain="c", req=0, h=0.6, h32=0.6)]}
    pd = profile_delta(rows)
    assert pd[0] == "x@profile" and abs(pd[1][0] - 10.0) < 1e-9, pd
    decide({"strata": {"fam": {"chat-short": rows}}}, out.append, dc="x")
    assert any("profile" in line for line in out), out
    pair = {DPR[0]: [dict(chain="c", req=i, h=0.5 + 0.01 * i) for i in range(4)],
            DPR[1]: [dict(chain="c", req=i, h=0.4) for i in range(4)],
            DC: [dict(chain="c", req=i, h=0.5 + 0.01 * i) for i in range(4)],
            "o": [dict(chain="c", req=i, h=0.6 + 0.01 * i) for i in range(4)]}
    assert abs(deltas(pair)["c"][0]) < 1e-9 and abs(deltas(pair, "o")["c"][0] - 10.0) < 1e-9, deltas(pair, "o")
    try:
        decide({"strata": {"fam": {"chat": pair}}}, out.append, dc="none")
        raise AssertionError("an absent companion word was read")
    except Refused:
        pass


CASES = [case_static_count, case_freeze_is_open, case_rule_twin, case_carry_serve, case_call_end,
         case_rolling_window, case_profile_split, case_belady_bound, case_pacing_late, case_cost_v41_clip,
         case_q38_delegate, case_call_pick, case_refusals, case_decide_table]


def self_test():
    failed = []
    for case in CASES:
        try:
            case()
        except (AssertionError, Refused, rr.ToolError) as e:
            failed.append(case.__name__)
            print(f"FAIL {case.__name__}: {e!r}", file=sys.stderr)
    if failed:
        print(f"resplay: self-test FAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"resplay: self-test ok ({len(CASES)} cases)")
    return 0


# ============================================================================ main


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"resplay.py: {message}", file=sys.stderr)
        sys.exit(64)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--self-test"]:
        return self_test()
    ap = Parser(prog="resplay.py", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True, parser_class=Parser)
    r = sub.add_parser("replay")
    r.add_argument("family")
    r.add_argument("sets", nargs="+")
    r.add_argument("--data", required=True)
    r.add_argument("--split")
    r.add_argument("--prompt")
    r.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    r.add_argument("--budget", default="card=plan")
    r.add_argument("--policies", default="")
    r.add_argument("--json")
    p = sub.add_parser("predict")
    p.add_argument("family")
    p.add_argument("--hit", type=float, required=True)
    p.add_argument("--flips", type=float, default=0.0)
    p.add_argument("--nvme", type=float, default=0.0)
    p.add_argument("--depth", type=int, default=512)
    d = sub.add_parser("decide")
    d.add_argument("json")
    d.add_argument("--companion", default=DC, help=f"the word Δ_c reads (default {DC})")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "replay":
            if a.steps < 1:
                raise Refused(f"--steps {a.steps}: at least 1")
            cmd_replay(a)
        elif a.cmd == "predict":
            cmd_predict(a)
        else:
            cmd_decide(a)
        return 0
    except Refused as e:
        print(f"resplay.py: refused: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
