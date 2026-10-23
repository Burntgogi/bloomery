//! The muse-spark load gate: the body's placed load of the gate plan, proven
//! against the plan. The plan is the gate card's (`gate_card`: the 3090's
//! bytes on the card in view in the real tier, the largest visible card in the
//! fixture tier, which under a budget lever keeps half of its experts on the
//! card so the host share is not empty), one stage card running every layer
//! and the head with no expert tier, at [`CTX`] positions. The body is opened
//! through the entry the engine opens the family with
//! (`GpuModel::<Body>::open_placed`). A correctness run: every time and byte
//! figure it prints is a runtime value. Every clause is self-consistency (the
//! load against its own plan and the file), so both tiers run all five.
//!
//! - `plan_places_every_expert`: the plan puts each routed expert in exactly
//!   one place — the card's prefix and the host's rest, no tier and no paged
//!   expert — the card's expert bytes are its experts' stacks in the card
//!   format's buffers (the loader's arithmetic, which pads each row to whole
//!   words), the host's are its experts' file bytes, and the card's KV term is
//!   the typed description's planes.
//! - `segments_are_the_plans`: every segment the plan puts on the card is
//!   resident at the plan's buffer bytes (measured from the buffers), nothing
//!   else is resident, and the weights' total is the card's dense and expert
//!   bytes.
//! - `expert_prefix_byte_for_byte`: on a sample of layers (the first, the
//!   middle and the last), each routed stack on the card is the file's own
//!   bytes of the first `n_l` experts, row count included; a sample of the
//!   F32 norms is the file's values bit for bit.
//! - `slot_map_is_the_prefix`: each layer's row of the tier's slot map holds
//!   the card's experts at slots `0..n_l` in id order and the host mark on
//!   the rest, and the layer's card count is the plan's.
//! - `resident_inside_the_plan`: what the load holds on the card (the
//!   weights, the K/V planes, the step's buffers, the host boundary) stays
//!   inside the plan's terms for them.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_musespark_load", gate::run())
}

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the card the gate plans on; the fixture arm reads its spec"
)]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "gpu")]
mod gate {
    use std::collections::BTreeSet;

    use crate::gate_card;
    use bloomery_gpu::GpuModel;
    use bloomery_gpu::arch::musespark::Body;
    use bloomery_gpu::host::served::TierBody;
    use bloomery_gpu::hybrid::HOST;
    use bloomery_gpu::weights::{DevWeight, Weights};
    use bloomery_gpu_gates::tier::{self, Tag};
    use bloomery_gpu_gates::{GateError, checks_failed, open_split, split_f32, verdict};
    use gguf::quant::GgmlType;
    use model::arch::Arch;
    use model::arch::musespark::hparams::Hparams;
    use model::arch::musespark::place::PlanInputs;
    use model::placement::{Device, Plan, PlanLevers, Role, accepts, workstation};

    type Model = GpuModel<Body>;

    const NAME: &str = "gate_musespark_load";

    /// The plan's and the load's context: the fixture's and the real file's
    /// prompts all fit, and the K/V planes stay small beside the weights.
    const CTX: u64 = 4_096;

    /// The host tier's configuration the load opens with: no lever is read.
    const HOST_CFG: bloomery_levers::HostCfg = bloomery_levers::HostCfg {
        populate: false,
        lock: false,
        card_dontneed: false,
        r8: false,
        route_trace: false,
    };

    /// The routed stacks, in the order the host leg reads them.
    const STACKS: [&str; 3] = ["gate", "up", "down"];

    /// The F32 norms whose values are compared bit for bit, per sampled layer.
    const NORMS: [&str; 3] = ["attn_norm", "ffn_post_exp_norm", "post_ffw_norm"];

    /// The layers a sample reads: the first, the middle and the last.
    fn sample(n_layer: usize) -> Vec<usize> {
        let mut layers = vec![0, n_layer / 2, n_layer - 1];
        layers.dedup();
        layers
    }

    /// The plan's levers: the fixture's card keeps half its experts (so the
    /// slot map has host marks and the prefix is a proper one), the real
    /// tier's the card's own bytes, which already leave most to the host.
    fn levers(
        inputs: &PlanInputs,
        machine: &model::placement::Machine,
        tier_now: tier::Tier,
    ) -> Result<PlanLevers, GateError> {
        if tier_now == tier::Tier::Real {
            return Ok(PlanLevers::default());
        }
        let all = inputs.plan(machine, CTX, &PlanLevers::default())?;
        let c = &all.cards[0];
        let card = &machine.cards[0];
        let kept = c.dense_bytes
            + c.rounding_bytes
            + c.kv_bytes
            + c.stage_rows_bytes
            + card
                .set_aside_bytes()
                .ok_or("the card's set-aside is unbounded")?
            + card.margin_bytes
            + c.expert_bytes / 2;
        Ok(PlanLevers {
            card_budget_bytes: Some(kept).into(),
        })
    }

    /// `want`'s buffers, measured from the device.
    fn buffers(dw: &DevWeight) -> Vec<u64> {
        dw.buffer_bytes().into_iter().map(|b| b as u64).collect()
    }

    /// The routed stacks' bytes where `plan` puts each layer's experts: its card prefix in the
    /// card format the plan loads it in (`accepts::leg_card_format`, measured by
    /// `CardFormat::buffer_bytes`: a row of a whole number of words, so a stack whose rows end
    /// inside a word takes more on the card than in the file), and the rest in the file's bytes.
    fn placed_stack_bytes(plan: &Plan<'_>, inputs: &PlanInputs) -> Result<(u64, u64), GateError> {
        let (mut card, mut host) = (0u64, 0u64);
        for t in inputs
            .model
            .tensors
            .iter()
            .filter(|t| t.role == Role::RoutedExperts)
        {
            let (Some(l), &[k, rows, experts]) = (t.layer, t.dims.as_slice()) else {
                return Err(format!("{}: a routed stack of dims {:?}", t.name, t.dims).into());
            };
            let on = plan.n_l[l];
            let per_card: u64 = accepts::leg_card_format(t.ty)
                .buffer_bytes(t.ty, k, rows)
                .ok_or_else(|| format!("{}: no card buffers of {rows} rows of {k}", t.name))?
                .iter()
                .sum();
            card += on * per_card;
            host += (experts - on) * (t.file_bytes / experts);
        }
        Ok((card, host))
    }

    fn plan_places_every_expert(
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        hp: &Hparams,
    ) -> Result<bool, GateError> {
        const CLAUSE: &str = "plan_places_every_expert";
        let card = &plan.cards[0];
        let on_card: u64 = plan.n_l.iter().sum();
        let total = (hp.n_layer * hp.n_expert) as u64;
        let (card_stacks, host_stacks) = placed_stack_bytes(plan, inputs)?;
        let planes = inputs.kv.bytes(0..hp.n_layer, plan.ctx_max);
        let mut bad = Vec::new();
        if on_card + plan.host.experts != total || card.experts != on_card {
            bad.push(format!(
                "{on_card} experts on the card (totals {}) + {} on the host, the file's {total}",
                card.experts, plan.host.experts
            ));
        }
        if (card.expert_bytes, plan.host.expert_bytes) != (card_stacks, host_stacks) {
            bad.push(format!(
                "expert bytes {} on the card + {} on the host, the stacks' {card_stacks} in the \
                 card format + {host_stacks} in the file",
                card.expert_bytes, plan.host.expert_bytes
            ));
        }
        if plan.host.nvme_expert_bytes != 0 || plan.tier_n_l.iter().flatten().any(|&n| n != 0) {
            bad.push(format!(
                "paged or tiered experts: nvme {} B, tier counts {:?}",
                plan.host.nvme_expert_bytes, plan.tier_n_l
            ));
        }
        if card.kv_bytes != planes {
            bad.push(format!(
                "kv bytes {} B, the planes' {planes} B",
                card.kv_bytes
            ));
        }
        println!(
            "{CLAUSE}: ctx {} card experts {on_card} ({} B), host experts {} ({} B), kv {} B \
             (plan values) {}",
            plan.ctx_max,
            card.expert_bytes,
            plan.host.experts,
            plan.host.expert_bytes,
            card.kv_bytes,
            verdict(bad.is_empty())
        );
        for b in &bad {
            println!("FAIL: {CLAUSE}: {b}");
        }
        Ok(bad.is_empty())
    }

    fn segments_are_the_plans(plan: &Plan<'_>, w: &Weights) -> Result<bool, GateError> {
        const CLAUSE: &str = "segments_are_the_plans";
        let (mut segments, mut planned, mut bad) = (0usize, BTreeSet::new(), Vec::new());
        for row in &plan.rows {
            let t = &plan.model.tensors[row.tensor];
            for seg in row.segments.iter().filter(|s| s.device == Device::Card(0)) {
                segments += 1;
                planned.insert(t.name.as_str());
                let want = seg.buffer_bytes(t, plan.model.experts);
                match (w.get(&t.name).map(buffers), want) {
                    (Some(got), Ok(want)) if got == want => {}
                    (Some(got), want) => bad.push(format!(
                        "{}: buffers {got:?} B resident, the plan's segment {want:?}",
                        t.name
                    )),
                    (None, _) => bad.push(format!("{}: not resident", t.name)),
                }
            }
        }
        for n in w.names().filter(|n| !planned.contains(n)) {
            bad.push(format!("{n}: resident, and the plan has no segment of it"));
        }
        let t = &plan.cards[0];
        let (got, want) = (w.resident_bytes() as u64, t.dense_bytes + t.expert_bytes);
        if got != want {
            bad.push(format!(
                "total {got} B, the plan's dense + experts {want} B"
            ));
        }
        println!(
            "{CLAUSE}: {segments} segments, {got} B resident, plan dense + experts {want} B {}",
            verdict(bad.is_empty())
        );
        for b in &bad {
            println!("FAIL: {CLAUSE}: {b}");
        }
        Ok(bad.is_empty())
    }

    fn expert_prefix_byte_for_byte(
        m: &Model,
        plan: &Plan<'_>,
        file: &gguf::Split,
        hp: &Hparams,
    ) -> Result<bool, GateError> {
        const CLAUSE: &str = "expert_prefix_byte_for_byte";
        let stream = m.gpu().stream();
        let w = m.weights();
        let (mut stacks, mut norms, mut bad) = (0usize, 0usize, Vec::new());
        for l in sample(hp.n_layer) {
            let n_l = usize::try_from(plan.n_l[l])?;
            for kind in STACKS {
                let name = format!("blk.{l}.ffn_{kind}_exps.weight");
                let (shard, info) = file
                    .find(&name)
                    .ok_or_else(|| format!("{name} is not in the file"))?;
                if info.ty != GgmlType::IQ2_XXS || info.dims.len() != 3 {
                    bad.push(format!(
                        "{name}: {:?} {:?}, want a IQ2_XXS stack",
                        info.ty, info.dims
                    ));
                    continue;
                }
                let per = usize::try_from(info.nbytes)? / hp.n_expert;
                let rows_per = usize::try_from(info.dims[1])?;
                let resident = w.get(&name);
                if n_l == 0 {
                    if resident.is_some() {
                        bad.push(format!(
                            "{name}: resident, and the plan keeps no expert of the layer"
                        ));
                    }
                    continue;
                }
                let Some(DevWeight::KQuant { ty, w: words, .. }) = resident else {
                    bad.push(format!("{name}: not resident as a K-quant stack"));
                    continue;
                };
                let got = words.buf().to_host_vec(stream)?;
                let bytes: Vec<u8> = got.iter().flat_map(|x| x.to_le_bytes()).collect();
                let data = file
                    .shard(shard)
                    .ok_or("shard index out of range")?
                    .data(info)?;
                let want = &data[..n_l * per];
                stacks += 1;
                if *ty != GgmlType::IQ2_XXS
                    || words.rows() != n_l * rows_per
                    || bytes.get(..n_l * per) != Some(want)
                {
                    bad.push(format!(
                        "{name}: {ty:?}, {} rows resident (want {}), first {} B {} the file's \
                         first {n_l} experts",
                        words.rows(),
                        n_l * rows_per,
                        n_l * per,
                        if bytes.get(..n_l * per) == Some(want) {
                            "equal"
                        } else {
                            "differ from"
                        }
                    ));
                }
            }
            for stem in NORMS {
                let name = format!("blk.{l}.{stem}.weight");
                let (_, info) = file
                    .find(&name)
                    .ok_or_else(|| format!("{name} is not in the file"))?;
                let n = usize::try_from(info.dims.iter().product::<u64>())?;
                let want = split_f32(file, &name, n)?;
                let Some(DevWeight::F32 { w: tensor, .. }) = w.get(&name) else {
                    bad.push(format!("{name}: not resident as an F32 tensor"));
                    continue;
                };
                let got = tensor.buf().to_host_vec(stream)?;
                norms += 1;
                if got.len() != want.len()
                    || got
                        .iter()
                        .zip(&want)
                        .any(|(a, b)| a.to_bits() != b.to_bits())
                {
                    bad.push(format!("{name}: values differ from the file's"));
                }
            }
        }
        println!(
            "{CLAUSE}: layers {:?}, {stacks} resident stacks and {norms} norms read back {}",
            sample(hp.n_layer),
            verdict(bad.is_empty())
        );
        for b in &bad {
            println!("FAIL: {CLAUSE}: {b}");
        }
        Ok(bad.is_empty())
    }

    fn slot_map_is_the_prefix(m: &Model, plan: &Plan<'_>, hp: &Hparams) -> Result<bool, GateError> {
        const CLAUSE: &str = "slot_map_is_the_prefix";
        let placed = m
            .body(NAME)?
            .placed()
            .ok_or("a placed load carries the host tier")?;
        let map = placed.hybrid().slots();
        let mut bad = Vec::new();
        for (l, &planned) in plan.n_l.iter().enumerate().take(hp.n_layer) {
            let n_l = usize::try_from(planned)?;
            let Some(row) = map.row(l) else {
                bad.push(format!("layer {l}: the map has no row"));
                continue;
            };
            let prefix = row
                .iter()
                .enumerate()
                .all(|(e, &s)| if e < n_l { s == e as u32 } else { s == HOST });
            if row.len() != hp.n_expert || !prefix || map.on_card(l)? != n_l {
                bad.push(format!(
                    "layer {l}: row of {} entries, card count {} (plan {n_l}), prefix {prefix}",
                    row.len(),
                    map.on_card(l)?
                ));
            }
        }
        println!(
            "{CLAUSE}: {} layers, the plan's card counts are the ids' prefix {}",
            hp.n_layer,
            verdict(bad.is_empty())
        );
        for b in &bad {
            println!("FAIL: {CLAUSE}: {b}");
        }
        Ok(bad.is_empty())
    }

    fn resident_inside_the_plan(m: &Model, plan: &Plan<'_>) -> Result<bool, GateError> {
        const CLAUSE: &str = "resident_inside_the_plan";
        let c = &plan.cards[0];
        let resident = m.resident_bytes() as u64;
        let planned =
            c.dense_bytes + c.expert_bytes + c.kv_bytes + c.stage_rows_bytes + c.scratch_bytes;
        let pass = resident <= planned;
        println!(
            "{CLAUSE}: resident {resident} B (runtime value) against the plan's dense + experts + \
             kv + stage rows + scratch {planned} B {}",
            verdict(pass)
        );
        Ok(pass)
    }

    pub fn run() -> Result<(), GateError> {
        let tier_now = tier::Tier::from_env()?;
        let file = open_split(Arch::MuseSpark, "gate-gpu-musespark-load")?;
        let hp = Hparams::read(&file).map_err(|e| format!("muse-spark header: {e}"))?;
        let inputs = PlanInputs::read(&file)?;
        let spec = gate_card::card()?;
        let machine = workstation::placed_on(spec, hp.n_layer, workstation::SCRATCH);
        let plan = inputs.plan(&machine, CTX, &levers(&inputs, &machine, tier_now)?)?;
        let mut ok = true;
        if tier::run_clause("plan_places_every_expert", Tag::SelfConsistency)? {
            let pass = plan_places_every_expert(&plan, &inputs, &hp)?;
            if !pass {
                println!("clause plan_places_every_expert FAIL");
            }
            ok &= pass;
        }
        let t = std::time::Instant::now();
        let m = Model::open_placed(
            open_split(Arch::MuseSpark, "gate-gpu-musespark-load")?,
            &plan,
            0,
            HOST_CFG,
        )?;
        println!(
            "load resident_bytes={} ctx={CTX} layers={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            m.layers().len(),
            t.elapsed().as_secs_f64()
        );
        type Clause<'a> = Box<dyn Fn() -> Result<bool, GateError> + 'a>;
        let clauses: Vec<(&str, Clause<'_>)> = vec![
            (
                "segments_are_the_plans",
                Box::new(|| segments_are_the_plans(&plan, m.weights())),
            ),
            (
                "expert_prefix_byte_for_byte",
                Box::new(|| expert_prefix_byte_for_byte(&m, &plan, &file, &hp)),
            ),
            (
                "slot_map_is_the_prefix",
                Box::new(|| slot_map_is_the_prefix(&m, &plan, &hp)),
            ),
            (
                "resident_inside_the_plan",
                Box::new(|| resident_inside_the_plan(&m, &plan)),
            ),
        ];
        // Every clause is self-consistency, run in both tiers; the first read
        // the plan alone, before the load.
        for (name, clause) in &clauses {
            if !tier::run_clause(name, Tag::SelfConsistency)? {
                continue;
            }
            let pass = clause()?;
            if !pass {
                println!("clause {name} FAIL");
            }
            ok &= pass;
        }
        println!("{}", tier::tally_line());
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} the plan puts every routed expert in one place and its bytes are the \
             file's; every segment on the card is resident at the plan's buffers and nothing \
             else is; the routed stacks and the sampled norms are the file's bytes; the slot \
             map is the ids' prefix; the load stays inside the plan"
        );
        Ok(())
    }
}
