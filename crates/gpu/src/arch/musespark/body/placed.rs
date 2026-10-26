//! A muse-spark load whose routed experts the card cannot all hold: the
//! plan's card share of each layer's stacks resident (the id prefix
//! `[0, n_card)`), the rest served by the common host tier ([`crate::host`]) in
//! the latent, SiLU-gated. A layer the plan holds whole runs the whole-card
//! half ([`super::ffn`]); a layer with a host share runs this module's three
//! parts around the tier's step port, the module doc of [`super`] in the host
//! leg's order:
//!
//! - **front**: the layer's attention half, then the FFN half's norm, the
//!   router on the full stream, the latent down and its weighted norm into the
//!   boundary's activation (the host reads the experts' own input, whose width
//!   is the latent's), the handoff of the ids, weights and that row into the
//!   page with each slot's place in the card's stacks, and the go;
//! - **shadow**, under the host's work: the card's slots (the row's q8_1, the
//!   gate·up and the down over the places the handoff wrote; a
//!   [`HOST`](crate::hybrid::HOST) slot's warps return before their first
//!   load), each slot's down row scaled by `ffn_post_exp_norm` then normed;
//! - **back**: the wait, the card's slots' weighted sum
//!   (`ds41_ffn_card_acc_9`, the places below the layer's card count) plus the
//!   host's, the latent up, the scale-then-norm and the α/β mix.
//!
//! The experts' normed rows enter the sum before the router weights, so the
//! host sums what the card sums: each host expert's own row is normed on the
//! host before its weight ([`HostRun::set_post_norm`]), and the two sums add.
//! Up to rounding the card's share and the host's are the whole-card layer's
//! (the host quantizes the latent with its own kernel, norms in its own order
//! and sums its slots in its own).
//!
//! The prompt runs as passes through the tier's batch port
//! ([`super::prefill`]): the same three parts over the pass's rows.
//!
//! A plan of an expert tier card (plan (b′)) hangs it under the host tier
//! ([`hosted_hybrid_at`], the family's node service [`super::tier`]): a
//! layer the tier holds experts of hands the tier its image with the host's
//! ([`TierParts::enqueue_step_go`]), its back waits for both, and the tier's
//! slots join the card's and the host's sums ([`enqueue_tier_sum`]).

use super::tier::{self, NODE_FAMILY, TierParts, TierSum, enqueue_tier_sum};
use super::{
    Acts, Kernels, Layer, STEP, Scratch, Shape, Taps, attention, copy_row, experts_card, ffn,
    ffn_post, ffn_pre,
};
use crate::arch::hybrid::GqaCx;
use crate::card_slots::CardSlotKernels;
use crate::host::handoff::{Handoff, HandoffKernels};
use crate::host::leg_tier::{Hang, LegOpen};
use crate::host::run::{HostRun, HostWidths};
use crate::host::served::{ResidencyParts, TierBody};
use crate::host::swap_source::{FileSwap, ResidencyGlue};
use crate::host::tier::TierOpen;
use crate::host::{StepLeg, refuse_tier_count};
use crate::hybrid::{Chain, HostResidency, Hybrid, SlotMap};
use crate::model::hosted_hybrid_at;
use crate::routed_block::handoff_go;
use crate::tensor::DeviceTensor;
use crate::{Gpu, GpuError};
use bloomery_levers::HostCfg;
use cuda_core::{CudaStream, DeviceBuffer};
use model::arch::musespark::host::{post_exp_gain, routed_layers};
use model::arch::musespark::hparams::Hparams;
use model::placement::Plan;
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};
use std::sync::Arc;

const WHAT: &str = "musespark placed load";

/// The slot map the plan makes over every layer of the chain, one row of
/// `stride` places a layer (the layers' expert count), with the expert tier
/// card's places when the plan hangs one. A plan of other than one stage card
/// holding every layer, or of more expert tiers than the host tier serves, is
/// refused by name.
pub(super) fn plan_map(plan: &Plan<'_>, hp: &Hparams) -> Result<(SlotMap, usize), GpuError> {
    let n = hp.n_layer;
    let spec = plan
        .machine
        .cards
        .first()
        .ok_or(GpuError::shape(WHAT, "a plan of no card"))?;
    let n_tiers = plan.machine.expert_tiers().count();
    refuse_tier_count(WHAT, n_tiers)?;
    if plan.machine.cards.len() != 1 || spec.layers != (0..n) {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "a plan of {} cards and {n_tiers} expert tiers, card 0 running layers {:?}; the \
                 body runs every layer (0..{n}) on one stage card beside at most one expert tier",
                plan.machine.cards.len(),
                spec.layers
            ),
        ));
    }
    let stride = hp.n_expert;
    let tiers: Vec<usize> = (0..n_tiers).map(|i| plan.machine.cards.len() + i).collect();
    Ok((
        SlotMap::of_plan_tiers(plan, 0, &tiers, 0..n, stride)?,
        stride,
    ))
}

/// What [`Placed::new`] builds a load's placed side from.
pub(super) struct PlacedOpen<'a> {
    /// The plan the load's weights were uploaded by.
    pub(super) plan: &'a Plan<'a>,
    /// The file, shared with the host tier.
    pub(super) file: Arc<gguf::Split>,
    pub(super) host: HostCfg,
    /// The plan's host set as the load left it.
    pub(super) set: HostResidency,
    /// The residency machine's glue: off on a load that runs none, which keeps
    /// the plan's slot map.
    pub(super) glue: ResidencyGlue,
}

/// The card's side of a placed load: which layers the host serves a share of,
/// the slot map's card copy, the handoff and sum launches and the buffers of
/// one layer's host leg.
pub(super) struct PlacedSide {
    pub(super) hosted: Vec<bool>,
    /// The slot map's card copy, shared with the residency machine, which
    /// rewrites a layer's row when it swaps an expert in.
    pub(super) slots: Arc<DeviceTensor<u32>>,
    /// The places of a row of the card copy: the layers' expert count.
    pub(super) stride: usize,
    handoff: HandoffKernels,
    pub(super) acc_k: CardSlotKernels,
    /// The routed slots' places in the card's stacks, or `HOST`.
    sel: DeviceBuffer<u32>,
    /// The card slots' weighted sum, a latent row.
    acc: DeviceBuffer<f32>,
    /// The stage card's side of the expert tier, when the plan hangs one.
    pub(super) tier: Option<TierParts>,
}

/// The placed side of a load: the host tier over the layers with a host
/// share, and what the walk reads beside it.
pub struct Placed {
    pub(super) hybrid: Hybrid<HostRun>,
    pub(super) side: PlacedSide,
    glue: ResidencyGlue,
}

impl Placed {
    /// The placed side over `layers` (the chain as loaded) of a plan whose
    /// slot map is `map` at `stride` places a row: the host run over the layers
    /// whose card count is under their experts, each with its post-expert norm
    /// (the file's gain at the file's post epsilon), the boundary at the
    /// latent's width, the tier watching the fault word and holding the host
    /// set, the plan's expert tier card hung under it ([`hosted_hybrid_at`]),
    /// its batch port and union slabs for prompt passes of up to `rows` rows,
    /// and the open's residency glue, given the host run's NVMe tier (started
    /// by the load once the body is sized). Refused by name: a layer whose card
    /// count the map and the layer disagree on. Load-time only.
    pub(super) fn new(
        gpu: &Gpu,
        hp: &Hparams,
        layers: &[Layer],
        (map, stride): (SlotMap, usize),
        (open, rows): (PlacedOpen<'_>, usize),
    ) -> Result<Placed, GpuError> {
        let n = layers.len();
        let mut hosted = vec![false; n];
        for (l, layer) in layers.iter().enumerate() {
            let on_card = map.on_card(l)?;
            if on_card != layer.n_card {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "layer {l}: the slot map puts {on_card} experts on the card, the layer \
                         loaded {}",
                        layer.n_card
                    ),
                ));
            }
            hosted[l] = on_card < hp.n_expert;
        }
        let set: Vec<usize> = (0..n).filter(|&l| hosted[l]).collect();
        let stream = gpu.stream();
        let slots = Arc::new(DeviceTensor::upload(stream, &map.stage_view(), n, stride)?);
        let width = hp.latent;
        let tiers = TierOpen::of_machine(open.plan.machine)?;
        let hang = (!tiers.is_empty()).then(|| Hang {
            tiers: &tiers,
            family: &NODE_FAMILY,
            open: LegOpen {
                n_expert: hp.n_expert,
                layers: n,
                hidden: width,
                n_used: hp.n_used,
                rows: tier::PAGE_ROWS,
            },
            cols: rows,
            card_dontneed: open.host.card_dontneed,
            bytes: crate::model::CardBytes::File,
        });
        let mut hybrid = hosted_hybrid_at(
            (gpu, &open.file, open.plan),
            (&set, n, map),
            (open.host, open.set, &open.glue),
            (
                HostWidths {
                    embd: width,
                    ff: hp.expert_width(),
                    n_used: hp.n_used,
                },
                1,
            ),
            hang.as_ref(),
            |src| routed_layers(src, hp, &set),
        )?;
        let experts = hybrid.host_mut();
        for &l in &set {
            experts.set_post_norm(l, post_exp_gain(&open.file, hp, l)?, hp.post_eps)?;
        }
        // A routed slot is a column of the union a post-norm layer's call is: a step's `n_used`, a prompt pass's
        // `rows · n_used`.
        experts.prepare_union(rows.max(1) * hp.n_used)?;
        hybrid.prepare_batch(gpu.context(), rows)?;
        let tier = if tiers.is_empty() {
            None
        } else {
            Some(TierParts::new(gpu, hybrid.slots(), n, (rows, hp.n_used))?)
        };
        Ok(Placed {
            hybrid,
            side: PlacedSide {
                hosted,
                slots,
                stride,
                handoff: HandoffKernels::load(gpu.context())?,
                acc_k: CardSlotKernels::load(gpu.context())?,
                sel: DeviceBuffer::zeroed(stream, hp.n_used)?,
                acc: DeviceBuffer::zeroed(stream, width)?,
                tier,
            },
            glue: open.glue,
        })
    }

    /// Device bytes the placed side holds: the slot map's card copy, the places
    /// and the sum, the tier's places, and the boundary.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        let s = &self.side;
        s.slots.buf().num_bytes()
            + s.sel.num_bytes()
            + s.acc.num_bytes()
            + s.tier.as_ref().map_or(0, TierParts::bytes)
            + self.hybrid.boundary().device_bytes()
    }

    /// The expert tier's experts of layer `l`: 0 off the tier and on a load
    /// that hangs none.
    #[must_use]
    pub fn tier_k(&self, l: usize) -> usize {
        self.side.tier.as_ref().map_or(0, |t| t.k(l))
    }

    /// Whether the host serves a share of layer `l`. A layer past the model
    /// panics by name.
    #[must_use]
    pub fn hosted(&self, l: usize) -> bool {
        self.side.hosted.get(l).copied().unwrap_or_else(|| {
            panic!(
                "hosted: layer {l} is past the model's {} layers",
                self.side.hosted.len()
            )
        })
    }

    /// The residency machine's source, when the load runs one.
    #[must_use]
    pub fn residency_source(&self) -> Option<&FileSwap> {
        self.glue.source()
    }

    /// The host tier's settling ([`crate::host::HostTier::settle`]).
    pub(super) fn reset(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        self.hybrid.settle(stream)
    }
}

impl TierBody for Placed {
    type Experts = HostRun;

    fn hybrid(&self) -> &Hybrid<HostRun> {
        &self.hybrid
    }

    fn hybrid_mut(&mut self) -> &mut Hybrid<HostRun> {
        &mut self.hybrid
    }

    /// The machine's parts when the glue holds one; a load that runs none
    /// keeps the plan's slot map.
    fn residency_parts(&mut self) -> Option<ResidencyParts<'_, HostRun>> {
        self.glue.source()?;
        Some(ResidencyParts {
            glue: &mut self.glue,
            hybrid: &mut self.hybrid,
            slots: &self.side.slots,
        })
    }

    fn serve_chain(&mut self, chain: Chain) -> Result<(), GpuError> {
        self.hybrid.serve_captured_of(chain)
    }
}

/// The captured decode step's walk `(1, 1, Step)` through the host tier's
/// step port: each layer's attention half, then its FFN half whole or as the
/// front, shadow and back around the host's go.
pub(super) struct StepWalk<'a> {
    pub(super) cx: GqaCx<'a>,
    pub(super) k: &'a Kernels,
    pub(super) sh: Shape,
    pub(super) layers: &'a mut [Layer],
    pub(super) s: &'a mut Scratch,
    pub(super) acts: &'a mut Acts,
    pub(super) side: &'a mut PlacedSide,
    pub(super) taps: Option<&'a mut Taps>,
    /// The head's input, the residual stream between layers, and the body's own
    /// buffer between a layer's two halves.
    pub(super) hin: &'a mut DeviceBuffer<f32>,
    pub(super) r: &'a mut DeviceBuffer<f32>,
}

impl<'a> StepWalk<'a> {
    /// Walk every layer through `leg`.
    pub(super) fn walk(mut self, leg: &mut StepLeg<'a, HostRun>) -> Result<(), GpuError> {
        let o = Overlap {
            units: 1,
            cols: 1,
            port: PortKind::Step,
        };
        let n = self.layers.len();
        sched::walk(o, n, leg, &mut self)
    }

    /// Whether the step runs layer `l`: every layer, but those a gate's span
    /// leaves out.
    fn runs(&self, l: usize) -> bool {
        self.taps.as_deref().is_none_or(|t| t.runs(l))
    }

    /// Layer `l`'s compared copies, when the taps are armed: the layer is over.
    fn snap(&mut self, l: usize) -> Result<(), GpuError> {
        match self.taps.as_deref_mut() {
            Some(t) => t.layers[l].snap(self.cx.gpu.stream(), self.s, self.r, self.hin, self.sh),
            None => Ok(()),
        }
    }
}

impl<'a> LayerProgram for StepWalk<'a> {
    type Port = StepLeg<'a, HostRun>;

    fn host_leg(&self, at: At) -> bool {
        self.side.hosted[at.layer] && self.runs(at.layer)
    }

    /// A layer with no host share whole; a hosted one up to its go.
    fn front(&mut self, leg: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        if !self.runs(l) {
            return Ok(());
        }
        let (gpu, sh) = (self.cx.gpu, self.sh);
        let stream = gpu.stream();
        if let Some((from, row)) = self.taps.as_deref().and_then(|t| t.inject.as_ref())
            && *from == l
        {
            copy_row(stream, row, self.hin, sh.hidden)?;
        }
        let a = &mut self.layers[l];
        attention(
            gpu,
            &self.cx,
            (l, a),
            (&mut *self.s, &mut *self.acts),
            (&*self.hin, &mut *self.r),
            sh,
        )?;
        if !self.side.hosted[l] {
            let a = &mut self.layers[l];
            ffn(
                gpu,
                &self.cx,
                self.k,
                (l, a),
                (&mut *self.s, &mut *self.acts),
                (&*self.r, &mut *self.hin),
                sh,
            )?;
            return self.snap(l);
        }
        let hy = leg.hybrid();
        let a = &mut self.layers[l];
        ffn_pre(
            gpu,
            &self.cx,
            self.k,
            (l, &mut *a),
            (&mut *self.s, &mut *self.acts),
            &*self.r,
            Some(hy.boundary_mut().normed_mut()),
            sh,
        )?;
        let PlacedSide {
            slots,
            stride,
            handoff,
            sel,
            tier,
            ..
        } = &mut *self.side;
        let h = Handoff {
            ids: a.route.ids(),
            weights: a.route.weights(),
            map: slots.buf(),
            row_off: l * *stride,
            n_expert: sh.n_expert,
        };
        match tier.as_mut().filter(|t| t.k(l) > 0) {
            Some(t) => t.enqueue_step_go(gpu, handoff, &h, hy, l, sel),
            None => handoff_go(gpu, handoff, &h, hy, (l, 0), sel),
        }
    }

    /// The card's slots of a hosted layer, under the host's work.
    fn shadow(&mut self, leg: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        if !self.side.hosted[l] || !self.runs(l) || self.layers[l].n_card == 0 {
            return Ok(());
        }
        experts_card(
            self.cx.gpu,
            &self.cx,
            self.k,
            (l, &self.layers[l]),
            (&mut *self.s, &mut *self.acts),
            (Some(leg.hybrid().boundary().normed()), Some(&self.side.sel)),
            self.sh,
        )
    }

    /// A hosted layer from the join on: the host's sum and the card's slots',
    /// then the FFN half's tail.
    fn back(&mut self, leg: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        if !self.side.hosted[l] || !self.runs(l) {
            return Ok(());
        }
        let (gpu, sh) = (self.cx.gpu, self.sh);
        let stream = gpu.stream();
        let a = &self.layers[l];
        let hy = leg.hybrid();
        let n_tier = self.side.tier.as_ref().map_or(0, |t| t.k(l));
        if n_tier > 0 {
            hy.enqueue_tier_back(stream, l, 0)?;
        } else {
            hy.boundary().enqueue_back_of(stream, 0, l)?;
        }
        let hsum = hy.boundary().hsum_of(0)?;
        if a.n_card > 0 {
            self.side.acc_k.enqueue_card_acc(
                stream,
                &sh.card_acc(
                    (&self.s.rows2, a.route.weights(), &self.side.sel),
                    (1, a.n_card),
                    gpu.layer_sink(l)?,
                ),
                &mut self.side.acc,
            )?;
        }
        // The card's slots, then the tier's, then the host's: `s.lat` holds
        // the tier's sum and `s.lat2` the two cards' (a hosted layer's
        // experts read the boundary's row, not those).
        let cards = if n_tier > 0 {
            let t = self.side.tier.as_ref().ok_or(GpuError::state(
                WHAT,
                "the stage card's tier side of a tier layer",
            ))?;
            let sum = TierSum {
                trows: hy.tier_rows(0)?,
                gain: self.cx.w.f32_buf(STEP, &a.post_exp_norm)?,
                post_eps: sh.post_eps,
                w: a.route.weights(),
                tsel: t.step_tsel()?,
                n_tier,
                n: sh.latent,
                m: 1,
                n_used: sh.top_k,
                fault: gpu.layer_sink(l)?,
            };
            enqueue_tier_sum(
                gpu,
                &self.side.acc_k,
                &sum,
                (&mut self.s.rows, &mut self.s.lat),
            )?;
            if a.n_card > 0 {
                gpu.elem().enqueue_add(
                    stream,
                    &self.side.acc,
                    &self.s.lat,
                    sh.latent,
                    &mut self.s.lat2,
                )?;
                Some(&self.s.lat2)
            } else {
                Some(&self.s.lat)
            }
        } else if a.n_card > 0 {
            Some(&self.side.acc)
        } else {
            None
        };
        match cards {
            Some(c) => gpu
                .elem()
                .enqueue_add(stream, c, hsum, sh.latent, &mut self.s.moe)?,
            None => copy_row(stream, hsum, &mut self.s.moe, sh.latent)?,
        }
        ffn_post(
            gpu,
            &self.cx,
            (l, a),
            (&mut *self.s, &mut *self.acts),
            (&*self.r, &mut *self.hin),
            sh,
        )?;
        self.snap(l)?;
        leg.hybrid().row_enqueued(l, 0)
    }
}
