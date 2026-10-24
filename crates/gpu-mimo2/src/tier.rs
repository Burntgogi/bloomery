//! The MiMo-V2 expert tier ([`bloomery_gpu::host::leg_tier`]): a card beside the stage card that
//! holds the next ids of each routed layer's experts after the stage card's prefix and computes the
//! routed slots the slot map sends it, in the host leg's shadow. The tier's layer is the common
//! [`LegTier`](bloomery_gpu::host::leg_tier::LegTier) over MiMo's leg shape ([`CHUNK`] tokens a
//! run), its stacks through the leg's gate·up arm at the routed SwiGLU (plain: MiMo's limit is 0)
//! and its `Q8Act` down arm; the stage card's side is the common [`TierSide`].
//!
//! A tier layer's join is the common one ([`tier_join`]): one weighted sum over the stage card's
//! and the tier's slots in slot order, plus `x1`, then the host's sum: the layer's output is the
//! one-card layer's over the union of the two cards' experts, bit for bit. The step's handoff
//! writes the tier's image and places with the host's (`RouteBlock::front_tiered`) and the back
//! waits for both ([`tier_join::enqueue_step_join`]); a prompt batch downloads each slot's tier
//! place with the host's rows ([`Hybrid::enqueue_download_tiered`]) and its back reads the tier's
//! rows through the batch port's join ([`bloomery_gpu::host::BatchLeg::join_tiered`]).
//!
//! What is MiMo's here is the node family that reads the file's description ([`NODE_FAMILY`]) and
//! the base the sum is added to: MiMo's routed block has no shared expert, so it is `x1` alone.

use std::ops::Range;

use bloomery_gpu::card_experts::{CardBatch, CardExperts};
use bloomery_gpu::host::leg_tier::{self, LegFamily, LegWidths, TierSide};
use bloomery_gpu::host::node::NodeFamily;
use bloomery_gpu::host::run::HostRun;
use bloomery_gpu::hybrid::{BatchKey, Hybrid, SlotMap};
use bloomery_gpu::routed_block::RouteBlock;
use bloomery_gpu::{Gpu, GpuError};
use bloomery_gpu_deepseek41::chain::ffn::FfnBatchKernels;
use bloomery_gpu_deepseek41::router::mimo2::{N_USED, RouterKernels};
pub(crate) use bloomery_gpu_deepseek41::tier_join::TierBatch;
use bloomery_gpu_deepseek41::tier_join::{self, TierSlots};
use cuda_core::DeviceBuffer;
use gguf::Split;
use model::ModelError;
use model::arch::mimo2::place::{self, PlanInputs};
use model::moe::HostLayer;
use model::placement::{ModelTensors, PlacementError};
use model::placement::workstation::TierBatchBytes;
use model::r8file::R8Source;
use models::LayerSpec;

use crate::body::prefill::T_MAX;
use crate::ffn::routed_layers;

/// What the tier's errors name.
const WHAT: &str = "mimo2 tier";

/// The tier page's rows: the step's one row in flight. A load of several sequences, whose pass
/// runs their rows as columns of one row, hangs no tier (the body refuses it by name).
pub(crate) const PAGE_ROWS: usize = 1;

/// Tokens one tier run covers: the card leg's capacity, which the plan reserves the leg's buffers
/// for ([`place::TIER_CHUNK`]); a block of a prompt unit's rows runs in runs of at most this many.
pub const CHUNK: usize = place::TIER_CHUNK as usize;

/// The tier's prompt-batch bytes the plan reserves on the tier card and the host for the file
/// `inputs` describes: [`place::tier_batch`] over blocks of a prompt unit's rows ([`T_MAX`]), the
/// columns the load makes the tier's batch legs for.
#[must_use]
pub fn tier_batch(inputs: &PlanInputs) -> Result<TierBatchBytes, PlacementError> {
    place::tier_batch(&inputs.hp, T_MAX as u64)
}

/// MiMo's architecture facts for the common tier node service: the file's reading, the step's one
/// row in flight ([`PAGE_ROWS`]) and the tier's chunk.
struct Mimo2;

impl LegFamily for Mimo2 {
    const WHAT: &'static str = WHAT;
    const WORD: &'static str = "mimo2";
    const ROWS: usize = PAGE_ROWS;
    const CHUNK: usize = CHUNK;
    type Inputs = PlanInputs;

    fn describe(file: &Split) -> Result<PlanInputs, GpuError> {
        PlanInputs::describe(file).map_err(|e| GpuError::plan(WHAT, e))
    }

    fn into_model(inputs: PlanInputs) -> ModelTensors {
        inputs.model
    }

    fn layers(inputs: &PlanInputs) -> &[LayerSpec] {
        &inputs.spec.layers
    }

    fn widths(inputs: &PlanInputs) -> LegWidths {
        let hp = &inputs.hp;
        LegWidths {
            n_expert: hp.n_expert,
            n_used: hp.n_used,
            n_embd: hp.n_embd,
            ff: hp.expert_ff,
            program: model::placement::accepts::Program::Mimo2Body,
        }
    }

    fn routed(
        src: R8Source<'_>,
        inputs: &PlanInputs,
        run: Range<usize>,
    ) -> Result<Vec<HostLayer>, ModelError> {
        routed_layers(src, &inputs.hp, run)
    }
}

/// MiMo's tier node service ([`leg_tier::node_family`]).
pub const NODE_FAMILY: NodeFamily = leg_tier::node_family::<Mimo2>();

/// The stage card's side of a tiered load: the common tier side (the slot map's tier view and the
/// step's tier places) and the join's kernels.
pub(crate) struct TierParts {
    side: TierSide,
    kernels: FfnBatchKernels,
    /// The model's width.
    n: usize,
}

impl TierParts {
    /// The parts for `map`'s tier over the description's `layers` layers, `n` wide. Load-time only.
    pub(crate) fn new(
        gpu: &Gpu,
        map: &SlotMap,
        layers: usize,
        n: usize,
    ) -> Result<TierParts, GpuError> {
        Ok(TierParts {
            side: TierSide::new(gpu, map, layers, (PAGE_ROWS, N_USED))?,
            kernels: FfnBatchKernels::load(gpu.context())?,
            n,
        })
    }

    /// Device bytes the step's side holds; the batch's are [`TierBatch::bytes`] (the common
    /// [`tier_join::TierBatch`] at [`N_USED`] slots a position).
    pub(crate) fn bytes(&self) -> usize {
        self.side.bytes()
    }

    /// The step's side, which the routed block's tiered front writes the step's tier places
    /// through.
    pub(crate) fn side_mut(&mut self) -> &mut TierSide {
        &mut self.side
    }

    /// Layer `l`'s back of the step on a tier layer (module doc): the common step join
    /// ([`tier_join::enqueue_step_join`]) of the page's one row over the stage card's slots
    /// (`card`'s step rows at the handoff's places `sel`) and the tier's rows with the router's
    /// `weights`, plus `x1`, then the host's sum, into `dest`. Three launches after the wait.
    pub(crate) fn enqueue_step_back(
        &self,
        gpu: &Gpu,
        (card, l): (&mut CardExperts, usize),
        (sel, weights): (&DeviceBuffer<u32>, &DeviceBuffer<f32>),
        hybrid: &Hybrid<HostRun>,
        (x1, dest): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
    ) -> Result<(), GpuError> {
        let slots = TierSlots {
            w: weights,
            tsel: self.side.tsel(0)?,
            n: self.n,
            n_card: card.n_card(l),
            n_tier: card.tier_k(l),
            n_used: N_USED,
        };
        tier_join::enqueue_step_join(
            gpu,
            &self.kernels,
            (hybrid, l, 0),
            (card.step_join(sel), &slots),
            (x1, dest),
        )
    }

    /// The download of a prompt unit's `m` rows of layer `key.layer` to the host's next batch set
    /// on a tier layer: each slot's tier place from the router's `ids` (the map's tier view) into
    /// `tb`'s places, then the rows, weights, ids and those places
    /// ([`Hybrid::enqueue_download_tiered`]), which the tier's service reads.
    pub(crate) fn enqueue_download_rows(
        &self,
        gpu: &Gpu,
        (routed, hybrid): (&RouteBlock<RouterKernels>, &mut Hybrid<HostRun>),
        (key, m): (BatchKey, usize),
        (normed, weights, ids): (&DeviceBuffer<f32>, &DeviceBuffer<f32>, &DeviceBuffer<u32>),
        tb: &mut TierBatch,
    ) -> Result<(), GpuError> {
        routed.tier_places_rows(
            gpu,
            (key.layer, m),
            ids,
            (self.side.places(), tb.places_mut()),
        )?;
        hybrid.enqueue_download_tiered(gpu.stream(), [normed, weights], ids, &[tb.places()], key)
    }

    /// Layer `l`'s join of a tier layer over a unit's `m` rows, once the host leg has landed the
    /// tier's rows `trows` ([`bloomery_gpu::host::BatchLeg::join_tiered`]): the common join
    /// ([`tier_join::enqueue_join`]) of every token's slots from the stage card's downs (`cb`'s,
    /// at its places) or the tier's rows, at the weights and places [`TierBatch::keep`]
    /// kept, then `x1` added, into `cb`'s sum plus the base, which the back adds the host's sums
    /// to.
    pub(crate) fn enqueue_join_rows(
        &self,
        gpu: &Gpu,
        (card, l, m): (&CardExperts, usize, usize),
        (cb, tb): (&mut CardBatch, &TierBatch),
        (trows, x1): (&DeviceBuffer<f32>, &DeviceBuffer<f32>),
    ) -> Result<(), GpuError> {
        let slots = tb.slots(self.n, (card.n_card(l), card.tier_k(l), N_USED));
        tier_join::enqueue_join(
            gpu,
            (gpu.stream(), &self.kernels),
            (&mut cb.join(), &slots),
            (trows, m),
            x1,
        )
    }
}
