//! The MiniMax-M3 expert tier ([`bloomery_gpu::host::leg_tier`]): a card beside the stage card that
//! holds some of each routed layer's experts and computes the routed slots the slot map sends it, in
//! the host leg's shadow. The tier's layer is the common
//! [`LegTier`](bloomery_gpu::host::leg_tier::LegTier) over MiniMax's leg shape ([`CHUNK`] rows a
//! run), its NVFP4 stacks in their planes through the leg's NVFP4 gate·up and down
//! arms; the stage card's side is the common
//! [`TierSide`](bloomery_gpu::host::leg_tier::TierSide) and its join the routed block's
//! ([`crate::ffn::Routed`]). What is MiniMax's here is the node family that reads the file's
//! description ([`NODE_FAMILY`]) and the activation an architecture fact fixes: every routed block
//! runs SwiGLU-OAI at the slope and clamp its description gives.

use std::error::Error;
use std::ops::Range;

use bloomery_gpu::GpuError;
use bloomery_gpu::host::leg_tier::{self, LegFamily, LegWidths};
use bloomery_gpu::host::node::NodeFamily;
use bloomery_gpu::kquant::Act;
use gguf::Split;
use model::arch::minimaxm3::host as minimax_host;
use model::arch::minimaxm3::place::{self, PlanInputs};
use model::moe::HostLayer;
use model::placement::ModelTensors;
use model::placement::workstation::TierBatchBytes;
use model::r8file::R8Source;
use models::{LayerSpec, Moe};

use crate::body::{LOAD_ROWS, routed_oai};
use crate::prefill::T_MAX;

/// What the tier's errors name.
const WHAT: &str = "minimaxm3 tier";

/// Tokens one tier run covers: the card leg's capacity, which the plan reserves the leg's buffers
/// for ([`place::TIER_CHUNK`]); a block of a prompt unit's rows runs in runs of at most this many.
pub const CHUNK: usize = place::TIER_CHUNK as usize;

/// The tier's prompt-batch bytes the plan reserves on the tier card and the host for the file
/// `inputs` describes: [`place::tier_batch`] over blocks of a prompt unit's rows ([`T_MAX`]), the
/// columns the load makes the tier's batch legs for.
#[must_use]
pub fn tier_batch(inputs: &PlanInputs) -> TierBatchBytes {
    place::tier_batch(inputs, T_MAX as u64)
}

/// MiniMax's architecture facts for the common tier node service: the file's reading, the rows in
/// flight of a load ([`LOAD_ROWS`]), the tier's chunk and the routed blocks' SwiGLU-OAI.
struct MiniMax;

impl LegFamily for MiniMax {
    const WHAT: &'static str = WHAT;
    const WORD: &'static str = "minimaxm3";
    const ROWS: usize = LOAD_ROWS;
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
        let experts = &inputs.hp.experts;
        LegWidths {
            n_expert: experts.n_expert,
            n_used: experts.n_used,
            n_embd: inputs.hp.n_embd,
            ff: experts.ff,
        }
    }

    /// Each routed layer of `run` as the host tier serves it, at the SwiGLU-OAI slope and clamp its
    /// description gives ([`minimax_host::layer`]); a layer of another activation is refused by
    /// name.
    fn routed_views(
        src: R8Source<'_>,
        inputs: &PlanInputs,
        run: Range<usize>,
    ) -> Result<Vec<HostLayer>, Box<dyn Error + Send + Sync + 'static>> {
        let mut views = Vec::with_capacity(run.len());
        for l in run {
            let (alpha, limit) = routed_oai(&inputs.spec, l)?;
            views.push(minimax_host::layer(src, &inputs.hp, l, alpha, limit)?);
        }
        Ok(views)
    }

    /// The block's SwiGLU-OAI at its slope and clamp, the rule the stage card's card experts run;
    /// any other activation is refused by name.
    fn tier_act(moe: &Moe) -> Result<Act, GpuError> {
        match moe.act {
            models::Act::SwigluOai { alpha, limit } => Ok(Act::SwigluOai { alpha, limit }),
            other => Err(GpuError::shape(
                WHAT,
                format!("a routed activation {other:?}; the tier runs SwiGLU-OAI"),
            )),
        }
    }
}

/// MiniMax's tier node service ([`leg_tier::node_family`]).
pub const NODE_FAMILY: NodeFamily = leg_tier::node_family::<MiniMax>();
