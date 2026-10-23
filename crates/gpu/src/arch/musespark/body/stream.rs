//! A placed load's prompt calls on the common host streaming ([`PromptStream`], the `BLOOMERY_XSTREAM` lever), and
//! the residency machine the stream reads its experts from ([`stacks`]): the body hands the common owners its own
//! facts as data — its stacks' names, its counts ([`STREAM_FLOOR`]), its card map (a layer streams when the host
//! serves a share of it and the card holds its stage stacks) and its expert shape's costs ([`XSTREAM_COSTS`]) — and
//! no stream code of its own.
//!
//! A call opens on the machine inside the pass its boundary opened ([`super::prefill`]); each pass is one walk of one
//! unit. At a hosted layer's shadow the pass picks ([`PromptStream::unit_pick`]): under `split` the host experts the
//! stream's rule sends to the card land in a half of its ring, in landing batches, and the serve leaves them out. The
//! card's route then runs the layer's stage slots, and the ring's slots batch by batch, each batch behind its own
//! landed event, through the same IQ2_XXS `_sel` launches over the half's stacks ([`RingPass`]); the back sums the
//! card's slots by the stream's union places (`n_card + i` for the `i`-th streamed expert), so a streamed expert's
//! row joins the sum where a host expert's would, before the host's sums add.
//!
//! The IQ2_XXS stacks are the file's blocks on the card, so the machine stages each part as the file holds it
//! ([`crate::host::swap::Transform::Identity`]): the stage slots, the ring's slots and the host read the same bytes.

use super::Body;
use crate::card_slots::CardSlotKernels;
use crate::host::prompt_stream::PromptStream;
use crate::host::swap::{CallPick, CallReport};
use crate::host::swap_source::{DEADLINE, PlanStacks, hosted_stacks};
use crate::host::xstream::{Costs, RingLayer, XCfg, XLayer, XMode, XReport, XStreamBody};
use crate::{Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
use model::placement::Plan;

const WHAT: &str = "musespark::stream";

/// The least count a pick admits on a load whose lane never probed: a call of fewer ids opens no streaming. The
/// value V4.1's and Qwen3.8's walks admit from: an admit moves an expert for the rest of the call, so it pays only
/// past the transient stream's own threshold (one expert's copy at the priced lane against [`XSTREAM_COSTS`]'s union
/// columns, about eleven columns [derived]), and the floor sits above that.
pub const STREAM_FLOOR: u32 = 32;

/// The stream rule's costs for the file's IQ2_XXS experts (latent 4,096 by 4,096 a part, three parts): the host
/// union's `t(m) = a + c·m` of `docs/research/muse-spark-attach.md` §5, "Prefill, P = 512" [derived: `a` the h1fold
/// fit, `c` the Q3_K column fit scaled by the expert's 50.33/35.39 MMAC a column and the upper IQ2_XXS unpack penalty,
/// 1.5, the end the load's measured pp512 sits at]; the card's, the `_sel` launches' column (each slot reads its
/// expert's 12.98 MB whole, so a column is one weight read at the card's bandwidth and nothing is fixed per expert
/// past its launch share) [derived, unmeasured]. No record of this file's union has fitted the row: the per-file
/// override ([`Body::set_xstream_costs`]) takes a calibrated one.
pub const XSTREAM_COSTS: Costs = Costs {
    host_us_per_col: 57.6,
    host_us_fixed: 28.1,
    card_us_fixed: 0.0,
    card_us_per_col: 18.5,
};

/// The residency machine's stacks of a placed load of `n_layer` layers: every layer's gate, up and down by the common
/// stack names, the types read from `plan`'s tensors, every layer routed (none holds no stack). Refused by name: a
/// layer missing one of its three stacks. Load-time only.
pub(super) fn stacks(plan: &Plan<'_>, n_layer: usize) -> Result<PlanStacks, GpuError> {
    hosted_stacks(plan.model, "musespark stacks", (n_layer, false))
}

/// The prompt calls' host streaming and the buffers the card's ring pass writes, once a `split` armed them.
#[derive(Default)]
pub(super) struct XSide {
    /// The calls' host streaming: off until [`Body::set_xstream`] sets it.
    pub(super) stream: PromptStream,
    /// The ring pass's places, from the first `split` on.
    ring: Option<RingPlaces>,
    /// Per ring pass of the last streaming call, its walk, its layer and the streamed experts its slots joined into
    /// the card's sum: what a gate holds the stream's records to ([`Body::take_stream_joined`]).
    joined: Vec<(usize, usize, usize)>,
    /// The walk the call runs, from 0: the joined entries' walk, as the records name theirs.
    walk: usize,
    /// The card bytes the ring may take past the stream's keep-free: the ring a streaming plan reserves
    /// ([`stream_ring_bytes`](model::arch::musespark::place::stream_ring_bytes)), so a ring never takes the card's margin;
    /// set at the load.
    ring_room: Option<u64>,
}

/// The ring pass's places, `slots` words each: a landing batch's ring slots of the pass's routed slots, and the
/// union's places the back sums by.
struct RingPlaces {
    sel: DeviceBuffer<u32>,
    union: DeviceBuffer<u32>,
}

/// One hosted layer's ring pass ([`super::prefill`]'s card route): the stream's view of the layer, the places' kernels
/// and the buffers they write.
pub(super) struct RingPass<'a> {
    pub(super) r: RingLayer,
    pub(super) acc_k: &'a CardSlotKernels,
    pub(super) sel: &'a mut DeviceBuffer<u32>,
    pub(super) union: &'a mut DeviceBuffer<u32>,
}

impl XSide {
    /// The ring's room cap, the plan's ring reserve. Load-time only.
    pub(super) fn set_ring_room(&mut self, bytes: u64) {
        self.ring_room = Some(bytes);
    }

    /// The ring pass's places for passes of up to `slots` routed slots. Load-time only (the first `split`); a second
    /// arming is refused by name.
    fn arm(&mut self, stream: &CudaStream, slots: usize) -> Result<(), GpuError> {
        if self.ring.is_some() {
            return Err(GpuError::state(
                WHAT,
                "a stream side not yet armed for the ring",
            ));
        }
        self.ring = Some(RingPlaces {
            sel: DeviceBuffer::zeroed(stream, slots)?,
            union: DeviceBuffer::zeroed(stream, slots)?,
        });
        self.joined.reserve(64);
        Ok(())
    }

    /// A call opens: the joined record and the walk index start over.
    pub(super) fn begin_call(&mut self) {
        self.joined.clear();
        self.walk = 0;
    }

    /// A pass's walk ran ([`PromptStream::walk_done`] beside it).
    pub(super) fn walk_done(&mut self) {
        self.stream.walk_done();
        self.walk += 1;
    }

    /// Layer `layer`'s ring pass over `r` with `acc_k`'s places: the armed buffers lent to it. Refused by name: a
    /// stream with no armed ring.
    pub(super) fn ring_pass<'a>(
        &'a mut self,
        r: RingLayer,
        acc_k: &'a CardSlotKernels,
    ) -> Result<RingPass<'a>, GpuError> {
        let p = self.ring.as_mut().ok_or(GpuError::state(
            WHAT,
            "the ring pass's places (the first split arms them)",
        ))?;
        Ok(RingPass {
            r,
            acc_k,
            sel: &mut p.sel,
            union: &mut p.union,
        })
    }

    /// The union's places the last ring pass wrote, which its layer's back sums the card's slots by. Refused by
    /// name: a stream with no armed ring.
    pub(super) fn union_places(&self) -> Result<&DeviceBuffer<u32>, GpuError> {
        self.ring.as_ref().map(|p| &p.union).ok_or(GpuError::state(
            WHAT,
            "the ring pass's places (the first split arms them)",
        ))
    }

    /// The ring pass of layer `layer` joined `n` streamed experts into the card's sum, in the current walk.
    pub(super) fn joined(&mut self, layer: usize, n: usize) {
        self.joined.push((self.walk, layer, n));
    }
}

impl Body {
    /// The stream rule's costs in place of [`XSTREAM_COSTS`] ([`PromptStream::set_costs`]). Refused by name: a load
    /// with no host tier, before `split` started the stream, and while a call streams.
    pub fn set_xstream_costs(&mut self, costs: Costs) -> Result<(), GpuError> {
        let p = self.placed.as_mut().ok_or(GpuError::state(
            "musespark::Body::set_xstream_costs",
            "a placed load (the host tier the stream serves)",
        ))?;
        self.xs.stream.set_costs(&mut p.hybrid, costs)
    }

    /// The last streaming prompt call's streamed layers, each with its walk, and the stream's end; empty and `None`
    /// after a call that did not stream. Taken ([`PromptStream::take_xstream_records`]).
    pub fn take_xstream_records(&mut self) -> (Vec<(usize, XLayer)>, Option<XReport>) {
        self.xs.stream.take_xstream_records()
    }

    /// The last streaming prompt call's picks, each with its walk, and its end; empty and `None` after a call that
    /// did not stream. Taken ([`PromptStream::take_pick_records`]).
    pub fn take_stream_records(&mut self) -> (Vec<(usize, CallPick)>, Option<CallReport>) {
        self.xs.stream.take_pick_records()
    }

    /// The last streaming call's ring passes: each one's walk, layer and the streamed experts its slots joined into
    /// the card's sum. Taken: a second read is empty. A gate's instrument.
    pub fn take_stream_joined(&mut self) -> Vec<(usize, usize, usize)> {
        std::mem::take(&mut self.xs.joined)
    }

    /// Open the prompt call's streaming over `n` ids ending at `end`, returning to its start when `returns`
    /// ([`PromptStream::begin`], the walks being every pass of a placed load); nothing on a whole-card load or while
    /// the lever is off.
    pub(super) fn stream_begin(
        &mut self,
        stream: &CudaStream,
        n: usize,
        end: usize,
        returns: bool,
    ) -> Result<(), GpuError> {
        let layers = self.layers.len();
        let Some(p) = self.placed.as_mut() else {
            return Ok(());
        };
        self.xs.begin_call();
        self.xs.stream.begin(
            &mut p.hybrid,
            stream,
            true,
            n,
            end,
            returns,
            STREAM_FLOOR,
            0.0,
            layers,
        )
    }

    /// End the prompt call's streaming ([`PromptStream::end`]), its placement `kept` for the decode after it; nothing
    /// on a whole-card load.
    pub(super) fn stream_end(&mut self, stream: &CudaStream, kept: bool) -> Result<(), GpuError> {
        match self.placed.as_mut() {
            Some(p) => self.xs.stream.end(&mut p.hybrid, stream, kept),
            None => Ok(()),
        }
    }

    /// Whether the residency machine's rule holds a decode history ([`crate::host::HostTier::call_history`]): a
    /// prompt call then returns its placement at its end. False on a whole-card load.
    pub(super) fn call_history(&self) -> bool {
        self.placed
            .as_ref()
            .is_some_and(|p| p.hybrid.call_history())
    }
}

impl XStreamBody for Body {
    /// What the next prompt calls move (`BLOOMERY_XSTREAM`): `off` until set here. `Split` starts the expert stream
    /// at its first setting ([`PromptStream::start_split`]: the ring sized from the card's free bytes past the
    /// stream's keep-free, the lane's probe, [`XSTREAM_COSTS`] for the rule) and arms the ring pass's places; the
    /// stream then stays for the load's life, idle under `admit` and `off`. Refused by name: `admit` or `split` on a
    /// whole-card load or one with no residency machine, `split` on a load with an expert tier card or when the
    /// stream cannot start (the card's room among the refusals), and any change while a call streams.
    fn set_xstream(&mut self, gpu: &Gpu, mode: XMode) -> Result<(), GpuError> {
        const WHAT_S: &str = "musespark::Body::set_xstream";
        let (experts, top_k, rows) = (self.sh.n_expert, self.sh.top_k, self.prompt.rows());
        let room = self.xs.ring_room;
        let Some(p) = self.placed.as_mut() else {
            if mode == XMode::Off {
                self.xs.stream.set_mode(mode);
                return Ok(());
            }
            return Err(GpuError::state(
                WHAT_S,
                "a placed load (the host tier the stream serves)",
            ));
        };
        self.xs
            .stream
            .check_mode(&p.hybrid, (mode, p.side.tier.is_some()))?;
        if mode == XMode::Split
            && PromptStream::start_split(&mut p.hybrid, gpu, || {
                XCfg::new(XSTREAM_COSTS, experts, top_k, DEADLINE, room)
            })?
            .is_some()
        {
            self.xs.arm(gpu.stream(), rows * top_k)?;
        }
        self.xs.stream.set_mode(mode);
        Ok(())
    }

    /// The load line's `xstream=` word ([`PromptStream::word`]); the mode alone on a whole-card load.
    fn xstream_word(&self) -> String {
        match &self.placed {
            Some(p) => self.xs.stream.word(&p.hybrid),
            None => self.xs.stream.mode().word().to_string(),
        }
    }
}
