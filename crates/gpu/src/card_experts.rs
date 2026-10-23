//! The routed block's card experts, common to every program whose card runs
//! the experts the slot map puts on it by the kernel table: the experts of a
//! layer computed by the common card leg ([`crate::card_leg`]: the per-slot
//! pair of the kernel table, launched by the entries the table names for a
//! stack's type) in the host leg's shadow, and summed here.
//!
//! One run is six launches, over `m` tokens of a layer's normed rows:
//! - the card leg's four (`card_leg`): the normed rows in q8_1, the gate·up
//!   `_sel` with the routed SwiGLU (slot `s` against column `s / n_used`), the
//!   card slots' `h` columns in the form the layer's down reads (q8_1 blocks of
//!   128 or q8 blocks of 32; the host's slots are left alone) and the down
//!   `_sel`;
//! - the card slots' weighted sum, a token's slots in slot order (the leg's
//!   sum stage, `card_slots`);
//! - the sum plus the base the layer's back adds the host's sum to
//!   ([`Io::x1`]): the residual after the attention, with a shared expert's
//!   output summed into it by the caller where the family has one.
//!
//! The one-token step is the run at `m = 1`, its places written by the
//! handoff; a prompt batch is the run at `m = t`, one launch each over the
//! batch's `t · n_used` slots, its places written by one launch
//! ([`CardSlotKernels::enqueue_places`]). Each slot's rows are what the step's
//! launches write for it, whatever `m`, so the two feeds leave the same bits.
//!
//! What a family fixes is a [`CardFamily`]: the slot map's row width, the
//! slots a token picks, the stacks' names and what the errors name. A load
//! whose map names an expert tier card builds them with [`CardExperts::tiered`]:
//! a layer the tier holds experts of runs the card leg without its sum, the
//! stage card's slots' downs stay in the leg's rows ([`TierJoin`]) and the
//! layer's join sums them with the tier's rows in one slot order. [`CardExperts::new`]
//! refuses a map that names a tier by name: those experts hang none.
//!
//! Under the expert stream ([`CardExperts::enqueue_rows_ring`]) the streamed
//! experts' slots of a prompt batch run as route GEMMs over the ring's half
//! (`card_leg::RouteBlock`, the 32-value GEMM over the half's planes), one
//! GEMM batch a landing batch of the stream behind that batch's copies, each
//! expert's weights read once a tile of up to 64 of its slots; the card's own
//! slots stay on the leg's `_sel` launches, whose rows are the step's.

use std::mem::ManuallyDrop;

use cuda_core::{CudaStream, DeviceBuffer};
use model::arch::models::{Ffn, LayerSpec};

use model::arch::place_hybrid::ring_pass_bytes;
use model::placement::kernels::{self, DownAct, DownForms};

use crate::card_leg::{
    self, CardLeg, LegBufs, LegRun, LegShape, LegStacks, LegSum, RouteBlock, RouteIo, RouteW,
};
use crate::card_slots::{CardSlotKernels, Places};
use crate::gemm::{Gemm32Kernels, GemmAct32, GemmRoute};
use crate::host::xstream::RingLayer;
use crate::hybrid::SlotMap;
use crate::kquant::Act;
use crate::weights::Weights;
use crate::{DeviceTensor, Gpu, GpuError};

/// What a family fixes of its card experts.
#[derive(Clone, Copy)]
pub struct CardFamily {
    /// What the card experts' errors name.
    pub what: &'static str,
    /// The routed experts of a layer: the slot map's row width.
    pub n_expert: usize,
    /// The routed experts a token picks.
    pub n_used: usize,
    /// Layer `l`'s gate, up and down stacks' names.
    pub stacks: fn(usize) -> [String; 3],
}

/// The kernels a run launches: the card leg's (its sum stage included), and
/// the places' of a prompt batch.
struct CardKernels {
    leg: CardLeg,
    places: CardSlotKernels,
}

impl CardFamily {
    /// The card leg's shape for runs of up to `cap` tokens of `n` values
    /// through experts of `ff` rows.
    fn shape(&self, n: usize, ff: usize, cap: usize) -> LegShape {
        LegShape {
            cap,
            slots_per_col: self.n_used,
            n,
            ff,
        }
    }
}

/// One run's buffers for up to `cap` tokens: the card leg's, the card sum and
/// the sum plus the base.
struct CardBufs {
    leg: LegBufs,
    acc: DeviceBuffer<f32>,
    pre: DeviceBuffer<f32>,
}

impl CardBufs {
    /// The buffers for runs of up to `cap` tokens of `n` values through
    /// experts of `ff` rows, the slots' columns held in each form of `downs`.
    /// Load-time or first-prompt only.
    fn new(
        stream: &CudaStream,
        family: &CardFamily,
        (n, ff, cap): (usize, usize, usize),
        downs: DownForms,
    ) -> Result<CardBufs, GpuError> {
        Ok(CardBufs {
            leg: LegBufs::for_downs(stream, family.shape(n, ff, cap), downs)?,
            acc: DeviceBuffer::zeroed(stream, cap * n)?,
            pre: DeviceBuffer::zeroed(stream, cap * n)?,
        })
    }

    /// Tokens the buffers hold.
    fn cap(&self) -> usize {
        self.leg.shape().cap
    }

    /// The buffers a tier layer's join reads and writes, the slots' places
    /// `sel` as the run read them.
    fn join<'a>(&'a mut self, sel: &'a DeviceBuffer<u32>) -> TierJoin<'a> {
        TierJoin {
            down: self.leg.down(),
            sel,
            acc: &mut self.acc,
            pre: &mut self.pre,
        }
    }

    /// Device bytes of the buffers.
    fn bytes(&self) -> usize {
        self.leg.bytes() + self.acc.num_bytes() + self.pre.num_bytes()
    }
}

/// A prompt batch's card buffers: one run's buffers for the batch's tokens
/// and the places its slots resolve to.
pub struct CardBatch {
    bufs: CardBufs,
    sel: DeviceBuffer<u32>,
    /// The ring pass's GEMM buffers, once the expert stream armed them
    /// ([`CardExperts::arm_ring`]).
    ring: Option<RingGemm>,
}

/// The ring pass's GEMM buffers of a prompt batch of up to `cap` tokens
/// ([`CardExperts::arm_ring`]): the batch's normed rows in the 32-value
/// GEMM's form, the gate and up rows of every slot, their SwiGLU in the
/// down's form and the route table over a half of the expert stream's ring.
/// Their bytes are `place_hybrid::ring_pass_bytes` of the same widths, the
/// one formula the plan reserves them by.
struct RingGemm {
    x: GemmAct32,
    g: DeviceBuffer<f32>,
    u: DeviceBuffer<f32>,
    h: GemmAct32,
    route: GemmRoute,
}

impl RingGemm {
    fn bytes(&self) -> usize {
        self.x.bytes()
            + self.g.num_bytes()
            + self.u.num_bytes()
            + self.h.bytes()
            + self.route.bytes()
    }
}

impl CardBatch {
    /// Device bytes of the buffers: what [`CardExperts::batch`] allocates,
    /// counted before it does by [`CardExperts::batch_bytes`], and the ring
    /// pass's once armed.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bufs.bytes() + self.sel.num_bytes() + self.ring.as_ref().map_or(0, RingGemm::bytes)
    }

    /// The sum plus the base of the last run over the batch, `t` rows.
    #[must_use]
    pub fn pre(&self) -> &DeviceBuffer<f32> {
        &self.bufs.pre
    }

    /// The buffers a tier layer's join reads and writes over the batch.
    pub fn join(&mut self) -> TierJoin<'_> {
        self.bufs.join(&self.sel)
    }
}

/// A tier layer's join over one run's buffers: the stage card's slots' down
/// rows (slot-major, `n` values a slot, left by the run that did not sum
/// them) and each slot's place in the card's stacks, the card sum the join
/// writes and the sum plus the base it then writes
/// ([`CardExperts::step_join`], [`CardBatch::join`]).
pub struct TierJoin<'a> {
    pub down: &'a DeviceBuffer<f32>,
    pub sel: &'a DeviceBuffer<u32>,
    pub acc: &'a mut DeviceBuffer<f32>,
    pub pre: &'a mut DeviceBuffer<f32>,
}

/// What [`CardExperts::enqueue_step`], [`CardExperts::enqueue_pass`] and
/// [`CardExperts::enqueue_rows`] read: `m` tokens' normed rows, their slots'
/// routing weights, and the base the card sum is added to.
pub struct Io<'a> {
    pub normed: &'a DeviceBuffer<f32>,
    pub weights: &'a DeviceBuffer<f32>,
    /// The residual after the attention, with a shared expert's output
    /// summed into it by the caller where the family has one.
    pub x1: &'a DeviceBuffer<f32>,
    pub m: usize,
}

/// A layer's card experts: how many its stacks hold, and the stacks' names
/// (gate, up, down), made once at load.
struct CardLayer {
    n_card: usize,
    names: [String; 3],
}

impl CardLayer {
    /// The stacks' names, as the card leg reads them.
    fn names(&self) -> [&str; 3] {
        self.names.each_ref().map(String::as_str)
    }
}

/// The form the down stack of `layer`, of type `ty`, reads ([`kernels::down`]);
/// a type the table names no down entry for is refused by name (`what`).
fn down_act(
    what: &'static str,
    layer: &CardLayer,
    ty: gguf::GgmlType,
) -> Result<DownAct, GpuError> {
    kernels::down(ty)
        .map(|(_, act)| act)
        .ok_or_else(|| GpuError::Tensor {
            what,
            name: layer.names[2].clone(),
            need: "a down stack of a type the kernel table names a down entry for",
        })
}

/// The forms a load's leg buffers hold: the forms `downs` the card layers'
/// downs read, or the 128-value [`DownAct::Q8Act`] on a load with no card
/// layer (a tier card's join reads only the buffers' down rows).
fn card_downs(downs: DownForms) -> DownForms {
    if downs == DownForms::NONE {
        DownForms::of(DownAct::Q8Act)
    } else {
        downs
    }
}

/// Layer `l`'s card experts of `layers`, refused by name (`what`) on a layer
/// with none.
fn layer_of<'a>(
    what: &'static str,
    layers: &'a [Option<CardLayer>],
    l: usize,
) -> Result<&'a CardLayer, GpuError> {
    layers
        .get(l)
        .and_then(Option::as_ref)
        .ok_or_else(|| GpuError::Shape {
            what,
            detail: format!("layer {l}: no card experts to run"),
        })
}

/// The card's routed experts of the step: the kernels, each layer's count
/// from the slot map, and the step's buffers. Made at load, after the
/// weights, from the stacks they hold; a layer the map puts experts on holds
/// its three stacks in a type the card leg runs, at `n_card` experts each, and
/// a layer it puts none on holds none — anything else is refused by name.
pub struct CardExperts {
    family: CardFamily,
    k: CardKernels,
    /// Per layer of the description, its card experts; `None` without.
    layers: Vec<Option<CardLayer>>,
    /// Per layer of the description, the experts the expert tier holds: 0
    /// off the tier, and on every layer of a load with no tier.
    tier: Vec<usize>,
    /// Routed expert width: a gate·up slot's rows.
    ff: usize,
    /// The model's width.
    n: usize,
    /// The forms the card layers' downs read, which every run's buffers hold
    /// ([`card_downs`]).
    downs: DownForms,
    /// The one-token step's buffers.
    step: CardBufs,
    /// The buffers of a pass of several rows, on a load that runs one.
    pass: Option<CardBufs>,
}

impl CardExperts {
    /// The card experts of `layers` (the description) as `map` places them and
    /// `w` holds them, `n` wide with routed experts `ff` wide, with buffers
    /// for a pass of `rows` rows when `rows` is more than one. A map that names
    /// an expert tier is refused by name ([`CardExperts::tiered`]). Load-time
    /// only.
    pub fn new(
        gpu: &Gpu,
        w: &Weights,
        family: CardFamily,
        layers: &[LayerSpec],
        map: &SlotMap,
        dims: (usize, usize),
        rows: usize,
    ) -> Result<CardExperts, GpuError> {
        Self::build(gpu, w, family, layers, map, dims, rows, false)
    }

    /// [`CardExperts::new`] for a load that hangs the map's expert tier card:
    /// a layer the tier holds experts of runs as a tier layer (module doc).
    pub fn tiered(
        gpu: &Gpu,
        w: &Weights,
        family: CardFamily,
        layers: &[LayerSpec],
        map: &SlotMap,
        dims: (usize, usize),
        rows: usize,
    ) -> Result<CardExperts, GpuError> {
        Self::build(gpu, w, family, layers, map, dims, rows, true)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the constructors' arguments and whether a tier hangs (rust-quality R8)"
    )]
    fn build(
        gpu: &Gpu,
        w: &Weights,
        family: CardFamily,
        layers: &[LayerSpec],
        map: &SlotMap,
        (n, ff): (usize, usize),
        rows: usize,
        tiers: bool,
    ) -> Result<CardExperts, GpuError> {
        let what = family.what;
        let mut per = Vec::with_capacity(layers.len());
        let mut tier_k = Vec::with_capacity(layers.len());
        let mut downs = DownForms::NONE;
        for (l, spec) in layers.iter().enumerate() {
            // A dense block is outside the host run and has no row: no card
            // experts, which the resident-rows check below holds it to. A
            // routed layer the map has no row for is refused by name.
            let (n_card, on_tier) = match spec.ffn {
                Ffn::Dense { .. } => (0, 0),
                Ffn::Moe(_) => {
                    let (card, tier) = (map.on_card(l)?, map.on_tier(l)?);
                    if tier > 0 && !tiers {
                        return Err(GpuError::Shape {
                            what,
                            detail: format!(
                                "layer {l}: the plan puts {tier} routed experts on an expert \
                                 tier; these card experts hang no tier"
                            ),
                        });
                    }
                    (card, tier)
                }
            };
            tier_k.push(on_tier);
            let layer = CardLayer {
                n_card,
                names: (family.stacks)(l),
            };
            if n_card == 0 {
                if let Some(name) = layer.names.iter().find(|name| w.get(name).is_some()) {
                    return Err(GpuError::Tensor {
                        what,
                        name: name.clone(),
                        need: "no resident rows: the slot map puts no expert of its layer on the \
                               card",
                    });
                }
                per.push(None);
                continue;
            }
            let stacks = LegStacks::of(w, layer.names())?;
            stacks.check(n_card, family.shape(n, ff, 1))?;
            downs = downs.with(down_act(what, &layer, stacks.down_ty())?);
            per.push(Some(layer));
        }
        let downs = card_downs(downs);
        Ok(CardExperts {
            family,
            k: CardKernels {
                leg: CardLeg::load(gpu)?,
                places: CardSlotKernels::load(gpu.context())?,
            },
            layers: per,
            tier: tier_k,
            ff,
            n,
            downs,
            step: CardBufs::new(gpu.stream(), &family, (n, ff, 1), downs)?,
            pass: (rows > 1)
                .then(|| CardBufs::new(gpu.stream(), &family, (n, ff, rows), downs))
                .transpose()?,
        })
    }

    /// Layer `l` computes experts on the card.
    #[must_use]
    pub fn has(&self, l: usize) -> bool {
        layer_of(self.family.what, &self.layers, l).is_ok()
    }

    /// Some layer computes experts on the card.
    #[must_use]
    pub fn any(&self) -> bool {
        self.layers.iter().any(Option::is_some)
    }

    /// The experts of layer `l` the expert tier holds: 0 off the tier, and on
    /// every layer of a load with none.
    #[must_use]
    pub fn tier_k(&self, l: usize) -> usize {
        self.tier.get(l).copied().unwrap_or(0)
    }

    /// Layer `l` has a card sum: card experts on the stage card or the tier.
    #[must_use]
    pub fn sums(&self, l: usize) -> bool {
        self.has(l) || self.tier_k(l) > 0
    }

    /// Some layer holds experts on the expert tier.
    #[must_use]
    pub fn any_tier(&self) -> bool {
        self.tier.iter().any(|&k| k > 0)
    }

    /// The stage card's experts of layer `l`: 0 off the card.
    #[must_use]
    pub fn n_card(&self, l: usize) -> usize {
        layer_of(self.family.what, &self.layers, l).map_or(0, |c| c.n_card)
    }

    /// The layers that compute experts on the card.
    #[must_use]
    pub fn card_layers(&self) -> Vec<bool> {
        self.layers.iter().map(Option::is_some).collect()
    }

    /// The routed experts' width: a gate·up slot's rows.
    #[must_use]
    pub fn ff(&self) -> usize {
        self.ff
    }

    /// The step's sum plus the base of the last card layer enqueued.
    #[must_use]
    pub fn pre(&self) -> &DeviceBuffer<f32> {
        &self.step.pre
    }

    /// The buffers a tier layer's join reads and writes over the step's row,
    /// the slots' places `sel` as the handoff wrote them.
    pub fn step_join<'a>(&'a mut self, sel: &'a DeviceBuffer<u32>) -> TierJoin<'a> {
        self.step.join(sel)
    }

    /// The sum plus the base of the last card layer a pass enqueued, one row
    /// a token of the pass.
    pub fn pass_pre(&self) -> Result<&DeviceBuffer<f32>, GpuError> {
        self.pass.as_ref().map(|p| &p.pre).ok_or(GpuError::State {
            what: self.family.what,
            missing: "the buffers of a pass of several rows",
        })
    }

    /// Device bytes of the step's buffers and a pass's.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.step.bytes() + self.pass.as_ref().map_or(0, CardBufs::bytes)
    }

    /// The buffers for prompt batches of up to `cap` tokens. Load-time or
    /// first-prompt only.
    pub fn batch(&self, gpu: &Gpu, cap: usize) -> Result<CardBatch, GpuError> {
        let stream = gpu.stream();
        Ok(CardBatch {
            bufs: CardBufs::new(stream, &self.family, (self.n, self.ff, cap), self.downs)?,
            sel: DeviceBuffer::zeroed(stream, cap * self.family.n_used)?,
            ring: None,
        })
    }

    /// The ring pass's GEMM buffers in `cb` for a half of `half_slots` slots
    /// of the expert stream's ring (`RingGemm`): made once, when the
    /// stream starts; a second arming at the same half is nothing. Refused by
    /// name: a re-arming at another half, and buffers whose made bytes are
    /// not `place_hybrid::ring_pass_bytes` of their widths. Load-time or
    /// first-prompt only.
    pub fn arm_ring(
        &self,
        gpu: &Gpu,
        cb: &mut CardBatch,
        half_slots: usize,
    ) -> Result<(), GpuError> {
        let what = self.family.what;
        if let Some(r) = &cb.ring {
            if r.route.n_experts() == half_slots {
                return Ok(());
            }
            return Err(GpuError::shape(
                what,
                format!(
                    "the ring pass's buffers armed for a half of {} slots, asked for {half_slots}",
                    r.route.n_experts()
                ),
            ));
        }
        let stream = gpu.stream();
        let cap = cb.bufs.cap();
        let slots = cap * self.family.n_used;
        let ring = RingGemm {
            x: GemmAct32::new(stream, cap, self.n)?,
            g: DeviceBuffer::zeroed(stream, slots * self.ff)?,
            u: DeviceBuffer::zeroed(stream, slots * self.ff)?,
            h: GemmAct32::new(stream, slots, self.ff)?,
            route: GemmRoute::new(stream, slots, half_slots)?,
        };
        let want = ring_pass_bytes(cap, self.family.n_used, self.n, self.ff, half_slots);
        if ring.bytes() != want {
            return Err(GpuError::shape(
                what,
                format!(
                    "the ring pass's buffers made {} bytes; the plan reserves {want} \
                     (place_hybrid::ring_pass_bytes)",
                    ring.bytes()
                ),
            ));
        }
        cb.ring = Some(ring);
        Ok(())
    }

    /// Device bytes of [`CardExperts::batch`]'s buffers for `cap` tokens
    /// ([`CardBatch::bytes`]'s formula): the card leg's in the card layers'
    /// down forms ([`kernels::leg_bytes`]), the card sum and the sum plus the
    /// base, and the places.
    #[must_use]
    pub fn batch_bytes(&self, cap: usize) -> usize {
        let shape = self.family.shape(self.n, self.ff, cap);
        let leg = kernels::leg_bytes(
            shape.cap as u64,
            shape.slots_per_col as u64,
            shape.n as u64,
            shape.ff as u64,
            self.downs,
        );
        usize::try_from(leg).expect("the leg's bytes fit a usize")
            + 2 * size_of::<f32>() * cap * self.n
            + size_of::<u32>() * cap * self.family.n_used
    }

    /// Layer `l`'s card slots of the step (module doc, `m = 1`), the slots'
    /// places `sel` as the handoff wrote them, into [`CardExperts::pre`].
    pub fn enqueue_step(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        l: usize,
        rule: Act,
        io: &Io<'_>,
        sel: &DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let layer = layer_of(self.family.what, &self.layers, l)?;
        let tiered = self.tier_k(l) > 0;
        self.k.enqueue(
            gpu,
            w,
            &mut self.step,
            (l, layer, self.n),
            (rule, tiered),
            io,
            sel,
        )
    }

    /// Layer `l`'s card slots of a pass of `io.m` rows (module doc), the
    /// slots' places `sel` as the handoffs wrote them, column after column,
    /// into [`CardExperts::pass_pre`]. A pass of more rows than the load's
    /// buffers hold is refused by name.
    pub fn enqueue_pass(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        l: usize,
        rule: Act,
        io: &Io<'_>,
        sel: &DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let what = self.family.what;
        let layer = layer_of(what, &self.layers, l)?;
        let bufs = self.pass.as_mut().ok_or(GpuError::State {
            what,
            missing: "the buffers of a pass of several rows",
        })?;
        if io.m == 0 || io.m > bufs.cap() {
            return Err(GpuError::Shape {
                what,
                detail: format!("a pass of {} rows in buffers of {}", io.m, bufs.cap()),
            });
        }
        let tiered = self.tier.get(l).is_some_and(|&k| k > 0);
        self.k
            .enqueue(gpu, w, bufs, (l, layer, self.n), (rule, tiered), io, sel)
    }

    /// Layer `l`'s card slots over a prompt batch of `io.m` tokens (module
    /// doc): every slot's place from the slot map `map` in one launch, `ids`
    /// the router's picks, then the run, into [`CardBatch::pre`]. The batch's
    /// buffers hold at least `io.m` tokens.
    #[allow(
        clippy::too_many_arguments,
        reason = "one batch's inputs: the layer and its rule, the batch's rows and routing, the \
                  slot map and its row"
    )]
    pub fn enqueue_rows(
        &self,
        gpu: &Gpu,
        w: &Weights,
        cb: &mut CardBatch,
        l: usize,
        rule: Act,
        io: &Io<'_>,
        (ids, map, row_off): (&DeviceBuffer<u32>, &DeviceTensor<u32>, usize),
    ) -> Result<(), GpuError> {
        let layer = layer_of(self.family.what, &self.layers, l)?;
        if io.m == 0 || io.m > cb.bufs.cap() {
            return Err(GpuError::Shape {
                what: self.family.what,
                detail: format!("a batch of {} tokens in buffers of {}", io.m, cb.bufs.cap()),
            });
        }
        self.k.places.enqueue_places(
            gpu.stream(),
            &Places {
                ids,
                n: io.m * self.family.n_used,
                map: map.buf(),
                row_off,
                n_expert: self.family.n_expert,
            },
            gpu.layer_sink(l)?,
            &mut cb.sel,
        )?;
        let tiered = self.tier.get(l).is_some_and(|&k| k > 0);
        self.k.enqueue(
            gpu,
            w,
            &mut cb.bufs,
            (l, layer, self.n),
            (rule, tiered),
            io,
            &cb.sel,
        )
    }

    /// [`CardExperts::enqueue_rows`] with the expert stream's ring `r` of
    /// layer `l` beside the card's stacks (a walk under `split`): the card's
    /// slots run over its stacks under the slot map's places (the leg's
    /// `_sel` launches); then the streamed experts' slots, one GEMM batch a
    /// landing batch of `r` ([`RingLayer::batches`]): the route table of the
    /// batch's map row (its experts at their ring slots, every other
    /// [`crate::hybrid::HOST`]), the engine stream's wait on the batch's
    /// copies, and the gate, up, SwiGLU under `rule` and down GEMMs over the
    /// half's stacks ([`card_leg::with_ring`], `RouteBlock`), each slot's down
    /// rows into the leg's rows beside the card's; then the card sum over the
    /// union row's places — the layer's `n_card` and the `r.n` streamed
    /// experts, a token's slots in slot order — plus the base into `cb`'s
    /// `pre`. A batch's GEMMs run while the copies of the batches after it
    /// land. The places share `cb`'s, each written after the run before it on
    /// the one stream. Refused by name: a layer an expert tier holds experts
    /// of (the ring serves the stage card of a load with no tier), a half
    /// holding more experts than its slots, a batch whose ring pass buffers
    /// were not armed ([`CardExperts::arm_ring`]) or were armed for another
    /// half, and ring stacks the 32-value GEMM does not read.
    pub fn enqueue_rows_ring(
        &self,
        (gpu, w, g32): (&Gpu, &Weights, &Gemm32Kernels),
        cb: &mut CardBatch,
        (l, rule): (usize, Act),
        io: &Io<'_>,
        (ids, map, row_off): (&DeviceBuffer<u32>, &DeviceTensor<u32>, usize),
        r: &RingLayer,
    ) -> Result<(), GpuError> {
        let what = self.family.what;
        let layer = layer_of(what, &self.layers, l)?;
        if io.m == 0 || io.m > cb.bufs.cap() {
            return Err(GpuError::Shape {
                what,
                detail: format!("a batch of {} tokens in buffers of {}", io.m, cb.bufs.cap()),
            });
        }
        if self.tier.get(l).is_some_and(|&k| k > 0) {
            return Err(GpuError::shape(
                what,
                format!(
                    "layer {l}: the expert stream's ring on a layer the expert tier holds experts of"
                ),
            ));
        }
        if r.n > r.half_slots {
            return Err(GpuError::shape(
                what,
                format!(
                    "layer {l}: a ring of {} experts in a half of {} slots",
                    r.n, r.half_slots
                ),
            ));
        }
        let CardBatch { bufs, sel, ring } = cb;
        let rg = ring.as_mut().ok_or(GpuError::State {
            what,
            missing: "the ring pass's GEMM buffers (CardExperts::arm_ring at the stream's start)",
        })?;
        if rg.route.n_experts() != r.half_slots {
            return Err(GpuError::shape(
                what,
                format!(
                    "layer {l}: the ring pass's buffers armed for a half of {} slots, the ring's \
                     is {}",
                    rg.route.n_experts(),
                    r.half_slots
                ),
            ));
        }
        let (stream, ctx) = (gpu.stream(), gpu.context());
        let fault = gpu.layer_sink(l)?;
        let (n_expert, n_used) = (self.family.n_expert, self.family.n_used);
        let n_slots = io.m * n_used;
        let places = |map: &DeviceBuffer<u32>, row_off: usize, sel: &mut DeviceBuffer<u32>| {
            self.k.places.enqueue_places(
                stream,
                &Places {
                    ids,
                    n: n_slots,
                    map,
                    row_off,
                    n_expert,
                },
                fault,
                sel,
            )
        };
        let stacks = LegStacks::of(w, layer.names())?;
        places(map.buf(), row_off, &mut *sel)?;
        let card = LegRun {
            layer: l,
            stacks: &stacks,
            normed: io.normed,
            m: io.m,
            sel: &*sel,
            n_card: layer.n_card,
            rule,
        };
        self.k.leg.enqueue(gpu, &mut bufs.leg, &card, None)?;
        g32.enqueue_quantize_gemm32(stream, io.normed, io.m, &mut rg.x, fault)?;
        card_leg::with_ring(&stacks, layer.n_card, r, ctx, |half| {
            let [gate, up, down] = half.routes()?;
            if [gate, up, down]
                .iter()
                .any(|w| matches!(w, RouteW::Grouped(..)))
            {
                return Err(GpuError::shape(
                    what,
                    format!(
                        "layer {l}: ring stacks the grouped GEMM reads; the ring pass runs the \
                         32-value GEMM alone"
                    ),
                ));
            }
            let blk = RouteBlock {
                gate,
                up,
                down,
                ff: self.ff,
                hidden: self.n,
            };
            for b in &r.batches {
                b.with_map(ctx, n_expert, |row| {
                    g32.enqueue_route_remap(stream, ids, row, n_slots, &mut rg.route, fault)
                })?;
                if let Some(e) = b.event.as_deref() {
                    stream.wait(e)?;
                }
                let RingGemm { x, g, u, h, route } = &mut *rg;
                blk.enqueue_rule(
                    gpu,
                    (None, g32),
                    (&*route, n_used),
                    n_slots,
                    (
                        RouteIo {
                            x128: None,
                            x32: Some(&*x),
                            g,
                            u,
                            h128: None,
                            h32: Some(h),
                            y: bufs.leg.down_mut(),
                        },
                        rule,
                    ),
                    fault,
                )?;
            }
            Ok(())
        })?;
        // SAFETY: the union row is `n_expert` words of the stream's mapped
        // rows (`XStream::ring_layer`), allocated while the stream lives, past
        // these launches; the window is given back below.
        let union_map = unsafe { crate::tensor::window::<u32>(r.union_map, n_expert, ctx) };
        let ran = (|| {
            places(&union_map, 0, &mut *sel)?;
            let CardBufs { leg, acc, pre } = &mut *bufs;
            let sum = LegSum {
                w: io.weights,
                acc: &mut *acc,
            };
            let n_card = layer.n_card + r.n;
            self.k
                .leg
                .sum_rows(gpu, leg, &*sel, (l, io.m, n_card), sum)?;
            gpu.elem()
                .enqueue_add(stream, acc, io.x1, io.m * self.n, pre)
        })();
        drop(ManuallyDrop::into_inner(union_map).into_raw_parts());
        ran
    }
}

impl CardKernels {
    /// The six launches (module doc) of layer `l` over `io.m` tokens, `layer`
    /// its card experts and `n` the model's width. On a `tiered` layer the
    /// card leg's four and no more: the slots' downs stay in the buffers for
    /// the layer's join, which sums them with the tier's rows.
    #[allow(
        clippy::too_many_arguments,
        reason = "one run's inputs: the weights, the buffers, the layer's shape and rule, the \
                  rows and their places"
    )]
    fn enqueue(
        &self,
        gpu: &Gpu,
        w: &Weights,
        bufs: &mut CardBufs,
        (l, layer, n): (usize, &CardLayer, usize),
        (rule, tiered): (Act, bool),
        io: &Io<'_>,
        sel: &DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let stacks = LegStacks::of(w, layer.names())?;
        let run = LegRun {
            layer: l,
            stacks: &stacks,
            normed: io.normed,
            m: io.m,
            sel,
            n_card: layer.n_card,
            rule,
        };
        let sum = (!tiered).then_some(LegSum {
            w: io.weights,
            acc: &mut bufs.acc,
        });
        self.leg.enqueue(gpu, &mut bufs.leg, &run, sum)?;
        if tiered {
            return Ok(());
        }
        let stream = gpu.stream();
        gpu.elem()
            .enqueue_add(stream, &bufs.acc, io.x1, io.m * n, &mut bufs.pre)
    }
}

#[cfg(test)]
mod tests {
    use super::{DownAct, DownForms, card_downs};

    /// A load's leg buffers hold the forms its card layers' downs read — the
    /// 32-value form for NVFP4's, both for a mix — and the 128-value form on a
    /// load with no card layer. Mutant: the buffers back on the 128-value form
    /// alone (an NVFP4 layer's down finds no q8 blocks of 32 and is refused at
    /// its first run).
    #[test]
    fn the_leg_buffers_hold_the_card_downs_forms() {
        let q8 = DownForms::of(DownAct::Q8Act);
        let b32 = DownForms::of(DownAct::Q8Blocks32);
        assert_eq!(card_downs(DownForms::NONE), q8);
        assert_eq!(card_downs(q8), q8);
        assert_eq!(card_downs(b32), b32);
        assert!(!card_downs(b32).holds(DownAct::Q8Act));
        let both = q8.with(DownAct::Q8Blocks32);
        assert_eq!(card_downs(both), both);
    }
}
