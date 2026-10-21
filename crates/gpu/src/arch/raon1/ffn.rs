//! Raon-1's feed-forward blocks over one row: the dense SiTU-GLU block of
//! the leading layers and the routed block with its shared expert
//! (llama.cpp `src/models/raon1.cpp`: `build_ffn` / `build_moe_ffn` at
//! `LLM_FFN_SITU`, the shared expert added after the routed sum).
//!
//! - dense: gate and up sites of the normed row, SiTU-GLU
//!   (`ElemKernels::enqueue_situ`), the down site;
//! - routed: the sigmoid router at 128 experts and the top 6
//!   ([`sigmoid_128_6`]: the selection bias steers the pick only, the kept
//!   unbiased scores renormalized and scaled), the card leg over every
//!   expert (`card_leg`: the gate·up `_sel` under [`Act::Situ`], the down
//!   `_sel`, the slots' weighted sum in slot order), then the shared expert
//!   as the dense block at its width, added to the routed sum.
//!
//! Every expert is resident: a slot's place in its stacks is its expert id,
//! so the router's ids are the leg's places.

use crate::arch::hybrid::{Acts, GqaCx, Site};
use crate::card_leg::{CardLeg, LegBufs, LegRun, LegShape, LegStacks, LegSum};
use crate::kquant::Act;
use crate::weights::Weights;
use crate::{Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
use model::placement::kernels::DownAct;

crate::sigmoid_router_instance! {
    /// The sigmoid router at 128 experts and the top 6 (Raon-1): the rule
    /// of `sigmoid_128_4` (MiniMax-M3) at another pick count, from the
    /// generic bodies. Named by its shape: any family at 128/6 imports it.
    pub mod sigmoid_128_6 {
        experts: 128,
        used: 6,
        kernels: sigmoid_128_6_router_kernels,
        entries: sigmoid_128_6_router, sigmoid_128_6_router_pick,
        prepare: prepare_sigmoid_128_6_router, prepare_sigmoid_128_6_router_pick,
    }
}

/// What the refusals of this file name.
const WHAT: &str = "raon1 ffn";

/// A gate·up·down block's three sites.
pub(crate) struct Glu {
    pub(crate) gate: Site,
    pub(crate) up: Site,
    pub(crate) down: Site,
    /// The block's width: the gate's and up's rows.
    pub(crate) ff: usize,
}

impl Glu {
    pub(crate) fn sites(&self) -> [&Site; 3] {
        [&self.gate, &self.up, &self.down]
    }
}

/// A routed layer's resident pieces.
pub(crate) struct Routed {
    /// The F32 router rows and the selection bias.
    pub(crate) router: String,
    pub(crate) bias: String,
    /// The gate, up and down stacks' names.
    pub(crate) stacks: [String; 3],
    pub(crate) shared: Glu,
}

/// A layer's feed-forward block.
pub(crate) enum Ffn {
    Dense(Glu),
    Routed(Routed),
}

/// The buffers the blocks run over, one row.
pub(crate) struct FfnScratch {
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    act: DeviceBuffer<f32>,
    shared: DeviceBuffer<f32>,
    routed: DeviceBuffer<f32>,
    route: sigmoid_128_6::RouterOut,
    leg: LegBufs,
}

impl FfnScratch {
    /// The buffers for rows of `n` values, the widest dense block `ff` and
    /// routed experts `expert_ff` wide. Load-time only.
    pub(crate) fn new(
        stream: &CudaStream,
        n: usize,
        ff: usize,
        expert_ff: usize,
    ) -> Result<FfnScratch, GpuError> {
        let z = |len: usize| -> Result<DeviceBuffer<f32>, GpuError> {
            Ok(DeviceBuffer::zeroed(stream, len.max(1))?)
        };
        let shape = LegShape {
            cap: 1,
            slots_per_col: sigmoid_128_6::N_USED,
            n,
            ff: expert_ff,
        };
        Ok(FfnScratch {
            gate: z(ff)?,
            up: z(ff)?,
            act: z(ff)?,
            shared: z(n)?,
            routed: z(n)?,
            route: sigmoid_128_6::RouterOut::new(stream)?,
            leg: LegBufs::for_down(stream, shape, DownAct::Q8Blocks32)?,
        })
    }

    pub(crate) fn bytes(&self) -> usize {
        [&self.gate, &self.up, &self.act, &self.shared, &self.routed]
            .iter()
            .map(|b| b.num_bytes())
            .sum::<usize>()
            + self.leg.bytes()
            + 3 * size_of::<f32>() * sigmoid_128_6::N_EXPERT
    }
}

/// The kernels the blocks launch beyond the sites'.
pub(crate) struct FfnKernels {
    pub(crate) router: sigmoid_128_6::RouterKernels,
    pub(crate) leg: CardLeg,
}

impl FfnKernels {
    pub(crate) fn load(gpu: &Gpu) -> Result<FfnKernels, GpuError> {
        Ok(FfnKernels {
            router: sigmoid_128_6::RouterKernels::load(gpu.context())?,
            leg: CardLeg::load(gpu)?,
        })
    }
}

/// `y = down(situ(gate(xn), up(xn)))` over one row.
fn glu(
    cx: &GqaCx<'_>,
    (acts, rule): (&mut Acts, Act),
    l: usize,
    g: &Glu,
    xn: &DeviceBuffer<f32>,
    s: (
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
    ),
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let (gate, up, act) = s;
    g.gate.project(cx, acts, l, xn, gate)?;
    g.up.project(cx, acts, l, xn, up)?;
    cx.gpu
        .elem()
        .enqueue_situ(cx.gpu.stream(), rule, (gate, up), g.ff, act)?;
    g.down.project(cx, acts, l, act, y)
}

/// Layer `l`'s block over its normed row `xn` into `y` (the branch's
/// output): the dense block, or the routed sum plus the shared expert.
#[allow(
    clippy::too_many_arguments,
    reason = "one block's inputs: the launch context, the kernels, the forms, the rule, the layer, \
              the row and the scratch"
)]
pub(crate) fn ffn(
    cx: &GqaCx<'_>,
    k: &FfnKernels,
    acts: &mut Acts,
    (rule, scale): (Act, f32),
    (l, f): (usize, &Ffn),
    xn: &DeviceBuffer<f32>,
    s: &mut FfnScratch,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let FfnScratch {
        gate,
        up,
        act,
        shared,
        routed,
        route,
        leg,
    } = s;
    match f {
        Ffn::Dense(g) => glu(cx, (acts, rule), l, g, xn, (gate, up, act), y),
        Ffn::Routed(r) => {
            let (gpu, w) = (cx.gpu, cx.w);
            let stream = gpu.stream();
            k.router.enqueue_router(
                stream,
                w.f32_tensor(WHAT, &r.router)?,
                xn,
                w.f32_buf(WHAT, &r.bias)?,
                scale,
                route,
                gpu.layer_sink(l)?,
            )?;
            routed_sum(gpu, w, &k.leg, (l, r, rule), xn, route, leg, routed)?;
            glu(cx, (acts, rule), l, &r.shared, xn, (gate, up, act), shared)?;
            gpu.elem().enqueue_add(stream, routed, shared, cx.hidden, y)
        }
    }
}

/// The card leg over every expert of layer `l`, the router's ids as the
/// slots' places, its weights the sum's, into `acc`.
#[allow(
    clippy::too_many_arguments,
    reason = "the leg's inputs: the card, the weights, the kernels, the layer, the row, the picks \
              and the buffers"
)]
fn routed_sum(
    gpu: &Gpu,
    w: &Weights,
    leg: &CardLeg,
    (l, r, rule): (usize, &Routed, Act),
    xn: &DeviceBuffer<f32>,
    route: &sigmoid_128_6::RouterOut,
    bufs: &mut LegBufs,
    acc: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let names = r.stacks.each_ref().map(String::as_str);
    let stacks = LegStacks::of_resident(w, names)?;
    let run = LegRun {
        layer: l,
        stacks: &stacks,
        normed: xn,
        m: 1,
        sel: &route.ids,
        n_card: sigmoid_128_6::N_EXPERT,
        rule,
    };
    leg.enqueue(
        gpu,
        bufs,
        &run,
        Some(LegSum {
            w: &route.weights,
            acc,
        }),
    )
}
