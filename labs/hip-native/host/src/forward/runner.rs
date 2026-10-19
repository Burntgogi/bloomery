use std::ffi::c_void;

use super::plan::{Matrix, Mix, Plan, Region, fit};
use crate::{Arg, Context};

#[derive(Clone, Copy)]
#[repr(usize)]
enum Kernel {
    Embed,
    Gemv,
    Norm,
    Glu,
    Add,
    Argmax,
    Conv,
    Decay,
    Delta,
    Gate,
    Prep,
    Attention,
}

const ENTRIES: [&str; 12] = [
    "bloomery_hip_embed",
    "bloomery_hip_gemv",
    "bloomery_hip_norm",
    "bloomery_hip_glu",
    "bloomery_hip_add",
    "bloomery_hip_argmax",
    "bloomery_hip_gdn_conv",
    "bloomery_hip_gdn_decay",
    "bloomery_hip_gdn_delta",
    "bloomery_hip_gdn_gate",
    "bloomery_hip_gqa_prep",
    "bloomery_hip_attention",
];

struct Buffers {
    x: *mut c_void,
    norm: *mut c_void,
    q: *mut c_void,
    k: *mut c_void,
    v: *mut c_void,
    gate: *mut c_void,
    up: *mut c_void,
    conv: *mut c_void,
    beta: *mut c_void,
    decay: *mut c_void,
    delta: *mut c_void,
    mixed: *mut c_void,
    logits: *mut c_void,
    fault: *mut c_void,
    summary: *mut c_void,
    rope: *mut c_void,
}

enum Store {
    Gdn {
        ring: *mut c_void,
        state: *mut c_void,
    },
    Attention {
        keys: *mut c_void,
        values: *mut c_void,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Ready(u32),
    Poisoned,
}

pub(super) struct Model<'c, 'p> {
    context: &'c Context,
    plan: &'p Plan,
    weights: Vec<*mut c_void>,
    functions: [*mut c_void; 12],
    arena: *mut c_void,
    buffers: Buffers,
    stores: Vec<Store>,
    state: State,
}

fn pointer(base: *mut c_void, region: Region, arena_bytes: usize) -> *mut c_void {
    assert!(
        region.offset <= arena_bytes && region.bytes <= arena_bytes - region.offset,
        "region is outside the planned arena"
    );
    // SAFETY: the one arena allocation covers every checked region, including an unused empty tail.
    unsafe { base.cast::<u8>().add(region.offset).cast() }
}

pub(super) fn rope_rows(base: f32, dims: u32, context: u32) -> Vec<f32> {
    // The CUDA table owner cannot link here. Retain gpu::rope_table's f32 theta recurrence,
    // changing only its interleaved cos/sin layout to the lab kernel's two planes.
    let mut rows = vec![0.0; dims as usize * context as usize];
    if dims == 0 {
        return rows;
    }
    let scale = base.powf(-2.0 / dims as f32);
    for pos in 0..context {
        let mut theta = pos as f32;
        for pair in 0..dims / 2 {
            let (sin, cos) = theta.sin_cos();
            rows[(pos * dims + pair) as usize] = cos;
            rows[(pos * dims + dims / 2 + pair) as usize] = sin;
            theta *= scale;
        }
    }
    rows
}

impl<'c, 'p> Model<'c, 'p> {
    pub fn load(context: &'c mut Context, plan: &'p Plan) -> Result<Self, String> {
        let (free, _) = context.memory()?;
        fit(plan.required()?, free)?;
        crate::selected::wave_size(context)?;
        let mut functions = [std::ptr::null_mut(); 12];
        for (slot, name) in functions.iter_mut().zip(ENTRIES) {
            *slot = context.function(name)?;
        }
        let mut weights = Vec::with_capacity(plan.tensors.len());
        for tensor in &plan.tensors {
            let shard = plan.file.shard(tensor.shard).expect("plan's tensor shard");
            weights.push(context.upload(shard.data(&tensor.info).map_err(|e| e.to_string())?)?);
        }
        let arena = context.alloc(plan.arena_bytes)?;
        let at = |region| pointer(arena, region, plan.arena_bytes);
        let s = &plan.scratch;
        let buffers = Buffers {
            x: at(s.x),
            norm: at(s.norm),
            q: at(s.q),
            k: at(s.k),
            v: at(s.v),
            gate: at(s.gate),
            up: at(s.up),
            conv: at(s.conv),
            beta: at(s.beta),
            decay: at(s.decay),
            delta: at(s.delta),
            mixed: at(s.mixed),
            logits: at(s.logits),
            fault: at(s.fault),
            summary: at(s.summary),
            rope: at(s.rope),
        };
        let stores = plan
            .layers
            .iter()
            .map(|l| match &l.mixer {
                Mix::Gdn(d) => Store::Gdn {
                    ring: at(d.ring),
                    state: at(d.state),
                },
                Mix::Attention(g) => Store::Attention {
                    keys: at(g.keys),
                    values: at(g.values),
                },
            })
            .collect();
        if plan.rope_dim != 0 {
            let rows = rope_rows(plan.rope_base, plan.rope_dim, plan.context);
            context.overwrite(buffers.rope, &rows)?;
        }
        let mut model = Self {
            context,
            plan,
            weights,
            functions,
            arena,
            buffers,
            stores,
            state: State::Poisoned,
        };
        model.reset()?;
        Ok(model)
    }

    fn call<const N: usize>(&self, kernel: Kernel, n: u32, args: [Arg; N]) -> Result<(), String> {
        self.context
            .call(self.functions[kernel as usize], n as usize, args)
    }

    fn tag(layer: u32, kernel: Kernel) -> Arg {
        Arg::u32((layer << 8) | (kernel as u32 + 1))
    }

    fn gemv(
        &self,
        matrix: Matrix,
        x: *mut c_void,
        y: *mut c_void,
        layer: u32,
    ) -> Result<(), String> {
        self.call(
            Kernel::Gemv,
            matrix.rows,
            [
                Arg::ptr(self.weights[matrix.tensor]),
                Arg::ptr(x),
                Arg::ptr(y),
                Arg::u32(matrix.k),
                Arg::u32(matrix.rows),
                Arg::u32(matrix.kind),
                Arg::ptr(self.buffers.fault),
                Self::tag(layer, Kernel::Gemv),
            ],
        )
    }

    fn norm(&self, gain: usize, layer: u32) -> Result<(), String> {
        self.call(
            Kernel::Norm,
            1,
            [
                Arg::ptr(self.buffers.x),
                Arg::ptr(self.weights[gain]),
                Arg::ptr(self.buffers.norm),
                Arg::u32(self.plan.spec.hidden),
                Arg::u32(1),
                Arg::f32(self.plan.spec.rms_eps),
                Arg::ptr(self.buffers.fault),
                Self::tag(layer, Kernel::Norm),
            ],
        )
    }

    fn add(&self, layer: u32) -> Result<(), String> {
        self.call(
            Kernel::Add,
            self.plan.spec.hidden,
            [
                Arg::ptr(self.buffers.norm),
                Arg::ptr(self.buffers.x),
                Arg::u32(self.plan.spec.hidden),
                Arg::ptr(self.buffers.fault),
                Self::tag(layer, Kernel::Add),
            ],
        )
    }

    pub fn reset(&mut self) -> Result<(), String> {
        self.state = State::Poisoned;
        if self.plan.state_bytes != 0 {
            self.context.fill(self.arena, 0, self.plan.state_bytes)?;
        }
        self.context.fill(self.buffers.fault, 255, 4)?;
        self.state = State::Ready(0);
        Ok(())
    }

    pub fn step(&mut self, token: u32) -> Result<u32, String> {
        let pos = match self.state {
            State::Ready(p) => p,
            State::Poisoned => return Err("forward is poisoned; reset is required".into()),
        };
        if token >= self.plan.spec.vocab {
            return Err(format!(
                "token {token} is outside vocabulary {}",
                self.plan.spec.vocab
            ));
        }
        if pos >= self.plan.context {
            return Err(format!("context exhausted at position {pos}"));
        }
        self.state = State::Poisoned;
        let b = &self.buffers;
        self.call(
            Kernel::Embed,
            self.plan.spec.hidden,
            [
                Arg::ptr(self.weights[self.plan.embedding.tensor]),
                Arg::ptr(b.x),
                Arg::u32(self.plan.spec.hidden),
                Arg::u32(token),
                Arg::u32(self.plan.embedding.kind),
                Arg::ptr(b.fault),
                Self::tag(0, Kernel::Embed),
            ],
        )?;
        for (l, (layer, store)) in self.plan.layers.iter().zip(&self.stores).enumerate() {
            let l = l as u32;
            self.norm(layer.before_mix, l)?;
            match (&layer.mixer, store) {
                (Mix::Gdn(d), Store::Gdn { ring, state }) => {
                    self.gemv(d.qkv, b.norm, b.q, l)?;
                    self.gemv(d.gate, b.norm, b.gate, l)?;
                    self.gemv(d.beta, b.norm, b.k, l)?;
                    self.gemv(d.alpha, b.norm, b.v, l)?;
                    self.call(
                        Kernel::Conv,
                        2 * d.nk + d.nv,
                        [
                            Arg::ptr(b.q),
                            Arg::ptr(self.weights[d.conv]),
                            Arg::ptr(*ring),
                            Arg::ptr(b.conv),
                            Arg::u32(d.nk),
                            Arg::u32(d.nv),
                            Arg::u32(pos),
                            Arg::f32(self.plan.spec.rms_eps),
                            Arg::ptr(b.fault),
                            Self::tag(l, Kernel::Conv),
                        ],
                    )?;
                    self.call(
                        Kernel::Decay,
                        d.nv,
                        [
                            Arg::ptr(b.k),
                            Arg::ptr(b.v),
                            Arg::ptr(self.weights[d.dt]),
                            Arg::ptr(self.weights[d.a]),
                            Arg::ptr(b.beta),
                            Arg::ptr(b.decay),
                            Arg::u32(d.nv),
                            Arg::ptr(b.fault),
                            Self::tag(l, Kernel::Decay),
                        ],
                    )?;
                    self.call(
                        Kernel::Delta,
                        d.nv * 128 * 8,
                        [
                            Arg::ptr(b.conv),
                            Arg::ptr(b.beta),
                            Arg::ptr(b.decay),
                            Arg::ptr(*state),
                            Arg::ptr(b.delta),
                            Arg::u32(d.nk),
                            Arg::u32(d.nv),
                            Arg::ptr(b.fault),
                            Self::tag(l, Kernel::Delta),
                        ],
                    )?;
                    self.call(
                        Kernel::Gate,
                        d.nv,
                        [
                            Arg::ptr(b.delta),
                            Arg::ptr(b.gate),
                            Arg::ptr(self.weights[d.gain]),
                            Arg::ptr(b.mixed),
                            Arg::u32(d.nv),
                            Arg::f32(self.plan.spec.rms_eps),
                            Arg::ptr(b.fault),
                            Self::tag(l, Kernel::Gate),
                        ],
                    )?;
                    self.gemv(d.output, b.mixed, b.norm, l)?;
                }
                (Mix::Attention(g), Store::Attention { keys, values }) => {
                    self.gemv(g.q, b.norm, b.q, l)?;
                    self.gemv(g.k, b.norm, b.k, l)?;
                    self.gemv(g.v, b.norm, b.v, l)?;
                    let shape = &g.shape;
                    self.call(
                        Kernel::Prep,
                        shape.heads + shape.kv_heads,
                        [
                            Arg::ptr(b.q),
                            Arg::ptr(b.k),
                            Arg::ptr(b.v),
                            Arg::ptr(self.weights[g.qgain]),
                            Arg::ptr(self.weights[g.kgain]),
                            Arg::ptr(b.rope),
                            Arg::ptr(b.delta),
                            Arg::ptr(*keys),
                            Arg::ptr(*values),
                            Arg::u32(shape.heads),
                            Arg::u32(shape.kv_heads),
                            Arg::u32(shape.head_dim),
                            Arg::u32(shape.rope.dims),
                            Arg::u32(pos),
                            Arg::u32(self.plan.context),
                            Arg::f32(self.plan.spec.rms_eps),
                            Arg::ptr(b.fault),
                            Self::tag(l, Kernel::Prep),
                        ],
                    )?;
                    self.call(
                        Kernel::Attention,
                        shape.heads,
                        [
                            Arg::ptr(b.delta),
                            Arg::ptr(b.q),
                            Arg::ptr(*keys),
                            Arg::ptr(*values),
                            Arg::ptr(b.mixed),
                            Arg::u32(shape.heads),
                            Arg::u32(shape.kv_heads),
                            Arg::u32(shape.head_dim),
                            Arg::u32(pos),
                            Arg::u32(self.plan.context),
                            Arg::ptr(b.fault),
                            Self::tag(l, Kernel::Attention),
                        ],
                    )?;
                    self.gemv(g.output, b.mixed, b.norm, l)?;
                }
                _ => unreachable!("plan and loaded store variants agree"),
            }
            self.add(l)?;
            self.norm(layer.before_ffn, l)?;
            self.gemv(layer.gate, b.norm, b.gate, l)?;
            self.gemv(layer.up, b.norm, b.up, l)?;
            self.call(
                Kernel::Glu,
                layer.gate.rows,
                [
                    Arg::ptr(b.gate),
                    Arg::ptr(b.up),
                    Arg::ptr(b.gate),
                    Arg::u32(layer.gate.rows),
                    Arg::ptr(b.fault),
                    Self::tag(l, Kernel::Glu),
                ],
            )?;
            self.gemv(layer.down, b.gate, b.norm, l)?;
            self.add(l)?;
        }
        let head_layer = self.plan.layers.len() as u32;
        self.norm(self.plan.output_gain, head_layer)?;
        self.gemv(self.plan.head, b.norm, b.logits, head_layer)?;
        self.call(
            Kernel::Argmax,
            1,
            [
                Arg::ptr(b.logits),
                Arg::ptr(b.summary),
                Arg::u32(self.plan.spec.vocab),
                Arg::ptr(b.fault),
                Self::tag(head_layer, Kernel::Argmax),
            ],
        )?;
        let mut summary = [u32::MAX; 2];
        self.context.download(b.summary, &mut summary)?;
        if summary[1] != u32::MAX {
            return Err(format!(
                "device fault at layer {}, lab-site {}; reset is required",
                summary[1] >> 8,
                summary[1] & 255
            ));
        }
        if summary[0] >= self.plan.spec.vocab {
            return Err("device argmax returned an out-of-range token; reset is required".into());
        }
        self.state = State::Ready(pos + 1);
        Ok(summary[0])
    }

    pub fn logits(&self, out: &mut [f32]) -> Result<(), String> {
        if !matches!(self.state,State::Ready(p) if p>0) {
            return Err("logits require a successful step since reset".into());
        }
        if out.len() != self.plan.spec.vocab as usize {
            return Err("logits buffer must cover exactly the vocabulary".into());
        }
        self.context.download(self.buffers.logits, out)
    }

    pub fn memory(&self) -> Result<(usize, usize), String> {
        self.context.memory()
    }
}

/// Exercises the actual model's refusal/recovery transitions on its private smoke weights.
pub(super) fn probe_contract(model: &mut Model<'_, '_>) -> Result<(), String> {
    let vocab = model.plan.spec.vocab;
    model.reset()?;
    let first = model.step(7)?;
    let mut baseline = vec![0.0; vocab as usize];
    let mut got = vec![0.0; vocab as usize];
    model.logits(&mut baseline)?;
    if model.step(vocab).is_ok() {
        return Err("model accepted an out-of-range token".into());
    }
    let second = model.step(23)?;
    model.logits(&mut got)?;
    let after_second = got.clone();
    model.reset()?;
    if model.step(7)? != first || model.step(23)? != second {
        return Err("token refusal advanced or poisoned the model".into());
    }
    model.logits(&mut got)?;
    if got
        .iter()
        .zip(&after_second)
        .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err("token refusal changed the subsequent numerical state".into());
    }
    let gain = model.weights[model.plan.output_gain];
    let nan = [f32::NAN];
    model.context.overwrite(gain, &nan)?;
    if model.step(11).is_ok() || model.state != State::Poisoned {
        return Err("nonfinite model arithmetic did not poison the model".into());
    }
    let mut poisoned = vec![0_u8; model.plan.state_bytes];
    let mut unchanged = vec![0_u8; model.plan.state_bytes];
    model.context.download(model.arena, &mut poisoned)?;
    if model.step(13).is_ok() {
        return Err("poisoned model accepted another token".into());
    }
    model.context.download(model.arena, &mut unchanged)?;
    if poisoned != unchanged {
        return Err("poisoned refusal changed recurrent/KV state".into());
    }
    let original = &model.plan.tensors[model.plan.output_gain];
    let bytes = model
        .plan
        .file
        .shard(original.shard)
        .expect("smoke gain shard")
        .data(&original.info)
        .map_err(|e| e.to_string())?;
    model.context.overwrite(gain, &bytes[..4])?;
    model.reset()?;
    if model.step(7)? != first {
        return Err("reset did not recover the original first token".into());
    }
    model.logits(&mut got)?;
    if got
        .iter()
        .zip(&baseline)
        .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err("reset did not restore first-token logits bitwise".into());
    }
    for pos in 1..model.plan.context {
        model.step((pos * 17) % vocab)?;
    }
    model.context.download(model.arena, &mut poisoned)?;
    if model.step(3).is_ok() {
        return Err("model accepted a token past context capacity".into());
    }
    model.context.download(model.arena, &mut unchanged)?;
    if poisoned != unchanged {
        return Err("context refusal modified numerical state".into());
    }
    model.reset()?;
    if model.step(7)? != first {
        return Err("reset did not recover from context exhaustion".into());
    }
    println!(
        "{{\"operation\":\"forward_model_state\",\"token_bounds\":true,\"context_bounds\":true,\"poison_refusal\":true,\"reset_bitwise_recovery\":true}}"
    );
    Ok(())
}
