use std::collections::BTreeMap;
use std::path::Path;

use gguf::{GgmlType, Split, TensorInfo};
use model::arch::models::{
    Act, Arch, DeltaKind, Ffn, GdnGate, Gqa, Mixer, ModelSpec, OutGate, Residual, Role, RopeMode,
};
use model::arch::{names, qwen35moe::names as qnames};

include!(concat!(env!("OUT_DIR"), "/fit_reserve.rs"));
pub(super) const MAX_CONTEXT: u32 = 64;

#[derive(Clone, Copy)]
pub(super) struct Region {
    pub offset: usize,
    pub bytes: usize,
}

#[derive(Default)]
struct Arena {
    bytes: usize,
}

impl Arena {
    fn add(&mut self, count: usize, width: usize) -> Result<Region, String> {
        let bytes = count.checked_mul(width).ok_or("arena size overflow")?;
        let offset = self
            .bytes
            .checked_next_multiple_of(8)
            .ok_or("arena offset overflow")?;
        self.bytes = offset.checked_add(bytes).ok_or("arena size overflow")?;
        Ok(Region { offset, bytes })
    }
}

#[derive(Clone, Copy)]
pub(super) struct Matrix {
    pub tensor: usize,
    pub k: u32,
    pub rows: u32,
    pub kind: u32,
}

pub(super) struct Tensor {
    pub shard: usize,
    pub info: TensorInfo,
}

pub(super) struct Gdn {
    pub qkv: Matrix,
    pub gate: Matrix,
    pub alpha: Matrix,
    pub beta: Matrix,
    pub output: Matrix,
    pub conv: usize,
    pub dt: usize,
    pub a: usize,
    pub gain: usize,
    pub nk: u32,
    pub nv: u32,
    pub ring: Region,
    pub state: Region,
}

pub(super) struct Attention {
    pub q: Matrix,
    pub k: Matrix,
    pub v: Matrix,
    pub output: Matrix,
    pub qgain: usize,
    pub kgain: usize,
    pub shape: Gqa,
    pub keys: Region,
    pub values: Region,
}

pub(super) enum Mix {
    Gdn(Gdn),
    Attention(Attention),
}

pub(super) struct Layer {
    pub before_mix: usize,
    pub before_ffn: usize,
    pub mixer: Mix,
    pub gate: Matrix,
    pub up: Matrix,
    pub down: Matrix,
}

pub(super) struct Scratch {
    pub x: Region,
    pub norm: Region,
    pub q: Region,
    pub k: Region,
    pub v: Region,
    pub gate: Region,
    pub up: Region,
    pub conv: Region,
    pub beta: Region,
    pub decay: Region,
    pub delta: Region,
    pub mixed: Region,
    pub logits: Region,
    pub fault: Region,
    pub summary: Region,
    pub rope: Region,
}

pub(super) struct Plan {
    pub file: Split,
    pub spec: ModelSpec,
    pub tensors: Vec<Tensor>,
    pub embedding: Matrix,
    pub head: Matrix,
    pub output_gain: usize,
    pub layers: Vec<Layer>,
    pub scratch: Scratch,
    pub context: u32,
    pub weight_bytes: usize,
    pub arena_bytes: usize,
    pub state_bytes: usize,
    pub rope_base: f32,
    pub rope_dim: u32,
}

struct Weights<'a> {
    file: &'a Split,
    tensors: Vec<Tensor>,
    names: BTreeMap<String, usize>,
    bytes: usize,
}

impl<'a> Weights<'a> {
    fn new(file: &'a Split) -> Self {
        Self {
            file,
            tensors: Vec::new(),
            names: BTreeMap::new(),
            bytes: 0,
        }
    }

    fn take(&mut self, name: String) -> Result<usize, String> {
        if let Some(&id) = self.names.get(&name) {
            return Ok(id);
        }
        let (shard, info) = self
            .file
            .find(&name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        self.bytes = self
            .bytes
            .checked_add(usize::try_from(info.nbytes).map_err(|e| e.to_string())?)
            .ok_or("weight byte count overflow")?;
        let id = self.tensors.len();
        self.tensors.push(Tensor {
            shard,
            info: info.clone(),
        });
        self.names.insert(name, id);
        Ok(id)
    }

    fn matrix(&mut self, name: String, k: u32, rows: u32) -> Result<Matrix, String> {
        let tensor = self.take(name)?;
        let info = &self.tensors[tensor].info;
        if info.dims != [u64::from(k), u64::from(rows)] {
            return Err(format!(
                "{}: expected [{k}, {rows}], got {:?}",
                info.name, info.dims
            ));
        }
        if !matches!(
            info.ty,
            GgmlType::F32 | GgmlType::Q4_K | GgmlType::Q6_K | GgmlType::Q8_0
        ) {
            return Err(format!(
                "{}: unsupported forward matrix format {:?}",
                info.name, info.ty
            ));
        }
        let block = info.ty.blck_size().ok_or("matrix block size is unknown")?;
        if k == 0 || rows == 0 || !u64::from(k).is_multiple_of(block) {
            return Err(format!("{}: invalid packed row width {k}", info.name));
        }
        Ok(Matrix {
            tensor,
            k,
            rows,
            kind: info.ty.as_u32(),
        })
    }

    fn f32(&mut self, name: String, dims: &[u64]) -> Result<usize, String> {
        let tensor = self.take(name)?;
        let info = &self.tensors[tensor].info;
        if info.ty != GgmlType::F32 || info.dims != dims {
            return Err(format!(
                "{}: expected F32 {dims:?}, got {:?} {:?}",
                info.name, info.ty, info.dims
            ));
        }
        Ok(tensor)
    }

    fn vector(&mut self, name: String, count: u32) -> Result<usize, String> {
        self.f32(name, &[u64::from(count)])
    }
}

fn product(a: u32, b: u32) -> Result<u32, String> {
    a.checked_mul(b)
        .ok_or_else(|| format!("forward dimension overflow: {a} * {b}"))
}

pub(super) fn fit(required: usize, free: usize) -> Result<(), String> {
    if required
        .checked_add(FIT_RESERVE)
        .is_none_or(|need| need > free)
    {
        return Err(format!(
            "load fit refused: {required} managed bytes + {FIT_RESERVE} reserve bytes, {free} free bytes; no model weights allocated"
        ));
    }
    Ok(())
}

impl Plan {
    pub fn open(path: &Path, context: u32) -> Result<Self, String> {
        if !(1..=MAX_CONTEXT).contains(&context) {
            return Err(format!(
                "forward context must be 1..={MAX_CONTEXT}, got {context}"
            ));
        }
        let file = Split::open(path).map_err(|e| e.to_string())?;
        let read = model::arch::spec(&file).map_err(|e| e.to_string())?;
        let spec = read.spec;
        if spec.arch != Arch::Qwen35
            || spec.hidden == 0
            || spec.vocab == 0
            || spec.layers.is_empty()
            || spec.hc.is_some()
            || spec.engram.is_some()
            || spec.encoder.is_some()
            || !spec.rms_eps.is_finite()
            || spec.rms_eps <= 0.0
            || context > spec.ctx_train
        {
            return Err("forward requires a nonempty dense Qwen3.5 text trunk with finite positive RMS epsilon and context within training bounds".into());
        }
        let mut w = Weights::new(&file);
        let embedding = w.matrix(names::token_embd(), spec.hidden, spec.vocab)?;
        let head = if file.find(&names::output()).is_some() {
            w.matrix(names::output(), spec.hidden, spec.vocab)?
        } else {
            embedding
        };
        let output_gain = w.vector(names::output_norm(), spec.hidden)?;
        let mut arena = Arena::default();
        let mut layers = Vec::with_capacity(spec.layers.len());
        let (mut max_q, mut max_kv, mut max_gate, mut max_conv, mut max_nv, mut max_mixed) =
            (0, 0, 0, 0, 0, 0);
        let mut rope = None;
        for (l, layer) in spec.layers.iter().enumerate() {
            if layer.residual != Residual::Plain || !layer.extras.is_empty() {
                return Err(format!(
                    "layer {l}: only plain residuals without extra sites are supported"
                ));
            }
            let ff = match layer.ffn {
                Ffn::Dense {
                    ff,
                    act: Act::SwiGlu { limit: None },
                } if ff > 0 => ff,
                _ => return Err(format!("layer {l}: requires dense unclipped SwiGLU")),
            };
            let before_mix = w.vector(names::blk(l, "attn_norm.weight"), spec.hidden)?;
            let before_ffn = w.vector(names::blk(l, "post_attention_norm.weight"), spec.hidden)?;
            let mixer = match &layer.mixer {
                Mixer::DeltaRule(d) => {
                    if !matches!(
                        d.kind,
                        DeltaKind::Gdn {
                            gate: GdnGate::Silu,
                            ..
                        }
                    ) || d.d != 128
                        || d.conv != 4
                        || d.k_heads == 0
                        || d.v_heads == 0
                        || !d.v_heads.is_multiple_of(d.k_heads)
                    {
                        return Err(format!(
                            "layer {l}: requires tiled GDN with head128, conv4 and valid key/value heads"
                        ));
                    }
                    let channels = product(
                        d.k_heads
                            .checked_mul(2)
                            .and_then(|v| v.checked_add(d.v_heads))
                            .ok_or("GDN head count overflow")?,
                        128,
                    )?;
                    let mixed = product(d.v_heads, 128)?;
                    max_q = max_q.max(channels);
                    max_conv = max_conv.max(channels);
                    max_gate = max_gate.max(mixed);
                    max_nv = max_nv.max(d.v_heads);
                    max_mixed = max_mixed.max(mixed);
                    let state = arena.add(product(mixed, 128)? as usize, 4)?;
                    let ring = arena.add(product(channels, 11)? as usize, 4)?;
                    let conv = w.f32(qnames::ssm_conv1d(l), &[4, u64::from(channels)])?;
                    Mix::Gdn(Gdn {
                        qkv: w.matrix(qnames::attn_qkv(l), spec.hidden, channels)?,
                        gate: w.matrix(qnames::attn_gate(l), spec.hidden, mixed)?,
                        alpha: w.matrix(qnames::ssm_alpha(l), spec.hidden, d.v_heads)?,
                        beta: w.matrix(qnames::ssm_beta(l), spec.hidden, d.v_heads)?,
                        output: w.matrix(qnames::ssm_out(l), mixed, spec.hidden)?,
                        conv,
                        dt: w.vector(qnames::ssm_dt_bias(l), d.v_heads)?,
                        a: w.vector(qnames::ssm_a(l), d.v_heads)?,
                        gain: w.vector(qnames::ssm_norm(l), 128)?,
                        nk: d.k_heads,
                        nv: d.v_heads,
                        ring,
                        state,
                    })
                }
                Mixer::Gqa(g) => {
                    if g.heads == 0
                        || g.kv_heads == 0
                        || !g.heads.is_multiple_of(g.kv_heads)
                        || g.head_dim == 0
                        || g.head_dim > 256
                        || !g.head_dim.is_multiple_of(2)
                        || g.value_dim != g.head_dim
                        || !g.qk_norm
                        || g.out_gate != OutGate::Packed
                        || g.select.is_some()
                        || g.window.is_some()
                        || g.sinks
                        || g.value_scale.is_some()
                        || g.rope.yarn.is_some()
                        || !matches!(g.rope.mode, RopeMode::Neox | RopeMode::Imrope { .. })
                        || g.rope.dims < 2
                        || !g.rope.dims.is_multiple_of(2)
                        || g.rope.dims > g.head_dim
                        || !g.rope.base.is_finite()
                        || g.rope.base <= 0.0
                    {
                        return Err(format!(
                            "layer {l}: requires gated normalized GQA with partial Neox/IMROPE, no scaling/window/pool/sink, head<=256"
                        ));
                    }
                    if rope.is_some_and(|r| r != g.rope) {
                        return Err(format!(
                            "layer {l}: differing layer RoPE parameters are unsupported"
                        ));
                    }
                    rope = Some(g.rope);
                    let mixed = product(g.heads, g.head_dim)?;
                    let q = product(mixed, 2)?;
                    let kv = product(g.kv_heads, g.head_dim)?;
                    max_q = max_q.max(q);
                    max_kv = max_kv.max(kv);
                    max_mixed = max_mixed.max(mixed);
                    let keys = arena.add(product(kv, context)? as usize, 2)?;
                    let values = arena.add(product(kv, context)? as usize, 2)?;
                    Mix::Attention(Attention {
                        q: w.matrix(qnames::attn_q(l), spec.hidden, q)?,
                        k: w.matrix(qnames::attn_k(l), spec.hidden, kv)?,
                        v: w.matrix(qnames::attn_v(l), spec.hidden, kv)?,
                        output: w.matrix(qnames::attn_output(l), mixed, spec.hidden)?,
                        qgain: w.vector(qnames::attn_q_norm(l), g.head_dim)?,
                        kgain: w.vector(qnames::attn_k_norm(l), g.head_dim)?,
                        shape: g.clone(),
                        keys,
                        values,
                    })
                }
                _ => return Err(format!("layer {l}: unsupported mixer")),
            };
            max_gate = max_gate.max(ff);
            layers.push(Layer {
                before_mix,
                before_ffn,
                mixer,
                gate: w.matrix(names::blk(l, "ffn_gate.weight"), spec.hidden, ff)?,
                up: w.matrix(names::blk(l, "ffn_up.weight"), spec.hidden, ff)?,
                down: w.matrix(names::blk(l, "ffn_down.weight"), ff, spec.hidden)?,
            });
        }
        for tensor in &read.tensors.tensors {
            if !matches!(tensor.role, Role::Unused | Role::Unread)
                && !w.names.contains_key(&tensor.name)
            {
                return Err(format!(
                    "unconsumed trunk tensor {} ({})",
                    tensor.name, tensor.role
                ));
            }
        }
        let state_bytes = arena.bytes;
        let rope_dim = rope.map_or(0, |r| r.dims);
        let rope_base = rope.map_or(0.0, |r| r.base);
        let scratch = Scratch {
            x: arena.add(spec.hidden as usize, 4)?,
            norm: arena.add(spec.hidden as usize, 4)?,
            q: arena.add(max_q as usize, 4)?,
            k: arena.add(max_kv.max(max_nv) as usize, 4)?,
            v: arena.add(max_kv.max(max_nv) as usize, 4)?,
            gate: arena.add(max_gate as usize, 4)?,
            up: arena.add(max_gate as usize, 4)?,
            conv: arena.add(max_conv as usize, 4)?,
            beta: arena.add(max_nv as usize, 4)?,
            decay: arena.add(max_nv as usize, 4)?,
            delta: arena.add(max_mixed as usize, 4)?,
            mixed: arena.add(max_mixed as usize, 4)?,
            logits: arena.add(spec.vocab as usize, 4)?,
            fault: arena.add(1, 4)?,
            summary: arena.add(2, 4)?,
            rope: arena.add(product(context, rope_dim)? as usize, 4)?,
        };
        let weight_bytes = w.bytes;
        let tensors = w.tensors;
        Ok(Self {
            file,
            spec,
            tensors,
            embedding,
            head,
            output_gain,
            layers,
            scratch,
            context,
            weight_bytes,
            arena_bytes: arena.bytes,
            state_bytes,
            rope_base,
            rope_dim,
        })
    }

    pub fn required(&self) -> Result<usize, String> {
        self.weight_bytes
            .checked_add(self.arena_bytes)
            .ok_or_else(|| "managed byte count overflow".into())
    }

    pub fn identity(&self) -> serde_json::Value {
        let files: Vec<_> = (0..self.file.shard_count())
            .map(|i| {
                let shard = self.file.shard(i).expect("plan's live shard");
                serde_json::json!({"file":self.file.shard_path(i).expect("plan's shard path"),
                "bytes":shard.mapping().len(), "sha256":gguf::digest::sha256_hex(shard.mapping())})
            })
            .collect();
        serde_json::json!({"files":files,"architecture":self.spec.arch.name(),
            "hidden":self.spec.hidden,"vocab":self.spec.vocab,"layers":self.layers.len(),
            "gdn_layers":self.layers.iter().filter(|l| matches!(l.mixer,Mix::Gdn(_))).count(),
            "gqa_layers":self.layers.iter().filter(|l| matches!(l.mixer,Mix::Attention(_))).count(),
            "context":self.context,"kv":"F16","ubatch":1,"mtp":false,"serving_admission":false,
            "tied_head":self.embedding.tensor==self.head.tensor,"loaded_tensors":self.tensors.len(),
            "weight_payload_bytes":self.weight_bytes,"arena_bytes":self.arena_bytes,
            "state_bytes":self.state_bytes,"fit_reserve_bytes":FIT_RESERVE,
            "allocation_estimate_excludes_driver_and_kernel_scratch":true,
            "activation_arithmetic":"F32 scalar; on-device GGUF dequantization; not CUDA Q8-activation bit parity"})
    }
}

#[cfg(test)]
mod tests {
    use super::{Arena, FIT_RESERVE, fit};

    #[test]
    fn load_fit_preserves_reserve_and_refuses_overflow() {
        assert!(fit(37, FIT_RESERVE + 37).is_ok());
        assert!(fit(37, FIT_RESERVE + 36).is_err());
        assert!(fit(usize::MAX, usize::MAX).is_err());
        let mut arena = Arena::default();
        let first = arena.add(3, 2).unwrap();
        let second = arena.add(1, 4).unwrap();
        assert_eq!(first.offset, 0);
        assert_eq!(second.offset, 8);
        assert_eq!(arena.bytes, 12);
        assert!(arena.add(usize::MAX, 4).is_err());
    }
}
