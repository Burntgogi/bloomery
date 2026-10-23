//! Flint's scalar widths and per-layer masks, checked before constructing a spec.

use super::roles;
use crate::arch::hybrid::{layers_agree, no_nextn};
use crate::arch::{
    check_kv, dims_agree, expert_counts, finite_positive, meta_bool, meta_usize, metadata, mul,
    n_vocab, one_expert_group, optional_usize, per_layer_bool, per_layer_count, plain_rope,
    positive, sigmoid_gating,
};
use crate::placement::PlacementError;
use gguf::Split;

pub(super) struct Hparams {
    pub layers: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub context: usize,
    pub eps: f32,
    pub heads: usize,
    pub kv: Vec<usize>,
    pub head: usize,
    pub ff: Vec<usize>,
    pub expert_ff: Vec<usize>,
    pub dense_lead: usize,
    pub experts: usize,
    pub used: usize,
    pub shared_ff: usize,
    pub norm: bool,
    pub scale: f32,
    pub window: usize,
    pub swa: Vec<bool>,
    pub rope: Vec<bool>,
    pub base: f32,
    pub defaults: Vec<String>,
}

impl Hparams {
    pub(super) fn read(split: &Split) -> Result<Self, PlacementError> {
        let layers = positive(split, "block_count")?;
        no_nextn(split)?;
        // This common read checks the layer bound before expanding any scalar.
        let kv = per_layer_count(split, "attention.head_count_kv", layers)?;
        let hidden = positive(split, "embedding_length")?;
        let context = positive(split, "context_length")?;
        let vocab = n_vocab(split)?;
        if vocab == 0 {
            return Err(metadata(split, "vocab_size", "is 0"));
        }
        let eps = finite_positive(split, "attention.layer_norm_rms_epsilon")?;
        let heads = positive(split, "attention.head_count")?;
        let head = positive(split, "attention.key_length")?;
        mul(split, "attention.key_length", heads, head)?;
        let value = positive(split, "attention.value_length")?;
        if value != head {
            return Err(metadata(
                split,
                "attention.value_length",
                format!("is {value}, not key_length {head}"),
            ));
        }
        if !head.is_multiple_of(2) {
            return Err(metadata(
                split,
                "attention.key_length",
                "must be even for NeoX rotation",
            ));
        }
        if let Some(dims) = optional_usize(split, "rope.dimension_count")?
            && dims != head
        {
            return Err(metadata(
                split,
                "rope.dimension_count",
                format!("is {dims}, not the full head {head}"),
            ));
        }
        plain_rope(split)?;
        let base = finite_positive(split, "rope.freq_base")?;
        let window = positive(split, "attention.sliding_window")?;
        let swa = per_layer_bool(split, "attention.sliding_window_pattern", layers)?;
        let rope = per_layer_bool(split, "attention.rope_pattern", layers)?;
        let dense_lead = meta_usize(split, "leading_dense_block_count")?;
        if dense_lead > layers {
            return Err(metadata(
                split,
                "leading_dense_block_count",
                format!("is {dense_lead}, above block_count {layers}"),
            ));
        }
        let ff = per_layer_count(split, "feed_forward_length", layers)?;
        let expert_ff = per_layer_count(split, "expert_feed_forward_length", layers)?;
        let (experts, used) = expert_counts(split)?;
        one_expert_group(split, "Flint's sigmoid router")?;
        let shared_count = positive(split, "expert_shared_count")?;
        let shared_ff = positive(split, "expert_shared_feed_forward_length")?;
        let norm = meta_bool(split, "expert_weights_norm")?;
        let mut defaults = Vec::new();
        sigmoid_gating(split, None, &mut defaults)?;
        let scale = match split.value(&split.arch_key("expert_weights_scale")) {
            None => {
                defaults.push("expert_weights_scale = 1 (llama.cpp's default)".into());
                1.0
            }
            Some(_) => finite_positive(split, "expert_weights_scale")?,
        };
        for (l, &count) in kv.iter().enumerate() {
            check_kv(split, heads, l, count)?;
            mul(split, "attention.key_length", count, head)?;
            if l < dense_lead {
                if ff[l] == 0 {
                    return Err(metadata(
                        split,
                        "feed_forward_length",
                        format!("is 0 at dense layer {l}"),
                    ));
                }
            } else {
                if expert_ff[l] == 0 {
                    return Err(metadata(
                        split,
                        "expert_feed_forward_length",
                        format!("is 0 at routed layer {l}"),
                    ));
                }
                let combined = mul(split, "expert_shared_count", expert_ff[l], shared_count)?;
                if combined != shared_ff {
                    return Err(metadata(
                        split,
                        "expert_shared_feed_forward_length",
                        format!(
                            "is {shared_ff}, not expert_feed_forward_length * expert_shared_count {combined} at layer {l}"
                        ),
                    ));
                }
            }
        }
        let hp = Self {
            layers,
            hidden,
            vocab,
            context,
            eps,
            heads,
            kv,
            head,
            ff,
            expert_ff,
            dense_lead,
            experts,
            used,
            shared_ff,
            norm,
            scale,
            window,
            swa,
            rope,
            base,
            defaults,
        };
        let kinds: Vec<bool> = (0..layers).map(|l| l >= dense_lead).collect();
        layers_agree(
            split,
            &kinds,
            |kind| roles::groups(kind).to_vec(),
            &[false, true],
        )?;
        dims_agree(split, hidden, vocab, (0..layers).map(|l| hp.dims(l)))?;
        Ok(hp)
    }

    pub(super) fn dims(&self, l: usize) -> Vec<(&'static str, Vec<u64>)> {
        let (e, d, q, kv) = (
            self.hidden as u64,
            self.head as u64,
            self.heads as u64 * self.head as u64,
            self.kv[l] as u64 * self.head as u64,
        );
        let mut out = vec![
            ("attn_norm.weight", vec![e]),
            ("attn_q.weight", vec![e, q]),
            ("attn_k.weight", vec![e, kv]),
            ("attn_v.weight", vec![e, kv]),
            ("attn_output.weight", vec![q, e]),
            ("attn_q_norm.weight", vec![d]),
            ("attn_k_norm.weight", vec![d]),
            ("attn_sinks.weight", vec![self.heads as u64]),
            ("ffn_norm.weight", vec![e]),
        ];
        if l < self.dense_lead {
            let ff = self.ff[l] as u64;
            out.extend([
                ("ffn_gate.weight", vec![e, ff]),
                ("ffn_up.weight", vec![e, ff]),
                ("ffn_down.weight", vec![ff, e]),
            ]);
        } else {
            let (ff, experts, shared) = (
                self.expert_ff[l] as u64,
                self.experts as u64,
                self.shared_ff as u64,
            );
            out.extend([
                ("ffn_gate_inp.weight", vec![e, experts]),
                (roles::SELECTION_BIAS, vec![experts]),
                ("ffn_gate_exps.weight", vec![e, ff, experts]),
                ("ffn_up_exps.weight", vec![e, ff, experts]),
                ("ffn_down_exps.weight", vec![ff, e, experts]),
                ("ffn_gate_shexp.weight", vec![e, shared]),
                ("ffn_up_shexp.weight", vec![e, shared]),
                ("ffn_down_shexp.weight", vec![shared, e]),
            ]);
        }
        out
    }
}
