//! Every `muse-spark.*` key the description reads. A layer is grouped-query
//! attention of one head width over a routed latent mixture of experts —
//! every layer routes, there is no dense layer — and the file's own
//! `attention.sliding_window_pattern` array decides, layer by layer, which
//! layers rope: a sliding layer ropes at `rope.freq_base_swa` inside the
//! window and a full layer does not rope at all. The array is read as the
//! file spells it, never reconstructed from a rule, and its length is the
//! block count.
//!
//! Two keys the header carries are not what the graph reads
//! (`muse-spark.feed_forward_length`, `muse-spark.rope.freq_base`: the graph
//! takes the sliding base), so the reader leaves them unread
//! ([`unread_keys`]) rather than taking them as facts.

use gguf::Split;

use super::names::{
    ATTN_GATE, ATTN_K, ATTN_K_NORM, ATTN_NORM, ATTN_OUTPUT, ATTN_Q, ATTN_Q_NORM, ATTN_RESID_ALPHA,
    ATTN_RESID_BETA, ATTN_V, EXP_PROBS_B, FFN_DOWN_EXPS, FFN_GATE_EXPS, FFN_GATE_INP,
    FFN_LATENT_DOWN, FFN_LATENT_UP, FFN_NORM, FFN_POST_EXP_NORM, FFN_PRE_EXP_NORM, FFN_RESID_ALPHA,
    FFN_RESID_BETA, FFN_UP_EXPS, POST_FFW_NORM,
};
use crate::arch::{
    check_kv, expert_counts, finite_positive, meta_bool, metadata, n_vocab, one_expert_group,
    optional_usize, per_layer_count, per_layer_items, positive, sigmoid_gating, unread_keys_of,
};

use crate::placement::PlacementError;

/// The hyperparameters of one `muse-spark` file.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// `block_count`.
    pub n_layer: usize,
    /// `embedding_length`.
    pub n_embd: usize,
    /// `context_length`.
    pub n_ctx: usize,
    /// `vocab_size`, else the token list's length.
    pub n_vocab: usize,
    /// `attention.layer_norm_rms_epsilon`.
    pub rms_eps: f32,
    /// `post_norm_epsilon`: the weightless post-sublayer norms' own epsilon.
    pub post_eps: f32,
    /// `attention.head_count`.
    pub n_head: usize,
    /// `attention.head_count_kv`.
    pub n_head_kv: usize,
    /// `attention.key_length`, the value length's too.
    pub head_dim: usize,
    /// `rope.dimension_count`, the head when the file carries no key.
    pub rope_dims: usize,
    /// `rope.freq_base_swa`: the sliding layers' base.
    pub rope_base: f32,
    /// `attention.sliding_window`.
    pub window: usize,
    /// `attention.sliding_window_pattern`: whether each layer slides.
    pub sliding: Vec<bool>,
    /// `expert_count`.
    pub n_expert: usize,
    /// `expert_used_count`.
    pub n_used: usize,
    /// `expert_feed_forward_length` at each layer: one expert's values.
    pub expert_ff: Vec<usize>,
    /// `moe_latent_size`: the values the routed experts read and write.
    pub latent: usize,
    /// `expert_weights_norm`: the kept weights renormalized to sum to one.
    pub weights_norm: bool,
    /// `expert_weights_scale`: the factor on the kept weights.
    pub weights_scale: f32,
    /// `logit_scale`: the factor on the head's logits.
    pub logit_scale: f32,
    /// `final_logit_softcapping`: the tanh softcap's bound.
    pub softcap: f32,
    /// The key defaults taken, each with its value and why.
    pub defaults: Vec<String>,
}

/// Every `muse-spark.` key [`Hparams::read`] reads, by suffix.
const READ: &[&str] = &[
    "block_count",
    "context_length",
    "embedding_length",
    "vocab_size",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.layer_norm_rms_epsilon",
    "attention.sliding_window",
    "attention.sliding_window_pattern",
    "rope.dimension_count",
    "rope.freq_base_swa",
    "post_norm_epsilon",
    "logit_scale",
    "final_logit_softcapping",
    "expert_count",
    "expert_used_count",
    "expert_feed_forward_length",
    "expert_gating_func",
    "expert_weights_norm",
    "expert_weights_scale",
    "moe_latent_size",
];

/// The architecture keys and `general.sampling.*` keys of `split` that
/// [`Hparams::read`] does not read: listed, not refused.
#[must_use]
pub fn unread_keys(split: &Split) -> Vec<String> {
    unread_keys_of(split, READ)
}

impl Hparams {
    /// Every value from `split`'s headers. A missing key, a key of a value
    /// the model cannot hold (another gating func, a nonpositive softcap, a
    /// value length other than the key length, a pattern of another length),
    /// or a tensor missing or of other dims than the keys give is an error
    /// naming the key or the tensor.
    pub fn read(split: &Split) -> Result<Hparams, PlacementError> {
        let n_layer = positive(split, "block_count")?;
        let n_embd = positive(split, "embedding_length")?;
        let n_ctx = positive(split, "context_length")?;
        let rms_eps = finite_positive(split, "attention.layer_norm_rms_epsilon")?;
        // The weightless post-sublayer norms run at their own epsilon, not
        // the block norm's (muse-spark.cpp reads `post_norm_epsilon`).
        let post_eps = finite_positive(split, "post_norm_epsilon")?;
        let n_head = positive(split, "attention.head_count")?;
        let n_head_kv = positive(split, "attention.head_count_kv")?;
        check_kv(split, n_head, 0, n_head_kv)?;
        let head_dim = positive(split, "attention.key_length")?;
        let value_len = positive(split, "attention.value_length")?;
        if value_len != head_dim {
            return Err(metadata(
                split,
                "attention.value_length",
                format!("is {value_len}, the key length {head_dim}; one width serves q, k and v"),
            ));
        }
        // The file carries no rope width key: the graph takes the head
        // (llama-hparams.cpp), recorded as the default it is.
        let mut defaults = Vec::new();
        let rope_dims = match optional_usize(split, "rope.dimension_count")? {
            Some(v) => v,
            None => {
                defaults.push(format!(
                    "rope.dimension_count = {head_dim} (the head, llama-hparams.cpp)"
                ));
                head_dim
            }
        };
        if rope_dims > head_dim || rope_dims % 2 != 0 {
            return Err(metadata(
                split,
                "rope.dimension_count",
                format!("is {rope_dims}, for a head of {head_dim}"),
            ));
        }
        let (n_expert, n_used) = expert_counts(split)?;
        // Every layer routes, so no layer's expert width is 0.
        let expert_ff = per_layer_count(split, "expert_feed_forward_length", n_layer)?;
        if let Some(l) = expert_ff.iter().position(|&ff| ff == 0) {
            return Err(metadata(
                split,
                "expert_feed_forward_length",
                format!("is 0 at layer {l}; every layer routes"),
            ));
        }
        // One width serves the step's launches and the plan's sizing, so a
        // file whose layers differ is refused rather than run at layer 0's.
        if let Some(l) = expert_ff.iter().position(|&ff| ff != expert_ff[0]) {
            return Err(metadata(
                split,
                "expert_feed_forward_length",
                format!(
                    "is {} at layer {l}, {} at layer 0; one expert width serves every layer",
                    expert_ff[l], expert_ff[0]
                ),
            ));
        }
        let latent = positive(split, "moe_latent_size")?;
        // The dialect builds the sigmoid router (muse-spark.cpp reads the key
        // and passes it to build_moe_ffn's sigmoid arm), so a file that says
        // another is refused; the key is the file's own, no default.
        sigmoid_gating(split, None, &mut defaults)?;
        let weights_norm = meta_bool(split, "expert_weights_norm")?;
        let weights_scale = finite_positive(split, "expert_weights_scale")?;
        let logit_scale = finite_positive(split, "logit_scale")?;
        let softcap = finite_positive(split, "final_logit_softcapping")?;
        one_expert_group(split, "the muse-spark router")?;
        let sliding = sliding(split, n_layer)?;
        let hp = Hparams {
            n_layer,
            n_embd,
            n_ctx,
            n_vocab: n_vocab(split)?,
            rms_eps,
            post_eps,
            n_head,
            n_head_kv,
            head_dim,
            rope_dims,
            rope_base: finite_positive(split, "rope.freq_base_swa")?,
            window: positive(split, "attention.sliding_window")?,
            sliding,
            n_expert,
            n_used,
            expert_ff,
            latent,
            weights_norm,
            weights_scale,
            logit_scale,
            softcap,
            defaults,
        };
        // Every tensor the keys name, present at its dims: a missing one is
        // refused by name with the count of the others.
        let (e, v) = (hp.n_embd as u64, hp.n_vocab as u64);
        let mut want = vec![
            (crate::arch::names::token_embd(), vec![e, v]),
            (crate::arch::names::output(), vec![e, v]),
            (crate::arch::names::output_norm(), vec![e]),
        ];
        for l in 0..hp.n_layer {
            want.extend(
                hp.want(l)
                    .into_iter()
                    .map(|(stem, dims)| (crate::arch::names::blk(l, stem), dims)),
            );
        }
        crate::arch::tensors_agree(split, want)?;
        Ok(hp)
    }

    /// One routed expert's width, the same at every layer ([`Hparams::read`]
    /// refuses a file whose layers differ). `expert_ff` holds one entry per
    /// block and `block_count` is positive.
    #[must_use]
    pub fn expert_width(&self) -> usize {
        self.expert_ff[0]
    }

    /// The dims the keys give layer `l`, ggml order. Every layer carries the
    /// same stems.
    fn want(&self, l: usize) -> Vec<(&'static str, Vec<u64>)> {
        let _ = l;
        let e = self.n_embd as u64;
        let d = self.head_dim as u64;
        let q = d * self.n_head as u64;
        let kv = d * self.n_head_kv as u64;
        let x = self.n_expert as u64;
        let w = self.expert_ff[l] as u64;
        let lat = self.latent as u64;
        vec![
            (ATTN_NORM, vec![e]),
            (ATTN_Q, vec![e, q]),
            (ATTN_K, vec![e, kv]),
            (ATTN_V, vec![e, kv]),
            (ATTN_Q_NORM, vec![d]),
            (ATTN_K_NORM, vec![d]),
            (ATTN_GATE, vec![e, q]),
            (ATTN_OUTPUT, vec![q, e]),
            (ATTN_RESID_ALPHA, vec![e]),
            (ATTN_RESID_BETA, vec![e]),
            (FFN_NORM, vec![e]),
            (FFN_LATENT_DOWN, vec![e, lat]),
            (FFN_PRE_EXP_NORM, vec![lat]),
            (FFN_GATE_INP, vec![e, x]),
            (EXP_PROBS_B, vec![x]),
            (FFN_GATE_EXPS, vec![lat, w, x]),
            (FFN_UP_EXPS, vec![lat, w, x]),
            (FFN_DOWN_EXPS, vec![w, lat, x]),
            (FFN_POST_EXP_NORM, vec![lat]),
            (FFN_LATENT_UP, vec![lat, e]),
            (POST_FFW_NORM, vec![e]),
            (FFN_RESID_ALPHA, vec![e]),
            (FFN_RESID_BETA, vec![e]),
        ]
    }
}

/// `attention.sliding_window_pattern` as one bool per layer. A scalar, a
/// length other than `n_layer` or a value that is no bool is refused by
/// name: the file's own form is the per-layer array, and its entries decide
/// which layers rope.
fn sliding(split: &Split, n_layer: usize) -> Result<Vec<bool>, PlacementError> {
    let key = "attention.sliding_window_pattern";
    per_layer_items(split, key, n_layer)?
        .iter()
        .enumerate()
        .map(|(l, v)| {
            v.as_bool()
                .ok_or_else(|| metadata(split, key, format!("has no bool at layer {l}")))
        })
        .collect()
}
