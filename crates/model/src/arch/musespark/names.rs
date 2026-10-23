//! Every tensor name a `muse-spark` file spells. The model-level names are
//! the common ones, and every layer carries the same stems as
//! `blk.{l}.{stem}` (llama.cpp creates each with no `TENSOR_NOT_REQUIRED`).

pub use crate::arch::names::{output, output_norm, token_embd};

/// The pre-attention norm's gain.
pub const ATTN_NORM: &str = "attn_norm.weight";
/// The query projection.
pub const ATTN_Q: &str = "attn_q.weight";
/// The key projection.
pub const ATTN_K: &str = "attn_k.weight";
/// The value projection.
pub const ATTN_V: &str = "attn_v.weight";
/// The per-head RMS gain on the query before the rope: the MetaP scale,
/// synthesised at conversion (ones scaled by `qk_scale_factor`).
pub const ATTN_Q_NORM: &str = "attn_q_norm.weight";
/// The per-head RMS gain on the key before the rope: ones at conversion.
pub const ATTN_K_NORM: &str = "attn_k_norm.weight";
/// The attention output gate: a projection of the layer input of its own,
/// the output multiplied by its sigmoid.
pub const ATTN_GATE: &str = "attn_gate.weight";
/// The attention output projection.
pub const ATTN_OUTPUT: &str = "attn_output.weight";
/// The residual mix after the attention sublayer: `alpha` scales the stream.
pub const ATTN_RESID_ALPHA: &str = "attn_resid_alpha.weight";
/// The residual mix after the attention sublayer: `beta` scales the branch.
pub const ATTN_RESID_BETA: &str = "attn_resid_beta.weight";
/// The pre-FFN norm's gain.
pub const FFN_NORM: &str = "ffn_norm.weight";
/// The projection of the normed row down to the latent the experts read.
pub const FFN_LATENT_DOWN: &str = "ffn_latent_down.weight";
/// The projection of the experts' weighted sum back up to the hidden row.
pub const FFN_LATENT_UP: &str = "ffn_latent_up.weight";
/// The RMS gain on the latent row before the experts.
pub const FFN_PRE_EXP_NORM: &str = "ffn_pre_exp_norm.weight";
/// The RMS gain on the experts' weighted sum, scaled then re-normalised
/// before the router weights apply.
pub const FFN_POST_EXP_NORM: &str = "ffn_post_exp_norm.weight";
/// The router: one score per expert, from the full stream (not the latent),
/// pre-scaled by the MetaP output multiplier at conversion.
pub const FFN_GATE_INP: &str = "ffn_gate_inp.weight";
/// The router's selection bias, added to the scores for the pick only.
pub const EXP_PROBS_B: &str = "exp_probs_b.bias";
/// The routed experts' gate stack.
pub const FFN_GATE_EXPS: &str = "ffn_gate_exps.weight";
/// The routed experts' up stack.
pub const FFN_UP_EXPS: &str = "ffn_up_exps.weight";
/// The routed experts' down stack.
pub const FFN_DOWN_EXPS: &str = "ffn_down_exps.weight";
/// The RMS gain on the FFN sublayer's output, scaled then re-normalised
/// (llama.cpp's `post_ffw_norm`).
pub const POST_FFW_NORM: &str = "post_ffw_norm.weight";
/// The residual mix after the FFN sublayer: `alpha` scales the stream.
pub const FFN_RESID_ALPHA: &str = "ffn_resid_alpha.weight";
/// The residual mix after the FFN sublayer: `beta` scales the branch.
pub const FFN_RESID_BETA: &str = "ffn_resid_beta.weight";
