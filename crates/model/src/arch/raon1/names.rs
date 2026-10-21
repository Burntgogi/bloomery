//! Every tensor name a Raon-1 step reads: the spellings every family takes from
//! [`crate::arch::names`], the delta rule's, the latent layer's and the hyper-connections' among
//! them, and the two gates of this family.

use crate::arch::names::blk;

pub use crate::arch::names::{
    Sub, attn_k, attn_k_b, attn_kv_a_mqa, attn_kv_a_norm, attn_norm, attn_output, attn_q, attn_q_a,
    attn_q_a_norm, attn_q_b, attn_v, attn_v_b, exp_probs_b, ffn_down, ffn_down_exps,
    ffn_down_shexp, ffn_gate, ffn_gate_exps, ffn_gate_inp, ffn_gate_shexp, ffn_norm, ffn_up,
    ffn_up_exps, ffn_up_shexp, hc_base, hc_fn, hc_scale, output, output_norm, ssm_a, ssm_beta,
    ssm_conv1d, ssm_dt_bias, ssm_f_a, ssm_f_b, ssm_norm, token_embd,
};

/// `blk.{l}.attn_gate.weight` — the latent layer's output gate projection, one value for each
/// value the heads write.
#[must_use]
pub fn attn_gate(l: usize) -> String {
    blk(l, "attn_gate.weight")
}

/// `blk.{l}.ssm_g.weight` — the delta layer's output gate projection, one value for each channel.
#[must_use]
pub fn ssm_g(l: usize) -> String {
    blk(l, "ssm_g.weight")
}

/// `blk.{l}.ssm_rda.weight` — the delta layer's write gate offset, one value for each head.
#[must_use]
pub fn ssm_rda(l: usize) -> String {
    blk(l, "ssm_rda.weight")
}
