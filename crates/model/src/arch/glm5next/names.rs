//! Every tensor name the glm5next step reads, in one place: the model-level
//! names, then a layer's `blk.{N}.*` names by the sub-layer that reads them.
//! Which layers carry which is the layer's [`LayerSpec`](models::LayerSpec),
//! read by [`super::spec`]; the roles table ([`super::roles`]) holds the same
//! stems for the classifier.

use crate::arch::names::blk;

pub use crate::arch::names::{
    Sub, attn_k_b, attn_kv_a_mqa, attn_kv_a_norm, attn_q_a, attn_q_a_norm, attn_q_b, attn_v_b,
    ffn_down_shexp, ffn_gate_shexp, ffn_up_shexp, hc_base, hc_fn, hc_scale, indexer_k_norm, output,
    output_norm, ssm_a, ssm_beta, ssm_conv1d, ssm_dt_bias, ssm_f_a, ssm_f_b, ssm_g_a, ssm_g_b,
    ssm_norm, token_embd,
};

/// The mixer's RMS gain, both kinds.
pub fn attn_norm(l: usize) -> String {
    blk(l, "attn_norm.weight")
}

/// The mixer's output projection, both kinds.
pub fn attn_output(l: usize) -> String {
    blk(l, "attn_output.weight")
}

// ---------------------------------------------------------------- KDA

pub fn attn_q(l: usize) -> String {
    blk(l, "attn_q.weight")
}

pub fn attn_k(l: usize) -> String {
    blk(l, "attn_k.weight")
}

pub fn attn_v(l: usize) -> String {
    blk(l, "attn_v.weight")
}

/// The q, k and v projections joined at load into one row stream, the conv's
/// channel order.
pub fn attn_qkv(l: usize) -> String {
    format!("derived.blk.{l}.attn_qkv")
}

/// The three conv tap sets joined at load in the q, k, v channel order.
pub fn ssm_conv1d_qkv(l: usize) -> String {
    format!("derived.blk.{l}.ssm_conv1d_qkv")
}

// ------------------------------------------------------------- latent

pub fn indexer_attn_k(l: usize) -> String {
    blk(l, "indexer.attn_k.weight")
}

pub fn indexer_k_norm_bias(l: usize) -> String {
    blk(l, "indexer.k_norm.bias")
}

pub fn indexer_compressor_gate(l: usize) -> String {
    blk(l, "indexer_compressor_gate.weight")
}

pub fn indexer_attn_q_b(l: usize) -> String {
    blk(l, "indexer.attn_q_b.weight")
}

pub fn indexer_proj(l: usize) -> String {
    blk(l, "indexer.proj.weight")
}

pub fn indexer_compressor_ape(l: usize) -> String {
    blk(l, "indexer_compressor_ape.weight")
}

/// The four projections of the normed input joined at load into one row
/// stream: `[q_a; latent; index key; pool gate]` per token.
pub fn attn_a_stack(l: usize) -> String {
    format!("derived.blk.{l}.attn_a_stack")
}

// -------------------------------------------------------- feed-forward

pub fn ffn_norm(l: usize) -> String {
    blk(l, "ffn_norm.weight")
}

pub fn ffn_gate(l: usize) -> String {
    blk(l, "ffn_gate.weight")
}

pub fn ffn_up(l: usize) -> String {
    blk(l, "ffn_up.weight")
}

pub fn ffn_down(l: usize) -> String {
    blk(l, "ffn_down.weight")
}

pub fn ffn_gate_inp(l: usize) -> String {
    blk(l, "ffn_gate_inp.weight")
}

pub fn exp_probs_b(l: usize) -> String {
    blk(l, "exp_probs_b.bias")
}

pub fn ffn_gate_exps(l: usize) -> String {
    blk(l, "ffn_gate_exps.weight")
}

pub fn ffn_up_exps(l: usize) -> String {
    blk(l, "ffn_up_exps.weight")
}

pub fn ffn_down_exps(l: usize) -> String {
    blk(l, "ffn_down_exps.weight")
}

// ---------------------------------------------------- hyper-connections

// ------------------------------------------------------- the next-token layer

/// The next-token layer's input projection of `[enorm(e); hnorm(h)]`.
pub fn nextn_eh_proj(l: usize) -> String {
    blk(l, "nextn.eh_proj.weight")
}

/// The next-token layer's RMS gain of the token's embedding row.
pub fn nextn_enorm(l: usize) -> String {
    blk(l, "nextn.enorm.weight")
}

/// The next-token layer's RMS gain of the target's hidden row.
pub fn nextn_hnorm(l: usize) -> String {
    blk(l, "nextn.hnorm.weight")
}

/// The next-token layer's RMS gain before the shared head.
pub fn nextn_shared_head_norm(l: usize) -> String {
    blk(l, "nextn.shared_head_norm.weight")
}
