//! The role of every tensor in a `muse-spark` file, by exact name. Every
//! layer carries the same 23 stems, all of them required (llama.cpp creates
//! each with no `TENSOR_NOT_REQUIRED`): the attention block with its q/k
//! norms, output gate and residual mix, and the routed latent block with its
//! norms, router, bias and stacks. The residual α/β vectors and the
//! sublayer gains sit with their sublayer (the attention mix on the mixer's
//! card rows, the FFN mix with the pre-FFN norm); the latent pair and the
//! two expert norms, like the router, are matrices of the routed block every
//! token reads on the layer's card, serial with the experts. A name the
//! table does not hold fails the file with every such name listed.

use gguf::Split;

use super::hparams::Hparams;
use super::names::{
    ATTN_GATE, ATTN_K, ATTN_K_NORM, ATTN_NORM, ATTN_OUTPUT, ATTN_Q, ATTN_Q_NORM, ATTN_RESID_ALPHA,
    ATTN_RESID_BETA, ATTN_V, EXP_PROBS_B, FFN_DOWN_EXPS, FFN_GATE_EXPS, FFN_GATE_INP,
    FFN_LATENT_DOWN, FFN_LATENT_UP, FFN_NORM, FFN_POST_EXP_NORM, FFN_PRE_EXP_NORM, FFN_RESID_ALPHA,
    FFN_RESID_BETA, FFN_UP_EXPS, POST_FFW_NORM,
};
use crate::arch::classify::{Counts, classify_groups};
use crate::arch::stems::{Stem, req};
use crate::placement::{ModelTensors, PlacementError, Role};

/// Every layer's stems, in the file's own order.
const STEMS: &[Stem] = &[
    req(ATTN_NORM, Role::Attention),
    req(ATTN_Q, Role::Attention),
    req(ATTN_K, Role::Attention),
    req(ATTN_V, Role::Attention),
    req(ATTN_Q_NORM, Role::Attention),
    req(ATTN_K_NORM, Role::Attention),
    req(ATTN_GATE, Role::Attention),
    req(ATTN_OUTPUT, Role::Attention),
    req(ATTN_RESID_ALPHA, Role::Attention),
    req(ATTN_RESID_BETA, Role::Attention),
    req(FFN_NORM, Role::FfnNorm),
    req(FFN_LATENT_DOWN, Role::Router),
    req(FFN_PRE_EXP_NORM, Role::Router),
    req(FFN_GATE_INP, Role::Router),
    req(EXP_PROBS_B, Role::Router),
    req(FFN_GATE_EXPS, Role::RoutedExperts),
    req(FFN_UP_EXPS, Role::RoutedExperts),
    req(FFN_DOWN_EXPS, Role::RoutedExperts),
    req(FFN_POST_EXP_NORM, Role::Router),
    req(FFN_LATENT_UP, Role::Router),
    req(POST_FFW_NORM, Role::FfnNorm),
    req(FFN_RESID_ALPHA, Role::FfnNorm),
    req(FFN_RESID_BETA, Role::FfnNorm),
];

/// Every tensor of `split` with its role and header facts.
pub fn classify(split: &Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    classify_groups(
        split,
        &[STEMS],
        Counts::new(hp.n_layer, hp.n_expert, hp.n_used),
    )
}

#[cfg(test)]
mod tests {
    use super::{STEMS, Stem};

    /// No stem is listed twice. Mutant: a stem repeated in the table.
    #[test]
    fn the_table_lists_each_stem_once() {
        let all: Vec<&Stem> = STEMS.iter().collect();
        assert_eq!(all.len(), 23);
        for (i, s) in all.iter().enumerate() {
            assert!(
                !all[..i].iter().any(|t| t.name == s.name),
                "{} twice",
                s.name
            );
        }
    }
}
