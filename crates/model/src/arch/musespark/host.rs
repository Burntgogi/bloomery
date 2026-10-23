//! The host tier's view of a muse-spark layer's routed experts: the three
//! stacks, the expert count and the widths a [`HostLayer`] serves by, read in
//! the latent the experts run in (the layer's input projected down to
//! `moe_latent_size` before them, so the experts are `latent -> ff -> latent`),
//! at the plain SwiGLU combine — the file's SiLU-gated experts carry no clamp.
//! The one place that wires them into a [`HostLayerSpec`], and the one reader
//! of the layer's post-expert gain: the host leg applies it to each expert's
//! own output row before the router weight does (the gpu crate's
//! `host::run::HostRun::set_post_norm`).

use gguf::{GgmlType, Split};

use super::hparams::Hparams;
use super::names::FFN_POST_EXP_NORM;
use crate::ModelError;
use crate::arch::names::{blk, routed_stacks};
use crate::moe::{HostLayer, HostLayerSpec};
use crate::r8file::R8Source;

/// Layer `l`'s routed experts as the host tier serves them, built from `src`
/// ([`HostLayer::build`]): all of the layer's `n_expert` experts, in the
/// latent's width, at the layer's own expert width. A layer past the model is
/// refused by name. Load-time only: the stacks are checked here, each with a
/// fused kernel at its row width.
pub fn layer(src: R8Source<'_>, hp: &Hparams, l: usize) -> Result<HostLayer, ModelError> {
    let Some(&ff) = hp.expert_ff.get(l) else {
        return Err(ModelError::MissingTensor(format!(
            "layer {l}'s routed stacks: a layer of the host run past the model's {}",
            hp.n_layer
        )));
    };
    let [gate, up, down] = routed_stacks(l);
    HostLayer::build(
        src,
        &HostLayerSpec {
            gate: &gate,
            up: &up,
            down: &down,
            n_expert: hp.n_expert,
            embd: hp.latent,
            ff,
            swiglu_limit: 0.0,
        },
    )
}

/// The host views of the routed layers `set` (strictly ascending model
/// layers that carry a host share), one per layer in the set's order, each
/// through [`layer`].
pub fn routed_layers(
    src: R8Source<'_>,
    hp: &Hparams,
    set: &[usize],
) -> Result<Vec<HostLayer>, ModelError> {
    set.iter().map(|&l| layer(src, hp, l)).collect()
}

/// Layer `l`'s post-expert gain, `moe_latent_size` f32 values, read from the
/// file: the weight the host leg scales each routed expert's output row by
/// before the row's weightless RMS norm. Refused by name: a layer without the
/// tensor, one that is not F32 of the latent's width.
pub fn post_exp_gain(split: &Split, hp: &Hparams, l: usize) -> Result<Vec<f32>, ModelError> {
    let name = blk(l, FFN_POST_EXP_NORM);
    let Some((_, t)) = split.find(&name) else {
        return Err(ModelError::MissingTensor(name));
    };
    if t.ty != GgmlType::F32 || t.dims != [hp.latent as u64] {
        return Err(ModelError::Metadata {
            key: name,
            detail: format!(
                "is {} {:?}, the post-expert gain is F32 of the latent's {}",
                t.ty, t.dims, hp.latent
            ),
        });
    }
    let bytes = split.data(&name)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}
