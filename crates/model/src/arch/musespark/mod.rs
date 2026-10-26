//! `muse-spark`: Muse Spark Open, a text decoder of one repeated layer. Every
//! layer is grouped-query attention — the sliding layers rope in NORM pairs
//! at the sliding base inside their window and the full layers are NoPE, all
//! of them with a per-head q/k norm (a constant gain synthesised at
//! conversion) and an output gated by a projection of the layer input — over
//! a routed mixture of SiLU-gated experts in a latent, with no dense layer.
//! The file's keys (`hparams`), every tensor's role (`roles`), the tensor
//! names (`names`), the typed description (`spec`) and the gate fixture
//! (`fixture`: a requantized mini of the real file). The card body
//! (`gpu/src/arch/musespark`) runs it, whole-card or placed: the plan is the
//! common hybrid owner's over this family's reading (`place`), and the experts
//! the plan leaves on the host are the host tier's (`host`). The dialect's authority is
//! llama.cpp's `src/models/muse-spark.cpp`.

pub mod fixture;
pub mod host;
pub mod hparams;
pub mod names;
pub mod place;
pub mod roles;
pub mod spec;
