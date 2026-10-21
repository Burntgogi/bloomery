//! `raon1` — Raon-1-mini on one card: the chain ([`body`]), Raon_mHC's
//! streams ([`mhc`]), the SiTU feed-forward blocks and the 128/6 router
//! ([`ffn`]), the RDA layer ([`rda`]) and the gated latent layer ([`mla`]).
//! The model description is `model::arch::raon1`.

mod body;
mod ffn;
mod mhc;
pub mod mla;
pub mod rda;

pub use body::Body;
pub use ffn::sigmoid_128_6;
pub use mhc::{HC_OUT, STREAMS};
