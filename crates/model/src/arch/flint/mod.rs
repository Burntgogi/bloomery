//! Flint's header description and exact tensor roles. Full and sliding attention
//! independently select NoPE or NeoX rotation; every layer has head-local q/k
//! norms and attention sinks. Routed blocks carry one ungated shared MLP whose
//! width is explicit, not a count of physical modules.
//!
//! This is a reader, not an executable program. Coverage retains `Need::Program`
//! and the missing numerical/site bindings; no fixture or serve seat is registered.

mod hparams;
mod roles;
pub mod spec;

#[cfg(test)]
mod tests;
