//! The MiniMax-M3.1-Flash family of the fixture tier: transformers' dump of the cut folder (`fixture cut`, its first shard
//! [`crate::fixture::first_shard`]), mirroring [`super::TF`] with the same writer, the same consumers and set names that carry
//! the `fx_` prefix. The identity is the fixture's: its first shard and its `# fixture` line
//! ([`crate::family::Identity::FixtureManifest`]). The cut's set e under the vendor's rule is dumped with the others and read by
//! no family, as [`super::POOLED`]'s real one is read by no gate.

use super::{ARCH, BUILD};
use crate::RefError;
use crate::family::{Build, Family, Identity};

/// [`super::A`] on the fixture.
pub const A: &str = "fx_ref_minimaxm31_a";
/// [`super::B`] on the fixture.
pub const B: &str = "fx_ref_minimaxm31_b";
/// [`super::C`] on the fixture.
pub const C: &str = "fx_ref_minimaxm31_c";
/// [`super::D`] on the fixture.
pub const D: &str = "fx_ref_minimaxm31_d";
/// [`super::E_PERHEAD`] on the fixture.
pub const E_PERHEAD: &str = "fx_ref_minimaxm31_e_perhead";

/// The fixture cut the tree runs, as a family's `runs`.
fn runs() -> Result<String, RefError> {
    crate::fixture::first_shard(ARCH)
}

/// [`super::TF`] on the fixture.
pub static TF: Family = Family {
    name: "fx-tf-minimaxm31",
    sets: &[A, B, C, D, E_PERHEAD],
    resolve: None,
    recipe: "BLOOMERY_TIER=fixture just dump-ref-minimaxm31 all",
    identity: Identity::FixtureManifest,
    arch: Some(ARCH),
    build: Some(Build::Is(BUILD)),
    runs: Some(runs),
    draft_runs: None,
    consumers: super::TF.consumers,
};
