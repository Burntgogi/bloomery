//! Layer groups shared by the tensor-set check, shape check and classification.

pub(super) use crate::arch::stems::{ATTENTION, DENSE, MOE, SELECTION_BIAS, SHARED};
use crate::arch::stems::{Stem, req};
use crate::placement::Role;

pub(super) const SINKS: &[Stem] = &[req("attn_sinks.weight", Role::Attention)];

const DENSE_GROUPS: &[&[Stem]] = &[ATTENTION, SINKS, DENSE];
const ROUTED_GROUPS: &[&[Stem]] = &[ATTENTION, SINKS, MOE, SHARED];

pub(super) fn groups(routed: bool) -> &'static [&'static [Stem]] {
    if routed { ROUTED_GROUPS } else { DENSE_GROUPS }
}
