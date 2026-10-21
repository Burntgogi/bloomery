//! The role tables' row, shared by the families whose tables name their
//! tensors by stem (`glm5next`, `mimo2`): what a layer of some kind carries,
//! and which of it the file must carry. The tables themselves stay in each
//! family's `roles` — the GGUF names are its architecture's strings — as
//! `classify_with`'s contract says.

use crate::placement::Role;

/// A tensor a layer of some kind carries, by stem: its role, and whether
/// the file must carry it (llama.cpp creates it without
/// `TENSOR_NOT_REQUIRED`).
#[derive(Clone, Copy, Debug)]
pub struct Stem {
    pub(crate) name: &'static str,
    pub(crate) role: Role,
    pub(crate) required: bool,
}

/// A stem the file must carry.
pub(crate) const fn req(name: &'static str, role: Role) -> Stem {
    Stem {
        name,
        role,
        required: true,
    }
}

/// A stem the file may carry.
pub(crate) const fn opt(name: &'static str, role: Role) -> Stem {
    Stem {
        name,
        role,
        required: false,
    }
}

/// The names of `stems` the file must carry.
pub fn required(stems: &[Stem]) -> impl Iterator<Item = &'static str> + '_ {
    stems.iter().filter(|s| s.required).map(|s| s.name)
}

/// The stems of `parts` in order as one table of `N`; `N` is their count, held by assertion.
pub(crate) const fn join<const N: usize>(parts: &[&[Stem]]) -> [Stem; N] {
    let mut out = [parts[0][0]; N];
    let (mut n, mut p) = (0, 0);
    while p < parts.len() {
        let mut i = 0;
        while i < parts[p].len() {
            out[n] = parts[p][i];
            n += 1;
            i += 1;
        }
        p += 1;
    }
    assert!(n == N, "join: the parts' stems are not N");
    out
}

/// The hyper-connection tensors a trunk layer carries: the mix of each sublayer's input and
/// output from the streams, which every family of hyper-connections names alike.
pub const HC: &[Stem] = &[
    req("hc_attn_fn.weight", Role::HyperConnection),
    req("hc_attn_base.weight", Role::HyperConnection),
    req("hc_attn_scale.weight", Role::HyperConnection),
    req("hc_ffn_fn.weight", Role::HyperConnection),
    req("hc_ffn_base.weight", Role::HyperConnection),
    req("hc_ffn_scale.weight", Role::HyperConnection),
];
