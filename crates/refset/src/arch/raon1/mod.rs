//! The Raon-1-mini-preview family. The node dumps are dumped from the Q4_K_M file, [`MODEL`], by KRAFTON's llama.cpp branch
//! `raon-1-mini-preview` at `fb895662d` (a CPU build with no patch, [`LCPP_BUILD`]): `src/models/raon1.cpp` is the vendor's own graph,
//! so the oracle and the public baseline are one engine. `tools/ref/raon1-dump.sh` is the writer (the shared dumper,
//! `tools/ref/cb_dump.cpp`): one set a prompt of `tools/ref/data/raon1`, each a dump_ref-format set (`MANIFEST.tsv`, one f32 file a node
//! and occurrence, an integer node's i32 twin and `int` row) that also holds `topk.tsv` (the last position's 40 largest logits) and
//! `greedy.tsv` (argmax tokens and their margins), read by [`prompt_tables`](super::prompt_tables). No fixture file exists yet, so the
//! family has no fixture tier.

#[cfg(test)]
mod tests;

use crate::RefError;
use crate::family::{Build, Family, Identity};

/// The architecture every raon1 manifest names in its `# arch` line: the GGUF `general.architecture`.
pub const ARCH: &str = "raon1";

/// The Q4_K_M file every raon1 set is dumped from and the engine runs: the identity each set's `# model` line states. A set of the file
/// moved elsewhere is stale by this line, not by its name.
pub const MODEL: &str = "/models/Raon-1-mini-preview/raon-1-mini-preview-Q4_K_M.gguf";

/// The build every raon1 set names: llama.cpp `fb895662d` (krafton-ai/llama.cpp, branch `raon-1-mini-preview`) and the first 12 hex
/// digits of the md5 of the `libllama.so` built from it (`/home/user/llama.cpp-raon`, the tree the profile's `LCPP` names;
/// `tools/ref/raon1-dump-build.sh` records the whole md5 beside the binary and the run refuses a library that moved).
// PIN(2026-10-21): the vendor branch's tip when the family was added; a rebuild of that tree moves this build, and the sets are dumped
// again.
pub const LCPP_BUILD: &str = "fb895662d+libllama-fcd56f0bf834";

/// Prompt a: an English paragraph, every per-layer tap and 32 greedy tokens.
pub const A: &str = "ref_raon1_a";
/// Prompt b: a Korean paragraph, each layer's output and router ids, and 32 greedy tokens.
pub const B: &str = "ref_raon1_b";
/// Prompt c: a Rust function, each layer's output and router ids, and 32 greedy tokens.
pub const C: &str = "ref_raon1_c";
/// Prompt d: the first 1,536 ids of wikitext-2, the last position's logits.
pub const D: &str = "ref_raon1_d";

/// The prompt sets, in the order the dump writes them.
pub const SETS: &[&str] = &[A, B, C, D];

/// [`MODEL`], as a family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// The vendor branch's node dumps, which the Raon-1 gates read once they exist.
pub static LCPP: Family = Family {
    name: "lcpp-raon1",
    sets: SETS,
    resolve: None,
    recipe: "just dump-ref-raon1",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &[],
};

/// The architecture's families, in the order `refset-check` lists them.
pub static FAMILIES: &[&Family] = &[&LCPP];
