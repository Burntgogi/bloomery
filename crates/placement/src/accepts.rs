//! What each layer program runs, by site and type: the one owner the plan's
//! rules and the loaders both read, so a plan its rule passes is a plan the
//! load takes. The card kernel table is [`kernels`]; a program's sets here
//! are the entries of it that its dispatch launches. The coverage check
//! (`model::arch::coverage`) names its items through these.

use gguf::GgmlType;
use models::Arch;

use crate::kernels::{self, DownEntry, GateUpEntry};

/// A layer program: the chain that runs a family's layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Program {
    /// The V4.1 chain (`crates/gpu-deepseek41`: `chain/`, `body.rs`).
    Deepseek41Chain,
    /// The whole-card qwen3moe body (`crates/gpu/src/arch/qwen3moe` `Body`).
    Qwen3moeBody,
    /// The whole-card qwen35moe and qwen35 body (`crates/gpu/src/arch/qwen3moe`
    /// `Body35`): gated-delta-rule and gated GQA layers, its sites' types by
    /// `crate::site`.
    Qwen35Body,
    /// The glm5next body (`crates/gpu-glm5next`): the routed experts the plan
    /// puts on a card (the common pair) and the rest on the host tier.
    Glm5nextBody,
    /// The qwen4exp body (`crates/gpu/src/arch/qwen3moe` `Body38`): the
    /// routed experts the plan puts on a card (its own gate·up and down
    /// entries) and the rest on the host tier.
    Qwen38Body,
    /// The mimo2 body (`crates/gpu-mimo2`): the routed experts the plan puts
    /// on a card (MXFP4 stacks, the common pair) and the rest on the host
    /// tier.
    Mimo2Body,
}

/// The program that runs `arch`'s layers; `None` when none does.
#[must_use]
pub fn program_of(arch: Arch) -> Option<Program> {
    match arch {
        Arch::Deepseek41 | Arch::Deepseek4 => Some(Program::Deepseek41Chain),
        Arch::Qwen3Moe => Some(Program::Qwen3moeBody),
        Arch::Qwen35Moe | Arch::Qwen35 => Some(Program::Qwen35Body),
        Arch::Glm5Next => Some(Program::Glm5nextBody),
        Arch::Qwen4Exp => Some(Program::Qwen38Body),
        Arch::MiMo2 => Some(Program::Mimo2Body),
    }
}

/// The card gate·up entry `Body35` runs a routed stack of `ty` with: the
/// card kernel table's row ([`kernels::gate_up`], the decode `_sel`) when
/// the table names the type's prompt GEMM too ([`kernels::gemm`]), among
/// the entries the family's FFN dispatch launches
/// (`gpu/src/arch/qwen3moe/dispatch.rs`): the K-quant family's Q4_K and
/// Q5_K. The table's Q8_0 row reads a stack in the file's block stream, its
/// i-quant rows are `iq_sel.rs`'s and its MXFP4 row is `mxfp4_sel.rs`'s; the
/// dispatch launches none of them, so
/// such a stack is refused by name.
#[must_use]
pub const fn qwen35_gate_up(ty: GgmlType) -> Option<GateUpEntry> {
    let (Some(entry), Some(_)) = (kernels::gate_up(ty), kernels::gemm(ty)) else {
        return None;
    };
    match entry {
        GateUpEntry::Q4k | GateUpEntry::Q5k => Some(entry),
        GateUpEntry::Q8_0
        | GateUpEntry::Iq2xs
        | GateUpEntry::Iq3xxs
        | GateUpEntry::Iq3s
        | GateUpEntry::Iq4xs
        | GateUpEntry::Mxfp4 => None,
    }
}

/// The card down entry `Body35` runs a routed stack of `ty` with: the
/// table's row ([`kernels::down`]) when it names the type's prompt GEMM too
/// ([`kernels::gemm`]), among the entries the FFN dispatch launches: the
/// downs over `Q8Act`, Q4_K, Q5_K and Q6_K. The downs over `Q8Blocks32`
/// (Q5_0, Q5_1, Q8_0, IQ4_NL) read an activation form the dispatch does not
/// quantize, so such a stack is refused by name.
#[must_use]
pub const fn qwen35_down(ty: GgmlType) -> Option<DownEntry> {
    let (Some((entry, _)), Some(_)) = (kernels::down(ty), kernels::gemm(ty)) else {
        return None;
    };
    match entry {
        DownEntry::Q4k | DownEntry::Q5k | DownEntry::Q6k => Some(entry),
        DownEntry::Q5_0
        | DownEntry::Q5_1
        | DownEntry::Q8_0
        | DownEntry::Iq2xs
        | DownEntry::Iq3xxs
        | DownEntry::Iq3s
        | DownEntry::Iq4nl
        | DownEntry::Iq4xs
        | DownEntry::Mxfp4 => None,
    }
}
