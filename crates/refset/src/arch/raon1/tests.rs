use super::{ARCH, LCPP, LCPP_BUILD, MODEL};
use crate::RefError;
use crate::arch::node_dump_contract::{self as contract, Row};

/// Raon's row: the Q4_K_M file, the vendor branch's build, this architecture; the sets it must refuse are another quantization of the
/// same model (the same vocabulary and architecture, so only the `# model` line tells it), the DFlash2 draft file beside it, the file's
/// name in another directory, the bare commit (no library named), another library, and mainline llama.cpp (it has no Raon graph).
const ROW: Row = Row {
    family: &LCPP,
    model: MODEL,
    build: LCPP_BUILD,
    arch: ARCH,
    other_files: &[
        "/models/Raon-1-mini-preview/raon-1-mini-preview-Q8_0.gguf",
        "/models/Raon-1-mini-preview/dflash-raon-1-mini-preview-Q8_0.gguf",
        "/models/elsewhere/raon-1-mini-preview-Q4_K_M.gguf",
    ],
    other_builds: &[
        "fb895662d",
        "fb895662d+libllama-000000000000",
        "cd0a5b6a1+libllama-d846809bb9cd",
    ],
    other_arches: &["raon-1", "minimax-m3"],
};

#[test]
fn the_family_takes_a_set_of_its_file_build_and_arch() -> Result<(), RefError> {
    contract::takes_a_set_of_its_file_build_and_arch(&ROW)
}

#[test]
fn the_family_refuses_a_set_of_another_file() -> Result<(), RefError> {
    contract::refuses_a_set_of_another_file(&ROW)
}

#[test]
fn the_family_refuses_a_set_of_another_build() -> Result<(), RefError> {
    contract::refuses_a_set_of_another_build(&ROW)
}

#[test]
fn the_family_refuses_a_set_of_another_arch() -> Result<(), RefError> {
    contract::refuses_a_set_of_another_arch(&ROW)
}

#[test]
fn the_family_refuses_a_set_without_its_trailer() -> Result<(), RefError> {
    contract::refuses_a_set_without_its_trailer(&ROW)
}
