use super::{ARCH, FAMILIES, MODEL, PROFILE, TOKENIZER};
use crate::RefError;
use crate::emb2::LCPP_BUILD;
use crate::emb2::tests::{check, header, remove, set_dir};
use crate::family::Identity;
use crate::tokenizer::tests::{CASES_MD5, RIGHT, Stated, set_dir as tokenizer_dir, write_set};

#[test]
fn text_families_take_their_sets_without_a_projector() -> Result<(), RefError> {
    let dir = set_dir("v1-own")?;
    for family in FAMILIES {
        let Identity::Emb2(kind, _) = family.identity else {
            panic!("embedding family")
        };
        assert_eq!(
            check(&dir, family, &header(&PROFILE, ARCH, kind, &[]), true)?,
            MODEL,
            "{}",
            family.name
        );
    }
    remove(&dir)
}

/// The tokenizer set is held to the oracle commit its `# build` line names, not to the md5s of one machine's binary: two builds of
/// the commit are the same oracle, while a set of another commit, and a set that states only its binary's md5s, are refused by name.
#[test]
fn the_tokenizer_set_is_pinned_by_the_oracle_commit() {
    let dir = tokenizer_dir("v1-commit");
    let set = |commit, binary, library| {
        write_set(
            &dir,
            &Stated {
                commit,
                binary,
                library,
                vocabulary: MODEL,
                ..RIGHT
            },
            CASES_MD5,
        );
        TOKENIZER.check_set(&dir)
    };
    for (binary, library) in [
        (RIGHT.binary, RIGHT.library),
        (
            "0123456789abcdef0123456789abcdef",
            "fedcba9876543210fedcba9876543210",
        ),
    ] {
        let own = set(Some(LCPP_BUILD), binary, library).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(own.build.as_deref(), Some(LCPP_BUILD));
    }
    for (commit, got) in [
        (Some("0000000ff"), "0000000ff"),
        (
            None,
            "llama-tokenize d03083b01a24f6a9e5c2190abf7d38c6 libllama.so f855ad0b949f0b7d871973be1fcc419f",
        ),
    ] {
        match set(commit, RIGHT.binary, RIGHT.library) {
            Err(RefError::Foreign {
                field: "build",
                family: "tokenizer-llama",
                got: stated,
                want,
                ..
            }) => assert_eq!((stated.as_str(), want.as_str()), (got, LCPP_BUILD)),
            r => panic!("a set of {commit:?}: {r:?}"),
        }
    }
    std::fs::remove_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
}
