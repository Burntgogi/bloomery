//! The raon vocabulary: our ids equal `llama-tokenize`'s of the reference tree, id for id, on every case of
//! `cases.txt` and `cases-raon.txt` in both parse modes, and the header's two end tokens end generation.
//! The reference's ids are `raon-ids.tsv`, pinned by `tools/raon-oracle.sh` (its header names the tree's
//! commit): the raon pre-tokenizer and the NFC normalizer exist in that tree only.
//!
//! `hw_`: needs the vocabulary file on the box (`bash tools/gate.sh --release -p bloomery-tokenizer --test raon
//! -- --ignored --nocapture`). The vocabulary only: seconds.

use std::path::PathBuf;

use tokenizer::Tokenizer;

/// The Q4_K_M file on the box.
fn vocabulary() -> PathBuf {
    PathBuf::from("/models/Raon-1-mini-preview/raon-1-mini-preview-Q4_K_M.gguf")
}

/// The bytes a row's hex column names.
fn unhex(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("a hex byte"))
        .collect()
}

/// A row's id column.
fn ids(col: &str) -> Vec<u32> {
    col.split_whitespace()
        .map(|id| id.parse().expect("an id"))
        .collect()
}

/// Every case encodes to the reference's ids, with special tokens parsed and without. FAIL-first:
/// with the NFC step or the digit grouping of the raon pre-tokenizer off, the rows of combining marks and
/// of digit runs differ.
#[test]
#[ignore = "gate: the raon vocabulary file on the box"]
fn hw_raon_ids_equal_the_reference() {
    let tok = Tokenizer::from_gguf(vocabulary()).expect("the raon vocabulary loads");
    let pins = include_str!("raon-ids.tsv");
    let mut differ = Vec::new();
    let mut rows = 0;
    let mut complete = None;
    for line in pins.lines() {
        if let Some(n) = line.strip_prefix("# complete\t") {
            complete = Some(n.parse::<usize>().expect("a count"));
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 3, "{line}");
        let text = unhex(cols[0]);
        rows += 1;
        for (parse, want) in [(true, ids(cols[1])), (false, ids(cols[2]))] {
            let got = tok.encode(&text, false, parse);
            if got != want {
                differ.push(format!(
                    "{:?} parse_special {parse}: ours {got:?}, reference {want:?}",
                    String::from_utf8_lossy(&text)
                ));
            }
        }
    }
    assert_eq!(complete, Some(rows), "the pins' trailer names every row");
    println!("raon: {rows} cases, {} differ", differ.len());
    assert!(differ.is_empty(), "{}", differ.join("\n"));
}

/// The header's `<|im_end|>` and `<|endoftext|>` end generation: the two ids the model's generation
/// config lists.
#[test]
#[ignore = "gate: the raon vocabulary file on the box"]
fn hw_raon_end_tokens_end_generation() {
    let tok = Tokenizer::from_gguf(vocabulary()).expect("the raon vocabulary loads");
    for text in ["<|im_end|>", "<|endoftext|>"] {
        let id = tok.encode(text, false, true);
        assert_eq!(id.len(), 1, "{text} is one token");
        assert!(
            tok.eog().contains(&id[0]),
            "{text} ({}) is not an end",
            id[0]
        );
    }
}
