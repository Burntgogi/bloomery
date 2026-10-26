//! The reference's Unicode handling, byte for byte: its lenient UTF-8 decoder,
//! the strict one of the newer tree that normalizes to NFC, the GPT-2 byte <->
//! codepoint table, and the "collapsed" byte its regex fallback matches on.

use unicode_normalization_alignments::UnicodeNormalization;

use crate::collapse_table::RUNS;
use crate::letter_case_table::RUNS as LETTER_CASE_RUNS;

/// U+FFFD, what the reference substitutes for a byte it cannot decode.
pub(crate) const REPLACEMENT: u32 = 0xFFFD;

/// Decode `bytes` the way the reference's `unicode_cpts_from_utf8` does: a lead
/// byte is trusted for its length, each continuation byte must be `10xxxxxx`,
/// overlong forms and surrogates pass through, and a byte that starts no valid
/// sequence becomes U+FFFD and advances one byte.
///
/// A four-byte form above U+10FFFF is refused; the reference throws on it
/// when it re-encodes the word.
pub(crate) fn decode_lenient(bytes: &[u8], out: &mut Vec<u32>) {
    out.clear();
    out.reserve(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let (cpt, n) = decode_one(bytes, i);
        assert!(cpt <= 0x10FFFF, "UTF-8 codepoint U+{cpt:X} above U+10FFFF");
        out.push(cpt);
        i += n;
    }
}

/// The codepoint at `b[i]` and its length, the lead byte trusted for the length: the sequence's
/// bits as written, overlong or above U+10FFFF or not.
fn decode_one(b: &[u8], i: usize) -> (u32, usize) {
    let cont = |k: usize| b.get(i + k).is_some_and(|&c| c & 0xC0 == 0x80);
    let lead = b[i];
    let bits = |k: usize| u32::from(b[i + k] & 0x3F);
    if lead & 0x80 == 0 {
        (u32::from(lead), 1)
    } else if lead & 0x40 == 0 {
        (REPLACEMENT, 1)
    } else if lead & 0x20 == 0 {
        if cont(1) {
            ((u32::from(lead & 0x1F) << 6) | bits(1), 2)
        } else {
            (REPLACEMENT, 1)
        }
    } else if lead & 0x10 == 0 {
        if cont(1) && cont(2) {
            ((u32::from(lead & 0x0F) << 12) | (bits(1) << 6) | bits(2), 3)
        } else {
            (REPLACEMENT, 1)
        }
    } else if lead & 0x08 == 0 {
        if cont(1) && cont(2) && cont(3) {
            let cpt = (u32::from(lead & 0x07) << 18) | (bits(1) << 12) | (bits(2) << 6) | bits(3);
            (cpt, 4)
        } else {
            (REPLACEMENT, 1)
        }
    } else {
        (REPLACEMENT, 1)
    }
}

/// Decode `bytes` the way the newer reference tree's `unicode_cpts_from_utf8` does, the one that
/// precedes its NFC step: [`decode_lenient`]'s rules with overlong forms and codepoints above
/// U+10FFFF refused too. A byte that starts no valid sequence is one U+FFFD and advances one byte;
/// surrogates pass through.
pub(crate) fn decode_strict(bytes: &[u8], out: &mut Vec<u32>) {
    out.clear();
    out.reserve(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let (cpt, n) = match decode_one(bytes, i) {
            (cpt, n) if n > 1 && cpt < [0, 0, 0x80, 0x800, 0x1_0000][n] => (REPLACEMENT, 1),
            (cpt, _) if cpt > 0x10FFFF => (REPLACEMENT, 1),
            ok => ok,
        };
        out.push(cpt);
        i += n;
    }
}

/// NFC of `cpts`, as the reference's `unicode_cpts_normalize_nfc` makes it: each run of
/// codepoints that are characters normalizes on its own, and a surrogate (no character, a starter
/// that composes with nothing) stands where it was.
pub(crate) fn normalize_nfc(cpts: &mut Vec<u32>) {
    let mut out = Vec::with_capacity(cpts.len());
    let mut run = String::new();
    let flush = |run: &mut String, out: &mut Vec<u32>| {
        out.extend(run.nfc().map(|(c, _)| u32::from(c)));
        run.clear();
    };
    for &cpt in cpts.iter() {
        match char::from_u32(cpt) {
            Some(c) => run.push(c),
            None => {
                flush(&mut run, &mut out);
                out.push(cpt);
            }
        }
    }
    flush(&mut run, &mut out);
    *cpts = out;
}

/// Append the UTF-8 form of `cpt` (<= U+10FFFF; surrogates are encoded as
/// three bytes, as the reference does).
pub(crate) fn encode_cpt(cpt: u32, out: &mut Vec<u8>) {
    // The casts below keep the low bits of a value already masked or shifted
    // into byte range.
    if cpt <= 0x7F {
        out.push(cpt as u8);
    } else if cpt <= 0x7FF {
        out.extend_from_slice(&[0xC0 | (cpt >> 6) as u8, 0x80 | (cpt & 0x3F) as u8]);
    } else if cpt <= 0xFFFF {
        out.extend_from_slice(&[
            0xE0 | (cpt >> 12) as u8,
            0x80 | ((cpt >> 6) & 0x3F) as u8,
            0x80 | (cpt & 0x3F) as u8,
        ]);
    } else {
        out.extend_from_slice(&[
            0xF0 | ((cpt >> 18) & 0x07) as u8,
            0x80 | ((cpt >> 12) & 0x3F) as u8,
            0x80 | ((cpt >> 6) & 0x3F) as u8,
            0x80 | (cpt & 0x3F) as u8,
        ]);
    }
}

/// The byte a codepoint collapses to: itself below 0x80, otherwise its
/// category byte from the reference's table (whitespace first).
pub(crate) fn collapse(cpt: u32) -> u8 {
    match u8::try_from(cpt) {
        Ok(b) if b < 0x80 => b,
        _ => {
            let i = RUNS.partition_point(|&(start, _)| start <= cpt);
            // RUNS starts at 0x80 and `cpt` is at least that, so `i >= 1`.
            RUNS[i - 1].1
        }
    }
}

/// The codepoints Unicode 15.0 and 15.1 added that the reference's table
/// classes as letters, marks or numbers. Hugging Face's tokenizers matches its
/// regex with Oniguruma 6.9.8, whose tables are Unicode 14.0's, so there they
/// are none of `\p{L}`, `\p{M}`, `\p{N}` or `\s`.
const AFTER_UNICODE_14: [(u32, u32); 20] = [
    (0x0CF3, 0x0CF3),
    (0x0ECE, 0x0ECE),
    (0x10EFD, 0x10EFF),
    (0x1123F, 0x11241),
    (0x11F00, 0x11F10),
    (0x11F12, 0x11F3A),
    (0x11F3E, 0x11F42),
    (0x11F50, 0x11F59),
    (0x1342F, 0x1342F),
    (0x13440, 0x13455),
    (0x1B132, 0x1B132),
    (0x1B155, 0x1B155),
    (0x1D2C0, 0x1D2D3),
    (0x1DF25, 0x1DF2A),
    (0x1E030, 0x1E06D),
    (0x1E08F, 0x1E08F),
    (0x1E4D0, 0x1E4F9),
    (0x2B739, 0x2B739),
    (0x2EBF0, 0x2EE5D),
    (0x31350, 0x323AF),
];

/// [`collapse`] as Hugging Face's regex classes the codepoint: a codepoint of
/// [`AFTER_UNICODE_14`] is 0xD0, the byte of an unassigned codepoint.
pub(crate) fn collapse_hf(cpt: u32) -> u8 {
    let i = AFTER_UNICODE_14.partition_point(|&(_, last)| last < cpt);
    match AFTER_UNICODE_14.get(i) {
        Some(&(first, _)) if first <= cpt => 0xD0,
        _ => collapse(cpt),
    }
}

/// A letter's category class in Unicode 14.0, the version Hugging Face's regex classes are: 1 for
/// `Lu` and `Lt`, 2 for `Ll`, 3 for `Lm` and `Lo`, 0 for a codepoint that is no letter.
pub(crate) fn letter_case(cpt: u32) -> u8 {
    match u8::try_from(cpt) {
        Ok(b) if b < 0x80 => {
            u8::from(b.is_ascii_uppercase()) + 2 * u8::from(b.is_ascii_lowercase())
        }
        _ => {
            let i = LETTER_CASE_RUNS.partition_point(|&(start, _)| start <= cpt);
            // The runs start at 0x80 and `cpt` is at least that, so `i >= 1`.
            LETTER_CASE_RUNS[i - 1].1
        }
    }
}

/// GPT-2's byte-level alphabet: the codepoint each byte is written as inside
/// a vocabulary string (`unicode_byte_to_utf8_map`).
pub fn byte_to_cpt_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut assigned = [false; 256];
    for b in (0x21..=0x7E).chain(0xA1..=0xAC).chain(0xAE..=0xFF) {
        table[b] = b as u32;
        assigned[b] = true;
    }
    let mut n = 0;
    for b in 0..256 {
        if !assigned[b] {
            table[b] = 256 + n;
            n += 1;
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use super::{decode_lenient, decode_strict, encode_cpt, normalize_nfc};

    #[test]
    fn upper_unicode_boundary_round_trips() {
        let bytes = b"\xF4\x8F\xBF\xBF";
        let mut cpts = Vec::new();
        decode_lenient(bytes, &mut cpts);
        assert_eq!(cpts, [0x10FFFF]);
        let mut encoded = Vec::new();
        encode_cpt(cpts[0], &mut encoded);
        assert_eq!(encoded, bytes);
    }

    /// The newer tree's reader: an overlong form, a four-byte form above U+10FFFF and a truncated
    /// one are U+FFFD a byte, a surrogate passes, and the NFC step composes what is a character
    /// run and leaves a surrogate where it stood.
    #[test]
    fn strict_decoding_replaces_a_byte_at_a_time() {
        let mut cpts = Vec::new();
        for (bytes, want) in [
            (&b"\xC0\x80"[..], vec![0xFFFD, 0xFFFD]),
            (b"\xC1\xBF", vec![0xFFFD, 0xFFFD]),
            (b"\xE0\x80\x80", vec![0xFFFD; 3]),
            (b"\xF0\x80\x80\x80", vec![0xFFFD; 4]),
            (b"\xF4\x90\x80\x80", vec![0xFFFD; 4]),
            (b"\xF0\x9F\x98", vec![0xFFFD; 3]),
            (b"\xED\xA0\x80", vec![0xD800]),
            (b"\xF4\x8F\xBF\xBF", vec![0x10FFFF]),
            (b"a\x80b", vec![0x61, 0xFFFD, 0x62]),
        ] {
            decode_strict(bytes, &mut cpts);
            assert_eq!(cpts, want, "{bytes:?}");
        }
        let mut cpts = vec![0x65, 0x301, 0xD800, 0x65, 0x301, 0x1112, 0x1161, 0x11AB];
        normalize_nfc(&mut cpts);
        assert_eq!(cpts, [0xE9, 0xD800, 0xE9, 0xD55C]);
    }

    #[test]
    #[should_panic(expected = "above U+10FFFF")]
    fn refuses_codepoint_above_unicode_limit() {
        decode_lenient(b"\xF4\x90\x80\x80", &mut Vec::new());
    }
}
