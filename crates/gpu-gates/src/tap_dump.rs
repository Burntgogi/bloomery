//! The f32 files of a tap dump (`forced_probe --dump`, which `exact_ref --dump` mirrors): little-endian, one tensor a file.
//! A dump read back is compared whole or not at all: a file that ends inside a value, or a tensor of another length, is an
//! error naming it, never a shorter tensor that compares as exact.

use crate::GateError;
use std::path::Path;

/// Every f32 of the file at `path`. A file that ends inside a value is an error naming the file and its length.
pub fn read_f32(path: &Path) -> Result<Vec<f32>, GateError> {
    let b = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let (floats, stray) = b.as_chunks::<4>();
    if !stray.is_empty() {
        return Err(format!(
            "read {}: {} bytes end in {} stray bytes, not a whole number of f32",
            path.display(),
            b.len(),
            stray.len()
        )
        .into());
    }
    Ok(floats.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

/// `‖a − b‖ / ‖b‖` and `max |a − b|` of two spans of one nonzero length. Any other pair has no distance (zipped, a
/// truncated or empty dump would compare as exact): it is an error naming `what`.
pub fn distance(what: &str, a: &[f32], b: &[f32]) -> Result<(f64, f64), GateError> {
    if a.len() != b.len() || a.is_empty() {
        return Err(format!(
            "{what}: {} values here and {} in the dump, want one nonzero length",
            a.len(),
            b.len()
        )
        .into());
    }
    let (mut num, mut den, mut mx) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let d = f64::from(*x) - f64::from(*y);
        num += d * d;
        den += f64::from(*y) * f64::from(*y);
        mx = mx.max(d.abs());
    }
    Ok((if den > 0.0 { (num / den).sqrt() } else { 0.0 }, mx))
}

#[cfg(test)]
mod tests {
    use super::{distance, read_f32};
    use std::path::PathBuf;

    fn dump_file(name: &str, bytes: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "bloomery-tap-dump-{}-{name}.f32",
            std::process::id()
        ));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    /// An empty dump file zips no pairs, so a distance that does not check lengths prints zero for a tap that has
    /// values here. Mutant: the length check deleted, the empty file read as an exact match.
    #[test]
    fn an_empty_dump_is_refused_not_read_as_exact() {
        let path = dump_file("empty", b"");
        let theirs = read_f32(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        let e = distance("L00.attn_norm.f32", &[1.0, 2.0, 3.0], &theirs)
            .expect_err("empty against three values")
            .to_string();
        assert!(
            e.contains("L00.attn_norm.f32") && e.contains("3 values here and 0 in the dump"),
            "{e}"
        );
        assert!(distance("t", &[], &[]).is_err());
    }

    /// A tensor cut short by a whole value is as undefined as an empty one. Mutant: the lengths zipped to the shorter.
    #[test]
    fn a_tensor_of_another_length_is_refused() {
        let e = distance("t", &[1.0, 2.0], &[1.0])
            .expect_err("two against one")
            .to_string();
        assert!(e.contains("2 values here and 1 in the dump"), "{e}");
        assert!(distance("t", &[1.0], &[1.0, 2.0]).is_err());
    }

    /// A file that ends inside a value is refused by name, its stray bytes counted. Mutant: the trailing bytes dropped,
    /// the file read as the values before them.
    #[test]
    fn a_dump_ending_inside_a_value_is_refused() {
        let path = dump_file("partial", &[0, 0, 128, 63, 7]);
        let e = read_f32(&path).expect_err("five bytes").to_string();
        std::fs::remove_file(&path).unwrap();
        assert!(
            e.contains(&path.display().to_string()) && e.contains("5 bytes end in 1 stray bytes"),
            "{e}"
        );
    }

    /// Whole files of one length compare as before: little-endian values, zero for equal spans, the relative and the
    /// absolute distance otherwise.
    #[test]
    fn whole_dumps_of_one_length_compare() {
        let bytes: Vec<u8> = [1.0f32, 2.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let path = dump_file("whole", &bytes);
        let theirs = read_f32(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(theirs, [1.0, 2.0, 4.0]);
        assert_eq!(distance("t", &theirs, &theirs).unwrap(), (0.0, 0.0));
        let (rel, mx) = distance("t", &[1.0, 2.0, 3.0], &theirs).unwrap();
        assert_eq!(mx, 1.0);
        assert!((rel - (1.0f64 / 21.0).sqrt()).abs() < 1e-12, "{rel}");
    }
}
