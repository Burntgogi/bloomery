//! DFlash2's selector over one block (llama.cpp `build_dflash2_selector` and
//! `common_speculative_selector_walk`). Every block row after the anchor keeps
//! the `k` ids its logits rank highest ([`top_k`]); each kept id is scored
//! against each id the row before kept ([`score_row`]): its own logit plus
//! `Σ_r succ[id][r] · pred[before][r] · gate[r]`, `gate` being the row's
//! selector projection. Row 1's only predecessor is the anchor id. A greedy
//! walk ([`walk`]) then takes, row by row, the best-scored id after the one it
//! took before. The scores are the [`Lattice`]; the walk alone reads it.
//!
//! Undefined input is refused by name: a NaN logit or score, a candidate list
//! shorter than `k`, factor rows of another width.

use std::fmt;

/// A selector input the module refuses, naming where.
#[derive(Clone, Debug, PartialEq)]
pub enum SelectorError {
    /// A NaN at `index` of `what`'s row `row`.
    Nan {
        what: &'static str,
        row: usize,
        index: usize,
    },
    /// Fewer than `k` values to keep `k` from.
    Short { have: usize, k: usize },
    /// A row of `what` of `have` values where `want` are needed.
    Width {
        what: &'static str,
        have: usize,
        want: usize,
    },
    /// A lattice row whose scores do not cover its predecessors.
    Lattice { row: usize, detail: String },
}

impl fmt::Display for SelectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SelectorError::Nan { what, row, index } => {
                write!(f, "selector: NaN in {what} row {row} at {index}")
            }
            SelectorError::Short { have, k } => {
                write!(f, "selector: {have} values cannot give a top {k}")
            }
            SelectorError::Width { what, have, want } => {
                write!(
                    f,
                    "selector: a {what} row of {have} values, {want} expected"
                )
            }
            SelectorError::Lattice { row, detail } => {
                write!(f, "selector: lattice row {row}: {detail}")
            }
        }
    }
}

impl std::error::Error for SelectorError {}

/// The `k` ids of `logits` (row `row` of the block, for the error) with the
/// highest values, highest first; equal values keep the lower id first. A NaN
/// anywhere in the row is refused by name.
pub fn top_k(logits: &[f32], k: usize, row: usize) -> Result<Vec<(u32, f32)>, SelectorError> {
    if logits.len() < k {
        return Err(SelectorError::Short {
            have: logits.len(),
            k,
        });
    }
    if let Some(index) = logits.iter().position(|v| v.is_nan()) {
        return Err(SelectorError::Nan {
            what: "logits",
            row,
            index,
        });
    }
    if u32::try_from(logits.len()).is_err() {
        return Err(SelectorError::Width {
            what: "logits",
            have: logits.len(),
            want: u32::MAX as usize,
        });
    }
    // Every id fits u32 (above), so the casts below keep it.
    let row_ids = logits.iter().enumerate().map(|(id, &v)| (id as u32, v));
    // `a` ranks below `b`: a lower value, or an equal value at a higher id.
    let weaker = |a: (u32, f32), b: (u32, f32)| a.1 < b.1 || (a.1 == b.1 && a.0 > b.0);
    let weakest = |kept: &[(u32, f32)]| {
        (1..kept.len()).fold(0, |w, i| if weaker(kept[i], kept[w]) { i } else { w })
    };
    let mut kept: Vec<(u32, f32)> = row_ids.clone().take(k).collect();
    if k == 0 {
        return Ok(kept);
    }
    let mut floor = weakest(&kept);
    for c in row_ids.skip(k) {
        if weaker(kept[floor], c) {
            kept[floor] = c;
            floor = weakest(&kept);
        }
    }
    kept.sort_by(|a, b| {
        if weaker(*b, *a) {
            std::cmp::Ordering::Less
        } else if weaker(*a, *b) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    Ok(kept)
}

/// One scored block row: its kept ids, and per predecessor `j` the scores of
/// every kept id after it, `scores[j * ids.len() + i]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Scored {
    pub ids: Vec<u32>,
    pub scores: Vec<f32>,
}

impl Scored {
    /// Predecessors this row was scored against.
    #[must_use]
    pub fn preds(&self) -> usize {
        self.scores.len().checked_div(self.ids.len()).unwrap_or(0)
    }
}

/// The block's scored rows, row 1 first: what the walk reads.
#[derive(Clone, Debug, PartialEq)]
pub struct Lattice {
    pub rows: Vec<Scored>,
}

/// Score block row `row`'s kept candidates `cands` (from [`top_k`]) against
/// the ids `preds` the row before kept (the anchor alone for row 1). `gate`
/// is the row's selector projection, `rank` wide; `pred` and `succ` write the
/// predecessor and successor factor rows of an id (`rank` values each).
///
/// `scores[j * k + i] = cands[i].1 + Σ_r succ(cands[i].0)[r] · (pred(preds[j])[r] · gate[r])`.
pub fn score_row(
    row: usize,
    cands: &[(u32, f32)],
    preds: &[u32],
    gate: &[f32],
    pred: &mut dyn FnMut(u32, &mut [f32]),
    succ: &mut dyn FnMut(u32, &mut [f32]),
) -> Result<Scored, SelectorError> {
    let rank = gate.len();
    if let Some(index) = gate.iter().position(|v| v.is_nan()) {
        return Err(SelectorError::Nan {
            what: "gate",
            row,
            index,
        });
    }
    let k = cands.len();
    let mut succ_rows = vec![0.0f32; k * rank];
    for (i, &(id, _)) in cands.iter().enumerate() {
        succ(id, &mut succ_rows[i * rank..(i + 1) * rank]);
    }
    let mut cond = vec![0.0f32; rank];
    let mut scores = vec![0.0f32; preds.len() * k];
    for (j, &p) in preds.iter().enumerate() {
        pred(p, &mut cond);
        for (c, g) in cond.iter_mut().zip(gate) {
            *c *= g;
        }
        for (i, &(_, unary)) in cands.iter().enumerate() {
            let s: f32 = succ_rows[i * rank..(i + 1) * rank]
                .iter()
                .zip(&cond)
                .map(|(a, b)| a * b)
                .sum();
            let v = s + unary;
            if v.is_nan() {
                return Err(SelectorError::Nan {
                    what: "scores",
                    row,
                    index: j * k + i,
                });
            }
            scores[j * k + i] = v;
        }
    }
    Ok(Scored {
        ids: cands.iter().map(|c| c.0).collect(),
        scores,
    })
}

/// The greedy walk over `lattice`: from predecessor 0 (row 1's only one, the
/// anchor), each row takes its best-scored id after the id taken before; the
/// first of equal scores wins. It stops at a row whose best score is −∞, or
/// whose best id's softmax over the row's scores is under `p_min` (0 never
/// stops). Returns the ids taken, at most one per row.
pub fn walk(lattice: &Lattice, p_min: f32) -> Result<Vec<u32>, SelectorError> {
    let mut out = Vec::with_capacity(lattice.rows.len());
    let mut before = 0usize;
    for (r, row) in lattice.rows.iter().enumerate() {
        let k = row.ids.len();
        if k == 0 || row.scores.len() != k * row.preds() || before >= row.preds() {
            return Err(SelectorError::Lattice {
                row: r + 1,
                detail: format!(
                    "{k} ids, {} scores, predecessor {before} of {}",
                    row.scores.len(),
                    row.preds()
                ),
            });
        }
        let s = &row.scores[before * k..(before + 1) * k];
        let best = (1..k).fold(0, |b, i| if s[i] > s[b] { i } else { b });
        if s[best] == f32::NEG_INFINITY {
            break;
        }
        if p_min > 0.0 {
            let sum: f32 = s.iter().map(|v| (v - s[best]).exp()).sum();
            if 1.0 / sum < p_min {
                break;
            }
        }
        out.push(row.ids[best]);
        before = best;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{Lattice, Scored, SelectorError, score_row, top_k, walk};

    #[test]
    fn top_k_orders_by_value_then_lower_id() {
        let logits = [0.5, 3.0, -1.0, 3.0, 2.0, 0.5];
        let got = top_k(&logits, 4, 1).expect("defined");
        assert_eq!(got, vec![(1, 3.0), (3, 3.0), (4, 2.0), (0, 0.5)]);
    }

    #[test]
    fn top_k_refuses_nan_and_short_rows() {
        assert_eq!(
            top_k(&[1.0, f32::NAN, 2.0], 2, 3),
            Err(SelectorError::Nan {
                what: "logits",
                row: 3,
                index: 1
            })
        );
        assert_eq!(
            top_k(&[1.0], 2, 1),
            Err(SelectorError::Short { have: 1, k: 2 })
        );
    }

    /// Rank 1 factors: `pred(id) = id`, `succ(id) = 1`, so a score is the
    /// candidate's logit plus `before · gate`.
    #[test]
    fn score_row_adds_the_pairwise_term_per_predecessor() {
        let cands = [(7, 1.0), (9, 0.0)];
        let mut pred = |id: u32, out: &mut [f32]| out[0] = id as f32;
        let mut succ = |id: u32, out: &mut [f32]| out[0] = if id == 9 { 2.0 } else { 1.0 };
        let got = score_row(2, &cands, &[1, 3], &[0.5], &mut pred, &mut succ).expect("defined");
        assert_eq!(got.ids, vec![7, 9]);
        // j=0 (pred 1): 1 + 1·0.5, 0 + 2·0.5; j=1 (pred 3): 1 + 1·1.5, 0 + 2·1.5.
        assert_eq!(got.scores, vec![1.5, 1.0, 2.5, 3.0]);
        assert_eq!(got.preds(), 2);
    }

    /// The walk follows the pairwise scores, not each row's own best: row 2's
    /// best after id 10 is id 21 though id 20 has the higher score after id
    /// 11.
    #[test]
    fn walk_follows_the_predecessor_taken() {
        let lattice = Lattice {
            rows: vec![
                Scored {
                    ids: vec![10, 11],
                    scores: vec![2.0, 1.0],
                },
                Scored {
                    ids: vec![20, 21],
                    scores: vec![0.0, 1.0, 9.0, 0.0],
                },
                Scored {
                    ids: vec![30, 31],
                    scores: vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 5.0, 4.0],
                },
            ],
        };
        assert_eq!(walk(&lattice, 0.0).expect("defined"), vec![10, 21, 30]);
    }

    #[test]
    fn walk_stops_at_minus_infinity_and_under_p_min() {
        let lattice = Lattice {
            rows: vec![
                Scored {
                    ids: vec![10, 11],
                    scores: vec![1.0, 0.0],
                },
                Scored {
                    ids: vec![20, 21],
                    scores: vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0, 0.0],
                },
            ],
        };
        assert_eq!(walk(&lattice, 0.0).expect("defined"), vec![10]);
        // softmax(1, 0) at its best is 0.731: under 0.75 the walk takes nothing.
        assert_eq!(walk(&lattice, 0.75).expect("defined"), Vec::<u32>::new());
    }

    #[test]
    fn walk_refuses_a_row_short_of_predecessors() {
        let lattice = Lattice {
            rows: vec![
                Scored {
                    ids: vec![10, 11],
                    scores: vec![0.0, 1.0],
                },
                Scored {
                    ids: vec![20, 21],
                    scores: vec![0.0, 1.0],
                },
            ],
        };
        assert!(matches!(
            walk(&lattice, 0.0),
            Err(SelectorError::Lattice { row: 2, .. })
        ));
    }
}
