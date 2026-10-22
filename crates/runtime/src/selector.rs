//! DFlash2's selector over one block (llama.cpp `build_dflash2_selector` and
//! `common_speculative_selector_walk`). Every block row after the anchor keeps
//! the `k` ids its logits rank highest ([`top_k`]); each kept id is scored
//! against each id the row before kept ([`score_row`]): its own logit plus
//! `Σ_r succ[id][r] · pred[before][r] · gate[r]`, `gate` being the row's
//! selector projection. Row 1's only predecessor is the anchor id. A greedy
//! walk ([`walk`]) then takes, row by row, the best-scored id after the one it
//! took before. The scores are the [`Lattice`]; the walk alone reads it. The
//! card writes the lattice as rows of words ([`Lattice::from_words`]).
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

/// Words of one lattice row as the card writes it: the `k` kept ids, then
/// `k * k` scores `scores[j * k + i]`.
#[must_use]
pub const fn row_words(k: usize) -> usize {
    k + k * k
}

impl Lattice {
    /// The lattice of `rows` block rows (row 1 first) from the card's words,
    /// each [`row_words`]`(k)` long: the `k` ids, then the scores of every
    /// kept id after each of `k` predecessors. The first row has one
    /// predecessor, the anchor, so only its first `k` scores are read. A
    /// buffer shorter than `rows` rows, or a NaN score, is refused by name.
    pub fn from_words(words: &[u32], rows: usize, k: usize) -> Result<Lattice, SelectorError> {
        let stride = row_words(k);
        let want = rows.saturating_mul(stride);
        if words.len() < want {
            return Err(SelectorError::Width {
                what: "lattice words",
                have: words.len(),
                want,
            });
        }
        let mut out = Vec::with_capacity(rows);
        for (r, row) in words.chunks_exact(stride).take(rows).enumerate() {
            let preds = if r == 0 { 1 } else { k };
            let scores: Vec<f32> = row[k..k + preds * k]
                .iter()
                .map(|&w| f32::from_bits(w))
                .collect();
            if let Some(index) = scores.iter().position(|v| v.is_nan()) {
                return Err(SelectorError::Nan {
                    what: "scores",
                    row: r + 1,
                    index,
                });
            }
            out.push(Scored {
                ids: row[..k].to_vec(),
                scores,
            });
        }
        Ok(Lattice { rows: out })
    }
}

/// Values a factor row holds on the card: the score kernel is built for it.
pub const CARD_RANK: usize = 256;

/// The dot of a successor row with a conditioned predecessor row
/// (`pred · gate`) in the card's order: 32 lane partials, lane `l` the fused
/// multiply-add chain from 0 over values `8l .. 8l + 8` in order, combined by
/// the xor butterfly (16, 8, 4, 2, 1) and read at lane 0. The host twin of
/// the card's score kernel, so a gate holds the card's scores to the bit.
/// Rows of another width are refused by name.
pub fn lane_dot(succ: &[f32], cond: &[f32]) -> Result<f32, SelectorError> {
    for (what, row) in [("successor", succ), ("conditioned predecessor", cond)] {
        if row.len() != CARD_RANK {
            return Err(SelectorError::Width {
                what,
                have: row.len(),
                want: CARD_RANK,
            });
        }
    }
    let lane_len = CARD_RANK / 32;
    let mut lane = [0.0f32; 32];
    for (l, acc) in lane.iter_mut().enumerate() {
        for q in 0..lane_len {
            *acc = succ[lane_len * l + q].mul_add(cond[lane_len * l + q], *acc);
        }
    }
    for off in [16usize, 8, 4, 2, 1] {
        let before = lane;
        for (l, v) in lane.iter_mut().enumerate() {
            *v = before[l] + before[l ^ off];
        }
    }
    Ok(lane[0])
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

/// An id the walk took and the largest softmax of its row's scores, the
/// probability the draft gives it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Taken {
    pub id: u32,
    pub p: f32,
}

/// The greedy walk over `lattice`: from predecessor 0 (row 1's only one, the
/// anchor), each row takes its best-scored id after the id taken before; the
/// first of equal scores wins. It stops at a row whose best score is −∞, or
/// whose best id's softmax over the row's scores is under `p_min` (0 never
/// stops). Returns the ids taken, at most one per row, each with that
/// softmax. A NaN score is refused by name.
pub fn walk_p(lattice: &Lattice, p_min: f32) -> Result<Vec<Taken>, SelectorError> {
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
        if let Some(index) = s.iter().position(|v| v.is_nan()) {
            return Err(SelectorError::Nan {
                what: "scores",
                row: r + 1,
                index: before * k + index,
            });
        }
        let best = (1..k).fold(0, |b, i| if s[i] > s[b] { i } else { b });
        if s[best] == f32::NEG_INFINITY {
            break;
        }
        let sum: f32 = s.iter().map(|v| (v - s[best]).exp()).sum();
        let p = 1.0 / sum;
        if p.is_nan() {
            return Err(SelectorError::Nan {
                what: "probabilities",
                row: r + 1,
                index: before * k + best,
            });
        }
        if p_min > 0.0 && p < p_min {
            break;
        }
        out.push(Taken {
            id: row.ids[best],
            p,
        });
        before = best;
    }
    Ok(out)
}

/// [`walk_p`]'s ids alone.
pub fn walk(lattice: &Lattice, p_min: f32) -> Result<Vec<u32>, SelectorError> {
    Ok(walk_p(lattice, p_min)?.into_iter().map(|t| t.id).collect())
}

#[cfg(test)]
mod tests {
    use super::{
        CARD_RANK, Lattice, Scored, SelectorError, lane_dot, row_words, score_row, top_k, walk,
        walk_p,
    };

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

    #[test]
    fn walk_reports_the_softmax_of_the_id_it_takes() {
        let lattice = Lattice {
            rows: vec![Scored {
                ids: vec![10, 11],
                scores: vec![1.0, 0.0],
            }],
        };
        let taken = walk_p(&lattice, 0.0).expect("defined");
        assert_eq!(taken.len(), 1);
        assert_eq!(taken[0].id, 10);
        // 1 / (1 + e^-1)
        assert!((taken[0].p - 0.731_058_6).abs() < 1e-6, "{}", taken[0].p);
    }

    #[test]
    fn walk_refuses_a_nan_score_by_name() {
        let lattice = Lattice {
            rows: vec![
                Scored {
                    ids: vec![10, 11],
                    scores: vec![1.0, 0.0],
                },
                Scored {
                    ids: vec![20, 21],
                    scores: vec![0.0, 0.0, 0.0, f32::NAN],
                },
            ],
        };
        // Predecessor 0 was taken: the NaN sits in predecessor 1's scores
        // and the walk never reads them.
        assert_eq!(walk(&lattice, 0.0).expect("defined"), vec![10, 20]);
        let nan_first = Lattice {
            rows: vec![Scored {
                ids: vec![10, 11],
                scores: vec![f32::NAN, 0.0],
            }],
        };
        assert_eq!(
            walk(&nan_first, 0.0),
            Err(SelectorError::Nan {
                what: "scores",
                row: 1,
                index: 0
            })
        );
    }

    /// The card's words: row 1 has the anchor's `k` scores only (what follows
    /// them is not read), later rows `k` predecessors' scores.
    #[test]
    fn from_words_reads_the_cards_rows() {
        let k = 2;
        assert_eq!(row_words(k), 6);
        let f = |v: f32| v.to_bits();
        let words = [
            // row 1: ids 7 8, the anchor's scores 1.5 0.5, then words the
            // card did not write
            7,
            8,
            f(1.5),
            f(0.5),
            f32::NAN.to_bits(),
            0,
            // row 2: ids 9 10, scores after predecessor 0 then 1
            9,
            10,
            f(1.0),
            f(2.0),
            f(3.0),
            f(4.0),
        ];
        let lattice = Lattice::from_words(&words, 2, k).expect("defined");
        assert_eq!(lattice.rows[0].ids, vec![7, 8]);
        assert_eq!(lattice.rows[0].scores, vec![1.5, 0.5]);
        assert_eq!(lattice.rows[0].preds(), 1);
        assert_eq!(lattice.rows[1].ids, vec![9, 10]);
        assert_eq!(lattice.rows[1].scores, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(lattice.rows[1].preds(), 2);
        // The walk takes 7 (1.5), then after it 10 (2.0 over 1.0).
        assert_eq!(walk(&lattice, 0.0).expect("defined"), vec![7, 10]);
    }

    #[test]
    fn from_words_refuses_a_short_buffer_and_a_nan_score() {
        let k = 2;
        assert_eq!(
            Lattice::from_words(&[0; 11], 2, k),
            Err(SelectorError::Width {
                what: "lattice words",
                have: 11,
                want: 12
            })
        );
        let mut words = [0u32; 12];
        words[8 + 1] = f32::NAN.to_bits();
        assert_eq!(
            Lattice::from_words(&words, 2, k),
            Err(SelectorError::Nan {
                what: "scores",
                row: 2,
                index: 1
            })
        );
    }

    /// Raon-1-mini's first block on set a: the ids the reference proposes
    /// (the walk), against the id each row's logits rank first. The scores
    /// here are made up (the oracle's own are on the box): each row's top
    /// logit is a decoy, the walk's id a runner-up, and a pairwise bonus
    /// for the id that follows the one taken before carries the walk.
    #[test]
    fn the_walk_takes_the_oracles_ids_where_each_rows_best_logit_is_another() {
        const PICKS: [u32; 7] = [166_387, 4_832, 23_388, 166_436, 10_189, 46, 1_794];
        const DECOYS: [u32; 7] = [7, 99, 5_000, 42, 123_456, 8, 60_000];
        const ANCHOR: u32 = 9_999;
        let vocab = 166_437;
        let rows: Vec<Vec<f32>> = (0..PICKS.len())
            .map(|r| {
                let mut l = vec![0.0f32; vocab];
                l[DECOYS[r] as usize] = 9.0;
                l[PICKS[r] as usize] = 8.5;
                l[r + 100] = 7.0;
                l[r + 200] = 6.0;
                l
            })
            .collect();
        // Rank 1 factors: pred(id) is 1 on the pick before, succ(id) is 1
        // on a pick, so a pick scores `gate` more after the pick before.
        let mut pred = |id: u32, out: &mut [f32]| {
            out[0] = f32::from(id == ANCHOR || PICKS.contains(&id));
        };
        let mut succ = |id: u32, out: &mut [f32]| out[0] = f32::from(PICKS.contains(&id));
        let mut lattice = Lattice { rows: Vec::new() };
        let mut before: Vec<u32> = vec![ANCHOR];
        let mut argmax = Vec::new();
        for (r, logits) in rows.iter().enumerate() {
            let cands = top_k(logits, 4, r + 1).expect("defined");
            argmax.push(cands[0].0);
            lattice.rows.push(
                score_row(r + 1, &cands, &before, &[2.0], &mut pred, &mut succ).expect("defined"),
            );
            before = cands.iter().map(|c| c.0).collect();
        }
        assert_eq!(argmax, DECOYS.to_vec());
        assert_eq!(walk(&lattice, 0.0).expect("defined"), PICKS.to_vec());
    }

    /// The card's order is not the sequential sum: a lane's chain loses
    /// the small values behind a large one in its own lane, and the
    /// butterfly keeps those that sit in other lanes and meet first.
    #[test]
    fn lane_dot_sums_in_the_cards_order() {
        let mut succ = vec![0.0f32; CARD_RANK];
        let cond = vec![1.0f32; CARD_RANK];
        succ[0] = 16_777_216.0;
        for e in [8, 16, 24] {
            succ[e] = 1.0;
        }
        // Lanes 0..4 hold 2^24, 1, 1, 1; the offset-2 step meets 2^24 and 1
        // (a tie, rounded to 2^24), the offset-1 step adds the two ones.
        assert_eq!(lane_dot(&succ, &cond), Ok(16_777_218.0));
        let sequential: f32 = succ.iter().zip(&cond).map(|(a, b)| a * b).sum();
        assert_eq!(sequential, 16_777_219.0 - 3.0);
        // The same values in one lane: the chain drops every one.
        let mut one_lane = vec![0.0f32; CARD_RANK];
        one_lane[0] = 16_777_216.0;
        one_lane[1..4].fill(1.0);
        assert_eq!(lane_dot(&one_lane, &cond), Ok(16_777_216.0));
    }

    #[test]
    fn lane_dot_is_the_dot_to_f32_rounding_and_refuses_other_widths() {
        let succ: Vec<f32> = (0..CARD_RANK)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 16.0)
            .collect();
        let cond: Vec<f32> = (0..CARD_RANK)
            .map(|i| ((i * 53 % 97) as f32 - 48.0) / 32.0)
            .collect();
        let exact: f64 = succ
            .iter()
            .zip(&cond)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let got = lane_dot(&succ, &cond).expect("defined");
        assert!((f64::from(got) - exact).abs() < 1e-4, "{got} vs {exact}");
        assert_eq!(
            lane_dot(&succ[..255], &cond),
            Err(SelectorError::Width {
                what: "successor",
                have: 255,
                want: CARD_RANK
            })
        );
        assert!(matches!(
            lane_dot(&succ, &cond[1..]),
            Err(SelectorError::Width {
                what: "conditioned predecessor",
                ..
            })
        ));
    }
}
