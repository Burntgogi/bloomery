//! The width rule of the split prompt walk: whether a prompt call's unit of
//! `m` rows is walked as two units — the second unit's front enqueued while
//! the host serves the first's — or as the one unit the walk takes today. A
//! pure function of the width and of costs the caller brings; no card, load
//! or model name enters it, and every cost is an input ([`SplitCosts`]).
//!
//! A split holds the card busy under the host's serve, so it saves up to the
//! card's time over the unit's rows, [`SplitCosts::card_row_us`] times `m`.
//! It costs the host side what one more layer of the walk adds: a second
//! union call ([`SplitCosts::call_us`] and, per expert that call lists,
//! [`SplitCosts::listing_us`]) and the enqueue of the second unit's front
//! ([`SplitCosts::enqueue_us`]). [`pays`] splits when the saving clears that
//! cost by [`MARGIN`].
//!
//! The costs are the caller's, fitted from the records of the loads the rule
//! prices ([`LoadKind::Resident`]); [`refuse`] says by name why the rule
//! prices no other load.

use std::fmt;

/// The factor the card's time must clear the split's added cost by: the
/// fitted costs carry error, and a split that only just pays is not worth the
/// second union call's variance.
pub const MARGIN: f64 = 1.5;

/// What the rule reads of a load, per layer of one walk. Every field is in
/// microseconds, the card's per row of the unit and the rest per layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitCosts {
    /// The card's time one layer takes per row of the unit [µs per row]: the
    /// time the split can hide the host's added work under.
    pub card_row_us: f64,
    /// The fixed cost of one more union call a layer [µs]: what a call costs
    /// with nothing listed.
    pub call_us: f64,
    /// The cost of one expert a union call lists [µs per expert].
    pub listing_us: f64,
    /// The host's enqueue of the second unit's front a layer [µs].
    pub enqueue_us: f64,
}

/// What a load keeps its routed experts on, as far as the rule's costs go.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadKind {
    /// Routed experts on the cards and in host memory alone: the loads the
    /// rule's costs are fitted on.
    Resident,
    /// An expert tier holds routed experts of some layers beside the host: a
    /// tiered layer's host serve leaves its upload to the join, which waits
    /// for the tier's rows, a leg the costs do not cover.
    Tier,
    /// The plan paged routed experts to the NVMe tier: the union call reads
    /// pages, which the costs do not cover.
    Paged,
}

/// What the rule refuses, by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitWalkError {
    /// A cost that is not finite or is negative.
    Cost { name: &'static str },
    /// A walk of no row.
    NoRows,
    /// A router that picks no expert a row.
    NoTopK,
    /// A layer with no host-resident routed expert: it has no union call a
    /// split could add to or overlap.
    NoHostExperts,
    /// A tier load, which the rule does not price ([`refuse`]).
    TierLoad,
    /// An NVMe-paged plan, which the rule does not price ([`refuse`]).
    PagedPlan,
}

impl fmt::Display for SplitWalkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SplitWalkError::Cost { name } => {
                write!(f, "split walk cost {name} must be finite and 0 or more")
            }
            SplitWalkError::NoRows => write!(f, "split walk of 0 rows: nothing to walk"),
            SplitWalkError::NoTopK => {
                write!(f, "split walk over a router that picks 0 experts a row")
            }
            SplitWalkError::NoHostExperts => write!(
                f,
                "split walk of a layer with 0 host-resident routed experts: no union call to overlap"
            ),
            SplitWalkError::TierLoad => write!(
                f,
                "split walk on a tier load: a tiered layer's host serve leaves its upload to the \
                 join after the tier's rows, which the width rule's costs do not cover"
            ),
            SplitWalkError::PagedPlan => write!(
                f,
                "split walk on an NVMe-paged plan: the union call reads pages, which the width \
                 rule's costs do not cover"
            ),
        }
    }
}

impl std::error::Error for SplitWalkError {}

/// The loads the rule prices: a resident load. Refused by name, with the
/// reason, on a tier load and on an NVMe-paged plan — the rule never splits
/// there, whatever the width.
///
/// # Errors
/// [`SplitWalkError::TierLoad`] and [`SplitWalkError::PagedPlan`].
pub fn refuse(load: LoadKind) -> Result<(), SplitWalkError> {
    match load {
        LoadKind::Resident => Ok(()),
        LoadKind::Tier => Err(SplitWalkError::TierLoad),
        LoadKind::Paged => Err(SplitWalkError::PagedPlan),
    }
}

/// Whether a walk of `m` rows pays to split into two units of `⌈m/2⌉` and
/// `⌊m/2⌋` rows: `card_row_us · m ≥ MARGIN · (call_us + listing_us · listed +
/// enqueue_us)`, where `listed = min(host_experts, top_k · ⌊m/2⌋)` — the
/// experts the second unit's union call can list, at most its rows' picks.
///
/// `host_experts` is **one layer's** host-resident routed experts, never the
/// model's total over its layers: a union call lists experts of the layer it
/// serves, so a sum over layers prices a call far wider than any layer's and
/// turns the answer false at every width.
///
/// A walk of one row has no second unit to hold: `Ok(false)`.
///
/// # Errors
/// Every refusal names its input; see [`SplitWalkError`].
pub fn pays(
    m: usize,
    costs: &SplitCosts,
    host_experts: usize,
    top_k: usize,
) -> Result<bool, SplitWalkError> {
    for (name, cost) in [
        ("card_row_us", costs.card_row_us),
        ("call_us", costs.call_us),
        ("listing_us", costs.listing_us),
        ("enqueue_us", costs.enqueue_us),
    ] {
        if !(cost.is_finite() && cost >= 0.0) {
            return Err(SplitWalkError::Cost { name });
        }
    }
    if m == 0 {
        return Err(SplitWalkError::NoRows);
    }
    if top_k == 0 {
        return Err(SplitWalkError::NoTopK);
    }
    if host_experts == 0 {
        return Err(SplitWalkError::NoHostExperts);
    }
    if m < 2 {
        return Ok(false);
    }
    let listed = host_experts.min(top_k.saturating_mul(m / 2));
    let added = costs.call_us + costs.listing_us * listed as f64 + costs.enqueue_us;
    Ok(costs.card_row_us * m as f64 >= MARGIN * added)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The router's picks a row of the model the rule's costs were first
    /// fitted for.
    const TOP_K: usize = 10;

    /// The 3090's band edge hardest for a split that pays: its row time at
    /// the fitted value and every host cost at the top of its fitted band.
    const EDGE_3090: SplitCosts = SplitCosts {
        card_row_us: 7.69,
        call_us: 2600.0,
        listing_us: 5.7,
        enqueue_us: 920.0,
    };
    /// One layer's host-resident routed experts on the 3090 server plan, and
    /// the same plan's over all its layers (both derived from the plan, not
    /// measured).
    const LAYER_HOST_EXPERTS: usize = 423;
    const PLAN_HOST_EXPERTS: usize = 20_307;

    /// The 5090's band edge hardest for a split that does not pay: its row
    /// time at the fitted value and every host cost at the bottom of its
    /// fitted band.
    const EDGE_5090: SplitCosts = SplitCosts {
        card_row_us: 2.61,
        call_us: 1200.0,
        listing_us: 0.0,
        enqueue_us: 470.0,
    };

    /// The 3090 at the widest walk: its card time over 4096 rows is more
    /// than the margin times the host's added cost with every cost at the
    /// top of its band, the edge hardest for a split that pays.
    #[test]
    fn the_3090_splits_a_wide_walk_at_the_hardest_band_edge() {
        assert_eq!(pays(4096, &EDGE_3090, LAYER_HOST_EXPERTS, TOP_K), Ok(true));
    }

    /// The 5090 at a narrow walk: its card time over 512 rows is below the
    /// margin times the host's added cost with every cost at the bottom of
    /// its band, the edge hardest for a split that does not.
    #[test]
    fn the_5090_keeps_a_narrow_walk_whole_at_the_hardest_band_edge() {
        assert_eq!(pays(512, &EDGE_5090, LAYER_HOST_EXPERTS, TOP_K), Ok(false));
    }

    /// The parameter is one layer's experts: the plan's total over its layers
    /// prices a union call no layer makes and turns the 3090's wide walk
    /// false.
    #[test]
    fn host_experts_is_one_layers_not_the_plans() {
        assert_eq!(pays(4096, &EDGE_3090, PLAN_HOST_EXPERTS, TOP_K), Ok(false));
    }

    /// The listed experts are the layer's host experts or the second unit's
    /// picks, whichever is fewer: a layer with far more host experts than a
    /// narrow walk's second half can route lists only those picks, and a
    /// layer with fewer lists them all.
    #[test]
    fn listed_is_the_fewer_of_the_layers_experts_and_the_second_units_picks() {
        let at = |listing_us| SplitCosts {
            card_row_us: 2.0,
            call_us: 0.0,
            listing_us,
            enqueue_us: 0.0,
        };
        // 2 picks a row over ⌊100/2⌋ rows: 100 listed at most. The card's
        // 200 clears the margin times 100.
        assert_eq!(pays(100, &at(1.0), 1_000_000, 2), Ok(true));
        // At 1.5 a listing the same 100 cost 150, times the margin 225.
        assert_eq!(pays(100, &at(1.5), 1_000_000, 2), Ok(false));
        // A layer of 40 host experts lists 40: 60 times the margin is 90.
        assert_eq!(pays(100, &at(1.5), 40, 2), Ok(true));
    }

    /// The margin is inclusive and exact where the products are: a card time
    /// equal to the margin times the added cost splits, one row fewer does
    /// not.
    #[test]
    fn the_margin_is_inclusive() {
        let c = SplitCosts {
            card_row_us: 3.0,
            call_us: 2000.0,
            listing_us: 0.0,
            enqueue_us: 0.0,
        };
        assert_eq!(pays(1000, &c, 1, 1), Ok(true));
        assert_eq!(pays(999, &c, 1, 1), Ok(false));
    }

    /// A walk of one row has no second unit; a walk of no row is refused.
    #[test]
    fn a_walk_of_one_row_is_not_split_and_none_is_refused() {
        let free = SplitCosts {
            card_row_us: 1e9,
            call_us: 0.0,
            listing_us: 0.0,
            enqueue_us: 0.0,
        };
        assert_eq!(pays(1, &free, 1, 1), Ok(false));
        assert_eq!(pays(2, &free, 1, 1), Ok(true));
        assert_eq!(pays(0, &free, 1, 1), Err(SplitWalkError::NoRows));
    }

    /// Every cost refuses by its own name when not finite or negative, and
    /// the router's and the layer's shape are refused by name.
    #[test]
    fn inputs_out_of_range_are_refused_by_name() {
        let ok = EDGE_3090;
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            let cases = [
                (
                    "card_row_us",
                    SplitCosts {
                        card_row_us: bad,
                        ..ok
                    },
                ),
                ("call_us", SplitCosts { call_us: bad, ..ok }),
                (
                    "listing_us",
                    SplitCosts {
                        listing_us: bad,
                        ..ok
                    },
                ),
                (
                    "enqueue_us",
                    SplitCosts {
                        enqueue_us: bad,
                        ..ok
                    },
                ),
            ];
            for (name, c) in cases {
                let e = pays(4096, &c, LAYER_HOST_EXPERTS, TOP_K).expect_err(name);
                assert_eq!(e, SplitWalkError::Cost { name }, "{name} = {bad}");
                assert!(e.to_string().contains(name), "{e}");
            }
        }
        assert_eq!(
            pays(4096, &ok, LAYER_HOST_EXPERTS, 0),
            Err(SplitWalkError::NoTopK)
        );
        assert_eq!(
            pays(4096, &ok, 0, TOP_K),
            Err(SplitWalkError::NoHostExperts)
        );
    }

    /// A resident load is priced; a tier load and an NVMe-paged plan are
    /// refused, each by a message that names its load.
    #[test]
    fn a_tier_load_and_a_paged_plan_are_never_split() {
        refuse(LoadKind::Resident).expect("a resident load");
        for (load, want, named) in [
            (LoadKind::Tier, SplitWalkError::TierLoad, "tier load"),
            (
                LoadKind::Paged,
                SplitWalkError::PagedPlan,
                "NVMe-paged plan",
            ),
        ] {
            let e = refuse(load).expect_err(named);
            assert_eq!(e, want);
            assert!(e.to_string().contains(named), "{e}");
        }
    }
}
