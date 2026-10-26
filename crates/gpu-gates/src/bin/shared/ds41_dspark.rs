//! Where the DSpark draft a V4.1 run serves comes from: its file
//! (`BLOOMERY_DSPARK_MODEL`) and its card (`BLOOMERY_DSPARK_CARD`). The draft
//! itself is the session's (`app::arch::deepseek41::Ds41Draft`).
//!
//! The draft's card is a card of the plan, by its place in the plan's device
//! list (`Place::draft_on`): unset, the first tier card (`--place bp`), else
//! the visible device of the fewest bytes (the stage when it is that card, as
//! on one card or under `gate`; beside plan (a) the 3090, which joins the plan
//! as a reserve-only member and holds no expert); the lever names one card as
//! a `--place` list word names it, found among this process's visible
//! devices. The plan sets the draft's bytes aside on that card before its
//! experts ([`draft_seat`]), and the load opens on that device
//! (`ds41_draft::open_dspark`).

use std::path::PathBuf;

use std::path::Path;

use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::generate::{DraftCard, DraftSeat, Place};
use bloomery_gpu_gates::tier::Tier;
use gguf::{Gguf, Split};
use model::arch::dspark::{self, DraftCardBytes, DraftHparams};
use model::placement::workstation;

/// The draft file: `$BLOOMERY_DSPARK_MODEL`, which the recipes export from
/// the V4.1 profile's `DSPARK_MODEL` (`tools/ref/models/deepseek41.sh`), and under
/// `ref-paths.sh`'s fixture tier the fixture's own draft. In the fixture tier the file must be a
/// whole fixture ([`bloomery_gpu_gates::tier`]): a run whose target is a fixture never reads the
/// real draft, whose `target_layers` point past the fixture's layers.
pub fn draft_path() -> Result<PathBuf, GateError> {
    let path = match std::env::var_os("BLOOMERY_DSPARK_MODEL") {
        Some(p) if !p.is_empty() => PathBuf::from(p),
        _ => {
            return Err(
                "BLOOMERY_DSPARK_MODEL unset — export the V4.1 profile's DSPARK_MODEL \
                  (`. tools/ref/ref-paths.sh` under BLOOMERY_MODEL=deepseek41), as the dspark \
                  recipes do"
                    .into(),
            );
        }
    };
    let tier = Tier::from_env()?;
    if tier == Tier::Fixture {
        let g = Gguf::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        tier.check_file(&path, g.iter_kv().map(|(k, _)| k))?;
    }
    Ok(path)
}

/// The draft file's hyperparameters, read from its header alone.
pub fn draft_hparams() -> Result<(Split, DraftHparams), GateError> {
    let path = draft_path()?;
    let draft = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let hp = DraftHparams::read(&draft)?;
    Ok((draft, hp))
}

/// The draft's card under `place` on this process's devices
/// ([`Place::draft_on`] of `BLOOMERY_DSPARK_CARD`, a flag's override before the
/// environment's value, and the census): its place
/// in the plan's device list, its device, and whether it joins the plan as a
/// reserve-only member.
pub fn draft_card(place: Place) -> Result<DraftCard, GateError> {
    let set = match bloomery_levers::override_of("BLOOMERY_DSPARK_CARD") {
        Some(v) => Some(v.to_owned()),
        None => match std::env::var("BLOOMERY_DSPARK_CARD") {
            Err(std::env::VarError::NotPresent) => None,
            Ok(v) => Some(v),
            Err(e) => return Err(format!("BLOOMERY_DSPARK_CARD: {e}").into()),
        },
    };
    place.draft_on(set.as_deref(), &bloomery_gpu::census()?)
}

/// The seat `place`'s plan makes for the DSpark draft `draft`: its resident
/// bytes ([`draft_bytes`]) reserved on the card it sits on ([`draft_card`]),
/// before any load. A load of the draft always has one: the card is the
/// stage, a tier, or a reserve-only member of the plan.
pub fn draft_seat(place: Place, draft: &Split, target: &Path) -> Result<DraftSeat, GateError> {
    Ok(draft_card(place)?.seat(draft_bytes(draft, target)?))
}

/// The draft's resident bytes on its card from the two files' headers
/// (`model::arch::dspark::card_bytes`, the target's head and mask row read
/// from `target`).
pub fn draft_bytes(draft: &Split, target: &Path) -> Result<u64, GateError> {
    let t = Split::open(target).map_err(|e| format!("open {}: {e}", target.display()))?;
    Ok(plan_bytes(draft, &t)?.total())
}

/// The DSpark draft's card bytes from the two files' headers, on the plan's
/// allocation granule ([`workstation::GRANULE`]): the one formula
/// [`draft_bytes`] sums and a load is held to (`ds41_draft::open_dspark`).
pub fn plan_bytes(draft: &Split, target: &Split) -> Result<DraftCardBytes, GateError> {
    dspark::card_bytes(draft, target, workstation::GRANULE)
        .map_err(|e| format!("the DSpark draft's card bytes: {e}").into())
}
