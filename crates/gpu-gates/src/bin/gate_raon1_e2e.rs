//! The raon1 end-to-end gate: the whole decode chain — the embedding into
//! Raon_mHC's four streams, the RDA and gated latent layers, the dense and
//! routed SiTU blocks, the streams' mean, the head and the argmax — on one
//! card, over the Raon-1-mini Q4_K_M file, one prompt id a step, against the
//! vendor branch's CPU sets (refset family `lcpp-raon1`, `just
//! dump-ref-raon1`).
//!
//! For each prompt set of [`SETS`] (English prose, Korean prose, Rust code):
//!
//! - `top1_is_vendors`: the argmax after the prompt is the set's top-k
//!   table's first id. The top-[`TOP_K`] overlap of the two last-position
//!   rows and `d`, the largest distance of our logit from the vendor's over
//!   the table's ids, are printed beside it.
//! - `logits_are_vendors`: the whole last-position row's relative distance
//!   from the vendor's `result_output` is at most [`LOGITS_REL_BAND`]. It
//!   bounds `d`, so a chain that moves every logit cannot widen the named-tie
//!   band below with its own error.
//! - `greedy_is_vendors`: the greedy continuation is the set's `greedy.tsv`,
//!   id for id, over its whole length, or up to a named tie: a first
//!   divergence at a step whose vendor margin is at most `2d`, the most two
//!   logits each `d` off can close. A divergence past `2d` is red. Either is
//!   printed with the vendor's margin and our own gap between the two ids;
//!   the steps after a tie continue another text and are not compared.
//!
//! Then, for each set, over the same file in the verifying load a seat with a draft opens
//! (`GpuModel::open_verifying`), as a session in graph mode unless named:
//!
//! - `prompt_rows_is_steps`: the prompt as the rows pass (`GpuModel::prompt_rows`) and one id a
//!   step: the same argmax, the same logits row by bits, the same steps after it.
//! - `verify_is_steps`, in eager and in graph mode: after the prompt call, a verify of the plain
//!   run's first 8 ids (and of its first 2) gives the plain run's next ids, each row's logits the
//!   plain step's by bits.
//! - `commit_keeps_steps`, in eager and in graph mode: a verify of 8 rows committed at 1, 3 and 8
//!   rows, then steps: the plain run's ids.
//! - `taps_are_steps`: the rows entering the draft's tapped layers, as a step taps them, are by
//!   bits the rows the rows pass hands its sink over the prompt (every position once, in order)
//!   and the rows a verify taps.
//! - `speculative_is_plain`: Raon-1-mini's DFlash2 draft beside it (`raon1_dflash`: the serve
//!   seat's pick and open), every pass a proposal of 7 ids verified by one pass of 8 rows: the
//!   first [`DRAFTED`] greedy ids are the plain session's (its prompt call, then a step an id), id
//!   for id. The passes, and the proposed ids the target kept, are printed beside it.
//!
//! A step that raises the card's fault word ends the gate by name. A set
//! absent from the box, dumped from another file or build, or unfinished ends
//! the gate by name before the card loads.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_raon1_e2e", gate::run())
}

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "gpu")]
#[path = "shared/greedy.rs"]
mod greedy;

#[cfg(feature = "gpu")]
#[path = "shared/quiet.rs"]
mod quiet;

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the serve seat and two gates share the opener; each takes the parts it reads"
)]
#[path = "shared/raon1_dflash.rs"]
mod raon1_dflash;

#[cfg(feature = "gpu")]
mod gate {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use crate::e2e::{rel, top_ids};
    use crate::greedy::greedy;
    use crate::quiet::Quiet;
    use crate::raon1_dflash::{self, Pick, RaonDraft};
    use app::Session;
    use bloomery_gpu::arch::raon1::{Body, HC_OUT, STREAMS};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{Gpu, GpuError, GpuModel};
    use bloomery_gpu_gates::{
        GateError, RefManifest, bits_equal, checks_failed, data_dir, open_split, ref_model_path,
        ref_tensor_logical_in, verdict,
    };
    use model::arch::Arch;
    use model::arch::dflash2::Hparams as DraftHparams;
    use refset::arch::prompt_tables::{self, Greedy, Top};
    use refset::arch::raon1::{A, B, C, LCPP};
    use runtime::{Advance, RowLogits, Speculative, Target, Verify, Want};

    /// The sets the step path is held to, in the order they run.
    const SETS: &[&str] = &[A, B, C];

    /// Rows of each latent layer's cache: the longest prompt and its
    /// continuation fit.
    const CTX: usize = 256;

    /// The file's layers: the last one's route is printed.
    const LAYERS: usize = 25;

    /// The last-position ids whose overlap is printed.
    const TOP_K: usize = 5;

    /// The last-position logits row's relative distance from the vendor's.
    /// PIN(2026-10-21): measured, not derived: the chain at 3.33e-2, 3.87e-2
    /// and 3.16e-2 on sets a, b and c; SiTU's gate beta 3 for 4 at 7.90e-2,
    /// 8.88e-2 and 1.98e-1.
    const LOGITS_REL_BAND: f64 = 5.0e-2;

    /// The greedy ids `speculative_is_plain` compares a set: past the vendor's table, so a run
    /// holds several dozen passes.
    const DRAFTED: usize = 64;

    /// A verify of a proposal: the token at the target's position, then the draft's 7 ids.
    const VERIFY_ROWS: usize = 8;

    /// The steps `prompt_rows_is_steps` and `commit_keeps_steps` run after the call they check.
    const AFTER: usize = 4;

    /// The rows `commit_keeps_steps` keeps of a verify of [`VERIFY_ROWS`].
    const KEEPS: [usize; 3] = [1, 3, VERIFY_ROWS];

    type Seat = Session<Body>;
    type Spec = Speculative<RaonDraft, VERIFY_ROWS>;

    /// One set's prompt and its two tables.
    struct Set {
        name: &'static str,
        prompt: Vec<u32>,
        top: Vec<Top>,
        greedy: Greedy,
        /// The vendor's whole logits row at the prompt's last position
        /// (`result_output`).
        logits: Vec<f32>,
        /// The vendor's route of the last layer's feed-forward site at the
        /// prompt's last position: `pre`, `post`, `comb` in [`HC_OUT`]'s order;
        /// `None` for a set dumped without those taps.
        route: Option<Vec<f32>>,
    }

    /// The last `n` values of tap `name`'s occurrence `occurrence` in `man`.
    fn last_of(
        man: &RefManifest,
        name: &str,
        occurrence: u32,
        n: usize,
    ) -> Result<Vec<f32>, GateError> {
        let row = man.tensor(name, occurrence)?;
        let all = ref_tensor_logical_in(&man.dir, row)?;
        let at = all
            .len()
            .checked_sub(n)
            .ok_or_else(|| format!("{name}: {} values, want {n}", all.len()))?;
        Ok(all[at..].to_vec())
    }

    fn open_set(name: &'static str) -> Result<Set, GateError> {
        let man = RefManifest::open(&LCPP.path_in(&data_dir(), name), &LCPP)?;
        let row = man.tensor("result_output", 0)?;
        let all = ref_tensor_logical_in(&man.dir, row)?;
        let vocab = usize::try_from(row.ne[0])?;
        if vocab == 0 || all.len() % vocab != 0 {
            return Err(format!("{name}: result_output is {:?}", row.ne).into());
        }
        let logits = all[all.len() - vocab..].to_vec();
        let last = LAYERS - 1;
        let route = match man.tensor(&format!("hc_comb-{last}"), 1) {
            Err(_) => None,
            Ok(_) => {
                let mut r = last_of(&man, &format!("hc_pre-{last}"), 1, STREAMS)?;
                r.extend(last_of(&man, &format!("hc_post-{last}"), 1, STREAMS)?);
                r.extend(last_of(
                    &man,
                    &format!("hc_comb-{last}"),
                    1,
                    STREAMS * STREAMS,
                )?);
                Some(r)
            }
        };
        Ok(Set {
            name,
            prompt: prompt_tables::tokens(&man)?.to_vec(),
            top: prompt_tables::topk(&man)?,
            greedy: prompt_tables::greedy(&man)?,
            logits,
            route,
        })
    }

    /// Our route of the last layer's feed-forward site at the prompt's last
    /// position beside the vendor's: `pre`'s and `post`'s relative distances,
    /// and `comb`'s from the vendor's and from its transpose (the two
    /// orientations of `comb[src · 4 + dst]`).
    fn print_route(m: &GpuModel<Body>, s: &Set) -> Result<(), GateError> {
        let Some(theirs) = &s.route else {
            println!(
                "route set={} has no layer-{} route taps",
                s.name,
                LAYERS - 1
            );
            return Ok(());
        };
        let ours = m.body("gate_raon1_e2e")?.last_route(m.gpu().stream())?;
        if ours.len() != HC_OUT || theirs.len() != HC_OUT {
            return Err(format!("routes of {} and {} values", ours.len(), theirs.len()).into());
        }
        let k = 2 * STREAMS;
        let transposed: Vec<f32> = (0..STREAMS * STREAMS)
            .map(|i| theirs[k + (i % STREAMS) * STREAMS + i / STREAMS])
            .collect();
        println!(
            "route set={} layer={} pre_rel={:.3e} post_rel={:.3e} comb_rel={:.3e} \
             comb_rel_transposed={:.3e}",
            s.name,
            LAYERS - 1,
            rel(&ours[..STREAMS], &theirs[..STREAMS]),
            rel(&ours[STREAMS..k], &theirs[STREAMS..k]),
            rel(&ours[k..], &theirs[k..]),
            rel(&ours[k..], &transposed),
        );
        Ok(())
    }

    /// The first step whose id is not the vendor's, and our logit gap
    /// between our id and the vendor's there.
    type Divergence = Option<(usize, f32)>;

    /// The greedy ids from `first` (the argmax of `row`, the prompt's last
    /// logits), each fed back, until the set's length or the first id that
    /// is not the vendor's: that step and our logit gap between our id and
    /// the vendor's there.
    fn continue_greedy(
        m: &mut GpuModel<Body>,
        s: &Set,
        first: u32,
        mut row: Vec<f32>,
    ) -> Result<(Vec<u32>, Divergence), GateError> {
        let mut ids = vec![first];
        loop {
            let i = ids.len() - 1;
            if ids[i] != s.greedy.ids[i] {
                let gap = row[ids[i] as usize] - row[s.greedy.ids[i] as usize];
                return Ok((ids, Some((i, gap))));
            }
            if ids.len() == s.greedy.ids.len() {
                return Ok((ids, None));
            }
            ids.push(m.step(&[ids[i]])?);
            row = m.logits()?;
        }
    }

    /// The DFlash2 draft's file beside the target, as the serve seat picks it, and the layers it
    /// taps.
    fn draft_file() -> Result<(PathBuf, Vec<usize>), GateError> {
        let target = ref_model_path()?;
        let path = match raon1_dflash::pick(None, &target)? {
            Pick::File(path, _) => path,
            Pick::Absent(dirs) => {
                return Err(format!(
                    "no DFlash2 draft beside {} (looked at {dirs:?})",
                    target.display()
                )
                .into());
            }
        };
        let split =
            gguf::Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let hp = DraftHparams::read(&split).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok((path, hp.target_layers))
    }

    /// The DFlash2 draft of the file at `path` on the seat's card, as the serve seat opens it.
    fn open_draft(seat: &Seat, path: &Path) -> Result<RaonDraft, GateError> {
        let file = Arc::new(open_split(Arch::Raon1, "gate-gpu-raon1-e2e")?);
        let draft = raon1_dflash::open(Gpu::new()?, path, (file, seat.model()), CTX, "gate")?;
        println!(
            "draft {} resident_bytes={}",
            path.display(),
            draft.resident_bytes()
        );
        Ok(draft)
    }

    /// `n` greedy ids of `prompt` from a reset seat through `spec`, every pass a proposal; the
    /// passes, and the proposed ids the target kept.
    fn drafted(
        spec: &mut Spec,
        seat: &mut Seat,
        prompt: &[u32],
        n: usize,
    ) -> Result<(Vec<u32>, usize, usize), GateError> {
        Target::reset(seat)?;
        let first = Advance::prompt(spec, seat, prompt)?;
        Advance::begin(spec, seat, prompt, first)?;
        let mut ids = vec![first];
        let (mut passes, mut kept) = (0, 0);
        while ids.len() < n {
            let last = ids[ids.len() - 1];
            let before = ids.len();
            let c = Advance::pass(spec, seat, last, &mut ids)?;
            if !c.proposed || c.rows != VERIFY_ROWS || ids.len() - before != c.kept {
                return Err(format!(
                    "a pass of {} rows (proposed {}) kept {} and appended {}: every pass is a \
                     proposal of {} verified by {VERIFY_ROWS} rows",
                    c.rows,
                    c.proposed,
                    c.kept,
                    ids.len() - before,
                    VERIFY_ROWS - 1
                )
                .into());
            }
            passes += 1;
            kept += c.kept - 1;
        }
        ids.truncate(n);
        Ok((ids, passes, kept))
    }

    /// Clause `prompt_rows_is_steps` on set `s`: the prompt one id a step and as the rows pass
    /// (`GpuModel::prompt_rows`), each from a reset model: the same argmax, the same logits row by
    /// bits, and the same [`AFTER`] steps after it.
    fn prompt_rows_is_steps(m: &mut GpuModel<Body>, s: &Set) -> Result<bool, GateError> {
        let mut run = |rows: bool| -> Result<(Vec<u32>, Vec<f32>), GateError> {
            m.reset()?;
            let first = if rows {
                m.prompt_rows(&s.prompt)?
            } else {
                m.step(&s.prompt)?
            };
            let row = m.logits()?;
            let mut ids = vec![first];
            while ids.len() <= AFTER {
                let last = ids[ids.len() - 1];
                ids.push(m.step(&[last])?);
            }
            Ok((ids, row))
        };
        let (steps, steps_row) = run(false)?;
        let (rows, rows_row) = run(true)?;
        let same_row = bits_equal(&steps_row, &rows_row);
        let pass = steps == rows && same_row;
        println!(
            "prompt_rows_is_steps set={} steps={steps:?} rows={rows:?} logits_same_bits={same_row} {}",
            s.name,
            verdict(pass)
        );
        Ok(pass)
    }

    /// A verify of `M` rows of the plain run `plain` from the prompt's end, through the session
    /// from a reset: whether the prompt call's id is `plain[0]` and the verify's ids are
    /// `plain[1..=M]`, and whether each row's logits are, by bits, `rows` one id later (`rows[r]`
    /// is the row whose argmax is `plain[r]`). The verify is left waiting for its commit.
    fn verified<const M: usize>(
        seat: &mut Seat,
        s: &Set,
        (plain, rows): (&[u32], &[Vec<f32>]),
    ) -> Result<(bool, bool), GateError> {
        Target::reset(seat)?;
        let first = Target::prompt(seat, &s.prompt, Want::Argmax)?.argmax();
        let out = Verify::verify::<M>(seat, <[u32; M]>::try_from(&plain[..M])?)?;
        let ids = first == plain[0] && out[..] == plain[1..=M];
        let mut logits = true;
        for (r, want) in rows[1..=M].iter().enumerate() {
            logits &= bits_equal(RowLogits::row_logits(seat, r)?, want);
        }
        Ok((ids, logits))
    }

    /// Clauses `verify_is_steps` (verifies of [`VERIFY_ROWS`] and 2 rows) and
    /// `commit_keeps_steps` (a verify of [`VERIFY_ROWS`] rows committed at each of [`KEEPS`], then
    /// [`AFTER`] steps, the plain run's ids) on set `s` in the seat's mode `mode`.
    fn verify_is_steps(seat: &mut Seat, s: &Set, mode: &str) -> Result<bool, GateError> {
        let (plain, rows) = greedy(seat, &s.prompt, VERIFY_ROWS + 1 + AFTER, true)?;
        let run = (&plain[..], &rows[..]);
        let mut ok = true;
        for (m, (ids, logits)) in [
            (VERIFY_ROWS, verified::<VERIFY_ROWS>(seat, s, run)?),
            (2, verified::<2>(seat, s, run)?),
        ] {
            let pass = ids && logits;
            println!(
                "verify_is_steps set={} mode={mode} rows={m} ids={ids} logits_same_bits={logits} {}",
                s.name,
                verdict(pass)
            );
            ok &= pass;
        }
        for k in KEEPS {
            let (ids, logits) = verified::<VERIFY_ROWS>(seat, s, run)?;
            Verify::commit(seat, k)?;
            let mut after = vec![plain[k]];
            while after.len() <= AFTER {
                let last = after[after.len() - 1];
                after.push(Target::step(seat, last, Want::Argmax)?.argmax());
            }
            let kept = after[..] == plain[k..=k + AFTER];
            let pass = ids && logits && kept;
            println!(
                "commit_keeps_steps set={} mode={mode} kept={k} after={:?} plain={:?} {}",
                s.name,
                &after[1..],
                &plain[k + 1..=k + AFTER],
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    /// Clause `taps_are_steps` on set `s`, taps on `layers`: the rows a step taps
    /// (`GpuModel::draft_taps` of 1 after each id of the prompt, then of the plain run's first
    /// [`VERIFY_ROWS`] ids) are, by bits, the rows the rows pass hands its sink over the prompt
    /// (`GpuModel::prompt_rows_tapped`, every position once and in order), then the rows a verify
    /// of those ids taps (`draft_taps` of [`VERIFY_ROWS`]). The taps are off again after it.
    fn taps_are_steps(
        m: &mut GpuModel<Body>,
        s: &Set,
        plain: &[u32],
        layers: &[usize],
    ) -> Result<bool, GateError> {
        m.set_draft_taps(layers)?;
        let w = m.draft_tap_width();
        m.reset()?;
        let mut stepped = Vec::new();
        for &id in s.prompt.iter().chain(&plain[..VERIFY_ROWS]) {
            m.step(&[id])?;
            stepped.extend(m.draft_taps(1)?);
        }
        m.reset()?;
        let p = s.prompt.len();
        let mut sunk = vec![f32::NAN; p * w];
        let (mut next, mut in_order) = (0usize, true);
        m.prompt_rows_tapped(&s.prompt, p, &mut |first: u32, rows: &[f32]| {
            let (first, n) = (first as usize, rows.len() / w);
            in_order &= first == next && first + n <= p && rows.len() == n * w;
            if in_order {
                sunk[first * w..(first + n) * w].copy_from_slice(rows);
            }
            next = first + n;
            Ok::<_, GpuError>(())
        })?;
        in_order &= next == p;
        m.step_rows::<VERIFY_ROWS>(<[u32; VERIFY_ROWS]>::try_from(&plain[..VERIFY_ROWS])?)?;
        let verified = m.draft_taps(VERIFY_ROWS)?;
        m.reset()?;
        m.set_draft_taps(&[])?;
        let prompt = in_order && bits_equal(&sunk, &stepped[..p * w]);
        let verify = bits_equal(&verified, &stepped[p * w..]);
        let pass = prompt && verify;
        println!(
            "taps_are_steps set={} layers={layers:?} width={w} prompt_rows={p} in_order={in_order} \
             prompt_same_bits={prompt} verify_same_bits={verify} {}",
            s.name,
            verdict(pass)
        );
        Ok(pass)
    }

    /// The clauses over the verifying load ([`GpuModel::open_verifying`], the one a seat with a
    /// draft opens), as a graph-mode session, on every set: `prompt_rows_is_steps`,
    /// `verify_is_steps` and `commit_keeps_steps` in eager and in graph mode, `taps_are_steps`,
    /// then `speculative_is_plain` with the draft beside it.
    fn verifying(sets: &[Set]) -> Result<bool, GateError> {
        let file = open_split(Arch::Raon1, "gate-gpu-raon1-e2e")?;
        let mut m = GpuModel::<Body>::open_verifying(Gpu::new()?, file, CTX)?;
        m.set_mode(StepMode::Graph);
        m.capture_step()?;
        let mut seat = Seat::from_model(m, u32::try_from(CTX)?);
        let (path, layers) = draft_file()?;
        let mut ok = true;
        for s in sets {
            ok &= prompt_rows_is_steps(seat.model_mut(), s)?;
            for (mode, word) in [(StepMode::Eager, "eager"), (StepMode::Graph, "graph")] {
                seat.model_mut().set_mode(mode);
                ok &= verify_is_steps(&mut seat, s, word)?;
            }
            let (plain, _) = greedy(&mut seat, &s.prompt, VERIFY_ROWS, false)?;
            ok &= taps_are_steps(seat.model_mut(), s, &plain, &layers)?;
            Target::reset(&mut seat)?;
        }
        Ok(speculative_is_plain(seat, &path, sets)? && ok)
    }

    /// Clause `speculative_is_plain` on every set, over `seat` with the draft of the file at
    /// `path` beside it.
    fn speculative_is_plain(mut seat: Seat, path: &Path, sets: &[Set]) -> Result<bool, GateError> {
        let draft = open_draft(&seat, path)?;
        let mut spec = seat.with_draft::<RaonDraft, VERIFY_ROWS>(draft, &mut Quiet)?;
        let mut ok = true;
        for s in sets {
            let (plain, _) = greedy(&mut seat, &s.prompt, DRAFTED, false)?;
            let (ids, passes, kept) = drafted(&mut spec, &mut seat, &s.prompt, DRAFTED)?;
            let differ = plain.iter().zip(&ids).position(|(a, b)| a != b);
            let pass = differ.is_none() && plain.len() == ids.len();
            let at = match differ {
                None => "none".to_owned(),
                Some(i) => format!("step {i} plain={} drafted={}", plain[i], ids[i]),
            };
            println!(
                "speculative_is_plain set={} ids={DRAFTED} passes={passes} kept_proposed={kept} \
                 first_difference={at} {}",
                s.name,
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let sets: Vec<Set> = SETS
            .iter()
            .map(|&n| open_set(n))
            .collect::<Result<_, _>>()?;
        for s in &sets {
            if s.greedy.ids.is_empty() || s.greedy.margins.len() != s.greedy.ids.len() {
                return Err(format!(
                    "{}: a greedy table of {} ids and {} margins",
                    s.name,
                    s.greedy.ids.len(),
                    s.greedy.margins.len()
                )
                .into());
            }
            let need =
                (s.prompt.len() + s.greedy.ids.len()).max(s.prompt.len() + DRAFTED + VERIFY_ROWS);
            if need > CTX {
                return Err(
                    format!("{}: {need} positions, the gate's caches hold {CTX}", s.name).into(),
                );
            }
        }
        let file = open_split(Arch::Raon1, "gate-gpu-raon1-e2e")?;
        let t = std::time::Instant::now();
        let mut m = GpuModel::<Body>::open(Gpu::new()?, file, CTX)?;
        println!(
            "load resident_bytes={} ctx={CTX} layers={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            m.layers().len(),
            t.elapsed().as_secs_f64()
        );
        if m.layers().len() != LAYERS {
            return Err(format!(
                "a file of {} layers; the sets' is {LAYERS}",
                m.layers().len()
            )
            .into());
        }
        let mut ok = true;
        for s in &sets {
            m.reset()?;
            let first = m.step(&s.prompt)?;
            let row = m.logits()?;
            let theirs = s
                .top
                .first()
                .ok_or_else(|| format!("{}: an empty top-k table", s.name))?;
            let ours5 = top_ids(&row, TOP_K);
            let d = s
                .top
                .iter()
                .map(|t| (row[t.id as usize] - t.logit).abs())
                .fold(0.0f32, f32::max);
            let logits_rel = rel(&row, &s.logits);
            let logits_ok = logits_rel <= LOGITS_REL_BAND;
            print_route(&m, s)?;
            let overlap = s
                .top
                .iter()
                .take(TOP_K)
                .filter(|t| ours5.contains(&(t.id as usize)))
                .count();
            let top1 = first == theirs.id;
            println!(
                "top1_is_vendors set={} prompt_ids={} ours={first} vendor={} top{TOP_K}_overlap={overlap}/{TOP_K} \
                 ours_top{TOP_K}={ours5:?} d={d} over {} ids {}",
                s.name,
                s.prompt.len(),
                theirs.id,
                s.top.len(),
                verdict(top1)
            );
            let (ids, diverge) = continue_greedy(&mut m, s, first, row)?;
            let (pass, at) = match diverge {
                None => (true, "none".to_owned()),
                Some((i, our_gap)) => {
                    let margin = s.greedy.margins[i];
                    let tie = margin <= 2.0 * d;
                    let word = if tie { "a named tie" } else { "past 2d" };
                    let at = format!(
                        "step {i} ours={} vendor={} vendor_margin={margin} our_gap={our_gap} \
                         2d={} ({word})",
                        ids[i],
                        s.greedy.ids[i],
                        2.0 * d
                    );
                    (tie, at)
                }
            };
            println!(
                "greedy_is_vendors set={} steps={} first_divergence={at} {}",
                s.name,
                ids.len(),
                verdict(pass)
            );
            println!("greedy set={} ours={ids:?}", s.name);
            println!(
                "logits_are_vendors set={} logits_rel={logits_rel:.4e} band={LOGITS_REL_BAND:.1e} {}",
                s.name,
                verdict(logits_ok)
            );
            ok &= top1 && pass && logits_ok;
        }
        // The plain load's card bytes go back before the verifying load takes its own.
        drop(m);
        ok &= verifying(&sets)?;
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: gate_raon1_e2e on every set of {SETS:?} the argmax after the prompt is the \
             vendor's, the greedy continuation is the vendor's id for id, or up to a named tie, and \
             the drafted greedy run is the plain one id for id"
        );
        Ok(())
    }
}
