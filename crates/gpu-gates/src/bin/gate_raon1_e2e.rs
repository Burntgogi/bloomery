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
mod gate {
    use crate::e2e::{rel, top_ids};
    use bloomery_gpu::arch::raon1::{Body, HC_OUT, STREAMS};
    use bloomery_gpu::{Gpu, GpuModel};
    use bloomery_gpu_gates::{
        GateError, RefManifest, checks_failed, data_dir, open_split, ref_tensor_logical_in, verdict,
    };
    use model::arch::Arch;
    use refset::arch::prompt_tables::{self, Greedy, Top};
    use refset::arch::raon1::{A, B, C, LCPP};

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
            let need = s.prompt.len() + s.greedy.ids.len();
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
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: gate_raon1_e2e on every set of {SETS:?} the argmax after the prompt is the \
             vendor's and the greedy continuation is the vendor's id for id, or up to a named tie"
        );
        Ok(())
    }
}
