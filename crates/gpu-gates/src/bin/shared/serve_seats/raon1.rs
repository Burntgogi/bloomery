//! `bloomery-serve --model raon1` — the llama-server-compatible HTTP API on
//! the Raon-1 engine with one sequence: a Raon-1-mini file (RDA and gated
//! latent layers, SiTU experts, Raon_mHC streams), the whole model on one
//! card, opened as `gate_raon1_e2e` opens it (`GpuModel::<raon1::Body>::open`;
//! the step captured in graph mode).
//!
//!     bloomery-serve --model raon1 [-m PATH | --hf <repo>[:<quant>]]
//!                    [--host 127.0.0.1] [--port 8080] [--ctx C] [--alias NAME]
//!                    [--chat-template-file PATH] [--place W] [--plan]
//!                    [--queue-depth Q] [--api-key KEY] [--api-key-file FNAME]
//!
//! The flags are the nemotron seat's, read the same way: the model's first
//! shard gives the vocabulary, `tokenizer.chat_template` the chat template
//! (`--chat-template-file` replaces it) and `general.name` the alias
//! (`--alias` replaces it; `raon1` for a file that names none).
//!
//! Three facts of the architecture set this seat's shape:
//!
//! - The body holds one sequence: its latent layers' caches of one context
//!   and its RDA layers' states and conv rings. It has no slot rows, so
//!   `--parallel` above 1 is refused by name before any plan.
//! - The prompt runs as steps, one id a position (`GpuModel::step`): the
//!   recurrence is the decode step's, so the states are exact.
//! - An RDA state cannot step back. A request keeps the prefix it shares
//!   with what the slot holds only when that prefix is the whole held
//!   sequence; any other shared prefix is kept 0, the slot reset and the
//!   prompt fed again. A rollback to a position the model does not stand at,
//!   a snapshot and a resume are refused by name, and the seat runs no
//!   prompt cache.
//!
//! `--place W` is the shared placement word (`generate::Place`); unset, the
//! common untiered rule decides on one census reading. A placement the plan
//! refuses (more than one card, a tier card, a card the file does not fit)
//! stops before any load by the plan's own refusal. `--ctx` is the slot's
//! context, chosen by `generate::choose_ctx` as every seat's is.
//!
//! The records on stderr: `place unset`, one `ctx` line, the `plan` record,
//! the `parallel` line, then the `load` record and, once the port is bound,
//! the `listening` record (`record::BLOOMERY_SERVE_RAON1`). `--plan` ends
//! the process after the `parallel` line, nothing loaded.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use bloomery_gpu::arch::raon1::Body;
use bloomery_gpu::model::{ChainBody, StepMode};
use bloomery_gpu::{Gpu, GpuModel};
use bloomery_gpu_gates::bind::{
    ChatSurface, Pass, Seat, SeatEngine, SeatFlags, Vocab, arm_screen, model_props, note_line,
    parallel_line, placement_of, slots_of,
};
use bloomery_gpu_gates::generate::{Place, choose_ctx};
use bloomery_gpu_gates::need;
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::{GateError, gpu_census, ref_model_path};
use gguf::Split;
use model::arch::Arch;
use model::arch::raon1::place::PlanInputs;
use model::placement::PlanLevers;
use model::placement::ctx_ask::{DEFAULT_SLOTS, probe_of};
use model::placement::workstation::DeviceId;
use serve::{CacheNote, EngineProps, Saved, ServeError};
use tokenizer::Tokenizer;

const NAME: &str = "bloomery-serve-raon1";

const USAGE: &str = "usage: bloomery-serve --model raon1 [-m PATH | --hf <repo>[:<quant>]] \
                     [--host H] [--port P] [--ctx C] [--alias NAME] [--chat-template-file PATH] \
                     [--place W] [--plan] [--queue-depth Q] [--api-key KEY] \
                     [--api-key-file FNAME]";

/// The levers this seat acts on: the plan's card budget, the start screen and
/// how a structured-output request's fixed runs are fed.
pub const ACTS_ON: &[&str] = &[
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::SCREEN,
    bloomery_levers::SCHEMAJUMP,
];

/// Why the seat takes no sequence state: it runs no prompt cache.
const NO_CACHE: &str = "the raon1 seat runs no prompt cache";

/// Why a prefix shorter than the held sequence is not kept.
const NO_STEP_BACK: &str =
    "the RDA state cannot step back: a prefix is kept only where the sequence stands";

/// What the engine thread opens the model with.
struct OpenSeat {
    card: String,
    device: Option<DeviceId>,
    path: PathBuf,
    /// The slot's positions the latent caches are sized for.
    ctx: usize,
}

/// The model, in graph mode with its step captured.
struct Raon1 {
    m: GpuModel<Body>,
}

impl Raon1 {
    /// The model of the file at `a.path` on the planned card, the `load`
    /// record on stderr.
    fn open(a: OpenSeat) -> Result<Raon1, GateError> {
        let t = Instant::now();
        let file = Split::open(&a.path).map_err(|e| format!("open {}: {e}", a.path.display()))?;
        let opened = t.elapsed();
        let mut m =
            GpuModel::<Body>::open(Gpu::open_card_after_hook(&a.card, a.device)?, file, a.ctx)?;
        m.set_mode(StepMode::Graph);
        let nodes = m.capture_step()?;
        let total = t.elapsed();
        Record::new(&record::LOAD_RAON1)
            .w("arch", Body::arch().name())
            .u("resident_bytes", m.resident_bytes())
            .u("ctx", m.ctx_max())
            .u("slots", 1)
            .u("layers", m.layers().len())
            .u("graph_nodes", nodes)
            .f("load_s", total.as_secs_f64())
            .eprint();
        record::eprint_load_phases(&m, total, opened);
        Ok(Raon1 { m })
    }
}

impl Seat for Raon1 {
    fn pos(&self) -> usize {
        self.m.pos() as usize
    }

    fn ctx_max(&self) -> usize {
        self.m.ctx_max()
    }

    /// The prompt as steps, one id a position.
    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        Ok(self.m.step(ids)?)
    }

    fn step(&mut self, last: u32) -> Result<u32, GateError> {
        Ok(self.m.step(&[last])?)
    }

    fn logits_into(&self, row: &mut [f32]) -> Result<(), GateError> {
        Ok(self.m.logits_into(row)?)
    }

    /// The model's reset: the RDA states and conv rings back to zero, the
    /// position to 0.
    fn reset(&mut self) -> Result<(), GateError> {
        Ok(self.m.reset()?)
    }

    /// Never asked for a position other than the one the model stands at
    /// ([`Seat::keep`] grants no other); any other is refused by name.
    fn rollback(&mut self, pos: u32) -> Result<(), GateError> {
        Err(format!(
            "a rollback to position {pos} from {}: {NO_STEP_BACK}",
            self.m.pos()
        )
        .into())
    }

    /// `n` when it is the whole held sequence, else nothing: the caller then
    /// resets the slot and feeds the prompt again.
    fn keep(&self, n: usize) -> (usize, Option<String>) {
        if n == self.pos() {
            (n, None)
        } else {
            (0, Some(NO_STEP_BACK.to_owned()))
        }
    }

    /// Nowhere: no prompt call is cut for a prefix a later request keeps.
    fn splits(&self, _first: usize, _end: usize, _marks: &[usize]) -> Vec<usize> {
        Vec::new()
    }

    /// Never asked: the seat's prompt cache budget is 0.
    fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
        Err(format!("a snapshot: {NO_CACHE}").into())
    }

    fn resume(&mut self, _state: &dyn Saved) -> Result<(), GateError> {
        Err(format!("a resume: {NO_CACHE}").into())
    }

    fn note(note: &CacheNote) {
        note_line(NAME, note);
    }
}

/// Loads the model and serves until the listener or the engine fails;
/// `Ok` carries why the server ended.
pub fn run(args: &[String]) -> Result<ServeError, GateError> {
    let levers = bloomery_levers::at_main(ACTS_ON)?;
    record::at_main(NAME, record::BLOOMERY_SERVE_RAON1);
    arm_screen(levers.screen());
    let a = SeatFlags::parse(args, USAGE, |_, _| Ok(false))?;
    let path = ref_model_path()?;
    let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let first = split
        .shard(0)
        .ok_or_else(|| format!("{} opened with no shard", path.display()))?;
    let arch = Arch::detect(first).map_err(|e| format!("{}: {e}", path.display()))?;
    if arch != Arch::Raon1 {
        return Err(format!(
            "{} is a {} file; the raon1 seat serves raon1 files",
            path.display(),
            arch.name()
        )
        .into());
    }
    if let Some(n) = a.parallel.filter(|&n| n > 1) {
        return Err(format!(
            "--parallel {n}: the raon1 body holds one sequence and no resident slots; the seat \
             serves --parallel 1"
        )
        .into());
    }
    let chat = ChatSurface::read(
        &path,
        split.value("tokenizer.chat_template"),
        split.value("general.name"),
        a.template_file.path(),
        "raon1",
    )?;
    // The coverage list has no raon1 program yet (`placement::accepts::program_of`), so the plan
    // reads the file as it is; the body's load refuses by name every part it does not run.
    let inputs = PlanInputs::describe(&split)?;
    let plan_levers = PlanLevers::from_levers(&levers)?;
    let (slots, from) = slots_of(a.parallel, a.ctx.is_some(), DEFAULT_SLOTS)?;
    let census = gpu_census::census()?;
    let chosen = Place::choose_untiered(a.place, &census, "raon1")?;
    chosen.record().eprint();
    let place = chosen.place;
    let machine = place.machine_untiered()?(inputs.model.layers);
    let asked = choose_ctx(&split, (a.ctx, slots), (None, None), |c| {
        probe_of(inputs.plan(&machine, c, &plan_levers))
            .map_err(|e| format!("--place {}: {e}", place.name()))
    })?;
    eprintln!("{}", asked.line());
    let slot_ctx = usize::try_from(asked.slot_ctx)?;
    let plan = asked.plan;
    record::plan(place.name(), &machine, &plan).eprint();
    need::print(&plan, 0, 0, true)?;
    parallel_line(slots, slot_ctx, from, Pass::One);
    if a.plan_only {
        std::process::exit(0);
    }
    let props = EngineProps {
        model: Some(model_props(&split, &inputs.model)),
        placement: placement_of(NAME, &machine, &plan),
        ..EngineProps::default()
    };
    let card = &machine.cards[0];
    let open = OpenSeat {
        card: card.name.clone(),
        device: card.device,
        path: path.clone(),
        ctx: slot_ctx,
    };
    let device = card.name.clone();
    drop(plan);
    drop(split);
    let vocab = Arc::new(Vocab::new(Tokenizer::from_gguf(&path)?)?);
    let engine = SeatEngine::spawn(move || Raon1::open(open), slot_ctx, vocab, device, props, 0)?;
    let server = a.bind(engine, chat, &path, slots, &levers, None)?;
    Record::new(&record::LISTENING_RAON1)
        .w("place", place.name())
        .u("ctx", slot_ctx)
        .u("slots", slots)
        .w("addr", server.local_addr()?)
        .eprint();
    Ok(server.run())
}
