//! The serve gate's clauses: the one owner of what a seat of a body that runs
//! one sequence on one card is held to — it refuses a second slot, it loads
//! the file at the context it was asked for, and it answers a chat turn with
//! its reasoning split at the model's own tags and its turn ending at the
//! model's own end token.
//!
//! A seat gives the clauses its own facts ([`Clauses`]): the model word, the
//! records its server prints, the load and the listening record, the
//! architecture word its load record names, the file's layers, the tags of its
//! think span, the token that closes its turn and the context the server is
//! asked for; and what its chat request asks beside the turn ([`Ask`]).

use std::path::Path;
use std::time::Duration;

use bloomery_gpu_gates::record;
use bloomery_gpu_gates::serve_client::{SeatGate, Served, check, curl, json_of, server_log};
use bloomery_gpu_gates::tier::{self, Tag};
use bloomery_gpu_gates::{GateError, checks_failed, ref_model_path};
use serde_json::json;

/// What a serve gate's seat gives the shared clauses.
pub struct Clauses {
    /// The gate's own name, as the `PASSED` line names it.
    pub gate: &'static str,
    /// The seat's model word (`--model`).
    pub model: &'static str,
    /// The records the seat's server prints.
    pub kinds: &'static [&'static record::Kind],
    /// The load record of the seat.
    pub load: &'static record::Kind,
    /// The listening record of the seat.
    pub listening: &'static record::Kind,
    /// The architecture word the load record names.
    pub arch: &'static str,
    /// The layers of the file the seat serves.
    pub layers: u64,
    /// The tags of the model's think span, as its vocabulary spells them; for a model with no
    /// span (Muse Spark's reasoning is a message to `self`), the two markers of its protocol
    /// the server must take out of the message.
    pub think: [&'static str; 2],
    /// The token that closes the model's turn, as its vocabulary spells it.
    pub turn_end: &'static str,
    /// The context the server is asked for.
    pub ctx: u64,
}

/// The tokens the chat request asks for at most, and the reasoning tokens the
/// budget lets the model spend before the server writes the close.
// PIN(2026-10-21): 64 → 512. MiniMax-M3's greedy answer to `TURN` ends its turn at
// 143 tokens (16 of reasoning, the rest content), past 64, so the turn-end clause
// read a cut answer (`finish_reason "length"`); 512 holds it 3.5×, and Solar-Open2
// stops under 64 either way. The clause still asks `stop` and no turn end in the
// message: red at 64, green at 512 on the same binary.
const MAX_TOKENS: u64 = 512;
const BUDGET: u64 = 16;

/// What the chat request asks beside the turn, and what its answer must hold: the reasoning
/// budget a think-span model's request carries, the effort a model with that knob is held to,
/// and whether the answer must carry reasoning.
#[derive(Clone, Copy)]
pub struct Ask {
    /// `reasoning_budget`: the reasoning tokens the server lets the model spend before it writes
    /// the span's close; `None` for a model with no span to close.
    pub budget: Option<u64>,
    /// `reasoning_effort`, where the template reads one.
    pub effort: Option<&'static str>,
    /// Whether [`CHAT`] requires reasoning in the message: false for a model that may answer a
    /// low-effort turn with none, whose reasoning is then read from the printed answer.
    pub reasons: bool,
}

impl Ask {
    /// A think-span model's request: the budget closes the span, the effort is the template's
    /// default and the message holds reasoning.
    pub const THINK_SPAN: Ask = Ask {
        budget: Some(BUDGET),
        effort: None,
        reasons: true,
    };
}

/// Polls of a second that bound the wait for a server to listen: a cold load
/// of the whole host set.
const LISTEN_POLLS: usize = 1000;

/// The chat turn the file answers.
const TURN: &str = "Name three primary colours.";

/// The words of the refusal of a second slot.
const ONE_SEQUENCE: &str = "holds one sequence";

const CHAT: &str = "chat_splits_the_reasoning_at_the_models_tags";
const ENDED: &str = "chat_ends_at_the_turn_end";
const ANSWERED: &str = "chat_answers_a_turn";

/// The seat of `c`, the one server the clauses drive.
fn seat(c: &Clauses) -> SeatGate {
    SeatGate {
        model: c.model,
        kinds: c.kinds,
    }
}

/// The clauses of the seat `c` names, on the file of the profile
/// (`BLOOMERY_REF_MODEL`; the fixture under the fixture tier): the seat
/// starts, loads the file and answers one chat request whose reasoning is
/// split at the model's own tags.
pub fn run(c: &Clauses) -> Result<(), GateError> {
    let dir = dir_of(c)?;
    std::fs::create_dir_all(&dir)?;
    let mut ok = parallel_above_one_is_refused(c, &dir)?;
    ok &= served(c, &dir)?;
    finish(c, ok)
}

/// The gate's own directory, the `--dir` flag every serve gate takes.
pub fn dir_of(c: &Clauses) -> Result<std::path::PathBuf, GateError> {
    seat(c).dir(&format!("usage: {} --dir <dir>", c.gate))
}

/// The gate's end: the `PASSED` line, or the failed clauses' own error.
pub fn finish(c: &Clauses, ok: bool) -> Result<(), GateError> {
    if !ok {
        return Err(checks_failed());
    }
    println!(
        "PASSED: {} the seat refuses a second slot, loads at the context it was asked for, and \
         answers a chat turn with its reasoning split at the model's tags",
        c.gate
    );
    Ok(())
}

/// `parallel_above_one_is_refused`: `--parallel 2 --plan` exits non-zero, its
/// stderr names the body's one sequence, and no `plan` record is printed (the
/// refusal comes before any plan). FAIL-first: a seat that takes a second slot
/// prints its plan and the clause is red.
pub fn parallel_above_one_is_refused(c: &Clauses, dir: &Path) -> Result<bool, GateError> {
    let mut ok = true;
    let file = ref_model_path()?;
    let (status, log, text) = seat(c).plan(dir, "refuse-parallel", &file, &["--parallel", "2"])?;
    let named = text.contains(ONE_SEQUENCE);
    let no_plan = log.first(&record::PLAN)?.is_none();
    println!(
        "refuse-parallel: exit non-zero {}; names {ONE_SEQUENCE:?} {named}; no plan record {no_plan}",
        !status.success()
    );
    if !named {
        println!("refuse-parallel stderr:\n{text}");
    }
    check(
        &mut ok,
        "parallel_above_one_is_refused",
        !status.success() && named && no_plan,
    );
    Ok(ok)
}

/// The clauses on one server: `load_and_listen` and the chat; the server is
/// stopped at the end.
pub fn served(c: &Clauses, dir: &Path) -> Result<bool, GateError> {
    served_with(c, dir, Ask::THINK_SPAN)
}

/// [`served`] with the chat request `ask` says.
pub fn served_with(c: &Clauses, dir: &Path, ask: Ask) -> Result<bool, GateError> {
    let mut ok = true;
    let d = dir.join("serve");
    std::fs::create_dir_all(&d)?;
    let file = ref_model_path()?;
    let (cmd, mut all) = seat(c).command(&file)?;
    all.extend(
        ["--port", "0", "--ctx", &c.ctx.to_string()]
            .iter()
            .map(|a| (*a).to_owned()),
    );
    let all: Vec<&str> = all.iter().map(String::as_str).collect();
    let mut s = Served::spawn_cmd(cmd, &all, &d)?;
    let err_log = d.join("server.err");
    let addr = match s.address(&err_log, LISTEN_POLLS, Duration::from_secs(1)) {
        Ok(addr) => addr,
        Err(e) => {
            println!("the server never listened: {e}");
            check(&mut ok, "load_and_listen", false);
            return Ok(ok);
        }
    };
    let records = server_log(&err_log, c.kinds)?;
    let load = records.one(c.load)?;
    let listen = records.one(c.listening)?;
    let url = |p: &str| format!("http://{addr}{p}");
    let (health, _) = curl(&url("/health"), None, false)?;
    let loaded = load.word("arch")? == c.arch
        && load.u64("slots")? == 1
        && load.u64("layers")? == c.layers
        && load.u64("ctx")? == c.ctx;
    let listening = listen.u64("slots")? == 1 && listen.u64("ctx")? == c.ctx;
    let in_order = load.at() < listen.at();
    println!(
        "serve at {addr}: load {:?}; listening {:?}; load names {}, one slot, {} layers and ctx \
         {} {loaded}; listening names one slot and ctx {} {listening}; load before listening \
         {in_order}; /health {health}",
        load.line(),
        listen.line(),
        c.arch,
        c.layers,
        c.ctx,
        c.ctx
    );
    check(
        &mut ok,
        "load_and_listen",
        loaded && listening && in_order && health == 200,
    );
    // The turn is answered on any file; its reasoning and its end read what the model writes, which
    // a cut of the model's layers does not write, so those two are the real file's.
    let (answered, split, ended) = match chat(c, ask, &url, &d) {
        Ok(verdicts) => verdicts,
        Err(e) => {
            println!("{CHAT}: {e}");
            (false, false, false)
        }
    };
    check(&mut ok, ANSWERED, answered);
    if tier::run_clause(CHAT, Tag::FileBound)? {
        check(&mut ok, CHAT, split);
    }
    if tier::run_clause(ENDED, Tag::FileBound)? {
        check(&mut ok, ENDED, ended);
    }
    println!("server stopped: {}", s.stop()?);
    Ok(ok)
}

/// [`ANSWERED`], [`CHAT`] and [`ENDED`], from one request: the turn is counted, carries no think
/// tag and stops by its end or its length; its message holds reasoning and content; it finishes
/// at the turn end.
fn chat(
    c: &Clauses,
    ask: Ask,
    url: &dyn Fn(&str) -> String,
    dir: &Path,
) -> Result<(bool, bool, bool), GateError> {
    let mut body = json!({
        "messages": [{ "role": "user", "content": TURN }],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "stream": false,
    });
    if let Some(budget) = ask.budget {
        body["reasoning_budget"] = json!(budget);
    }
    if let Some(effort) = ask.effort {
        body["reasoning_effort"] = json!(effort);
    }
    let (status, text) = curl(&url("/v1/chat/completions"), Some(&body), false)?;
    std::fs::write(dir.join("chat.json"), &text)?;
    let answer = json_of("/v1/chat/completions", status, &text)?;
    let message = &answer["choices"][0]["message"];
    let reasoning = message["reasoning_content"].as_str().unwrap_or("");
    let content = message["content"].as_str().unwrap_or("");
    let leaked = c
        .think
        .iter()
        .any(|tag| reasoning.contains(tag) || content.contains(tag));
    let usage = &answer["usage"];
    let counted = usage["prompt_tokens"].as_u64().is_some_and(|n| n > 0)
        && usage["completion_tokens"]
            .as_u64()
            .is_some_and(|n| (1..=MAX_TOKENS).contains(&n));
    let finish = answer["choices"][0]["finish_reason"].as_str().unwrap_or("");
    let ended =
        finish == "stop" && !content.contains(c.turn_end) && !reasoning.contains(c.turn_end);
    println!(
        "chat: usage {usage}; finish_reason {finish:?}; reasoning {} chars, content {} chars; a \
         think tag in the message {leaked}; the turn end in the message {}; reasoning \
         {reasoning:?}; content {content:?}",
        reasoning.trim().len(),
        content.trim().len(),
        content.contains(c.turn_end) || reasoning.contains(c.turn_end)
    );
    let stopped = matches!(finish, "stop" | "length");
    Ok((
        counted && !leaked && stopped,
        counted
            && (!ask.reasons || !reasoning.trim().is_empty())
            && !content.trim().is_empty()
            && !leaked,
        ended,
    ))
}
