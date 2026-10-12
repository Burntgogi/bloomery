//! The start screen: a view for a person over the record lines a seat printed
//! while it started, drawn once on stderr below them when the server begins
//! to serve. A seat arms it ([`arm`]) when `BLOOMERY_SCREEN` is on and stderr
//! is a terminal, and hands it each record line it prints ([`note`]); piped
//! stderr is never armed, so it carries the record lines alone.
//!
//! The screen reads the lines' text as printed and nothing else: every value
//! on it is a record's, shown as the record wrote it, a byte count in GiB to
//! one decimal. A row is read from the last line of its record's head, so the
//! order of lines of different heads does not move it; a fact whose record
//! the start did not print is left off, and two records that name different
//! placements are not joined. `NO_COLOR` (set and not empty) and `TERM=dumb`
//! draw it without colour; `TERM=dumb` or a locale that does not name UTF-8
//! draws it in plain ASCII.

use std::io::IsTerminal;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// How the screen is drawn: with colour, and with UTF-8 glyphs or plain ASCII.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Style {
    pub color: bool,
    pub utf8: bool,
}

/// The terminal variables [`Style::of`] reads, each as set (`None` unset).
#[derive(Clone, Copy, Debug, Default)]
pub struct TermVars<'a> {
    pub term: Option<&'a str>,
    pub no_color: Option<&'a str>,
    pub lc_all: Option<&'a str>,
    pub lc_ctype: Option<&'a str>,
    pub lang: Option<&'a str>,
}

impl Style {
    /// The style of `v`: colour unless `NO_COLOR` is set and not empty or
    /// `TERM` is `dumb`; UTF-8 when the first non-empty of `LC_ALL`,
    /// `LC_CTYPE` and `LANG` names it, and never on a `dumb` terminal.
    #[must_use]
    pub fn of(v: TermVars<'_>) -> Style {
        fn set(x: Option<&str>) -> Option<&str> {
            x.filter(|x| !x.is_empty())
        }
        let dumb = set(v.term) == Some("dumb");
        let no_color = set(v.no_color).is_some();
        let utf8 = set(v.lc_all)
            .or(set(v.lc_ctype))
            .or(set(v.lang))
            .is_some_and(|l| {
                let l = l.to_ascii_lowercase();
                l.contains("utf-8") || l.contains("utf8")
            });
        Style {
            color: !no_color && !dumb,
            utf8: utf8 && !dumb,
        }
    }

    /// The style of the process's terminal variables ([`Style::of`]).
    #[must_use]
    pub fn of_terminal() -> Style {
        let term = std::env::var_os("TERM");
        let no_color = std::env::var_os("NO_COLOR");
        let lc_all = std::env::var_os("LC_ALL");
        let lc_ctype = std::env::var_os("LC_CTYPE");
        let lang = std::env::var_os("LANG");
        let s =
            |v: &Option<std::ffi::OsString>| v.as_ref().map(|v| v.to_string_lossy().into_owned());
        let (term, no_color, lc_all, lc_ctype, lang) =
            (s(&term), s(&no_color), s(&lc_all), s(&lc_ctype), s(&lang));
        Style::of(TermVars {
            term: term.as_deref(),
            no_color: no_color.as_deref(),
            lc_all: lc_all.as_deref(),
            lc_ctype: lc_ctype.as_deref(),
            lang: lang.as_deref(),
        })
    }
}

/// Whether the screen is armed and not drawn yet: [`note`] reads it before
/// it takes the tape's lock, so a line printed after the draw costs one load.
static ACTIVE: AtomicBool = AtomicBool::new(false);

/// The lines noted since [`arm`] and the style to draw them in; `None`
/// before an arming, after [`draw`], and in a process that never armed.
static TAPE: Mutex<Option<(Style, Vec<String>)>> = Mutex::new(None);

/// Arms the screen when `on` (the seat's `BLOOMERY_SCREEN`) and stderr is a
/// terminal, in the terminal's [`Style`]; returns whether it armed. A seat
/// whose screen armed hands it each record line it prints.
pub fn arm(on: bool) -> bool {
    if !on || !std::io::stderr().is_terminal() {
        return false;
    }
    *tape() = Some((Style::of_terminal(), Vec::new()));
    ACTIVE.store(true, Ordering::Release);
    true
}

/// Keeps `line`, a record line the seat printed, for the screen; nothing
/// unless armed and not drawn yet.
pub fn note(line: &str) {
    if !ACTIVE.load(Ordering::Acquire) {
        return;
    }
    if let Some((_, lines)) = tape().as_mut() {
        lines.push(line.to_owned());
    }
}

/// Draws the screen of the noted lines on stderr, once, and disarms: the
/// server calls it when it begins to serve. Unarmed, or with no `listening`
/// line noted, it draws nothing.
pub fn draw() {
    ACTIVE.store(false, Ordering::Release);
    let Some((style, lines)) = tape().take() else {
        return;
    };
    if let Some(screen) = render(&lines, style) {
        eprint!("{screen}");
    }
}

fn tape() -> std::sync::MutexGuard<'static, Option<(Style, Vec<String>)>> {
    TAPE.lock().unwrap_or_else(|e| e.into_inner())
}

/// The literal every seat's `listening` record carries before its address.
const LISTENING: &str = " listening on http://";

/// One record line as printed, read by key.
#[derive(Clone, Copy)]
struct Line<'a>(&'a str);

impl<'a> Line<'a> {
    /// The text after ` key=`, up to the next ` name=` or the line's end.
    fn text(self, key: &str) -> Option<&'a str> {
        let pat = format!(" {key}=");
        let start = self.0.find(&pat)? + pat.len();
        let rest = &self.0[start..];
        Some(rest[..next_key(rest)].trim_end())
    }

    /// The first word of `key`'s text.
    fn word(self, key: &str) -> Option<&'a str> {
        self.text(key)?.split_whitespace().next()
    }

    /// `key`'s first word as a whole number.
    fn num(self, key: &str) -> Option<u64> {
        self.word(key)?.parse().ok()
    }

    /// The `(<n> B)` written beside `key`'s value.
    fn bytes_beside(self, key: &str) -> Option<u64> {
        let (_, inner) = self.text(key)?.split_once('(')?;
        inner.strip_suffix(" B)")?.parse().ok()
    }

    /// Whether this line may be joined to the `listening` line's placement
    /// `place`: it names the same one, or one of the two names none.
    fn same_place(self, place: Option<&str>) -> bool {
        match (self.word("place"), place) {
            (Some(mine), Some(theirs)) => mine == theirs,
            _ => true,
        }
    }
}

/// Where the next ` name=` starts in `rest`, or its length.
fn next_key(rest: &str) -> usize {
    let b = rest.as_bytes();
    let mut i = 0;
    while let Some(off) = rest[i..].find(' ') {
        let at = i + off;
        let name = b[at + 1..]
            .iter()
            .take_while(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || **c == b'_')
            .count();
        if name > 0 && b.get(at + 1 + name) == Some(&b'=') {
            return at;
        }
        i = at + 1;
    }
    rest.len()
}

/// The last noted line that starts with `head` and carries ` key=`.
fn last<'a, S: AsRef<str>>(lines: &'a [S], head: &str, key: &str) -> Option<Line<'a>> {
    lines
        .iter()
        .map(|l| Line(l.as_ref()))
        .rfind(|l| l.0.starts_with(head) && l.text(key).is_some())
}

/// `b` bytes in GiB, one decimal.
fn gib(b: u64) -> String {
    #[allow(
        clippy::cast_precision_loss,
        reason = "a display rounding to one decimal of GiB"
    )]
    let g = b as f64 / f64::from(1u32 << 30);
    format!("{g:.1} GiB")
}

/// What the screen shows, read from the record lines.
struct Facts {
    /// The seat's name, its `listening` line's head.
    seat: String,
    /// The address the server listens on.
    addr: String,
    /// Each row's label and value, in the screen's order.
    rows: Vec<(&'static str, String)>,
}

/// The separators a row's value is written with, per [`Style`].
#[derive(Clone, Copy)]
struct Marks {
    dot: &'static str,
    times: &'static str,
}

/// The screen's facts from the records `lines` holds: the seat's name and
/// address from its `listening` line, `None` without one.
fn facts<S: AsRef<str>>(lines: &[S], m: Marks) -> Option<Facts> {
    let listen = lines
        .iter()
        .map(|l| Line(l.as_ref()))
        .rfind(|l| l.0.contains(LISTENING))?;
    let (head, addr) = listen.0.split_once(LISTENING)?;
    let seat = head.split(':').next()?.trim().to_owned();
    let addr = addr.trim().to_owned();
    let place = listen.word("place");
    let mut rows = vec![("url", format!("http://{addr}"))];
    rows.extend(seat_rows(lines, listen, m));
    let plan = last(lines, "plan ", "place").filter(|l| l.same_place(place));
    let load = last(lines, "load ", "resident_bytes").filter(|l| l.same_place(place));
    rows.extend(where_rows(lines, plan, load, m));
    rows.extend(context_row(lines, listen, m));
    if let Some(ram) = last(lines, "cache ", "ram").and_then(|c| c.num("ram")) {
        rows.push(match ram {
            0 => ("cache", "no prompt cache".to_owned()),
            r => ("cache", format!("{} of prompt cache in RAM", gib(r))),
        });
    }
    let secs = load
        .and_then(|l| l.0.rsplit_once(" in "))
        .and_then(|(_, t)| t.split_once(" s"))
        .and_then(|(s, _)| s.parse::<f64>().ok().map(|_| s));
    if let Some(s) = secs {
        rows.push(("loaded", format!("in {s} s")));
    }
    Some(Facts { seat, addr, rows })
}

/// The rows the `listening` line itself holds: the placement (with the
/// `place unset` record's why when it names the same one), the
/// architecture, and a decide seat's model and head.
fn seat_rows<S: AsRef<str>>(
    lines: &[S],
    listen: Line<'_>,
    m: Marks,
) -> Vec<(&'static str, String)> {
    let mut rows = Vec::new();
    if let Some(place) = listen.word("place") {
        let why = last(lines, "place unset ", "why")
            .filter(|p| p.word("place") == Some(place))
            .and_then(|p| p.text("why"));
        rows.push((
            "place",
            match why {
                Some(w) => format!("{place} ({w})"),
                None => place.to_owned(),
            },
        ));
    }
    if let Some(arch) = listen.word("arch") {
        rows.push(("arch", arch.to_owned()));
    }
    if let Some(model) = listen.word("model") {
        let quant = listen.text("quant").map(|q| format!(" {} {q}", m.dot));
        rows.push(("model", format!("{model}{}", quant.unwrap_or_default())));
    }
    if let Some(head) = listen.word("head") {
        let row = listen.word("row").map(|r| format!(" {} row {r}", m.dot));
        rows.push(("head", format!("{head}{}", row.unwrap_or_default())));
    }
    rows
}

/// Where the model went: the card (the plan's card and experts, the load's
/// resident bytes), every plan card when there are several, the tier card,
/// the host's experts and the NVMe tier — the arena the load was granted
/// from its `nvtier` record, else the plan's, named as planned.
fn where_rows<S: AsRef<str>>(
    lines: &[S],
    plan: Option<Line<'_>>,
    load: Option<Line<'_>>,
    m: Marks,
) -> Vec<(&'static str, String)> {
    let experts = |l: Line<'_>, key: &str| {
        l.num(key).map(|n| match l.bytes_beside(key) {
            Some(b) => format!("{n} experts ({})", gib(b)),
            None => format!("{n} experts"),
        })
    };
    let mut rows = Vec::new();
    let mut card = Vec::new();
    if let Some(p) = plan {
        card.extend(p.word("card").map(str::to_owned));
        card.extend(experts(p, "card_experts"));
    }
    if let Some(b) = load.and_then(|l| l.num("resident_bytes")) {
        card.push(format!("{} resident", gib(b)));
    }
    if !card.is_empty() {
        rows.push(("card", card.join(&format!(" {} ", m.dot))));
    }
    let Some(p) = plan else {
        return rows;
    };
    if let Some(devices) = p.word("devices").filter(|d| d.contains(',')) {
        rows.push(("devices", devices.to_owned()));
    }
    if let (Some(tier), Some(n)) = (p.word("tier"), p.num("tier_experts")) {
        rows.push(("tier", format!("{tier} {} {n} experts", m.dot)));
    }
    if let Some(host) = experts(p, "host_experts") {
        rows.push(("host", host));
    }
    let nvtier = last(lines, "nvtier ", "arena_granted");
    let granted = nvtier.and_then(|n| Some((n.num("paged")?, n.num("arena_granted")?)));
    match (granted, p.num("paged"), p.num("arena")) {
        (Some((paged, granted)), _, _) => {
            let planned = nvtier
                .and_then(|n| n.num("arena_planned"))
                .filter(|&a| a != granted)
                .map(|a| format!(" ({} planned)", gib(a)));
            rows.push((
                "nvme",
                format!(
                    "{} paged through a {} RAM arena{}",
                    gib(paged),
                    gib(granted),
                    planned.unwrap_or_default()
                ),
            ));
        }
        (None, Some(0), _) => rows.push(("nvme", "nothing paged".to_owned())),
        (None, Some(paged), Some(arena)) => rows.push((
            "nvme",
            format!("{} paged, a {} RAM arena planned", gib(paged), gib(arena)),
        )),
        (None, Some(paged), None) => rows.push(("nvme", format!("{} paged", gib(paged)))),
        (None, None, _) => {}
    }
    rows
}

/// The slots and their context: the `parallel` line's, else the
/// `listening` line's.
fn context_row<S: AsRef<str>>(
    lines: &[S],
    listen: Line<'_>,
    m: Marks,
) -> Option<(&'static str, String)> {
    let (dot, times) = (m.dot, m.times);
    let parallel = last(lines, "parallel ", "slots");
    let text = match parallel.and_then(|p| Some((p.num("slots")?, p.word("slot_ctx")?))) {
        Some((1, ctx)) => format!("1 slot {dot} {ctx} positions"),
        Some((n, ctx)) => match parallel.and_then(|p| p.word("total")) {
            Some(total) => format!("{n} slots {times} {ctx} positions ({total} in all)"),
            None => format!("{n} slots {times} {ctx} positions"),
        },
        None => {
            let ctx = listen.word("ctx")?;
            match listen.num("slots") {
                Some(1) => format!("1 slot {dot} {ctx} positions"),
                Some(n) => format!("{n} slots {times} {ctx} positions"),
                None => format!("{ctx} positions"),
            }
        }
    };
    Some(("context", text))
}

/// The columns `c` takes on a terminal: 0 for a combining mark, 2 for a wide
/// (East Asian wide or fullwidth) character, else 1.
fn char_width(c: char) -> usize {
    const ZERO: &[(u32, u32)] = &[
        (0x0300, 0x036F),
        (0x0483, 0x0489),
        (0x0591, 0x05BD),
        (0x0610, 0x061A),
        (0x064B, 0x065F),
        (0x1160, 0x11FF),
        (0x1AB0, 0x1AFF),
        (0x1DC0, 0x1DFF),
        (0x200B, 0x200F),
        (0x20D0, 0x20FF),
        (0x302A, 0x302F),
        (0x3099, 0x309A),
        (0xFE00, 0xFE0F),
        (0xFE20, 0xFE2F),
    ];
    const WIDE: &[(u32, u32)] = &[
        (0x1100, 0x115F),
        (0x2E80, 0x303E),
        (0x3041, 0x33FF),
        (0x3400, 0x4DBF),
        (0x4E00, 0x9FFF),
        (0xA000, 0xA4CF),
        (0xA960, 0xA97F),
        (0xAC00, 0xD7A3),
        (0xF900, 0xFAFF),
        (0xFE30, 0xFE4F),
        (0xFF00, 0xFF60),
        (0xFFE0, 0xFFE6),
        (0x1F1E6, 0x1F1FF),
        (0x1F300, 0x1F64F),
        (0x1F680, 0x1F6FF),
        (0x1F900, 0x1F9FF),
        (0x20000, 0x3FFFD),
    ];
    let u = u32::from(c);
    let within = |t: &[(u32, u32)]| t.iter().any(|&(a, b)| (a..=b).contains(&u));
    if within(ZERO) {
        0
    } else if within(WIDE) {
        2
    } else {
        1
    }
}

/// The columns `s` takes on a terminal ([`char_width`]).
fn width(s: &str) -> usize {
    s.chars().map(char_width).sum()
}

/// The furnace beside the rows: smoke, the stack, the bloom glowing in its
/// hearth and the tuyere on its side. `*` is the glow, drawn per [`Style`].
const FURNACE: [&str; 9] = [
    r"      (   )     ",
    r"     (  )  )    ",
    r"      )  (      ",
    r"    .-'--'-.    ",
    r"   /  ____  \   ",
    r"  |  /****\  |  ",
    r"  |  \****/  |==",
    r"  |__________|  ",
    r" ~~~~~~~~~~~~~~ ",
];

/// The smoke's rows at the top of [`FURNACE`].
const SMOKE_ROWS: usize = 3;

const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const GLOW: &str = "\x1b[1;33m";
const URL: &str = "\x1b[1;36m";
const RESET: &str = "\x1b[0m";

/// The screen of the record lines `lines` in `style`, or `None` when they
/// hold no `listening` line.
#[must_use]
pub fn render<S: AsRef<str>>(lines: &[S], style: Style) -> Option<String> {
    let marks = if style.utf8 {
        Marks {
            dot: "·",
            times: "×",
        }
    } else {
        Marks {
            dot: "-",
            times: "x",
        }
    };
    let Facts { seat, addr, rows } = facts(lines, marks)?;
    let paint = |code: &str, text: &str| {
        if style.color {
            format!("{code}{text}{RESET}")
        } else {
            text.to_owned()
        }
    };
    // A record value is shown without its control characters, and in ASCII
    // with `?` for every character past it.
    let plain = |text: &str| -> String {
        text.chars()
            .filter(|c| !c.is_control())
            .map(|c| if style.utf8 || c.is_ascii() { c } else { '?' })
            .collect()
    };

    let seat = plain(&seat);
    let addr = plain(&addr);
    let title = format!("{seat} is ready");
    let label_w = rows.iter().map(|(l, _)| l.len()).max().unwrap_or(0);
    let body: Vec<(String, String)> = rows
        .iter()
        .map(|(label, value)| {
            let value = plain(value);
            let shown = format!("{label:<label_w$}  {value}");
            let label = paint(DIM, &format!("{label:<label_w$}"));
            let value = if shown.starts_with("url ") {
                paint(URL, &value)
            } else {
                value
            };
            (shown, format!("{label}  {value}"))
        })
        .collect();
    let inner = body
        .iter()
        .map(|(s, _)| width(s))
        .chain([width(&title)])
        .max()
        .unwrap_or(0);
    let (tl, tr, bl, br, h, v) = if style.utf8 {
        ("╭", "╮", "╰", "╯", "─", "│")
    } else {
        ("+", "+", "+", "+", "-", "|")
    };
    let rule = h.repeat(inner + 2);
    let mut boxed = vec![format!("{tl}{rule}{tr}")];
    let mut row = |shown: &str, painted: &str| {
        let pad = " ".repeat(inner - width(shown));
        boxed.push(format!("{v} {painted}{pad} {v}"));
    };
    row(&title, &paint(BOLD, &title));
    row("", "");
    for (shown, painted) in &body {
        row(shown, painted);
    }
    boxed.push(format!("{bl}{rule}{br}"));

    let mut out = String::from("\n");
    for i in 0..FURNACE.len().max(boxed.len()) {
        let art = FURNACE.get(i).copied().unwrap_or("                ");
        let art = furnace_row(art, i, style);
        match boxed.get(i) {
            Some(b) => out.push_str(&format!("{art} {b}\n")),
            None => out.push_str(&format!("{}\n", art.trim_end())),
        }
    }
    out.push('\n');
    let url = format!("http://{addr}");
    if seat.ends_with("-decide") {
        out.push_str(&format!(
            "  try:  POST {}\n",
            paint(URL, &format!("{url}/v1/systemone"))
        ));
    } else {
        out.push_str("  try:\n");
        out.push_str(&format!(
            "    curl {} \\\n",
            paint(URL, &format!("{url}/v1/chat/completions"))
        ));
        out.push_str("      -H 'Content-Type: application/json' \\\n");
        out.push_str("      -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}'\n");
    }
    out.push('\n');
    out.push_str(&paint(
        DIM,
        &format!(
            "  Ctrl+C stops the server {} BLOOMERY_SCREEN=off hides this screen",
            marks.dot
        ),
    ));
    out.push('\n');
    Some(out)
}

/// Row `i` of [`FURNACE`] in `style`: the smoke dimmed, the glow as `▓` in
/// UTF-8 and `#` in ASCII, coloured when the style is.
fn furnace_row(art: &str, i: usize, style: Style) -> String {
    let glow = if style.utf8 { "▓" } else { "#" };
    if !style.color {
        return art.replace('*', glow);
    }
    if i < SMOKE_ROWS {
        return format!("{DIM}{art}{RESET}");
    }
    match (art.find('*'), art.rfind('*')) {
        (Some(a), Some(b)) => format!(
            "{}{GLOW}{}{RESET}{}",
            &art[..a],
            glow.repeat(b - a + 1),
            &art[b + 1..]
        ),
        _ => art.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::{Style, TermVars, render, width};

    /// The record lines of a V4.1 seat's start, as its log carries them.
    const START: &str = include_str!("../tests/fixtures/screen-start.log");
    /// The record lines of a Qwen3.8 seat's paged start, its arena granted
    /// smaller than planned.
    const START38: &str = include_str!("../tests/fixtures/screen-start-qwen38.log");

    const ASCII: Style = Style {
        color: false,
        utf8: false,
    };
    const UTF8: Style = Style {
        color: false,
        utf8: true,
    };

    fn lines(text: &str) -> Vec<&str> {
        text.lines().collect()
    }

    /// The screen of a fixed start, in UTF-8 and in ASCII, each the golden
    /// text; with colour, the same screen once the escapes are taken out.
    #[test]
    fn renders_the_golden_screens() {
        let got = render(&lines(START), UTF8).expect("the start has a listening line");
        assert_eq!(got, include_str!("../tests/fixtures/screen-utf8.txt"));
        let got = render(&lines(START), ASCII).expect("the start has a listening line");
        assert_eq!(got, include_str!("../tests/fixtures/screen-ascii.txt"));
        assert!(got.is_ascii(), "the ASCII screen holds a byte past ASCII");

        let colored = render(
            &lines(START),
            Style {
                color: true,
                utf8: true,
            },
        )
        .unwrap();
        assert!(
            colored.contains("\x1b["),
            "no escape in the coloured screen"
        );
        assert_eq!(
            strip_escapes(&colored),
            render(&lines(START), UTF8).unwrap()
        );
    }

    /// A Qwen3.8 start that pages its experts: the NVMe row names the arena
    /// the `nvtier` record says the load was granted, beside the one planned,
    /// the golden text.
    #[test]
    fn renders_the_granted_arena() {
        let got = render(&lines(START38), UTF8).expect("the start has a listening line");
        assert_eq!(got, include_str!("../tests/fixtures/screen-qwen38.txt"));
        let no_nvtier: Vec<&str> = lines(START38)
            .into_iter()
            .filter(|l| !l.starts_with("nvtier "))
            .collect();
        let got = render(&no_nvtier, UTF8).unwrap();
        assert!(
            !got.contains("nvme"),
            "an arena from no nvtier record:\n{got}"
        );
    }

    /// The fixed start with its `plan`, `load` and `place unset` lines taken
    /// out: the screen leaves off the rows those records hold (card, host,
    /// nvme, loaded, the place's why) and keeps the rest, the golden text.
    /// The same lines in reverse order draw the same screen, and so does the
    /// whole start reversed: each head has one line here, and a row is read
    /// from the last line of its head, never from where it sits among the
    /// others.
    #[test]
    fn missing_and_reordered_lines_keep_to_the_records() {
        let missing: Vec<&str> = lines(START)
            .into_iter()
            .filter(|l| {
                !["plan ", "load ", "place unset "]
                    .iter()
                    .any(|h| l.starts_with(h))
            })
            .collect();
        let golden = include_str!("../tests/fixtures/screen-missing.txt");
        assert_eq!(render(&missing, ASCII).unwrap(), golden);
        let reversed: Vec<&str> = missing.iter().rev().copied().collect();
        assert_eq!(render(&reversed, ASCII).unwrap(), golden);

        let reversed: Vec<&str> = lines(START).into_iter().rev().collect();
        assert_eq!(
            render(&reversed, UTF8).unwrap(),
            include_str!("../tests/fixtures/screen-utf8.txt")
        );
    }

    /// Records that name another placement than the `listening` line are not
    /// joined to it: a `place unset` line gives no why, a `plan` and a `load`
    /// line no card, host or load row.
    #[test]
    fn records_of_another_place_are_left_off() {
        let start: Vec<String> = lines(START)
            .into_iter()
            .map(|l| {
                if l.starts_with("place unset ") {
                    "place unset place=bp why=two cards, tier at or past the break-even".to_owned()
                } else if l.starts_with("plan ") || l.starts_with("load ") {
                    l.replace(" place=a ", " place=bp ")
                } else {
                    l.to_owned()
                }
            })
            .collect();
        let got = render(&start, ASCII).unwrap();
        assert!(got.contains("| place    a "), "{got}");
        for gone in ["break-even", "| card ", "| host ", "| loaded "] {
            assert!(
                !got.contains(gone),
                "{gone} from a record of place bp:\n{got}"
            );
        }
    }

    /// A record the start did not print leaves its row off: a listening line
    /// alone shows the address and what that line says, nothing more.
    #[test]
    fn a_missing_record_leaves_its_row_off() {
        let only = [
            "bloomery-serve-qwen38: place=a ctx=8192 slots=2 slot_ctx=8192 listening on http://0.0.0.0:9000",
        ];
        let got = render(&only, ASCII).unwrap();
        for row in ["url", "place", "context"] {
            assert!(got.contains(&format!("| {row} ")), "no {row} row:\n{got}");
        }
        for row in ["card", "host", "nvme", "cache", "loaded"] {
            assert!(
                !got.contains(&format!("| {row} ")),
                "a {row} row from no record:\n{got}"
            );
        }
        assert!(got.contains("2 slots x 8192 positions"), "{got}");
    }

    /// A load wall that is no number leaves the `loaded` row off.
    #[test]
    fn a_wall_that_is_no_number_is_left_off() {
        let start: Vec<String> = lines(START)
            .into_iter()
            .map(|l| l.replace(" in 41.2 s ", " in soon s "))
            .collect();
        let got = render(&start, ASCII).unwrap();
        assert!(!got.contains("| loaded "), "{got}");
    }

    /// Several plan cards: the `devices` row is the plan's whole list.
    #[test]
    fn several_cards_show_the_whole_list() {
        let start = [
            "plan place=bp card=A6000 ctx_max=16384 card_experts=2552 (38801506304 B) \
             host_experts=8045 (123885584384 B) host_shadow=0 B n_l=65..66 on 39 layers \
             card_budget=none paged=0 arena=0 arena_slots=0 card_free=50668240896 \
             devices=[stage:A6000:cuda1:50952404992,tier0:3090:cuda0:25350373376] \
             cuda_order=unset(FASTEST_FIRST)",
            "bloomery-serve-glm: place=bp ctx=16384 listening on http://127.0.0.1:8080",
        ];
        let got = render(&start, ASCII).unwrap();
        assert!(
            got.contains(
                "| devices  [stage:A6000:cuda1:50952404992,tier0:3090:cuda0:25350373376] "
            ),
            "{got}"
        );
    }

    /// A value with Hangul and a combining mark pads by the columns a
    /// terminal draws, so every row of the box ends in the same column; a
    /// control character in a value is not drawn.
    #[test]
    fn wide_and_control_characters_keep_the_box_square() {
        let start = [
            "bloomery-serve-decide: model=삽살이-Q4_K_M.gguf quant=q4_K×1988 q6_K×97 \
             head=e\u{301}tiquette.safetensors\x1b[31m row=clef listening on http://127.0.0.1:8080",
        ];
        let got = render(&start, UTF8).unwrap();
        assert!(
            !got.contains('\x1b'),
            "an escape from a record value:\n{got}"
        );
        let boxed: Vec<&str> = got.lines().filter(|l| l.contains('│')).collect();
        assert!(boxed.iter().any(|l| l.contains("삽살이")), "{got}");
        let ends: Vec<usize> = boxed.iter().map(|l| width(l)).collect();
        assert!(ends.windows(2).all(|w| w[0] == w[1]), "{ends:?}\n{got}");
        assert_eq!(width("삽살이"), 6);
        assert_eq!(width("か\u{3099}"), width("が"));
        assert_eq!(width("\u{1F680}"), 2);
        assert_eq!(width("e\u{301}"), 1);
    }

    /// No `listening` line, no screen: a start that did not reach its
    /// address draws nothing.
    #[test]
    fn no_listening_line_draws_nothing() {
        let start: Vec<&str> = lines(START)
            .into_iter()
            .filter(|l| !l.contains(" listening on http://"))
            .collect();
        assert_eq!(
            render(
                &start,
                Style {
                    color: true,
                    utf8: true
                }
            ),
            None
        );
    }

    /// The terminal's variables: `NO_COLOR` set and not empty, or
    /// `TERM=dumb`, turn colour off; the first non-empty locale variable
    /// decides UTF-8, which `TERM=dumb` turns off too.
    #[test]
    fn the_style_follows_the_terminal_variables() {
        let utf8 = Some("ko_KR.UTF-8");
        let style = |v: TermVars<'_>| Style::of(v);
        assert_eq!(
            style(TermVars {
                lang: utf8,
                ..TermVars::default()
            }),
            Style {
                color: true,
                utf8: true
            }
        );
        assert_eq!(
            style(TermVars {
                no_color: Some("1"),
                lang: utf8,
                ..TermVars::default()
            }),
            Style {
                color: false,
                utf8: true
            }
        );
        assert!(
            style(TermVars {
                no_color: Some(""),
                ..TermVars::default()
            })
            .color,
            "an empty NO_COLOR is unset"
        );
        assert_eq!(
            style(TermVars {
                term: Some("dumb"),
                lang: utf8,
                ..TermVars::default()
            }),
            Style {
                color: false,
                utf8: false
            }
        );
        assert!(
            !style(TermVars {
                lc_all: Some("C"),
                lang: utf8,
                ..TermVars::default()
            })
            .utf8,
            "LC_ALL wins over LANG"
        );
        assert!(
            style(TermVars {
                lc_all: Some(""),
                lc_ctype: Some("en_US.utf8"),
                lang: Some("C"),
                ..TermVars::default()
            })
            .utf8,
            "an empty LC_ALL is unset, and LC_CTYPE comes before LANG"
        );
        assert!(!style(TermVars::default()).utf8, "no locale is not UTF-8");
    }

    fn strip_escapes(s: &str) -> String {
        let mut out = String::new();
        let mut it = s.chars();
        while let Some(c) = it.next() {
            if c == '\x1b' {
                for c in it.by_ref() {
                    if c == 'm' {
                        break;
                    }
                }
            } else {
                out.push(c);
            }
        }
        out
    }
}
