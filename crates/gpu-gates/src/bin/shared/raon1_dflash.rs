//! Raon-1's DFlash2 draft, driven by the common loop (`app::dspark::CardDspark`): the common owner
//! (`DsparkBody`) over DFlash2's layers with the selector head, on the target's card, borrowing the
//! target file's embedding rows and output head. [`pick`] finds the draft's file (refset's
//! beside-file row), [`reserve`] is what a plan sets aside for it on the target's card,
//! [`open_body`] loads the owner and holds it to that formula ([`PlanBytes::hold`]), [`open`] puts
//! it behind the common loop.
//!
//! The owner holds `fc` and its norm, the head's norm and the selector's three tables, and reads the
//! target's output head where [`HeadFrom`] says: the target model's own resident matrix in place
//! (the seat's and the drafted gate's load, on the target's card), or a copy from the target's file
//! (the draft gate's, with no target loaded). The layers hold every `blk.` tensor and embed the
//! block's rows from the target's `token_embd`. A `dflash` file without a selector head (V4.1's DSpark draft) is refused by name
//! before any tensor is read, and every tensor is held to the reader's statement
//! (`model::arch::dflash2::check`).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use app::dspark::CardDspark;
use bloomery_gpu::arch::raon1::Body;
use bloomery_gpu::dspark::dense::DenseLayers;
use bloomery_gpu::dspark::head::{HeadTail, HeadWeights};
use bloomery_gpu::dspark::selector::SelectorTail;
use bloomery_gpu::dspark::{
    DraftLayers, DsparkBody, EmbedSource, FcWeights, HeadSource, InPlace, Q8Rows, TargetHead,
};
use bloomery_gpu::weights::{DevWeight, Weights, head_tensor};
use bloomery_gpu::{Gpu, GpuError, GpuModel};
use bloomery_gpu_gates::GateError;
use gguf::{GgmlType, Split};
use model::arch::dflash2::{self, Hparams, names};
use model::arch::names as arch_names;
use model::embed::EmbedTable;
use refset::arch::DraftFrom;
use refset::arch::raon1::DFLASH_BESIDE;
use runtime::dspark::DsparkShape;

/// The dense layers DFlash2's load builds ([`DenseLayers::load_dflash2`]), the common owner's `L`
/// for this draft.
pub type Layers = DenseLayers;

/// Raon-1's DFlash2 draft: the common loop over DFlash2's layers.
pub type RaonDraft = CardDspark<Layers>;

const WHAT: &str = "raon1 dflash2";

/// Where the draft's head reads the target's output matrix.
#[derive(Clone, Copy)]
pub enum HeadFrom<'m> {
    /// A copy on the draft's card, read from the target's file.
    Copy,
    /// The resident matrix of `GpuModel` the draft drafts for, on the same card, read in place.
    InPlace(&'m GpuModel<Body>),
}

/// The head's scratch a plan sets aside: its normed rows, logits and the selector's walk buffers
/// for a block's rows, an allowance [assumed] the load is held under ([`PlanBytes::hold`]).
const HEAD_SCRATCH: u64 = 64 << 20;

/// What the draft takes on the target's card, from the two files' headers before any load, each
/// term in bytes before the allocator rounds a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanBytes {
    /// The layers' ([`DenseLayers::dflash2_card_bytes`]).
    pub layers: u64,
    /// The owner's weights as the files hold them: fc, the norms and the selector's tables from
    /// the draft, and the target's output head when the draft holds a copy of it.
    pub weights: u64,
    /// The owner's working buffers, the tapped rows' inbox among them
    /// ([`DsparkBody::working_bytes`]).
    pub working: u64,
    /// The head's scratch allowance ([`HEAD_SCRATCH`]).
    pub head_scratch: u64,
}

impl PlanBytes {
    /// Every term, summed.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.layers + self.weights + self.working + self.head_scratch
    }

    /// Hold `body`, loaded for `ctx` positions, to these terms: the layers', the owner's weights
    /// and its working buffers each equal to the buffers the load counted of itself, the head's
    /// scratch within its allowance. A term off is refused by name.
    pub fn hold(&self, body: &DsparkBody<Layers>, ctx: usize) -> Result<(), GateError> {
        let own = body.own_held();
        let scratch = body.head().scratch_bytes();
        let terms = [
            ("the layers", body.layers().card_bytes(ctx), self.layers),
            (
                "the owner's weights",
                own.weights.iter().sum(),
                self.weights,
            ),
            (
                "the owner's working buffers",
                own.working.saturating_sub(scratch),
                self.working,
            ),
        ];
        for (term, held, planned) in terms {
            if held != planned {
                return Err(format!(
                    "{WHAT}: the load holds {held} B of {term}; the plan's formula has {planned} B"
                )
                .into());
            }
        }
        if scratch > self.head_scratch {
            return Err(format!(
                "{WHAT}: the head's scratch holds {scratch} B, past the {} B the plan sets aside",
                self.head_scratch
            )
            .into());
        }
        Ok(())
    }
}

/// [`PlanBytes`] of the draft `draft` (its header `hp`) beside the target `target`, for a load of
/// `ctx` positions whose head reads the target's matrix in place (`copy` false) or a copy of it.
/// A tensor either file does not hold is refused by name.
pub fn plan_bytes(
    draft: &Split,
    hp: &Hparams,
    target: &Split,
    (ctx, copy): (usize, bool),
) -> Result<PlanBytes, GateError> {
    let file_bytes = |split: &Split, name: &str| {
        split
            .find(name)
            .map(|(_, t)| t.nbytes)
            .ok_or_else(|| format!("{WHAT}: no tensor {name} in the file"))
    };
    let mut weights = if copy {
        file_bytes(target, head_tensor(target))?
    } else {
        0
    };
    for name in owned() {
        weights += file_bytes(draft, &name)?;
    }
    Ok(PlanBytes {
        layers: Layers::dflash2_card_bytes(hp, ctx)?,
        weights,
        working: DsparkBody::<Layers>::working_bytes(&dflash2::shape(hp), hp.n_embd, false),
        head_scratch: HEAD_SCRATCH,
    })
}

/// What a plan sets aside on the target's card for the draft `draft` (its header `hp`) beside the
/// target `target`, its head reading the target's matrix in place: [`PlanBytes::total`] at the
/// draft's window, which the layers' ring holds at most, so the figure bounds a load of any
/// context.
pub fn reserve(draft: &Split, hp: &Hparams, target: &Split) -> Result<u64, GateError> {
    Ok(plan_bytes(draft, hp, target, (hp.window, false))?.total())
}

/// What [`pick`] found.
#[derive(Debug, PartialEq, Eq)]
pub enum Pick {
    /// The file to open, and where it came from.
    File(PathBuf, DraftFrom),
    /// No draft file: every path looked at.
    Absent(Vec<PathBuf>),
}

/// The draft file a run of `target` opens, by Raon-1's beside-file row ([`DFLASH_BESIDE`]): `set`
/// when given (`BLOOMERY_DSPARK_MODEL`), which must be a regular file; else the row's file name
/// beside the target (as the path spells it, then with its symlinks resolved), else the row's path.
/// The row's path with no file there is no draft.
pub fn pick(set: Option<&Path>, target: &Path) -> Result<Pick, GateError> {
    if let Some(p) = set.filter(|p| !p.is_file()) {
        return Err(format!(
            "{WHAT}: {}={} is not a regular file",
            DFLASH_BESIDE.lever,
            p.display()
        )
        .into());
    }
    let (path, from) = DFLASH_BESIDE.pick(set, target);
    if matches!(from, DraftFrom::Family(_)) && !path.is_file() {
        return Ok(Pick::Absent(DFLASH_BESIDE.places(target)));
    }
    Ok(Pick::File(path, from))
}

/// The draft of the file at `path` on `gpu`, for a target whose file is `target` and whose
/// context holds `ctx` positions: the file's geometry, and the common owner over DFlash2's layers
/// with the selector head reading the target's matrix from `head`, its graphs captured, held to
/// the plan's formula ([`PlanBytes::hold`]). The layers embed the block from the target's
/// `token_embd`. Load-time only.
pub fn open_body(
    gpu: &Gpu,
    path: &Path,
    (target, head): (Arc<Split>, HeadFrom<'_>),
    ctx: usize,
) -> Result<(DsparkShape, DsparkBody<Layers>), GateError> {
    let draft = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let hp = Hparams::read(&draft).map_err(|e| format!("{}: {e}", path.display()))?;
    dflash2::check(&draft, &hp).map_err(|e| format!("{}: {e}", path.display()))?;
    let shape = dflash2::shape(&hp);
    let embed = EmbedTable::new(Arc::clone(&target), &arch_names::token_embd(), hp.n_embd)
        .map_err(|e| format!("{WHAT}: the target's embedding: {e}"))?;
    let layers = Layers::load_dflash2(gpu, &draft, &hp, ctx, embed)?;
    let copy = matches!(head, HeadFrom::Copy);
    let (fc, head) = owner_weights(gpu, &draft, &hp, (&target, head))?;
    let body = DsparkBody::new(gpu, &shape, layers, fc, head, EmbedSource::Target)?;
    plan_bytes(&draft, &hp, &target, (ctx, copy))?.hold(&body, ctx)?;
    Ok((shape, body))
}

/// [`open_body`] behind the common loop, on the target's card `card` (the draft's `gpu`, its
/// primary context shared with the target's), drafting for the target `m` loaded from `target` at
/// `ctx` positions, its head reading `m`'s resident matrix in place. The target's card context is
/// bound again before this returns. The draft's resident bytes are the buffers its load allocated;
/// the card's free-bytes loss around the load is kept as a diagnostic.
pub fn open(
    gpu: Gpu,
    path: &Path,
    (target, m): (Arc<Split>, &GpuModel<Body>),
    ctx: usize,
    card: &'static str,
) -> Result<RaonDraft, GateError> {
    let target_ctx = Arc::clone(m.gpu().context());
    let loaded = (|| {
        let (free, _) = gpu.mem_info()?;
        let (shape, body) = open_body(&gpu, path, (target, HeadFrom::InPlace(m)), ctx)?;
        let (left, _) = gpu.mem_info()?;
        Ok::<_, GateError>((shape, body, free.saturating_sub(left)))
    })();
    target_ctx.bind_to_thread().map_err(GpuError::from)?;
    let (shape, body, taken) = loaded?;
    let held = body.layers().held(&body.own_held());
    Ok(CardDspark::new::<Body>(
        body, gpu, target_ctx, card, shape, held, taken,
    )?)
}

/// The tensors the common owner holds: from the draft, fc and its norm, the head's norm and the
/// selector's tables; from the target, its output head as `head` says (Q6_K rows of the draft's
/// width, the type the owner's head kernel reads; any other is refused by name).
fn owner_weights(
    gpu: &Gpu,
    draft: &Split,
    hp: &Hparams,
    (target, head): (&Split, HeadFrom<'_>),
) -> Result<(FcWeights, HeadWeights), GateError> {
    let stream = gpu.stream();
    let owned = owned();
    let mut w = Weights::load_where(stream, draft, |n| owned.iter().any(|o| o == n))?;
    let (n_embd, rank, vocab) = (hp.n_embd, hp.selector.rank, hp.n_vocab);
    let fc = FcWeights {
        fc: take_q8(&mut w, &names::fc(), hp.fc_in(), n_embd)?,
        norm: take_f32(&mut w, &names::enc_output_norm(), n_embd)?,
        eps: hp.rms_eps,
    };
    let out = head_tensor(target);
    let projection = match head {
        HeadFrom::Copy => match Weights::load_where(stream, target, |n| n == out)?.take(out) {
            Some(DevWeight::KQuant {
                ty: GgmlType::Q6_K,
                w,
                k,
            }) if k == n_embd => TargetHead::Q6K(w),
            other => return Err(not_q6k(other.as_ref(), out, n_embd)),
        },
        HeadFrom::InPlace(m) => {
            if m.gpu().context().cu_ctx() != gpu.context().cu_ctx() {
                return Err(format!(
                    "{WHAT}: the target's {out} is read in place, on the target's card context; \
                     the draft opened on another"
                )
                .into());
            }
            let w = match m.weights().get(out) {
                Some(DevWeight::KQuant {
                    ty: GgmlType::Q6_K,
                    w,
                    k,
                }) if *k == n_embd => w,
                other => return Err(not_q6k(other, out, n_embd)),
            };
            // SAFETY: `w` is the target model's resident matrix, allocated at its load on this
            // card and freed only when the model drops; the draft is held beside the model's
            // session (the serve seat's `Raon1`, the gate's seat) and launches only inside that
            // session's calls, so the matrix stays allocated and in place while the draft reads it.
            TargetHead::Q6KInPlace(unsafe { InPlace::new(w, gpu.context()) })
        }
    };
    let head = HeadWeights {
        norm: take_f32(&mut w, &names::output_norm(), n_embd)?,
        eps: hp.rms_eps,
        source: HeadSource::Target(projection),
        tail: HeadTail::Selector(SelectorTail {
            hidden: take_q8(&mut w, &names::selector_hidden(), n_embd, rank)?,
            pred: take_q8(&mut w, &names::selector_predecessor(), rank, vocab)?,
            succ: take_q8(&mut w, &names::selector_successor(), rank, vocab)?,
            top_k: hp.selector.top_k,
        }),
    };
    Ok((fc, head))
}

/// The refusal of `w`, the target's `name`, which is not Q6_K rows of `k` values.
fn not_q6k(w: Option<&DevWeight>, name: &str, k: usize) -> GateError {
    match w {
        Some(DevWeight::KQuant { ty, k: wk, .. }) => format!(
            "{WHAT}: the target's {name} is {ty:?} rows of {wk}; the draft's head reads Q6_K rows \
             of {k}"
        ),
        Some(_) => format!(
            "{WHAT}: the target's {name} is not a K-quant matrix; the draft's head reads Q6_K"
        ),
        None => format!("{WHAT}: the target holds no {name}"),
    }
    .into()
}

/// The draft's tensors the common owner holds: fc and its norm, the head's norm and the selector's
/// three tables.
fn owned() -> [String; 6] {
    [
        names::fc(),
        names::enc_output_norm(),
        names::output_norm(),
        names::selector_hidden(),
        names::selector_predecessor(),
        names::selector_successor(),
    ]
}

/// Tensor `name` of `w`, moved out: Q8_0 rows of `k` values, `rows` of them.
fn take_q8(w: &mut Weights, name: &str, k: usize, rows: usize) -> Result<Q8Rows, GateError> {
    match w.take(name) {
        Some(DevWeight::Q8_0 { qs, d, k: wk }) if wk == k && d.rows() == rows => {
            Ok(Q8Rows { qs, d })
        }
        Some(DevWeight::Q8_0 { d, k: wk, .. }) => Err(format!(
            "{WHAT}: {name} is {} rows of {wk}; the draft reads {rows} of {k}",
            d.rows()
        )
        .into()),
        _ => Err(format!("{WHAT}: {name} is not resident as Q8_0 rows").into()),
    }
}

/// Tensor `name` of `w`, moved out: an f32 row of `len` values.
fn take_f32(
    w: &mut Weights,
    name: &str,
    len: usize,
) -> Result<bloomery_gpu::DeviceTensor<f32>, GateError> {
    match w.take(name) {
        Some(DevWeight::F32 { w, .. }) if w.buf().len() == len => Ok(w),
        Some(DevWeight::F32 { w, .. }) => Err(format!(
            "{WHAT}: {name} holds {} f32; the draft reads {len}",
            w.buf().len()
        )
        .into()),
        _ => Err(format!("{WHAT}: {name} is not resident as f32").into()),
    }
}
