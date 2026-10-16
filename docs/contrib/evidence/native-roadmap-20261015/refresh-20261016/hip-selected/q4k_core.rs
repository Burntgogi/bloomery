// Generated from the pinned crates/gpu/src/cores.rs; inspect device/build.rs.
const Q4K_ITER_UNROLL: u32 = 2;
#[inline(always)]
fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let mag = if exp == 0x1f {
        0x7f800000 | (mant << 13)
    } else if exp == 0 {
        if mant == 0 {
            0
        } else {
            let mut e = 127 - 14;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (e << 23) | ((m & 0x3ff) << 13)
        }
    } else {
        ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(sign | mag)
}
#[inline(always)]
fn q4k_nibble(qsw: u32, nib_sh: u32) -> u32 {
    ((((qsw >> nib_sh) & 0x0f0f0f0f) | 0x80808080).wrapping_sub(0x08080808)) ^ 0x80808080
}
#[inline(always)]
fn q4k_a_chain(vi: &[u32; 8], q: &[u32], qb: usize) -> i32 {
    // SAFETY: qb + 224 stays inside the caller's column span by this fn's
    // contract: within quad group g the largest qb is 256*g + 31, so the
    // last load is at 256*g + 255 < 256*ceil(n_sb/4), the column's words.
    let (w0, w1, w2, w3, w4, w5, w6, w7) = unsafe {
        (
            *q.get_unchecked(qb),
            *q.get_unchecked(qb + 32),
            *q.get_unchecked(qb + 64),
            *q.get_unchecked(qb + 96),
            *q.get_unchecked(qb + 128),
            *q.get_unchecked(qb + 160),
            *q.get_unchecked(qb + 192),
            *q.get_unchecked(qb + 224),
        )
    };
    let a = dp4a_s32(vi[0], w0, 0);
    let a = dp4a_s32(vi[1], w1, a);
    let a = dp4a_s32(vi[2], w2, a);
    let a = dp4a_s32(vi[3], w3, a);
    let a = dp4a_s32(vi[4], w4, a);
    let a = dp4a_s32(vi[5], w5, a);
    let a = dp4a_s32(vi[6], w6, a);
    dp4a_s32(vi[7], w7, a)
}
#[inline(always)]
fn q4k_scale_min(s: usize, w1: u32, w2: u32, w3: u32) -> (i32, i32) {
    if s < 4 {
        let sh = 8 * s as u32;
        (((w1 >> sh) & 63) as i32, ((w2 >> sh) & 63) as i32)
    } else {
        let sh = 8 * (s - 4) as u32;
        let qs = (w2 >> sh) & 0xff; // scales byte s
        let qs4 = (w3 >> sh) & 0xff; // scales byte s+4
        let qsm = (w1 >> sh) & 0xff; // scales byte s-4
        (
            ((qs4 & 0x0f) as i32) | (((qsm >> 6) as i32) << 4),
            ((qs4 >> 4) as i32) | (((qs >> 6) as i32) << 4),
        )
    }
}
#[inline(always)]
fn q4k_sub_value(a: i32, b: i32, cda: f32, cdb: f32) -> f32 {
    fma_rn_f32(cda, a as f32, mul_rn_f32(cdb, b as f32))
}
#[inline(always)]
fn q4k_acc(f: f32, x: f32, e: f32, split: bool) -> f32 {
    if split {
        add_rn_f32(f, mul_rn_f32(e, x))
    } else {
        fma_rn_f32(e, x, f)
    }
}
#[inline(always)]
fn q4k_sb_decode(w: &[u32], wk: usize, s: usize) -> ([u32; 8], f32, f32) {
    // d/dmin word + the 12 scale bytes in words 1..3.
    // SAFETY: wk + 3 <= wk + 35, inside `w` by this fn's contract.
    let (w0, w1, w2, w3) = unsafe {
        (
            *w.get_unchecked(wk),
            *w.get_unchecked(wk + 1),
            *w.get_unchecked(wk + 2),
            *w.get_unchecked(wk + 3),
        )
    };
    let d = half_to_f32((w0 & 0xffff) as u16);
    let dmin = half_to_f32((w0 >> 16) as u16);

    let (sc, mi) = q4k_scale_min(s, w1, w2, w3);
    // Column-independent chain coefficients.
    let (cda, cdb) = q4k_coeff(d, dmin, sc, mi);

    // qs word base: 8 words from super-block word 4 + 8*(s>>1); nibble
    // select is the sub-block parity, 0 (low) or 4 (high). Hoisted:
    // the window and its SWAR decode are column-independent, so
    // decode once per iteration and let every column's A chain reuse the
    // registers.
    let qsk = wk + 4 + 8 * (s >> 1);
    let nib_sh = (s as u32 & 1) * 4;
    // SAFETY: qsk + 7 <= wk + 35, the same bound as w0..w3 above.
    let vi = unsafe {
        [
            q4k_nibble(*w.get_unchecked(qsk), nib_sh),
            q4k_nibble(*w.get_unchecked(qsk + 1), nib_sh),
            q4k_nibble(*w.get_unchecked(qsk + 2), nib_sh),
            q4k_nibble(*w.get_unchecked(qsk + 3), nib_sh),
            q4k_nibble(*w.get_unchecked(qsk + 4), nib_sh),
            q4k_nibble(*w.get_unchecked(qsk + 5), nib_sh),
            q4k_nibble(*w.get_unchecked(qsk + 6), nib_sh),
            q4k_nibble(*w.get_unchecked(qsk + 7), nib_sh),
        ]
    };
    (vi, cda, cdb)
}
#[allow(clippy::too_many_arguments, reason = "original device core: flat kernel arguments (rust-quality R8)")]
#[inline(always)]
fn q4k_iter_term(
    w: &[u32],
    q: &[u32],
    s8: &[i32],
    d8: &[f32],
    n_sb: usize,
    it: u32,
    row_abs: usize,
    qb0: usize,
    s8b0: usize,
    d8b0: usize,
    s: usize,
    grp: usize,
    lane: usize,
) -> (f32, f32) {
    let sbp = 4 * it as usize + grp;
    // SAFETY: sbp < n_sb by this fn's contract, so the super-block's words
    // wk..wk+35 are inside row `row_abs` of `w`.
    let (vi, cda, cdb) = q4k_sb_decode(w, row_abs * 36 * n_sb + 36 * sbp, s);
    let qb = 256 * it as usize + lane;
    let s8b = 32 * it as usize + lane;
    let d8b = 2 * sbp + (s >> 2);
    let a = q4k_a_chain(&vi, q, qb0 + qb);
    // SAFETY: sbp < n_sb keeps s8b < 8*n_sb, and s8.len() >=
    // (col0+1)*8*n_sb by the contract.
    let b = unsafe { *s8.get_unchecked(s8b0 + s8b) };
    // SAFETY: sbp < n_sb keeps d8b < 2*n_sb, and d8.len() >=
    // (col0+1)*2*n_sb by the contract.
    let e0 = unsafe { *d8.get_unchecked(d8b0 + d8b) };
    (q4k_sub_value(a, b, cda, cdb), e0)
}
#[allow(clippy::too_many_arguments, reason = "original device core: flat kernel arguments (rust-quality R8)")]
#[inline(always)]
fn q4k_row_dot_1col(
    w: &[u32],
    q: &[u32],
    s8: &[i32],
    d8: &[f32],
    n_sb: usize,
    iters: u32,
    row_abs: usize,
    col0: usize,
    lane: usize,
) -> f32 {
    let q_col = 256 * iters as usize; // q8 words per column
    let s8_col = 8 * n_sb; // 32-value groups per column
    let d8_col = 2 * n_sb; // 128-value blocks per column

    let s = lane & 7; // sub-block within the super-block
    let grp = lane >> 3; // super-block within this iteration (0..4)

    let mut f0 = 0.0f32;

    let qb0 = col0 * q_col;
    let s8b0 = col0 * s8_col;
    let d8b0 = col0 * d8_col;

    // Iterations every lane takes: `sbp = 4*it + grp <= 4*full - 1 <= n_sb
    // - 1` for `it < full`, whatever `grp` is, so the walk's guard is dead
    // here and the backend sees one straight counted loop.
    let full = (n_sb / 4) as u32;
    let mut it: u32 = 0;
    while it + Q4K_ITER_UNROLL <= full {
        // SAFETY: it + Q4K_ITER_UNROLL <= full, so each term's `sbp < n_sb`;
        // the rest of the contract is this fn's.
        let (x0, e0) = q4k_iter_term(
            w, q, s8, d8, n_sb, it, row_abs, qb0, s8b0, d8b0, s, grp, lane,
        );
        // `q4k_acc`'s split, its product formed before the second term.
        let t0 = mul_rn_f32(e0, x0);
        let (x1, e1) = q4k_iter_term(
            w,
            q,
            s8,
            d8,
            n_sb,
            it + 1,
            row_abs,
            qb0,
            s8b0,
            d8b0,
            s,
            grp,
            lane,
        );
        f0 = add_rn_f32(f0, t0);
        f0 = q4k_acc(f0, x1, e1, false);
        it += Q4K_ITER_UNROLL;
    }
    while it < iters {
        let sbp = 4 * it as usize + grp;
        // As `q4k_row_dot`: the guard makes a partial final iteration safe
        // and is always true when n_sb is a multiple of 4.
        if sbp < n_sb {
            // SAFETY: the guard is this term's `sbp < n_sb` precondition.
            let (x, e) = q4k_iter_term(
                w, q, s8, d8, n_sb, it, row_abs, qb0, s8b0, d8b0, s, grp, lane,
            );
            f0 = q4k_acc(f0, x, e, false);
        }

        it += 1;
    }

    f0
}
