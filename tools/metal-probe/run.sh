#!/bin/bash
# tools/metal-probe/run.sh -- build and sit the Metal probes on this Mac.
#
# What it measures
#   M2.0 (default sitting)
#     c_d   GPU time per dependent dispatch, by encoding mode (plain, concurrent,
#           icb, one command buffer every K), from probe dispatch/fit records.
#     f_bw  fraction of datasheet bandwidth a streaming read reaches alone (bw) and
#           inside a 580-dispatch chain over 1.8 GB (chain).
#   M2.2 (--kernels)
#     f_q   fraction of the same-run streaming bandwidth a real Q4_K / Q6_K
#           mat-vec reaches (gemv), and the 580-dispatch dependent mat-vec chain
#           (gemvchain), plain and with two independent mat-vecs between barriers.
#
# How to run
#   bash tools/metal-probe/run.sh            the M2.0 sitting
#   bash tools/metal-probe/run.sh --kernels  the M2.2 kernel sitting
#   Builds into tools/metal-probe/build/ and writes one TSV log under
#   tools/metal-probe/out/. Both are gitignored.
#
# Records in the log (tab-separated, kind first): info, dispatch, fit, bw, chain,
# gemv, gemvchain, gemvcheck, selftest. See the header of probe.swift for each
# field.
#
# These numbers are development evidence from this Mac, never a public number:
# there is no lease and no timing runner for Metal yet.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD="$HERE/build"
OUT="$HERE/out"
mkdir -p "$BUILD" "$OUT"

SIT="m20"
case "${1:-}" in
    "") ;;
    --kernels) SIT="kernels" ;;
    *)
        echo "usage: bash $0 [--kernels]" >&2
        exit 64
        ;;
esac

# The Metal toolchain component is optional (Xcode 26+ downloads it separately).
# With it, kernels compile offline; without it, probe.swift compiles kernels.metal
# at run time through device.makeLibrary(source:).
KERNELS=""
if xcrun -sdk macosx metal -c "$HERE/kernels.metal" -o "$BUILD/kernels.air" 2>"$BUILD/metal.err"; then
    xcrun -sdk macosx metallib "$BUILD/kernels.air" -o "$BUILD/kernels.metallib"
    KERNELS="$BUILD/kernels.metallib"
    BUILD_NOTE="metallib"
else
    KERNELS="$HERE/kernels.metal"
    BUILD_NOTE="source"
    echo "run.sh: xcrun metal unavailable ($(sed -n 1p "$BUILD/metal.err")); compiling kernels.metal at run time" >&2
fi

xcrun swiftc -O "$HERE/probe.swift" -o "$BUILD/probe"

export PROBE_KERNELS="$KERNELS"
PROBE="$BUILD/probe"

CHIP_RAW="$(sysctl -n machdep.cpu.brand_string)"
CHIP="$(printf '%s' "$CHIP_RAW" | tr ' ' '-' | tr -cd 'A-Za-z0-9._-')"
TS="$(date +%Y%m%dT%H%M%S)"
LOG="$OUT/probe-${CHIP}-${TS}.tsv"

{
    echo "# metal-probe chip=$CHIP_RAW kernels=$BUILD_NOTE sit=$SIT started=$TS load_at_start=$(sysctl -n vm.loadavg)"
} > "$LOG"

INFO="$("$PROBE" info)"
printf '%s\n' "$INFO" >> "$LOG"

WS="$(printf '%s\n' "$INFO" | sed -n 's/.*recommended_max_working_set=\([0-9][0-9]*\).*/\1/p')"
BW_BYTES=2147483648
CHAIN_BYTES=1800000000
if [ -n "$WS" ] && [ "$WS" -lt 4294967296 ]; then
    BW_BYTES=1073741824
    CHAIN_BYTES=900000000
    printf '# recommendedMaxWorkingSetSize %s is under 4 GiB: byte sizes halved (bw %s, chain %s)\n' \
        "$WS" "$BW_BYTES" "$CHAIN_BYTES" >> "$LOG"
fi

set +e
"$PROBE" --self-test > "$BUILD/selftest.log" 2>&1
SELFTEST_RC=$?
set -e
cat "$BUILD/selftest.log" >> "$LOG"
cat "$BUILD/selftest.log"
if [ "$SELFTEST_RC" -ne 0 ]; then
    echo "run.sh: self-test failed (rc $SELFTEST_RC), log $LOG" >&2
    exit "$SELFTEST_RC"
fi

if [ "$SIT" = "kernels" ]; then
    # The denominator: one streaming read of the decode token's byte size.
    "$PROBE" bw --bytes "$BW_BYTES" --tg 1024 --reps 20 >> "$LOG"

    # rows x cols: square, a 512-row expert down slice, a tall gate/up, and the
    # largest matrix this probe holds. Every shape's weight bytes are replicated
    # until the working set is at least 1 GiB, and each rep reads another copy,
    # so each arm is a DRAM number (the log's working_set_bytes says it). --reps is the
    # least dispatches in a timed command buffer; the probe raises it until a buffer reads
    # 768 MiB of weights (the record's dispatches=), long enough to outlast the clock ramp.
    for TYPE in q4k q6k; do
        for SHAPE in 2048:2048 4096:4096 512:2048 1536:4096 8192:4096; do
            ROWS="${SHAPE%%:*}"
            COLS="${SHAPE##*:}"
            for VARIANT in ggml own brk ownbrk; do
                echo "run.sh: gemv $TYPE $ROWS x $COLS $VARIANT" >&2
                "$PROBE" gemv --type "$TYPE" --rows "$ROWS" --cols "$COLS" \
                    --variant "$VARIANT" --reps 30 >> "$LOG"
            done
        done
    done

    # The decode skeleton: 580 dependent mat-vecs over 580 weight slices, one at a time and
    # two independent ones between barriers (a concurrent encoder).
    for VARIANT in ggml brk; do
        for CONC in 1 2; do
            echo "run.sh: gemvchain q4k 2048 x 2048 n=580 $VARIANT concurrent=$CONC" >&2
            "$PROBE" gemvchain --type q4k --rows 2048 --cols 2048 --n 580 \
                --variant "$VARIANT" --concurrent "$CONC" --reps 20 >> "$LOG"
        done
    done
else
    for MODE in plain concurrent icb cbper:1 cbper:50; do
        "$PROBE" dispatch --n 1,16,64,256,580 --reps 30 --mode "$MODE" >> "$LOG"
    done

    "$PROBE" bw --bytes "$BW_BYTES" --tg 256,512,1024 --reps 20 >> "$LOG"

    for MODE in plain icb; do
        "$PROBE" chain --bytes "$CHAIN_BYTES" --n 580 --mode "$MODE" --reps 20 >> "$LOG"
    done
fi

echo "run.sh: log $LOG"
printf '# metal-probe finished=%s load_at_end=%s\n' "$(date +%Y%m%dT%H%M%S)" "$(sysctl -n vm.loadavg)" >> "$LOG"
grep -E '^(fit|bw|chain|gemv|gemvchain|gemvcheck)	' "$LOG" || true
