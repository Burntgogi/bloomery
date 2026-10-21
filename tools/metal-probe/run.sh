#!/bin/bash
# tools/metal-probe/run.sh -- build and sit the M2.0 Metal probe on this Mac.
#
# What it measures
#   c_d   GPU time per dependent dispatch, by encoding mode (plain, concurrent,
#         icb, one command buffer every K), from probe dispatch/fit records.
#   f_bw  fraction of datasheet bandwidth a streaming read reaches alone (bw) and
#         inside a 580-dispatch chain over 1.8 GB (chain).
#
# How to run
#   bash tools/metal-probe/run.sh
#   Builds into tools/metal-probe/build/ and writes one TSV log under
#   tools/metal-probe/out/. Both are gitignored.
#
# Records in the log (tab-separated, kind first): info, dispatch, fit, bw, chain,
# selftest. See the header of probe.swift for each field.
#
# These numbers are development evidence from this Mac, never a public number:
# there is no lease and no timing runner for Metal yet.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD="$HERE/build"
OUT="$HERE/out"
mkdir -p "$BUILD" "$OUT"

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
    echo "# metal-probe chip=$CHIP_RAW kernels=$BUILD_NOTE started=$TS"
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

for MODE in plain concurrent icb cbper:1 cbper:50; do
    "$PROBE" dispatch --n 1,16,64,256,580 --reps 30 --mode "$MODE" >> "$LOG"
done

"$PROBE" bw --bytes "$BW_BYTES" --tg 256,512,1024 --reps 20 >> "$LOG"

for MODE in plain icb; do
    "$PROBE" chain --bytes "$CHAIN_BYTES" --n 580 --mode "$MODE" --reps 20 >> "$LOG"
done

echo "run.sh: log $LOG"
grep -E '^(fit|bw|chain)	' "$LOG" || true
