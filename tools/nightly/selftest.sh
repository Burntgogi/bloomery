#!/usr/bin/env bash
# `bash tools/nightly/selftest.sh --self-test`: the commit label of tools/nightly/units.sh (`tree_commit`) and the two scripts
# that print it, tools/contrib-check.sh and tools/nightly/run.sh, on temp repositories. No card, no box, no cargo, no network.
# One line per check, `ok <name>` or `FAIL <name>: got … want …`, the verdict last (bash 3.2 on the Mac).
#
# What it blocks: a `git status` that fails (a corrupt index, an unreadable tree) prints nothing, and a test of that output
# reads nothing as a clean tree. A run on such a tree then published its HEAD as the commit it tested, without `+local`, and went
# on. The label is `?` or the run stops (contrib-check 70, nightly rc 2 RED) naming the tree and git's exit code. The three other
# scripts that name a tree's commit (gate-batch.sh, ptx-compare.sh, narrow-scan.sh) test that in their own self-tests.
set -uo pipefail

if [ "${1:-}" != --self-test ]; then
  echo "usage: tools/nightly/selftest.sh --self-test" >&2
  exit 64
fi

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
# shellcheck source=tools/nightly/units.sh
source "$HERE/units.sh"

T=$(mktemp -d "${TMPDIR:-/tmp}/tree-commit.XXXXXX")
trap 'rm -rf "$T"' EXIT
fails=0 checks=0
check() {
  checks=$((checks + 1))
  if [ "$2" = "$3" ]; then
    echo "ok $1"
  else
    echo "FAIL $1: got [$2] want [$3]"
    fails=$((fails + 1))
  fi
}
# has NAME HAYSTACK NEEDLE: `ok` when HAYSTACK contains NEEDLE, else the haystack's first line in the failure.
has() {
  case $2 in
    *"$3"*) check "$1" ok ok ;;
    *) check "$1" "${2%%$'\n'*}" "…$3…" ;;
  esac
}

g() { git -c user.email=t@t -c user.name=t -c commit.gpgsign=false "$@"; }
# mkrepo DIR: a repository with one commit and one tracked file.
mkrepo() {
  git init -q --template= "$1" && echo a > "$1/f" && g -C "$1" add f && g -C "$1" commit -q -m one
}
sha8() { git -C "$1" rev-parse --short=8 HEAD; }
corrupt_index() { printf garbage > "$1/.git/index"; }

# ---- tree_commit ----
mkrepo "$T/clean"
tree_commit "$T/clean"
check clean "$?|$TREE_COMMIT|$TREE_GIT_RC" "0|$(sha8 "$T/clean")|0"
check clean-parts "$TREE_SHA|$TREE_DIRTY" "$(sha8 "$T/clean")|0"
tree_commit "$T/clean" 12
check width "$?|$TREE_SHA|$TREE_COMMIT" "0|$(git -C "$T/clean" rev-parse --short=12 HEAD)|$(git -C "$T/clean" rev-parse --short=12 HEAD)"

mkrepo "$T/untracked"
echo b > "$T/untracked/new"
tree_commit "$T/untracked"
check untracked-is-local "$?|$TREE_COMMIT" "0|$(sha8 "$T/untracked")+local"
check untracked-parts "$TREE_SHA|$TREE_DIRTY" "$(sha8 "$T/untracked")|1"

mkrepo "$T/modified"
echo b >> "$T/modified/f"
tree_commit "$T/modified"
check modified-is-local "$?|$TREE_COMMIT" "0|$(sha8 "$T/modified")+local"

# The failure mode: HEAD reads, status fails with 128 and prints nothing. Never the bare commit, never `+local` guessed.
mkrepo "$T/badindex"
corrupt_index "$T/badindex"
check badindex-head-reads "$(sha8 "$T/badindex" > /dev/null 2>&1 && echo yes)" yes
tree_commit "$T/badindex" 2> /dev/null
check badindex-refused "$?|$TREE_COMMIT|$TREE_GIT_RC" "2|?|128"
check badindex-no-part-states-a-clean-tree "$TREE_SHA|$TREE_DIRTY" "?|?"

git init -q --template= "$T/nocommit"
tree_commit "$T/nocommit"
check no-head "$?|$TREE_COMMIT|$TREE_GIT_RC|$TREE_SHA|$TREE_DIRTY" "1|?|0|?|?"

mkdir "$T/plain"
GIT_CEILING_DIRECTORIES=$T tree_commit "$T/plain"
check not-a-repository "$?|$TREE_COMMIT|$TREE_GIT_RC" "1|?|0"

# ---- tools/contrib-check.sh: the real script in the real tree, its index pointed at a corrupt file ----
# Only the injected unit could run if the stop were missing, so a regression ends this check red and short, never in a full run.
printf garbage > "$T/index.bad"
out=$(cd "$ROOT" && GIT_INDEX_FILE=$T/index.bad CONTRIB_INJECT_FAIL=1 bash tools/contrib-check.sh --tier t0 --only '^inject-fail$' 2> "$T/cc.err")
rc=$?
check contrib-check-stops "$rc" 70
check contrib-check-no-header "$out" ""
has contrib-check-names-the-tree "$(cat "$T/cc.err")" "git status failed in $ROOT (git exit 128)"

# A tree with no readable HEAD (an unpacked archive) is the `?` label, not the stop: the header names commit=? and the units run.
out=$(cd "$ROOT" && GIT_DIR=$T/no-git CONTRIB_INJECT_FAIL=1 bash tools/contrib-check.sh --tier t0 --only '^inject-fail$' 2> "$T/cc-archive.err")
rc=$?
check contrib-check-archive-runs "$rc" 1
has contrib-check-archive-header "$out" "contrib-check tier=t0 commit=?"
has contrib-check-archive-unit-ran "$out" "inject-fail rc=101"

# ---- tools/nightly/run.sh: a clone under a temp NIGHTLY_HOME, no fetch ----
# nightly_run NAME CLONE: the files of the one run it leaves in $T/NAME/state/<date>/. flock is a stub on the nightly's PATH
# ($HOME/bin comes first there), so a host without it (macOS) runs the same code. The clone has no rust-toolchain.toml, so a run
# that passes the tree check ends at "no channel": rc 2 either way, and the commit file and the failed line tell the cases apart.
nightly_run() {
  local name=$1 home=$T/$1/home
  mkdir -p "$home/bin" "$T/$name/state"
  printf '#!/bin/sh\nexit 0\n' > "$home/bin/flock"
  chmod +x "$home/bin/flock"
  mv "$2" "$home/repo"
  NIGHTLY_HOME=$home NIGHTLY_STATE=$T/$name/state NIGHTLY_NO_FETCH=1 bash "$HERE/run.sh" > /dev/null 2>&1
  NIGHTLY_RC=$?
  NIGHTLY_DIR=
  for NIGHTLY_DIR in "$T/$name"/state/[0-9]*; do break; done
  NIGHTLY_CLONE=$home/repo
}
nightly_file() { cat "$NIGHTLY_DIR/$1" 2> /dev/null || echo "(no $1)"; }

mkrepo "$T/fx-n1"
corrupt_index "$T/fx-n1"
nightly_run n1 "$T/fx-n1"
check nightly-badindex-rc "$NIGHTLY_RC|$(nightly_file rc)|$(nightly_file status)" "2|2|RED"
has nightly-badindex-names-the-clone "$(nightly_file failed)" "git status of $NIGHTLY_CLONE failed (git exit 128)"
check nightly-badindex-publishes-no-commit "$(nightly_file commit)" "(no commit)"

mkrepo "$T/fx-n2"
nightly_run n2 "$T/fx-n2"
check nightly-clean-commit "$(nightly_file commit)" "$(sha8 "$NIGHTLY_CLONE")"
has nightly-clean-goes-on "$(nightly_file failed)" "no channel"

mkrepo "$T/fx-n3"
echo b > "$T/fx-n3/new"
nightly_run n3 "$T/fx-n3"
check nightly-dirty-commit "$(nightly_file commit)" "$(sha8 "$NIGHTLY_CLONE")+local"

git init -q --template= "$T/fx-n4"
nightly_run n4 "$T/fx-n4"
check nightly-no-head-rc "$NIGHTLY_RC|$(nightly_file status)" "2|RED"
has nightly-no-head-named "$(nightly_file failed)" "HEAD of $NIGHTLY_CLONE cannot be read"

echo "tree-commit self-test: $([ "$fails" = 0 ] && echo ok || echo FAILED) ($checks checks, $fails failures)"
[ "$fails" = 0 ]
