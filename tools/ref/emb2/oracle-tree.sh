#!/usr/bin/env bash
# shellcheck shell=bash
# The mainline oracle tree of the EmbeddingGemma sets, checked and not trusted. Sourced; never executed.
#
#   oracle_tree WHO TREE PIN [BINARY...]
#
# TREE is the commit's build that `just build-emb2-lcpp` made (owned by `user` on the box): its HEAD is PIN, it has no tracked
# change, and each BINARY under TREE/build/bin (default llama-server) exists and reports a commit that is a prefix of PIN. A
# tree rebuilt without a readable git reports `commit unknown` there and is refused: it is not the commit's build.
# Returns 0 and prints nothing, or exits 2 with a line that begins WHO and names what is wrong. It reads the tree and writes
# nothing into it: a script that wants the oracle calls this and never builds one.
oracle_tree() {
  local who=$1 tree=$2 pin=$3 have bin report
  local re='commit ([0-9a-f]{7,40})([^0-9a-f]|$)'
  shift 3
  [ -d "$tree" ] || { echo "$who: no oracle tree at $tree (just build-emb2-lcpp)" >&2; exit 2; }
  have=$(git -c safe.directory="$tree" -C "$tree" rev-parse HEAD) || { echo "$who: $tree is not a git tree" >&2; exit 2; }
  [ "$have" = "$pin" ] || { echo "$who: $tree is at $have, the profile pins $pin" >&2; exit 2; }
  have=$(git -c safe.directory="$tree" -C "$tree" status --porcelain --untracked-files=no) ||
    { echo "$who: cannot inspect tracked changes in $tree" >&2; exit 2; }
  [ -z "$have" ] || { echo "$who: $tree has tracked changes; the oracle is the commit's build" >&2; exit 2; }
  [ $# -gt 0 ] || set -- llama-server
  for bin in "$@"; do
    [ -x "$tree/build/bin/$bin" ] || { echo "$who: no $tree/build/bin/$bin (just build-emb2-lcpp)" >&2; exit 2; }
    report=$("$tree/build/bin/$bin" --version 2>&1) || { echo "$who: $tree/build/bin/$bin --version failed: $report" >&2; exit 2; }
    # The whole commit token, 7 hex digits or more, a prefix of the pin.
    [[ $report =~ $re ]] && [ "${pin:0:${#BASH_REMATCH[1]}}" = "${BASH_REMATCH[1]}" ] ||
      { echo "$who: $tree/build/bin/$bin is not built at ${pin:0:7}: $report" >&2; exit 2; }
  done
}
