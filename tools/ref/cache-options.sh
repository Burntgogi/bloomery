#!/usr/bin/env bash
# shellcheck shell=bash
# Source-only CMake-cache verdicts for reference build wrappers.
#
# ref_cache_flags TOOL TREE writes sorted GGML/LLAMA option rows from
# TREE/build/CMakeCache.txt, excluding discovery-only _FOUND entries. Status 0
# includes readable empty caches and caches with no selected options; status 2
# means a read, filter, or sort failed and there is no usable snapshot.
#
# ref_cache_check TOOL REFERENCE_TREE CANDIDATE_TREE checks both snapshots before
# comparing them. Options present in only one tree are informational stdout in C key order;
# unequal values for a common key are stderr and status 1. Agreement (including
# empty or one-sided snapshots) is status 0; any snapshot/comparison failure is
# status 2. Keys include :TYPE, and values retain everything after the first '='.
# Both functions isolate pipefail and locale settings from the sourcing shell.
# shellcheck source=tools/grep-answer.sh
source "$(dirname "${BASH_SOURCE[0]}")/../grep-answer.sh"

ref_cache_flags() (
  set -o pipefail
  export LC_ALL=C
  local tool=$1 cache=$2/build/CMakeCache.txt
  if ! grep_answer "$tool" -E '^(GGML|LLAMA)_[A-Z0-9_]*:[A-Z]+=' "$cache" \
    | grep_answer "$tool" -v '_FOUND:' | sort; then
    printf '%s: cannot snapshot %s, no verdict\n' "$tool" "$cache" >&2
    return 2
  fi
)

ref_cache_check() (
  set -o pipefail
  export LC_ALL=C
  local tool=$1 reference_tree=$2 candidate_tree=$3
  local reference candidate only differences
  local reference_cache=$reference_tree/build/CMakeCache.txt
  local candidate_cache=$candidate_tree/build/CMakeCache.txt

  if ! reference=$(ref_cache_flags "$tool" "$reference_tree"); then
    return 2
  fi
  if ! candidate=$(ref_cache_flags "$tool" "$candidate_tree"); then
    return 2
  fi

  # Process substitutions emit only checked strings, never read a cache. Empty
  # snapshots must emit zero bytes rather than an artificial empty-key record.
  if ! only=$(join -t '=' -v 1 -v 2 \
    <(printf '%s' "${reference:+$reference$'\n'}") \
    <(printf '%s' "${candidate:+$candidate$'\n'}")); then
    printf '%s: cannot compare %s and %s (join), no verdict\n' \
      "$tool" "$reference_cache" "$candidate_cache" >&2
    return 2
  fi
  if ! differences=$(awk '
    {
      separator = index($0, "=")
      key = substr($0, 1, separator - 1)
      value = substr($0, separator + 1)
    }
    FILENAME == ARGV[1] { reference[key] = value; next }
    key in reference && reference[key] != value {
      printf "    %s: reference=%s candidate=%s\n", key, reference[key], value
    }
  ' <(printf '%s' "${reference:+$reference$'\n'}") \
    <(printf '%s' "${candidate:+$candidate$'\n'}")); then
    printf '%s: cannot compare %s and %s (awk), no verdict\n' \
      "$tool" "$reference_cache" "$candidate_cache" >&2
    return 2
  fi

  if [[ -n $only ]]; then
    printf '%s: cache options only one of %s and %s has:\n%s\n' \
      "$tool" "$reference_cache" "$candidate_cache" "$only"
  fi
  if [[ -n $differences ]]; then
    printf '%s: common cache options differ (reference %s, candidate %s):\n%s\n' \
      "$tool" "$reference_cache" "$candidate_cache" "$differences" >&2
    return 1
  fi
  return 0
)
