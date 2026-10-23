#!/usr/bin/env bash
# A pull request's diff, base...head as GitHub shows it, for the review bot's two jobs.
#
#   tools/review/pr-diff.sh OWNER/NAME PR PR_TREE OUT
#
# GitHub's diff media type refuses a diff over 20000 lines (HTTP 406). Then the diff is computed in PR_TREE, the
# head's checkout: its shallow history is deepened from the base repository (which serves the head as
# refs/pull/PR/head, a fork's too) until it holds the merge base, and `git diff` runs against that commit. Any other
# API failure, or no merge base within the fetched history, ends the script with its reason and a non-zero code.
set -euo pipefail

if [ $# -ne 4 ]; then
    echo "pr-diff: usage: pr-diff.sh OWNER/NAME PR PR_TREE OUT" >&2
    exit 64
fi
repo=$1 pr=$2 tree=$3 out=$4

err=$(mktemp)
trap 'rm -f "$err"' EXIT
if gh api "repos/$repo/pulls/$pr" -H 'Accept: application/vnd.github.diff' > "$out" 2> "$err"; then
    exit 0
fi
if ! grep -q 'HTTP 406' "$err"; then
    cat "$err" >&2
    exit 1
fi

base=$(gh api "repos/$repo/pulls/$pr" --jq .base.sha)
head=$(git -C "$tree" rev-parse HEAD)
url="https://github.com/$repo"
git -C "$tree" fetch -q --no-tags --depth=64 "$url" "$base" "refs/pull/$pr/head"
tries=0
until mb=$(git -C "$tree" merge-base "$base" "$head" 2> /dev/null); do
    tries=$((tries + 1))
    if [ "$tries" -gt 8 ]; then
        echo "pr-diff: no merge base of $base and $head within the fetched history" >&2
        exit 1
    fi
    git -C "$tree" fetch -q --no-tags --deepen=256 "$url" "$base" "refs/pull/$pr/head"
done
git -C "$tree" diff --no-color --no-ext-diff "$mb" "$head" > "$out"
echo "pr-diff: GitHub refused the diff as too large; built from $mb..$head ($(wc -l < "$out") lines)" >&2
