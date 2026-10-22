#!/usr/bin/env bash
# raon-oracle.sh — the reference ids of the raon vocabulary for tests/raon.rs, on stdout as the file
# tests/raon-ids.tsv: one row per case of tests/cases.txt and tests/cases-raon.txt (printf %b escapes), the
# text's bytes in hex, then llama-tokenize's ids with special tokens parsed (its default) and with
# --no-parse-special, each a space-separated list.
#
# The reference is the krafton-ai llama.cpp branch raon-1-mini-preview built on the box (its pre-tokenizer
# `raon` and its NFC normalizer are in that tree only): LCPP names the tree, MODEL the vocabulary file. No
# lease, no card (CUDA_VISIBLE_DEVICES is emptied; llama-tokenize loads the vocabulary only).
#
#   LCPP=/home/user/llama.cpp-raon bash crates/tokenizer/tools/raon-oracle.sh > crates/tokenizer/tests/raon-ids.tsv
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TESTS=$HERE/../tests
LCPP=${LCPP:-/home/user/llama.cpp-raon}
MODEL=${MODEL:-/models/Raon-1-mini-preview/raon-1-mini-preview-Q4_K_M.gguf}
TOKENIZE=$LCPP/build/bin/llama-tokenize
[ -x "$TOKENIZE" ] || { echo "raon-oracle: no $TOKENIZE" >&2; exit 66; }
[ -f "$MODEL" ] || { echo "raon-oracle: no vocabulary file at $MODEL" >&2; exit 66; }
commit=$(git -c safe.directory='*' -C "$LCPP" rev-parse HEAD)
dirty=$(git -c safe.directory='*' -C "$LCPP" status --porcelain --untracked-files=no | wc -l | tr -d ' ')
[ "$dirty" = 0 ] || { echo "raon-oracle: the tree $LCPP has $dirty tracked change(s)" >&2; exit 65; }

tmp=$(mktemp)
trap 'rm -f "$tmp" "$tmp.raw"' EXIT

# ids <flag...> — llama-tokenize --ids of $tmp, a space-separated list.
ids() {
  CUDA_VISIBLE_DEVICES= "$TOKENIZE" -m "$MODEL" -f "$tmp" --ids --log-disable --no-bos --no-escape "$@" > "$tmp.raw"
  tr -d '[] ' < "$tmp.raw" | tr ',' ' ' | sed 's/ *$//'
}

printf '# reference\tkrafton-ai/llama.cpp raon-1-mini-preview at %s\n' "$commit"
printf '# vocabulary\t%s\n' "$MODEL"
printf '# columns\ttext bytes in hex, ids (special tokens parsed), ids (--no-parse-special)\n'
n=0
for cases in "$TESTS/cases.txt" "$TESTS/cases-raon.txt"; do
  while IFS= read -r line || [ -n "$line" ]; do
    case $line in '#'*) continue ;; esac
    n=$((n + 1))
    printf '%b' "$line" > "$tmp"
    hex=$(od -An -v -tx1 "$tmp" | tr -d ' \n')
    printf '%s\t%s\t%s\n' "$hex" "$(ids)" "$(ids --no-parse-special)"
  done < "$cases"
done
printf '# complete\t%s\n' "$n"
