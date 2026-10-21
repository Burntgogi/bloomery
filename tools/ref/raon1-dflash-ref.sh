#!/usr/bin/env bash
# The reference's DFlash2 run for Raon-1-mini: the oracle tree (models/raon1.sh's $LCPP, KRAFTON's llama.cpp branch, read only)
# copied to $OUT/src with two log lines changed in examples/speculative-simple (each draft's ids and each verify's accepted count at
# info level), built CPU-only as that one target, then run greedy over tools/ref/data/raon1/{a,b,c}.ids with the reference's flags
# (--spec-type draft-dflash --spec-draft-n-max 7). The prompts go in as text decoded from the ids by the file's vocabulary (byte-level
# BPE); each text is tokenized back by the oracle tree's llama-tokenize and must give its ids again (rc 65 otherwise). With the
# argument `chat` the runs are instead four chat-template prompts (ChatML, the file's template with thinking off for three and on for
# one), 96 tokens each: what the serve seat feeds.
#
#   BLOOMERY_REMOTE='~/repo/bloomery-raon-dflash' tools/box.sh 'setsid -f bash tools/ref/raon1-dflash-ref.sh [chat] </dev/null >/dev/null 2>&1'
#   BLOOMERY_REMOTE='~/repo/bloomery-raon-dflash' BLOOMERY_BOX_READONLY=1 tools/box.sh 'cat /root/raon1-dflash-ref/rc; grep -E "^(raon1-dflash-ref|draft ids|accepted|n_drafted|n_accept|accept )" /root/raon1-dflash-ref/log'
#
# No card and no lease: a CPU build and CPU runs under `nice -n 19`, 12 build jobs, 24 threads, every step bounded (tools/gate-bound.sh).
# Writes pid, log and rc (last; absent = running or killed) under /root/raon1-dflash-ref.
set -uo pipefail
HERE=$(cd "${BASH_SOURCE[0]%/*}" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
# shellcheck source=tools/gate-bound.sh
source "$ROOT/tools/gate-bound.sh"

main() {
  local dir=/root/raon1-dflash-ref out=/home/user/stage/raon1-dflash-ref
  local src=/home/user/llama.cpp-raon
  local target=/models/Raon-1-mini-preview/raon-1-mini-preview-Q4_K_M.gguf
  local draft=/models/Raon-1-mini-preview/dflash-raon-1-mini-preview-Q8_0.gguf
  local p ids back
  mkdir -p "$dir" "$out"
  rm -f "$dir/rc"
  echo $$ > "$dir/pid"
  exec > "$dir/log" 2>&1
  fail() {
    echo "raon1-dflash-ref: $1"
    echo "${2:-2}" > "$dir/rc"
    exit "${2:-2}"
  }
  echo "raon1-dflash-ref: start $(date -Is) pid $$"
  for f in "$target" "$draft" "$src/build/bin/llama-tokenize"; do
    [ -e "$f" ] || fail "$f is not there" 66
  done
  bounded 300 rsync -a --delete --exclude build --exclude .git "$src/" "$out/src/" || fail "copying $src" 70
  local ex=$out/src/examples/speculative-simple/speculative-simple.cpp
  grep -q '            common_speculative_draft(spec);' "$ex" || fail "$ex has no draft call to log after" 65
  sed -i 's|^            common_speculative_draft(spec);$|            common_speculative_draft(spec);\n            { std::string s; for (auto t : draft) { s += std::to_string(t) + " "; } LOG_INF("draft ids pos %d after %d: %s\\n", n_past, id_last, s.c_str()); }|' "$ex"
  sed -i 's|LOG_DBG("accepted %d/%d draft tokens|LOG_INF("accepted %d/%d draft tokens|' "$ex"
  grep -c 'draft ids pos\|LOG_INF("accepted' "$ex"
  bounded 600 nice -n 19 cmake -S "$out/src" -B "$out/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_CURL=OFF \
    -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF || fail "configure" 70
  bounded 1500 nice -n 19 cmake --build "$out/build" --target llama-speculative-simple -j 12 || fail "build" 70
  echo "raon1-dflash-ref: built $(date -Is)"
  local run=(nice -n 19 "$out/build/bin/llama-speculative-simple" -m "$target" -md "$draft" --spec-type draft-dflash
    --spec-draft-n-max 7 --temp 0 -t 24 -c 4096)
  if [ "${1:-}" = chat ]; then
    # `-f` drops one trailing newline from its file: each prompt carries one more than the template ends with.
    local off='<think></think>\n\n\n'
    printf '<|im_start|>user\nExplain what a lighthouse keeper did on an ordinary day in the nineteenth century, in three short paragraphs.<|im_end|>\n<|im_start|>assistant\n%b' "$off" > "$out/chat-en.txt"
    printf '<|im_start|>user\n서울에서 부산까지 기차로 여행할 때 들를 만한 곳을 세 군데 추천해 줘.<|im_end|>\n<|im_start|>assistant\n%b' "$off" > "$out/chat-ko.txt"
    printf '<|im_start|>user\nWrite a Python function that returns the n-th Fibonacci number iteratively, with a docstring.<|im_end|>\n<|im_start|>assistant\n%b' "$off" > "$out/chat-code.txt"
    printf '<|im_start|>user\nWhat is 17 times 23? Reason briefly.<|im_end|>\n<|im_start|>assistant\n<think>\n\n' > "$out/chat-think.txt"
    for p in chat-en chat-ko chat-code chat-think; do
      echo "raon1-dflash-ref: prompt $p"
      bounded 900 "${run[@]}" -f "$out/$p.txt" -n 96 || fail "the run on $p" 70
      echo "raon1-dflash-ref: prompt $p done $(date -Is)"
    done
    echo 0 > "$dir/rc"
    return
  fi
  for p in a b c; do
    ids=$ROOT/tools/ref/data/raon1/$p.ids
    bounded 120 python3 -I - "$src/gguf-py" "$target" "$ids" "$out/$p.txt" <<'EOF' || fail "decoding $p" 70
import sys
sys.path.insert(0, sys.argv[1])
from gguf import GGUFReader
r = GGUFReader(sys.argv[2])
f = r.fields['tokenizer.ggml.tokens']
toks = [bytes(f.parts[i]).decode('utf-8') for i in f.data]
bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
cs = bs[:]
n = 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + n)
        n += 1
dec = {chr(c): b for b, c in zip(bs, cs)}
ids = [int(x) for x in open(sys.argv[3]).read().split()]
data = bytes(dec[ch] for i in ids for ch in toks[i])
open(sys.argv[4], 'wb').write(data)
EOF
    LEASE_BOUNDED_OUT=$out/$p.back LEASE_BOUNDED_ERR=/dev/null \
      bounded 300 "$src/build/bin/llama-tokenize" -m "$target" -f "$out/$p.txt" --ids --no-escape --log-disable || fail "tokenizing $p" 70
    back=$(tr -d '[] ' < "$out/$p.back" | tr ',' '\n' | grep -v '^$')
    [ "$back" = "$(grep -v '^$' "$ids")" ] || fail "$p: the decoded text does not tokenize back to $ids" 65
    echo "raon1-dflash-ref: prompt $p ids $(wc -l < "$ids") round trip ok"
    bounded 900 "${run[@]}" -f "$out/$p.txt" -n 32 || fail "the run on $p" 70
    echo "raon1-dflash-ref: prompt $p done $(date -Is)"
  done
  echo 0 > "$dir/rc"
}

main "$@"
