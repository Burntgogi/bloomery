#!/usr/bin/env bash
# Builds the Raon-1 oracle tree: KRAFTON's llama.cpp branch raon-1-mini-preview at LCPP_COMMIT (models/raon1.sh), a CPU build with no
# patch, in $LCPP/build as shared libraries (the library tools/ref/raon1-dump-build.sh links the dumper against) plus the CLIs the
# sanity check and the prompt tokenizer need (llama-cli, llama-simple, llama-tokenize). The tree is put on the box by rsync of the
# branch's checkout (it is a shallow clone: `git rev-parse HEAD` names the commit). A tree at another commit or with tracked
# changes is refused (rc 65), never moved or reset.
#
#   BLOOMERY_MODEL=raon1 BLOOMERY_REMOTE='~/repo/bloomery-<track>' tools/box.sh 'setsid -f bash tools/ref/raon1-tree.sh </dev/null >/dev/null 2>&1'
#   ssh ws 'cat /root/raon1-tree/pid; tail -n 3 /root/raon1-tree/log; cat /root/raon1-tree/rc'
#
# The build runs under `nice -n 19` with 12 jobs and a 29-minute bound, and starts only while the timing lease is free (rc 75).
# Writes pid, log and rc (last; absent = running or killed) under /root/raon1-tree. The body is one function, parsed before it runs,
# so a later sync of this file does not change a build in flight. The log ends with `raon1-tree commit <sha> libllama md5 <md5>`.
set -uo pipefail
HERE=$(cd "${BASH_SOURCE[0]%/*}" && pwd)
# shellcheck source=tools/ref/ref-paths.sh
source "$HERE/ref-paths.sh"
# shellcheck source=tools/ref/lease.sh
source "$HERE/lease.sh"

main() {
  local dir=/root/raon1-tree have dirty t0 rc=0
  mkdir -p "$dir"
  rm -f "$dir/rc"
  echo $$ > "$dir/pid"
  exec > "$dir/log" 2>&1
  fail() {
    echo "raon1-tree: $1"
    echo "${2:-2}" > "$dir/rc"
    exit "${2:-2}"
  }
  [ "$MODEL_NAME" = raon1 ] || fail "the profile is $MODEL_NAME; run under BLOOMERY_MODEL=raon1" 64
  : "${LCPP_COMMIT:?models/raon1.sh names no LCPP_COMMIT}"
  echo "raon1-tree: start $(now) pid $$ tree $LCPP pin $LCPP_COMMIT"
  lease_free || case $? in
    1) fail "the timing lease is held (a sitting is running); start again after it" 75 ;;
    *) fail "the timing lease cannot be tested (above); not starting" 70 ;;
  esac
  have=$(git -c safe.directory='*' -C "$LCPP" rev-parse HEAD) || fail "$LCPP is not a git tree" 65
  [ "$have" = "$LCPP_COMMIT" ] || fail "$LCPP is at $have, the pin is $LCPP_COMMIT: move it by hand, knowing which dumps came from which" 65
  dirty=$(git -c safe.directory='*' -C "$LCPP" status --porcelain --untracked-files=no) || fail "cannot inspect tracked changes in $LCPP" 65
  [ -z "$dirty" ] || fail "$LCPP has tracked changes; the oracle is the commit's build, not an edited tree" 65
  t0=$(date +%s)
  cmake -S "$LCPP" -B "$LCPP/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_TESTS=OFF || fail "cmake configure failed"
  lease_bounded 1740 nice -n 19 cmake --build "$LCPP/build" --config Release -j 12 \
    --target llama llama-cli llama-simple llama-tokenize || rc=$?
  echo "raon1-tree: build rc $rc in $(($(date +%s) - t0)) s, end $(now)"
  if [ "$rc" = 0 ]; then
    echo "raon1-tree commit $have libllama md5 $(md5sum < "$(readlink -f "$LCPP/build/bin/libllama.so")" | cut -d' ' -f1)"
  fi
  echo "$rc" > "$dir/rc"
  exit "$rc"
}
main "$@"
