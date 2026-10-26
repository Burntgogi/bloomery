#!/usr/bin/env bash
# Link musespark_dump (tools/ref/cb_dump.cpp, with tools/ref/ref_io.h) against the existing mainline build
# of the muse-spark profile ($LCPP/build/bin: llama.cpp ba3c05f71, PR 30616's merge, CPU, no patch; its
# tracked files must be clean). This does not configure or rebuild that tree: a rebuild moves the libllama
# md5 the oracle sets name. Output: $BLOOMERY_DATA/bin/musespark_dump and musespark_dump.build beside it
# (MUSESPARK_OUT moves them), the record tools/ref/musespark-dump.sh compares: the source sha256 (cpp then
# header), the tree's commit and the md5 of the libllama the binary loads — the 12 digits the refset family
# pins. The build is tools/ref/cb-dump.sh's; the family's part is musespark-dump.sh.
set -euo pipefail
exec bash "${BASH_SOURCE[0]%/*}/musespark-dump.sh" --build
