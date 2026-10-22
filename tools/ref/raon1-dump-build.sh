#!/usr/bin/env bash
# Link raon1_dump (tools/ref/cb_dump.cpp, with tools/ref/ref_io.h) against the Raon-1 oracle tree (KRAFTON's llama.cpp branch
# raon-1-mini-preview, CPU, no patch; tools/ref/raon1-tree.sh builds it, and its tracked files must be clean). This does not
# configure or rebuild that tree: a rebuild moves the libllama md5 the oracle sets name. Output: $BLOOMERY_DATA/bin/raon1_dump and
# raon1_dump.build beside it (RAON1_OUT moves them), the record tools/ref/raon1-dump.sh compares: the source sha256 (cpp then
# headers), the tree's commit and the md5 of the libllama the binary loads. The build is tools/ref/cb-dump.sh's; the family's part is
# raon1-dump.sh.
set -euo pipefail
exec bash "${BASH_SOURCE[0]%/*}/raon1-dump.sh" --build
