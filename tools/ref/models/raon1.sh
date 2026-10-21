#!/usr/bin/env bash
# shellcheck shell=bash
# The raon1 (Raon-1-mini-preview Q4_K_M, 28.8B total / 4.3B active, KRAFTON) profile: what the reference harness needs that is a
# property of the model. Sourced by ref-paths.sh when BLOOMERY_MODEL=raon1; never executed, and it exports nothing.
#
# The name is the GGUF general.architecture value, so `grep raon1` finds this profile, the registry row and the oracle family
# (crates/refset/src/arch/raon1).
#
#   MODEL   the Q4_K_M file (19.7 GB, one file). BLOOMERY_REF_MODEL moves it; a caller's own MODEL= is ignored. The path is the
#           family's identity (refset::arch::raon1::MODEL): a set names its file's full path, so the file is not moved or deleted
#           before its sets are re-dumped.
#   LCPP    KRAFTON's llama.cpp branch raon-1-mini-preview at LCPP_COMMIT (/home/user/llama.cpp-raon, a CPU build, no patch):
#           src/models/raon1.cpp is the vendor's own graph, so the oracle is that branch and the public baseline is the same
#           engine, named as a vendor patch in every table. tools/ref/raon1-tree.sh builds it. ik has no Raon graph; the default IK
#           is only so IKBIN can be formed.
#
# The dump (tools/ref/raon1-dump.sh) hides CUDA and takes the CPU lease.
#
# SC2034: every name here is read by the file that sources this one.
# shellcheck disable=SC2034
MODEL_NAME=raon1
: "${IK:=/home/user/ik-idxkey}"
: "${LCPP:=/home/user/llama.cpp-raon}"
LCPP_COMMIT=fb895662d25868756a19253ca5e0688f99b18994
MODEL=${BLOOMERY_REF_MODEL:-/models/Raon-1-mini-preview/raon-1-mini-preview-Q4_K_M.gguf}
