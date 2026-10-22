#!/usr/bin/env bash
# Dump the Raon-1-mini-preview oracle sets: KRAFTON's llama.cpp branch raon-1-mini-preview (CPU build, no patch, models/raon1.sh) on the
# Q4_K_M file over the four prompts of tools/ref/data/raon1, from ONE model load, as dump_ref-format sets under $BLOOMERY_DATA:
#   ref_raon1_a  English paragraph   every per-layer tap below, 32 greedy tokens
#   ref_raon1_b  Korean paragraph    each layer output and the router's expert ids, 32 greedy tokens
#   ref_raon1_c  Rust function       the same as b
#   ref_raon1_d  1,536 ids of wikitext-2: the last position's logits only (more than 16 positions: the RDA chunked scan runs many
#                chunks, the MLA layers attend a context of 1,536)
# Each set also holds result_output (the last position's whole logits row), topk.tsv (its 40 largest logits with their f64
# log-softmax) and greedy.tsv (the argmax tokens with their margins). Set names are the family's (crates/refset/src/arch/raon1);
# Raon has no fixture file yet, so BLOOMERY_TIER=fixture is refused by name (exit 66, ref-paths.sh).
#
# The harness is the shared one: tools/ref/cb_dump.cpp is the dumper, tools/ref/cb-dump.sh the build and the run (staging, lease,
# install); this file is Raon's part of them, the tree, the taps and the prompts. `bash tools/ref/raon1-dump.sh --build` links the
# dumper against the profile's tree (tools/ref/raon1-dump-build.sh is that call); tools/ref/raon1-tree.sh builds the tree.
#
# The taps are the graph's cb() names (src/models/raon1.cpp, delta-net-base.cpp, llama-graph.cpp build_ffn / build_moe_ffn), a layer's
# name `<node>-<il>`; a name that a layer's kind does not have is absent from that layer, and hc_* repeat once a site:
#   attn_norm                  the attention site's input after the mHC pre-mix and the norm
#   hc_mixes, hc_comb, hc_pre, hc_post   an mHC site's selector rows, its doubly stochastic mixing matrix, the pre-mix weights and the
#                              post weights (twice a layer: the attention site first, then the ffn site)
#   rda_q_conv, rda_k_conv, rda_v_conv   the short convolutions' outputs after SiLU (RDA layers: all but 3, 7, 11, 15, 19, 23, 24)
#   rda_g1, rda_v_scaled, rda_beta   the decay gate, the value scaled by the write/erase ratio, the erase gate
#   rda_scan_out, new_state    the scan's output and the recurrent state after the ubatch ([128, 128, 40], the last position's state
#                              for a prompt of one ubatch)
#   rda_g2, rda_normed, rda_out   the output gate, the RMS-normed scan output and the layer output after wo
#   mla_gated                  the gated attention output before wo (MLA layers: 3, 7, 11, 15, 19, 23, 24)
#   hc_attn_out                the residual streams after the attention site's post-mix ([2560, 4 streams, tokens])
#   ffn_norm                   the ffn site's input after the mHC pre-mix and the norm
#   ffn_situ                   the dense layer's and the shared expert's SiTU activation (their down projection is unnamed in the graph)
#   ffn_moe_probs, ffn_moe_topk, ffn_moe_weights_norm   the router: sigmoid probabilities, the chosen expert ids (an i32 node: the f32
#                              file, its i32 twin and its `int` row), their normalized weights (the scale is 1: no *_scaled node)
#   ffn_moe_situ, ffn_moe_out  the routed experts' SiTU activation and their summed output (shared expert not included)
#   l_out, result_norm         the residual streams after the ffn site's post-mix, and the normed mean of the streams at the last row
#
# CPU only: CUDA_VISIBLE_DEVICES is emptied and the dumper asks for no GPU layers, so the reference is a CPU reduction order, as
# dump.sh's. -fa on: the engine's attention is the flash-attention one. The run takes the CPU lease (REF_DUMP_LEASE=0 only for a dry
# reading). DUMP_DRY=1 prints what would run and stops with 0, before the lease and anything on disk.
#
# Wall [derived, docs/cards/raon1-dump.card].
set -euo pipefail
export BLOOMERY_MODEL=raon1
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
# shellcheck source=tools/ref/cb-dump.sh
source "${BASH_SOURCE[0]%/*}/cb-dump.sh"

CB_FAMILY=raon1
CB_ENV=RAON1
CB_SCRIPT=raon1-dump.sh
CB_TREE=$LCPP
CB_PATCH=
CB_ARCH=raon1
CB_TITLE="raon1-dump — KRAFTON's llama.cpp branch raon-1-mini-preview intermediate tensors, raw f32, little-endian"
CB_IDS=raon1
ROUTER=ffn_moe_topk
TAPS=attn_norm+hc_mixes+hc_comb+hc_pre+hc_post+rda_q_conv+rda_k_conv+rda_v_conv+rda_g1+rda_v_scaled+rda_beta+rda_scan_out+new_state+rda_g2+rda_normed+rda_out+mla_gated+hc_attn_out+ffn_norm+ffn_situ+ffn_moe_probs+$ROUTER+ffn_moe_weights_norm+ffn_moe_situ+ffn_moe_out+l_out+result_norm
# name:ids:nodes:greedy — nodes joined by `+`.
CB_PROMPTS=("a:a:$TAPS:32" "b:b:l_out+$ROUTER:32" "c:c:l_out+$ROUTER:32" "d:d::0")
CB_FLAGS=(-fa on)

case ${1:-} in
  --build) cb_dump_build ;;
  '')
    # shellcheck source=tools/ref/lease.sh
    source "${BASH_SOURCE[0]%/*}/lease.sh"
    cb_dump_run
    ;;
  *) echo "raon1-dump.sh: no argument (the dump) or --build, got $1" >&2; exit 64 ;;
esac
