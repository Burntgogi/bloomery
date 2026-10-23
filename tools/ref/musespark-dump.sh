#!/usr/bin/env bash
# Dump the Muse Spark Open oracle sets: mainline llama.cpp's (PR 30616's merge, ba3c05f71, CPU build, no
# patch) forward pass over the five prompts of tools/ref/data/musespark on the profile's UD-IQ2_XXS file,
# from ONE model load, as dump_ref-format sets under $BLOOMERY_DATA:
#   ref_musespark_a  every per-layer tap below, 64 greedy tokens
#   ref_musespark_b  (Korean) each layer output l_out and the router's expert ids, 64 greedy tokens
#   ref_musespark_c  (Rust) the same as b
#   ref_musespark_d  1,536 ids of wikitext-2: the last position's logits only
#   ref_musespark_e  2,560 ids of wikitext-2 (d's ids and 1,024 more): each layer's output — the only
#                    context past the 2,048-position sliding window, so the sliding layers read a window of
#                    real cells and the 16 full layers the whole context (the iswa keep rule)
# Each set also holds topk.tsv (the last position's 40 largest logits with their f64 log-softmax) and
# greedy.tsv (the argmax tokens with their margins). Set names are the family's
# (crates/refset/src/arch/musespark); under BLOOMERY_TIER=fixture each is the fixture file's, behind `fx_`,
# with its `# fixture` line.
#
# The harness is the shared one: tools/ref/cb_dump.cpp is the dumper, tools/ref/cb-dump.sh the build and the
# run (staging, lease, install); this file is Muse Spark's part of them, the tree, the taps and the prompts.
# `bash tools/ref/musespark-dump.sh --build` links the dumper against the muse tree
# (tools/ref/musespark-dump-build.sh is that call).
#
# The ids are tokens of the muse tree's llama-tokenize on the file, taken once and kept in
# tools/ref/data/musespark; the dumper never tokenizes (a tokenizer difference would read as a kernel bug).
# The file sets add_bos_token, so the ids are the text's own (--no-bos), the same mode the tokenizer
# oracle's set takes. Prompt a/b/c text: /home/user/stage/sorc/prompts/{a,b,c}.txt (sorc round, 2026-10-20,
# Solar's prompts); d and e are the first 1,536 and 2,560 ids of /home/user/eval/wikitext-2-raw/wiki.test.raw.
# To take them: one llama-tokenize a prompt, reshaped one id a line —
#   CUDA_VISIBLE_DEVICES= $LCPP/build/bin/llama-tokenize -m "$MODEL" -f TEXT --ids --log-disable \
#     --no-bos --no-escape | tr -d '[] ' | tr ',' '\n' | grep -v '^$' > tools/ref/data/musespark/NAME.ids
#
# The taps are the graph's cb() names (src/models/muse-spark.cpp, llama-graph.cpp build_moe_ffn), a layer's
# name `<node>-<il>`:
#   embd_norm, result_norm                      the weightless embedding norm and the head's norm
#   attn_norm                                   the layer input, normed
#   attn_gate_proj, attn_gate_sig               the output gate's projection and its sigmoid
#   Qcur_normed, Kcur_normed                    the per-head q/k norms (the q one carries the MetaP gain)
#   Qcur_rope, Kcur_rope                        the roped q/k — the sliding layers only; the full layers
#                                              rope nothing, so no occurrence exists on them
#   attn_out, attn_o_norm, attn_gated, attn_o_proj  the attention output, its per-head weightless norm,
#                                              the gated product and the output projection
#   attn_post_norm, ffn_inp                     the weightless 1e-8 post-norm and the α/β residual
#   ffn_norm, ffn_latent_down, ffn_pre_exp_norm the latent entry: the norm, the down projection, the
#                                              experts' own norm
#   ffn_moe_logits                              the router's logits from the full stream: the file's
#                                              ffn_gate_inp is pre-scaled (the MetaP output multiplier
#                                              folded at conversion), and a body that multiplies again
#                                              diverges here by a constant per logit
#   ffn_moe_probs, ffn_moe_topk, ffn_moe_weights_scaled   the sigmoid scores, the chosen expert ids (an
#                                              i32 node: the f32 file, its i32 twin and its `int` row —
#                                              the selection the exp_probs_b bias steers) and the
#                                              unbiased weights, renormalised, the router applies
#   ffn_moe_out, ffn_latent_up, ffn_post_norm   the experts' weighted sum, the latent exit and the
#                                              scale-then-normalise post-FFN norm
#   l_out                                       the layer output, the FFN α/β residual through it
# `result_output` (the softcapped logits: ×0.1767767, /20, tanh, ×20) is always the last logits row.
#
# CPU only: CUDA_VISIBLE_DEVICES is emptied and the dumper asks for no GPU layers, so the reference is a
# CPU reduction order, as dump.sh's. -b 4096 so e is one llama_decode. The run takes the CPU lease
# (REF_DUMP_LEASE=0 only for a dry reading). DUMP_DRY=1 prints what would run and stops with 0, before the
# lease and anything on disk.
#
# Wall [derived, docs/cards/musespark-dump.card]: the 225.6 GB file is mapped, and this tree's CPU backend
# repacks none of its types on this AVX2 CPU (IQ2_XXS, Q6_K, Q8_0 and Q4_K all stay page-cache reads; they
# fit the 251 GB box with ~25 GB to spare). Prefill of the five prompts' ~4,300 ids at 2 × 44.5 G active
# MACs a token is ~770 TFLOP of unrepacked IQ2_XXS dot products, and greedy 192 tokens read ~6.4 GB of
# experts a token through ~120 GB/s; the first run also pages in the file at the root drive's ~3 GB/s.
set -euo pipefail
export BLOOMERY_MODEL=muse-spark
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
# shellcheck source=tools/ref/cb-dump.sh
source "${BASH_SOURCE[0]%/*}/cb-dump.sh"

CB_FAMILY=musespark
CB_ENV=MUSESPARK
CB_SCRIPT=musespark-dump.sh
CB_TREE=$LCPP
CB_PATCH=
CB_ARCH=muse-spark
CB_TITLE="musespark-dump — mainline llama.cpp's intermediate tensors, raw f32, little-endian"
CB_IDS=musespark
ROUTER=ffn_moe_topk
TAPS=embd_norm+attn_norm+attn_gate_proj+Qcur_normed+Kcur_normed+Qcur_rope+Kcur_rope+attn_out+attn_o_norm+attn_gate_sig+attn_gated+attn_o_proj+attn_post_norm+ffn_inp+ffn_norm+ffn_latent_down+ffn_pre_exp_norm+ffn_moe_logits+ffn_moe_probs+$ROUTER+ffn_moe_weights_scaled+ffn_moe_out+ffn_latent_up+ffn_post_norm+l_out+result_norm
# name:ids:nodes:greedy — nodes joined by `+`.
CB_PROMPTS=("a:a:$TAPS:64" "b:b:l_out+$ROUTER:64" "c:c:l_out+$ROUTER:64" "d:d::0" "e:e:l_out:0")
CB_FLAGS=(-b 4096)

case ${1:-} in
  --build) cb_dump_build ;;
  '')
    # shellcheck source=tools/ref/lease.sh
    source "${BASH_SOURCE[0]%/*}/lease.sh"
    cb_dump_run
    ;;
  *) echo "musespark-dump.sh: no argument (the dump) or --build, got $1" >&2; exit 64 ;;
esac
