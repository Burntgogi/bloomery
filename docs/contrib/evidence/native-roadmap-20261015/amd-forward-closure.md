# AMD Body35 whole-forward closure: exact 0.8B candidate

Read-only implementation plan, 2026-10-15. Source inspected: selected-kernel stack
`ce87db9ac79c8e9b8658578b2b0289498c326119`, upstream base
`5e837723d31cf860088c96197ba4be52d47c3c3d`. No model was loaded on a GPU and no
whole-forward result is claimed. This is stage 4 preparation; the maintainer's stage 3
A6000/27B validation remains separate.

## Exact input and initial execution lane

Use the fully downloaded `notschmee/Qwen3.5-0.8B-GGUF` revision
`1870603ac8ab59c7ddb9d94cccf3b297e379538f`, `Qwen3.5-0.8B-Q4_K_M.gguf`,
590,880,672 bytes, SHA-256
`be5b3cb8c2f3ebd791349c9c775ea38950f23a2b9e792ca2f0411597dd172988`.
Identity and full-file verification are in `small-model/candidate-identity.json`.

The full-file admission run found F32 133, Q4_K 162, Q6_K 24, Q8_0 1; zero unused
tensors. It still exits 1 because this chat template has no tool-call parser. Preserve
that refusal. A numerical harness over explicit token IDs does not establish chat/tool
serving admission.

First proposed lane: one slot, eager execution, whole-card weights, text only, MTP off,
one recurrent-state lane, F16 KV, context 256, prompt ubatch 1. The existing head-256
four-query-head scalar flash is available (`mma=false`); the metadata's `flash_mma=true`
reports a CUDA-supported option, not a requirement to begin with tensor cores. Keep the
same-file llama.cpp HIP configuration, token IDs, cache type, context and generation
length in the reference record. Increase context to 4096 only after the short lane passes.

## Actual model, from tensors

- Hidden 1024, dense FFN 3584, vocabulary 248320, 24 trunk layers, no routed experts.
- Six gated GQA layers: 3, 7, 11, 15, 19, 23. Eight query heads, two KV heads, width 256,
  group 4; packed query-plus-gate output is 4096 values, K and V 512 each, attention
  output 2048. First 64 head values use RoPE; text positions use the existing table.
- Eighteen GDN layers: the remaining indices. `LinearShape { n_k:16, n_v:16,
  map:Tiled }`, head 128, conv taps 4, channels `(2*16+16)*128=6144`. Output gate 2048,
  beta and alpha 16 each. Recurrence state is value-major `[lane][16][128 v][128 k]`.
- Ten GDN QKV matrices are Q6_K and eight Q4_K. All GDN gate/beta/alpha/output matrices
  are Q4_K. In GQA, Q/K/output are Q4_K; V is Q6_K in two layers and Q4_K in four.
- All 24 FFN gate/up pairs are Q4_K; 12 down matrices are Q6_K and 12 Q4_K.
- The only Q8_0 tensor is `token_embd.weight`, shape `[1024,248320]`. There is no
  `output.weight`: the common `weights::head_tensor`/`Weights::get` tie the output head
  to the same allocation. Do not requantize it or allocate a duplicate head weight.
- F32 tensors are gains, conv coefficients and GDN parameters, so this exact lane does
  not require a general F32 matrix multiplication implementation.

## Ordered implementation slices

Each slice should stay reviewable, retain the original arithmetic/array contracts, and
carry exact-head operator evidence before the next slice is linked. Stage 2's Q4 selected
kernel is a starting primitive; its host Q8 quantization does not close the following list.

| Order | Port / original owner | Format and dependency | Acceptance before moving on |
|---|---|---|---|
| 1 | `weights.rs::{head_tensor,q8_0_planes,Weights::get}`, GGUF `Split`/spec/role readers; HIP buffers, stream, fault allocation | Original Q4/Q6 bytes; Q8_0 packed signed-code words plus F16 block scales; tied head alias; F32 gains | Exact bytes and tensor shapes; no duplicate tied allocation; allocation before steady step; bounds/fault/reset checks |
| 2 | `lib.rs::{q8_1_quant_block,q8_1_quant_vals}`, `fused.rs::norm_quant`; original `cores.rs::{q4_slot,q6_slot,q8_quad}` | F32 RMSNorm -> Q8 blocks of 128; original q4/q6 plane permutations, sums and scales | Packed planes against `qdot::card_q8_1_scalar`; ties-away division/rounding, zero, NaN/Inf refusal, subnormals; same reductions and NormQuant/QuantColumn faults |
| 3 | Q4 core already ported; add `lib.rs::q6k_gemv`, `q6k_sel.rs::q6k_gemv_sel`, `cores.rs` Q6 decode/chain | Q4_K and Q6_K -> F32 using Q8 activation; K=1024/2048/3584, original row/column order | Existing qdot card-rule oracle and existing family-gate contracts; selected and plain outputs match, including Q6 byte windows/half-word funnels |
| 4 | `arch/qwen3moe/proj.rs::{qwen3moe_qkv_q4k,qwen3moe_o_resid_q4k}`, `experts.rs::{qwen3moe_gate_up_swiglu_q4k,qwen3moe_combine}` | Original fused Q4 projection groups/residual and gate-up-SiLU; dense route is fixed id 0, weight 1, one slot | Projection and entire FFN outputs versus existing one-row path; Q6 down and Q6 QKV/V branches exercised; no learned router needed |
| 5 | `linear/conv.rs::gdn_conv_prep`, `linear/delta.rs::gdn_delta`, `linear/norm_gate.rs::gdn_norm_gate` | F32 conv -> normalized Q/K/V, beta/decay, in-place recurrent state, gated RMSNorm; original exp/log polynomials | Reuse `conv_prep_host`, `delta_host`, `norm_gate_host` and `gate_linear` clauses; token positions, ring wrap, reset, nonfinite faults, state transpose relative to llama reference |
| 6 | `rope_neox.rs::head_norm_neox_append_256`, `flash_gqa.rs::{gqa_flash_seg_256_p4,gqa_flash_merge_256}`, `gated_quant.rs::q8_1_quantize_gated` | F32 Q/K norm and 64-value RoPE; F16 KV append; scalar group-4 attention; sigmoid gate -> Q8 output-projection input | Existing gated-attention clauses at positions 0/boundaries/depth; causal live counts, exact cache layout, finite/fault checks; derive error allowance from existing reference gates |
| 7 | `q38.rs::embed_rows_q8_0`, `head.rs::Head::enqueue`, `q8f32.rs::q8_0_gemv`, elementwise RMSNorm and `argmax_finite_fault` | Q8_0 embedding -> F32; final RMSNorm -> Q8_0-weight GEMV reading F32 activations; F32 logits | Exact embedding decode/tied lookup and token bounds; full logits and greedy token against exact-file reference; NaN head refuses/poisons |
| 8 | Existing `Body35::enqueue_chain` -> `Program::walk` -> `dispatch::{mixer,ffn}` -> head, with original `Pre35` layer plans | All preceding slices, original layer ordering and state ownership; no libllama forward | One token, then forced-ID sequence, then short greedy generation against same-file llama.cpp HIP; per-layer taps locate first divergence; reset/repeat identity |
| 9 | Original common model/session capability boundary, followed by original seat | Stream/buffer/kernel/graph ownership must support HIP before `Session<Body35>` can own it | Same Session/SeatEngine path, then HTTP versus CLI, slots/prefix reset/cancellation; only then call it native Bloomery serving |

Slices 4/5 close GDN layers; slices 4/6 close GQA layers; every layer also needs the dense
FFN. Scalar probes from the earlier lab do not substitute for these original reductions
or for the recurrence. A GPU forward with host quantization between each operation would
not satisfy the intended all-device steady chain.

## Resources and formulas (derived, not allocated or measured)

Let H=1024, F=3584, V=248320, Q=4096, A=2048, KV=512, C=6144, nv=16,
context T, prompt rows U, and one recurrent lane. These formulas describe existing owners;
reuse their sizing rather than introducing a second memory planner.

| Resource | Existing owner / formula | Exact-shape value |
|---|---|---|
| Packed weight payload | Tensor products times block bytes: Q4_K 144/256, Q6_K 210/256, Q8_0 34/32, F32 4 | 579,919,104 B (Q8_0 alone 270,172,160 B); GGUF header/padding separate |
| Recurrent states | `LinearShape::state_len`, 18 * lanes * nv * 128 * 128 * 4 | 18,874,368 B at one lane |
| Conv rings | `LinearShape::ring_len`, 18 * **11** * C * 4 (`RING_ROWS=3+PASS_ROWS`, PASS_ROWS=8) | 4,866,048 B; do not size as only three previous rows |
| F16 KV | `KvPlanes::new`, 6 layers * 2 planes * 2 KV heads * T * 256 * 2 | 12,288*T B: 3,145,728 B at T256; 50,331,648 B at T4096 |
| Text RoPE table | `RopeRows`, T * 64 * 4 | 65,536 B at T256; 1,048,576 B at T4096 |
| Full Q8Act per column | `wide::act_col_bytes(K)`, n=K/256: 512*ceil(n/2)+4*(256*ceil(n/4)+128*ceil(n/2))+40*n | K1024=3,232 B; K2048=6,464 B; K3584=11,824 B |
| GDN temporaries | `GdnArena`, 4*[R*(2*C+2*A+4*nv) + (R>1 ? min(R,8)*C : 0)] | R1=65,792 B; R8=722,944 B |
| Step/pass/prompt arenas | `wide::arena_bytes` and `Arena::with`; candidate has fused gate/up, no separate GLU, no unfused projection `cols` | A(1)=818,624 B, A(8)=7,455,040 B; existing Body35 allocates A(1)+A(8)+A(U), even with eager/ubatch1 |
| Decode head | `Head::with_norm`, 2*H*4 + Q8Act(H) + V*4 + 3*4 | 1,004,716 B; Q8Act allocated by common owner although this Q8_0 head reads F32 |
| Fault words | `FAULT_WORDS=65537`, 4 bytes each | 262,148 B |

The arena formula includes fixed scalar-flash partials: 80 segments * 8 heads * R *
(256+2) F32 values, and all narrow-width Q8 scratch forms for m=1..min(R,8).
At T256/U1 the listed components total 617,229,936 B (~588.64 MiB). This subtotal
excludes small input/prompt records, checkpoint/pinned-host storage, kernel bundles,
driver/runtime allocation overhead and the common load's **256 MiB FIT_RESERVE**.
It is not a VRAM residency measurement. Do not infer available memory from a 16 GiB card's
capacity while Strata owns that card. Measure free bytes at the eventual authorized load
window and retain the existing fit refusal.

## Where the shared serving path currently requires CUDA

- `crates/app/src/lib.rs::Session<B>` owns concrete `GpuModel<B>`; `Open`, `Prompt`,
  `Keep`, `Loaded::ready`, slot and logits methods all take that type. It is not generic
  over an arbitrary `runtime::Target`.
- `crates/gpu/src/model.rs::ChainBody::refresh` and `Rows::plan_rows` take `CudaStream`;
  `enqueue_chain` takes concrete `Gpu`, `Weights`, `Head`. `GpuModel` owns CUDA graphs,
  those concrete resources, parked-slot captures, and poison/position state.
- `Gpu` owns `Arc<CudaContext>`, `Arc<CudaStream>`, CUDA modules and `DeviceBuffer` fault
  storage. `Weights`, `DeviceTensor`, `Q8Act`, Body35 arenas/stores, head, copy/checkpoint
  operations and captured addresses use CUDA buffer/context APIs directly.
- `Kernels::load(gpu,true)` eagerly loads the common Qwen CUDA module set, including
  branches this dense one-token lane does not execute. Operator closure alone therefore
  does not make current Body35 constructors usable with HIP.

After exact-file whole-forward evidence, review the concrete device-resource and enqueue
boundary with the maintainer. Keep GGUF/spec/placement, Body35 layer plans, `Program`,
Session, token scheduling and the HTTP seat as common owners. Do not implement a guessed
backend trait or a separate serving engine now; replacing only the HTTP Engine adapter
would not meet the same-engine requirement. Graph replay, wide prefill/GEMM, multi-slot
states/checkpoints and performance comparison follow the first shared-path eager proof.

## Stage 4 completion evidence

Record source and llama.cpp commits, exact model SHA, target/runtime, every command and
exit code, chosen token IDs, load allocation totals, per-layer taps/logits where needed,
forced-ID and greedy-token comparisons, reset/repeat results, and explicit supported
features. Raw-token forward success and tool-parser/chat admission are separate facts.
Do not close stage 4 on a document, operator tests, or an unexecuted benchmark manifest.
