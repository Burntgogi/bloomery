// Metal kernels for the M2.0 probe (tools/metal-probe/probe.swift).
//
// Three kernels, each observable so a skipped dispatch cannot go unnoticed:
//
//   empty       one threadgroup of 32 threads; thread 0 adds 1 to an atomic counter.
//   stream_read reads `count` uint4 words with a grid-stride loop, XORs them into one
//               uint per threadgroup. A dispatch that reads nothing still writes its
//               slot, so the host checksum sees the reads.
//   chain_slice stream_read over one slice of a large buffer, plus the same counter add.
//               Dispatch i gets the pointer at i * slice and the slice's word count;
//               the counter add proves the dispatch ran.
//
// The M2.2 quantized mat-vec kernels (issue #593):
//
//   q4k_gemv_ggml  y[r] = sum_c W[r,c] * x[c] for one f32 activation vector x,
//                  W in GGUF block_q4_K layout, one dispatch. Adapted from
//                  llama.cpp's kernel_mul_mv_q4_K_f32:
//                    ggml/src/ggml-metal/kernels/mul_mv.metal
//                    llama.cpp commit 36a73916e (MIT licence)
//                  The lane map (ix/it/iq/ir), the 4-block loop, the uint16
//                  nibble trick, the row loop that advances its weight pointers
//                  by a row's bytes, and the simd_sum reduction are llama.cpp's;
//                  2 rows per SIMD group, 2 SIMD groups per threadgroup.
//   q6k_gemv_ggml  the same for block_q6_K, from kernel_mul_mv_q6_K_f32 in the
//                  same file and commit.
//   q4k_gemv_own / q6k_gemv_own  the probe's variant: the ggml loop with 4 rows per SIMD
//                  group and 4 SIMD groups per threadgroup (16 rows a
//                  threadgroup instead of 4), so the activation loads and the
//                  loop overhead are shared by 4 rows.
//   q4k_gemv_brk / q6k_gemv_brk  the ggml lane map with another row loop: each
//                  row's base is recomputed from W and the loop exits at the last
//                  row. ownbrk is the same with 4 x 4. Measured against ggml in
//                  the sitting (the loop form changes the register count).
//
// Block layouts are ggml/src/ggml-common.h (block_q4_K, block_q6_K); the
// dequantizers the probe's CPU reference follows are ggml/src/ggml-quants.c
// (dequantize_row_q4_K, dequantize_row_q6_K), same commit.
//
// Compile offline (when the Metal toolchain component is installed):
//   xcrun -sdk macosx metal -c kernels.metal -o build/kernels.air
//   xcrun -sdk macosx metallib build/kernels.air -o build/kernels.metallib
// or let probe.swift compile this source at run time (device.makeLibrary(source:)).

#include <metal_stdlib>
using namespace metal;

kernel void empty(device atomic_uint *counter [[buffer(0)]],
                  uint tid [[thread_position_in_threadgroup]]) {
    if (tid == 0) {
        atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    }
}

kernel void stream_read(device const uint4 *src [[buffer(0)]],
                        device uint *out [[buffer(1)]],
                        constant ulong &count [[buffer(2)]],
                        uint gid [[thread_position_in_grid]],
                        uint lid [[thread_index_in_threadgroup]],
                        uint tgid [[threadgroup_position_in_grid]],
                        uint tg_size [[threads_per_threadgroup]],
                        uint tg_count [[threadgroups_per_grid]]) {
    uint acc = 0;
    ulong stride = (ulong)tg_count * (ulong)tg_size;
    for (ulong i = (ulong)gid; i < count; i += stride) {
        uint4 v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    uint group_xor = simd_xor(acc);
    threadgroup uint scratch[32];
    uint lane = lid % 32;
    uint simd_id = lid / 32;
    if (lane == 0) scratch[simd_id] = group_xor;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        uint total = 0;
        uint simd_count = (tg_size + 31) / 32;
        for (uint i = 0; i < simd_count; ++i) total ^= scratch[i];
        out[tgid] = total;
    }
}

kernel void chain_slice(device const uint4 *src [[buffer(0)]],
                        device uint *out [[buffer(1)]],
                        constant ulong &count [[buffer(2)]],
                        device atomic_uint *counter [[buffer(3)]],
                        uint gid [[thread_position_in_grid]],
                        uint lid [[thread_index_in_threadgroup]],
                        uint tgid [[threadgroup_position_in_grid]],
                        uint tg_size [[threads_per_threadgroup]],
                        uint tg_count [[threadgroups_per_grid]]) {
    uint acc = 0;
    ulong stride = (ulong)tg_count * (ulong)tg_size;
    for (ulong i = (ulong)gid; i < count; i += stride) {
        uint4 v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    uint group_xor = simd_xor(acc);
    threadgroup uint scratch[32];
    uint lane = lid % 32;
    uint simd_id = lid / 32;
    if (lane == 0) scratch[simd_id] = group_xor;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        uint total = 0;
        uint simd_count = (tg_size + 31) / 32;
        for (uint i = 0; i < simd_count; ++i) total ^= scratch[i];
        out[tgid] = total;
    }
    if (lid == 0 && tgid == 0) {
        atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    }
}

// MARK: - M2.2 quantized mat-vec (issue #593)

#define QK_K 256
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

struct block_q4_K {
    half    d;
    half    dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    char    scales[16];
    half    d;
};

struct GemvArgs {
    int rows;
    int cols;
    int nblocks;
    int pad0;
};

static_assert(sizeof(block_q4_K) == 144, "block_q4_K is 144 bytes");
static_assert(sizeof(block_q6_K) == 210, "block_q6_K is 210 bytes");

// The row loop is llama.cpp's: no per-row bound check, the weight pointers advance by
// one row's bytes. The host pads each matrix to a whole threadgroup of rows, so the rows
// past args.rows read padding and are never stored.
template <short NR0, short NSG>
inline void q4k_gemv_body(device const block_q4_K * W,
                          device const float     * x,
                          device float           * y,
                          constant GemvArgs      & args,
                          uint3  tgpig,
                          ushort tiisg,
                          ushort sgitg) {
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg/8;  // 0...3
    const short it = tiisg%8;  // 0...7
    const short iq = it/4;     // 0 or 1
    const short ir = it%4;     // 0...3

    const int nb = args.nblocks;
    const int first_row = (int)(tgpig.x * NSG + sgitg) * NR0;
    const ulong row_stride = (ulong)nb * (sizeof(block_q4_K) / 2);  // in uint16 (and half)

    device const block_q4_K * xb = W + (ulong)first_row * (ulong)nb;

    float yl[16];
    float yh[16];
    float sumf[NR0];
    for (short r = 0; r < NR0; ++r) sumf[r] = 0.f;

    device const float * y4 = x + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        device const uint16_t * sc = (device const uint16_t *)xb[ib].scales + iq;
        device const uint16_t * q1 = (device const uint16_t *)xb[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &xb[ib].d;

        for (short row = 0; row < NR0; ++row) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                  (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                  sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += row_stride;
            sc += row_stride;
            dh += row_stride;
        }

        y4 += 4 * QK_K;
    }

    for (short row = 0; row < NR0 && first_row + row < args.rows; ++row) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) y[first_row + row] = sum_all;
    }
}

// Same row loop as above, from kernel_mul_mv_q6_K_f32 (row pointers advance by row_bytes).
template <short NR0, short NSG>
inline void q6k_gemv_body(device const block_q6_K * W,
                          device const float     * x,
                          device float           * y,
                          constant GemvArgs      & args,
                          uint3  tgpig,
                          ushort tiisg,
                          ushort sgitg) {
    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = args.nblocks;
    const int first_row = (int)(tgpig.x * NSG + sgitg) * NR0;
    const ulong row_bytes = (ulong)nb * sizeof(block_q6_K);

    device const block_q6_K * xb = W + (ulong)first_row * (ulong)nb;

    float sumf[NR0];
    for (short r = 0; r < NR0; ++r) sumf[r] = 0.f;

    float yl[16];

    const short tid = tiisg/2;
    const short ix  = tiisg%2;
    const short ip  = tid/8;   // 0 or 1
    const short il  = tid%8;
    const short l0  = 4*il;
    const short is  = 8*ip + l0/16;

    const short y_offset   = 128*ip + l0;
    const short q_offset_l =  64*ip + l0;
    const short q_offset_h =  32*ip + l0;

    for (int i = ix; i < nb; i += 2) {
        device const uint8_t * q1 = xb[i].ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = xb[i].qh + q_offset_h;
        device const char    * sc = xb[i].scales + is;
        device const half    * dh = &xb[i].d;

        device const float * yb = x + i * QK_K + y_offset;

        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = yb[l +  0];
            yl[4*l + 1] = yb[l + 32];
            yl[4*l + 2] = yb[l + 64];
            yl[4*l + 3] = yb[l + 96];
        }

        for (short row = 0; row < NR0; ++row) {
            float4 sums = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((char)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((char)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((char)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((char)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] +
                                  sums[2] * sc[4] + sums[3] * sc[6]);

            q1 += row_bytes;
            q2 += row_bytes;
            qh += row_bytes;
            sc += row_bytes;
            dh += row_bytes / 2;
        }
    }

    for (short row = 0; row < NR0 && first_row + row < args.rows; ++row) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) y[first_row + row] = sum_all;
    }
}

// The previous loop form: each row's base is recomputed from W and the row loop leaves
// at the matrix's last row. The compiler cannot hoist a row's loads above the previous
// row's check, so fewer values are live and the pipeline state allows larger threadgroups
// (pso_max_tg in the gemv record).
template <short NR0, short NSG>
inline void q4k_gemv_body_brk(device const block_q4_K * W,
                          device const float     * x,
                          device float           * y,
                          constant GemvArgs      & args,
                          uint3  tgpig,
                          ushort tiisg,
                          ushort sgitg) {
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg/8;  // 0...3
    const short it = tiisg%8;  // 0...7
    const short iq = it/4;     // 0 or 1
    const short ir = it%4;     // 0...3

    const int nb = args.nblocks;
    const int first_row = (int)(tgpig.x * NSG + sgitg) * NR0;

    float yl[16];
    float yh[16];
    float sumf[NR0];
    for (short r = 0; r < NR0; ++r) sumf[r] = 0.f;

    device const float * y4 = x + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        for (short row = 0; row < NR0; ++row) {
            const int r = first_row + row;
            if (r >= args.rows) break;
            device const block_q4_K * b = W + (ulong)r * (ulong)nb + (ulong)ib;

            device const uint16_t * sc = (device const uint16_t *)b->scales + iq;
            device const uint16_t * q1 = (device const uint16_t *)b->qs + 16 * iq + 4 * ir;
            device const uint16_t * q2 = q1 + 32;
            const half dh0 = b->d;
            const half dh1 = b->dmin;

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += float(dh0) * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                       (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                       (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                       (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         float(dh1) * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                       sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }

        y4 += 4 * QK_K;
    }

    for (short row = 0; row < NR0; ++row) {
        const int r = first_row + row;
        if (r >= args.rows) break;
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) y[r] = sum_all;
    }
}

template <short NR0, short NSG>
inline void q6k_gemv_body_brk(device const block_q6_K * W,
                          device const float     * x,
                          device float           * y,
                          constant GemvArgs      & args,
                          uint3  tgpig,
                          ushort tiisg,
                          ushort sgitg) {
    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = args.nblocks;
    const int first_row = (int)(tgpig.x * NSG + sgitg) * NR0;

    float sumf[NR0];
    for (short r = 0; r < NR0; ++r) sumf[r] = 0.f;

    float yl[16];

    const short tid = tiisg/2;
    const short ix  = tiisg%2;
    const short ip  = tid/8;   // 0 or 1
    const short il  = tid%8;
    const short l0  = 4*il;
    const short is  = 8*ip + l0/16;

    const short y_offset   = 128*ip + l0;
    const short q_offset_l =  64*ip + l0;
    const short q_offset_h =  32*ip + l0;

    for (int i = ix; i < nb; i += 2) {
        device const float * yb = x + i * QK_K + y_offset;

        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = yb[l +  0];
            yl[4*l + 1] = yb[l + 32];
            yl[4*l + 2] = yb[l + 64];
            yl[4*l + 3] = yb[l + 96];
        }

        for (short row = 0; row < NR0; ++row) {
            const int r = first_row + row;
            if (r >= args.rows) break;
            device const block_q6_K * b = W + (ulong)r * (ulong)nb + (ulong)i;

            device const uint8_t * q1 = b->ql + q_offset_l;
            device const uint8_t * q2 = q1 + 32;
            device const uint8_t * qh = b->qh + q_offset_h;
            device const char    * sc = b->scales + is;

            float4 sums = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((char)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((char)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((char)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((char)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf[row] += float(b->d) * (sums[0] * sc[0] + sums[1] * sc[2] +
                                        sums[2] * sc[4] + sums[3] * sc[6]);
        }
    }

    for (short row = 0; row < NR0; ++row) {
        const int r = first_row + row;
        if (r >= args.rows) break;
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) y[r] = sum_all;
    }
}

kernel void q4k_gemv_ggml(device const block_q4_K * W [[buffer(0)]],
                          device const float     * x [[buffer(1)]],
                          device float           * y [[buffer(2)]],
                          constant GemvArgs      & args [[buffer(3)]],
                          uint3  tgpig [[threadgroup_position_in_grid]],
                          ushort tiisg [[thread_index_in_simdgroup]],
                          ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    q4k_gemv_body<2, 2>(W, x, y, args, tgpig, tiisg, sgitg);
}

kernel void q4k_gemv_own(device const block_q4_K * W [[buffer(0)]],
                         device const float     * x [[buffer(1)]],
                         device float           * y [[buffer(2)]],
                         constant GemvArgs      & args [[buffer(3)]],
                         uint3  tgpig [[threadgroup_position_in_grid]],
                         ushort tiisg [[thread_index_in_simdgroup]],
                         ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    q4k_gemv_body<4, 4>(W, x, y, args, tgpig, tiisg, sgitg);
}

kernel void q6k_gemv_ggml(device const block_q6_K * W [[buffer(0)]],
                          device const float     * x [[buffer(1)]],
                          device float           * y [[buffer(2)]],
                          constant GemvArgs      & args [[buffer(3)]],
                          uint3  tgpig [[threadgroup_position_in_grid]],
                          ushort tiisg [[thread_index_in_simdgroup]],
                          ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    q6k_gemv_body<2, 2>(W, x, y, args, tgpig, tiisg, sgitg);
}

kernel void q6k_gemv_own(device const block_q6_K * W [[buffer(0)]],
                         device const float     * x [[buffer(1)]],
                         device float           * y [[buffer(2)]],
                         constant GemvArgs      & args [[buffer(3)]],
                         uint3  tgpig [[threadgroup_position_in_grid]],
                         ushort tiisg [[thread_index_in_simdgroup]],
                         ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    q6k_gemv_body<4, 4>(W, x, y, args, tgpig, tiisg, sgitg);
}

kernel void q4k_gemv_brk(device const block_q4_K * W [[buffer(0)]],
                         device const float     * x [[buffer(1)]],
                         device float           * y [[buffer(2)]],
                         constant GemvArgs      & args [[buffer(3)]],
                         uint3  tgpig [[threadgroup_position_in_grid]],
                         ushort tiisg [[thread_index_in_simdgroup]],
                         ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    q4k_gemv_body_brk<2, 2>(W, x, y, args, tgpig, tiisg, sgitg);
}

kernel void q4k_gemv_ownbrk(device const block_q4_K * W [[buffer(0)]],
                            device const float     * x [[buffer(1)]],
                            device float           * y [[buffer(2)]],
                            constant GemvArgs      & args [[buffer(3)]],
                            uint3  tgpig [[threadgroup_position_in_grid]],
                            ushort tiisg [[thread_index_in_simdgroup]],
                            ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    q4k_gemv_body_brk<4, 4>(W, x, y, args, tgpig, tiisg, sgitg);
}

kernel void q6k_gemv_brk(device const block_q6_K * W [[buffer(0)]],
                         device const float     * x [[buffer(1)]],
                         device float           * y [[buffer(2)]],
                         constant GemvArgs      & args [[buffer(3)]],
                         uint3  tgpig [[threadgroup_position_in_grid]],
                         ushort tiisg [[thread_index_in_simdgroup]],
                         ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    q6k_gemv_body_brk<2, 2>(W, x, y, args, tgpig, tiisg, sgitg);
}

kernel void q6k_gemv_ownbrk(device const block_q6_K * W [[buffer(0)]],
                            device const float     * x [[buffer(1)]],
                            device float           * y [[buffer(2)]],
                            constant GemvArgs      & args [[buffer(3)]],
                            uint3  tgpig [[threadgroup_position_in_grid]],
                            ushort tiisg [[thread_index_in_simdgroup]],
                            ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    q6k_gemv_body_brk<4, 4>(W, x, y, args, tgpig, tiisg, sgitg);
}
