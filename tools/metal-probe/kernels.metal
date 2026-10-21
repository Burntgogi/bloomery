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
