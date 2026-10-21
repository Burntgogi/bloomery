// tools/metal-probe/icb-barrier-control.swift -- does MTLIndirectCommandEncoder.setBarrier
// serialize the commands of one indirect compute command buffer?
//
// One kernel (an empty dispatch that bumps a counter), one ICB of N commands, executed
// twice: with a barrier on every command and with none. If setBarrier orders each command
// behind the previous one (Apple's documentation), the barriered run is the sum of N
// dependent dispatches while the unbarriered run is one launch of N independent ones.
//
// Build and run:
//   xcrun swiftc -O tools/metal-probe/icb-barrier-control.swift -o /tmp/icb-barrier-control
//   /tmp/icb-barrier-control
//
// Prints one line per (barrier, N): the best GPU time of 20 executions and the per-command
// cost. This is a mechanism check, not a sitting: it carries no thermal witness and no log.

import Foundation
import Metal

let source = """
#include <metal_stdlib>
using namespace metal;
kernel void empty(device atomic_uint *c [[buffer(0)]], uint tid [[thread_position_in_threadgroup]]) {
    if (tid == 0) atomic_fetch_add_explicit(c, 1u, memory_order_relaxed);
}
"""

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(("icb-barrier-control: " + message + "\n").data(using: .utf8)!)
    exit(1)
}

guard let device = MTLCreateSystemDefaultDevice() else { fail("no Metal device") }
guard let queue = device.makeCommandQueue() else { fail("no command queue") }
guard let library = try? device.makeLibrary(source: source, options: nil) else { fail("compiling the kernel failed") }
guard let function = library.makeFunction(name: "empty") else { fail("empty kernel not found") }

let descriptor = MTLComputePipelineDescriptor()
descriptor.computeFunction = function
descriptor.supportIndirectCommandBuffers = true
let pipeline = try! device.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)
guard pipeline.supportIndirectCommandBuffers else { fail("the device refuses indirect command buffers") }

func run(n: Int, barrier: Bool) {
    let icbDescriptor = MTLIndirectCommandBufferDescriptor()
    icbDescriptor.commandTypes = .concurrentDispatch
    icbDescriptor.inheritPipelineState = true
    icbDescriptor.inheritBuffers = true
    guard let icb = device.makeIndirectCommandBuffer(descriptor: icbDescriptor, maxCommandCount: n, options: []) else {
        fail("no indirect command buffer for n=\(n)")
    }
    for i in 0..<n {
        let command = icb.indirectComputeCommandAt(i)
        command.concurrentDispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                               threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        if barrier { command.setBarrier() }
    }

    guard let counter = device.makeBuffer(length: 4, options: .storageModeShared) else { fail("no counter buffer") }
    var best = Double.infinity
    for _ in 0..<20 {
        counter.contents().storeBytes(of: UInt32(0), as: UInt32.self)
        guard let cb = queue.makeCommandBuffer(), let encoder = cb.makeComputeCommandEncoder(dispatchType: .concurrent) else {
            fail("no command buffer")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(counter, offset: 0, index: 0)
        encoder.useResource(counter, usage: .write)
        encoder.executeCommandsInBuffer(icb, range: 0..<n)
        encoder.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { fail("command buffer error: \(error)") }
        let count = counter.contents().load(as: UInt32.self)
        if count != UInt32(n) { fail("counter \(count) != \(n) (barrier=\(barrier))") }
        let gpu = cb.gpuEndTime - cb.gpuStartTime
        if !(gpu > 0) { fail("no GPU time") }
        best = min(best, gpu * 1e6)
    }
    let label = (barrier ? "yes" : "no") as NSString
    print(String(format: "barrier=%@ n=%d gpu_us=%.1f us_per_command=%.2f",
                 label, n, best, best / Double(n)))
}

for n in [1, 64, 580] {
    run(n: n, barrier: true)
    run(n: n, barrier: false)
}
