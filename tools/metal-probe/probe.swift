// tools/metal-probe/probe.swift
//
// What it measures
//   The two terms a decode chain on Apple Silicon needs and no derivation closes:
//
//   c_d   the GPU time one *dependent* dispatch adds inside a token's chain, by
//         encoding mode: plain (one serial compute encoder with a buffer barrier
//         between dispatches), concurrent (a concurrent encoder with the same
//         barriers), icb (an MTLIndirectCommandBuffer with one compute command per
//         dispatch and a barrier on every command), and cbper:K (a new command
//         buffer every K dispatches).
//   f_bw  the fraction of the datasheet bandwidth a streaming read kernel reaches,
//         alone (bw) and inside an N-dispatch chain that reads B bytes in total
//         (chain), which is the decode token's byte skeleton.
//
// How to run
//   bash tools/metal-probe/run.sh          build (offline metallib when the Metal
//                                          toolchain component is present, else
//                                          run-time compilation of kernels.metal),
//                                          self-test, then the standard sitting
//   build/probe info
//   build/probe dispatch --n 1,16,64,256,580 --reps 30 --mode plain
//   build/probe bw --bytes 2147483648 --tg 256,512,1024 --reps 20
//   build/probe chain --bytes 1800000000 --n 580 --mode plain --reps 20
//   build/probe gemv --type q4k|q6k --rows R --cols C --variant ggml|own|brk|ownbrk --reps N [--trials T]
//   build/probe gemvchain --type q4k|q6k --rows R --cols C --n N --variant ggml|own|brk|ownbrk
//                         [--concurrent K] --reps R
//   build/probe --self-test
//   Set PROBE_KERNELS (or --kernels PATH) to a .metallib or a .metal source.
//
// Records (tab-separated, the kind in the first field)
//   info      chip, gpu_cores, os, recommended_max_working_set, max_buffer_length,
//             thermal, low_power, power; role=start|end|info marks the witness
//             block printed before and after every measuring subcommand
//   dispatch  one N of a dispatch run: gpu_us_min/median and host_us_median
//   fit       least-squares slope (c_d, µs per dispatch), intercept and r2 over
//             the N list, from each N's best GPU time
//   bw        one threadgroup size: GB/s best/median/p10/p90 and the checksum state
//   chain     one chain run: per-token ms, achieved and standalone GB/s, the byte
//             floor B / bw_best, their ratio, and the fixed cost per dispatch
//   gemv      one quantized mat-vec arm (M2.2, issue #593). gbps_best/median are the
//             headline: T command buffers of N dispatches each, one buffer barrier
//             between dispatches, every dispatch another copy of the weights, so each
//             dispatch is a DRAM read (working_set_bytes >= 1 GiB) and a command
//             buffer's start/commit is not in the number. perrep_* is the same
//             mat-vec with one command buffer per dispatch, kept beside it only: a
//             submit gap between ~60 us kernels reads far below the stream rate.
//             bw_best_gbps is the same run's stream_read over the same byte ranges, in
//             the same one-buffer structure (best of tg 256, 1024); f_q = gbps_best /
//             bw_best_gbps; rel_err is max|y - y_ref| / max|y_ref| against a CPU
//             reference in f64 and gates the run (exit 1 above 1e-3). Every
//             dispatch's y is checked bit for bit against the first one's.
//   gemvchain N dependent mat-vecs over N weight slices of one buffer, a buffer
//             barrier after every K (--concurrent) mat-vecs, all in one command
//             buffer (K > 1 uses a concurrent encoder): chain_ms_*, the byte floor at
//             the same run's best standalone read of the whole buffer, their ratio.
//   gemvcheck a correctness case: rel_err of one mat-vec against the CPU reference
//   selftest  the self-test's verdict
//
// Numbers from this tool are development evidence, not a timing-runner number:
// no lease, no witness beyond the info block.

import Foundation
import Metal

let GMAX = 8192
let EMPTY_THREADS = 32

func now() -> Double { ProcessInfo.processInfo.systemUptime }

func fmt(_ x: Double, _ digits: Int = 3) -> String {
    String(format: "%.\(digits)f", locale: Locale(identifier: "en_US_POSIX"), arguments: [x])
}

func fail(_ name: String, _ detail: String) -> Never {
    FileHandle.standardError.write("probe: \(name): \(detail)\n".data(using: .utf8)!)
    exit(1)
}

func usage() -> Never {
    FileHandle.standardError.write("""
    usage: probe <info|dispatch|bw|chain|gemv|gemvchain|--self-test> [options]
      dispatch --n N[,N...] --mode plain|concurrent|icb|cbper:K --reps R
      bw --bytes B --tg T[,T...] --reps R
      chain --bytes B --n N --mode plain|icb --reps R
      gemv --type q4k|q6k --rows R --cols C --variant ggml|own|brk|ownbrk --reps N [--trials T]
      gemvchain --type q4k|q6k --rows R --cols C --n N --variant ggml|own|brk|ownbrk [--concurrent K] --reps R
      --kernels PATH   .metallib or .metal source (default $PROBE_KERNELS)
    """.data(using: .utf8)!)
    exit(64)
}

func record(_ fields: [String]) {
    print(fields.joined(separator: "\t"))
}

func thermalName() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}

func refuseIfHot() {
    let state = ProcessInfo.processInfo.thermalState
    if state == .serious || state == .critical {
        FileHandle.standardError.write(
            "probe: refused: thermal state \(thermalName()) (exit 75)\n".data(using: .utf8)!)
        exit(75)
    }
}

func powerSource() -> String {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: "/usr/bin/pmset")
    p.arguments = ["-g", "batt"]
    let pipe = Pipe()
    p.standardOutput = pipe
    p.standardError = pipe
    do { try p.run() } catch { return "[unavailable]" }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    p.waitUntilExit()
    let text = String(decoding: data, as: UTF8.self)
        .replacingOccurrences(of: "\t", with: "; ")
        .replacingOccurrences(of: "\n", with: "; ")
        .trimmingCharacters(in: CharacterSet(charactersIn: "; "))
    return text.isEmpty ? "[unavailable]" : text
}

func cpuXor(_ buffer: MTLBuffer, words: Int) -> UInt32 {
    let p = buffer.contents().bindMemory(to: UInt32.self, capacity: words)
    var x: UInt32 = 0
    for i in 0..<words { x ^= p[i] }
    return x
}

func makeFilledBuffer(_ device: MTLDevice, bytes: Int) -> MTLBuffer {
    guard let buf = device.makeBuffer(length: bytes, options: .storageModeShared) else {
        fail("buffer", "cannot allocate \(bytes) bytes")
    }
    arc4random_buf(buf.contents(), bytes)
    return buf
}

func percentile(_ xs: [Double], _ p: Double) -> Double {
    let s = xs.sorted()
    if s.isEmpty { return 0 }
    let i = Int((p * Double(s.count - 1)).rounded())
    return s[min(max(i, 0), s.count - 1)]
}

func fit(_ points: [(x: Double, y: Double)]) -> (slope: Double, intercept: Double, r2: Double) {
    let n = Double(points.count)
    guard n >= 2 else { return (0, 0, 0) }
    let mx = points.map(\.x).reduce(0, +) / n
    let my = points.map(\.y).reduce(0, +) / n
    var sxx = 0.0, sxy = 0.0, syy = 0.0
    for p in points {
        sxx += (p.x - mx) * (p.x - mx)
        sxy += (p.x - mx) * (p.y - my)
        syy += (p.y - my) * (p.y - my)
    }
    let slope = sxx > 0 ? sxy / sxx : 0
    let intercept = my - slope * mx
    let r2 = (sxx > 0 && syy > 0) ? (sxy * sxy) / (sxx * syy) : 0
    return (slope, intercept, r2)
}

enum Mode: Equatable {
    case plain
    case concurrent
    case icb
    case cbper(Int)

    var name: String {
        switch self {
        case .plain: return "plain"
        case .concurrent: return "concurrent"
        case .icb: return "icb"
        case .cbper(let k): return "cbper:\(k)"
        }
    }

    static func parse(_ s: String) -> Mode? {
        switch s {
        case "plain": return .plain
        case "concurrent": return .concurrent
        case "icb": return .icb
        default:
            if s.hasPrefix("cbper:"), let k = Int(s.dropFirst("cbper:".count)), k >= 1 {
                return .cbper(k)
            }
            return nil
        }
    }
}

struct RepTime {
    let gpu: Double
    let host: Double
}

final class Probe {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary
    let emptyPSO: MTLComputePipelineState
    let streamPSO: MTLComputePipelineState
    let chainPSO: MTLComputePipelineState

    init(kernels: String) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fail("device", "no Metal device")
        }
        guard let queue = device.makeCommandQueue() else {
            fail("queue", "cannot make a command queue")
        }
        let library: MTLLibrary
        do {
            if kernels.hasSuffix(".metallib") {
                library = try device.makeLibrary(URL: URL(fileURLWithPath: kernels))
            } else {
                let source = try String(contentsOfFile: kernels, encoding: .utf8)
                library = try device.makeLibrary(source: source, options: nil)
            }
        } catch {
            fail("kernels", "\(kernels): \(error)")
        }
        func makePSO(_ name: String, indirect: Bool) -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                fail("kernels", "no kernel named \(name)")
            }
            do {
                if indirect {
                    let desc = MTLComputePipelineDescriptor()
                    desc.computeFunction = function
                    desc.supportIndirectCommandBuffers = true
                    return try device.makeComputePipelineState(descriptor: desc, options: [], reflection: nil)
                }
                return try device.makeComputePipelineState(function: function)
            } catch {
                fail("pipeline", "\(name): \(error)")
            }
        }
        emptyPSO = makePSO("empty", indirect: true)
        streamPSO = makePSO("stream_read", indirect: false)
        chainPSO = makePSO("chain_slice", indirect: true)
        if !emptyPSO.supportIndirectCommandBuffers || !chainPSO.supportIndirectCommandBuffers {
            fail("icb-unsupported", "the device reports supportIndirectCommandBuffers = false for the empty/chain pipelines")
        }
        self.device = device
        self.queue = queue
        self.library = library
    }

    func emitInfo(role: String) {
        record(["info",
                "role=\(role)",
                "chip=\(device.name)",
                "gpu_cores=[unknown]",
                "os=\(ProcessInfo.processInfo.operatingSystemVersionString)",
                "recommended_max_working_set=\(device.recommendedMaxWorkingSetSize)",
                "max_buffer_length=\(device.maxBufferLength)",
                "thermal=\(thermalName())",
                "low_power=\(ProcessInfo.processInfo.isLowPowerModeEnabled)",
                "power=\(powerSource())"])
    }

    func submit(_ cb: MTLCommandBuffer) -> RepTime {
        let t0 = now()
        cb.commit()
        cb.waitUntilCompleted()
        let t1 = now()
        if cb.status == .error {
            fail("command-buffer", cb.error.map { "\($0)" } ?? "unknown error")
        }
        let gpu = cb.gpuEndTime - cb.gpuStartTime
        if !(gpu > 0) {
            fail("gpu-time", "the command buffer reports no GPU start/end time")
        }
        return RepTime(gpu: gpu, host: t1 - t0)
    }

    func emptyCommandBuffer(n: Int, dispatchType: MTLDispatchType, counter: MTLBuffer) -> MTLCommandBuffer {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder(dispatchType: dispatchType)!
        enc.setComputePipelineState(emptyPSO)
        enc.setBuffer(counter, offset: 0, index: 0)
        let groups = MTLSize(width: 1, height: 1, depth: 1)
        let threads = MTLSize(width: EMPTY_THREADS, height: 1, depth: 1)
        for _ in 0..<n {
            enc.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
            enc.memoryBarrier(scope: .buffers)
        }
        enc.endEncoding()
        return cb
    }

    func makeEmptyICB(n: Int) -> MTLIndirectCommandBuffer {
        let desc = MTLIndirectCommandBufferDescriptor()
        desc.commandTypes = .concurrentDispatch
        desc.inheritPipelineState = true
        desc.inheritBuffers = true
        desc.maxKernelBufferBindCount = 0
        guard let icb = device.makeIndirectCommandBuffer(descriptor: desc,
                                                         maxCommandCount: n,
                                                         options: []) else {
            fail("icb", "cannot allocate an indirect command buffer of \(n) commands")
        }
        let groups = MTLSize(width: 1, height: 1, depth: 1)
        let threads = MTLSize(width: EMPTY_THREADS, height: 1, depth: 1)
        for i in 0..<n {
            let cmd = icb.indirectComputeCommandAt(i)
            cmd.concurrentDispatchThreadgroups(groups, threadsPerThreadgroup: threads)
            cmd.setBarrier()
        }
        return icb
    }

    func runEmptyICB(_ icb: MTLIndirectCommandBuffer, n: Int, counter: MTLBuffer) -> RepTime {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder(dispatchType: .concurrent)!
        enc.setComputePipelineState(emptyPSO)
        enc.setBuffer(counter, offset: 0, index: 0)
        enc.useResource(counter, usage: .write)
        enc.executeCommandsInBuffer(icb, range: 0..<n)
        enc.endEncoding()
        return submit(cb)
    }

    func counterValue(_ buffer: MTLBuffer) -> UInt32 {
        buffer.contents().load(fromByteOffset: 0, as: UInt32.self)
    }

    func dispatchRun(ns: [Int], mode: Mode, reps: Int) {
        var points: [(x: Double, y: Double)] = []
        for n in ns {
            guard let counter = device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                                 options: .storageModeShared) else {
                fail("counter", "cannot allocate the counter")
            }
            counter.contents().storeBytes(of: UInt32(0), toByteOffset: 0, as: UInt32.self)
            var gpuSamples: [Double] = []
            var hostSamples: [Double] = []
            let icb = (mode == .icb) ? makeEmptyICB(n: n) : nil
            for rep in 0..<reps {
                let t: RepTime
                switch mode {
                case .plain:
                    t = submit(emptyCommandBuffer(n: n, dispatchType: .serial, counter: counter))
                case .concurrent:
                    t = submit(emptyCommandBuffer(n: n, dispatchType: .concurrent, counter: counter))
                case .icb:
                    t = runEmptyICB(icb!, n: n, counter: counter)
                case .cbper(let k):
                    var gpu = 0.0, host = 0.0
                    var done = 0
                    while done < n {
                        let take = min(k, n - done)
                        let cb = emptyCommandBuffer(n: take, dispatchType: .serial, counter: counter)
                        let one = submit(cb)
                        gpu += one.gpu
                        host += one.host
                        done += take
                    }
                    t = RepTime(gpu: gpu, host: host)
                }
                let want = UInt32(n * (rep + 1))
                let got = counterValue(counter)
                if got != want {
                    fail("counter-mismatch",
                         "mode=\(mode.name) n=\(n) rep=\(rep + 1) expected=\(want) got=\(got)")
                }
                gpuSamples.append(t.gpu)
                hostSamples.append(t.host)
            }
            let minGpu = gpuSamples.min() ?? 0
            let medGpu = percentile(gpuSamples, 0.5)
            let medHost = percentile(hostSamples, 0.5)
            record(["dispatch",
                    "mode=\(mode.name)",
                    "n=\(n)",
                    "reps=\(reps)",
                    "gpu_us_min=\(fmt(minGpu * 1e6))",
                    "gpu_us_median=\(fmt(medGpu * 1e6))",
                    "host_us_median=\(fmt(medHost * 1e6))"])
            points.append((x: Double(n), y: minGpu * 1e6))
        }
        let f = fit(points)
        record(["fit",
                "mode=\(mode.name)",
                "ns=\(ns.map(String.init).joined(separator: ","))",
                "slope_us=\(fmt(f.slope, 4))",
                "intercept_us=\(fmt(f.intercept, 3))",
                "r2=\(fmt(f.r2, 4))",
                "basis=min_gpu_us"])
    }

    func runBW(bytes: Int, tgs: [Int], reps: Int, verify: Bool) -> Bool {
        guard bytes % 16 == 0 else { fail("bw-bytes", "\(bytes) is not a multiple of 16") }
        let words = bytes / 16
        var wordCount = UInt64(words)
        let buffer = makeFilledBuffer(device, bytes: bytes)
        guard let out = device.makeBuffer(length: GMAX * MemoryLayout<UInt32>.size,
                                          options: .storageModeShared) else {
            fail("bw-out", "cannot allocate the output buffer")
        }
        var ok = true
        let cpu = verify ? cpuXor(buffer, words: bytes / 4) : 0
        for tg in tgs {
            guard tg % 32 == 0 && tg > 0 && tg <= 1024 else {
                fail("bw-tg", "\(tg) is not a multiple of 32 in 1...1024")
            }
            let groups = max(1, min((words + tg - 1) / tg, GMAX))
            var gbps: [Double] = []
            for _ in 0..<reps {
                let cb = queue.makeCommandBuffer()!
                let enc = cb.makeComputeCommandEncoder(dispatchType: .serial)!
                enc.setComputePipelineState(streamPSO)
                enc.setBuffer(buffer, offset: 0, index: 0)
                enc.setBuffer(out, offset: 0, index: 1)
                enc.setBytes(&wordCount, length: MemoryLayout<UInt64>.size, index: 2)
                enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
                enc.endEncoding()
                let t = submit(cb)
                gbps.append(Double(bytes) / t.gpu / 1e9)
            }
            var checksum = "skipped"
            if verify {
                var gpu: UInt32 = 0
                let p = out.contents().bindMemory(to: UInt32.self, capacity: groups)
                for i in 0..<groups { gpu ^= p[i] }
                if gpu == cpu {
                    checksum = "ok"
                } else {
                    checksum = "mismatch(gpu=\(gpu) cpu=\(cpu))"
                    ok = false
                }
            }
            let best = gbps.max() ?? 0
            record(["bw",
                    "bytes=\(bytes)",
                    "tg=\(tg)",
                    "groups=\(groups)",
                    "reps=\(reps)",
                    "gbps_best=\(fmt(best, 2))",
                    "gbps_median=\(fmt(percentile(gbps, 0.5), 2))",
                    "gbps_p10=\(fmt(percentile(gbps, 0.1), 2))",
                    "gbps_p90=\(fmt(percentile(gbps, 0.9), 2))",
                    "checksum=\(checksum)"])
        }
        return ok
    }

    func standaloneBW(buffer: MTLBuffer, bytes: Int, tg: Int, reps: Int) -> Double {
        let words = bytes / 16
        var wordCount = UInt64(words)
        let groups = max(1, min((words + tg - 1) / tg, GMAX))
        guard let out = device.makeBuffer(length: groups * MemoryLayout<UInt32>.size,
                                          options: .storageModeShared) else {
            fail("chain-out", "cannot allocate the streaming output buffer")
        }
        var gbps: [Double] = []
        for _ in 0..<reps {
            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder(dispatchType: .serial)!
            enc.setComputePipelineState(streamPSO)
            enc.setBuffer(buffer, offset: 0, index: 0)
            enc.setBuffer(out, offset: 0, index: 1)
            enc.setBytes(&wordCount, length: MemoryLayout<UInt64>.size, index: 2)
            enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            enc.endEncoding()
            let t = submit(cb)
            gbps.append(Double(bytes) / t.gpu / 1e9)
        }
        return gbps.max() ?? 0
    }

    func chainRun(bytes: Int, n: Int, mode: Mode, reps: Int) {
        guard mode == .plain || mode == .icb else {
            fail("chain-mode", "\(mode.name) is not plain or icb")
        }
        guard bytes % 16 == 0 else { fail("chain-bytes", "\(bytes) is not a multiple of 16") }
        let words = bytes / 16
        guard n >= 1 && n <= words else { fail("chain-n", "\(n) dispatches over \(words) uint4") }
        let sliceWords = (words + n - 1) / n
        let tg = 256
        let groups = max(1, min((sliceWords + tg - 1) / tg, GMAX))

        let buffer = makeFilledBuffer(device, bytes: bytes)
        guard let counts = device.makeBuffer(length: n * MemoryLayout<UInt64>.size,
                                             options: .storageModeShared) else {
            fail("chain-counts", "cannot allocate the slice count buffer")
        }
        let countPtr = counts.contents().bindMemory(to: UInt64.self, capacity: n)
        for i in 0..<n {
            let start = i * sliceWords
            countPtr[i] = UInt64(max(0, min(sliceWords, words - start)))
        }
        guard let out = device.makeBuffer(length: n * groups * MemoryLayout<UInt32>.size,
                                          options: .storageModeShared) else {
            fail("chain-out", "cannot allocate the per-dispatch output buffer")
        }
        guard let counter = device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                              options: .storageModeShared) else {
            fail("chain-counter", "cannot allocate the counter")
        }
        counter.contents().storeBytes(of: UInt32(0), toByteOffset: 0, as: UInt32.self)

        let threads = MTLSize(width: tg, height: 1, depth: 1)
        var icb: MTLIndirectCommandBuffer? = nil
        if mode == .icb {
            let desc = MTLIndirectCommandBufferDescriptor()
            desc.commandTypes = .concurrentDispatch
            desc.inheritPipelineState = true
            desc.inheritBuffers = false
            desc.maxKernelBufferBindCount = 4
            guard let made = device.makeIndirectCommandBuffer(descriptor: desc,
                                                              maxCommandCount: n,
                                                              options: []) else {
                fail("chain-icb", "cannot allocate an indirect command buffer of \(n) commands")
            }
            for i in 0..<n {
                let cmd = made.indirectComputeCommandAt(i)
                cmd.setKernelBuffer(buffer, offset: i * sliceWords * 16, at: 0)
                cmd.setKernelBuffer(out, offset: i * groups * 4, at: 1)
                cmd.setKernelBuffer(counts, offset: i * 8, at: 2)
                cmd.setKernelBuffer(counter, offset: 0, at: 3)
                cmd.concurrentDispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                                   threadsPerThreadgroup: threads)
                cmd.setBarrier()
            }
            icb = made
        }

        var gpuSamples: [Double] = []
        for rep in 0..<reps {
            let t: RepTime
            if mode == .icb {
                let cb = queue.makeCommandBuffer()!
                let enc = cb.makeComputeCommandEncoder(dispatchType: .concurrent)!
                enc.setComputePipelineState(chainPSO)
                // Explicit per-command bindings are invisible to the encoder's
                // hazard tracking; without useResource the driver stages the
                // buffers on every execution (measured ~3 GB/s, ~100x slower).
                enc.useResource(buffer, usage: .read)
                enc.useResource(out, usage: .write)
                enc.useResource(counts, usage: .read)
                enc.useResource(counter, usage: .write)
                enc.executeCommandsInBuffer(icb!, range: 0..<n)
                enc.endEncoding()
                t = submit(cb)
            } else {
                let cb = queue.makeCommandBuffer()!
                let enc = cb.makeComputeCommandEncoder(dispatchType: .serial)!
                enc.setComputePipelineState(chainPSO)
                for i in 0..<n {
                    enc.setBuffer(buffer, offset: i * sliceWords * 16, index: 0)
                    enc.setBuffer(out, offset: i * groups * 4, index: 1)
                    enc.setBuffer(counts, offset: i * 8, index: 2)
                    enc.setBuffer(counter, offset: 0, index: 3)
                    enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                             threadsPerThreadgroup: threads)
                    enc.memoryBarrier(scope: .buffers)
                }
                enc.endEncoding()
                t = submit(cb)
            }
            let want = UInt32(n * (rep + 1))
            let got = counterValue(counter)
            if got != want {
                fail("counter-mismatch",
                     "chain mode=\(mode.name) rep=\(rep + 1) expected=\(want) got=\(got)")
            }
            gpuSamples.append(t.gpu)
        }

        let bestGB = standaloneBW(buffer: buffer, bytes: bytes, tg: tg, reps: 5)
        let minSec = gpuSamples.min() ?? 0
        let medSec = percentile(gpuSamples, 0.5)
        let floorSec = Double(bytes) / (bestGB * 1e9)
        let ratio = floorSec > 0 ? minSec / floorSec : 0
        let fixedUs = (minSec - floorSec) * 1e6 / Double(n)
        record(["chain",
                "bytes=\(bytes)",
                "n=\(n)",
                "mode=\(mode.name)",
                "reps=\(reps)",
                "token_ms_min=\(fmt(minSec * 1e3))",
                "token_ms_median=\(fmt(medSec * 1e3))",
                "bw_gbps=\(fmt(Double(bytes) / minSec / 1e9, 2))",
                "bw_best_gbps=\(fmt(bestGB, 2))",
                "byte_floor_ms=\(fmt(floorSec * 1e3))",
                "ratio=\(fmt(ratio, 3))",
                "fixed_us_per_dispatch=\(fmt(fixedUs, 3))"])
    }

    func selfTest() {
        let modes: [Mode] = [.plain, .concurrent, .icb, .cbper(1), .cbper(50)]
        for mode in modes {
            dispatchRun(ns: [1, 7], mode: mode, reps: 3)
        }
        let ok = runBW(bytes: 64 * 1024 * 1024, tgs: [256], reps: 2, verify: true)
        if !ok { fail("self-test", "bw checksum mismatch") }
        gemvSelfTest()
        record(["selftest", "subcommands=dispatch,bw,gemv,gemvchain", "status=ok"])
        print("self-test: ok")
    }
}

// MARK: - M2.2 quantized mat-vec (issue #593)

let QK_K = 256
let GEMV_TOLERANCE = 1e-3
let GEMV_WORKING_SET = 1 << 30
let GEMV_TRIALS = 8
let GEMV_WARMUPS = 2
let GEMV_WINDOW_BYTES = 768 << 20
let GEMV_STREAM_TGS = [256, 1024]

// Mirrors GemvArgs in kernels.metal.
struct GemvArgs {
    var rows: Int32
    var cols: Int32
    var nblocks: Int32
    var pad0: Int32 = 0
}

enum QType: String {
    case q4k
    case q6k

    var blockBytes: Int { self == .q4k ? 144 : 210 }
}

// NR0 rows per SIMD group and NSG SIMD groups per threadgroup, as instantiated in
// kernels.metal: ggml is llama.cpp's 2 x 2, own is 4 x 4 on the same row loop; brk and
// ownbrk are those two on the other row loop (per-row base, exit at the last row).
enum GemvVariant: String {
    case ggml
    case own
    case brk
    case ownbrk

    var nr0: Int { self == .ggml || self == .brk ? 2 : 4 }
    var nsg: Int { nr0 }
    var rowsPerGroup: Int { nr0 * nsg }
}

struct SplitMix64: RandomNumberGenerator {
    var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    // Uniform in [0, 1).
    mutating func unit() -> Double { Double(next() >> 11) / Double(1 << 53) }
}

struct GemvCase {
    let type: QType
    let variant: GemvVariant
    let rows: Int
    let cols: Int
    let nb: Int
    let bytes: Int       // one matrix
    let stride: Int      // between copies: the matrix, then pad rows, a multiple of 256
    let copies: Int
    let weights: MTLBuffer
    let x: MTLBuffer
    let pso: MTLComputePipelineState
    let reference: [Double]

    var args: GemvArgs { GemvArgs(rows: Int32(rows), cols: Int32(cols), nblocks: Int32(nb)) }
    var groups: Int { (rows + variant.rowsPerGroup - 1) / variant.rowsPerGroup }
    var threads: Int { 32 * variant.nsg }
    var workingSet: Int { copies * stride }
}

func gemvWeights(type: QType, rows: Int, nb: Int, seed: UInt64) -> [UInt8] {
    var rng = SplitMix64(state: seed)
    let bb = type.blockBytes
    var bytes = [UInt8](repeating: 0, count: rows * nb * bb)
    bytes.withUnsafeMutableBytes { raw in
        var i = 0
        while i + 8 <= raw.count {
            raw.storeBytes(of: rng.next(), toByteOffset: i, as: UInt64.self)
            i += 8
        }
        while i < raw.count {
            raw[i] = UInt8(truncatingIfNeeded: rng.next())
            i += 1
        }
        // d (and dmin) as finite f16 in [0.0005, 0.004]
        func small() -> UInt16 { Float16(Float(0.0005 + 0.0035 * rng.unit())).bitPattern }
        for b in 0..<(rows * nb) {
            switch type {
            case .q4k:
                raw.storeBytes(of: small(), toByteOffset: b * bb, as: UInt16.self)
                raw.storeBytes(of: small(), toByteOffset: b * bb + 2, as: UInt16.self)
            case .q6k:
                raw.storeBytes(of: small(), toByteOffset: b * bb + 208, as: UInt16.self)
            }
        }
    }
    return bytes
}

func f16Value(_ w: [UInt8], _ at: Int) -> Double {
    Double(Float(Float16(bitPattern: UInt16(w[at]) | (UInt16(w[at + 1]) << 8))))
}

// get_scale_min_k4 from ggml-quants.c
func scaleMinK4(_ j: Int, _ q: [UInt8], _ at: Int) -> (Double, Double) {
    if j < 4 {
        return (Double(q[at + j] & 63), Double(q[at + j + 4] & 63))
    }
    let d = (q[at + j + 4] & 0xF) | ((q[at + j - 4] >> 6) << 4)
    let m = (q[at + j + 4] >> 4) | ((q[at + j] >> 6) << 4)
    return (Double(d), Double(m))
}

// dequantize_row_q4_K, f64, folded into the dot with x.
func dotQ4K(_ w: [UInt8], _ base: Int, _ x: [Float], _ xo: Int) -> Double {
    let d = f16Value(w, base)
    let dmin = f16Value(w, base + 2)
    var acc = 0.0
    var xi = xo
    var qo = base + 16
    var isc = 0
    for _ in stride(from: 0, to: QK_K, by: 64) {
        let (sc1, mn1) = scaleMinK4(isc, w, base + 4)
        let (sc2, mn2) = scaleMinK4(isc + 1, w, base + 4)
        let d1 = d * sc1, m1 = dmin * mn1
        let d2 = d * sc2, m2 = dmin * mn2
        for l in 0..<32 { acc += (d1 * Double(w[qo + l] & 0xF) - m1) * Double(x[xi + l]) }
        for l in 0..<32 { acc += (d2 * Double(w[qo + l] >> 4) - m2) * Double(x[xi + 32 + l]) }
        xi += 64
        qo += 32
        isc += 2
    }
    return acc
}

// dequantize_row_q6_K, f64, folded into the dot with x.
func dotQ6K(_ w: [UInt8], _ base: Int, _ x: [Float], _ xo: Int) -> Double {
    let d = f16Value(w, base + 208)
    var acc = 0.0
    var xi = xo
    var ql = base
    var qh = base + 128
    var sc = base + 192
    for _ in stride(from: 0, to: QK_K, by: 128) {
        for l in 0..<32 {
            let isx = l / 16
            let q1 = Int((w[ql + l] & 0xF) | (((w[qh + l] >> 0) & 3) << 4)) - 32
            let q2 = Int((w[ql + l + 32] & 0xF) | (((w[qh + l] >> 2) & 3) << 4)) - 32
            let q3 = Int((w[ql + l] >> 4) | (((w[qh + l] >> 4) & 3) << 4)) - 32
            let q4 = Int((w[ql + l + 32] >> 4) | (((w[qh + l] >> 6) & 3) << 4)) - 32
            func s(_ k: Int) -> Double { Double(Int8(bitPattern: w[sc + isx + k])) }
            acc += d * s(0) * Double(q1) * Double(x[xi + l])
            acc += d * s(2) * Double(q2) * Double(x[xi + l + 32])
            acc += d * s(4) * Double(q3) * Double(x[xi + l + 64])
            acc += d * s(6) * Double(q4) * Double(x[xi + l + 96])
        }
        xi += 128
        ql += 64
        qh += 32
        sc += 8
    }
    return acc
}

func referenceGemv(type: QType, w: [UInt8], x: [Float], rows: Int, nb: Int) -> [Double] {
    var y = [Double](repeating: 0, count: rows)
    for r in 0..<rows {
        var acc = 0.0
        for ib in 0..<nb {
            let base = (r * nb + ib) * type.blockBytes
            acc += type == .q4k ? dotQ4K(w, base, x, ib * QK_K) : dotQ6K(w, base, x, ib * QK_K)
        }
        y[r] = acc
    }
    return y
}

// max|y - ref| / max|ref|; a non-finite y is an infinite error.
func relativeError(_ y: [Float], _ ref: [Double]) -> Double {
    var worst = 0.0
    var scale = 0.0
    for (a, b) in zip(y, ref) {
        if !a.isFinite { return .infinity }
        worst = max(worst, abs(Double(a) - b))
        scale = max(scale, abs(b))
    }
    return scale > 0 ? worst / scale : worst
}

extension Probe {
    func makeGemvPSO(_ type: QType, _ variant: GemvVariant) -> MTLComputePipelineState {
        let name = "\(type.rawValue)_gemv_\(variant.rawValue)"
        guard let function = library.makeFunction(name: name) else {
            fail("kernels", "no kernel named \(name)")
        }
        do {
            return try device.makeComputePipelineState(function: function)
        } catch {
            fail("pipeline", "\(name): \(error)")
        }
    }

    // Random blocks from a fixed seed (variant-independent, so both variants read the
    // same bytes), replicated into `copies` places so that a run of dispatches never
    // reads one place twice inside the cache's reach.
    func makeGemvCase(type: QType, rows: Int, cols: Int, variant: GemvVariant,
                      minWorkingSet: Int, copies fixedCopies: Int? = nil) -> GemvCase {
        guard rows > 0, cols > 0, cols % QK_K == 0 else {
            fail("gemv-shape", "\(rows)x\(cols): cols must be a positive multiple of \(QK_K)")
        }
        let nb = cols / QK_K
        let bytes = rows * nb * type.blockBytes
        // The kernels have no per-row bound check: the last threadgroup reads up to a
        // whole group of rows past `rows`, which the copy's tail holds as zeros.
        let padRows = (variant.rowsPerGroup - rows % variant.rowsPerGroup) % variant.rowsPerGroup
        let stride = (bytes + padRows * nb * type.blockBytes + 255) / 256 * 256
        let copies = fixedCopies ?? max(1, (minWorkingSet + stride - 1) / stride)
        guard let weights = device.makeBuffer(length: copies * stride, options: .storageModeShared) else {
            fail("gemv-buffer", "cannot allocate \(copies * stride) bytes of weights")
        }
        guard let xbuf = device.makeBuffer(length: cols * 4, options: .storageModeShared) else {
            fail("gemv-buffer", "cannot allocate the activation vector")
        }
        let seed = 0x6B70_726F_6265 ^ (UInt64(rows) << 24) ^ (UInt64(cols) << 4) ^ (type == .q6k ? 1 : 0)
        let w = gemvWeights(type: type, rows: rows, nb: nb, seed: seed)
        var rng = SplitMix64(state: seed ^ 0xA5A5_A5A5)
        let x = (0..<cols).map { _ in Float(2 * rng.unit() - 1) }
        let base = weights.contents()
        w.withUnsafeBytes { base.copyMemory(from: $0.baseAddress!, byteCount: bytes) }
        for c in 1..<max(copies, 1) {
            (base + c * stride).copyMemory(from: base, byteCount: bytes)
        }
        x.withUnsafeBytes { xbuf.contents().copyMemory(from: $0.baseAddress!, byteCount: cols * 4) }
        return GemvCase(type: type, variant: variant, rows: rows, cols: cols, nb: nb,
                        bytes: bytes, stride: stride, copies: copies,
                        weights: weights, x: xbuf, pso: makeGemvPSO(type, variant),
                        reference: referenceGemv(type: type, w: w, x: x, rows: rows, nb: nb))
    }

    func makeYBuffer(_ floats: Int) -> MTLBuffer {
        guard let y = device.makeBuffer(length: floats * 4, options: .storageModeShared) else {
            fail("gemv-buffer", "cannot allocate \(floats * 4) bytes of output")
        }
        return y
    }

    // `count` mat-vecs in one command buffer. Mat-vec j reads copy (first + j) % copies
    // and writes y slice slice + j; a buffer barrier follows every `group` mat-vecs
    // (group > 1 runs on a concurrent encoder, so the mat-vecs of one group may overlap).
    func gemvBuffer(_ c: GemvCase, y: MTLBuffer, count: Int, first: Int, slice: Int = 0,
                    group: Int) -> MTLCommandBuffer {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder(dispatchType: group > 1 ? .concurrent : .serial)!
        enc.setComputePipelineState(c.pso)
        enc.setBuffer(c.x, offset: 0, index: 1)
        var args = c.args
        for j in 0..<count {
            enc.setBuffer(c.weights, offset: ((first + j) % c.copies) * c.stride, index: 0)
            enc.setBuffer(y, offset: (slice + j) * c.rows * 4, index: 2)
            enc.setBytes(&args, length: MemoryLayout<GemvArgs>.stride, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: c.groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: c.threads, height: 1, depth: 1))
            if (j + 1) % group == 0 && j + 1 < count { enc.memoryBarrier(scope: .buffers) }
        }
        enc.endEncoding()
        return cb
    }

    // The same byte ranges read by stream_read, one buffer barrier between dispatches.
    func streamBuffer(_ c: GemvCase, out: MTLBuffer, tg: Int, count: Int, first: Int) -> MTLCommandBuffer {
        let words = c.bytes / 16
        var wordCount = UInt64(words)
        let groups = max(1, min((words + tg - 1) / tg, GMAX))
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder(dispatchType: .serial)!
        enc.setComputePipelineState(streamPSO)
        enc.setBuffer(out, offset: 0, index: 1)
        for j in 0..<count {
            enc.setBuffer(c.weights, offset: ((first + j) % c.copies) * c.stride, index: 0)
            enc.setBytes(&wordCount, length: MemoryLayout<UInt64>.size, index: 2)
            enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            if j + 1 < count { enc.memoryBarrier(scope: .buffers) }
        }
        enc.endEncoding()
        return cb
    }

    // Commits every buffer at once and returns each one's GPU seconds. A buffer that
    // starts after an idle gap runs at the gap's clocks, and a 30-dispatch buffer is
    // shorter than the ramp, so a timed buffer follows another that is already running.
    func submitBackToBack(_ cbs: [MTLCommandBuffer]) -> [Double] {
        for cb in cbs { cb.commit() }
        return cbs.map { cb in
            cb.waitUntilCompleted()
            if cb.status == .error {
                fail("command-buffer", cb.error.map { "\($0)" } ?? "unknown error")
            }
            let gpu = cb.gpuEndTime - cb.gpuStartTime
            if !(gpu > 0) { fail("gpu-time", "the command buffer reports no GPU start/end time") }
            return gpu
        }
    }

    func poison(_ buffer: MTLBuffer) {
        memset(buffer.contents(), 0xFF, buffer.length)
    }

    // Every y slice must be bit-identical to `golden`: the weights are copies of one
    // matrix, and a slice a dispatch never wrote still holds the NaN poison.
    func verifySlices(_ y: MTLBuffer, count: Int, golden: [Float], what: String) {
        let rows = golden.count
        let p = y.contents().bindMemory(to: Float.self, capacity: count * rows)
        for j in 0..<count {
            for r in 0..<rows where p[j * rows + r].bitPattern != golden[r].bitPattern {
                fail("gemv-dispatch-mismatch", "\(what): slice \(j) row \(r) is \(p[j * rows + r]), the checked dispatch gave \(golden[r])")
            }
        }
    }

    // One dispatch on copy 0, checked against the CPU reference. Prints the gemvcheck
    // record and exits 1 above the tolerance.
    func gemvCheck(_ c: GemvCase) -> (relErr: Double, golden: [Float]) {
        let y = makeYBuffer(c.rows)
        poison(y)
        _ = submit(gemvBuffer(c, y: y, count: 1, first: 0, group: 1))
        let p = y.contents().bindMemory(to: Float.self, capacity: c.rows)
        let golden = (0..<c.rows).map { p[$0] }
        let err = relativeError(golden, c.reference)
        record(["gemvcheck",
                "type=\(c.type.rawValue)",
                "variant=\(c.variant.rawValue)",
                "rows=\(c.rows)",
                "cols=\(c.cols)",
                "rel_err=\(fmt(err, 8))"])
        if !(err <= GEMV_TOLERANCE) {
            fail("gemv-\(c.type.rawValue)-\(c.variant.rawValue)",
                 "relative error \(fmt(err, 6)) > 1e-3 at \(c.rows)x\(c.cols)")
        }
        return (err, golden)
    }

    // `reps` is the least number of dispatches in a timed command buffer; a buffer holds
    // at least GEMV_WINDOW_BYTES of weight reads so that it outlasts the clock ramp.
    // GEMV_WARMUPS untimed buffers run first, then `trials` timed ones, all committed back
    // to back; each buffer's mat-vecs read the next copies in turn.
    func gemvRun(type: QType, rows: Int, cols: Int, variant: GemvVariant, reps: Int, trials: Int) {
        let c = makeGemvCase(type: type, rows: rows, cols: cols, variant: variant,
                             minWorkingSet: GEMV_WORKING_SET)
        let (relErr, golden) = gemvCheck(c)
        let dispatches = max(reps, (GEMV_WINDOW_BYTES + c.bytes - 1) / c.bytes)
        let buffers = GEMV_WARMUPS + trials
        let y = makeYBuffer(buffers * dispatches * rows)
        let out = makeYBuffer(GMAX)

        poison(y)
        let gpu = submitBackToBack((0..<buffers).map {
            gemvBuffer(c, y: y, count: dispatches, first: $0 * dispatches, slice: $0 * dispatches, group: 1)
        })
        verifySlices(y, count: buffers * dispatches, golden: golden, what: "gemv in-buffer")
        let inBuffer = gpu.dropFirst(GEMV_WARMUPS).map { $0 / Double(dispatches) }
        var index = buffers * dispatches

        // One command buffer per dispatch, beside it.
        var perRep: [Double] = []
        for _ in 0..<reps {
            poison(y)
            let t = submit(gemvBuffer(c, y: y, count: 1, first: index, group: 1))
            verifySlices(y, count: 1, golden: golden, what: "gemv per-rep")
            index += 1
            perRep.append(t.gpu)
        }
        // The denominator: stream_read over the same ranges in the same structure.
        var bwSec = Double.infinity
        var bwTg = 0
        for tg in GEMV_STREAM_TGS {
            let times = submitBackToBack((0..<buffers).map {
                streamBuffer(c, out: out, tg: tg, count: dispatches, first: index + $0 * dispatches)
            })
            index += buffers * dispatches
            for t in times.dropFirst(GEMV_WARMUPS) where t / Double(dispatches) < bwSec {
                bwSec = t / Double(dispatches)
                bwTg = tg
            }
        }
        let streamBytes = Double(c.bytes / 16 * 16)
        let bwBest = streamBytes / bwSec / 1e9
        let perGB = inBuffer.map { Double(c.bytes) / $0 / 1e9 }
        let repGB = perRep.map { Double(c.bytes) / $0 / 1e9 }
        let best = perGB.max() ?? 0
        record(["gemv",
                "type=\(type.rawValue)",
                "variant=\(variant.rawValue)",
                "rows=\(rows)",
                "cols=\(cols)",
                "reps=\(reps)",
                "dispatches=\(dispatches)",
                "trials=\(trials)",
                "bytes=\(c.bytes)",
                "copies=\(c.copies)",
                "working_set_bytes=\(c.workingSet)",
                "gbps_best=\(fmt(best, 2))",
                "gbps_median=\(fmt(percentile(perGB, 0.5), 2))",
                "us_best=\(fmt((inBuffer.min() ?? 0) * 1e6, 2))",
                "perrep_gbps_best=\(fmt(repGB.max() ?? 0, 2))",
                "perrep_gbps_median=\(fmt(percentile(repGB, 0.5), 2))",
                "bw_best_gbps=\(fmt(bwBest, 2))",
                "bw_us_best=\(fmt(bwSec * 1e6, 2))",
                "bw_tg=\(bwTg)",
                "f_q=\(fmt(best / bwBest, 3))",
                "pso_max_tg=\(c.pso.maxTotalThreadsPerThreadgroup)",
                "rel_err=\(fmt(relErr, 8))"])
    }

    // N dependent mat-vecs over N weight slices in one command buffer, a barrier after
    // every `concurrent` of them. One untimed buffer, then `reps` timed ones, back to back.
    func gemvChainRun(type: QType, rows: Int, cols: Int, n: Int, variant: GemvVariant,
                      concurrent: Int, reps: Int) {
        let c = makeGemvCase(type: type, rows: rows, cols: cols, variant: variant,
                             minWorkingSet: 0, copies: n)
        let (relErr, golden) = gemvCheck(c)
        let buffers = 1 + reps
        let y = makeYBuffer(buffers * n * rows)
        poison(y)
        let times = submitBackToBack((0..<buffers).map {
            gemvBuffer(c, y: y, count: n, first: 0, slice: $0 * n, group: concurrent)
        })
        verifySlices(y, count: buffers * n, golden: golden, what: "gemvchain")
        let gpu = Array(times.dropFirst())
        let totalBytes = n * c.bytes
        var bwBest = 0.0
        for tg in GEMV_STREAM_TGS {
            bwBest = max(bwBest, standaloneBW(buffer: c.weights, bytes: c.workingSet, tg: tg, reps: 5))
        }
        let minSec = gpu.min() ?? 0
        let floorSec = Double(totalBytes) / (bwBest * 1e9)
        record(["gemvchain",
                "type=\(type.rawValue)",
                "variant=\(variant.rawValue)",
                "rows=\(rows)",
                "cols=\(cols)",
                "n=\(n)",
                "concurrent=\(concurrent)",
                "encoder=\(concurrent > 1 ? "concurrent" : "serial")",
                "bytes=\(totalBytes)",
                "reps=\(reps)",
                "chain_ms_min=\(fmt(minSec * 1e3))",
                "chain_ms_median=\(fmt(percentile(gpu, 0.5) * 1e3))",
                "bw_gbps=\(fmt(Double(totalBytes) / minSec / 1e9, 2))",
                "bw_best_gbps=\(fmt(bwBest, 2))",
                "byte_floor_ms=\(fmt(floorSec * 1e3))",
                "ratio=\(fmt(floorSec > 0 ? minSec / floorSec : 0, 3))",
                "us_per_step=\(fmt(minSec * 1e6 / Double(n), 3))",
                "rel_err=\(fmt(relErr, 8))"])
    }

    // Correctness at two small shapes per type and variant (the second leaves a
    // partial threadgroup of rows and a block count that is not a multiple of 4), and a
    // smoke of the timing machinery: grouped dispatches on both encoders and the
    // bit-for-bit slice check.
    func gemvSelfTest() {
        for type in [QType.q4k, .q6k] {
            for variant in [GemvVariant.ggml, .own, .brk, .ownbrk] {
                for (rows, cols) in [(8, 512), (18, 768)] {
                    let c = makeGemvCase(type: type, rows: rows, cols: cols, variant: variant, minWorkingSet: 0)
                    _ = gemvCheck(c)
                }
                let c = makeGemvCase(type: type, rows: 64, cols: 1024, variant: variant, minWorkingSet: 1 << 22)
                let (_, golden) = gemvCheck(c)
                let y = makeYBuffer(6 * c.rows)
                let out = makeYBuffer(GMAX)
                for group in [1, 2] {
                    poison(y)
                    _ = submit(gemvBuffer(c, y: y, count: 6, first: 1, group: group))
                    verifySlices(y, count: 6, golden: golden, what: "gemv self-test group \(group)")
                }
                _ = submit(streamBuffer(c, out: out, tg: 256, count: 6, first: 1))
            }
        }
    }
}

// MARK: - argument parsing

func parseFlags(_ args: [String]) -> [String: String] {
    var flags: [String: String] = [:]
    var i = 0
    while i < args.count {
        let a = args[i]
        guard a.hasPrefix("--"), i + 1 < args.count else { usage() }
        flags[String(a.dropFirst(2))] = args[i + 1]
        i += 2
    }
    return flags
}

func intList(_ s: String) -> [Int]? {
    var out: [Int] = []
    for part in s.split(separator: ",") {
        guard let v = Int(part), v > 0 else { return nil }
        out.append(v)
    }
    return out.isEmpty ? nil : out
}

func kernelsPath() -> String {
    if let env = ProcessInfo.processInfo.environment["PROBE_KERNELS"], !env.isEmpty {
        return env
    }
    let exe = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
    let lib = exe.appendingPathComponent("kernels.metallib").path
    if FileManager.default.fileExists(atPath: lib) { return lib }
    return exe.appendingPathComponent("../kernels.metal").path
}

// MARK: - main

var args = Array(CommandLine.arguments.dropFirst())
let subcommand: String
if args.first == "--self-test" {
    subcommand = "--self-test"
    args.removeFirst()
} else {
    guard let first = args.first else { usage() }
    subcommand = first
    args.removeFirst()
}

var flags = parseFlags(args)
if subcommand != "info" { refuseIfHot() }

let probe = Probe(kernels: flags.removeValue(forKey: "kernels") ?? kernelsPath())

switch subcommand {
case "info":
    probe.emitInfo(role: "info")
case "--self-test":
    probe.emitInfo(role: "start")
    probe.selfTest()
    probe.emitInfo(role: "end")
case "dispatch":
    guard let ns = flags["n"].flatMap(intList), let modeStr = flags["mode"],
          let mode = Mode.parse(modeStr), let reps = flags["reps"].flatMap(Int.init), reps >= 1 else {
        usage()
    }
    probe.emitInfo(role: "start")
    probe.dispatchRun(ns: ns, mode: mode, reps: reps)
    probe.emitInfo(role: "end")
case "bw":
    guard let bytes = flags["bytes"].flatMap(Int.init), bytes > 0,
          let tgs = flags["tg"].flatMap(intList),
          let reps = flags["reps"].flatMap(Int.init), reps >= 1 else {
        usage()
    }
    probe.emitInfo(role: "start")
    _ = probe.runBW(bytes: bytes, tgs: tgs, reps: reps, verify: false)
    probe.emitInfo(role: "end")
case "chain":
    guard let bytes = flags["bytes"].flatMap(Int.init), bytes > 0,
          let n = flags["n"].flatMap(Int.init), n > 0,
          let modeStr = flags["mode"], let mode = Mode.parse(modeStr),
          let reps = flags["reps"].flatMap(Int.init), reps >= 1 else {
        usage()
    }
    probe.emitInfo(role: "start")
    probe.chainRun(bytes: bytes, n: n, mode: mode, reps: reps)
    probe.emitInfo(role: "end")
case "gemv":
    guard let type = flags["type"].flatMap(QType.init(rawValue:)),
          let rows = flags["rows"].flatMap(Int.init), rows > 0,
          let cols = flags["cols"].flatMap(Int.init), cols > 0,
          let variant = flags["variant"].flatMap(GemvVariant.init(rawValue:)),
          let reps = flags["reps"].flatMap(Int.init), reps >= 1 else {
        usage()
    }
    let trials = flags["trials"].flatMap(Int.init) ?? GEMV_TRIALS
    probe.emitInfo(role: "start")
    probe.gemvRun(type: type, rows: rows, cols: cols, variant: variant, reps: reps, trials: max(trials, 1))
    probe.emitInfo(role: "end")
case "gemvchain":
    guard let type = flags["type"].flatMap(QType.init(rawValue:)),
          let rows = flags["rows"].flatMap(Int.init), rows > 0,
          let cols = flags["cols"].flatMap(Int.init), cols > 0,
          let n = flags["n"].flatMap(Int.init), n > 0,
          let variant = flags["variant"].flatMap(GemvVariant.init(rawValue:)),
          let reps = flags["reps"].flatMap(Int.init), reps >= 1 else {
        usage()
    }
    let concurrent = flags["concurrent"].flatMap(Int.init) ?? 1
    guard concurrent >= 1 else { usage() }
    probe.emitInfo(role: "start")
    probe.gemvChainRun(type: type, rows: rows, cols: cols, n: n, variant: variant,
                       concurrent: concurrent, reps: reps)
    probe.emitInfo(role: "end")
default:
    usage()
}
