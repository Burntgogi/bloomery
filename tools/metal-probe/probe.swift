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
    usage: probe <info|dispatch|bw|chain|--self-test> [options]
      dispatch --n N[,N...] --mode plain|concurrent|icb|cbper:K --reps R
      bw --bytes B --tg T[,T...] --reps R
      chain --bytes B --n N --mode plain|icb --reps R
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
        record(["selftest", "subcommands=dispatch,bw", "status=ok"])
        print("self-test: ok")
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
default:
    usage()
}
