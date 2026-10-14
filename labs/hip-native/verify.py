"""Build and check the HIP lab; keep evidence tied to a clean Git tree."""

import argparse
import ctypes
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys


LAB = Path(__file__).resolve().parent
ROOT = LAB.parents[1]


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clean():
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("source checkout is dirty; commit the source before verification")


def card():
    hip = ctypes.CDLL(str(Path(os.environ["HIP_LIB_DIR"]) / "libamdhip64.so"))

    def call(name, types, *args):
        fn = getattr(hip, name)
        fn.argtypes, fn.restype = types, ctypes.c_int
        code = fn(*args)
        if code:
            raise RuntimeError(f"{name} returned HIP error {code}")

    call("hipInit", [ctypes.c_uint], 0)
    call("hipSetDevice", [ctypes.c_int], 0)
    name = ctypes.create_string_buffer(256)
    call("hipDeviceGetName", [ctypes.c_char_p, ctypes.c_int, ctypes.c_int], name, len(name), 0)
    result = {"name": name.value.decode(), "device_ordinal": 0}
    for symbol, field in [("hipRuntimeGetVersion", "runtime_version"),
                          ("hipDriverGetVersion", "driver_version")]:
        value = ctypes.c_int()
        call(symbol, [ctypes.POINTER(ctypes.c_int)], ctypes.byref(value))
        result[field] = value.value
    result["visibility"] = {key: os.environ.get(key) for key in
                            ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"]}
    result["runtime_library"] = str((Path(os.environ["HIP_LIB_DIR"]) / "libamdhip64.so").resolve())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new evidence directory outside the Git checkout")
    parser.add_argument("--q4-rows", type=Path, help="optional packed Q4_K row fixture (never uploaded)")
    parser.add_argument("--k", type=int)
    parser.add_argument("--rows", type=int)
    args = parser.parse_args()
    if args.q4_rows is None and (args.k is not None or args.rows is not None):
        parser.error("--k and --rows require --q4-rows")
    if args.q4_rows is not None and (not args.k or not args.rows):
        parser.error("--q4-rows requires positive --k and --rows")
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        parser.error("evidence must be outside the source checkout")
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema": 1, "status": "fail", "checks": [], "cases": [],
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "host": {"platform": platform.platform(), "cpu": platform.processor()}}

    def run(label, command, flags=None):
        env = os.environ.copy()
        env["CARGO_TARGET_DIR"] = str(LAB / "target")
        if flags is not None:
            env["RUSTFLAGS"] = flags
        log = output / f"{label}.log"
        entry = {"name": label, "command": command, "rustflags": env.get("RUSTFLAGS"),
                 "log": log.name, "status": "fail"}
        result["checks"].append(entry)
        with log.open("w") as stream:
            try:
                done = subprocess.run(command, cwd=LAB, env=env, stdout=stream,
                                      stderr=subprocess.STDOUT, timeout=900, check=False)
                entry["exit_code"] = done.returncode
            except subprocess.TimeoutExpired:
                entry["exit_code"] = 124
        entry["log_sha256"] = digest(log)
        if entry["exit_code"]:
            raise RuntimeError(f"{label} failed ({entry['exit_code']}): {log}")
        entry["status"] = "pass"
        print(f"{label}: pass", flush=True)
        return log.read_text()

    try:
        clean()
        head = git("rev-parse", "HEAD")
        result["source"] = {"commit": head, "tree": git("rev-parse", "HEAD^{tree}"),
                            "lab_tree": git("rev-parse", "HEAD:labs/hip-native"), "clean": True}
        if os.environ.get("CARGO_ENCODED_RUSTFLAGS"):
            raise RuntimeError("unset CARGO_ENCODED_RUSTFLAGS; verification sets explicit RUSTFLAGS")
        if not os.environ.get("HIP_LIB_DIR"):
            raise RuntimeError("HIP_LIB_DIR must name the directory containing libamdhip64.so")
        result["toolchain"] = subprocess.check_output(["rustc", "-vV"], text=True)
        result["target"] = {"triple": "amdgcn-amd-amdhsa", "cpu": "gfx1100"}
        host_flags = "-C target-cpu=native"
        device_flags = "-C target-cpu=gfx1100 -C panic=abort"
        run("fmt", ["cargo", "fmt", "--all", "--", "--check"])
        run("host-test", ["cargo", "test", "--locked", "-p", "bloomery-hip-spike"], host_flags)
        run("host-clippy", ["cargo", "clippy", "--locked", "--no-deps", "-p",
                            "bloomery-hip-spike", "--all-targets", "--", "-D", "warnings"], host_flags)
        device = ["cargo", "-Zbuild-std=core"]
        target = ["--release", "--target", "amdgcn-amd-amdhsa", "-p", "bloomery-hip-spike-device"]
        run("device-clippy", device + ["clippy", "--locked", "--no-deps"] + target +
            ["--", "-D", "warnings"], device_flags)
        run("device-build", device + ["build", "--locked"] + target, device_flags)
        run("host-build", ["cargo", "build", "--locked", "--release", "-p", "bloomery-hip-spike"], host_flags)
        binaries = [LAB / "target/release/bloomery-hip-spike",
                    LAB / "target/amdgcn-amd-amdhsa/release/bloomery_hip_spike_device.elf"]
        result["binaries"] = [{"path": str(p.relative_to(LAB)), "sha256": digest(p)} for p in binaries]
        result["card"] = card()
        command = [str(p) for p in binaries]
        if args.q4_rows:
            path = args.q4_rows.resolve()
            result["fixture"] = {"name": path.name, "sha256": digest(path), "k": args.k,
                                 "rows": args.rows, "bytes": path.stat().st_size}
            command += [str(path), str(args.k), str(args.rows)]
        raw = run("gpu", command)
        result["cases"] = [json.loads(line) for line in raw.splitlines() if line.startswith("{")]
        expected = 6 if args.q4_rows else 4
        if len(result["cases"]) != expected:
            raise RuntimeError(f"expected {expected} GPU cases, got {len(result['cases'])}")
        for case in result["cases"]:
            if not math.isfinite(case["nmse"]) or case["nmse"] > 1e-6:
                raise RuntimeError("GPU case failed the fixed NMSE <= 1e-6 band")
            case["status"] = "pass"
        result["acceptance"] = {"nmse_max": 1e-6, "finite_outputs": True}
        clean()
        if git("rev-parse", "HEAD") != head:
            raise RuntimeError("source HEAD moved during verification")
        if any(digest(p) != b["sha256"] for p, b in zip(binaries, result["binaries"])):
            raise RuntimeError("a binary changed during verification")
        result["status"] = "pass"
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        result["error"] = str(error)
        print(f"verify: {error}", file=sys.stderr)
    finally:
        result["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(output / "result.json", flush=True)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
