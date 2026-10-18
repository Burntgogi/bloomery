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
import struct
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


def forward_evidence(directory):
    """Check the actual full-logit files, independently of the writer's success flags."""
    data = json.loads((directory / "forward.json").read_text())
    prompt, greedy = data["prompt_ids"], data["greedy_ids"]
    consumed, predictions = data["consumed_ids"], data["predictions"]
    expected = prompt + greedy[:-1] if greedy else prompt
    rows, vocab = len(consumed), data["model"]["vocab"]
    if not prompt or consumed != expected or len(predictions) != rows or len(data["logit_sha256"]) != rows:
        raise RuntimeError("forward prompt/continuation row contract differs")
    if greedy and predictions[len(prompt) - 1:] != greedy:
        raise RuntimeError("forward greedy ids differ from their GPU predictions")
    if data["model"]["gdn_layers"] < 1 or data["model"]["gqa_layers"] < 1:
        raise RuntimeError("bounded hybrid proof did not exercise both mixer kinds")
    model = directory / "smoke.gguf"
    if digest(model) != data["model"]["files"][0]["sha256"]:
        raise RuntimeError("bounded hybrid model changed after execution")
    for repeat in range(2):
        for row in range(rows):
            path = directory / f"run-{repeat}" / f"logits-{row:04}.f32"
            raw = path.read_bytes()
            if len(raw) != vocab * 4 or hashlib.sha256(raw).hexdigest() != data["logit_sha256"][row]:
                raise RuntimeError("forward logit length, digest, or reset/repeat bits differ")
            values = [value for (value,) in struct.iter_unpack("<f", raw)]
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError("forward observed a nonfinite logit")
            if max(range(vocab), key=values.__getitem__) != predictions[row]:
                raise RuntimeError("forward GPU argmax differs from the complete finite logit row")
    return {"model": data["model"], "rows_per_repeat": rows, "greedy_ids": greedy,
            "all_logit_files_finite": True, "reset_repeat_bitwise_equal": True,
            "independent_argmax_equal": True, "same_file_real_model_llama_HIP": False}


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
    result = {"schema": 3, "status": "fail", "checks": [], "cases": [],
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
        if os.environ.get("CARGO_BUILD_TARGET"):
            raise RuntimeError("unset CARGO_BUILD_TARGET; verification selects the native host target")
        if not os.environ.get("HIP_LIB_DIR"):
            raise RuntimeError("HIP_LIB_DIR must name the directory containing libamdhip64.so")
        result["toolchain"] = subprocess.check_output(["rustc", "-vV"], text=True)
        host = next(line.removeprefix("host: ") for line in result["toolchain"].splitlines() if line.startswith("host: "))
        result["host_target"] = host
        result["target"] = {"triple": "amdgcn-amd-amdhsa", "cpu": "gfx1100", "wave_size": 32}
        result["upstream_sources"] = [{"path": name, "sha256": digest(ROOT / name)} for name in [
            "crates/gpu/src/cores.rs", "crates/gpu/src/q4k_sel.rs", "crates/qdot/src/lib.rs"]]
        host_flags = "-C target-cpu=native"
        device_flags = "-C target-cpu=gfx1100 -C panic=abort"
        run("fmt", ["cargo", "fmt", "--all", "--", "--check"])
        run("host-test", ["cargo", "test", "--locked", "--target", host, "-p", "bloomery-hip-spike"], host_flags)
        run("host-clippy", ["cargo", "clippy", "--locked", "--no-deps", "-p",
                            "bloomery-hip-spike", "--target", host, "--all-targets", "--", "-D", "warnings"], host_flags)
        device = ["cargo"]
        target = ["--release", "--target", "amdgcn-amd-amdhsa", "-p", "bloomery-hip-spike-device"]
        run("device-clippy", device + ["clippy", "-Zbuild-std=core", "--locked", "--no-deps"] + target +
            ["--", "-D", "warnings"], device_flags)
        build_output = run("device-build", device + ["build", "-Zbuild-std=core", "--locked", "--message-format=json"] + target, device_flags)
        run("host-build", ["cargo", "build", "--locked", "--target", host, "--release", "-p", "bloomery-hip-spike"], host_flags)
        binaries = [LAB / f"target/{host}/release/bloomery-hip-spike",
                    LAB / "target/amdgcn-amd-amdhsa/release/bloomery_hip_spike_device.elf"]
        result["binaries"] = [{"path": str(p.relative_to(LAB)), "sha256": digest(p)} for p in binaries]
        messages = [json.loads(line) for line in build_output.splitlines() if line.startswith("{")]
        builds = [entry for entry in messages if entry.get("reason") == "build-script-executed"
                  and "bloomery-hip-spike-device" in entry.get("package_id", "")]
        if len(builds) != 1:
            raise RuntimeError("expected one current device build-script-executed record")
        generated = Path(builds[0]["out_dir"]) / "q4k_core.rs"
        if not generated.resolve().is_relative_to(LAB / "target"):
            raise RuntimeError("generated Q4_K core is outside the lab target directory")
        result["generated_core"] = {"path": str(generated.relative_to(LAB)), "sha256": digest(generated)}
        (output / "q4k_core.rs").write_bytes(generated.read_bytes())
        result["card"] = card()
        command = [str(p) for p in binaries]
        if args.q4_rows:
            path = args.q4_rows.resolve()
            result["fixture"] = {"name": path.name, "sha256": digest(path), "k": args.k,
                                 "rows": args.rows, "bytes": path.stat().st_size}
            command += [str(path), str(args.k), str(args.rows)]
        raw = run("gpu", command)
        result["cases"] = [json.loads(line) for line in raw.splitlines() if line.startswith("{")]
        numeric = [case for case in result["cases"] if case["operation"] in ("q4_matmul", "dense_ffn")]
        selected = [case for case in result["cases"] if case["operation"] == "q4_selected"]
        expected = 6 if args.q4_rows else 4
        if len(numeric) != expected or len(selected) != 11 + bool(args.q4_rows) or len(numeric) + len(selected) != len(result["cases"]):
            raise RuntimeError("unexpected GPU contract case inventory")
        for case in numeric:
            if not math.isfinite(case["nmse"]) or case["nmse"] > 1e-6:
                raise RuntimeError("GPU case failed the fixed NMSE <= 1e-6 band")
            case["status"] = "pass"
        if sorted(case["k"] for case in selected if not case["real_weights"]) != [256, 512, 768, 1024, 1280, 1792, 2048, 2304, 2816, 6144, 20480]:
            raise RuntimeError("missing selected Q4_K tail or paired-prefix case")
        real_cases = [case for case in selected if case["real_weights"]]
        if args.q4_rows and (len(real_cases) != 1 or real_cases[0]["k"] != args.k or real_cases[0]["rows_per_expert"] * 4 != args.rows):
            raise RuntimeError("selected real Q4_K case differs from the recorded fixture")
        for case in selected:
            if case.get("contract") != "selected_bitwise" or case.get("wave_size") != 32 or case.get("launches") != 9:
                raise RuntimeError("selected Q4_K launch contract differs")
            if any(case.get(key) is not True for key in ["bitwise_equal", "finite_expected_outputs", "nan_propagation", "fault_and_sentinel"]):
                raise RuntimeError("selected Q4_K bitwise/finite/fault contract failed")
            if case.get("steady_host_buffer_allocations") != 0:
                raise RuntimeError("selected Q4_K steady launch allocated")
            case["status"] = "pass"
        result["acceptance"] = {"nmse_max": 1e-6, "finite_outputs": True,
                                "selected_q4": "bitwise finite, NaN propagation, exact sticky fault and sentinel"}
        run("forward-gpu", [str(binaries[0]), "--forward-probe", str(binaries[1]),
                            str(output / "forward")])
        result["forward"] = forward_evidence(output / "forward")
        result["acceptance"]["bounded_hybrid_forward"] = "full-logit finite/argmax/reset checked"
        result["acceptance"]["same_file_real_model_llama_HIP"] = False
        clean()
        if git("rev-parse", "HEAD") != head:
            raise RuntimeError("source HEAD moved during verification")
        if any(digest(p) != b["sha256"] for p, b in zip(binaries, result["binaries"])):
            raise RuntimeError("a binary changed during verification")
        if digest(generated) != result["generated_core"]["sha256"]:
            raise RuntimeError("generated original Q4_K source changed during verification")
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
