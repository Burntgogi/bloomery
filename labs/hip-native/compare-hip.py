#!/usr/bin/env python3
"""Compare the frozen small hybrid GGUF on native HIP and observed llama.cpp HIP.

This diagnostic has no floating-point accuracy band and never grants real-model acceptance.
"""
import argparse
import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

LAB = Path(__file__).resolve().parent
ROOT = LAB.parents[1]
DRIVER = ROOT / "tools/ref/dense-logits.py"
_spec = importlib.util.spec_from_file_location("hip_dense_reference", DRIVER)
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)
PROMPT = [3, 17, 41, 9, 23]
STEPS = 8
VOCAB = 258
MODEL_BYTES = 685984
MODEL_SHA = "420f680e27e9ec75c7c29d39298574f7fd2aff8808e9b2566f3915664016ed06"
TARGET = "hip-lab-hybrid"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_file(path):
    return json.loads(Path(path).read_text())


def ids_file(path, values):
    path.write_text("".join(f"{value}\n" for value in values))


def argmax(row):
    require(len(row) > 0 and all(math.isfinite(v) for v in row),
            "argmax requires a complete finite row")
    # Python max retains the first index on a tie, matching the raw-logit reader.
    return max(range(len(row)), key=row.__getitem__)


def aligned_rows(native_rows, prompt_length, count):
    require(prompt_length > 0 and count > 0, "invalid prompt cut or row count")
    offset = prompt_length - 1
    require(len(native_rows) == offset + count, "native consumed-row count differs at prompt cut")
    return native_rows[offset:]


def canonical_identity(data, source):
    require(data.get("schema") == 3 and data.get("status") == "pass",
            "canonical native result must be a passing schema-3 verify.py result")
    recorded = data.get("source", {})
    require(recorded.get("clean") is True, "canonical native evidence is not clean")
    for key in ("commit", "tree", "lab_tree"):
        require(recorded.get(key) == source[key], f"canonical native source differs at {key}")
    require(data.get("forward", {}).get("all_logit_files_finite") is True and
            data["forward"].get("reset_repeat_bitwise_equal") is True and
            data["forward"].get("independent_argmax_equal") is True,
            "canonical native forward acceptance is incomplete")


def model_identity(data, model, target, context):
    files = data.get("files", [])
    require(len(files) == 1 and Path(files[0]["file"]).resolve() == model and
            files[0].get("bytes") == target["size"] and files[0].get("sha256") == target["sha256"],
            "native metadata is not the same frozen model file")
    require(data.get("vocab") == target["vocab"] and data.get("layers") == 2 and
            data.get("gdn_layers") == 1 and data.get("gqa_layers") == 1,
            "native metadata differs from the frozen hybrid geometry")
    require(data.get("context") == context and data.get("ubatch") == 1 and data.get("kv") == "F16",
            "native context/ubatch/KV contract differs")


def native_rows(directory, prompt, steps, model, target, context):
    data = json_file(directory / "forward.json")
    model_identity(data["model"], model, target, context)
    greedy = data["greedy_ids"]
    require(data["prompt_ids"] == prompt and len(greedy) == steps,
            "native prompt or requested greedy length differs")
    require(all(type(i) is int and 0 <= i < target["vocab"] for i in greedy),
            "native greedy IDs are outside the frozen vocabulary")
    consumed = prompt + greedy[:-1] if steps else prompt
    require(data["consumed_ids"] == consumed and len(data["predictions"]) == len(consumed) and
            len(data["logit_sha256"]) == len(consumed), "native consumed/prediction/logit row contract differs")
    require(data.get("reset_repeat_runs") == 2 and data.get("reset_repeat_bitwise_equal") is True and
            data.get("all_observed_outputs_finite") is True and
            data.get("native_math") == "Rust AMDGPU/HIP" and data.get("cpu_compute_fallback") is False,
            "native HIP/reset/finite attribution is incomplete")
    require(ref.token_ids(directory / "prompt.ids", target["vocab"]) == prompt,
            "native prompt snapshot differs")
    require((directory / "forced.ids").read_text() == "".join(f"{i}\n" for i in greedy),
            "native generated-ID snapshot differs")
    rows = []
    artifacts = []
    for index, prediction in enumerate(data["predictions"]):
        paths = [directory / f"run-{repeat}" / f"logits-{index:04}.f32" for repeat in range(2)]
        first = paths[0].read_bytes()
        require(first == paths[1].read_bytes(), f"native reset mismatch at row {index}: {paths}")
        require(ref.digest(paths[0]) == data["logit_sha256"][index],
                f"native row {index} digest differs")
        values = list(ref.raw_rows(paths[0], 1, target["vocab"]))
        # Read both repeats through the common shape/finite reader, not only their flags.
        ref.raw_rows(paths[1], 1, target["vocab"])
        require(argmax(values) == prediction, f"native full-row argmax differs at row {index}")
        rows.append(values)
        artifacts.append({"row": index, "files": [str(p) for p in paths],
                          "sha256": ref.digest(paths[0]), "reset_bitwise_equal": True})
    if steps:
        require(data["predictions"][len(prompt) - 1:] == greedy,
                "native greedy IDs differ from independently read argmax rows")
    return data, rows, artifacts


def reference_rows(directory, prompt, forced, source, binary_sha, model, target, reserve):
    header, flat = ref.bundle(directory)
    meta = ref.read_meta(directory / "logits.0.f32.meta")
    expected = {"backend": "hip", "engine": "reference", "target": TARGET,
                "source_head": source["commit"], "source_dirty": "false", "binary_sha256": binary_sha,
                "model_sha256": target["sha256"], "model": str(model),
                "ctx": "64", "ubatch": "1", "kv": "f16", "prefill": "step", "mode": "reference",
                "rows": str(len(forced)), "vocab": str(target["vocab"]),
                "reset_repeats": "2", "reset_repeat_bitwise_equal": "true", "hip_output": "true"}
    for key, value in expected.items():
        require(header.get(key) == value, f"reference identity differs at {key}: {header.get(key)!r}")
    require(ref.token_ids(directory / "prompt.ids", target["vocab"]) == prompt and
            ref.token_ids(directory / "forced.ids", target["vocab"]) == forced,
            "reference prompt/forced snapshots differ")
    require(meta["prompt_ids"] == ",".join(map(str, prompt)) and
            meta["forced_ids"] == ",".join(map(str, forced)), "reference raw metadata IDs differ")
    placement = ref.hip_metadata(meta, reserve, 16 << 20)
    for key, value in placement.items():
        require(header.get(key) == value, f"reference placement/header attribution differs at {key}")
    for key in ("backend", "reset_repeats", "reset_repeat_bitwise_equal", "repeat_file"):
        require(header.get(key) == meta.get(key), f"reference raw/header attribution differs at {key}")
    require(int(header["hip_model_layers"]) == 2,
            "reference has not observed both frozen-model HIP trunk layers")
    require(int(header["hip_cpu_math_nodes"]) == int(header["hip_cpu_input_nodes"]),
            "reference CPU math exceeds the observed standard input-embedding leg")
    repeat_evidence = ref.verify_repeat(directory, meta, len(forced), target["vocab"])
    for key, value in repeat_evidence.items():
        require(header.get(key) == value, f"reference repeat/header attribution differs at {key}")
    repeat = header["repeat_file"]
    require((directory / "logits.0.f32").read_bytes() == (directory / repeat).read_bytes(),
            "reference reset raw bits differ")
    rows = [list(flat[i * target["vocab"]:(i + 1) * target["vocab"]]) for i in range(len(forced))]
    return rows, {"directory": str(directory), "manifest_sha256": ref.digest(directory / "MANIFEST.tsv"),
                  "header": header, "meta": meta, "invocation": json_file(directory / "invocation.json"),
                  "exit": json_file(directory / "exit.json"), "reset_raw_sha256": ref.digest(directory / repeat)}


def row_diagnostic(index, reference, native):
    require(len(reference) == len(native), "comparison row vocabulary differs")
    reference_id, native_id = argmax(reference), argmax(native)
    delta = math.fsum((float(a) - float(b)) ** 2 for a, b in zip(native, reference))
    norm = math.fsum(float(v) ** 2 for v in reference)
    return {"row": index, "native_row": len(PROMPT) - 1 + index,
            "reference_argmax": reference_id, "native_forced_argmax": native_id,
            "argmax_equal": reference_id == native_id,
            "max_abs": max(abs(float(a) - float(b)) for a, b in zip(native, reference)),
            "relative_l2": math.sqrt(delta / norm) if norm else (0.0 if not delta else None),
            "reference_norm_zero": norm == 0,
            "bitwise_equal": struct.pack(f"<{len(native)}f", *native) ==
                             struct.pack(f"<{len(reference)}f", *reference)}


def first_token_divergence(rows):
    return next((row for row in rows if not all(row[key] for key in
                 ("argmax_equal", "greedy_token_equal", "reference_iterative_final_equal"))), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new evidence directory outside the source checkout")
    parser.add_argument("--reference", type=Path, required=True, help="HIP reference binary with .build.json")
    parser.add_argument("--native-evidence", type=Path, required=True, help="current canonical verify.py result.json")
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.is_relative_to(ROOT), "comparison evidence must be outside the checkout")
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema": 1, "status": "error", "real_model_HIP_acceptance": False,
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "original_prompt_ids": PROMPT, "reference_greedy_ids": [], "rows": [],
              "reference_captures": [], "commands": [],
              "numeric_acceptance": "diagnostic_only_no_numeric_band",
              "memory_guard": "stage preflight only; not an OS VRAM cap or measured peak"}
    try:
        head = ref.clean_head(ROOT)
        source = {"commit": head, "tree": ref.git(ROOT, "rev-parse", "HEAD^{tree}"),
                  "lab_tree": ref.git(ROOT, "rev-parse", "HEAD:labs/hip-native"), "clean": True}
        result["source"] = source
        target = ref.TARGETS[TARGET]
        require((target["sha256"], target["size"], target["vocab"]) == (MODEL_SHA, MODEL_BYTES, VOCAB),
                "reference target differs from the approved frozen small GGUF")
        evidence_path = args.native_evidence.resolve()
        canonical = json_file(evidence_path)
        canonical_identity(canonical, source)
        model = (evidence_path.parent / "forward/smoke.gguf").resolve()
        model_stat = ref.immutable_model(model, target)
        binaries = canonical["binaries"]
        require(len(binaries) == 2, "canonical evidence must bind both native host and device binaries")
        host = (LAB / "target" / canonical["host_target"] / "release/bloomery-hip-spike").resolve()
        device = (LAB / "target/amdgcn-amd-amdhsa/release/bloomery_hip_spike_device.elf").resolve()
        require(host.is_relative_to((LAB / "target").resolve()) and
                device.is_relative_to((LAB / "target").resolve()), "native binaries must remain inside lab target")
        for record, path in zip(binaries, (host, device)):
            require((LAB / record["path"]).resolve() == path and ref.digest(path) == record["sha256"],
                    f"native binary path or SHA differs from canonical evidence: {path}")
        _, _, canonical_rows = native_rows(evidence_path.parent / "forward", PROMPT, STEPS, model, target, 16)
        require(canonical["forward"]["model"] == json_file(evidence_path.parent / "forward/forward.json")["model"],
                "canonical forward summary differs from its actual model identity")
        reference = args.reference.resolve()
        build = ref.check_build(reference, "hip")
        require(build.get("hip_build") is True, "reference was not built with HIP")
        require(build.get("libraries_rebuilt") is True and
                build.get("library_source_proof") == "successful_incremental_build",
                "cached-library smoke attribution cannot establish this HIP comparison")
        result["model"] = {"path": str(model), "target": TARGET, **target}
        result["canonical_native"] = {"path": str(evidence_path), "sha256": ref.digest(evidence_path),
                                      "source": canonical["source"], "binaries": binaries,
                                      "card": canonical["card"], "reset_rows": canonical_rows}
        result["reference_build"] = build
        identities = {path: ref.digest(path) for path in (host, device, reference, DRIVER, Path(__file__).resolve(),
                                                         evidence_path, Path(str(reference) + ".build.json"))}
        result["file_identities"] = [{"path": str(path), "sha256": sha} for path, sha in identities.items()]
        reserve = canonical["forward"]["model"]["fit_reserve_bytes"]
        require(reserve == ref.fit_reserve(),
                "canonical native reserve differs from the current production owner")

        def unchanged():
            require(ref.clean_head(ROOT) == head, "source HEAD moved during comparison")
            for path, sha in identities.items():
                require(ref.digest(path) == sha, f"comparison input changed: {path}")
            ref.model_unchanged(model, model_stat)

        def capture(label, prompt, forced):
            unchanged()
            prompt_path, forced_path = output / f"{label}.prompt.ids", output / f"{label}.forced.ids"
            ids_file(prompt_path, prompt)
            ids_file(forced_path, forced)
            directory = output / label
            command = [sys.executable, str(DRIVER), "run", "--kind", "reference", "--backend", "hip",
                       "--target", TARGET, "--model", str(model), "--binary", str(reference),
                       "--prompt", str(prompt_path), "--forced", str(forced_path), "--out", str(directory),
                       "--prefill", "step", "--ctx", "64", "--ubatch", "1", "--reset-repeats", "2",
                       "--hip-max-model-bytes", str(16 << 20)]
            entry = {"name": label, "equivalent_driver_argv": command,
                     "execution": "dense-logits.run API; actual child argv/exit/log in bundle or .staging",
                     "directory": str(directory), "status": "error"}
            result["commands"].append(entry)
            # In-process use preserves bounded_run's ownership of the actual GPU child;
            # wrapping the driver in a second process would create nested owned sessions.
            ref.run(argparse.Namespace(kind="reference", backend="hip", target=TARGET, model=model,
                    binary=reference, prompt=prompt_path, forced=forced_path, out=directory,
                    prefill="step", mode="graph", ctx=64, ubatch=1, reset_repeats=2,
                    hip_max_model_bytes=16 << 20))
            rows, attribution = reference_rows(directory, prompt, forced, source, identities[reference],
                                               model, target, reserve)
            result["reference_captures"].append(attribution)
            entry["status"] = "pass"
            return rows

        greedy = result["reference_greedy_ids"]
        iterative = []
        for index in range(STEPS):
            row = capture(f"reference-step-{index}", PROMPT + greedy, [0])[0]
            iterative.append(row)
            greedy.append(argmax(row))
        reference_values = capture("reference-final", PROMPT, greedy)
        final_ids = [argmax(row) for row in reference_values]
        result["reference_final_argmax_ids"] = final_ids
        result["reference_iterative_final_agreement"] = final_ids == greedy
        result["reference_iterative_final_rows"] = [row_diagnostic(i, iterative[i], reference_values[i])
                                                    for i in range(STEPS)]

        def native(label, prompt, steps):
            unchanged()
            prompt_path = output / f"{label}.prompt.ids"
            ids_file(prompt_path, prompt)
            directory = output / label
            argv = [str(host), "--forward", str(device), str(model), str(prompt_path), str(directory),
                    str(steps), "64"]
            log = output / f"{label}.log"
            entry = {"name": label, "argv": argv, "log": str(log), "status": "error"}
            result["commands"].append(entry)
            try:
                with log.open("w") as stream:
                    entry["exit_code"] = ref.bounded_run(argv, os.environ.copy(), stream, 900)
            except subprocess.TimeoutExpired:
                entry.update({"exit_code": 124, "timeout": True})
                raise ValueError(f"{label} timed out; incomplete evidence retained at {directory}") from None
            finally:
                if log.exists():
                    entry["log_sha256"] = ref.digest(log)
            require(entry["exit_code"] == 0, f"{label} exited {entry['exit_code']}; see {log}")
            data, values, artifacts = native_rows(directory, prompt, steps, model, target, 64)
            entry["status"] = "pass"
            result[label.replace("-", "_")] = {"metadata": data, "reset_rows": artifacts,
                                                  "directory": str(directory)}
            return data, values

        native_data, native_greedy_rows = native("native-greedy", PROMPT, STEPS)
        forced_data, native_forced_rows = native("native-forced", PROMPT + greedy[:-1], 0)
        native_greedy_rows = aligned_rows(native_greedy_rows, len(PROMPT), STEPS)
        native_forced_rows = aligned_rows(native_forced_rows, len(PROMPT), STEPS)
        result["native_greedy_ids"] = native_data["greedy_ids"]
        result["native_forced_argmax_ids"] = [argmax(row) for row in native_forced_rows]
        for index in range(STEPS):
            diagnostic = row_diagnostic(index, reference_values[index], native_forced_rows[index])
            diagnostic["reference_greedy_id"] = greedy[index]
            diagnostic["native_greedy_id"] = native_data["greedy_ids"][index]
            diagnostic["native_greedy_argmax"] = argmax(native_greedy_rows[index])
            diagnostic["greedy_token_equal"] = native_data["greedy_ids"][index] == greedy[index]
            diagnostic["reference_iterative_final_equal"] = final_ids[index] == greedy[index]
            result["rows"].append(diagnostic)
        first = first_token_divergence(result["rows"])
        if first is not None:
            index = first["row"]
            divergence = {"diagnostic": first, "reference_prefix_ids": PROMPT + greedy[:index],
                          "native_greedy_prefix_ids": PROMPT + native_data["greedy_ids"][:index],
                          "reference_logits": reference_values[index],
                          "native_reference_forced_logits": native_forced_rows[index],
                          "native_greedy_logits": native_greedy_rows[index],
                          "reference_raw": str(output / "reference-final/logits.0.f32"),
                          "reference_byte_offset": index * VOCAB * 4,
                          "native_row_file": str(output / f"native-forced/run-0/logits-{len(PROMPT)-1+index:04}.f32")}
            ref.write_json(output / "first-divergence.json", divergence)
            result["first_divergence"] = {"row": index, "artifact": "first-divergence.json"}
        unchanged()
        ref.check_build(reference, "hip")
        result["status"] = "mismatch" if first is not None else "pass"
    except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError, subprocess.SubprocessError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
        print(result["error"], file=sys.stderr)
    except KeyboardInterrupt:
        result["error"] = "interrupted; bounded child cleanup requested"
    finally:
        result["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        ref.write_json(output / "comparison.json", result)
        print(output / "comparison.json", flush=True)
    return 0 if result["status"] == "pass" else 1


def self_test():
    """Exercise observable row cuts, finite/argmax and evidence identity rejection without a GPU."""
    def rejects(call):
        try:
            call()
        except ValueError:
            return
        raise AssertionError("invalid evidence was accepted")

    # Distinct prompt-only predictions make the plausible off-by-one cut observable.
    native = [[9., 0., 0.], [0., 8., 0.], [0., 0., 7.], [6., 0., 0.]]
    assert [argmax(row) for row in aligned_rows(native, 3, 2)] == [2, 0]
    rejects(lambda: aligned_rows(native, 2, 2))
    rejects(lambda: aligned_rows(native, 0, 2))
    assert argmax([2., 2., 1.]) == 0
    assert argmax([-3., -1., -2.]) == 1
    for value in (math.nan, math.inf, -math.inf):
        rejects(lambda value=value: argmax([1., value]))
    rejects(lambda: row_diagnostic(0, [1., 2.], [1.]))
    assert row_diagnostic(0, [0., 0.], [1., 0.])["relative_l2"] is None
    assert not row_diagnostic(0, [1., 2.], [3., 2.])["argmax_equal"]
    good = {"argmax_equal": True, "greedy_token_equal": True, "reference_iterative_final_equal": True}
    assert first_token_divergence([good]) is None
    for key in good:
        bad = {**good, key: False}
        assert first_token_divergence([good, bad, good]) is bad
    source = {"commit": "head", "tree": "tree", "lab_tree": "lab", "clean": True}
    canonical = {"schema": 3, "status": "pass", "source": dict(source),
                 "forward": {"all_logit_files_finite": True, "reset_repeat_bitwise_equal": True,
                             "independent_argmax_equal": True}}
    canonical_identity(canonical, source)
    for key, value in (("commit", "old"), ("tree", "old"), ("lab_tree", "old"), ("clean", False)):
        bad = {**canonical, "source": {**source, key: value}}
        rejects(lambda bad=bad: canonical_identity(bad, source))
    rejects(lambda: canonical_identity({**canonical, "status": "fail"}, source))
    # A valid raw file must still be rejected when its immutable identity is wrong.
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "row.f32"
        path.write_bytes(struct.pack("<3f", 1., 2., 3.))
        target = {"size": 12, "sha256": ref.digest(path)}
        snapshot = ref.immutable_model(path, target)
        assert argmax(ref.raw_rows(path, 1, 3)) == 2
        rejects(lambda: ref.immutable_model(path, {**target, "sha256": "0" * 64}))
        rejects(lambda: ref.raw_rows(path, 2, 3))
        path.write_bytes(struct.pack("<3f", 1., math.nan, 3.))
        rejects(lambda: ref.raw_rows(path, 1, 3))
        rejects(lambda: ref.model_unchanged(path, snapshot))
    print("HIP comparator no-GPU consumer contract self-test passed")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        raise SystemExit(self_test())
    raise SystemExit(main())
