#!/usr/bin/env python3
"""Cache verdicts and complete oracle-builder CLI ordering, without CUDA or a card.

Run with --self-test. CACHE_OPTIONS_TEST_ROOT selects a checkout for FAIL-before
runs; CACHE_OPTIONS_TEST_FILTER selects unittest method names (for example,
wrapper_missing). Each wrapper case copies its complete, unchanged entry script,
plus tools/ref/cache-options.sh and tools/grep-answer.sh when present. No other
production implementation is copied: ref-paths.sh, ref-build-common.sh, lease.sh
and build-dump.sh are fixture-only boundaries. The actual wrapper runs its Git/prerequisite,
configure, comparison and lease/build control flow.

Git, patch identity and privileged commands accept only enumerated fixture invocations.
BASH_ENV maps only `cd /home/user` and the fixed reference-cache grep operand
to private temporary paths; sudo validates the real wrapper's exact env prefix
but preserves the private PATH. Configure records an event without creating a
cache; build records an event and deliberately exits 88. These fixtures prove
ordering, option semantics and exit/error handling, not CUDA compilation, real
Git provenance, privilege switching, lease contention or dumper linking. They
never access a real user home, repository, card or machine lease.
"""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(os.environ.get("CACHE_OPTIONS_TEST_ROOT", Path(__file__).resolve().parents[2])).resolve()
WRAPPERS = ("build-ik-cand.sh", "build-ik-mimo2.sh")
SELECTED = "GGML_CUDA:BOOL=ON\nLLAMA_MODE:STRING=alpha=beta=gamma\n"
NOISE = "CMAKE_BUILD_TYPE:STRING=Release\n// comment\nggml_lower:BOOL=ON\nGGML_BAD-KEY:BOOL=ON\n"
FOUND = "GGML_CUDA_FOUND:BOOL=ON\nLLAMA_LIBRARY_FOUND:STRING=fixture\n"


def put(path, text, executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)


# One dispatcher keeps the accepted subprocess boundaries explicit. An unknown
# invocation fails closed rather than falling through to an installed utility.
BOUNDARY = r'''import os
from pathlib import Path
import subprocess
import sys

name = Path(sys.argv[0]).name
args = sys.argv[1:]
candidate = os.environ["FIXTURE_CANDIDATE"]
kind = os.environ["FIXTURE_KIND"]
source = "/home/user/ik_llama.cpp"
pin = {"cand": "ed27bf7e", "mimo2": "043ced9a", "draft": "db517b69"}[kind]
patched = ("src/llama-hparams.h", "src/llama-hparams.cpp", "src/llama-load-tensors.cpp",
           "src/graphs/build_deepseek4.cpp")
scratch = Path(os.environ["FIXTURE_HOME"]).parent / "scratch"

def reject():
    print("unexpected fixture invocation: " + repr([name, *args]), file=sys.stderr)
    sys.exit(97)

def event(value):
    with open(os.environ["FIXTURE_EVENTS"], "a", encoding="utf-8") as out:
        out.write(value + "\n")

with open(os.environ["FIXTURE_AUDIT"], "a", encoding="utf-8") as out:
    out.write(name + ": " + repr(args) + "\n")

if name == "sudo":
    prefix = ["-u", "user", "env", "HOME=/home/user",
              "PATH=/usr/local/cuda/bin:/usr/bin:/bin"]
    allowed = ("git", "cmake", "tee", "mktemp", "mkdir", "sed", "python3") if kind == "draft" else ("git", "cmake", "tee")
    if args[:5] != prefix or len(args) < 6 or args[5] not in allowed:
        reject()
    # Do not install the wrapper's real-machine PATH or HOME in this fixture.
    sys.exit(subprocess.run(args[5:], env=os.environ).returncode)
elif name == "git":
    if kind == "draft":
        if args in (["-C", "/home/user/ik-idxkey", "rev-parse", pin + "^{commit}"],
                    ["-C", candidate, "rev-parse", "HEAD"]):
            print("fixture-base")
        elif (len(args) == 4 and args[:3] == ["-C", candidate, "show"] and
              args[3] in [pin + ":" + path for path in patched]):
            print("original")
        elif args == ["-C", candidate, "diff", "--name-only", pin]:
            print("\n".join(patched))
        elif args == ["-C", candidate, "diff", "--stat", pin]:
            print("fixture patched files")
        elif args != ["-C", candidate, "status", "--porcelain", "--untracked-files=all"]:
            reject()
    elif args == ["patch-id", "--stable"]:
        patch = sys.stdin.read()
        if patch not in ("fixture-patch-49ef19d0\n", "fixture-patch-5bf8f0fe\n"):
            reject()
        print(patch.strip().removeprefix("fixture-patch-") + " fixture-commit")
    elif args == ["-C", source, "rev-parse", pin + "^{commit}"]:
        print("fixture-base")
    elif args == ["-C", candidate, "rev-parse", "HEAD^" if kind == "cand" else "HEAD~2"]:
        print("fixture-base")
    elif args == ["-C", candidate, "rev-parse", "--short=8", "HEAD"]:
        print("abc12345")
    elif args == ["-C", candidate, "status", "--porcelain", "--untracked-files=all"]:
        pass
    elif len(args) == 4 and args[:3] == ["-C", candidate, "show"]:
        allowed = {"49ef19d0": "49ef19d0", "HEAD": "49ef19d0"} if kind == "cand" else {
            "49ef19d0": "49ef19d0", "5bf8f0fe": "5bf8f0fe",
            "HEAD~1": "49ef19d0", "HEAD~0": "5bf8f0fe"}
        if args[3] not in allowed:
            reject()
        print("fixture-patch-" + allowed[args[3]])
    elif (kind == "mimo2" and len(args) == 6 and
          args[:5] == ["-C", candidate, "log", "-1", "--format=%s"] and
          args[5] in (pin, "49ef19d0", "5bf8f0fe")):
        print("fixture subject")
    else:
        reject()
elif name == "find":
    if kind != "mimo2" or args != [source + "/.git", "!", "-user", "user", "-print", "-quit"]:
        reject()
elif name == "tee":
    draft_path = len(args) == 1 and Path(args[0]).resolve().is_relative_to(scratch)
    if not ((kind == "draft" and draft_path) or (kind == "mimo2" and args == [candidate + "/PROVENANCE"])):
        reject()
    value = sys.stdin.buffer.read()
    Path(args[0]).write_bytes(value)
    sys.stdout.buffer.write(value)
elif name == "cmake":
    configure = ["-S", candidate, "-B", candidate + "/build", "-DCMAKE_BUILD_TYPE=Release",
                 "-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=86",
                 "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc"]
    targets = ["llama", "common"]
    if kind == "mimo2":
        configure.append("-DGGML_IQK_FA_ALL_QUANTS=ON")
        targets.append("llama-tokenize")
    elif kind == "draft":
        targets.append("llama-spec-bench")
    if args == configure:
        event("configure")
    elif (len(args) == 5 + len(targets) and args[:3] == ["--build", candidate + "/build", "-j"]
          and args[3].isdigit() and int(args[3]) > 0 and args[4:] == ["--target", *targets]):
        event("build")
        sys.exit(88)
    else:
        reject()
elif name == "sha256sum":
    if kind != "draft" or args != ["/home/user/ik-dsv41-draft.py"]:
        reject()
    # The fixed patch identity is an external provenance boundary, like Git above.
    print("4eeb56aa1d75b470b556bab87ab90a467c8e4942b3269c9c15a3c78db0b1bc60  " + args[0])
elif name == "mktemp":
    if kind != "draft" or args != ["-d", "/tmp/dsv41-draft.XXXXXX"]:
        reject()
    scratch.mkdir()
    print(scratch)
elif name == "mkdir":
    if kind != "draft" or len(args) != 2 or args[0] != "-p" or not Path(args[1]).resolve().is_relative_to(scratch):
        reject()
    Path(args[1]).mkdir(parents=True, exist_ok=True)
elif name == "sed":
    expected = 's|^ROOT = .*|ROOT = "' + str(scratch) + '/src/"|'
    if kind != "draft" or args != [expected, "/home/user/ik-dsv41-draft.py"]:
        reject()
    sys.exit(subprocess.run(["/usr/bin/sed", expected, os.environ["FIXTURE_PATCH"]]).returncode)
elif name == "python3":
    if kind != "draft" or args != [str(scratch / "patch.py")]:
        reject()
    sys.exit(subprocess.run([sys.executable, *args]).returncode)
else:
    reject()
'''


class Fixture:
    def __init__(self, directory):
        self.root = Path(directory)
        self.reference = self.root / "reference"
        self.candidate = self.root / "candidate"
        self.bin = self.root / "bin"
        for path in (self.reference / "build", self.candidate / "build", self.bin, self.root / "home"):
            path.mkdir(parents=True)
        self.events = self.root / "events"
        self.audit = self.root / "audit"
        self.env = {
            "PATH": str(self.bin) + os.pathsep + "/usr/bin:/bin",
            "HOME": str(self.root / "home"),
            "LC_ALL": "C",
            "FIXTURE_HOME": str(self.root / "home"),
            "FIXTURE_REFERENCE": str(self.reference),
            "FIXTURE_CANDIDATE": str(self.candidate),
            "FIXTURE_EVENTS": str(self.events),
            "FIXTURE_AUDIT": str(self.audit),
            "BLOOMERY_BUILD_BOUND": "60",
        }

    def cache(self, tree, state):
        path = tree / "build/CMakeCache.txt"
        if state is None:
            return
        if state == "directory":
            path.mkdir()
        else:
            put(path, state)

    def fail_utility(self, name):
        put(self.bin / name, "#!/bin/bash\nprintf 'fixture " + name + " failure\\n' >&2\nexit 73\n", True)

    def helper(self, function, *args):
        return subprocess.run(
            ["/bin/bash", "-c", 'set +o pipefail; source "$1"; shift; "$@"', "cache-options-test",
             str(ROOT / "tools/ref/cache-options.sh"), function, "cache-fixture", *map(str, args)],
            cwd=self.root, env=self.env, capture_output=True, text=True, timeout=30,
        )

    def wrapper(self, script):
        # copyfile preserves the complete wrapper body: no rewriting its constants
        # or substituting an extracted comparison fragment.
        for relative in ("tools/ref/" + script, "tools/ref/cache-options.sh", "tools/grep-answer.sh"):
            source = ROOT / relative
            if source.exists():
                target = self.root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        kind = {"build-ik-cand.sh": "cand", "build-ik-mimo2.sh": "mimo2", "build-dump-draft.sh": "draft"}[script]
        self.env["FIXTURE_KIND"] = kind
        boundaries = ["sudo", "git", "find", "tee", "cmake"]
        if kind == "draft":
            boundaries += ["sha256sum", "mktemp", "mkdir", "sed", "python3"]
            self.env["FIXTURE_PATCH"] = str(self.root / "draft-patch.py")
            patched = ("src/llama-hparams.h", "src/llama-hparams.cpp", "src/llama-load-tensors.cpp",
                       "src/graphs/build_deepseek4.cpp")
            for relative in patched:
                put(self.candidate / relative, "patched\n")
            put(Path(self.env["FIXTURE_PATCH"]),
                'ROOT = ""\nfrom pathlib import Path\n'
                f'for name in {tuple(path.removeprefix("src/") for path in patched)!r}:\n'
                '    Path(ROOT, name).write_text("patched\\n")\n')
            put(self.root / "tools/ref/ref-build-common.sh",
                "REF_BIN=" + shlex.quote(str(self.root / "out")) + "\nHERE=" + shlex.quote(str(self.root)) + "\n")
            self.env["DRAFT_IK"] = str(self.candidate)
        for name in boundaries:
            put(self.bin / name, "#!" + sys.executable + "\n" + BOUNDARY, True)
        profile = (
            "IK=" + shlex.quote(str(self.reference if kind == "cand" else self.candidate)) + "\n"
            "CAND_IK=" + shlex.quote(str(self.candidate)) + "\n"
            "BLOOMERY_DATA=" + shlex.quote(str(self.root / "data")) + "\n"
            "REF_BIN_NAME=bin-mimo2\n"
        )
        put(self.root / "tools/ref/ref-paths.sh", profile)
        put(self.root / "tools/ref/lease.sh", r'''lease_take() { printf 'lease\n' >> "$FIXTURE_EVENTS"; }
lease_release() { printf 'release\n' >> "$FIXTURE_EVENTS"; }
lease_bounded() {
  [ "$1" = 60 ] || return 97
  shift
  [ "$1" = bash ] || return 97
  "$@"
}
''')
        put(self.root / "tools/ref/build-dump.sh",
            '#!/bin/bash\nprintf "dumper\\n" >> "$FIXTURE_EVENTS"\nexit 97\n')
        put(self.root / "boundary-env.sh", r'''cd() {
  if [ "$#" = 1 ] && [ "$1" = /home/user ]; then
    builtin cd "$FIXTURE_HOME"
  else
    builtin cd "$@"
  fi
}
grep() {
  local arg
  local -a mapped=()
  for arg in "$@"; do
    if [ "$arg" = /home/user/ik-idxkey/build/CMakeCache.txt ]; then
      mapped+=("$FIXTURE_REFERENCE/build/CMakeCache.txt")
    else
      mapped+=("$arg")
    fi
  done
  command grep "${mapped[@]}"
}
''')
        self.env["BASH_ENV"] = str(self.root / "boundary-env.sh")
        return subprocess.run(
            ["/bin/bash", str(self.root / "tools/ref" / script)],
            cwd=self.root, env=self.env, capture_output=True, text=True, timeout=30,
        )

    def fail_cache_sort(self):
        put(self.bin / "sort", '#!/bin/bash\n'
            'if [ -f "$FIXTURE_EVENTS" ] && /usr/bin/grep -q "^configure$" "$FIXTURE_EVENTS"; then\n'
            '  printf "fixture cache sort failure\\n" >&2\n  exit 73\nfi\n'
            'exec /usr/bin/sort "$@"\n', True)


class CacheOptions(unittest.TestCase):
    def fixture(self):
        directory = tempfile.TemporaryDirectory(prefix="cache-options-")
        self.addCleanup(directory.cleanup)
        return Fixture(directory.name)

    def verdict(self, result, expected, label):
        self.assertEqual(result.returncode, expected,
                         f"{label}: rc={result.returncode}, expected={expected}\n"
                         f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    def context(self, result, tool, *paths):
        self.assertIn(tool, result.stderr)
        for path in paths:
            self.assertIn(str(path), result.stderr)

    def ordered_rows(self, output, rows):
        positions = [output.find(row) for row in rows]
        self.assertTrue(all(index >= 0 for index in positions), repr((rows, output)))
        self.assertEqual(positions, sorted(positions), output)

    def test_helper_selected_snapshots(self):
        cases = (
            ("empty", "", ""),
            ("noise", NOISE, ""),
            ("found-only", FOUND, ""),
            ("selection-and-full-values", "LLAMA_MODE:STRING=alpha=beta=gamma\n" + NOISE + FOUND +
             "GGML_CUDA:BOOL=ON\n", SELECTED),
        )
        for name, cache, expected in cases:
            with self.subTest(case=name):
                fixture = self.fixture()
                fixture.cache(fixture.reference, cache)
                result = fixture.helper("ref_cache_flags", fixture.reference)
                self.verdict(result, 0, name)
                # Exact bytes distinguish a genuinely empty snapshot from a phantom newline.
                self.assertEqual(result.stdout, expected)
                self.assertEqual(result.stderr, "")

    def test_helper_missing_and_directory_caches(self):
        for side in ("reference", "candidate"):
            for state in (None, "directory"):
                with self.subTest(side=side, state=state):
                    fixture = self.fixture()
                    fixture.cache(fixture.reference, state if side == "reference" else SELECTED)
                    fixture.cache(fixture.candidate, state if side == "candidate" else SELECTED)
                    result = fixture.helper("ref_cache_check", fixture.reference, fixture.candidate)
                    self.verdict(result, 2, f"{side}-{state}")
                    tree = getattr(fixture, side)
                    self.context(result, "cache-fixture", tree / "build/CMakeCache.txt")
                    self.assertEqual(result.stdout, "")

    def test_helper_utility_failures_without_pipefail(self):
        for utility in ("grep", "sort", "join", "awk"):
            with self.subTest(utility=utility):
                fixture = self.fixture()
                fixture.cache(fixture.reference, SELECTED)
                fixture.cache(fixture.candidate, SELECTED)
                fixture.fail_utility(utility)
                result = fixture.helper("ref_cache_check", fixture.reference, fixture.candidate)
                self.verdict(result, 2, utility)
                self.context(result, "cache-fixture")
                if utility in ("grep", "sort"):
                    self.context(result, "cache-fixture", fixture.reference / "build/CMakeCache.txt")
                self.assertEqual(result.stdout, "")

    def comparison_cases(self):
        return (
            ("identical", SELECTED, SELECTED, 0, ()),
            ("common-mismatch", "GGML_CUDA:BOOL=OFF\n", "GGML_CUDA:BOOL=ON\n", 1, ("OFF", "ON")),
            ("full-equals-identical", "LLAMA_MODE:STRING=a=b=c\n", "LLAMA_MODE:STRING=a=b=c\n", 0, ()),
            ("full-equals-mismatch", "LLAMA_MODE:STRING=a=b=c\n", "LLAMA_MODE:STRING=a=b=d\n", 1,
             ("a=b=c", "a=b=d")),
            ("one-sided-sorted", "LLAMA_Z:BOOL=OFF\nGGML_COMMON:BOOL=ON\nGGML_A:STRING=a=b\n",
             "LLAMA_B:BOOL=ON\nGGML_COMMON:BOOL=ON\n", 0,
             ("GGML_A:STRING=a=b", "LLAMA_B:BOOL=ON", "LLAMA_Z:BOOL=OFF")),
            ("typed-keys", "GGML_OPTION:BOOL=ON\n", "GGML_OPTION:STRING=ON\n", 0,
             ("GGML_OPTION:BOOL=ON", "GGML_OPTION:STRING=ON")),
            ("both-empty", "", "", 0, ()),
            ("both-noise", NOISE, NOISE, 0, ()),
            ("both-found-only", FOUND, FOUND, 0, ()),
            ("empty-reference", "", SELECTED, 0, tuple(SELECTED.splitlines())),
            ("empty-candidate", SELECTED, "", 0, tuple(SELECTED.splitlines())),
            ("noise-reference", NOISE + FOUND, SELECTED, 0, tuple(SELECTED.splitlines())),
        )

    def check_options_output(self, result, expected, rows, tool, reference, candidate):
        if expected == 1:
            self.context(result, tool, reference, candidate)
            for value in rows:
                self.assertIn(value, result.stderr)
            self.assertIn("GGML_CUDA:BOOL" if rows == ("OFF", "ON") else "LLAMA_MODE:STRING", result.stderr)
        elif rows:
            self.ordered_rows(result.stdout, rows)
            for value in (tool, str(reference), str(candidate)):
                self.assertIn(value, result.stdout)
            for row in rows:
                self.assertNotIn(row, result.stderr)
        else:
            self.assertEqual(result.stderr, "")

    def test_helper_comparisons(self):
        for name, reference, candidate, expected, rows in self.comparison_cases():
            with self.subTest(case=name):
                fixture = self.fixture()
                fixture.cache(fixture.reference, reference)
                fixture.cache(fixture.candidate, candidate)
                result = fixture.helper("ref_cache_check", fixture.reference, fixture.candidate)
                self.verdict(result, expected, name)
                self.check_options_output(result, expected, rows, "cache-fixture", fixture.reference, fixture.candidate)
                if expected == 0 and not rows:
                    self.assertEqual(result.stdout, "")

    def wrapper_events(self, fixture, result, expected, label):
        self.verdict(result, expected, label)
        events = fixture.events.read_text().splitlines() if fixture.events.exists() else []
        self.assertEqual(events, ["configure", "lease", "build"] if expected == 88 else ["configure"],
                         f"{label}: unexpected control-flow events {events}\n{result.stdout}\n{result.stderr}")
        audit = fixture.audit.read_text()
        operations = ("rev-parse", "show", "diff", "status") if fixture.env["FIXTURE_KIND"] == "draft" else ("rev-parse", "show", "patch-id", "status")
        for operation in operations:
            self.assertIn(operation, audit)
        if fixture.env["FIXTURE_KIND"] == "mimo2":
            self.assertIn("find:", audit)
            self.assertIn("tee:", audit)

    def test_wrapper_missing_and_directory_caches(self):
        for script in WRAPPERS:
            for side in ("reference", "candidate"):
                for state in (None, "directory"):
                    with self.subTest(wrapper=script, side=side, state=state):
                        fixture = self.fixture()
                        fixture.cache(fixture.reference, state if side == "reference" else SELECTED)
                        fixture.cache(fixture.candidate, state if side == "candidate" else SELECTED)
                        result = fixture.wrapper(script)
                        self.wrapper_events(fixture, result, 2, f"{script}-{side}-{state}")
                        reference = Path("/home/user/ik-idxkey") if script == "build-ik-mimo2.sh" else fixture.reference
                        tree = reference if side == "reference" else fixture.candidate
                        self.context(result, script.removesuffix(".sh"), tree / "build/CMakeCache.txt")

    def test_wrapper_comparisons(self):
        for script in WRAPPERS:
            for name, reference_cache, candidate_cache, expected, rows in self.comparison_cases():
                with self.subTest(wrapper=script, case=name):
                    fixture = self.fixture()
                    fixture.cache(fixture.reference, reference_cache)
                    fixture.cache(fixture.candidate, candidate_cache)
                    result = fixture.wrapper(script)
                    self.wrapper_events(fixture, result, 88 if expected == 0 else expected, f"{script}-{name}")
                    reference = Path("/home/user/ik-idxkey") if script == "build-ik-mimo2.sh" else fixture.reference
                    self.check_options_output(result, expected, rows, script.removesuffix(".sh"), reference, fixture.candidate)

    def test_draft_failed_snapshots_stop_before_lease(self):
        fixture = self.fixture()
        fixture.cache(fixture.reference, SELECTED)
        fixture.cache(fixture.candidate, SELECTED)
        fixture.fail_cache_sort()
        result = fixture.wrapper("build-dump-draft.sh")
        self.wrapper_events(fixture, result, 2, "draft-failed-cache-sorts")
        self.context(result, "build-dump-draft", Path("/home/user/ik-idxkey/build/CMakeCache.txt"))

    def test_draft_missing_and_directory_caches(self):
        for side in ("reference", "candidate"):
            for state in (None, "directory"):
                with self.subTest(side=side, state=state):
                    fixture = self.fixture()
                    fixture.cache(fixture.reference, state if side == "reference" else SELECTED)
                    fixture.cache(fixture.candidate, state if side == "candidate" else SELECTED)
                    result = fixture.wrapper("build-dump-draft.sh")
                    self.wrapper_events(fixture, result, 2, f"draft-{side}-{state}")
                    tree = Path("/home/user/ik-idxkey") if side == "reference" else fixture.candidate
                    self.context(result, "build-dump-draft", tree / "build/CMakeCache.txt")

    def test_draft_strict_comparison(self):
        for name, reference, candidate, expected, _ in self.comparison_cases():
            strict = 88 if name in ("identical", "full-equals-identical", "both-empty", "both-noise", "both-found-only") else 1
            with self.subTest(case=name):
                fixture = self.fixture()
                fixture.cache(fixture.reference, reference)
                fixture.cache(fixture.candidate, candidate)
                result = fixture.wrapper("build-dump-draft.sh")
                self.wrapper_events(fixture, result, strict, f"draft-{name}")

    def test_draft_comparison_error_stops_before_lease(self):
        fixture = self.fixture()
        fixture.cache(fixture.reference, SELECTED)
        fixture.cache(fixture.candidate, SELECTED)
        fixture.fail_utility("diff")
        result = fixture.wrapper("build-dump-draft.sh")
        self.wrapper_events(fixture, result, 2, "draft-failed-comparison")


def self_test():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CacheOptions)
    selected = os.environ.get("CACHE_OPTIONS_TEST_FILTER")
    if selected:
        suite = unittest.TestSuite(test for test in suite if selected in test.id())
        if suite.countTestCases() == 0:
            raise RuntimeError(f"cache-options-test: no test matches {selected!r}")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if sys.argv[1:] != ["--self-test"]:
        print("usage: cache-options-test.py --self-test", file=sys.stderr)
        sys.exit(64)
    sys.exit(self_test())
