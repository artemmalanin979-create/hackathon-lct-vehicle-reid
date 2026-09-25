#!/usr/bin/env python3
"""Run the existing M5/M8 behavioral tests in disposable, offline containers.

This orchestrates existing unittest cases; it does not implement a second
validator. Only a temporary source copy is mutated. No dataset or database is
mounted. A FIFO timeout inside its dedicated test is the expected M8 behavior;
an outer container timeout is an infrastructure failure, never a killed mutant.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
TARGET = Path("04-solution/service/app/input_checks.py")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    original = (ROOT / TARGET).read_bytes()
    report = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": sha(original), "image": args.image, "runs": [], "mutations": {}}
    report["image_id"] = subprocess.check_output(
        ["podman", "image", "inspect", "--format", "{{.Id}}", args.image], text=True).strip()
    report["test_hashes"] = {name: sha((ROOT / name).read_bytes()) for name in (
        "04-solution/service/tests/test_image_names.py", "04-solution/reproduce/test_cli_inputs.py")}
    with tempfile.TemporaryDirectory(prefix="input-mutations-", dir=out) as temporary:
        copy = Path(temporary)
        for name in ("04-solution/service/app", "04-solution/service/tests"):
            shutil.copytree(ROOT / name, copy / name, ignore=shutil.ignore_patterns("__pycache__"))
        rep = copy / "04-solution/reproduce"
        rep.mkdir(parents=True)
        shutil.copy2(ROOT / "04-solution/reproduce/test_cli_inputs.py", rep)
        target = copy / TARGET

        def run(label, suite):
            cwd, test = suite
            cmd = ["podman", "run", "--rm", "--pull=never", "--network=none",
                   "--cpus=1", "--memory=512m", "--security-opt", "label=disable",
                   "-e", "OPENBLAS_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1",
                   "-e", "PYTHONPYCACHEPREFIX=/tmp/isolated-pycache",
                   "-v", f"{copy}:/case:ro", "-w", f"/case/{cwd}", args.image,
                   "python", "-B", "-m", "unittest", *test, "-v"]
            started = time.monotonic()
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=75)
            log = proc.stdout + proc.stderr
            (out / f"{label}.log").write_text(log)
            count = re.search(r"Ran (\d+) tests?", log)
            if not count or "skipped=" in log or "SyntaxError" in log or "ImportError" in log:
                raise AssertionError(f"Invalid test execution: {label}; inspect log")
            report["runs"].append({"label": label, "command": cmd, "exit_code": proc.returncode,
                                   "tests": int(count[1]), "skipped": 0,
                                   "elapsed_s": round(time.monotonic() - started, 3),
                                   "source_sha256": sha(target.read_bytes()), "log": f"{label}.log"})
            return proc.returncode, log

        suites = {
            "M5": ("04-solution/service", ["discover", "-s", "tests", "-p", "test_image_names.py"]),
            "M8": ("04-solution/reproduce", ["test_cli_inputs.CliInputTests.test_stream_preflight_does_not_consume_or_open_csv"]),
        }
        changes = {
            "M5": ('if "" in suffixes and Path(image_id).suffix.lower() in IMAGE_SUFFIXES:',
                   'if Path(image_id).suffix.lower() in IMAGE_SUFFIXES:'),
            "M8": ('        if not Path(path).is_file():\n            continue\n', ''),
        }
        for name, suite in suites.items():
            assert run(name + "-baseline", suite)[0] == 0
            before, after = changes[name]
            text = original.decode()
            assert text.count(before) == 1, f"Mutation {name} must apply exactly once"
            changed = text.replace(before, after, 1)
            assert changed != text
            diff = ''.join(difflib.unified_diff(text.splitlines(True), changed.splitlines(True),
                                               fromfile=str(TARGET), tofile=str(TARGET)))
            (out / f"{name}.diff").write_text(diff)
            target.write_text(changed)
            report["mutations"][name] = {"diff": f"{name}.diff", "sha256": sha(target.read_bytes())}
            try:
                rc, log = run(name + "-mutated", suite)
                assert rc != 0, f"Surviving mutation {name}"
                if name == "M5":
                    assert "test_jpeg_only_research_preflight" in log and "car.png" in log
                else:
                    assert "stream_preflight" in log
                    assert "TimeoutExpired" in log and "'fifo'" in log
            finally:
                target.write_bytes(original)
            assert sha(target.read_bytes()) == sha(original)
            assert run(name + "-restored", suite)[0] == 0
        report["status"] = "PASS"
    assert (ROOT / TARGET).read_bytes() == original
    (out / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "runs": len(report["runs"]), "result": str(out / "result.json")}))


if __name__ == "__main__":
    main()
