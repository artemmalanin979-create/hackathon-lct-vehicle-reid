"""Restore the P2 geometry bug in a disposable copy; require a numeric failure."""
import difflib
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "experiment.py"


def run(directory):
    return subprocess.run([sys.executable, "-m", "unittest", "test_selection", "-v"],
                          cwd=directory, capture_output=True, text=True, timeout=30)


def main():
    original = SOURCE.read_text()
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    before = run(HERE)
    if before.returncode:
        raise SystemExit(before.stdout + before.stderr)
    needle = 'release_lw = dict(np.load(files["whitening_release"], allow_pickle=False))'
    assert original.count(needle) == 1
    mutated = original.replace(needle, 'release_lw = dict(np.load(out / "teacher_fit_only.npz", allow_pickle=False))')
    diff = "".join(difflib.unified_diff(original.splitlines(True), mutated.splitlines(True),
                                         fromfile="release-geometry", tofile="wrong-teacher-geometry"))
    assert diff and mutated != original
    with tempfile.TemporaryDirectory(prefix="lct-selection-mutation-") as temp:
        directory = Path(temp)
        (directory / "experiment.py").write_text(mutated)
        for name in ["contracts.py", "test_selection.py"]:
            shutil.copy2(HERE / name, directory / name)
        result = run(directory)
        output = result.stdout + result.stderr
        killed = (result.returncode == 1 and "0.707106" in output and "0.894427" in output
                  and "FAILED (failures=1)" in output and "errors=" not in output)
    after = run(HERE)
    restored = hashlib.sha256(SOURCE.read_bytes()).hexdigest() == digest
    report = {"baseline_before_exit": before.returncode, "baseline_after_exit": after.returncode,
              "tests_per_baseline": 1, "baseline_sha256": digest, "original_hash_unchanged": restored,
              "mutation": {"status": "KILLED" if killed else "SURVIVED_OR_INVALID", "diff": diff,
                           "exit_code": result.returncode, "output": output}}
    (HERE / "results/selection-mutation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["mutation"]["status"], "baseline_after_exit": after.returncode,
                      "original_hash_unchanged": restored}))
    if not killed or after.returncode or not restored:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
