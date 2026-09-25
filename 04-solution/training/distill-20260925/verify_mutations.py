"""Four bounded guard removals in disposable copies; no repository mutation."""
import difflib
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "contracts.py"


def run(directory):
    return subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(directory), "-p", "test_contracts.py", "-v"],
                          capture_output=True, text=True, timeout=30)


def main():
    original = SOURCE.read_text()
    digest = hashlib.sha256(original.encode()).hexdigest()
    before = run(HERE)
    if before.returncode:
        raise SystemExit(before.stdout + before.stderr)
    cases = {
        "row_order": 'raise ValueError("feature row order differs from metadata")',
        "finite_nonzero": 'raise ValueError("nonfinite or zero feature")',
        "identity_leakage": 'raise ValueError(f"identity overlap: {sorted(a & b)[:10]}")',
        "file_hash": 'raise ValueError(f"hash mismatch: {path}: {actual} != {expected}")',
    }
    records = []
    for name, needle in cases.items():
        if original.count(needle) != 1:
            raise AssertionError("mutation not unique")
        mutated = original.replace(needle, "pass  # isolated semantic guard-removal mutation")
        diff = "".join(difflib.unified_diff(original.splitlines(True), mutated.splitlines(True), fromfile="baseline", tofile=name))
        if not diff or mutated == original:
            raise AssertionError("empty mutation")
        with tempfile.TemporaryDirectory(prefix="lct-distill-mutation-") as d:
            directory=Path(d)
            (directory / "contracts.py").write_text(mutated)
            shutil.copyfile(HERE / "test_contracts.py", directory / "test_contracts.py")
            result=run(directory)
            text=result.stdout+result.stderr
            killed=result.returncode == 1 and "ValueError not raised" in text and "FAILED (failures=" in text and "errors=" not in text
            records.append({"mutation":name,"status":"KILLED" if killed else "SURVIVED_OR_INVALID","diff":diff,"exit_code":result.returncode,"output":text})
    after=run(HERE)
    restored=hashlib.sha256(SOURCE.read_bytes()).hexdigest() == digest
    report={"baseline_before_exit":before.returncode,"baseline_after_exit":after.returncode,"tests_per_baseline":5,"baseline_sha256":digest,"original_hash_unchanged":restored,"mutations":records}
    (HERE / "results/mutations.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"killed":sum(r["status"]=="KILLED" for r in records),"total":len(records),"baseline_after_exit":after.returncode,"original_hash_unchanged":restored}))
    if after.returncode or not restored or any(r["status"]!="KILLED" for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
