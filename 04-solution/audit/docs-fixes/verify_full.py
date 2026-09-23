#!/usr/bin/env python3
"""Recheck full cached research outputs, then infer and score the real validation.

Run on an isolated writable copy with organizer data and baseline/postproc vectors:
  python verify_full.py --repo /repo --data /data --out /checks
The output directory must be new; the runtime container supplies the shipped models.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True)
    env = {**os.environ, "REID_DATA_DIR": str(args.data), "REID_MODEL_PATH": os.environ["MODEL_PATH"],
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    solution = args.repo / "04-solution"
    results = []

    def run(name, cwd, command):
        with (args.out / f"{name}.txt").open("w") as log:
            p = subprocess.run([sys.executable, "-B", *command], cwd=cwd, env=env,
                               stdout=log, stderr=subprocess.STDOUT)
        results.append({"name": name, "cwd": str(cwd), "command": [sys.executable, "-B", *command],
                        "exit": p.returncode, "log": f"{name}.txt"})
        (args.out / "commands.json").write_text(json.dumps(results, indent=2) + "\n")
        print(name, p.returncode, flush=True)
        assert p.returncode == 0, (args.out / f"{name}.txt").read_text()

    def without_times(obj):
        if isinstance(obj, dict):
            return {k: without_times(v) for k, v in obj.items() if k != "rerank_seconds"}
        if isinstance(obj, list):
            return [without_times(v) for v in obj]
        return obj

    baseline, postproc, service = (solution / x for x in ("baseline", "postproc", "service"))
    expected = {str(p): json.loads(p.read_text()) for p in
                (baseline / "out/metrics_summary.json", postproc / "out/s01_verify_baseline.json",
                 postproc / "out/final_val.json")}
    artifacts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (baseline / "artifacts").iterdir()
                 if p.name in ("embeddings.npy", "submission.csv", "candidates.csv")}
    run("baseline-eval-full", baseline, ["scripts/run_eval.py"])
    run("baseline-submission-full", baseline, ["scripts/make_submission.py", "--threshold", "0.34921352213815304"])
    for name, digest in artifacts.items():
        assert hashlib.sha256((baseline / "artifacts" / name).read_bytes()).hexdigest() == digest, name
    run("baseline-order-full", baseline, ["scripts/check_embeddings_order.py"])
    assert json.loads((baseline / "out/order_check.json").read_text())["VERDICT"] == "OK"
    run("postproc-baseline-full", postproc, ["scripts/s01_verify_baseline.py"])
    run("postproc-tta-full", postproc, ["scripts/s05b_make_tta_vectors.py", "--sets", "val"])
    run("postproc-final-full", postproc, ["scripts/s06_final_val.py"])
    for path, value in expected.items():
        assert without_times(json.loads(Path(path).read_text())) == without_times(value), path
    (args.out / "research-comparison.json").write_text(json.dumps({
        "identical_except_timing": list(expected), "baseline_artifact_sha256": artifacts,
        "baseline_order": json.loads((baseline / "out/order_check.json").read_text())}, indent=2) + "\n")
    run("validation-inputs", solution / "reproduce", ["check_inputs.py", "--mode", "val",
        "--data-dir", str(args.data), "--report", str(args.out / "validation-inputs.json")])
    run("batch-validation", service, ["-m", "app.batch", "--images-dir", str(args.data / "images"),
        "--query", str(solution / "split/files/val_query.csv"),
        "--gallery", str(solution / "split/files/val_gallery.csv"),
        "--out-dir", str(args.out / "val"), "--threads", "2"])
    run("metrics-validation", service, ["tools/eval_split.py", str(args.out / "val/embeddings.npy")])
    metrics = json.loads((args.out / "metrics-validation.txt").read_text())
    assert metrics["rerank"]["mAP"] == 0.7740915539438481, metrics
    assert metrics["rerank"]["Rank-1"] == 0.7307692307692307, metrics
    assert metrics["scales_agree"], metrics
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("PASS: original research outputs unchanged; real validation = 0.7740915539438481 / 0.7307692307692307", flush=True)


if __name__ == "__main__":
    main()
