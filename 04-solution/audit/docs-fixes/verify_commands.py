#!/usr/bin/env python3
"""Run changed CLIs with real runtime dependencies and isolated fixture outputs.

Example (inside the runtime container, 2 CPU / 4 GiB):
  python verify_commands.py --repo /repo --before /before --out /checks
The --before tree contains the original service/app from commit 3e2cc9a.
No organizer data is needed; model paths come from the runtime environment.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--before", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fixture = args.out / "fixture"
    solution = fixture / "04-solution"
    for directory in ("baseline/scripts", "postproc/scripts", "eval", "service/app"):
        shutil.copytree(args.repo / "04-solution" / directory, solution / directory)
    for directory in ("baseline/out", "baseline/artifacts", "postproc/out", "split/files"):
        (solution / directory).mkdir(parents=True, exist_ok=True)
    data = fixture / "data"
    images = data / "images"
    images.mkdir(parents=True)
    rng = np.random.default_rng(89)
    fields = ["image_id", "x", "y", "w", "h", "vehicle_id", "camera_id", "has_mate"]
    for part, count in (("query", 2), ("gallery", 10)):
        rows = []
        for i in range(count):
            image_id = f"{part}{i}"
            Image.fromarray(rng.integers(0, 256, (96, 112, 3), dtype=np.uint8)).save(images / f"{image_id}.jpg")
            rows.append([image_id, 3, 2, 88, 90, str(i), "q" if part == "query" else "g", "1"])
        for path in (data / f"test_{part}.csv", solution / f"split/files/val_{part}.csv"):
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(fields)
                writer.writerows(rows)
    env = {**os.environ, "REID_DATA_DIR": str(data),
           "REID_MODEL_PATH": os.environ["MODEL_PATH"], "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    results = []

    def run(name, cwd, command, overrides=None):
        p = subprocess.run([sys.executable, "-B", *command], cwd=cwd,
                           env={**env, **(overrides or {})}, text=True, capture_output=True)
        log = args.out / f"{name}.txt"
        log.write_text(p.stdout + p.stderr)
        results.append({"name": name, "cwd": str(cwd), "command": [sys.executable, "-B", *command],
                        "overrides": overrides or {}, "exit": p.returncode, "log": log.name})
        (args.out / "commands.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print(name, p.returncode, flush=True)
        if p.returncode:
            raise AssertionError(log.read_text())
        return p.stdout

    baseline, postproc, service = (solution / name for name in ("baseline", "postproc", "service"))
    for split in ("val", "test"):
        for part in ("query", "gallery"):
            csv_path = solution / f"split/files/val_{part}.csv" if split == "val" else data / f"test_{part}.csv"
            variants = [("", [])] if split == "test" else [
                ("", []), ("_mask30", ["--mask-bottom", "0.30"]),
                ("_gray", ["--grayscale"]), ("_mask30gray", ["--mask-bottom", "0.30", "--grayscale"])]
            for suffix, flags in variants:
                command = ["scripts/extract_embeddings.py", "--csv", str(csv_path),
                           "--images-dir", str(images), "--model", env["MODEL_PATH"],
                           "--out", f"out/{split}_{part}{suffix}.npy", "--threads", "2", *flags]
                if not suffix:
                    command += ["--ids-out", f"out/{split}_{part}.ids"]
                run(f"extract-{split}-{part}{suffix}", baseline, command)
    run("baseline-eval", baseline, ["scripts/run_eval.py"])
    run("baseline-submission", baseline, ["scripts/make_submission.py", "--threshold", "0.34921352213815304"])
    for part in ("query", "gallery"):
        for ext in ("npy", "ids"):
            shutil.copy2(baseline / f"out/val_{part}.{ext}", postproc / "out")
    run("postproc-baseline", postproc, ["scripts/s01_verify_baseline.py"])
    run("postproc-extract", postproc, ["scripts/s03_extract.py", "--sets", "val", "--variants", "208,208f,256"])
    run("postproc-tta", postproc, ["scripts/s05b_make_tta_vectors.py", "--sets", "val"])
    run("postproc-final", postproc, ["scripts/s06_final_val.py"])
    run("postproc-cached-without-images", postproc,
        ["scripts/s03_extract.py", "--sets", "val", "--variants", "208f,256"],
        {"REID_DATA_DIR": str(fixture / "absent-data")})

    # Closed tests may use JPEG, PNG and complete filenames, not only organizer JPGs.
    mixed = data / "mixed"
    mixed.mkdir()
    for i, path in enumerate(sorted(images.iterdir())):
        suffix = (".jpg", ".jpeg", ".png", "")[i % 4]
        with Image.open(path) as im:
            im.save(mixed / (path.stem + suffix), format="PNG" if suffix in (".png", "") else "JPEG")
    comparisons = {}
    for mode, flags, overrides in (("rerank", [], {}), ("cosine-flag", ["--no-rerank"], {}),
                                   ("cosine-env", [], {"REID_RERANK": "0"})):
        dirs = []
        for version, cwd in (("before", args.before / "04-solution/service"), ("after", service)):
            out = args.out / f"batch-{version}-{mode}"
            dirs.append(out)
            run(f"batch-{version}-{mode}", cwd, ["-m", "app.batch", "--query", str(data / "test_query.csv"),
                "--gallery", str(data / "test_gallery.csv"), "--images-dir", str(mixed),
                "--out-dir", str(out), "--threads", "2", *flags], overrides)
        hashes = {}
        for filename in ("embeddings.npy", "submission.csv", "candidates.csv"):
            pair = [hashlib.sha256((d / filename).read_bytes()).hexdigest() for d in dirs]
            assert pair[0] == pair[1], (mode, filename, pair)
            hashes[filename] = pair[0]
        emb = np.load(dirs[1] / "embeddings.npy")
        assert emb.shape == (12, 512) and emb.dtype == np.float32 and np.isfinite(emb).all()
        assert np.max(np.abs(np.linalg.norm(emb, axis=1) - 1)) < 1e-6
        comparisons[mode] = hashes
    (args.out / "before-after.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    print("PASS: all commands, all supported image suffixes; batch outputs byte-identical", flush=True)


if __name__ == "__main__":
    main()
