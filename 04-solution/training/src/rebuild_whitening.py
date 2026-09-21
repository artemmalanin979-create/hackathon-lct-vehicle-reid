#!/usr/bin/env python3
"""New portable job_73 entry point; calls the historical lib45.learn_lw unchanged.

Inputs are the own train_fit raw crops used in training. The complete historical
candidate study s02_eval.py is preserved separately and also needs old experiments.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE / "postprocess"), str(REPO / "04-solution/eval"),
                str(REPO / "04-solution/postproc/scripts")]
from lib45 import learn_lw, l2n


def extract(path, crops):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    vectors = np.empty((len(crops), 512), dtype=np.float32)
    for start in range(0, len(crops), 16):
        x = np.asarray(crops[start:start+16], dtype=np.float32).transpose(0, 3, 1, 2)
        vectors[start:start+len(x)] = session.run(None, {session.get_inputs()[0].name: x})[0]
        if start % 800 == 0:
            print(path.name, start, len(crops), flush=True)
    # Historical extraction stores normalized f32, then lib45 normalizes in f64.
    return l2n(l2n(vectors).astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model1", type=Path, default=REPO / "04-solution/service/model/osnet_ain_x1_0_vehicle_reid.onnx")
    parser.add_argument("--model2", type=Path, default=REPO / "04-solution/service/model/osnet_ain_combined_v1.onnx")
    args = parser.parse_args()
    with args.crops.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != "ea77290d1ad1ae6c245bc33e2ba19bc7c676a8185eda69f9028fe4d1b9e05057":
            raise SystemExit("Expected the historical own crops_208.npy in train_fit order")
    with (REPO / "04-solution/split/files/train_fit.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    crops = np.load(args.crops, mmap_mode="r", allow_pickle=False)
    if crops.shape != (len(rows), 208, 208, 3) or len(rows) != 7248:
        raise SystemExit("train_fit/crops shape mismatch")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit("Use an empty output directory")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    first, second = extract(args.model1, crops), extract(args.model2, crops)
    lw = learn_lw(l2n(first + second), np.array([r["vehicle_id"] for r in rows]),
                  np.array([r["camera_id"] for r in rows]), 0.5)
    np.savez(args.out_dir / "lw_f64.npz", P=lw["P"], m=lw["m"])
    np.savez(args.out_dir / "lw_f32.npz", P=lw["P"].astype(np.float32), m=lw["m"].astype(np.float32))
    report = {k: v for k, v in lw.items() if k not in ("P", "m")}
    report["files"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out_dir.glob("*.npz")}
    (args.out_dir / "whitening.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
