#!/usr/bin/env python3
"""New portable adapter (job_73), not the historical preparation entry point.

Calls the existing prep_train_crops.work unchanged, serially; derives the raw
208 crops from those JPEGs as in reid_train.py::load_crops. No validation data.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    source = repo / "04-solution/training/scripts/prep_train_crops.py"
    spec = importlib.util.spec_from_file_location("original_prep", source)
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    prep.DATA = args.images_dir
    split = repo / "04-solution/split/files/train_fit.csv"
    with split.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 7248:
        raise SystemExit("Expected the published 7248-row train_fit split")
    missing = [r["image_id"] for r in rows if not (args.images_dir / (r["image_id"] + ".jpg")).is_file()]
    if missing:
        raise SystemExit(f"Missing {len(missing)} training images in {args.images_dir}: {missing[:10]}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    packed_path, raw_path = args.out_dir / "train_crops.npz", args.out_dir / "crops_208.npy"
    if packed_path.exists() or raw_path.exists():
        raise SystemExit("Use a new output directory; existing training arrays will not be overwritten")
    raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.uint8, shape=(len(rows), 208, 208, 3))
    blobs = []
    for index, row in enumerate(rows):
        _, blob = prep.work((index, row))
        blobs.append(blob)
        with Image.open(io.BytesIO(blob)) as image:
            raw[index] = np.asarray(image.convert("RGB").resize((208, 208), Image.BILINEAR), dtype=np.uint8)
        if index % 500 == 0:
            print(f"own {index}/{len(rows)}", flush=True)
    raw.flush()
    offsets = np.zeros(len(blobs) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(b) for b in blobs])
    np.savez(packed_path, data=np.frombuffer(b"".join(blobs), dtype=np.uint8), offsets=offsets,
             vehicle_id=np.array([int(r["vehicle_id"]) for r in rows], dtype=np.int64),
             camera_id=np.array([int(r["camera_id"]) for r in rows], dtype=np.int64),
             image_id=np.array([r["image_id"] for r in rows]),
             size=np.array([prep.SIZE]), quality=np.array([prep.QUALITY]))
    expected = {"train_crops.npz": "a910e8a3a10986845fb83f3a3fcd401bfc8213ac697d50864e36b8c84fefff9d",
                "crops_208.npy": "ea77290d1ad1ae6c245bc33e2ba19bc7c676a8185eda69f9028fe4d1b9e05057"}
    result = {}
    for path in (packed_path, raw_path):
        with path.open("rb") as stream:
            h = hashlib.file_digest(stream, "sha256").hexdigest()
        result[path.name] = {"sha256": h, "bytes": path.stat().st_size, "matches_historical": h == expected[path.name]}
    (args.out_dir / "prepare_own.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not all(r["matches_historical"] for r in result.values()):
        raise SystemExit("Prepared arrays differ from the historical hashes; inspect versions/inputs before training")


if __name__ == "__main__":
    main()
