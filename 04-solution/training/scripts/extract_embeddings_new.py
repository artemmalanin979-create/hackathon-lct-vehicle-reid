#!/usr/bin/env python3
"""Векторы дообученной модели по CSV с bbox — тем же путём, что у бейзлайна.

кроп по bbox -> RGB -> (вмешательство) -> resize 208x208 (PIL BILINEAR) -> float32 0..255
-> NCHW -> ONNX -> вектор (модель сама нормирует вход и L2-нормирует выход).

Отличие от baseline/scripts/extract_embeddings.py только в модели: нормировка
ImageNet и L2 зашиты внутрь графа, поэтому внешних преобразований не добавляется.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

SIZE = 208


def read_rows(csv_path: Path):
    with open(csv_path, newline="") as f:
        return [(r["image_id"], int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))
                for r in csv.DictReader(f)]


def load_crop(images_dir: Path, row) -> np.ndarray:
    image_id, x, y, w, h = row
    with Image.open(images_dir / f"{image_id}.jpg") as im:
        crop = im.convert("RGB").crop((x, y, x + w, y + h))
    crop = crop.resize((SIZE, SIZE), Image.BILINEAR)
    return np.asarray(crop, dtype=np.float32).transpose(2, 0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ids-out", type=Path, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--timing", type=Path, default=None)
    args = ap.parse_args()

    rows = read_rows(args.csv)
    o = ort.SessionOptions()
    o.log_severity_level = 3
    sess = ort.InferenceSession(str(args.model), sess_options=o,
                                providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    dim = sess.get_outputs()[0].shape[-1]
    out = np.empty((len(rows), int(dim)), dtype=np.float32)

    t0 = time.perf_counter()
    for s in range(0, len(rows), args.batch):
        chunk = rows[s:s + args.batch]
        batch = np.stack([load_crop(args.images_dir, r) for r in chunk])
        (emb,) = sess.run(None, {name: batch})
        out[s:s + len(chunk)] = emb
    elapsed = time.perf_counter() - t0

    m = out.astype(np.float64)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, m.astype(np.float32))
    if args.ids_out:
        args.ids_out.write_text("".join(r[0] + "\n" for r in rows))
    stats = {"csv": str(args.csv), "rows": len(rows), "dim": int(dim),
             "batch": args.batch, "elapsed_s": round(elapsed, 3),
             "obj_per_s": round(len(rows) / elapsed, 2),
             "ms_per_obj": round(elapsed / len(rows) * 1e3, 2)}
    if args.timing:
        args.timing.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
