#!/usr/bin/env python3
"""Векторы дообученной модели для всех вариантов вмешательства в зону пластины.

Схема — та же, что в 04-solution/plate-ablation/scripts/extract_variants.py:
маска накладывается на кроп ДО resize, контроли равной площади берутся готовой
функцией `mask_ops.controls` (импорт, код не меняется). Отличие только в модели
и в размере входа (208x208, как у бейзлайна).
"""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image

ABL = Path("/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/plate-ablation/scripts")
sys.path.insert(0, str(ABL))
from mask_ops import fill, controls  # noqa: E402

DATA = Path("/home/artem/projects/hackathon-lct-vehicle-reid/data")
SPLIT = Path("/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/split/files")
JOB = Path(__file__).resolve().parent.parent
SIZE = 208
PAD = 0.12

VARIANTS = ["base",
            "plate_ring", "shift_ring", "mirror_ring", "side_ring", "rand1_ring", "rand2_ring",
            "platepad_ring", "shiftpad_ring",
            "plate_gray127", "shift_gray127",
            "plate_desat", "shift_desat"]


def pad_box(b, shape, f=PAD):
    H, W = shape
    x, y, w, h = b
    dx, dy = int(round(w * f)), int(round(h * f))
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(W, x + w + dx), min(H, y + h + dy)
    return (x0, y0, x1 - x0, y1 - y0)


def make_variants(arr, box, seed):
    out = {"base": arr}
    if box is None:
        for v in VARIANTS[1:]:
            out[v] = arr
        return out
    H, W = arr.shape[:2]
    ctl = controls(box, (H, W), seed)
    pbox = pad_box(box, (H, W))
    pctl = controls(pbox, (H, W), seed)
    out["plate_ring"] = fill(arr, box, "ring")
    out["shift_ring"] = fill(arr, ctl["shift"], "ring")
    out["mirror_ring"] = fill(arr, ctl["mirror"], "ring")
    out["side_ring"] = fill(arr, ctl["side"], "ring")
    out["rand1_ring"] = fill(arr, ctl["rand1"], "ring")
    out["rand2_ring"] = fill(arr, ctl["rand2"], "ring")
    out["platepad_ring"] = fill(arr, pbox, "ring")
    out["shiftpad_ring"] = fill(arr, pctl["shift"], "ring")
    out["plate_gray127"] = fill(arr, box, "gray127")
    out["shift_gray127"] = fill(arr, ctl["shift"], "gray127")
    out["plate_desat"] = fill(arr, box, "desat")
    out["shift_desat"] = fill(arr, ctl["shift"], "desat")
    return out


def to_input(a):
    im = Image.fromarray(a).resize((SIZE, SIZE), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32).transpose(2, 0, 1)


def main():
    """Аргументы: МОДЕЛЬ ТЕГ СПЛИТ [START END] — кусками, чтобы одна сессия была короткой."""
    model = Path(sys.argv[1])
    tag = sys.argv[2]
    split = sys.argv[3]
    outdir = JOB / "work" / f"emb_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    boxes = json.load(open(JOB / "work" / "boxes.json"))
    o = ort.SessionOptions(); o.log_severity_level = 3
    sess = ort.InferenceSession(str(model), sess_options=o, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    dim = int(sess.get_outputs()[0].shape[-1])

    rows = [(r["image_id"], int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))
            for r in csv.DictReader(open(SPLIT / f"{split}.csv", newline=""))]
    start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    end = min(int(sys.argv[5]), len(rows)) if len(sys.argv) > 5 else len(rows)
    bx = boxes[split]
    acc = {v: np.empty((end - start, dim), np.float32) for v in VARIANTS}
    t0 = time.perf_counter(); B = 12
    for s in range(start, end, B):
        chunk = rows[s:min(s + B, end)]
        per = {v: [] for v in VARIANTS}
        for j, (iid, x, y, w, h) in enumerate(chunk):
            with Image.open(DATA / "images" / f"{iid}.jpg") as im:
                a = np.asarray(im.convert("RGB").crop((x, y, x + w, y + h)))
            b = bx[s + j]
            box = tuple(b[:4]) if b else None
            vs = make_variants(a, box, seed=1000 + s + j)
            for v in VARIANTS:
                per[v].append(to_input(vs[v]))
        for v in VARIANTS:
            acc[v][s - start:s - start + len(chunk)] = sess.run(None, {inp: np.stack(per[v])})[0]
        if (s - start) % 120 == 0:
            print(f"  {split} {s}/{end} {time.perf_counter()-t0:.0f}s", flush=True)
    for v in VARIANTS:
        np.save(outdir / f"{split}_{v}_{start:05d}.npy", acc[v])
    print(f"{split}[{start}:{end}] x {len(VARIANTS)} вариантов за "
          f"{time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
