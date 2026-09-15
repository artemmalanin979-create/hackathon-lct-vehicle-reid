#!/usr/bin/env python3
"""Независимая проверка: соответствие строк npy объектам CSV + инвариантности препроцессинга.

Кроп делаю СВОИМ кодом (numpy-срез arr[y:y+h, x:x+w]), не PIL.crop, чтобы поймать
перепутанные оси/трактовку w,h. Сверяю с val_query.npy / val_gallery.npy /
artifacts/embeddings.npy построчно (cos) и ближайшей строкой.
"""
import os
import csv, json, sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image

JOB = Path(__file__).resolve().parent.parent
REPO = Path(__file__).resolve().parents[3]  # корень репозитория
DATA = Path(os.environ.get("REID_DATA_DIR", REPO / "data"))
SPLIT = REPO / "04-solution/split/files"
MODEL = JOB / "subject" / "osnet_ain_x1_0_vehicle_reid.onnx"

def rd(p):
    with open(p, newline="") as f:
        return [(r["image_id"], int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))
                for r in csv.DictReader(f)]

sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
INP = sess.get_inputs()[0].name

def embed(pix_chw_batch):
    out = sess.run(None, {INP: np.asarray(pix_chw_batch, dtype=np.float32)})[0].astype(np.float64)
    return out / np.linalg.norm(out, axis=1, keepdims=True)

def my_crop(row, order="RGB", scale=1.0, mean=None, std=None, size=208):
    iid, x, y, w, h = row
    with Image.open(DATA / "images" / f"{iid}.jpg") as im:
        arr = np.asarray(im.convert("RGB"))          # H,W,3 RGB
    sub = arr[y:y+h, x:x+w]                          # numpy-срез, не PIL.crop
    pil = Image.fromarray(sub).resize((size, size), Image.BILINEAR)
    a = np.asarray(pil, dtype=np.float32)
    if order == "BGR":
        a = a[:, :, ::-1]
    a = a * scale
    if mean is not None:
        a = (a - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return a.transpose(2, 0, 1)

def check_block(tag, rows, mat, picks):
    res = []
    for i in picks:
        v = embed([my_crop(rows[i])])[0]
        cos_all = mat.astype(np.float64) @ v
        res.append((i, float(cos_all[i]), int(np.argmax(cos_all))))
    worst = min(r[1] for r in res)
    mis = [r for r in res if r[2] != r[0] and not np.isclose(r[1], cos_all[r[2]])]
    print(f"{tag}: n={len(picks)} min_cos_at_row={worst:.6f} nearest!=row: {len([r for r in res if r[2]!=r[0]])}")
    return res

rng = np.random.default_rng(777)
vq_rows, vg_rows = rd(SPLIT / "val_query.csv"), rd(SPLIT / "val_gallery.csv")
tq_rows, tg_rows = rd(DATA / "test_query.csv"), rd(DATA / "test_gallery.csv")
vq = np.load(JOB / "subject/out/val_query.npy"); vg = np.load(JOB / "subject/out/val_gallery.npy")
art = np.load(JOB / "subject/artifacts/embeddings.npy")

r1 = check_block("val_query.npy", vq_rows, vq, sorted(rng.choice(len(vq_rows), 12, replace=False).tolist()))
r2 = check_block("val_gallery.npy", vg_rows, vg, sorted(rng.choice(len(vg_rows), 12, replace=False).tolist()))
all_test = tq_rows + tg_rows
r3 = check_block("artifacts/embeddings.npy", all_test, art,
                 sorted(set([0, 1109, 1110, 1859]) | set(rng.choice(1860, 12, replace=False).tolist())))

# --- инвариантности препроцессинга на 6 кропах ---
sample = [vq_rows[i] for i in rng.choice(len(vq_rows), 6, replace=False)]
base = embed([my_crop(r) for r in sample])
div255 = embed([my_crop(r, scale=1/255.0) for r in sample])
imnet = embed([my_crop(r, scale=1/255.0, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) for r in sample])
bgr  = embed([my_crop(r, order="BGR") for r in sample])
def cmp(tag, a, b):
    cs = np.sum(a*b, axis=1)
    print(f"{tag}: cos min={cs.min():.6f} mean={cs.mean():.6f}")
cmp("x vs x/255", base, div255)
cmp("x vs imagenet-norm", base, imnet)
cmp("RGB vs BGR", base, bgr)
