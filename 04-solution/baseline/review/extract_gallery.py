#!/usr/bin/env python3
"""Одним проходом по картинкам val-сплита извлекаю варианты препроцессинга.

Варианты: base208 (контроль равенства их файлам), bgr208, flip208 (гориз. отражение),
res256, res288, letterbox208 (сохранение пропорций, паддинг 127).
Выход: work/emb/<split>_<variant>.npy
"""
import os
import csv, sys, time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image

JOB = Path(__file__).resolve().parent.parent
REPO = Path(__file__).resolve().parents[3]  # корень репозитория
DATA = Path(os.environ.get("REID_DATA_DIR", REPO / "data"))
SPLIT = REPO / "04-solution/split/files"
OUT = JOB / "work" / "emb"; OUT.mkdir(exist_ok=True)

opts = ort.SessionOptions(); opts.log_severity_level = 3
sess = ort.InferenceSession(str(JOB/"subject/osnet_ain_x1_0_vehicle_reid.onnx"),
                            sess_options=opts, providers=["CPUExecutionProvider"])
INP = sess.get_inputs()[0].name

def rd(p):
    with open(p, newline="") as f:
        return [(r["image_id"], int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))
                for r in csv.DictReader(f)]

def to_chw(pil):
    return np.asarray(pil, dtype=np.float32).transpose(2, 0, 1)

def letterbox(crop, size=208, fill=127):
    w, h = crop.size
    s = size / max(w, h)
    nw, nh = max(1, round(w*s)), max(1, round(h*s))
    r = crop.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    canvas.paste(r, ((size-nw)//2, (size-nh)//2))
    return canvas

VARIANTS = ["base208", "bgr208", "flip208", "res256", "res288", "letterbox208"]

def variants_of(crop):
    r208 = crop.resize((208, 208), Image.BILINEAR)
    a208 = to_chw(r208)
    out = {
        "base208": a208,
        "bgr208": a208[::-1].copy(),
        "flip208": to_chw(r208.transpose(Image.FLIP_LEFT_RIGHT)),
        "res256": to_chw(crop.resize((256, 256), Image.BILINEAR)),
        "res288": to_chw(crop.resize((288, 288), Image.BILINEAR)),
        "letterbox208": to_chw(letterbox(crop)),
    }
    return out

def run_split(name, rows, batch=24):
    accs = {v: np.empty((len(rows), 512), np.float32) for v in VARIANTS}
    t0 = time.perf_counter()
    for s in range(0, len(rows), batch):
        chunk = rows[s:s+batch]
        per_var = {v: [] for v in VARIANTS}
        for iid, x, y, w, h in chunk:
            with Image.open(DATA/"images"/f"{iid}.jpg") as im:
                crop = im.convert("RGB").crop((x, y, x+w, y+h))
            for v, arr in variants_of(crop).items():
                per_var[v].append(arr)
        for v in VARIANTS:
            emb = sess.run(None, {INP: np.stack(per_var[v])})[0]
            accs[v][s:s+len(chunk)] = emb
        if s % (batch*10) == 0:
            print(f"{name}: {s}/{len(rows)} elapsed={time.perf_counter()-t0:.0f}s", flush=True)
    for v in VARIANTS:
        m = accs[v].astype(np.float64)
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        np.save(OUT/f"{name}_{v}.npy", m.astype(np.float32))
    print(f"{name} done in {time.perf_counter()-t0:.0f}s", flush=True)

#skip query (уже извлечено)
run_split("val_gallery", rd(SPLIT/"val_gallery.csv"))

# контроль: полное сравнение base208 с их файлами
for nm, ref in [("val_query", "subject/out/val_query.npy"), ("val_gallery", "subject/out/val_gallery.npy")]:
    a = np.load(OUT/f"{nm}_base208.npy").astype(np.float64)
    b = np.load(JOB/ref).astype(np.float64)
    cos = np.sum(a*b, axis=1)
    print(f"base208 vs {ref}: min cos = {cos.min():.8f}, max|diff|={np.abs(a-b).max():.2e}")
