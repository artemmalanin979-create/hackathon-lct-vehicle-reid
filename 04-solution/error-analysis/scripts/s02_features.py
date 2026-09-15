#!/usr/bin/env python3
"""Признаки объектов (запросы и галерея), считаемые из кадра и bbox.

Каждый признак — проверяемая величина, вычисленная из пикселей/аннотации, без модели:
  area_frac  доля площади кадра, занятая bbox
  aspect     w/h bbox (прокси ракурса: фас/корма ~1.0-1.3, борт >1.7)
  edges      сколько сторон bbox упирается в край кадра (срез объекта)
  frameL     средняя яркость всего кадра 0..255 (прокси «ночь»)
  cropL      средняя яркость кропа
  sat        средняя насыщенность (HSV S) центральной части кропа
  lab_*      средний Lab центральной части кропа (цвет кузова без фона)
  sharp      средний модуль градиента кропа, приведённого к 128x128 (прокси резкости/зума)
"""
import os
import csv, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

JOB = Path(__file__).resolve().parent.parent
REPO = Path(__file__).resolve().parents[3]  # корень репозитория
P = REPO
IMG = P / "data/images"
SPLIT = P / "04-solution/split/files"


def rgb2lab(rgb):
    """rgb: (...,3) float 0..255 -> CIE Lab (D65)."""
    c = rgb / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1)


def feats_for(csv_path, tag):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    out = {k: np.zeros(n, dtype=np.float32) for k in
           ("area_frac", "aspect", "edges", "frameL", "cropL", "sat", "sharp",
            "labL", "laba", "labb", "W", "H", "bw", "bh")}
    t0 = time.perf_counter()
    for i, r in enumerate(rows):
        x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
        with Image.open(IMG / f"{r['image_id']}.jpg") as im:
            im = im.convert("RGB")
            W, H = im.size
            small = np.asarray(im.resize((160, 90), Image.BILINEAR), dtype=np.float32)
            crop = im.crop((x, y, x + w, y + h))
        out["W"][i], out["H"][i] = W, H
        out["bw"][i], out["bh"][i] = w, h
        out["area_frac"][i] = w * h / float(W * H)
        out["aspect"][i] = w / float(h)
        out["edges"][i] = ((x <= 2) + (y <= 2) + (x + w >= W - 2) + (y + h >= H - 2))
        out["frameL"][i] = (0.299 * small[..., 0] + 0.587 * small[..., 1] +
                            0.114 * small[..., 2]).mean()
        c128 = np.asarray(crop.resize((128, 128), Image.BILINEAR), dtype=np.float32)
        lum = 0.299 * c128[..., 0] + 0.587 * c128[..., 1] + 0.114 * c128[..., 2]
        out["cropL"][i] = lum.mean()
        gx = np.abs(np.diff(lum, axis=1)).mean()
        gy = np.abs(np.diff(lum, axis=0)).mean()
        out["sharp"][i] = 0.5 * (gx + gy)
        core = c128[32:96, 32:96]           # центральные 50% — кузов, меньше фона
        mx, mn = core.max(-1), core.min(-1)
        out["sat"][i] = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0).mean() * 255
        lab = rgb2lab(core.reshape(-1, 3)).mean(0)
        out["labL"][i], out["laba"][i], out["labb"][i] = lab
        if (i + 1) % 300 == 0:
            print(f"{tag} {i+1}/{n} {time.perf_counter()-t0:.0f}s", flush=True)
    np.savez(JOB / f"out/feats_{tag}.npz", **out,
             image_id=np.array([r["image_id"] for r in rows]),
             vehicle_id=np.array([int(r["vehicle_id"]) for r in rows]),
             camera_id=np.array([int(r["camera_id"]) for r in rows]))
    print(tag, "done", n, f"{time.perf_counter()-t0:.0f}s")


feats_for(SPLIT / "val_query.csv", "query")
feats_for(SPLIT / "val_gallery.csv", "gallery")
