"""Карта влияния для пары «запрос — кандидат»: чувствительность к перекрытию.

Закрываем окно во входе сети (208x208, то есть уже после resize — ровно то, что
модель видит), пересчитываем вектор и смотрим, насколько упала близость к
ВТОРОМУ изображению пары. Δ = cos(base) − cos(masked).

Δ > 0 — область держит совпадение; Δ < 0 — область мешает.

Параметры зафиксированы до просмотра результатов (см. journal.md):
окно 48, шаг 16 → 11x11 = 121 позиция на сторону; заливка — средний цвет кольца
вокруг окна (mask_ops.ring_color), тот же приём, что в абляции по пластине.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

REPO = Path(os.environ.get("REID_REPO", "/home/artem/projects/hackathon-lct-vehicle-reid"))
DATA = Path(os.environ.get("REID_DATA_DIR", REPO / "data"))
SPLIT = REPO / "04-solution/split/files"
MODEL = REPO / "04-solution/service/model/osnet_ain_x1_0_vehicle_reid.onnx"
sys.path.insert(0, str(REPO / "04-solution/plate-ablation/scripts"))
from mask_ops import ring_color, fill, controls  # noqa: E402

INPUT = 208
WIN = 48
STRIDE = 16
POS = list(range(0, INPUT - WIN + 1, STRIDE))  # 0..160 шаг 16 -> 11 позиций
NG = len(POS)

_SESSION = None


def session(threads: int = 0) -> ort.InferenceSession:
    global _SESSION
    if _SESSION is None:
        o = ort.SessionOptions()
        o.log_severity_level = 3
        if threads:
            o.intra_op_num_threads = threads
        _SESSION = ort.InferenceSession(str(MODEL), sess_options=o,
                                        providers=["CPUExecutionProvider"])
    return _SESSION


def read_split(name: str):
    with open(SPLIT / f"{name}.csv", newline="") as f:
        return list(csv.DictReader(f))


def crop_raw(row) -> np.ndarray:
    """Кроп по bbox в исходном разрешении, RGB uint8."""
    x, y, w, h = (int(row[k]) for k in ("x", "y", "w", "h"))
    with Image.open(DATA / "images" / f"{row['image_id']}.jpg") as im:
        return np.asarray(im.convert("RGB").crop((x, y, x + w, y + h)))


def net_input(row) -> np.ndarray:
    """Вход сети: кроп -> resize 208x208 BILINEAR -> uint8 HxWx3 (как в baseline)."""
    x, y, w, h = (int(row[k]) for k in ("x", "y", "w", "h"))
    with Image.open(DATA / "images" / f"{row['image_id']}.jpg") as im:
        c = im.convert("RGB").crop((x, y, x + w, y + h))
    return np.asarray(c.resize((INPUT, INPUT), Image.BILINEAR))


def embed(imgs, batch: int = 32) -> np.ndarray:
    """imgs: список uint8 HxWx3 -> L2-нормированные векторы N x 512 float32."""
    s = session()
    name = s.get_inputs()[0].name
    out = np.empty((len(imgs), 512), np.float32)
    for i in range(0, len(imgs), batch):
        chunk = imgs[i:i + batch]
        b = np.stack([a.astype(np.float32).transpose(2, 0, 1) for a in chunk])
        (e,) = s.run(None, {name: b})
        out[i:i + len(chunk)] = e
    v = out.astype(np.float64)
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def grid_pos(win: int = WIN, stride: int = STRIDE):
    return list(range(0, INPUT - win + 1, stride))


def masked_variants(img: np.ndarray, mode: str = "ring",
                    win: int = WIN, stride: int = STRIDE):
    """Копии входа с закрытым окном; порядок — построчно по сетке позиций."""
    out = []
    for gy in grid_pos(win, stride):
        for gx in grid_pos(win, stride):
            out.append(fill(img, (gx, gy, win, win), mode))
    return out


def windows_to_map(delta: np.ndarray, win: int = WIN,
                   stride: int = STRIDE) -> np.ndarray:
    """Δ по окнам (n*n,) -> карта 208x208: усреднение по перекрывающимся окнам."""
    acc = np.zeros((INPUT, INPUT), np.float64)
    cnt = np.zeros((INPUT, INPUT), np.float64)
    flat = delta.ravel()
    k = 0
    for gy in grid_pos(win, stride):
        for gx in grid_pos(win, stride):
            acc[gy:gy + win, gx:gx + win] += flat[k]
            cnt[gy:gy + win, gx:gx + win] += 1
            k += 1
    return acc / np.maximum(cnt, 1)


def pair_maps(q_img: np.ndarray, g_img: np.ndarray, mode: str = "ring",
              sides=("q", "g"), win: int = WIN, stride: int = STRIDE):
    """Карты влияния для пары. Возвращает dict с сетками Δ по окнам и базовым cos."""
    n = len(grid_pos(win, stride))
    base = embed([q_img, g_img])
    fq, fg = base[0], base[1]
    s0 = float(fq @ fg)
    res = {"cos": s0}
    if "q" in sides:
        vq = embed(masked_variants(q_img, mode, win, stride))
        res["wq"] = (s0 - (vq @ fg)).reshape(n, n)
    if "g" in sides:
        vg = embed(masked_variants(g_img, mode, win, stride))
        res["wg"] = (s0 - (vg @ fq)).reshape(n, n)
    return res


# ---------- скаляры, объявленные до просмотра результатов ----------

CENTERS = np.array([p + WIN // 2 for p in POS])          # 24..184
_IN60 = (CENTERS >= 0.2 * INPUT) & (CENTERS <= 0.8 * INPUT)
BORDER60 = ~(_IN60[:, None] & _IN60[None, :])            # центр вне центральных 60 %
RING1 = np.zeros((NG, NG), bool)                          # внешнее кольцо сетки
RING1[0, :] = RING1[-1, :] = RING1[:, 0] = RING1[:, -1] = True


def stats(w: np.ndarray) -> dict:
    """w: сетка Δ по окнам NG x NG."""
    flat = w.ravel()
    pos = np.clip(flat, 0, None)
    k = max(1, int(round(0.10 * flat.size)))
    top = np.sort(pos)[-k:]
    mean = float(flat.mean())
    b60 = float(flat[BORDER60.ravel()].mean())
    c60 = float(flat[~BORDER60.ravel()].mean())
    r1 = float(flat[RING1.ravel()].mean())
    return {
        "mean_drop": mean,
        "max_drop": float(flat.max()),
        "min_drop": float(flat.min()),
        "top10_share": float(top.sum() / pos.sum()) if pos.sum() > 0 else float("nan"),
        # рамка против центра — в АБСОЛЮТНЫХ единицах Δcos, не отношением:
        # средний Δ по карте бывает около нуля, и отношение к нему теряет смысл
        "border_mean": b60,
        "center_mean": c60,
        "border_minus_center": b60 - c60,
        "ring1_mean": r1,
        "ring1_minus_center": r1 - c60,
        "neg_share": float((flat < 0).mean()),
    }


def argmax_window(w: np.ndarray):
    """(x, y, w, h) самого горячего окна в координатах входа сети."""
    iy, ix = np.unravel_index(int(np.argmax(w)), w.shape)
    return (POS[ix], POS[iy], WIN, WIN), float(w[iy, ix])
