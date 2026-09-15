"""Вмешательства в кроп: заливка/обесцвечивание прямоугольника и выбор контролей.

Контроль равной площади: тот же по размеру прямоугольник, помещённый туда, где
пластины нет. Размер и форма совпадают побитово, значит совпадает и число
потерянных пикселей, и вклад в статистики InstanceNormalization на входе сети.
"""
from __future__ import annotations
import numpy as np


def ring_color(arr, box, pad=0.5):
    """Средний цвет кольца вокруг бокса (нейтральная замена вместо серого 127).

    Первый узел сети — InstanceNormalization, то есть заливка фиксированным
    цветом сдвигает статистики входа тем сильнее, чем сильнее она отличается от
    окружения. Цвет кольца — наименее возмущающая заливка: средняя яркость и
    цветность кропа почти не меняются, искусственная граница минимальна.
    """
    H, W = arr.shape[:2]
    x, y, w, h = box
    px, py = max(2, int(w * pad)), max(2, int(h * pad))
    X0, Y0 = max(0, x - px), max(0, y - py)
    X1, Y1 = min(W, x + w + px), min(H, y + h + py)
    outer = arr[Y0:Y1, X0:X1].reshape(-1, 3)
    m = np.ones((Y1 - Y0, X1 - X0), bool)
    m[max(0, y - Y0):max(0, y - Y0) + h, max(0, x - X0):max(0, x - X0) + w] = False
    ring = outer[m.ravel()]
    if ring.size == 0:
        return np.array([127, 127, 127], np.uint8)
    return ring.mean(axis=0).round().astype(np.uint8)


def fill(arr, box, mode):
    """mode: 'ring' | 'gray127' | 'desat' (обесцветить только этот прямоугольник)."""
    x, y, w, h = box
    out = arr.copy()
    sl = (slice(y, y + h), slice(x, x + w))
    if mode == "ring":
        out[sl] = ring_color(arr, box)
    elif mode == "gray127":
        out[sl] = 127
    elif mode == "desat":
        patch = out[sl].astype(np.float64)
        lum = patch @ np.array([0.299, 0.587, 0.114])
        out[sl] = np.repeat(lum[:, :, None], 3, axis=2).round().clip(0, 255).astype(np.uint8)
    else:
        raise ValueError(mode)
    return out


def _fits(b, H, W):
    return b[0] >= 0 and b[1] >= 0 and b[0] + b[2] <= W and b[1] + b[3] <= H


def _overlap(a, b):
    return not (a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or
                a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1])


def controls(box, shape, seed):
    """4 контроля той же площади и формы: вверх/вниз, зеркало, вбок, случайно x2."""
    H, W = shape
    x, y, w, h = box
    res = {}

    # 1) сдвиг вдоль вертикали на 1.6 высоты бокса (бампер/решётка/капот вместо пластины)
    d = int(round(1.6 * h))
    up, down = (x, y - d, w, h), (x, y + d, w, h)
    cand = [c for c in (up, down) if _fits(c, H, W)]
    if cand:
        # из подходящих берём ту, что дальше от края кадра
        res["shift"] = max(cand, key=lambda c: min(c[1], H - c[1] - c[3]))
    else:
        res["shift"] = (x, max(0, min(H - h, y - d)), w, h)

    # 2) зеркало по вертикали относительно центра кропа
    my = H - y - h
    mir = (x, my, w, h)
    res["mirror"] = mir if _fits(mir, H, W) and not _overlap(mir, box) else res["shift"]

    # 3) сдвиг вбок на 1.3 ширины
    dx = int(round(1.3 * w))
    cands = [c for c in ((x - dx, y, w, h), (x + dx, y, w, h)) if _fits(c, H, W)]
    res["side"] = (max(cands, key=lambda c: min(c[0], W - c[0] - c[2]))
                   if cands else res["shift"])

    # 4) случайное место без пересечения с пластиной
    rng = np.random.default_rng(seed)
    for k in (1, 2):
        pick = None
        for _ in range(200):
            c = (int(rng.integers(0, max(1, W - w))), int(rng.integers(0, max(1, H - h))), w, h)
            if _fits(c, H, W) and not _overlap(c, box):
                pick = c; break
        res[f"rand{k}"] = pick if pick else res["shift"]
    return res
