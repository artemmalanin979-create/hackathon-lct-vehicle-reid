"""Локализация номерной пластины по следу пикселизации.

Организатор заменяет пластину мозаикой: прямоугольные ячейки постоянного цвета.
Ключевой признак — не «гладкость» (гладкие панели кузова тоже гладкие) и не
«резкие края» (текстура тоже резкая), а РАЗМЕР ячейки постоянного цвета:

  Rx, Ry — длины максимальных почти-постоянных (|ΔI| <= 1) отрезков по строке и столбцу,
  проходящих через пиксель. В мозаике Rx x Ry = размер ячейки (единицы-десятки пикселей),
  в текстуре Rx*Ry мал, в большой однородной области (небо, асфальт, борт) — огромен.

  «mosaic-like» пиксель: Rx,Ry в ОГРАНИЧЕННОМ диапазоне и площадь ячейки >= 40 px.

Счёт окна = density(mosaic-like)^3 * std(яркость): пластина одновременно плотно
покрыта ячейками и контрастна (светлое поле знака против тёмных символов).
Куб у density — чтобы плотность доминировала над контрастом: контрастных мест много,
а плотно-ячеистых мало.

Этапы: плотная карта по набору форм окна -> лучшее окно -> уточнение границ по маске
ячеек -> проверки структуры (>=2 ступеньки по обеим осям) -> отсев по абсолютному счёту.
"""
from __future__ import annotations
import numpy as np

TOL = 1.0            # допуск «тот же цвет» внутри ячейки
CELL_MIN_X, CELL_MAX_X = 4, 90
CELL_MIN_Y, CELL_MAX_Y = 3, 45
CELL_MIN_AREA = 40


def runlen(g, axis, tol=TOL):
    a = g if axis == 1 else g.T
    H, W = a.shape
    brk = np.ones((H, W), bool)
    brk[:, 1:] = np.abs(np.diff(a, axis=1)) > tol
    idx = np.cumsum(brk.ravel()) - 1
    R = np.bincount(idx)[idx].reshape(H, W).astype(np.float64)
    return R if axis == 1 else R.T


def mosaic_mask(g):
    """Маска «мозаичных» пикселей + карты размеров ячейки Rx, Ry."""
    Rx, Ry = runlen(g, 1), runlen(g, 0)
    M = ((Rx >= CELL_MIN_X) & (Rx <= CELL_MAX_X) &
         (Ry >= CELL_MIN_Y) & (Ry <= CELL_MAX_Y) &
         (Rx * Ry >= CELL_MIN_AREA)).astype(np.float64)
    return M, Rx, Ry


def _integ(a):
    return np.pad(np.cumsum(np.cumsum(a, 0), 1), ((1, 0), (1, 0)))


def _bm(ii, h, w):
    return ii[h:, w:] - ii[:-h, w:] - ii[h:, :-w] + ii[:-h, :-w]


def _shapes(H, W):
    out = set()
    for frac in (0.05, 0.07, 0.10, 0.14, 0.19, 0.26, 0.34):
        w = int(round(W * frac))
        for ar in (1.2, 1.8, 2.6, 3.6, 4.8):
            h = int(round(w / ar))
            if 5 <= h < H and 8 <= w < W:
                out.add((h, w))
    return sorted(out)


def _grads(g):
    gx = np.zeros_like(g); gx[:, :-1] = np.abs(np.diff(g, axis=1))
    gy = np.zeros_like(g); gy[:-1, :] = np.abs(np.diff(g, axis=0))
    return gx, gy


def _npeaks(prof, min_abs=3.0, rel=0.3):
    if prof.size < 3:
        return 0
    thr = max(min_abs, rel * float(prof.max()))
    s = np.flatnonzero(prof >= thr)
    if s.size == 0:
        return 0
    return 1 + int((np.diff(s) > 1).sum())


def grow(M, Rx, Ry, box, shape, max_area=0.14):
    """Наращивание бокса по согласованности размера ячейки.

    Мозаика — регулярная сетка: у соседней полосы тот же размер ячейки. Гладкий
    бампер рядом тоже «плоский», но размер его постоянных отрезков другой,
    поэтому рост останавливается на границе пластины, а не растекается по кузову.
    """
    H, W = shape
    x, y, w, h = box
    sl = (slice(y, y + h), slice(x, x + w))
    cw = float(np.median(Rx[sl])); ch = float(np.median(Ry[sl]))
    sx = max(2, int(round(cw / 2))); sy = max(1, int(round(ch / 2)))
    alive = {"l": True, "r": True, "u": True, "d": True}
    for _ in range(80):
        if not any(alive.values()) or w * h > max_area * W * H:
            break
        for d in ("l", "r", "u", "d"):
            if not alive[d]:
                continue
            if d == "l":
                nx = max(0, x - sx); strip = (slice(y, y + h), slice(nx, x))
            elif d == "r":
                nx2 = min(W, x + w + sx); strip = (slice(y, y + h), slice(x + w, nx2))
            elif d == "u":
                ny = max(0, y - sy); strip = (slice(ny, y), slice(x, x + w))
            else:
                ny2 = min(H, y + h + sy); strip = (slice(y + h, ny2), slice(x, x + w))
            sub = M[strip]
            if sub.size == 0:
                alive[d] = False; continue
            mrx = float(np.median(Rx[strip])); mry = float(np.median(Ry[strip]))
            ok = (sub.mean() >= 0.5 and 0.4 * cw <= mrx <= 2.5 * cw
                  and 0.4 * ch <= mry <= 2.5 * ch)
            if not ok:
                alive[d] = False; continue
            if d == "l":
                w += x - nx; x = nx
            elif d == "r":
                w = nx2 - x
            elif d == "u":
                h += y - ny; y = ny
            else:
                h = ny2 - y
    return (x, y, w, h)


def detect(rgb, topn=1, return_map=False):
    """rgb uint8 HxWx3 -> [(x,y,w,h,score,feat)], лучший первый."""
    g = rgb.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    H, W = g.shape
    M, Rx, Ry = mosaic_mask(g)
    iiM, iiV, iiV2 = _integ(M), _integ(g), _integ(g * g)
    best = []
    for (h, w) in _shapes(H, W):
        n = h * w
        dens = _bm(iiM, h, w) / n
        m1 = _bm(iiV, h, w) / n
        std = np.sqrt(np.maximum(_bm(iiV2, h, w) / n - m1 * m1, 0.0))
        S = dens ** 3 * std
        k = min(8, S.size)
        idx = np.argpartition(S.ravel(), -k)[-k:]
        ys, xs = np.unravel_index(idx, S.shape)
        for yy, xx in zip(ys, xs):
            if S[yy, xx] > 0:
                best.append([float(S[yy, xx]), int(xx), int(yy), int(w), int(h)])
    best.sort(key=lambda c: -c[0])
    out = []
    seen = []
    gx, gy = _grads(g)
    for s, x, y, w, h in best[:120]:
        bx, by, bw, bh = grow(M, Rx, Ry, (x, y, w, h), (H, W))
        cand = (bx, by, bw, bh)
        if any(_iou(cand, o) > 0.35 for o in seen):
            continue
        seen.append(cand)
        win = g[by:by + bh, bx:bx + bw]
        if win.size < 40:
            continue
        dens = float(M[by:by + bh, bx:bx + bw].mean())
        std = float(win.std())
        nx = _npeaks(gx[by:by + bh, bx:bx + bw - 1].mean(axis=0)) if bw > 3 else 0
        ny = _npeaks(gy[by:by + bh - 1, bx:bx + bw].mean(axis=1)) if bh > 3 else 0
        # площадь в счёте: из двух согласованно-мозаичных областей пластина — большая
        score = dens ** 3 * std * (bw * bh) ** 0.5
        feat = dict(dens=dens, std=std, nx=nx, ny=ny, w=bw, h=bh,
                    ar=bw / max(bh, 1), area_frac=bw * bh / (W * H), prop=s)
        ok = (dens >= 0.55 and std >= 15 and 0.8 <= bw / max(bh, 1) <= 9.0
              and 0.0004 <= bw * bh / (W * H) <= 0.10)
        feat["ok"] = bool(ok)
        out.append([score if ok else 0.0, bx, by, bw, bh, feat])
    out = [o for o in out if o[0] > 0]
    out.sort(key=lambda b: -b[0])
    res = out[:topn]
    return (res, M) if return_map else res


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)
