#!/usr/bin/env python3
"""Агрегаты по точным картам вклада + прямой тест «рамка кропа против середины».

Зачем: окно скользящей карты — 48 px (23 % стороны), оно физически не может
отделить полоску шириной в 8 % кропа. Карта вклада имеет сетку 13x13, то есть
ячейка = 16 px = 7,7 % стороны: её внешнее кольцо и есть рамка кропа.

Плюс причинный тест той же мысли: закрываем ВСЮ рамку (наружные 8 % с каждой
стороны, 29,4 % площади) и, для сравнения, центральный квадрат ТОЙ ЖЕ площади.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

JOB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JOB / "scripts"))
import occl  # noqa: E402
from mask_ops import ring_color  # noqa: E402

RNG = np.random.default_rng(20260916)
B = 10000
FRAME = 0.08                      # доля стороны на рамку


def ci(x):
    x = np.asarray(x, float)
    idx = RNG.integers(0, x.size, (B, x.size))
    bs = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def cells_overlapping(box_in, n):
    """Ячейки сетки n x n входа 208x208, пересекающие бокс (в координатах входа)."""
    s = occl.INPUT / n
    bx, by, bw, bh = box_in
    m = np.zeros((n, n), bool)
    for iy in range(n):
        for ix in range(n):
            x0, y0 = ix * s, iy * s
            if not (x0 + s <= bx or bx + bw <= x0 or y0 + s <= by or by + bh <= y0):
                m[iy, ix] = True
    return m


def box_to_input(box, shape):
    H, W = shape
    x, y, w, h = box
    return (x * occl.INPUT / W, y * occl.INPUT / H,
            w * occl.INPUT / W, h * occl.INPUT / H)


qm, gm = occl.read_split("val_query"), occl.read_split("val_gallery")
S = {s: json.load(open(JOB / f"out/stats_{s}.json"))["rows"] for s in ("cases", "ref")}
CM = {s: np.load(JOB / f"out/contrib_{s}.npz") for s in ("cases", "ref")}
CJ = {s: json.load(open(JOB / f"out/contrib_{s}.json"))["rows"] for s in ("cases", "ref")}
PL = {s: {(r["case"], r["side"]): r for r in json.load(open(JOB / f"out/plate_{s}.json"))["rows"]}
      for s in ("cases", "ref")}

out = {}

# ---- A. рамка кропа против середины, по карте вклада 13x13 ------------------
ring, inner, kinds = [], [], []
for st in ("cases", "ref"):
    for k, rec in enumerate(S[st]):
        for side in ("q", "g"):
            g = CM[st][f"{k}_c{side}"]
            n = g.shape[0]
            m = np.zeros((n, n), bool)
            m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
            ring.append(float(g[m].mean()))
            inner.append(float(g[~m].mean()))
            kinds.append(rec["kind"])
ring, inner = np.array(ring), np.array(inner)
m, lo, hi = ci(ring - inner)
out["contrib_frame_vs_inner"] = {
    "n": len(ring), "cell_px": occl.INPUT // 13,
    "frame_mean": float(ring.mean()), "inner_mean": float(inner.mean()),
    "diff": m, "ci": [lo, hi], "share_frame_hotter": float((ring > inner).mean()),
    "ratio": float(ring.mean() / inner.mean()) if inner.mean() else None,
}

# ---- B. зона пластины по карте вклада 13x13 ---------------------------------
pp, aa, ss = [], [], []
for st in ("cases", "ref"):
    for k, rec in enumerate(S[st]):
        for side in ("q", "g"):
            p = PL[st].get((k, side))
            if not p or not p["found"]:
                continue
            g = CM[st][f"{k}_c{side}"]
            n = g.shape[0]
            mb = cells_overlapping(box_to_input(p["box"], p["crop_hw"]), n)
            if not mb.any():
                continue
            pp.append(float(g[mb].mean()))
            aa.append(float(g.mean()))
            ss.append(float(mb.sum()))
m, lo, hi = ci(np.array(pp) - np.array(aa))
out["contrib_plate"] = {"n": len(pp), "plate_mean": float(np.mean(pp)),
                        "map_mean": float(np.mean(aa)), "diff": m, "ci": [lo, hi],
                        "share_hotter": float((np.array(pp) > np.array(aa)).mean()),
                        "cells_med": float(np.median(ss)),
                        "cells_share": float(np.median(ss) / 169)}

# ---- C. причинный тест: закрыть рамку против закрыть центр той же площади ----
f = FRAME
side_in = int(round(occl.INPUT * (1 - 2 * f)))          # сторона внутреннего окна
area = occl.INPUT ** 2 - side_in ** 2                    # площадь рамки
cs = int(round(np.sqrt(area)))                           # сторона центрального квадрата
c0 = (occl.INPUT - cs) // 2
rows = []
t0 = time.perf_counter()
for st in ("cases", "ref"):
    for k, rec in enumerate(S[st]):
        qim, gim = occl.net_input(qm[rec["qi"]]), occl.net_input(gm[rec["gi"]])
        base = occl.embed([qim, gim])
        cos0 = float(base[0] @ base[1])
        for side, img in (("q", qim), ("g", gim)):
            partner = base[1] if side == "q" else base[0]
            fr = img.copy()
            b = int(round(occl.INPUT * f))
            col = ring_color(img, (b, b, occl.INPUT - 2 * b, occl.INPUT - 2 * b), pad=0.0)
            fr[:b, :] = fr[-b:, :] = fr[:, :b] = fr[:, -b:] = col
            ce = img.copy()
            ce[c0:c0 + cs, c0:c0 + cs] = ring_color(img, (c0, c0, cs, cs))
            e = occl.embed([fr, ce])
            rows.append({"set": st, "case": k, "kind": rec["kind"], "side": side,
                         "d_frame": cos0 - float(e[0] @ partner),
                         "d_center": cos0 - float(e[1] @ partner)})
df = np.array([r["d_frame"] for r in rows])
dc = np.array([r["d_center"] for r in rows])
m, lo, hi = ci(dc - df)
out["frame_vs_center_causal"] = {
    "n": len(rows), "frame_px": int(round(occl.INPUT * f)),
    "frame_area_frac": float(area / occl.INPUT ** 2),
    "center_side_px": cs, "center_area_frac": float(cs ** 2 / occl.INPUT ** 2),
    "d_frame_mean": float(df.mean()), "d_center_mean": float(dc.mean()),
    "center_minus_frame": m, "ci": [lo, hi],
    "share_center_costlier": float((dc > df).mean()),
    "seconds": round(time.perf_counter() - t0, 1),
}

# ---- D. концентрация и негативный контроль на картах вклада ------------------
t10p = np.array([r[s]["top10_share"] for st in ("cases", "ref") for r in CJ[st] for s in ("q", "g")])
t10n = np.array([r["null"]["top10_share"] for st in ("cases", "ref") for r in CJ[st]])
cn = np.array([r["corr_null"] for st in ("cases", "ref") for r in CJ[st]])
ca = np.array([r["corr_alt"] for r in CJ["cases"] if "corr_alt" in r])
out["contrib_concentration"] = {"top10_pair": float(t10p.mean()),
                                "top10_null": float(t10n.mean())}
m, lo, hi = ci(cn)
out["contrib_corr_null"] = {"n": len(cn), "mean": m, "ci": [lo, hi],
                            "median": float(np.median(cn))}
if ca.size:
    m, lo, hi = ci(ca)
    out["contrib_corr_alt"] = {"n": len(ca), "mean": m, "ci": [lo, hi],
                               "median": float(np.median(ca))}
# согласие двух методов
for key in ("corr_occl", "corr_win"):
    c = np.array([r[f"{key}_{s}"] for st in ("cases", "ref") for r in CJ[st] for s in ("q", "g")])
    m, lo, hi = ci(c)
    out[f"agree_{key}"] = {"n": len(c), "mean": m, "ci": [lo, hi],
                           "median": float(np.median(c)), "min": float(c.min()),
                           "share_pos": float((c > 0).mean())}
d = np.array([r[f"argmax_dist_{s}"] for st in ("cases", "ref") for r in CJ[st] for s in ("q", "g")])
out["agree_argmax"] = {"median_px": float(np.median(d)),
                       "share_within_48px": float((d <= 48).mean())}

(JOB / "out/contrib_aggregate.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
