#!/usr/bin/env python3
"""Точные карты вклада по тем же парам + сверка с картами перекрытия.

Два независимых метода на одних и тех же парах:
  occlusion — причинный вопрос: что будет, если участка не станет;
  contrib   — учётный вопрос: сколько эта позиция вносит в саму оценку.
Согласие — проверка обоих; расхождение — находка.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

JOB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JOB / "scripts"))
import occl  # noqa: E402
import contrib  # noqa: E402


def to208(g):
    """Сетка 13x13 -> карта 208x208 (для сравнения с картой перекрытия)."""
    return np.asarray(Image.fromarray(g.astype(np.float32), mode="F")
                      .resize((occl.INPUT, occl.INPUT), Image.BILINEAR)).astype(np.float64)


def to_occl_grid(g):
    """Карта вклада 13x13, усреднённая ПО ТЕМ ЖЕ окнам 48/16, что и карта перекрытия.

    Честное сравнение: карта перекрытия сглажена окном 48 px по построению,
    поэтому и вклад надо смотреть через то же окно, иначе сравниваются разные
    масштабы, а не разные методы.
    """
    big = to208(g)
    out = np.empty((occl.NG, occl.NG))
    for iy, gy in enumerate(occl.POS):
        for ix, gx in enumerate(occl.POS):
            out[iy, ix] = big[gy:gy + occl.WIN, gx:gx + occl.WIN].mean()
    return out


def grid_stats(g):
    """Те же скаляры, что у карты перекрытия, но на сетке 13x13."""
    n = g.shape[0]
    cen = (np.arange(n) + 0.5) / n
    inner = (cen >= 0.2) & (cen <= 0.8)
    border = ~(inner[:, None] & inner[None, :])
    ring1 = np.zeros((n, n), bool)
    ring1[0, :] = ring1[-1, :] = ring1[:, 0] = ring1[:, -1] = True
    flat = g.ravel()
    pos = np.clip(flat, 0, None)
    k = max(1, int(round(0.10 * flat.size)))
    return {
        "mean": float(flat.mean()), "max": float(flat.max()), "min": float(flat.min()),
        "sum": float(flat.sum()),
        "top10_share": float(np.sort(pos)[-k:].sum() / pos.sum()) if pos.sum() > 0 else float("nan"),
        "border_mean": float(flat[border.ravel()].mean()),
        "center_mean": float(flat[~border.ravel()].mean()),
        "border_minus_center": float(flat[border.ravel()].mean() - flat[~border.ravel()].mean()),
        "ring1_minus_center": float(flat[ring1.ravel()].mean() - flat[~border.ravel()].mean()),
        "neg_share": float((flat < 0).mean()),
    }


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--set", default="cases", choices=["cases", "ref"])
args = ap.parse_args()

qm, gm = occl.read_split("val_query"), occl.read_split("val_gallery")
S = json.load(open(JOB / f"out/stats_{args.set}.json"))["rows"]
M = np.load(JOB / f"out/maps_{args.set}.npz")

maps, rows = {}, []
t0 = time.perf_counter()
for k, rec in enumerate(S):
    qim, gim = occl.net_input(qm[rec["qi"]]), occl.net_input(gm[rec["gi"]])
    r = contrib.pair_contrib(qim, gim)
    maps[f"{k}_cq"], maps[f"{k}_cg"] = r["cq"], r["cg"]
    row = {"case": k, "kind": rec["kind"], "qi": rec["qi"], "gi": rec["gi"],
           "cos": r["cos"], "bias_q": r["bq"], "bias_g": r["bg"],
           "sum_check_q": r["cq_sum_check"], "sum_check_g": r["cg_sum_check"],
           "q": grid_stats(r["cq"]), "g": grid_stats(r["cg"])}
    # согласие с картой перекрытия: корреляция на уровне пикселей 208x208
    for side, ck, ok_ in (("q", "cq", f"{k}_wq"), ("g", "cg", f"{k}_wg")):
        a = to208(r[ck]).ravel()
        b = occl.windows_to_map(M[ok_]).ravel()
        row[f"corr_occl_{side}"] = float(np.corrcoef(a, b)[0, 1])
        # то же, но через одинаковое окно 48/16 у обеих карт
        cg_ = to_occl_grid(r[ck])
        row[f"corr_win_{side}"] = float(np.corrcoef(cg_.ravel(), M[ok_].ravel())[0, 1])
        iy1, ix1 = np.unravel_index(int(np.argmax(cg_)), cg_.shape)
        iy2, ix2 = np.unravel_index(int(np.argmax(M[ok_])), M[ok_].shape)
        row[f"argmax_dist_{side}"] = float(np.hypot(
            occl.POS[ix1] - occl.POS[ix2], occl.POS[iy1] - occl.POS[iy2]))
    if rec.get("gi_alt") is not None:
        ra = contrib.pair_contrib(qim, occl.net_input(gm[rec["gi_alt"]]))
        maps[f"{k}_cq_alt"] = ra["cq"]
        row["cos_alt"] = ra["cos"]
        row["corr_alt"] = float(np.corrcoef(r["cq"].ravel(), ra["cq"].ravel())[0, 1])
    rn = contrib.pair_contrib(qim, occl.net_input(gm[rec["null_gi"]]))
    maps[f"{k}_cq_null"] = rn["cq"]
    row["cos_null"] = rn["cos"]
    row["corr_null"] = float(np.corrcoef(r["cq"].ravel(), rn["cq"].ravel())[0, 1])
    row["null"] = grid_stats(rn["cq"])
    rows.append(row)
    print(f"[{k+1}/{len(S)}] {rec['kind']:12s} cos={r['cos']:+.3f} "
          f"corr_occl q={row['corr_occl_q']:+.2f} g={row['corr_occl_g']:+.2f}", flush=True)

np.savez_compressed(JOB / f"out/contrib_{args.set}.npz", **maps)
(JOB / f"out/contrib_{args.set}.json").write_text(
    json.dumps({"seconds": round(time.perf_counter() - t0, 1), "rows": rows},
               ensure_ascii=False, indent=1))
for key, lab in (("corr_occl", "пиксели"), ("corr_win", "общее окно 48/16")):
    c = np.array([r[f"{key}_q"] for r in rows] + [r[f"{key}_g"] for r in rows])
    print(f"согласие двух методов ({lab}): медиана r = {np.median(c):+.3f}, "
          f"min {c.min():+.3f}, доля r>0.5: {(c > 0.5).mean():.2f}, "
          f"доля r>0: {(c > 0).mean():.2f}")
d = np.array([r["argmax_dist_q"] for r in rows] + [r["argmax_dist_g"] for r in rows])
print(f"расстояние между максимумами: медиана {np.median(d):.0f} px "
      f"(вход 208 px), доля <=48 px: {(d <= 48).mean():.2f}")
sc = np.abs([r["sum_check_q"] for r in rows] + [r["sum_check_g"] for r in rows])
print(f"невязка суммы вкладов: max {sc.max():.2e}")
print("всего", round(time.perf_counter() - t0, 1), "с")
