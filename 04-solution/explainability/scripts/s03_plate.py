#!/usr/bin/env python3
"""Соотнесение карты влияния с абляцией по номерной пластине.

Два независимых показания на одних и тех же парах:

1. ТОЧЕЧНЫЙ ТЕСТ (протокол абляции, но читается по паре, а не по mAP):
   закрываем ровно бокс пластины в исходном кропе (до resize) и те же 4 контроля
   равной площади из mask_ops.controls (сдвиг/зеркало/вбок/случайно x2), плюс
   расширенный на 12 % бокс и его контроль-сдвиг. Читаем падение близости к
   партнёру по паре: Δ = cos(base) − cos(вариант).

2. ЧТЕНИЕ СКОЛЬЗЯЩЕЙ КАРТЫ: средний Δ по окнам карты, перекрывающим бокс
   пластины (бокс переведён в координаты входа 208x208), против среднего Δ по
   всей карте и против окон, перекрывающих контроль-сдвиг.

Если пластина «горячая» хоть по одному показанию — это противоречие с абляцией,
и о нём надо сказать.
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
from mask_ops import fill, controls  # noqa: E402
sys.path.insert(0, str(occl.REPO / "04-solution/plate-ablation/scripts"))
from plate_detect import detect  # noqa: E402
from extract_variants import pad_box  # noqa: E402

VARIANTS = ["plate_ring", "shift_ring", "mirror_ring", "side_ring",
            "rand1_ring", "rand2_ring", "platepad_ring", "shiftpad_ring",
            "plate_gray127", "shift_gray127"]


def to_input(a):
    return np.asarray(Image.fromarray(a).resize((208, 208), Image.BILINEAR))


def make_variants(arr, box, seed):
    H, W = arr.shape[:2]
    ctl = controls(box, (H, W), seed)
    pbox = pad_box(box, (H, W))
    pctl = controls(pbox, (H, W), seed)
    return {
        "plate_ring": fill(arr, box, "ring"),
        "shift_ring": fill(arr, ctl["shift"], "ring"),
        "mirror_ring": fill(arr, ctl["mirror"], "ring"),
        "side_ring": fill(arr, ctl["side"], "ring"),
        "rand1_ring": fill(arr, ctl["rand1"], "ring"),
        "rand2_ring": fill(arr, ctl["rand2"], "ring"),
        "platepad_ring": fill(arr, pbox, "ring"),
        "shiftpad_ring": fill(arr, pctl["shift"], "ring"),
        "plate_gray127": fill(arr, box, "gray127"),
        "shift_gray127": fill(arr, ctl["shift"], "gray127"),
    }, ctl, pbox


def box_to_input(box, shape):
    """Бокс из координат кропа в координаты входа сети 208x208."""
    H, W = shape
    x, y, w, h = box
    kx, ky = occl.INPUT / W, occl.INPUT / H
    return (x * kx, y * ky, w * kx, h * ky)


def windows_overlapping(box_in):
    """Маска окон сетки (NG x NG), пересекающих бокс в координатах входа."""
    bx, by, bw, bh = box_in
    m = np.zeros((occl.NG, occl.NG), bool)
    for iy, gy in enumerate(occl.POS):
        for ix, gx in enumerate(occl.POS):
            if not (gx + occl.WIN <= bx or bx + bw <= gx or
                    gy + occl.WIN <= by or by + bh <= gy):
                m[iy, ix] = True
    return m


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--set", default="cases", choices=["cases", "ref"])
args = ap.parse_args()

qm, gm = occl.read_split("val_query"), occl.read_split("val_gallery")
stats = json.load(open(JOB / f"out/stats_{args.set}.json"))["rows"]
maps = np.load(JOB / f"out/maps_{args.set}.npz")

rows = []
t0 = time.perf_counter()
for k, rec in enumerate(stats):
    qi, gi = rec["qi"], rec["gi"]
    imgs = {"q": (qm[qi], 1000 + qi, f"{k}_wq"), "g": (gm[gi], 1000 + gi, f"{k}_wg")}
    base = occl.embed([occl.net_input(qm[qi]), occl.net_input(gm[gi])])
    for side, (row, seed, mapkey) in imgs.items():
        partner = base[1] if side == "q" else base[0]
        arr = occl.crop_raw(row)
        det = detect(arr, topn=1)
        if not det:
            rows.append({"case": k, "kind": rec["kind"], "side": side, "found": False})
            continue
        s, bx, by, bw, bh, feat = det[0]
        box = (int(bx), int(by), int(bw), int(bh))
        vs, ctl, pbox = make_variants(arr, box, seed)
        names = list(vs)
        e = occl.embed([to_input(vs[n]) for n in names])
        cos0 = float(base[0] @ base[1])
        d = {n: cos0 - float(e[j] @ partner) for j, n in enumerate(names)}

        w = maps[mapkey]
        mb = windows_overlapping(box_to_input(box, arr.shape[:2]))
        ms = windows_overlapping(box_to_input(ctl["shift"], arr.shape[:2]))
        rows.append({
            "case": k, "kind": rec["kind"], "side": side, "found": True,
            "qi": qi, "gi": gi, "cos": cos0,
            "box": box, "crop_hw": [int(arr.shape[0]), int(arr.shape[1])],
            "box_area_frac": float(bw * bh / (arr.shape[0] * arr.shape[1])),
            "delta": {n: round(v, 5) for n, v in d.items()},
            "plate_minus_shift": round(d["plate_ring"] - d["shift_ring"], 5),
            "plate_minus_ctl_mean": round(
                d["plate_ring"] - np.mean([d["shift_ring"], d["mirror_ring"],
                                           d["side_ring"], d["rand1_ring"],
                                           d["rand2_ring"]]), 5),
            "map_plate_mean": float(w[mb].mean()) if mb.any() else None,
            "map_shift_mean": float(w[ms].mean()) if ms.any() else None,
            "map_all_mean": float(w.mean()),
            "map_n_windows_plate": int(mb.sum()),
        })
    if (k + 1) % 5 == 0:
        print(f"  {k+1}/{len(stats)} {time.perf_counter()-t0:.0f}s", flush=True)

(JOB / f"out/plate_{args.set}.json").write_text(
    json.dumps({"n": len(rows), "seconds": round(time.perf_counter() - t0, 1),
                "rows": rows}, ensure_ascii=False, indent=1))
found = [r for r in rows if r["found"]]
print(f"{len(rows)} кропов, бокс найден у {len(found)}, "
      f"{time.perf_counter()-t0:.0f}s")
if found:
    pm = np.array([r["plate_minus_ctl_mean"] for r in found])
    print(f"Δ(пластина) − Δ(средний контроль): среднее {pm.mean():+.4f}, "
          f"медиана {np.median(pm):+.4f}, доля >0: {(pm>0).mean():.2f}")
