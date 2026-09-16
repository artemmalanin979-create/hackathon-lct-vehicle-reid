#!/usr/bin/env python3
"""Заранее объявленные проверки устойчивости метода.

Подвыборка (объявлена до запуска): первые 2 случая каждого вида из out/cases.json
плюс первые 2 из опорной группы. Для каждого — карта запроса, пересчитанная с
другой заливкой и с другим размером окна, и корреляция с основной картой
(на уровне пикселей 208x208, потому что сетки окон разные).

Вывод: если вывод «модель смотрит сюда» меняется от заливки или размера окна,
метод шаткий, и об этом надо сказать.
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

VARIANTS = [("gray127_w48_s16", "gray127", 48, 16),
            ("ring_w32_s16", "ring", 32, 16),
            ("ring_w64_s16", "ring", 64, 16)]

qm, gm = occl.read_split("val_query"), occl.read_split("val_gallery")
cases = json.load(open(JOB / "out/stats_cases.json"))["rows"]
ref = json.load(open(JOB / "out/stats_ref.json"))["rows"]
mc = np.load(JOB / "out/maps_cases.npz")
mr = np.load(JOB / "out/maps_ref.npz")

sel = []
seen = {}
for k, r in enumerate(cases):
    seen.setdefault(r["kind"], 0)
    if seen[r["kind"]] < 2:
        seen[r["kind"]] += 1
        sel.append(("cases", k, r, mc[f"{k}_wq"]))
for k in range(2):
    sel.append(("ref", k, ref[k], mr[f"{k}_wq"]))

rows = []
t0 = time.perf_counter()
for tag, k, r, w0 in sel:
    qim = occl.net_input(qm[r["qi"]])
    gim = occl.net_input(gm[r["gi"]])
    base_map = occl.windows_to_map(w0)
    rec = {"set": tag, "case": k, "kind": r["kind"], "qi": r["qi"], "gi": r["gi"],
           "cos": r["cos"], "base": {"mean": float(w0.mean()), "max": float(w0.max()),
                                     "argmax": occl.argmax_window(w0)[0]}}
    for name, mode, win, stride in VARIANTS:
        res = occl.pair_maps(qim, gim, mode=mode, sides=("q",), win=win, stride=stride)
        wv = res["wq"]
        mv = occl.windows_to_map(wv, win, stride)
        iy, ix = np.unravel_index(int(np.argmax(wv)), wv.shape)
        gp = occl.grid_pos(win, stride)
        rec[name] = {
            "mean": float(wv.mean()), "max": float(wv.max()),
            "argmax": (gp[ix], gp[iy], win, win),
            "corr_pixels": float(np.corrcoef(base_map.ravel(), mv.ravel())[0, 1]),
            "argmax_dist_px": float(np.hypot(gp[ix] + win / 2 - (occl.argmax_window(w0)[0][0] + occl.WIN / 2),
                                             gp[iy] + win / 2 - (occl.argmax_window(w0)[0][1] + occl.WIN / 2))),
        }
    rows.append(rec)
    print(f"{tag} #{k} {r['kind']:12s} " +
          "  ".join(f"{n}: r={rec[n]['corr_pixels']:+.2f} d={rec[n]['argmax_dist_px']:.0f}px"
                    for n, *_ in VARIANTS), flush=True)

(JOB / "out/checks.json").write_text(
    json.dumps({"seconds": round(time.perf_counter() - t0, 1), "rows": rows},
               ensure_ascii=False, indent=1))
for name, *_ in VARIANTS:
    c = np.array([r[name]["corr_pixels"] for r in rows])
    d = np.array([r[name]["argmax_dist_px"] for r in rows])
    print(f"{name}: корреляция с основной картой медиана {np.median(c):+.3f} "
          f"(мин {c.min():+.3f}); сдвиг максимума медиана {np.median(d):.0f} px")
print("всего", round(time.perf_counter() - t0, 1), "с")
