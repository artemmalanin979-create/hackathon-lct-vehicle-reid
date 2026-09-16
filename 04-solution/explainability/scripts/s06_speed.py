#!/usr/bin/env python3
"""Сколько стоит объяснение: замер времени и проверка дешёвой сетки.

Основной вопрос отчёта: годится ли объяснимость для живого показа оператору или
только для разбора постфактум. Меряем на том же CPU, что и сервис.

Считаем:
  - время одного прогона сети (батч 32, как в карте);
  - время полной карты пары (две стороны, окно 48 / шаг 16);
  - время дешёвой сетки (окно 48 / шаг 32 и окно 64 / шаг 48) и её корреляцию
    с полной картой — чтобы сказать, теряется ли смысл при удешевлении.
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

COARSE = [("w48_s32", 48, 32), ("w64_s48", 64, 48)]

qm, gm = occl.read_split("val_query"), occl.read_split("val_gallery")
cases = json.load(open(JOB / "out/stats_cases.json"))["rows"]
mc = np.load(JOB / "out/maps_cases.npz")

# подвыборка: первые 2 случая каждого вида — та же, что в s04_checks
sel, seen = [], {}
for k, r in enumerate(cases):
    seen.setdefault(r["kind"], 0)
    if seen[r["kind"]] < 2:
        seen[r["kind"]] += 1
        sel.append((k, r))

out = {"n_positions_full": occl.NG ** 2, "win": occl.WIN, "stride": occl.STRIDE}

# 1) чистое время прогона сети
s = occl.session()
name = s.get_inputs()[0].name
x = np.random.rand(32, 3, 208, 208).astype(np.float32) * 255
s.run(None, {name: x})
t0 = time.perf_counter()
for _ in range(3):
    s.run(None, {name: x})
out["ms_per_forward_b32"] = round((time.perf_counter() - t0) / 3 / 32 * 1000, 2)

# 2) время загрузки кропа
t0 = time.perf_counter()
for k, r in sel:
    occl.net_input(qm[r["qi"]])
out["ms_crop_load"] = round((time.perf_counter() - t0) / len(sel) * 1000, 1)

rows = []
for k, r in sel:
    qim, gim = occl.net_input(qm[r["qi"]]), occl.net_input(gm[r["gi"]])
    rec = {"case": k, "kind": r["kind"]}
    t0 = time.perf_counter()
    full = occl.pair_maps(qim, gim)                       # обе стороны
    rec["s_full_pair"] = round(time.perf_counter() - t0, 2)
    t0 = time.perf_counter()
    occl.pair_maps(qim, gim, sides=("q",))
    rec["s_full_one_side"] = round(time.perf_counter() - t0, 2)
    base_map = occl.windows_to_map(full["wq"])
    for nm, win, stride in COARSE:
        t0 = time.perf_counter()
        cm = occl.pair_maps(qim, gim, sides=("q",), win=win, stride=stride)
        dt = time.perf_counter() - t0
        mv = occl.windows_to_map(cm["wq"], win, stride)
        rec[nm] = {"s_one_side": round(dt, 2),
                   "n_pos": int(cm["wq"].size),
                   "corr": float(np.corrcoef(base_map.ravel(), mv.ravel())[0, 1])}
    rows.append(rec)
    print(json.dumps(rec, ensure_ascii=False), flush=True)

out["rows"] = rows
out["s_full_pair_median"] = float(np.median([r["s_full_pair"] for r in rows]))
out["s_full_one_side_median"] = float(np.median([r["s_full_one_side"] for r in rows]))
for nm, _, _ in COARSE:
    out[f"{nm}_s_median"] = float(np.median([r[nm]["s_one_side"] for r in rows]))
    out[f"{nm}_corr_median"] = float(np.median([r[nm]["corr"] for r in rows]))
    out[f"{nm}_corr_min"] = float(np.min([r[nm]["corr"] for r in rows]))
(JOB / "out/speed.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "rows"},
                 ensure_ascii=False, indent=1))
