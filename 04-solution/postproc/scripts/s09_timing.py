#!/usr/bin/env python3
"""Шаг 9: тайминги на свободной машине.

1) OSNet, батч 1: мс/объект на 200 первых строках val_query для вариантов
   208 / 208f / 256 / 288 (стоимость конфигурации = сумма её проходов).
2) Re-ranking: время на нашем размере (1110+750) для трёх наборов параметров
   и кривая роста на случайных векторах при N x {0.5, 1, 2, 4} с сохранением
   пропорции query:gallery. Память: рабочие матрицы ~4-5 x N^2 float64 —
   при N=7440 это ~2 ГБ, дальше не идём (лимит машины 15 ГБ на всех).
Выход: out/s09_timing.json. Печать loadavg до/после — контроль фона.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import JOB, SPLIT, read_meta, rerank
from s03_extract import make_session, extract

out = {"loadavg_start": Path("/proc/loadavg").read_text().split()[:3]}

# --- 1) OSNet батч 1 ---
rows = read_meta(SPLIT / "val_query.csv")[:200]
session = make_session()
osnet = {}
global BATCH
import s03_extract
s03_extract.BATCH = 1
for size, flip in [(208, False), (208, True), (256, False), (288, False)]:
    tag = f"{size}{'f' if flip else ''}"
    _, dt = s03_extract.extract(session, rows, size, flip)   # прогретая сессия после 1-го
    _, dt = s03_extract.extract(session, rows, size, flip)
    osnet[tag] = round(1000 * dt / len(rows), 1)
    print(f"osnet batch1 {tag}: {osnet[tag]} мс/объект", flush=True)
out["osnet_batch1_ms_per_obj"] = osnet
out["osnet_config_ms"] = {
    "1 проход (208)": osnet["208"],
    "2 прохода (208+256)": round(osnet["208"] + osnet["256"], 1),
    "3 прохода (208+208f+256)": round(osnet["208"] + osnet["208f"] + osnet["256"], 1),
    "4 прохода (208+256+288+208f)": round(osnet["208"] + osnet["256"] + osnet["288"] + osnet["208f"], 1),
}

# --- 2) re-ranking на реальном размере ---
q = np.load(JOB / "out/val_query_208.npy")
g = np.load(JOB / "out/val_gallery_208.npy")
rr = {}
for params in [(20, 6, 0.3), (6, 3, 0.3), (10, 3, 0.45)]:
    ts = []
    for _ in range(3):
        _, dt = rerank(q, g, *params)
        ts.append(dt)
    rr[str(params)] = {"best_s": round(min(ts), 2), "runs_s": [round(t, 2) for t in ts]}
    print(f"rerank {params}: {min(ts):.2f} с (1110x750)", flush=True)
out["rerank_1860"] = rr

# --- кривая роста, случайные вектора ---
rng = np.random.default_rng(0)
growth = []
for scale in (0.5, 1.0, 2.0, 4.0):
    nq, ng = int(1110 * scale), int(750 * scale)
    qq = rng.standard_normal((nq, 512)).astype(np.float32)
    gg = rng.standard_normal((ng, 512)).astype(np.float32)
    t0 = time.perf_counter()
    rerank(qq, gg, 10, 3, 0.45)
    dt = time.perf_counter() - t0
    growth.append({"N_total": nq + ng, "seconds": round(dt, 2),
                   "ms_per_query": round(1000 * dt / nq, 1)})
    print(f"rerank N={nq+ng}: {dt:.1f} с", flush=True)
out["rerank_growth_random"] = growth
# показатель роста по последним двум точкам
a, b = growth[-2], growth[-1]
out["rerank_growth_exponent_last"] = round(
    np.log(b["seconds"] / a["seconds"]) / np.log(b["N_total"] / a["N_total"]), 2)

out["loadavg_end"] = Path("/proc/loadavg").read_text().split()[:3]
(JOB / "out/s09_timing.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(out, indent=2, ensure_ascii=False))
