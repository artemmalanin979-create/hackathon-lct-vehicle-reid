#!/usr/bin/env python3
"""Шаг 1: сверка бейзлайна и чисел re-ranking из разбора на своих векторах.

Ожидания (context-critique.md, находка 2; там — на векторах автора, у нас —
переизвлечённые тем же скриптом, разбор показал max|diff|=0.0):
  база                     mAP 0.6566  R1 0.6310  mAP@10 0.6451
  rerank k1=20,k2=6,l=0.3  mAP 0.6666  R1 0.6322  mAP@10 0.6549
  rerank k1=6, k2=3, l=0.3 mAP 0.6937  R1 0.6635  mAP@10 0.6834
"""
import json
import sys
import time
from pathlib import Path

from inputs import JOB, SPLIT, require_files, BASE_HINT

require_files(
    [SPLIT / f"val_{part}.csv" for part in ("query", "gallery")]
    + [JOB / f"out/val_{part}.{ext}" for part in ("query", "gallery")
       for ext in ("ids", "npy")], hint=BASE_HINT)

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import JOB, SPLIT, read_meta, check_ids, eval_scores, eval_embeddings, rerank

qm = read_meta(SPLIT / "val_query.csv")
gm = read_meta(SPLIT / "val_gallery.csv")
check_ids(qm, JOB / "out/val_query.ids")
check_ids(gm, JOB / "out/val_gallery.ids")
q = np.load(JOB / "out/val_query.npy")
g = np.load(JOB / "out/val_gallery.npy")

res = {}
t0 = time.perf_counter()
res["base"] = eval_embeddings(q, g, qm, gm)
print("base:", json.dumps(res["base"]), f"eval={time.perf_counter()-t0:.1f}s", flush=True)

for name, (k1, k2, lam) in [("rerank_paper", (20, 6, 0.3)), ("rerank_critique", (6, 3, 0.3))]:
    dist, dt = rerank(q, g, k1, k2, lam)
    r = eval_scores(-dist, qm, gm)
    r["rerank_seconds"] = round(dt, 2)
    res[name] = r
    print(f"{name} (k1={k1},k2={k2},lam={lam}):", json.dumps(r), flush=True)

(JOB / "out/s01_verify_baseline.json").write_text(json.dumps(res, indent=2) + "\n")
