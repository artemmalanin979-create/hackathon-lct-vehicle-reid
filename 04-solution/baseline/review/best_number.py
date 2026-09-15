#!/usr/bin/env python3
"""Лучшее число ТЕМИ ЖЕ весами без обучения: TTA + k-reciprocal re-ranking.
Сетка k1/k2/lambda — чтобы видеть, не выброс ли лучшая точка."""
import csv, json, sys
from pathlib import Path
import numpy as np
JOB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JOB/"work"))
sys.path.insert(0, "/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/eval")
from reid_metrics import evaluate, scores_from_embeddings
from postproc import k_reciprocal, l2, QV, GV, QC, GC, ABS, report
EMB = JOB/"work"/"emb"
def load(v): return l2(np.load(EMB/f"val_query_{v}.npy").astype(np.float64)), l2(np.load(EMB/f"val_gallery_{v}.npy").astype(np.float64))
V={v:load(v) for v in ["base208","res256","res288","flip208"]}
qb,gb = V["base208"]
qt = l2(V["base208"][0]+V["res256"][0]+V["res288"][0]+V["flip208"][0])
gt = l2(V["base208"][1]+V["res256"][1]+V["res288"][1]+V["flip208"][1])
grid=[]
for k1 in (6,8,10,12,15,20):
    for k2 in (1,2,3,4,6):
        for lam in (0.0,0.3):
            d = k_reciprocal(qt, gt, k1, k2, lam)
            r = report(f"TTA4+rerank k1={k1} k2={k2} l={lam}", -d.astype(np.float64))
            grid.append(r)
best = max(grid, key=lambda r: r["mAP"])
print("ЛУЧШЕЕ:", json.dumps(best, ensure_ascii=False))
# то же на голом base (без TTA), для разделения вкладов
for k1,k2,lam in [(best["name"].split()[1].split("=")[1], best["name"].split()[2].split("=")[1], best["name"].split()[3].split("=")[1])]:
    d = k_reciprocal(qb, gb, int(k1), int(k2), float(lam))
    report(f"base+rerank(те же k) k1={k1} k2={k2} l={lam}", -d.astype(np.float64))
json.dump(grid, open(JOB/"work/best_number.json","w"), indent=1)
