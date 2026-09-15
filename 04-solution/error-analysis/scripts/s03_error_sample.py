#!/usr/bin/env python3
"""Случайная выборка ошибок top-1 для ручной разметки + контрольные листы."""
import csv, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sheets_lib import build_sheet, crop_img, P

JOB = Path(__file__).resolve().parent.parent
SPLIT = P / "04-solution/split/files"


def read_meta(p):
    rows = list(csv.DictReader(open(p, newline="")))
    for r in rows:
        for k in ("x", "y", "w", "h", "vehicle_id", "camera_id"):
            r[k] = int(r[k])
    return rows


qm, gm = read_meta(SPLIT / "val_query.csv"), read_meta(SPLIT / "val_gallery.csv")
pb = np.load(JOB / "out/perquery_base.npz", allow_pickle=True)
pr = np.load(JOB / "out/perquery_rr.npz", allow_pickle=True)
S = pb["score_matrix"]
known = pb["status"] == "known"
idx = np.flatnonzero(known)
r1, fr, top = pb["rank1"], pb["first_rank"], pb["top_gidx"]
gv = np.array([r["vehicle_id"] for r in gm]); gc = np.array([r["camera_id"] for r in gm])
qv = np.array([r["vehicle_id"] for r in qm]); qc = np.array([r["camera_id"] for r in qm])

err = idx[r1[idx] == 0]
rng = np.random.default_rng(20260915)
sample = np.sort(rng.choice(err, size=64, replace=False))
(JOB / "out/error_sample.json").write_text(json.dumps(
    {"n_errors": int(len(err)), "sample": sample.tolist(), "seed": 20260915}, indent=2) + "\n")


def best_mate(i):
    m = np.flatnonzero((gv == qv[i]) & (gc != qc[i]))
    return m[np.argmax(S[i, m])]


rows_all, meta_rows = [], []
for n, i in enumerate(sample, 1):
    j = best_mate(i); k = int(top[i])
    meta_rows.append(dict(case=n, qi=int(i), mate=int(j), wrong=k,
                          rank=int(fr[i]), cos_mate=float(S[i, j]), cos_wrong=float(S[i, k]),
                          rank_rr=int(pr["first_rank"][i]), r1_rr=int(pr["rank1"][i]),
                          q_img=qm[i]["image_id"], g_img=gm[j]["image_id"], w_img=gm[k]["image_id"],
                          q_cam=int(qc[i]), mate_cam=int(gc[j]), wrong_cam=int(gc[k]),
                          q_vid=int(qv[i]), wrong_vid=int(gv[k])))
    rows_all.append([
        (crop_img(qm, i), f"#{n} запрос v{qv[i]} cam{qc[i]}"),
        (crop_img(gm, j), f"верный: ранг {int(fr[i])}, cos {S[i,j]:.3f}\ncam{gc[j]}"),
        (crop_img(gm, k), f"top-1 ОШИБКА: cos {S[i,k]:.3f}\nv{gv[k]} cam{gc[k]}"),
    ])

(JOB / "out/error_sample_meta.json").write_text(json.dumps(meta_rows, indent=2) + "\n")
outs = []
for s in range(0, 64, 8):
    p = JOB / f"sheets/errsample_{s//8+1:02d}.jpg"
    build_sheet(rows_all[s:s + 8], ["ЗАПРОС", "ВЕРНЫЙ ОТВЕТ (в галерее)", "ЧТО ВЫДАНО ПЕРВЫМ"],
                f"Случайная выборка ошибок top-1, лист {s//8+1}/8 (кейсы {s+1}-{s+8} из 64; ген. сов. 307)",
                p, border_by=[(40, 90, 200), (20, 140, 60), (200, 40, 40)])
    outs.append(str(p))
print("\n".join(outs))
