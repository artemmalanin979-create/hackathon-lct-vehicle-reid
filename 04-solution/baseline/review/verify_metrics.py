#!/usr/bin/env python3
"""Воспроизведение метрик контуром из их npy + СОБСТВЕННАЯ реализация mAP/Rank-1/TNR.

Собственная реализация написана с нуля (без заглядывания в контур):
- market: из ранжирования выбрасываем галерею с (same vehicle_id AND same camera_id);
- AP = среднее precision@позиция_попадания, знаменатель = число позитивов в допущенной галерее;
- mAP по запросам, у которых есть >=1 допущенный позитив (has_mate=1);
- Rank-1/5 по тем же запросам;
- top-10: сначала топ-10 по сырым скором ПО ВСЕЙ галерее (как в submission), затем фильтр,
  AP-знаменатель = все допущенные позитивы полной галереи;
- refusal presence: событие = top-1 допущенной галереи каждого запроса, порог >=,
  positive = запрос с mate, F1/TNR как обычно.
"""
import os
import csv, json, sys
from pathlib import Path
import numpy as np

JOB = Path(__file__).resolve().parent.parent
REPO = Path(__file__).resolve().parents[3]  # корень репозитория
SPLIT = REPO / "04-solution/split/files"
sys.path.insert(0, str(REPO / "04-solution/eval"))
from reid_metrics import evaluate, scores_from_embeddings  # noqa: E402

def rd(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))

qm, gm = rd(SPLIT/"val_query.csv"), rd(SPLIT/"val_gallery.csv")
q = np.load(JOB/"subject/out/val_query.npy").astype(np.float64)
g = np.load(JOB/"subject/out/val_gallery.npy").astype(np.float64)
qn = q/np.linalg.norm(q,axis=1,keepdims=True); gn = g/np.linalg.norm(g,axis=1,keepdims=True)
S = qn @ gn.T   # собственный расчёт скоров (обычный косинус)

qv = np.array([r["vehicle_id"] for r in qm]); gv = np.array([r["vehicle_id"] for r in gm])
qc = np.array([r["camera_id"] for r in qm]); gc = np.array([r["camera_id"] for r in gm])
mate = np.array([r["has_mate"]=="1" for r in qm])

def my_eval(S, threshold, exclude_same_cam_same_id=True):
    n_q, n_g = S.shape
    aps, r1s, r5s, aps10 = [], [], [], []
    ev_s, ev_pos = [], []
    for i in range(n_q):
        same_id = gv == qv[i]
        keep = ~(same_id & (gc == qc[i])) if exclude_same_cam_same_id else np.ones(n_g, bool)
        idx = np.flatnonzero(keep)
        order = idx[np.argsort(-S[i, idx], kind="stable")]
        rel = (gv[order] == qv[i])
        npos = int(rel.sum())
        # refusal-событие: top-1 допущенной галереи
        ev_s.append(S[i, order[0]]); ev_pos.append(bool(mate[i]))
        if npos:
            pos = np.flatnonzero(rel) + 1
            prec = np.arange(1, npos+1) / pos
            aps.append(prec.mean())
            r1s.append(1.0 if rel[0] else 0.0)
            r5s.append(1.0 if rel[:5].any() else 0.0)
            # top-10 как в submission: топ-10 по сырым скором всей галереи, потом фильтр
            top10 = np.argsort(-S[i], kind="stable")[:10]
            top10 = [j for j in top10 if keep[j]]
            rel10 = np.array([gv[j] == qv[i] for j in top10])
            k = int(rel10.sum())
            if k:
                p10 = np.flatnonzero(rel10) + 1
                aps10.append((np.arange(1, k+1) / p10).sum() / npos)
            else:
                aps10.append(0.0)
    ev_s = np.array(ev_s); ev_pos = np.array(ev_pos)
    acc = ev_s >= threshold
    tp = int((acc & ev_pos).sum()); fp = int((acc & ~ev_pos).sum())
    fn = int(ev_pos.sum()) - tp
    tn = int((~acc & ~ev_pos).sum())
    return dict(mAP=float(np.mean(aps)), R1=float(np.mean(r1s)), R5=float(np.mean(r5s)),
                mAP10=float(np.mean(aps10)), n_valid=len(aps),
                F1=2*tp/(2*tp+fp+fn), TNR=tn/int((~ev_pos).sum()),
                precision=tp/(tp+fp), recall=tp/(tp+fn))

T = 0.34921352213815304
mine = my_eval(S, T)
print("МОЯ реализация (market):", json.dumps(mine, indent=2))
mine_nx = my_eval(S, T, exclude_same_cam_same_id=False)
print("МОЯ без исключения:", json.dumps({k: mine_nx[k] for k in ("mAP","R1","mAP10")}, indent=2))

# случайные векторы, как у автора
rows = {}
for seed in (1,2,3):
    rng = np.random.default_rng(seed)
    rq = rng.standard_normal((len(qm),512)); rg = rng.standard_normal((len(gm),512))
    rqn = rq/np.linalg.norm(rq,axis=1,keepdims=True); rgn = rg/np.linalg.norm(rg,axis=1,keepdims=True)
    rows[seed] = my_eval(rqn @ rgn.T, 0.0)["mAP"]
print("random mAP по сидам:", rows, "mean:", np.mean(list(rows.values())))

# контур на их скорах — контрольное воспроизведение
S_ctr = scores_from_embeddings(q, g, metric="cosine")
print("макс |мой cos - контурный cos|:", float(np.abs(S - S_ctr).max()))
res = evaluate(S_ctr, query_ids=qv.tolist(), gallery_ids=gv.tolist(),
               query_cameras=qc.tolist(), gallery_cameras=gc.tolist(),
               known_absent=~mate, threshold=T, camera_policy="market", refusal_mode="presence")
print("КОНТУР full:", {k: res["ranking_full_gallery"][k] for k in ("mAP","Rank-1","Rank-5")})
print("КОНТУР top10 mAP:", res["ranking_top_k"]["mAP"])
print("КОНТУР refusal:", {k: res["refusal"][k] for k in ("f1","tnr","precision","recall")})
