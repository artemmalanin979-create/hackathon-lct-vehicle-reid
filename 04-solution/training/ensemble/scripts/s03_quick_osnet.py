#!/usr/bin/env python3
"""Быстрая проверка только базовой модели на своих векторах train_fit (пока идёт s02b)."""
import sys, json
from pathlib import Path
import numpy as np
HOME = Path.home() / "lct-reid"; S = HOME / "repo/04-solution"; OUT = HOME / "jobs/job_43/out"
sys.path.insert(0, str(S / "eval")); sys.path.insert(0, str(S / "postproc/scripts"))
from reid_metrics import evaluate, scores_from_embeddings
from common import rerank
E = np.load(OUT / "emb_train_osnet.npy").astype(np.float64)
vids = np.load(OUT / "train_meta_vehicle_id.npy"); cams = np.load(OUT / "train_meta_camera_id.npy")
uniq = np.unique(vids); rng = np.random.default_rng(20260916)
dev_ids = set(uniq[rng.permutation(len(uniq))[:100]].tolist())
q_idx, g_idx = [], []
for v in sorted(dev_ids):
    rows = np.flatnonzero(vids == v); g_here, q_here = [], []
    for c in np.unique(cams[rows]):
        grp = rows[cams[rows] == c]; g_here.append(int(grp[0])); q_here.extend(int(i) for i in grp[1:])
    if not q_here and len(g_here) >= 2: q_here.append(g_here.pop())
    q_idx.extend(q_here); g_idx.extend(g_here)
q_idx, g_idx = np.array(sorted(q_idx)), np.array(sorted(g_idx))
r = evaluate(scores_from_embeddings(E[q_idx], E[g_idx], metric="cosine"),
             query_ids=[int(vids[i]) for i in q_idx], gallery_ids=[int(vids[i]) for i in g_idx],
             query_cameras=[int(cams[i]) for i in q_idx], gallery_cameras=[int(cams[i]) for i in g_idx],
             known_absent=np.zeros(len(q_idx), dtype=bool), threshold=0.0, camera_policy="market", refusal_mode="presence")
f = r["ranking_full_gallery"]
print(json.dumps({"dev": {"n_q": int(len(q_idx)), "n_g": int(len(g_idx)), "mAP": round(f["mAP"], 5), "Rank-1": round(f["Rank-1"], 5), "expected_mAP": 0.71358, "expected_nq_ng": [353, 241]}}))
