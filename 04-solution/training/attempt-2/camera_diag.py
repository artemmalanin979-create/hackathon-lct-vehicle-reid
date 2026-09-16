#!/usr/bin/env python3
"""F1/F2 из §8 рецепта на векторах валидации: камерный оракул и pseudo-F по камерам.

F1 — каждый вектор заменяется средним своей камеры и прогоняется тем же контуром:
     если контур при этом даёт заметный mAP, он сам пропускает камеру.
F2 — Calinski-Harabasz по меткам камер (матожидание при независимости = 1,0) плюс
     перестановочный нуль (метки камер перемешаны).
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

JOB = Path(__file__).resolve().parent.parent
REPO = Path("/home/artem/projects/hackathon-lct-vehicle-reid")
sys.path.insert(0, str(REPO / "04-solution/eval"))
from reid_metrics import evaluate, scores_from_embeddings  # noqa: E402
SPLIT = REPO / "04-solution/split/files"


def rd(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def ch_score(x, lab):
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    uniq = np.unique(lab)
    k = len(uniq)
    mean = x.mean(0)
    bss = wss = 0.0
    for c in uniq:
        g = x[lab == c]
        mg = g.mean(0)
        bss += len(g) * float(((mg - mean) ** 2).sum())
        wss += float(((g - mg) ** 2).sum())
    return (bss / (k - 1)) / (wss / (n - k))


def run(scores, qm, gm):
    r = evaluate(scores,
                 query_ids=[x["vehicle_id"] for x in qm],
                 gallery_ids=[x["vehicle_id"] for x in gm],
                 query_cameras=[x["camera_id"] for x in qm],
                 gallery_cameras=[x["camera_id"] for x in gm],
                 known_absent=np.array([x["has_mate"] == "0" for x in qm]),
                 threshold=0.0, camera_policy="market", refusal_mode="presence")
    f = r["ranking_full_gallery"]
    return {"mAP": round(f["mAP"], 4), "Rank-1": round(f["Rank-1"], 4)}


def main():
    tags = sys.argv[1:]
    qm, gm = rd(SPLIT / "val_query.csv"), rd(SPLIT / "val_gallery.csv")
    qc = np.array([int(x["camera_id"]) for x in qm])
    gc = np.array([int(x["camera_id"]) for x in gm])
    out = {}
    rng = np.random.default_rng(20260916)
    for tag in tags:
        q = np.load(JOB / "out" / f"val_query_{tag}.npy").astype(np.float64)
        g = np.load(JOB / "out" / f"val_gallery_{tag}.npy").astype(np.float64)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        allv, allc = np.vstack([q, g]), np.concatenate([qc, gc])
        ch = ch_score(allv, allc)
        perm = [ch_score(allv, rng.permutation(allc)) for _ in range(5)]
        # F1: вектор -> среднее своей камеры
        cmean = {c: allv[allc == c].mean(0) for c in np.unique(allc)}
        qo = np.stack([cmean[c] for c in qc])
        go = np.stack([cmean[c] for c in gc])
        qo /= np.linalg.norm(qo, axis=1, keepdims=True)
        go /= np.linalg.norm(go, axis=1, keepdims=True)
        out[tag] = {"F2_CH_cameras": round(ch, 4),
                    "F2_CH_permuted_mean": round(float(np.mean(perm)), 4),
                    "F1_camera_oracle": run(scores_from_embeddings(qo, go, metric="cosine"), qm, gm)}
        print(tag, json.dumps(out[tag], ensure_ascii=False), flush=True)
    (JOB / "out" / "camera_diag.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")


main()
