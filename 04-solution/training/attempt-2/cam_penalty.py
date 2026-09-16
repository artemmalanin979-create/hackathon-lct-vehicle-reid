#!/usr/bin/env python3
"""Камерный штраф (VOC-ReID, §7.7 рецепта) поверх готовой матрицы схожести.

score_new = cos_reid(q,g) - lam * cos_cam(q,g)

Камерный вектор даёт ОТДЕЛЬНАЯ сеть (camnet), camera_id на инференсе не нужен.
Метрики — тем же контуром reid_metrics.py; переранжирование — канонический порт Zhong.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

JOB = Path(__file__).resolve().parent.parent
REPO = Path("/home/artem/projects/hackathon-lct-vehicle-reid")
sys.path.insert(0, str(REPO / "04-solution/eval"))
sys.path.insert(0, str(REPO / "04-solution/postproc/scripts"))
from reid_metrics import evaluate, scores_from_embeddings  # noqa: E402
from common import rerank  # noqa: E402
SPLIT = REPO / "04-solution/split/files"


def rd(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def l2(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def run(scores, qm, gm, fake_cameras=False):
    qc = [f"q:{i}" for i in range(len(qm))] if fake_cameras else [r["camera_id"] for r in qm]
    gc = [f"g:{i}" for i in range(len(gm))] if fake_cameras else [r["camera_id"] for r in gm]
    r = evaluate(scores,
                 query_ids=[x["vehicle_id"] for x in qm],
                 gallery_ids=[x["vehicle_id"] for x in gm],
                 query_cameras=qc, gallery_cameras=gc,
                 known_absent=np.array([x["has_mate"] == "0" for x in qm]),
                 threshold=0.0, camera_policy="market", refusal_mode="presence")
    f = r["ranking_full_gallery"]
    aps = np.array([row["ap"] for row in r["per_query"] if row["status"] == "known"])
    return {"mAP": round(f["mAP"], 4), "Rank-1": round(f["Rank-1"], 4)}, aps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="тег re-id векторов в out/")
    ap.add_argument("--cam-tag", default="cam", help="тег камерных векторов в out/")
    ap.add_argument("--lams", default="0,0.05,0.1,0.2,0.3,0.5")
    ap.add_argument("--rerank", default="6,3,0.3")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    qm, gm = rd(SPLIT / "val_query.csv"), rd(SPLIT / "val_gallery.csv")
    q = np.load(JOB / "out" / f"val_query_{a.tag}.npy")
    g = np.load(JOB / "out" / f"val_gallery_{a.tag}.npy")
    qc = l2(np.load(JOB / "out" / f"val_query_{a.cam_tag}.npy"))
    gc = l2(np.load(JOB / "out" / f"val_gallery_{a.cam_tag}.npy"))
    s_reid = scores_from_embeddings(q, g, metric="cosine")
    s_cam = qc @ gc.T
    k1, k2, lam = a.rerank.split(",")
    d_rr, _ = rerank(q, g, int(k1), int(k2), float(lam))
    s_rr = -d_rr

    # насколько камерная сеть вообще распознаёт камеру на валидации
    same_cam = (np.array([int(x["camera_id"]) for x in qm])[:, None]
                == np.array([int(x["camera_id"]) for x in gm])[None, :])
    diag = {"cos_cam_same_mean": round(float(s_cam[same_cam].mean()), 4),
            "cos_cam_diff_mean": round(float(s_cam[~same_cam].mean()), 4),
            "cos_reid_same_mean": round(float(s_reid[same_cam].mean()), 4),
            "cos_reid_diff_mean": round(float(s_reid[~same_cam].mean()), 4)}
    table = {"_diag": diag}
    for lm in [float(x) for x in a.lams.split(",")]:
        m, _ = run(s_reid - lm * s_cam, qm, gm)
        table[f"lam={lm}"] = m
        m2, _ = run(s_reid - lm * s_cam, qm, gm, fake_cameras=True)
        table[f"lam={lm}_no_camera_excl"] = m2
        m3, _ = run(s_rr - lm * s_cam, qm, gm)
        table[f"lam={lm}_rerank"] = m3
    print(json.dumps(table, indent=2, ensure_ascii=False))
    if a.out:
        a.out.write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n")


main()
