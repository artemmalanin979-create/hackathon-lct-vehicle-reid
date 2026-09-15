#!/usr/bin/env python3
"""Прогон измерительного контура + выгрузка per-query фактов.

Контур (04-solution/eval/reid_metrics.py) и rerank (04-solution/postproc/scripts/common.py)
импортируются без модификаций. Сверяем воспроизведение опубликованных чисел бейзлайна,
затем сохраняем per-query таблицу для разбора ошибок.
"""
import os
import csv, json, sys
from pathlib import Path
import numpy as np

JOB = Path(__file__).resolve().parent.parent
REPO = Path(__file__).resolve().parents[3]  # корень репозитория
P = REPO
sys.path.insert(0, str(P / "04-solution/eval"))
sys.path.insert(0, str(P / "04-solution/postproc/scripts"))
from reid_metrics import evaluate, scores_from_embeddings  # noqa: E402
from common import rerank  # noqa: E402

SPLIT = P / "04-solution/split/files"


def read_meta(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


qm, gm = read_meta(SPLIT / "val_query.csv"), read_meta(SPLIT / "val_gallery.csv")
qids = (JOB / "out/val_query.ids").read_text().split()
gids = (JOB / "out/val_gallery.ids").read_text().split()
assert qids == [r["image_id"] for r in qm] and gids == [r["image_id"] for r in gm]

qe = np.load(JOB / "out/val_query.npy")
ge = np.load(JOB / "out/val_gallery.npy")
print("embeddings:", qe.shape, ge.shape)


def run(scores):
    return evaluate(
        scores,
        query_ids=[r["vehicle_id"] for r in qm],
        gallery_ids=[r["vehicle_id"] for r in gm],
        query_cameras=[r["camera_id"] for r in qm],
        gallery_cameras=[r["camera_id"] for r in gm],
        known_absent=np.array([r["has_mate"] == "0" for r in qm]),
        threshold=0.0, camera_policy="market", refusal_mode="presence",
        include_rankings=True,
    )


cos = scores_from_embeddings(qe, ge, metric="cosine")
res_base = run(cos)
dist, secs = rerank(qe, ge, 6, 3, 0.3)
res_rr = run(-dist)

published = {"base": (0.6565692724907637, 0.6310096153846154),
             "rr": (0.6936584724586873, 0.6634615384615384)}
check = {}
for name, res in [("base", res_base), ("rr", res_rr)]:
    f = res["ranking_full_gallery"]
    got = (f["mAP"], f["Rank-1"])
    check[name] = {"mAP": got[0], "Rank-1": got[1], "Rank-5": f["Rank-5"], "mINP": f["mINP"],
                   "valid": f["num_valid_queries"],
                   "d_mAP": got[0] - published[name][0], "d_R1": got[1] - published[name][1]}
    print(name, json.dumps(check[name]))
check["rerank_seconds"] = secs
(JOB / "out/eval_check.json").write_text(json.dumps(check, indent=2) + "\n")

# per-query выгрузка: индексы рангов сохраняем в npz (ranking — массив переменной длины,
# но eligible_count почти константа; храним первые 50 + ранг верного ответа)
def dump(res, cos_or_score, tag):
    rows = res["per_query"]
    out = {k: [] for k in ("status", "ap", "rank1", "rank5", "inp", "num_relevant",
                           "first_rank", "top_gidx", "top_score", "eligible")}
    top50 = np.full((len(rows), 50), -1, dtype=np.int32)
    for i, r in enumerate(rows):
        out["status"].append(r["status"])
        out["ap"].append(np.nan if r["ap"] is None else r["ap"])
        out["rank1"].append(np.nan if r["rank1"] is None else r["rank1"])
        out["rank5"].append(np.nan if r["rank5"] is None else r["rank5"])
        out["inp"].append(np.nan if r["inp"] is None else r["inp"])
        out["num_relevant"].append(r["num_relevant"])
        out["first_rank"].append(r["positive_ranks"][0] if r["positive_ranks"] else -1)
        out["top_gidx"].append(-1 if r["top_gallery_index"] is None else r["top_gallery_index"])
        out["top_score"].append(np.nan if r["top_score"] is None else r["top_score"])
        out["eligible"].append(r["eligible_count"])
        rk = r["ranking"][:50]
        top50[i, :len(rk)] = rk
    np.savez(JOB / f"out/perquery_{tag}.npz",
             status=np.array(out["status"]),
             ap=np.array(out["ap"]), rank1=np.array(out["rank1"]),
             rank5=np.array(out["rank5"]), inp=np.array(out["inp"]),
             num_relevant=np.array(out["num_relevant"]),
             first_rank=np.array(out["first_rank"]),
             top_gidx=np.array(out["top_gidx"]), top_score=np.array(out["top_score"]),
             eligible=np.array(out["eligible"]), top50=top50,
             score_matrix=cos_or_score.astype(np.float32))
    print(tag, "dumped", len(rows))


dump(res_base, cos, "base")
dump(res_rr, -dist, "rr")
