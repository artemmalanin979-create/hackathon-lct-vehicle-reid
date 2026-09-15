"""Общий код job_11: обёртка контура метрик, каноническое k-reciprocal re-ranking.

Контур метрик используется импортом без модификаций (как в baseline/scripts/run_eval.py).
Re-ranking — дословный порт эталонной реализации Zhong et al. 2017
(person-re-ranking/python-version/re_ranking_ranklist.py, она же в fast-reid),
с одним отличием: dtype float64 вместо float32 и факторизация по (k1) -> (k2) -> λ,
чтобы сетка параметров не пересчитывала общие части. Пошаговая структура и порядок
операций сохранены.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

JOB = Path(__file__).resolve().parent.parent
EVAL_DIR = Path("/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/eval")
SPLIT = Path("/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/split/files")
DATA = Path("/home/artem/projects/hackathon-lct-vehicle-reid/data")
sys.path.insert(0, str(EVAL_DIR))
from reid_metrics import evaluate, scores_from_embeddings  # noqa: E402


def read_meta(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def check_ids(meta, ids_path: Path):
    ids = ids_path.read_text().split()
    assert ids == [r["image_id"] for r in meta], f"порядок строк {ids_path} != CSV"


def eval_scores(scores, qm, gm, *, camera_policy="market"):
    """Ранговые метрики контура (порог 0 — на ранжирование не влияет)."""
    res = evaluate(
        scores,
        query_ids=[r["vehicle_id"] for r in qm],
        gallery_ids=[r["vehicle_id"] for r in gm],
        query_cameras=[r["camera_id"] for r in qm],
        gallery_cameras=[r["camera_id"] for r in gm],
        known_absent=np.array([r.get("has_mate", "1") == "0" for r in qm]),
        threshold=0.0, camera_policy=camera_policy, refusal_mode="presence",
    )
    full, topk = res["ranking_full_gallery"], res["ranking_top_k"]
    return {
        "mAP": full["mAP"], "Rank-1": full["Rank-1"], "Rank-5": full["Rank-5"],
        "mAP@10": topk["mAP"], "valid_queries": full["num_valid_queries"],
    }


def eval_embeddings(q, g, qm, gm, **kw):
    return eval_scores(scores_from_embeddings(q, g, metric="cosine"), qm, gm, **kw)


def l2norm(m):
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def average_embeddings(mats):
    """TTA: среднее L2-нормированных векторов, затем повторная L2-нормировка."""
    acc = np.zeros(mats[0].shape, dtype=np.float64)
    for m in mats:
        acc += l2norm(m.astype(np.float64))
    return l2norm(acc / len(mats)).astype(np.float32)


# ----------------------------------------------------------------------------
# k-reciprocal re-ranking (Zhong et al., CVPR 2017)
# ----------------------------------------------------------------------------

def _k_reciprocal(initial_rank, i, k):
    forward = initial_rank[i, : k + 1]
    backward = initial_rank[forward, : k + 1]
    return forward[np.where(backward == i)[0]]


def cos_dist_all(q_emb, g_emb):
    """Полная матрица (Q+G)x(Q+G) косинус-дистанций 1-cos."""
    feats = np.concatenate([q_emb, g_emb]).astype(np.float64)
    feats = l2norm(feats)
    return 1.0 - np.clip(feats @ feats.T, -1.0, 1.0)


def rerank_prepare(q_emb, g_emb, k1, dist_all=None):
    """Часть, зависящая только от k1: original_dist, initial_rank, V.

    dist_all — готовая матрица дистанций (Q+G)x(Q+G) (для слияния моделей);
    если не задана, берётся косинус-дистанция по векторам.
    """
    if dist_all is None:
        dist_all = cos_dist_all(q_emb, g_emb)
    # Канон (torch-версия Zhong): original_dist = квадрат евклида, затем нормировка
    # на максимум по столбцу. Для L2-нормированных векторов euclid^2 = 2*(1-cos);
    # множитель 2 сокращается нормировкой, поэтому достаточно 1-cos без возведения.
    original_dist = (dist_all / np.max(dist_all, axis=0)).T
    all_num = original_dist.shape[0]
    initial_rank = np.argsort(original_dist, axis=1)

    V = np.zeros_like(original_dist)
    for i in range(all_num):
        k_recip = _k_reciprocal(initial_rank, i, k1)
        expansion = k_recip
        half = int(np.around(k1 / 2.0))
        for cand in k_recip:
            cand_recip = _k_reciprocal(initial_rank, cand, half)
            if len(np.intersect1d(cand_recip, k_recip)) > 2.0 / 3 * len(cand_recip):
                expansion = np.append(expansion, cand_recip)
        expansion = np.unique(expansion)
        weight = np.exp(-original_dist[i, expansion])
        V[i, expansion] = weight / np.sum(weight)
    return original_dist, initial_rank, V


def rerank_finalize(original_dist, initial_rank, V, query_num, k2, lambdas):
    """Часть, зависящая от k2 и λ. Возвращает {λ: final_dist (Q x G)}."""
    all_num = V.shape[0]
    if k2 != 1:
        V_qe = np.zeros_like(V)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
    orig_q = original_dist[:query_num, :]
    inv_index = [np.where(V[:, i] != 0)[0] for i in range(all_num)]
    jaccard = np.zeros_like(orig_q)
    for i in range(query_num):
        temp_min = np.zeros(all_num)
        ind_nonzero = np.where(V[i, :] != 0)[0]
        for j in ind_nonzero:
            imgs = inv_index[j]
            temp_min[imgs] += np.minimum(V[i, j], V[imgs, j])
        jaccard[i] = 1 - temp_min / (2.0 - temp_min)
    return {
        lam: (jaccard * (1 - lam) + orig_q * lam)[:, query_num:]
        for lam in lambdas
    }


def rerank(q_emb, g_emb, k1=20, k2=6, lam=0.3):
    """Полный проход; возвращает (final_dist Q x G, время_подготовки+финализации)."""
    t0 = time.perf_counter()
    original_dist, initial_rank, V = rerank_prepare(q_emb, g_emb, k1)
    out = rerank_finalize(original_dist, initial_rank, V, len(q_emb), k2, [lam])
    return out[lam], time.perf_counter() - t0
