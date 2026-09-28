"""k-reciprocal переранжирование (Zhong et al., CVPR 2017) — порядок кандидатов.

Работает ПОСЛЕ извлечения векторов и не трогает их: `embeddings.npy` пишется до
переранжирования и от него не зависит. Меняется только порядок кандидатов и
шкала уверенности, поэтому замеряемое организатором время «изображение → вектор»
остаётся прежним.

Код — порт эталонной реализации Zhong et al.
(`person-re-ranking/python-version/re_ranking_ranklist.py`, она же в fast-reid),
дословно повторяющий экспериментальный `04-solution/postproc/scripts/common.py`:
float64, тот же порядок операций. Зависимость — только numpy.

Шкала. Метод возвращает дистанцию (меньше = ближе), смесь жаккардовой дистанции
по k-взаимным окрестностям и нормированной косинус-дистанции. Сервис работает в
одной шкале «больше = лучше», поэтому наружу отдаётся уверенность 1 − d
(`distances_to_scores`): порядок сохраняется, а `ranking.py`/`submission.py`
остаются общими для обоих режимов.

Память квадратична по числу объектов: (Q+G)² float64 на матрицу, несколько матриц
одновременно. Замер на 1860 объектах выданного теста: пик выделений 169 МиБ,
maxRSS 219 МиБ. Практический потолок обычной реализации — порядка 10⁴ объектов
(≈5 ГБ); для 10⁶ нужен разреженный вариант.
"""
from __future__ import annotations

import numpy as np

from ..numeric_inputs import validate_rerank_params

STREAM_RERANK_TOP_K = 50


def _k_reciprocal(initial_rank: np.ndarray, i: int, k: int) -> np.ndarray:
    """k-взаимные соседи объекта i: прямые соседи, для которых i — тоже сосед."""
    forward = initial_rank[i, : k + 1]
    backward = initial_rank[forward, : k + 1]
    return forward[np.where(backward == i)[0]]


def rerank_distances(query: np.ndarray, gallery: np.ndarray,
                     k1: int, k2: int, lam: float) -> np.ndarray:
    """Переранжированные дистанции len(query) x len(gallery), float64.

    Считается на объединении «все запросы прогона + вся галерея» — именно так
    формируется сдача одним прогоном.
    """
    q = np.asarray(query, dtype=np.float64)
    g = np.asarray(gallery, dtype=np.float64)
    if q.ndim == 1:
        q = q[None]
    if q.shape[1] != g.shape[1] or q.shape[1] == 0:
        raise ValueError("размерности эмбеддингов не совпадают или нулевые")
    validate_rerank_params(k1, k2, lam)
    query_num = len(q)

    feats = np.concatenate([q, g])
    if not np.isfinite(feats).all():
        raise ValueError("эмбеддинги содержат NaN или бесконечность")
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("переранжирование не определено для нулевого вектора")
    feats = feats / norms
    dist_all = 1.0 - np.clip(feats @ feats.T, -1.0, 1.0)

    # Канон (torch-версия Zhong): original_dist = квадрат евклида, затем нормировка
    # на максимум по столбцу. Для L2-нормированных векторов euclid^2 = 2*(1-cos);
    # множитель 2 сокращается нормировкой, поэтому достаточно 1-cos без возведения.
    column_max = np.max(dist_all, axis=0)
    # Нулевой максимум означает столбец совпадающих объектов: дистанции уже
    # равны нулю. Нормировка на 1 сохраняет их, не создавая 0/0 -> NaN.
    # Ненулевые максимумы и порядок всех остальных операций не меняются.
    column_max[column_max == 0] = 1.0
    original_dist = (dist_all / column_max).T
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
    return (jaccard * (1 - lam) + orig_q * lam)[:, query_num:]


def distances_to_scores(distances: np.ndarray) -> np.ndarray:
    """Дистанция -> уверенность (1 − d): монотонно, порядок не меняется."""
    return 1.0 - np.asarray(distances, dtype=np.float64)


def rerank_scores(query: np.ndarray, gallery: np.ndarray,
                  k1: int, k2: int, lam: float) -> np.ndarray:
    """Матрица уверенностей len(query) x len(gallery) на шкале переранжирования."""
    return distances_to_scores(rerank_distances(query, gallery, k1, k2, lam))


def rerank_distances_independent(query: np.ndarray, gallery: np.ndarray,
                                 k1: int, k2: int, lam: float) -> np.ndarray:
    """KR внутри cosine top-50 одного query; остальные объекты ниже top-50.

    Используется только текущий query и выбранные из статичной gallery
    кандидаты. Внешние кандидаты сохраняют cosine-порядок; их дистанция больше
    дистанции любого выбранного кандидата. Их шкала — техническое дополнение
    для полного CSV, а не калиброванная уверенность совпадения. Порог отказа
    для этого режима требуется выбрать отдельно до релиза.
    """
    q = np.asarray(query, dtype=np.float64)
    g = np.asarray(gallery, dtype=np.float64)
    if q.ndim == 1:
        q = q[None]
    if q.ndim != 2 or g.ndim != 2 or q.shape[1] != g.shape[1] or q.shape[1] == 0:
        raise ValueError("размерности эмбеддингов не совпадают или нулевые")
    validate_rerank_params(k1, k2, lam)
    if not np.isfinite(q).all() or not np.isfinite(g).all():
        raise ValueError("эмбеддинги содержат NaN или бесконечность")
    q_norms = np.linalg.norm(q, axis=1)
    g_norms = np.linalg.norm(g, axis=1)
    if np.any(q_norms == 0) or np.any(g_norms == 0):
        raise ValueError("переранжирование не определено для нулевого вектора")
    distances = np.empty((len(q), len(g)), dtype=np.float64)
    if len(q) == 0 or len(g) == 0:
        return distances

    g_unit = g / g_norms[:, None]
    for i, row in enumerate(q):
        cosine = np.clip(g_unit @ (row / q_norms[i]), -1.0, 1.0)
        selected = np.argsort(-cosine, kind="stable")[:min(STREAM_RERANK_TOP_K, len(g))]
        selected_dist = rerank_distances(row[None], g[selected], k1, k2, lam)[0]
        distances[i, selected] = selected_dist
        if len(selected) < len(g):
            outside = np.ones(len(g), dtype=bool)
            outside[selected] = False
            distances[i, outside] = (np.max(selected_dist) + 1.0
                                     + (1.0 - cosine[outside]))
    return distances


def rerank_scores_independent(query: np.ndarray, gallery: np.ndarray,
                              k1: int, k2: int, lam: float) -> np.ndarray:
    """Уверенности потокового KR: ни один query не влияет на другие query."""
    return distances_to_scores(
        rerank_distances_independent(query, gallery, k1, k2, lam))
