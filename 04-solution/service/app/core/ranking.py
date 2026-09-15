"""Косинусные близости и ранжирование кандидатов.

Формулы 1:1 с измерительным контуром проекта (scores_from_embeddings, metric=
"cosine"): весь счёт в float64, нормировка через максабс-масштабирование,
clip в [-1, 1]. Порядок при равных близостях — стабильная сортировка, как в
проверенных сдаваемых файлах бейзлайна.
"""
from __future__ import annotations

import numpy as np


def cosine_scores(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Матрица косинусов len(query) x len(gallery), float64."""
    q = np.asarray(query, dtype=np.float64)
    g = np.asarray(gallery, dtype=np.float64)
    if q.ndim == 1:
        q = q[None]
    if q.shape[1] != g.shape[1] or q.shape[1] == 0:
        raise ValueError("размерности эмбеддингов не совпадают или нулевые")

    def normalize(x: np.ndarray) -> np.ndarray:
        scale = np.max(np.abs(x), axis=1, keepdims=True)
        if np.any(scale == 0):
            raise ValueError("косинус не определён для нулевого вектора")
        x = x / scale
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    return np.clip(normalize(q) @ normalize(g).T, -1.0, 1.0)


def ranked_indices(score_row: np.ndarray) -> np.ndarray:
    """Индексы галереи по убыванию близости; при равенстве — порядок галереи."""
    return np.argsort(-score_row, kind="stable")


def accepted_candidates(score_row: np.ndarray, threshold: float) -> list[int]:
    """Индексы кандидатов с близостью >= порога, по убыванию; пусто = отказ."""
    order = ranked_indices(score_row)
    return [int(j) for j in order if score_row[j] >= threshold]
