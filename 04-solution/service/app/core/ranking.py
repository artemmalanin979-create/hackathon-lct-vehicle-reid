"""Косинусные близости и ранжирование кандидатов.

Формулы 1:1 с измерительным контуром проекта (scores_from_embeddings, metric=
"cosine"): весь счёт в float64, нормировка через максабс-масштабирование,
clip в [-1, 1]. Порядок при равных близостях — стабильная сортировка, как в
проверенных сдаваемых файлах бейзлайна.
"""
from __future__ import annotations

import numpy as np


def validate_scores(scores: np.ndarray, threshold: float | None = None) -> None:
    """Неконечные числа — ошибка вычисления/ввода, а не отказ модели."""
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError("порог threshold должен быть конечным (не NaN/Inf)")
    if not np.isfinite(scores).all():
        raise ValueError("scores содержат NaN или бесконечность")


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
    """Индексы галереи по убыванию скора; при равенстве — порядок галереи.

    Шкала любая, лишь бы «больше = лучше»: косинус или уверенность
    переранжирования. Для дистанции d это 1 - d (см. core/rerank.py).
    """
    validate_scores(score_row)
    return np.argsort(-score_row, kind="stable")


def accepted_candidates(score_row: np.ndarray, threshold: float) -> list[int]:
    """Индексы кандидатов со скором >= порога, по убыванию; пусто = отказ."""
    validate_scores(score_row, threshold)
    order = ranked_indices(score_row)
    return [int(j) for j in order if score_row[j] >= threshold]
