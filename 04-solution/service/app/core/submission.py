"""Запись сдаваемых файлов: submission.csv, candidates.csv, embeddings.npy.

Форматы — по README датасета («Как сдавать решение»); побайтово совпадают с
файлами, которые собирал проверенный бейзлайн (тот же csv.writer, те же
заголовки, confidence с шестью знаками).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .ranking import accepted_candidates, ranked_indices, validate_scores
from .validation import validate_embeddings

TOP_K_SUBMISSION = 10  # столько кандидатов принимает submission.csv


def save_embeddings(path: Path, query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """embeddings.npy: сначала все query по порядку файла, затем вся галерея."""
    validate_embeddings(query, name="query")
    validate_embeddings(gallery, name="gallery")
    emb = np.concatenate([query, gallery]).astype(np.float32)
    np.save(path, emb)
    return emb


def write_submission(path: Path, q_ids: list[str], g_ids: list[str], scores: np.ndarray) -> None:
    """submission.csv: query_id, gallery_id_1..gallery_id_10 по убыванию скора.

    Скор — косинус либо уверенность переранжирования (1 - дистанция); шкалу
    выбирает вызывающий, порядок строк и формат от неё не зависят.
    """
    validate_scores(scores)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id"] + [f"gallery_id_{k}" for k in range(1, TOP_K_SUBMISSION + 1)])
        for i, qid in enumerate(q_ids):
            top = ranked_indices(scores[i])[:TOP_K_SUBMISSION]
            w.writerow([qid] + [g_ids[j] for j in top])


def write_candidates(path: Path, q_ids: list[str], g_ids: list[str],
                     scores: np.ndarray, threshold: float) -> dict:
    """candidates.csv: все кандидаты со скором >= порога; запрос без строк = отказ.

    Порог обязан быть на той же шкале, что и скоры (см. config.DEFAULT_THRESHOLD
    для косинуса и config.DEFAULT_THRESHOLD_RERANK для переранжирования).
    """
    validate_scores(scores, threshold)
    n_accepted = n_refused = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "gallery_id", "confidence"])
        for i, qid in enumerate(q_ids):
            accepted = accepted_candidates(scores[i], threshold)
            if accepted:
                n_accepted += 1
                for j in accepted:
                    w.writerow([qid, g_ids[j], f"{scores[i, j]:.6f}"])
            else:
                n_refused += 1
    return {
        "threshold": threshold,
        "queries_with_candidates": n_accepted,
        "queries_refused": n_refused,
        "refused_share": round(n_refused / len(q_ids), 4) if q_ids else 0.0,
    }
