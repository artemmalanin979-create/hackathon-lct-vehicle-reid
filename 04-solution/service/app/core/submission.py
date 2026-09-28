"""Запись сдаваемых файлов: submission.csv, candidates.csv, embeddings.npy.

Формат сверяется с example_submission.zip и evaluate.py организаторов:
submission.csv без заголовка, candidates.csv с заголовком и confidence
с шестью знаками после запятой.
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
        # Organizer evaluate.py and example_submission.zip require no header here.
        for i, qid in enumerate(q_ids):
            top = ranked_indices(scores[i])[:TOP_K_SUBMISSION]
            w.writerow([qid] + [g_ids[j] for j in top])


def write_candidates(path: Path, q_ids: list[str], g_ids: list[str],
                     scores: np.ndarray, threshold: float) -> dict:
    """candidates.csv: все кандидаты со скором >= порога; запрос без строк = отказ.

    Порог обязан быть на той же шкале, что и скоры. Пакетная сдача передаёт
    косинусные оценки и config.DEFAULT_THRESHOLD независимо от ранжирования.
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
