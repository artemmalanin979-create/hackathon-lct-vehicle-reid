#!/usr/bin/env python3
"""Метрики ранжирования на проверочном сплите — через измерительный контур.

Не входит в сдаваемый контур сервиса (в образ не копируется): это инструмент
перепроверки чисел README. Скоры считаются кодом сервиса, метрики — контуром
`04-solution/eval/reid_metrics.py` без изменений; собственных формул здесь нет.

    python tools/eval_split.py /путь/к/embeddings.npy

где embeddings.npy — выход `python -m app.batch` на файлах
`04-solution/split/files/val_query.csv` и `val_gallery.csv`.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

SERVICE = Path(__file__).resolve().parents[1]
SOLUTION = SERVICE.parent
sys.path.insert(0, str(SOLUTION / "eval"))
sys.path.insert(0, str(SERVICE))
from reid_metrics import evaluate  # noqa: E402
from app.core import config  # noqa: E402
from app.core.ranking import cosine_scores  # noqa: E402
from app.core.rerank import distances_to_scores, rerank_distances  # noqa: E402


def read_meta(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarize(result: dict) -> dict:
    full, topk = result["ranking_full_gallery"], result["ranking_top_k"]
    return {"mAP": full["mAP"], "Rank-1": full["Rank-1"], "Rank-5": full["Rank-5"],
            "mAP@10": topk["mAP"], "valid_queries": full["num_valid_queries"]}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    split = SOLUTION / "split/files"
    emb = np.load(Path(sys.argv[1]), allow_pickle=False)
    qm, gm = read_meta(split / "val_query.csv"), read_meta(split / "val_gallery.csv")
    if emb.shape != (len(qm) + len(gm), config.EMBEDDING_DIM):
        raise SystemExit(f"{emb.shape}: это не embeddings.npy прогона на сплите")
    q, g = emb[: len(qm)], emb[len(qm):]
    kwargs = dict(query_ids=[r["vehicle_id"] for r in qm],
                  gallery_ids=[r["vehicle_id"] for r in gm],
                  query_cameras=[r["camera_id"] for r in qm],
                  gallery_cameras=[r["camera_id"] for r in gm],
                  known_absent=np.array([r["has_mate"] == "0" for r in qm]),
                  camera_policy="market", refusal_mode="presence")

    distances = rerank_distances(q, g, config.RERANK_K1, config.RERANK_K2, config.RERANK_LAMBDA)
    out = {
        "no_rerank": summarize(evaluate(cosine_scores(q, g), threshold=0.0, **kwargs)),
        # шкала сервиса: уверенность 1 - дистанция (порог на ранжирование не влияет)
        "rerank": summarize(evaluate(distances_to_scores(distances), threshold=-1e9, **kwargs)),
        # та же матрица как дистанция — штатный режим контура, контроль шкалы
        "rerank_as_distance": summarize(
            evaluate(distances, threshold=float("inf"), score_kind="distance", **kwargs)),
        "rerank_params": [config.RERANK_K1, config.RERANK_K2, config.RERANK_LAMBDA],
    }
    out["scales_agree"] = out["rerank"] == out["rerank_as_distance"]
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
