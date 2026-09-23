#!/usr/bin/env python3
"""Сборка сдаваемых артефактов на выданном тесте.

  artifacts/embeddings.npy  — float32 (1860, 512): сперва все image_id из
                              test_query.csv по порядку файла, затем все из
                              test_gallery.csv по порядку файла;
  artifacts/submission.csv  — query_id, gallery_id_1..gallery_id_10 (по убыванию cos);
  artifacts/candidates.csv  — query_id, gallery_id, confidence: все кандидаты
                              с cos >= порога, по убыванию; запрос без строк = отказ.

Порог передаётся аргументом (берётся из out/metrics_summary.json — argmax F1
на валидационном сплите, см. run_eval.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from inputs import require_files, VECTOR_HINT

JOB = Path(__file__).resolve().parent.parent
REPO = Path(__file__).resolve().parents[3]  # корень репозитория (путь считается от файла, а не зашит)
DATA = Path(os.environ.get("REID_DATA_DIR", REPO / "data"))
sys.path.insert(0, str(REPO / "04-solution/eval"))


def read_ids(csv_path: Path):
    with open(csv_path, newline="") as f:
        return [r["image_id"] for r in csv.DictReader(f)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, required=True)
    args = ap.parse_args()
    require_files(
        [DATA / f"test_{part}.csv" for part in ("query", "gallery")]
        + [JOB / f"out/test_{part}.{ext}" for part in ("query", "gallery")
           for ext in ("ids", "npy")], hint=VECTOR_HINT)

    import numpy as np
    from reid_metrics import scores_from_embeddings

    out, art = JOB / "out", JOB / "artifacts"
    art.mkdir(exist_ok=True)

    q_ids_csv = read_ids(DATA / "test_query.csv")
    g_ids_csv = read_ids(DATA / "test_gallery.csv")
    q_ids_run = (out / "test_query.ids").read_text().split()
    g_ids_run = (out / "test_gallery.ids").read_text().split()
    assert q_ids_run == q_ids_csv, "порядок строк test_query при извлечении != CSV"
    assert g_ids_run == g_ids_csv, "порядок строк test_gallery при извлечении != CSV"

    q = np.load(out / "test_query.npy")
    g = np.load(out / "test_gallery.npy")
    assert q.shape == (len(q_ids_csv), 512) and g.shape == (len(g_ids_csv), 512)

    np.save(art / "embeddings.npy", np.concatenate([q, g]).astype(np.float32))

    scores = scores_from_embeddings(q, g, metric="cosine")  # 1110 x 750

    with open(art / "submission.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id"] + [f"gallery_id_{k}" for k in range(1, 11)])
        for i, qid in enumerate(q_ids_csv):
            top = np.argsort(-scores[i], kind="stable")[:10]
            w.writerow([qid] + [g_ids_csv[j] for j in top])

    n_accepted = n_refused = 0
    with open(art / "candidates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "gallery_id", "confidence"])
        for i, qid in enumerate(q_ids_csv):
            order = np.argsort(-scores[i], kind="stable")
            accepted = [j for j in order if scores[i, j] >= args.threshold]
            if accepted:
                n_accepted += 1
                for j in accepted:
                    w.writerow([qid, g_ids_csv[j], f"{scores[i, j]:.6f}"])
            else:
                n_refused += 1

    info = {
        "threshold": args.threshold,
        "queries_with_candidates": n_accepted, "queries_refused": n_refused,
        "refused_share": round(n_refused / len(q_ids_csv), 4),
        "embeddings_shape": [int(x) for x in np.load(art / "embeddings.npy").shape],
    }
    (art / "threshold.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
