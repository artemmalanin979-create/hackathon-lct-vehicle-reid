#!/usr/bin/env python3
"""Оценка дообученной модели измерительным контуром 04-solution/eval/reid_metrics.py.

Контур импортируется без модификаций, своего расчёта метрик нет. Прогоны те же,
что у бейзлайна (04-solution/baseline/scripts/run_eval.py):
  1) market — исключение совпадений с камеры запроса (главное число);
  2) БЕЗ исключения камеры — фиктивные непересекающиеся метки камер;
  3) случайные векторы той же размерности, seeds 1..3 — проверка, что мерим честно.
Дополнительно те же три прогона для склейки «новая модель + OSNet» и для самого OSNet
(контрольный пересчёт бейзлайна тем же кодом на том же сплите).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

JOB = Path(__file__).resolve().parent.parent
EVAL_DIR = Path("/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/eval")
SPLIT = Path("/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/split/files")
sys.path.insert(0, str(EVAL_DIR))
from reid_metrics import evaluate, scores_from_embeddings  # noqa: E402


def read_meta(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def check_ids(meta, ids_path: Path):
    ids = ids_path.read_text().split()
    assert ids == [r["image_id"] for r in meta], f"порядок строк {ids_path} != CSV"


def run(name, scores, qm, gm, *, threshold=0.0, fake_cameras=False):
    qc = [f"q:{i}" for i in range(len(qm))] if fake_cameras else [r["camera_id"] for r in qm]
    gc = [f"g:{i}" for i in range(len(gm))] if fake_cameras else [r["camera_id"] for r in gm]
    res = evaluate(
        scores,
        query_ids=[r["vehicle_id"] for r in qm],
        gallery_ids=[r["vehicle_id"] for r in gm],
        query_cameras=qc, gallery_cameras=gc,
        known_absent=np.array([r["has_mate"] == "0" for r in qm]),
        threshold=threshold, camera_policy="market", refusal_mode="presence",
    )
    res["run_name"] = name
    return res


def brief(res):
    f = res["ranking_full_gallery"]
    return {"mAP": f["mAP"], "Rank-1": f["Rank-1"], "Rank-5": f["Rank-5"],
            "mINP": f["mINP"], "valid_queries": f["num_valid_queries"],
            "filtered_queries": res["counts"]["filtered_queries"]}


def l2(x):
    x = x.astype(np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="new", help="суффикс файлов векторов в out/")
    ap.add_argument("--out", type=Path, default=JOB / "out" / "metrics_new.json")
    args = ap.parse_args()

    out = JOB / "out"
    qm, gm = read_meta(SPLIT / "val_query.csv"), read_meta(SPLIT / "val_gallery.csv")
    check_ids(qm, out / "val_query.ids")
    check_ids(gm, out / "val_gallery.ids")

    emb = {}
    for tag in (args.tag, "osnet"):
        q = out / f"val_query_{tag}.npy"
        g = out / f"val_gallery_{tag}.npy"
        if q.exists() and g.exists():
            emb[tag] = (np.load(q), np.load(g))

    results, summary = {}, {}
    for tag, (q, g) in emb.items():
        sc = scores_from_embeddings(q, g, metric="cosine")
        results[f"{tag}_market"] = run(f"{tag}_market", sc, qm, gm)
        results[f"{tag}_no_camera_excl"] = run(f"{tag}_no_camera_excl", sc, qm, gm,
                                               fake_cameras=True)

    # склейка двух моделей: конкатенация L2-нормированных векторов (равный вес)
    if args.tag != "osnet" and args.tag in emb and "osnet" in emb:
        qn, gn = emb[args.tag]
        qo, go = emb["osnet"]
        qc = l2(np.hstack([l2(qn), l2(qo)]))
        gc = l2(np.hstack([l2(gn), l2(go)]))
        sc = scores_from_embeddings(qc, gc, metric="cosine")
        results["concat_market"] = run("concat_market", sc, qm, gm)
        results["concat_no_camera_excl"] = run("concat_no_camera_excl", sc, qm, gm,
                                               fake_cameras=True)

    dim = emb[args.tag][0].shape[1]
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        rq = rng.standard_normal((len(qm), dim))
        rg = rng.standard_normal((len(gm), dim))
        results[f"random_seed{seed}"] = run(
            f"random_seed{seed}", scores_from_embeddings(rq, rg, metric="cosine"), qm, gm)

    for k, v in results.items():
        summary[k] = brief(v)
    rnd = [summary[f"random_seed{s}"] for s in (1, 2, 3)]
    summary["random_mean"] = {k: float(np.mean([r[k] for r in rnd]))
                              for k in ("mAP", "Rank-1", "Rank-5", "mINP")}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
