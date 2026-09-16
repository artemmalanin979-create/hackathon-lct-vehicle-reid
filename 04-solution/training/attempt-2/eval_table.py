#!/usr/bin/env python3
"""Итоговая таблица прогонов. Метрики — ТОЛЬКО через 04-solution/eval/reid_metrics.py.

Переранжирование берётся готовой канонической реализацией из
04-solution/postproc/scripts/common.py (порт Zhong et al. 2017) — импортом,
без изменений. Свой расчёт метрик не пишется нигде.

Строки: база / +переранжирование / без исключения камеры / случайные векторы.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

JOB = Path(__file__).resolve().parent.parent
REPO = Path("/home/artem/projects/hackathon-lct-vehicle-reid")
EVAL = REPO / "04-solution/eval"
SPLIT = REPO / "04-solution/split/files"
POSTPROC = REPO / "04-solution/postproc/scripts"
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(POSTPROC))
from reid_metrics import evaluate, scores_from_embeddings  # noqa: E402
from common import rerank  # noqa: E402  (канонический k-reciprocal, не наш код)


def rd(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def check_ids(meta, ids_path: Path):
    ids = ids_path.read_text().split()
    assert ids == [r["image_id"] for r in meta], f"порядок строк {ids_path} != CSV"


def run(scores, qm, gm, *, fake_cameras=False):
    qc = [f"q:{i}" for i in range(len(qm))] if fake_cameras else [r["camera_id"] for r in qm]
    gc = [f"g:{i}" for i in range(len(gm))] if fake_cameras else [r["camera_id"] for r in gm]
    r = evaluate(scores,
                 query_ids=[r["vehicle_id"] for r in qm],
                 gallery_ids=[r["vehicle_id"] for r in gm],
                 query_cameras=qc, gallery_cameras=gc,
                 known_absent=np.array([r["has_mate"] == "0" for r in qm]),
                 threshold=0.0, camera_policy="market", refusal_mode="presence")
    f = r["ranking_full_gallery"]
    return ({"mAP": f["mAP"], "Rank-1": f["Rank-1"], "Rank-5": f["Rank-5"],
             "mINP": f["mINP"], "valid_queries": f["num_valid_queries"]},
            np.array([row["ap"] for row in r["per_query"] if row["status"] == "known"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["osnet"])
    ap.add_argument("--rerank", nargs="*", default=["6,3,0.3"],
                    help="наборы k1,k2,lam через запятую")
    ap.add_argument("--out", type=Path, default=JOB / "out" / "table.json")
    ap.add_argument("--aps-out", type=Path, default=None)
    args = ap.parse_args()

    qm, gm = rd(SPLIT / "val_query.csv"), rd(SPLIT / "val_gallery.csv")
    check_ids(qm, JOB / "out" / "val_query.ids")
    check_ids(gm, JOB / "out" / "val_gallery.ids")

    table, aps = {}, {}
    for tag in args.tags:
        q = np.load(JOB / "out" / f"val_query_{tag}.npy")
        g = np.load(JOB / "out" / f"val_gallery_{tag}.npy")
        sc = scores_from_embeddings(q, g, metric="cosine")
        table[f"{tag}"], aps[f"{tag}"] = run(sc, qm, gm)
        table[f"{tag}_no_camera_excl"], _ = run(sc, qm, gm, fake_cameras=True)
        for spec in args.rerank:
            k1, k2, lam = spec.split(",")
            d, secs = rerank(q, g, int(k1), int(k2), float(lam))
            key = f"{tag}_rerank_{k1}_{k2}_{lam}"
            table[key], aps[key] = run(-d, qm, gm)
            table[key]["rerank_seconds"] = round(secs, 2)
            table[f"{key}_no_camera_excl"], _ = run(-d, qm, gm, fake_cameras=True)
        print(f"[{tag}] готово", flush=True)

    dim = np.load(JOB / "out" / f"val_query_{args.tags[0]}.npy").shape[1]
    rnd = []
    for seed in (1, 2, 3):
        rg = np.random.default_rng(seed)
        m, _ = run(scores_from_embeddings(rg.standard_normal((len(qm), dim)),
                                          rg.standard_normal((len(gm), dim)),
                                          metric="cosine"), qm, gm)
        table[f"random_seed{seed}"] = m
        rnd.append(m)
    table["random_mean"] = {k: float(np.mean([r[k] for r in rnd]))
                            for k in ("mAP", "Rank-1", "Rank-5", "mINP")}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=2) + "\n")
    if args.aps_out:
        np.savez(args.aps_out, **aps)
    for k, v in table.items():
        print(f"{k:44s} mAP={v['mAP']:.4f} R1={v['Rank-1']:.4f}")


if __name__ == "__main__":
    main()
