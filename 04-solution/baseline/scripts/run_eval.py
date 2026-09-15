#!/usr/bin/env python3
"""Оценка на проверочном сплите измерительным контуром 04-solution/eval/reid_metrics.py.

Контур используется импортом, без модификаций. Прогоны:
  1) OSNet базовый, camera_policy=market (исключение одноместных совпадений);
  2) случайные векторы N(0,1) dim=512, seeds 1..3, market;
  3) OSNet базовый БЕЗ исключения камеры (камерам запросов/галереи даны
     непересекающиеся фиктивные метки "q:*"/"g:*" — same_camera всегда False;
     это использование контракта evaluate, а не его модификация);
  4) OSNet базовый, camera_policy=all_same_camera (справочно);
  5) абляции пластины: mask30 / gray / mask30+gray, market.

Порог отказа: argmax F1 по PR-кривой самого контура (режим presence) на базовом
прогоне; затем контур перезапускается с этим порогом — его refusal-метрики и
берутся в отчёт.
"""
from __future__ import annotations

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


def run(name, scores, qm, gm, *, threshold, camera_policy, fake_cameras=False):
    qcams = [f"q:{i}" for i in range(len(qm))] if fake_cameras else [r["camera_id"] for r in qm]
    gcams = [f"g:{i}" for i in range(len(gm))] if fake_cameras else [r["camera_id"] for r in gm]
    res = evaluate(
        scores,
        query_ids=[r["vehicle_id"] for r in qm],
        gallery_ids=[r["vehicle_id"] for r in gm],
        query_cameras=qcams, gallery_cameras=gcams,
        known_absent=np.array([r["has_mate"] == "0" for r in qm]),
        threshold=threshold, camera_policy=camera_policy, refusal_mode="presence",
    )
    res["run_name"] = name
    return res


def brief(res):
    """Обе ветки контура: полная галерея и усечение до K=10 (как в submission.csv)."""
    f = res["refusal"]
    full, topk = res["ranking_full_gallery"], res["ranking_top_k"]

    def block(r):
        return {"mAP": r["mAP"], "Rank-1": r["Rank-1"], "Rank-5": r["Rank-5"], "mINP": r["mINP"]}

    return {
        "full_gallery": block(full),
        "top_k10": {**block(topk),
                    "mINP_zero_if_incomplete":
                        topk["mINP_by_incomplete_policy"]["zero_if_incomplete"],
                    "num_incomplete_queries": topk["num_incomplete_queries"]},
        "valid_queries": full["num_valid_queries"],
        "filtered_queries": res["counts"]["filtered_queries"],
        "threshold": f["threshold"], "F1": f["f1"], "TNR": f["tnr"],
        "precision": f["precision"], "recall": f["recall"], "auc_pr": f["auc_pr"],
    }


def best_f1_threshold(res):
    """argmax F1 по PR-кривой контура (presence: события — top-1 каждого запроса)."""
    curve = res["refusal"]["pr_curve"]
    positives = curve["positive_count"]
    best_t, best_f1 = None, -1.0
    for t, tp, fp in zip(curve["thresholds"], curve["tp"], curve["fp"]):
        if t is None:
            continue
        fn = positives - tp
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def save(res, path: Path):
    drop = {"per_query", "per_query_top_k", "per_query_top_k_by_filter_order"}
    slim = {k: v for k, v in res.items() if k not in drop}
    slim["refusal"] = {k: v for k, v in slim["refusal"].items() if k != "pr_curve"}
    if "refusal_by_filtered_positive_policy" in slim:
        slim["refusal_by_filtered_positive_policy"] = {
            p: {"counts": b["counts"],
                "refusal": {k: v for k, v in b["refusal"].items() if k != "pr_curve"}}
            for p, b in slim["refusal_by_filtered_positive_policy"].items()}
    path.write_text(json.dumps(slim, indent=2) + "\n")


def main():
    out = JOB / "out"
    qm = read_meta(SPLIT / "val_query.csv")
    gm = read_meta(SPLIT / "val_gallery.csv")
    check_ids(qm, out / "val_query.ids")
    check_ids(gm, out / "val_gallery.ids")

    emb = {v: (np.load(out / f"val_query{s}.npy"), np.load(out / f"val_gallery{s}.npy"))
           for v, s in [("base", ""), ("mask30", "_mask30"), ("gray", "_gray"),
                        ("mask30gray", "_mask30gray")]}
    scores = {v: scores_from_embeddings(q, g, metric="cosine") for v, (q, g) in emb.items()}

    results = {}

    # 1) базовый market при пороге 0 -> выбор порога по PR-кривой контура -> перезапуск
    base0 = run("base_market_t0", scores["base"], qm, gm, threshold=0.0, camera_policy="market")
    t_star, f1_star = best_f1_threshold(base0)
    results["base_market"] = run("base_market", scores["base"], qm, gm,
                                 threshold=t_star, camera_policy="market")

    # 2) случайные векторы: та же размерность 512, seeds 1..3
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        rq = rng.standard_normal((len(qm), 512))
        rg = rng.standard_normal((len(gm), 512))
        results[f"random_seed{seed}"] = run(
            f"random_seed{seed}", scores_from_embeddings(rq, rg, metric="cosine"),
            qm, gm, threshold=0.0, camera_policy="market")

    # 3) без исключения камеры (фиктивные непересекающиеся camera_id)
    results["base_no_camera_excl"] = run("base_no_camera_excl", scores["base"], qm, gm,
                                         threshold=t_star, camera_policy="market",
                                         fake_cameras=True)

    # 4) справочно: all_same_camera
    results["base_all_same_camera"] = run("base_all_same_camera", scores["base"], qm, gm,
                                          threshold=t_star, camera_policy="all_same_camera")

    # 5) абляции пластины (market, порог тот же для сопоставимости)
    for v in ("mask30", "gray", "mask30gray"):
        results[f"ablate_{v}"] = run(f"ablate_{v}", scores[v], qm, gm,
                                     threshold=t_star, camera_policy="market")

    evalrun = out / "evalrun"
    evalrun.mkdir(exist_ok=True)
    for name, res in results.items():
        save(res, evalrun / f"{name}.json")

    summary = {name: brief(res) for name, res in results.items()}
    rnd = [summary[f"random_seed{s}"] for s in (1, 2, 3)]

    def mean_block(scope):
        out = {}
        for k in ("mAP", "Rank-1", "Rank-5", "mINP"):
            vals = [r[scope][k] for r in rnd]
            out[k] = None if any(v is None for v in vals) else float(np.mean(vals))
        return out

    summary["random_mean"] = {"full_gallery": mean_block("full_gallery"),
                              "top_k10": mean_block("top_k10")}
    summary["_threshold_selection"] = {
        "method": "argmax F1 по PR-кривой контура (refusal_mode=presence, top-1 cos), базовый прогон market",
        "t_star": t_star, "f1_at_selection": f1_star,
    }
    (out / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
