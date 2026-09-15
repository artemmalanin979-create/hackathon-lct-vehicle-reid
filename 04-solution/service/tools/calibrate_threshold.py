#!/usr/bin/env python3
"""Калибровка порога режима отказа — обе шкалы, правило одно.

Не входит в сдаваемый контур сервиса (в образ не копируется): это инструмент
перепроверки чисел README.

Методика не изобретается: берётся обоснованная в `04-solution/refusal/` — её же
адаптер (`refusal/scripts/metric_adapter.py`, импорт без изменений) поверх
неизменённого контура `04-solution/eval/reid_metrics.py`. Правило выбора
зафиксировано до просмотра результатов: максимум от
min(TNR, F1 при долях отказных 0.10/0.25/0.40).

Векторы — те, что написал пакетный прогон сервиса на сплите; скоры — кодом
сервиса (`app.core.ranking` / `app.core.rerank`); метрики — контуром.

    python tools/calibrate_threshold.py /путь/к/embeddings.npy [каталог_вывода]

Выбранный порог печатается в headline.json: шкала косинуса -> DEFAULT_THRESHOLD,
шкала переранжирования -> DEFAULT_THRESHOLD_RERANK в app/core/config.py.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

SERVICE = Path(__file__).resolve().parents[1]
SOLUTION = SERVICE.parent
SPLIT = SOLUTION / "split/files"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else SERVICE / "tools/out"
sys.path.insert(0, str(SOLUTION / "eval"))
sys.path.insert(0, str(SOLUTION / "refusal/scripts"))
sys.path.insert(0, str(SERVICE))
import reid_metrics as rm  # noqa: E402
from metric_adapter import Sweep, metric_kernel  # noqa: E402
from app.core import config  # noqa: E402
from app.core.ranking import cosine_scores  # noqa: E402
from app.core.rerank import rerank_distances, distances_to_scores  # noqa: E402

PRIORS = (0.10, 0.25, 0.40)
FIELDS = ("threshold", "f1", "precision", "recall", "tnr", "tp", "fp", "fn",
          "tn_unknown", "fp_unknown", "auc_pr_trapezoid", "ap_pr_step")
OLD_COSINE_THRESHOLD = 0.5495953464415451
OLD_DRAFT_THRESHOLD = 0.34921352213815304


def read_meta(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    emb = np.load(Path(sys.argv[1]), allow_pickle=False)
    qm, gm = read_meta(SPLIT / "val_query.csv"), read_meta(SPLIT / "val_gallery.csv")
    nq, ng = len(qm), len(gm)
    assert emb.shape == (nq + ng, 512), emb.shape
    q, g = emb[:nq], emb[nq:]

    qids = np.array([r["vehicle_id"] for r in qm])
    gids = np.array([r["vehicle_id"] for r in gm])
    known = np.array([r["has_mate"] == "1" for r in qm])
    assert np.array_equal(known, np.isin(qids, gids)), "has_mate расходится с галереей"
    kwargs = dict(query_ids=qids, gallery_ids=gids,
                  query_cameras=[r["camera_id"] for r in qm],
                  gallery_cameras=[r["camera_id"] for r in gm], known_absent=~known)

    kernel, kernel_source = metric_kernel(rm)
    (OUT / "metric_kernel_extracted.txt").write_text(kernel_source)

    scales = {
        "cosine": cosine_scores(q, g),
        "rerank": distances_to_scores(
            rerank_distances(q, g, config.RERANK_K1, config.RERANK_K2, config.RERANK_LAMBDA)),
    }
    summary, verification = {}, []
    for scale, scores in scales.items():
        for policy in ("market", "all_same_camera", "unfiltered"):
            # "unfiltered" — как работает пакетный прогон на выданном тесте: камер
            # там нет вовсе, значит камерный фильтр ничего не убирает.
            options = dict(kwargs)
            if policy == "unfiltered":
                options.update(query_cameras=["q"] * nq, gallery_cameras=["g"] * ng)
            actual_policy = "market" if policy == "unfiltered" else policy
            base = rm.evaluate(scores, **options, threshold=-1e9, camera_policy=actual_policy,
                               refusal_mode="presence", include_rankings=True)
            assert base["counts"]["filtered_queries"] == 0
            rankings = [row["ranking"] for row in base["per_query"]]
            first = np.array([r[0] for r in rankings])
            s1 = scores[np.arange(nq), first]
            top_correct = known & (gids[first] == qids)
            for mode in ("presence", "top1"):
                tag = f"{scale}_{policy}_{mode}"
                sweep = Sweep(rm, kernel, s1, known, top_correct, mode)
                priors = {str(p): Sweep(rm, kernel, s1, known, top_correct, mode, prior=p)
                          for p in PRIORS}
                write_csv(OUT / f"curve_{tag}.csv", sweep.rows, FIELDS)
                robust = max(sweep.rows, key=lambda r: (
                    min([r["tnr"]] + [ps.at(r["threshold"])["f1"] for ps in priors.values()]),
                    r["f1"], r["tnr"]))
                selected = {
                    "accept_all": sweep.at(float(np.min(s1))),
                    "reject_all": sweep.at(float(np.nextafter(np.max(s1), np.inf))),
                    "best_f1": sweep.best_f1(),
                    "balanced": sweep.balanced(),
                    "robust_balanced": robust,
                    **{f"tnr_{i/10:.1f}": sweep.target_tnr(i / 10) for i in range(11)},
                }
                if scale == "cosine":
                    selected["old_calibrated"] = sweep.at(OLD_COSINE_THRESHOLD)
                    selected["old_draft_argmax_f1"] = sweep.at(OLD_DRAFT_THRESHOLD)
                write_csv(OUT / f"selected_{tag}.csv",
                          [dict(selection=k, **v) for k, v in selected.items()],
                          ["selection", *FIELDS])
                summary[tag] = {
                    "selected": selected,
                    "auc_pr_trapezoid": sweep.curve["auc_pr_trapezoid"],
                    "ap_pr_step": sweep.curve["ap_pr_step"],
                    "num_operating_points": len(sweep.rows),
                    "priors_at_robust": {p: ps.at(robust["threshold"]) for p, ps in priors.items()},
                    "score_quantiles": {
                        label: np.quantile(s1[mask], [0, .1, .25, .5, .7, .9, 1]).tolist()
                        for label, mask in (("known", known), ("unknown", ~known))},
                }
                # Контрольная сверка выбранных точек напрямую неизменённым контуром.
                for cutoff in sorted({row["threshold"] for row in selected.values()}):
                    direct = rm.evaluate(scores, **options, threshold=cutoff,
                                         camera_policy=actual_policy, refusal_mode=mode)["refusal"]
                    got = sweep.at(cutoff)
                    errors = {k: abs(got[k] - direct[k]) for k in FIELDS
                              if got[k] is not None and direct[k] is not None}
                    assert max(errors.values()) < 2e-12, (tag, cutoff, errors)
                    verification.append({"case": tag, "threshold": cutoff,
                                         "max_error": max(errors.values())})
            print(f"готово {scale}/{policy}", flush=True)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (OUT / "verification.json").write_text(json.dumps(
        {"checks": len(verification), "max_abs_error": max(v["max_error"] for v in verification),
         "detail": verification}, indent=2) + "\n")

    main_tag = "rerank_market_presence"
    old_tag = "cosine_market_presence"
    head = {
        "rerank_threshold": summary[main_tag]["selected"]["robust_balanced"]["threshold"],
        "rerank_point": summary[main_tag]["selected"]["robust_balanced"],
        "cosine_old_point": summary[old_tag]["selected"]["old_calibrated"],
        "cosine_rule_reproduced": summary[old_tag]["selected"]["robust_balanced"],
        "auc_pr": {"rerank": summary[main_tag]["auc_pr_trapezoid"],
                   "cosine": summary[old_tag]["auc_pr_trapezoid"]},
        "verification_checks": len(verification),
    }
    (OUT / "headline.json").write_text(json.dumps(head, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(head, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
