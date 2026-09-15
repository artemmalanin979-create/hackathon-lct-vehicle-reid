#!/usr/bin/env python3
"""Synchronous threshold calibration; all writes remain beside this script."""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True
import numpy as np

from metric_adapter import Sweep, metric_kernel

# ROOT — каталог этапа (scripts/ лежит внутри него): там же лежат входные векторы
# и туда пишутся results/, metadata/, evaluator_snapshot/ — как в исходном прогоне.
ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT.parents[1]  # корень репозитория
SNAPSHOT = ROOT / "evaluator_snapshot"
META = ROOT / "metadata"
OUT = ROOT / "results"
PRIORS = (0.10, 0.25, 0.40)
FIELDS = ("threshold", "f1", "precision", "recall", "tnr", "tp", "fp", "fn", "tn_unknown", "fp_unknown", "auc_pr_trapezoid", "ap_pr_step")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = list(fields or rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def snapshot(source, destination):
    if not destination.exists():
        shutil.copyfile(source, destination)
    if source.exists() and hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(destination.read_bytes()).digest():
        raise RuntimeError(f"Source changed after snapshot: {source}")


def metadata(name):
    with (META / (name + ".csv")).open() as f:
        rows = list(csv.DictReader(f))
    ids = (ROOT / (name + ".ids")).read_text().splitlines()
    assert ids == [r["image_id"] for r in rows], f"ID order mismatch: {name}"
    assert len(ids) == len(set(ids))
    return rows


def slim(result):
    return {
        "protocol": result["protocol"], "counts": result["counts"],
        "ranking_full_gallery": result["ranking_full_gallery"],
        "ranking_top_k_by_filter_order": result["ranking_top_k_by_filter_order"],
        "refusal": {k: v for k, v in result["refusal"].items() if k != "pr_curve"},
    }


def compare(expected, observed):
    errors = {}
    for key in FIELDS:
        a, b = expected[key], observed[key]
        if a is None or b is None:
            assert a is b, (key, a, b)
        else:
            errors[key] = abs(a - b)
            assert errors[key] < 2e-12, (key, a, b)
    return max(errors.values(), default=0.)


def main():
    for folder in (SNAPSHOT, META, OUT):
        folder.mkdir(exist_ok=True)
    for name in ("reid_metrics.py", "scope_metrics.py"):
        snapshot(PROJECT / "04-solution/eval" / name, SNAPSHOT / name)
    for name in ("val_query.csv", "val_gallery.csv", "manifest.json"):
        snapshot(PROJECT / "04-solution/split/files" / name, META / name)
    sys.path.insert(0, str(SNAPSHOT))
    rm = importlib.import_module("reid_metrics")
    kernel, kernel_source = metric_kernel(rm)
    (OUT / "metric_kernel_extracted.txt").write_text(kernel_source)
    qm, gm = metadata("val_query"), metadata("val_gallery")
    q = np.load(ROOT / "val_query.npy", allow_pickle=False)
    g = np.load(ROOT / "val_gallery.npy", allow_pickle=False)
    assert q.shape == (1110, 512) and g.shape == (750, 512)
    assert np.isfinite(q).all() and np.isfinite(g).all()
    scores = rm.scores_from_embeddings(q, g)
    qids = np.array([r["vehicle_id"] for r in qm])
    gids = np.array([r["vehicle_id"] for r in gm])
    known = np.array([r["has_mate"] == "1" for r in qm])
    assert np.array_equal(known, np.isin(qids, gids))
    assert not set(r["image_id"] for r in qm) & set(r["image_id"] for r in gm)
    kwargs = dict(query_ids=qids, gallery_ids=gids,
                  query_cameras=[r["camera_id"] for r in qm],
                  gallery_cameras=[r["camera_id"] for r in gm], known_absent=~known)
    inputs = [ROOT / name for name in ("val_query.npy", "val_gallery.npy", "val_query.ids", "val_gallery.ids", "context-tz.txt", "context-split.md")]
    # sorted: порядок ключей манифеста не должен зависеть от порядка каталога ФС
    inputs += sorted(SNAPSHOT.glob("*.py")) + sorted(META.iterdir())
    manifest = {
        "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "python": platform.python_version(), "numpy": np.__version__,
        "queries": len(q), "gallery": len(g), "known": int(known.sum()), "unknown": int((~known).sum()),
        "known_identities": len(set(qids[known])), "unknown_identities": len(set(qids[~known])),
        "actual_refusal_share": float((~known).mean()),
        "max_l2_norm_error": {"query": float(np.max(np.abs(np.linalg.norm(q.astype(float), axis=1)-1))),
                              "gallery": float(np.max(np.abs(np.linalg.norm(g.astype(float), axis=1)-1)))},
        "score_method": "reid_metrics.scores_from_embeddings(metric=cosine), float64",
        "prior_method": "exact integer replication; same conditional empirical distributions and same gallery",
        "prior_assumption": "Only P(has_mate) changes; P(score | has_mate) is held fixed. Not fresh splits.",
    }
    write_json(OUT / "manifest.json", manifest)
    all_summary, verification = {}, []
    feature_arrays = {"known": known, "query_identity": qids}
    for policy in ("market", "all_same_camera", "unfiltered"):
        options = dict(kwargs)
        if policy == "unfiltered":
            options.update(query_cameras=["q"] * len(q), gallery_cameras=["g"] * len(g))
        actual_policy = policy if policy != "unfiltered" else "market"
        base = rm.evaluate(scores, **options, threshold=0., camera_policy=actual_policy,
                           refusal_mode="presence", include_rankings=True)
        assert base["counts"]["filtered_queries"] == 0
        write_json(OUT / f"ranking_{policy}.json", slim(base))
        rankings = [row["ranking"] for row in base["per_query"]]
        assert min(map(len, rankings)) >= 2
        first = np.array([r[0] for r in rankings])
        second = np.array([r[1] for r in rankings])
        s1, s2 = scores[np.arange(len(q)), first], scores[np.arange(len(q)), second]
        top_correct = known & (gids[first] == qids)
        feature_arrays[policy + "_correct_top1"] = top_correct
        raw_top_k = base["per_query_top_k_by_filter_order"]["top_k_then_filter"]
        topk_check = {
            "min_eligible_candidates_in_raw_top10": min(r["candidate_count"] for r in raw_top_k),
            "top1_changed_by_raw_top10": sum(not r["ranking"] or r["ranking"][0] != rankings[i][0] for i, r in enumerate(raw_top_k)),
            "top2_changed_by_raw_top10": sum(r["ranking"][:2] != rankings[i][:2] for i, r in enumerate(raw_top_k)),
            "refusal_scope_reported_by_evaluator": base["protocol"]["refusal_scope"],
        }
        write_json(OUT / f"top10_check_{policy}.json", topk_check)
        per_query_rows = [dict(image_id=qm[i]["image_id"], vehicle_id=qids[i], camera_id=qm[i]["camera_id"],
                               has_mate=int(known[i]), top1_id=gm[first[i]]["image_id"],
                               top1_vehicle_id=gids[first[i]], correct_top1=int(top_correct[i]),
                               top1_score=float(s1[i]), top2_score=float(s2[i]), gap=float(s1[i]-s2[i]))
                          for i in range(len(q))]
        write_csv(OUT / f"per_query_{policy}.csv", per_query_rows)
        for feature, confidence in (("absolute", s1), ("gap", s1-s2)):
            feature_arrays[policy + "_" + feature] = confidence
            for mode in ("presence", "top1"):
                tag = f"{policy}_{feature}_{mode}"
                sweep = Sweep(rm, kernel, confidence, known, top_correct, mode)
                priors = {str(p): Sweep(rm, kernel, confidence, known, top_correct, mode, prior=p) for p in PRIORS}
                write_csv(OUT / f"curve_{tag}.csv", sweep.rows, FIELDS)
                for p, ps in priors.items():
                    write_csv(OUT / f"curve_{tag}_prior_{p}.csv", ps.rows, FIELDS)
                robust = max(sweep.rows, key=lambda r: (
                    min([r["tnr"]] + [ps.at(r["threshold"])["f1"] for ps in priors.values()]),
                    r["f1"], r["tnr"]))
                selected = {"accept_all": sweep.at(float(np.min(confidence))),
                            "reject_all": sweep.at(float(np.nextafter(np.max(confidence), np.inf))),
                            "best_f1": sweep.best_f1(), "balanced": sweep.balanced(),
                            "robust_balanced": robust,
                            **{f"tnr_{i/10:.1f}": sweep.target_tnr(i/10) for i in range(11)}}
                if feature == "absolute":
                    selected["old_exact"] = sweep.at(0.34921352213815304)
                    selected["old_rounded"] = sweep.at(0.349214)
                write_csv(OUT / f"selected_{tag}.csv", [dict(selection=k, **v) for k,v in selected.items()], ["selection", *FIELDS])
                grid = np.linspace(-1., 1., 41) if feature == "absolute" else np.linspace(0., 1., 41)
                write_csv(OUT / f"grid_{tag}.csv", [sweep.at(t) for t in grid], FIELDS)
                summary = {"selected": selected, "auc_pr_trapezoid": sweep.curve["auc_pr_trapezoid"],
                           "ap_pr_step": sweep.curve["ap_pr_step"], "num_operating_points": len(sweep.rows),
                           "score_quantiles": {label: np.quantile(confidence[mask], [0,.1,.25,.5,.7,.9,1]).tolist()
                                               for label,mask in (("known",known),("unknown",~known))},
                           "priors": {p: {"weights": ps.class_weights, "best_f1": ps.best_f1(),
                                           "balanced": ps.balanced(),
                                           "fixed_best_f1_actual": ps.at(selected["best_f1"]["threshold"]),
                                           "fixed_robust_balanced": ps.at(robust["threshold"]),
                                           "fixed_tnr_0.7": ps.at(selected["tnr_0.7"]["threshold"])}
                                      for p,ps in priors.items()}}
                all_summary[tag] = summary
                # Directly verify all report operating points with the unmodified
                # public evaluator; gap is a row translation preserving rankings.
                unique_cuts = sorted({row["threshold"] for row in selected.values()})
                transformed = scores if feature == "absolute" else scores-s2[:,None]
                for cutoff in unique_cuts:
                    direct = rm.evaluate(transformed, **options, threshold=cutoff,
                                         camera_policy=actual_policy, refusal_mode=mode)
                    error = compare(sweep.at(cutoff), direct["refusal"])
                    verification.append(dict(case=tag, threshold=cutoff, max_error=error))
                # PR curve of the evaluator is also identical at every event.
                assert sweep.curve == direct["refusal"]["pr_curve"]
        print(f"finished {policy}", flush=True)
    np.savez_compressed(OUT / "features.npz", **feature_arrays)
    write_json(OUT / "summary.json", all_summary)
    write_json(OUT / "verification.json", {"checks": verification, "count": len(verification),
                                         "max_abs_error": max(r["max_error"] for r in verification)})
    print(json.dumps({"market_absolute_presence": all_summary["market_absolute_presence"],
                      "market_gap_presence": all_summary["market_gap_presence"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
