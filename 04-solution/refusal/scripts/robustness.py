#!/usr/bin/env python3
"""Prior transfer, observable margin, and calibration uncertainty. No new metrics."""
from __future__ import annotations

import argparse
import importlib
import json
import sys

from analyze import ROOT, OUT, SNAPSHOT, FIELDS, PRIORS, write_csv, write_json, metadata, compare
from metric_adapter import Sweep, metric_kernel
import numpy as np


def quantiles(values):
    return dict(zip(("lo", "median", "hi"), map(float, np.quantile(values, [.025, .5, .975]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--cv-repeats", type=int, default=5)
    args = ap.parse_args()
    sys.path.insert(0, str(SNAPSHOT))
    rm = importlib.import_module("reid_metrics")
    kernel, _ = metric_kernel(rm)
    data = dict(np.load(OUT / "features.npz", allow_pickle=False))
    summary = json.loads((OUT / "summary.json").read_text())
    known, ids = data["known"], data["query_identity"]
    correct = data["market_correct_top1"]
    features = {"absolute": data["market_absolute"], "gap": data["market_gap"],
                "raw_gap": data["unfiltered_gap"]}
    qm, gm = metadata("val_query"), metadata("val_gallery")
    kwargs = dict(query_ids=ids, gallery_ids=np.array([r["vehicle_id"] for r in gm]),
                  query_cameras=np.array([r["camera_id"] for r in qm]),
                  gallery_cameras=np.array([r["camera_id"] for r in gm]), known_absent=~known)
    raw = rm.scores_from_embeddings(np.load(ROOT / "val_query.npy"), np.load(ROOT / "val_gallery.npy"))
    checks = []
    extra = {}
    for mode in ("presence", "top1"):
        sweep = Sweep(rm, kernel, features["raw_gap"], known, correct, mode)
        choices = {"best_f1": sweep.best_f1(), "balanced": sweep.balanced(),
                   "tnr_0.7": sweep.target_tnr(.7)}
        extra[mode] = {"selected": choices, "auc_pr_trapezoid": sweep.curve["auc_pr_trapezoid"],
                       "ap_pr_step": sweep.curve["ap_pr_step"]}
        write_csv(OUT / f"curve_market_raw_gap_{mode}.csv", sweep.rows, FIELDS)
        # This row translation sets the eligible top confidence to the observable
        # raw margin, while preserving the camera-filtered candidate ranking.
        translated = (raw - data["market_absolute"][:,None]) + features["raw_gap"][:,None]
        for row in choices.values():
            result = rm.evaluate(translated, **kwargs, threshold=row["threshold"],
                                 camera_policy="market", refusal_mode=mode)
            checks.append({"case": "raw_gap_"+mode, "max_error": compare(row, result["refusal"])})
    transfers, prior_summary = [], {}
    for feature, confidence in features.items():
        sweeps = {p: Sweep(rm, kernel, confidence, known, correct, "presence", prior=p) for p in PRIORS}
        prior_summary[feature] = {}
        for p, sw in sweeps.items():
            prior_summary[feature][str(p)] = {"best_f1": sw.best_f1(), "weights": sw.class_weights,
                                             "balanced": sw.balanced()}
            assert abs(sw.unknown_count / (sw.unknown_count + sw.positive_count) - p) < 1e-15
            if feature != "raw_gap":
                fixed = summary[f"market_{feature}_presence"]["selected"]["robust_balanced"]["threshold"]
                prior_summary[feature][str(p)]["fixed_recommendation"] = sw.at(fixed)
            for calibration_prior, calibration in sweeps.items():
                row = sw.at(calibration.best_f1()["threshold"])
                transfers.append(dict(feature=feature, calibration_prior=calibration_prior,
                                      evaluation_prior=p, **row,
                                      f1_regret=sw.best_f1()["f1"]-row["f1"]))
        reference = sweeps[.25]
        for p, sw in sweeps.items():
            max_error = max(abs(a[key]-b[key]) for a,b in zip(sw.rows,reference.rows)
                            for key in ("recall", "tnr"))
            assert max_error < 1e-14
            checks.append({"case": f"prior_invariance_{feature}_{p}", "max_error": max_error})
    write_csv(OUT / "prior_transfer.csv", transfers,
              ["feature", "calibration_prior", "evaluation_prior", *FIELDS, "f1_regret"])

    # A direct public-API check on repeated query rows verifies multiplicities.
    rng = np.random.default_rng(20260915)
    mult = rng.integers(0, 3, len(known))
    indexes = np.repeat(np.arange(len(known)), mult)
    repeated = dict(kwargs)
    for field in ("query_ids", "query_cameras", "known_absent"):
        repeated[field] = repeated[field][indexes]
    for mode in ("presence", "top1"):
        sw = Sweep(rm, kernel, features["absolute"], known, correct, mode, multiplicity=mult)
        cutoff = .55
        direct = rm.evaluate(raw[indexes], **repeated, threshold=cutoff,
                             camera_policy="market", refusal_mode=mode)
        checks.append({"case": "replication_"+mode, "max_error": compare(sw.at(cutoff), direct["refusal"])})

    # Bootstrap entire vehicle identities, stratified by mate presence.
    # The gallery and model stay fixed; repetitions are NOT new sample size.
    groups = {label: np.unique(ids[known == label]) for label in (True, False)}
    assert not set(groups[True]) & set(groups[False])
    fixed_thresholds = {feature: {name: summary[f"market_{feature}_presence"]["selected"][name]["threshold"]
                                 for name in ("best_f1", "tnr_0.5", "tnr_0.7", "robust_balanced")}
                        for feature in ("absolute", "gap")}
    bootstrap = []
    differences = []
    for rep in range(args.bootstrap):
        multiplicity = np.zeros(len(ids), dtype=np.int64)
        for label, members in groups.items():
            drawn, count = np.unique(rng.choice(members, len(members), replace=True), return_counts=True)
            for identity, weight in zip(drawn, count):
                multiplicity[ids == identity] = weight
        sweeps = {feature: Sweep(rm, kernel, features[feature], known, correct,
                                 multiplicity=multiplicity) for feature in ("absolute", "gap")}
        for feature, sw in sweeps.items():
            for selection, cutoff in fixed_thresholds[feature].items():
                bootstrap.append(dict(rep=rep, feature=feature, selection=selection, **sw.at(cutoff)))
        differences.append({
            "rep": rep,
            "auc_absolute_minus_gap": sweeps["absolute"].curve["auc_pr_trapezoid"] - sweeps["gap"].curve["auc_pr_trapezoid"],
            "f1_absolute_minus_gap_fixed_tnr07_cuts":
                sweeps["absolute"].at(fixed_thresholds["absolute"]["tnr_0.7"])["f1"] -
                sweeps["gap"].at(fixed_thresholds["gap"]["tnr_0.7"])["f1"],
        })
    write_csv(OUT / "bootstrap.csv", bootstrap, ["rep", "feature", "selection", *FIELDS])
    write_csv(OUT / "bootstrap_differences.csv", differences)
    intervals = {feature: {selection: {metric: quantiles([r[metric] for r in bootstrap
                    if r["feature"] == feature and r["selection"] == selection])
                    for metric in ("f1", "precision", "recall", "tnr")}
                    for selection in fixed_thresholds[feature]}
                    for feature in fixed_thresholds}
    intervals["differences"] = {field: quantiles([row[field] for row in differences])
                               for field in differences[0] if field != "rep"}
    print("bootstrap finished", flush=True)

    # Repeated five-fold calibration, identities disjoint between calibration
    # and evaluation queries. The embedding model is not refit in these folds.
    cv_rows, fold_rows = [], []
    for repeat in range(args.cv_repeats):
        fold_id = np.empty(len(ids), dtype=int)
        cv_rng = np.random.default_rng(20260915 + repeat)
        for label, members in groups.items():
            for position, identity in enumerate(cv_rng.permutation(members)):
                fold_id[ids == identity] = position % 5
        for feature in ("absolute", "gap"):
            centered = np.empty(len(ids))
            for fold in range(5):
                train, test = fold_id != fold, fold_id == fold
                assert not set(ids[train]) & set(ids[test])
                calibration = Sweep(rm, kernel, features[feature][train], known[train], correct[train], prior=.4)
                cutoff = calibration.balanced()["threshold"]
                centered[test] = features[feature][test] - cutoff
                heldout = Sweep(rm, kernel, features[feature][test], known[test], correct[test]).at(cutoff)
                fold_rows.append(dict(repeat=repeat, fold=fold, feature=feature,
                                      calibration_query_count=int(train.sum()), heldout_query_count=int(test.sum()), **heldout))
            pooled = Sweep(rm, kernel, centered, known, correct).at(0.)
            cv_rows.append(dict(repeat=repeat, feature=feature, **pooled))
    write_csv(OUT / "crossfit_folds.csv", fold_rows,
              ["repeat", "fold", "feature", "calibration_query_count", "heldout_query_count", *FIELDS])
    write_csv(OUT / "crossfit_pooled.csv", cv_rows, ["repeat", "feature", *FIELDS])
    cv_summary = {feature: {metric: {"mean": float(np.mean([r[metric] for r in cv_rows if r["feature"] == feature])),
                                    "min": float(np.min([r[metric] for r in cv_rows if r["feature"] == feature])),
                                    "max": float(np.max([r[metric] for r in cv_rows if r["feature"] == feature]))}
                             for metric in ("f1", "precision", "recall", "tnr")}
                  for feature in ("absolute", "gap")}
    for feature in cv_summary:
        cv_summary[feature]["calibration_thresholds"] = quantiles([r["threshold"] for r in fold_rows if r["feature"] == feature])
    recommendation = summary["market_absolute_presence"]["selected"]["robust_balanced"]
    actual_sw = Sweep(rm, kernel, features["absolute"], known, correct)
    write_json(OUT / "recommendation.json", {
        "feature": "max_cosine_over_eligible_gallery", "threshold": recommendation["threshold"],
        "accept_rule": ">=", "reject_rule": "<", "mode": "presence", "camera_policy": "market",
        "selection_rule": "maximize min(TNR, F1_at_p0.10, F1_at_p0.25, F1_at_p0.40)",
        "status": "provisional validation operating point; jury aggregation and camera protocol unknown",
        "actual_validation": recommendation,
        "nominal_prior_scenarios": prior_summary["absolute"],
        "rounded_6_decimals_check": actual_sw.at(round(recommendation["threshold"], 6)),
        "raw_service_refusals_on_validation": int(np.count_nonzero(data["unfiltered_absolute"] < recommendation["threshold"])),
        "eligible_list_refusals_on_validation": int(np.count_nonzero(features["absolute"] < recommendation["threshold"])),
        "bootstrap_95_percentile": intervals["absolute"]["robust_balanced"],
        "crossfit": cv_summary["absolute"],
    })
    output = {"seed": 20260915, "bootstrap_replicates": args.bootstrap, "cv_repeats": args.cv_repeats,
              "group_unit": "vehicle_id", "fixed_gallery": True,
              "observable_raw_gap": extra, "prior_summary": prior_summary,
              "bootstrap_intervals": intervals, "crossfit_summary": cv_summary,
              "verification": checks}
    write_json(OUT / "robustness.json", output)
    print(json.dumps({"crossfit": cv_summary, "intervals": intervals["absolute"]["robust_balanced"],
                      "observable_raw_gap": extra}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
