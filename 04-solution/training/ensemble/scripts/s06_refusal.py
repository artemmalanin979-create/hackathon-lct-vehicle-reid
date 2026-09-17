#!/usr/bin/env python3
"""job_43, шаг 5: калибровка порога отказа тем же правилом, что в 04-solution/refusal/ и
service/tools/calibrate_threshold.py: максимум min(TNR, F1 при долях отказных 0,10/0,25/0,40).
Адаптер refusal/scripts/metric_adapter.py импортируется без изменений поверх контура.
Конфигурации: osnet (контроль: должны воспроизвестись 0,5496→0,733/0,698 и 0,4994→0,806/0,777),
ainv2, avg (лучший ансамбль), cat_w1.0; шкалы: косинус и уверенность 1−d переранжирования.
"""
import csv, json, sys
from pathlib import Path
import numpy as np

HOME = Path.home() / "lct-reid"; REPO = HOME / "repo"; S = REPO / "04-solution"
OUT = HOME / "jobs/job_43/out"
sys.path.insert(0, str(S / "eval")); sys.path.insert(0, str(S / "postproc/scripts")); sys.path.insert(0, str(S / "refusal/scripts"))
import reid_metrics as rm  # noqa
from reid_metrics import scores_from_embeddings  # noqa
from common import rerank  # noqa
from metric_adapter import Sweep, metric_kernel  # noqa

A2 = S / "training/attempt-2/out"
K1, K2, LAM = 6, 3, 0.3
PRIORS = (0.10, 0.25, 0.40)
FIELDS = ("threshold", "f1", "precision", "recall", "tnr", "tp", "fp", "fn", "tn_unknown", "fp_unknown", "auc_pr_trapezoid")
OLD = {"cos": 0.5495953464415451, "rr": 0.49937235233589916}

def rd(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))
qm, gm = rd(S / "split/files/val_query.csv"), rd(S / "split/files/val_gallery.csv")
nq, ng = len(qm), len(gm)
qids = np.array([r["vehicle_id"] for r in qm]); gids = np.array([r["vehicle_id"] for r in gm])
known = np.array([r["has_mate"] == "1" for r in qm])
assert np.array_equal(known, np.isin(qids, gids))
kwargs = dict(query_ids=qids, gallery_ids=gids, query_cameras=[r["camera_id"] for r in qm],
              gallery_cameras=[r["camera_id"] for r in gm], known_absent=~known)
kernel, _ = metric_kernel(rm)

def l2(m):
    m = m.astype(np.float64); return m / np.linalg.norm(m, axis=1, keepdims=True)
def load(tag):
    return l2(np.load(A2 / f"val_query_{tag}.npy")), l2(np.load(A2 / f"val_gallery_{tag}.npy"))
base, v2 = load("osnet"), load("ainv2")
configs = {"osnet": base, "ainv2": v2,
           "avg": tuple(l2(b + a) for b, a in zip(base, v2)),
           "cat_w1.0": tuple(np.concatenate([b, a], axis=1) for b, a in zip(base, v2))}

def calibrate(scores):
    base_eval = rm.evaluate(scores, **kwargs, threshold=-1e9, camera_policy="market",
                            refusal_mode="presence", include_rankings=True)
    assert base_eval["counts"]["filtered_queries"] == 0
    first = np.array([row["ranking"][0] for row in base_eval["per_query"]])
    s1 = scores[np.arange(nq), first]
    top_correct = known & (gids[first] == qids)
    sweep = Sweep(rm, kernel, s1, known, top_correct, "presence")
    priors = {str(p): Sweep(rm, kernel, s1, known, top_correct, "presence", prior=p) for p in PRIORS}
    robust = max(sweep.rows, key=lambda r: (min([r["tnr"]] + [ps.at(r["threshold"])["f1"] for ps in priors.values()]), r["f1"], r["tnr"]))
    def pick(row): return {k: row[k] for k in FIELDS}
    out = {"robust": pick(robust),
           "robust_f1_at_priors": {p: ps.at(robust["threshold"])["f1"] for p, ps in priors.items()},
           "best_f1": pick(sweep.best_f1()),
           "tnr_0.7": pick(sweep.target_tnr(0.7)), "tnr_0.8": pick(sweep.target_tnr(0.8)),
           "auc_pr_trapezoid": sweep.curve["auc_pr_trapezoid"], "num_operating_points": len(sweep.rows)}
    # контрольная сверка выбранной точки напрямую контуром
    direct = rm.evaluate(scores, **kwargs, threshold=robust["threshold"], camera_policy="market", refusal_mode="presence")["refusal"]
    out["direct_check_max_err"] = max(abs(direct[k] - robust[k]) for k in ("f1", "precision", "recall", "tnr"))
    return out, s1

res = {}
for name, (q, g) in configs.items():
    res[name] = {}
    cos = scores_from_embeddings(q, g, metric="cosine")
    d, _ = rerank(q, g, K1, K2, LAM)
    for scale, sc in (("cos", cos), ("rr", 1.0 - d)):
        r, s1 = calibrate(sc)
        old = rm.evaluate(sc, **kwargs, threshold=OLD[scale], camera_policy="market", refusal_mode="presence")["refusal"]
        r["old_base_threshold_applied"] = {"threshold": OLD[scale], **{k: old[k] for k in ("f1", "precision", "recall", "tnr", "tp", "fp", "fn", "tn_unknown", "fp_unknown")}}
        r["score_quantiles"] = {lab: np.quantile(s1[m], [0, .1, .25, .5, .75, .9, 1]).tolist() for lab, m in (("known", known), ("unknown", ~known))}
        res[name][scale] = r
        rb = r["robust"]
        print(f"[{name:9s} {scale}] t={rb['threshold']:.6f} F1={rb['f1']:.4f} P={rb['precision']:.4f} R={rb['recall']:.4f} TNR={rb['tnr']:.4f} "
              f"AUC-PR={r['auc_pr_trapezoid']:.4f} | F1@TNR0.7={r['tnr_0.7']['f1']:.4f} F1@TNR0.8={r['tnr_0.8']['f1']:.4f} | "
              f"старый порог {OLD[scale]:.4f}: F1={old['f1']:.4f} TNR={old['tnr']:.4f} | check {r['direct_check_max_err']:.1e}", flush=True)
(OUT / "s06_refusal.json").write_text(json.dumps(res, indent=2) + "\n")
print("DONE")
