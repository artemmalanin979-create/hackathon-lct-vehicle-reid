"""job_42: общий код. Метрики — ТОЛЬКО контуром 04-solution/eval/reid_metrics.py (импорт без
модификаций), KR — 04-solution/postproc/scripts/common.rerank (тот же, что дал 0,694).
Здесь только обвязка: загрузка, per-query выгрузка, производные числа шага 0, бутстрэп."""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
import numpy as np

REPO = Path.home() / "lct-reid/repo"
EVAL = REPO / "04-solution/eval"
SPLIT = REPO / "04-solution/split/files"
BASE_OUT = REPO / "04-solution/baseline/out"
TUNE = REPO / "04-solution/postproc/tune"
JOB = Path.home() / "lct-reid/jobs/job_42"
OUT = JOB / "out"
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(REPO / "04-solution/postproc/scripts"))
from reid_metrics import evaluate, scores_from_embeddings  # noqa: E402
from common import rerank  # noqa: E402  (k-reciprocal, канон Zhong, float64)

KR = (6, 3, 0.3)  # параметры действующей системы (run_info.json)
BASELINE_MAP = 0.6936584724586873


def read_meta(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_val():
    qm, gm = read_meta(SPLIT / "val_query.csv"), read_meta(SPLIT / "val_gallery.csv")
    qids = (BASE_OUT / "val_query.ids").read_text().split()
    gids = (BASE_OUT / "val_gallery.ids").read_text().split()
    assert qids == [r["image_id"] for r in qm] and gids == [r["image_id"] for r in gm]
    q = np.load(BASE_OUT / "val_query.npy"); g = np.load(BASE_OUT / "val_gallery.npy")
    assert q.shape == (1110, 512) and g.shape == (750, 512)
    return q, g, qm, gm


def l2n(m):
    m = np.asarray(m, dtype=np.float64)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def run_eval(scores, qm, gm, *, fake_cameras=False, threshold=0.0):
    """Контур: market (исключить same-ID&same-camera), presence. fake_cameras=True —
    непересекающиеся фиктивные camera_id => ничего не исключается (как в baseline/run_eval.py)."""
    qc = [f"q:{i}" for i in range(len(qm))] if fake_cameras else [r["camera_id"] for r in qm]
    gc = [f"g:{i}" for i in range(len(gm))] if fake_cameras else [r["camera_id"] for r in gm]
    return evaluate(
        scores,
        query_ids=[r["vehicle_id"] for r in qm], gallery_ids=[r["vehicle_id"] for r in gm],
        query_cameras=qc, gallery_cameras=gc,
        known_absent=np.array([r.get("has_mate", "1") == "0" for r in qm]),
        threshold=threshold, camera_policy="market", refusal_mode="presence",
        include_rankings=True)


def per_query(res):
    """Плоская выгрузка per-query фактов контура (полная галерея + top-10 ветки)."""
    rows = res["per_query"]
    tk_f = res["per_query_top_k_by_filter_order"]["filter_then_top_k"]
    tk_s = res["per_query_top_k_by_filter_order"]["top_k_then_filter"]
    n = len(rows)
    d = dict(known=np.zeros(n, bool), ap=np.zeros(n), R=np.zeros(n, int),
             first=np.full(n, -1, int), pos_ranks=[None] * n,
             ap10_R_f=np.zeros(n), ap10_retr_f=np.zeros(n),
             ap10_R_s=np.zeros(n), ap10_retr_s=np.zeros(n), n_sub_s=np.zeros(n, int))
    for i, r in enumerate(rows):
        d["known"][i] = r["status"] == "known"
        d["R"][i] = r["num_relevant"]
        if r["status"] == "known":
            d["ap"][i] = r["ap"]
            d["first"][i] = r["positive_ranks"][0]
            d["pos_ranks"][i] = list(r["positive_ranks"])
            d["ap10_R_f"][i] = tk_f[i]["ap_by_denominator"]["all_gallery_positives"]
            d["ap10_retr_f"][i] = tk_f[i]["ap_by_denominator"]["retrieved_positives"]
            d["ap10_R_s"][i] = tk_s[i]["ap_by_denominator"]["all_gallery_positives"]
            d["ap10_retr_s"][i] = tk_s[i]["ap_by_denominator"]["retrieved_positives"]
        d["n_sub_s"][i] = tk_s[i]["candidate_count"]
    assert (res["counts"]["filtered_queries"] == 0)
    return d


def ap_at_k_from_ranks(pos_ranks, R, k=10, denom="R"):
    """AP@k (step) по позициям верных в кросс-камерном ранжире контура."""
    num = 0.0
    hit = 0
    for r in pos_ranks:
        if r > k:
            break
        hit += 1
        num += hit / r
    Z = R if denom == "R" else min(R, k)
    return num / Z


def ceilings(pos_ranks, R, M, k=10):
    """Потолки AP@10 при подъёме верных с позиций (10, M] в десятку.
    lift_min: поднятые занимают места самых НИЖНИХ неверных в десятке (ранги прочих не меняются).
    lift_max: поднятые занимают места самых ВЕРХНИХ неверных в десятке (ранги прочих не меняются).
    ideal:    все верные из top-M ставятся на 1..min(10, r_M) (U_M из ideas-08; при M=10 — чистая
              переупорядоченность десятки, резерв на ранних двойниках)."""
    in10 = [r for r in pos_ranks if r <= k]
    lifted = [r for r in pos_ranks if k < r <= M]
    rM = len(in10) + len(lifted)
    neg_slots = sorted(set(range(1, k + 1)) - set(in10))
    out = {}
    for name, slots in (("lift_min", neg_slots[::-1][:len(lifted)]), ("lift_max", neg_slots[:len(lifted)])):
        new = sorted(in10 + slots)
        num = sum((h + 1) / r for h, r in enumerate(new))
        for denom in ("R", "min"):
            Z = R if denom == "R" else min(R, k)
            out[f"{name}_{denom}"] = num / Z
    for denom in ("R", "min"):
        Z = R if denom == "R" else min(R, k)
        out[f"ideal_{denom}"] = min(k, rM) / Z
    out["n_lifted"] = len(lifted)
    return out


BUCKETS = [("1", 1, 1), ("2-5", 2, 5), ("6-10", 6, 10), ("11-20", 11, 20),
           ("21-50", 21, 50), ("51+", 51, 10**9)]


def step0_numbers(d, label):
    """Все числа шага 0 в двух шкалах: 832 (с парой) и 1110 (AP=0 у 278 без пары)."""
    known = d["known"]; n_all = len(known); n_known = int(known.sum())
    first = d["first"][known]
    dist = {name: int(((first >= lo) & (first <= hi)).sum()) for name, lo, hi in BUCKETS}
    dist["not_found"] = int((first < 0).sum())
    dist["no_pair_exists(1110-scale)"] = n_all - n_known
    # позиции ВСЕХ верных (не только первого)
    all_ranks = np.concatenate([np.array(p) for p, k in zip(d["pos_ranks"], known) if k])
    dist_all = {name: int(((all_ranks >= lo) & (all_ranks <= hi)).sum()) for name, lo, hi in BUCKETS}
    q_with_pos_11_20 = int(sum(any(10 < r <= 20 for r in p) for p, k in zip(d["pos_ranks"], known) if k))
    q_with_pos_11_50 = int(sum(any(10 < r <= 50 for r in p) for p, k in zip(d["pos_ranks"], known) if k))
    q_with_pos_21_50 = int(sum(any(20 < r <= 50 for r in p) for p, k in zip(d["pos_ranks"], known) if k))
    ap10_R = np.array([ap_at_k_from_ranks(p, R, 10, "R") if k else 0.0
                       for p, R, k in zip(d["pos_ranks"], d["R"], known)])
    ap10_min = np.array([ap_at_k_from_ranks(p, R, 10, "min") if k else 0.0
                         for p, R, k in zip(d["pos_ranks"], d["R"], known)])
    assert np.allclose(ap10_R[known], d["ap10_R_f"][known], atol=1e-12), "AP@10(R) != контур"
    ap20_R = np.array([ap_at_k_from_ranks(p, R, 20, "R") if k else 0.0 for p, R, k in zip(d["pos_ranks"], d["R"], known)])
    ap50_R = np.array([ap_at_k_from_ranks(p, R, 50, "R") if k else 0.0 for p, R, k in zip(d["pos_ranks"], d["R"], known)])

    def two_scales(v):
        return {"832": float(v[known].mean()), "1110": float(v.sum() / n_all)}

    rec = {}
    for K in (10, 20, 50):
        hit = (d["first"] > 0) & (d["first"] <= K)
        macro = np.array([np.mean([r <= K for r in p]) if k else 0.0 for p, k in zip(d["pos_ranks"], known)])
        rec[f"Recall@{K}_first_correct(CMC)"] = two_scales(hit.astype(float))
        rec[f"Recall@{K}_all_positives(macro)"] = two_scales(macro)
    ceil = {}
    for M in (10, 20, 50):
        cs = [ceilings(p, R, M) if k else None for p, R, k in zip(d["pos_ranks"], d["R"], known)]
        for key in ("lift_min_R", "lift_min_min", "lift_max_R", "lift_max_min", "ideal_R", "ideal_min"):
            v = np.array([c[key] if c else 0.0 for c in cs])
            ceil[f"M{M}_{key}"] = two_scales(v)
        ceil[f"M{M}_n_lifted_positives"] = int(sum(c["n_lifted"] for c in cs if c))
    return {
        "label": label, "n_queries": n_all, "n_with_pair": n_known,
        "positives_per_query": {"mean": float(d["R"][known].mean()), "min": int(d["R"][known].min()),
                                "max": int(d["R"][known].max()),
                                "hist": {str(k): int(v) for k, v in zip(*np.unique(d["R"][known], return_counts=True))}},
        "first_correct_position": dist,
        "all_correct_positions": dist_all,
        "queries_with_positive_at_11_20": q_with_pos_11_20,
        "queries_with_positive_at_21_50": q_with_pos_21_50,
        "queries_with_positive_at_11_50": q_with_pos_11_50,
        "mAP_full": two_scales(d["ap"]),
        "Rank-1": two_scales((d["first"] == 1).astype(float)),
        "recall": rec,
        "AP@10_denom_R": two_scales(ap10_R),
        "AP@10_denom_minR10": two_scales(ap10_min),
        "AP@10_denom_retrieved(контур, справочно)": two_scales(d["ap10_retr_f"]),
        "AP@10_submission_semantics_top10_then_filter_denom_R": two_scales(d["ap10_R_s"]),
        "AP@20_denom_R": two_scales(ap20_R), "AP@50_denom_R": two_scales(ap50_R),
        "ceilings_AP@10": ceil,
    }


def paired_bootstrap(a, b, B=4000, seed=20260916, groups=None):
    """a - b по среднему per-query значению; парный бутстрэп по запросам (или по группам-ID)."""
    a = np.asarray(a, float); b = np.asarray(b, float); n = len(a)
    assert len(b) == n
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, n, size=(B, n))
        da = a[idx].mean(1) - b[idx].mean(1)
    else:
        g = np.asarray(groups); ug = np.unique(g); members = [np.flatnonzero(g == u) for u in ug]
        da = np.empty(B)
        for t in range(B):
            pick = rng.integers(0, len(ug), size=len(ug))
            sel = np.concatenate([members[p] for p in pick])
            da[t] = a[sel].mean() - b[sel].mean()
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "delta": float(a.mean() - b.mean()),
            "ci95": [float(np.quantile(da, 0.025)), float(np.quantile(da, 0.975))],
            "p_two_sided": float(2 * min((da <= 0).mean(), (da >= 0).mean())),
            "sd_delta": float(da.std(ddof=1)), "n": int(n), "B": B, "seed": seed,
            "grouped_by_id": groups is not None}


def sd_of_mean(a, B=4000, seed=20260916):
    a = np.asarray(a, float); n = len(a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    return float(a[idx].mean(1).std(ddof=1))


def summary(res):
    f, t = res["ranking_full_gallery"], res["ranking_top_k_by_filter_order"]["filter_then_top_k"]
    return {"mAP": f["mAP"], "Rank-1": f["Rank-1"], "Rank-5": f["Rank-5"], "mINP": f["mINP"],
            "mAP@10(R)": t["mAP"], "valid": f["num_valid_queries"],
            "mAP_1110": f["mAP_by_query_average"]["all_queries_zero"]}


def dump_json(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n")
