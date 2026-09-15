#!/usr/bin/env python3
"""Сквозная проверка сплита измерительным контуром на двух вырожденных входах.

1) СЛУЧАЙНЫЕ эмбеддинги → метрики обязаны совпасть со случайным уровнем,
   посчитанным аналитически из состава НАШЕЙ галереи (формулы ORACLES.md):
     E[AP_step] = H_N/N + (M-1)(N-H_N)/(N(N-1));  E[Rank-k] = 1 - C(N-M,k)/C(N,k);
     E[INP] = sum_{r=M..N} (M/r) C(r-1,M-1)/C(N,M),
   где N — допустимые кандидаты запроса после камерной маски, M — верные среди них.
   Допуск: |z| <= 5 стандартных ошибок среднего (как в ORACLES.md).
   Порог отказа 0.0: max из ~749 косинусов < 0 с вероятностью ~2^-749 → ожидание:
   все запросы приняты, recall=1, TNR=0, precision = доля известных = 832/1110.

2) ИДЕАЛЬНЫЕ эмбеддинги (one-hot идентичности) → все ранжирующие метрики и все
   показатели отказа при пороге 0.5 обязаны быть ровно 1.0.

Контур не модифицируется: reid_metrics импортируется из 04-solution/eval (без записи
байткода в чужую папку); дополнительно тот же вход прогоняется через CLI reid_metrics
и сверяется с импортным вызовом.
"""
import json
import math
import os
import subprocess
import sys
from functools import lru_cache

import numpy as np

sys.dont_write_bytecode = True
EVAL_DIR = "/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/eval"
sys.path.insert(0, EVAL_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import reid_metrics as RM

RUN_DIR = os.path.join(C.OUT, "evalrun")
PROBLEMS = []


def load_split(split_dir):
    import csv
    with open(os.path.join(split_dir, "val_query.csv"), newline="") as f:
        q = list(csv.DictReader(f))
    with open(os.path.join(split_dir, "val_gallery.csv"), newline="") as f:
        g = list(csv.DictReader(f))
    qv = np.array([int(r["vehicle_id"]) for r in q])
    qc = np.array([int(r["camera_id"]) for r in q])
    gv = np.array([int(r["vehicle_id"]) for r in g])
    gc = np.array([int(r["camera_id"]) for r in g])
    absent = np.array([r["has_mate"] == "0" for r in q])
    return q, g, qv, qc, gv, gc, absent


def harmonic(n):
    return float(sum(1.0 / r for r in range(1, n + 1)))


@lru_cache(maxsize=None)
def expected_query(N, M):
    """(E[AP_step], E[R1], E[R5], E[INP]) для случайной перестановки."""
    if M == 0:
        return None
    if N == 1:
        return (1.0, 1.0, 1.0, 1.0)
    H = harmonic(N)
    ap = H / N + (M - 1) * (N - H) / (N * (N - 1))
    r1 = M / N
    k = min(5, N)
    r5 = 1.0 - math.comb(N - M, k) / math.comb(N, k) if N - M >= k else 1.0
    cnm = math.comb(N, M)
    inp = sum((M / r) * math.comb(r - 1, M - 1) for r in range(M, N + 1)) / cnm
    return (ap, r1, r5, inp)


def analytic(qv, qc, gv, gc, absent, policy):
    exp_rows = []
    ng = len(gv)
    for i in range(len(qv)):
        if absent[i]:
            continue
        same_id = gv == qv[i]
        same_cam = gc == qc[i]
        M = int((same_id & ~same_cam).sum())
        if policy == "market":
            N = ng - int((same_id & same_cam).sum())
        else:
            N = ng - int(same_cam.sum())
        exp_rows.append(expected_query(N, M))
    arr = np.array(exp_rows)
    return {"mAP": float(arr[:, 0].mean()), "Rank-1": float(arr[:, 1].mean()),
            "Rank-5": float(arr[:, 2].mean()), "mINP": float(arr[:, 3].mean()),
            "n_known": len(exp_rows)}


def observed_stats(result):
    per = [r for r in result["per_query"] if r["ap"] is not None]
    def col(k):
        v = np.array([r[k] for r in per], dtype=np.float64)
        sem = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0
        return float(v.mean()), sem
    return {k: col(k2) for k, k2 in
            [("mAP", "ap"), ("Rank-1", "rank1"), ("Rank-5", "rank5"), ("mINP", "inp")]}


def note(cond, msg):
    status = "ok " if cond else "ПРОБЛЕМА"
    print(f"  [{status}] {msg}")
    if not cond:
        PROBLEMS.append(msg)


def run_random(qv, qc, gv, gc, absent, seed, policy):
    rng = np.random.default_rng(seed)
    E = rng.standard_normal((len(qv) + len(gv), 128))
    scores = RM.scores_from_embeddings(E[:len(qv)], E[len(qv):])
    res = RM.evaluate(scores, qv, gv, qc, gc, absent, threshold=0.0,
                      camera_policy=policy, refusal_mode="presence")
    exp = analytic(qv, qc, gv, gc, absent, policy)
    obs = observed_stats(res)
    row = {"seed": seed, "policy": policy, "counts": res["counts"],
           "expected": exp, "observed": {}, "z": {}}
    print(f"-- случайные эмбеддинги, seed={seed}, policy={policy}")
    for k in ["mAP", "Rank-1", "Rank-5", "mINP"]:
        o, sem = obs[k]
        z = (o - exp[k]) / sem if sem else 0.0
        row["observed"][k] = o
        row["z"][k] = round(z, 2)
        note(abs(z) <= 5, f"{k}: наблюдаемое {o:.4f}, ожидаемое {exp[k]:.4f}, "
                          f"z={z:+.2f} (допуск |z|<=5)")
    ref = res["refusal"]
    note(res["counts"]["filtered_queries"] == 0,
         f"filtered_queries=0 (все пары кросс-камерные): {res['counts']['filtered_queries']}")
    note(ref["recall"] == 1.0 and ref["tnr"] == 0.0,
         f"порог 0: recall={ref['recall']}, TNR={ref['tnr']} (ожидание 1 и 0)")
    prev = res["counts"]["known_queries"] / res["counts"]["refusal_queries"]
    note(abs(ref["precision"] - prev) < 1e-9,
         f"precision={ref['precision']:.4f} = доля известных {prev:.4f}")
    note(abs(ref["auc_pr"] - prev) < 0.05,
         f"AUC-PR={ref['auc_pr']:.4f} ~ базовый уровень {prev:.4f} (случайные score)")
    row["refusal"] = {k: ref[k] for k in
                      ["precision", "recall", "tnr", "auc_pr", "tp", "fp", "fn"]}
    return row, res, scores


def run_ideal(qv, qc, gv, gc, absent, policy):
    vids = sorted(set(qv.tolist()) | set(gv.tolist()))
    vmap = {v: i for i, v in enumerate(vids)}
    Q = np.zeros((len(qv), len(vids)))
    G = np.zeros((len(gv), len(vids)))
    Q[np.arange(len(qv)), [vmap[v] for v in qv]] = 1.0
    G[np.arange(len(gv)), [vmap[v] for v in gv]] = 1.0
    scores = RM.scores_from_embeddings(Q, G)
    res = RM.evaluate(scores, qv, gv, qc, gc, absent, threshold=0.5,
                      camera_policy=policy, refusal_mode="presence")
    print(f"-- идеальные эмбеддинги (one-hot {len(vids)} идентичностей), policy={policy}")
    row = {"policy": policy, "ranking": res["ranking"], "refusal": {}}
    for k in ["mAP", "Rank-1", "Rank-5", "mINP"]:
        v = res["ranking"][k]
        note(abs(v - 1.0) < 1e-12, f"{k} = {v} (ожидание ровно 1)")
    ref = res["refusal"]
    for k in ["precision", "recall", "f1", "tnr"]:
        row["refusal"][k] = ref[k]
        note(ref[k] == 1.0, f"отказ {k} = {ref[k]} (ожидание ровно 1)")
    note(ref["fp"] == 0 and ref["fn"] == 0, f"fp={ref['fp']}, fn={ref['fn']} (ожидание 0)")
    note(res["counts"]["filtered_queries"] == 0, "filtered_queries = 0")
    return row


def run_cli_crosscheck(qv, qc, gv, gc, absent, scores, imported_res):
    npz = os.path.join(RUN_DIR, "random_seed1.npz")
    np.savez(npz, scores=scores, query_ids=qv, gallery_ids=gv,
             query_cameras=qc, gallery_cameras=gc, known_absent=absent)
    out = os.path.join(RUN_DIR, "cli_random_seed1.json")
    subprocess.run(
        [sys.executable, "-B", os.path.join(EVAL_DIR, "reid_metrics.py"), npz,
         "--threshold", "0.0", "--camera-policy", "market",
         "--refusal-mode", "presence", "--output", out],
        check=True, cwd=RUN_DIR)
    with open(out) as f:
        cli = json.load(f)
    same = all(abs(cli["ranking"][k] - imported_res["ranking"][k]) < 1e-12
               for k in ["mAP", "Rank-1", "Rank-5", "mINP"])
    print("-- CLI reid_metrics.py на том же входе")
    note(same, "CLI и импортный вызов дали одинаковое ранжирование")
    return {"cli_equals_import": same}


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    split_dir = sys.argv[1] if len(sys.argv) > 1 else C.SPLIT_DIR
    q, g, qv, qc, gv, gc, absent = load_split(split_dir)
    report = {"split_dir": split_dir,
              "counts": {"queries": len(qv), "gallery": len(gv),
                         "known": int((~absent).sum()), "unknown": int(absent.sum())}}
    report["random"] = []
    keep = None
    for seed in [1, 2, 3]:
        row, res, scores = run_random(qv, qc, gv, gc, absent, seed, "market")
        report["random"].append(row)
        if seed == 1:
            keep = (res, scores)
    row, _, _ = run_random(qv, qc, gv, gc, absent, 1, "all_same_camera")
    report["random"].append(row)
    report["ideal"] = [run_ideal(qv, qc, gv, gc, absent, "market"),
                       run_ideal(qv, qc, gv, gc, absent, "all_same_camera")]
    report["cli"] = run_cli_crosscheck(qv, qc, gv, gc, absent, keep[1], keep[0])
    report["problems"] = PROBLEMS
    with open(os.path.join(RUN_DIR, "degenerate_results.json"), "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    if PROBLEMS:
        print(f"\nНАХОДКИ ({len(PROBLEMS)}) — см. degenerate_results.json")
        sys.exit(1)
    print("\nOK: обе вырожденные проверки сошлись с ожиданиями")


if __name__ == "__main__":
    main()
