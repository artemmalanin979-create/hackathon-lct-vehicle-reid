#!/usr/bin/env python3
"""Шаг 4: сетка параметров k-reciprocal на заданном наборе векторов.

Использование:
  s04_grid.py <label> <query.npy> <gallery.npy> <query_meta.csv> <gallery_meta.csv> [--fast]

Сетка: k1 x k2 x lambda (факторизовано: prepare на k1, finalize на k2, blend на lambda).
Вывод: out/grid_<label>.json — все точки + argmax по mAP (полная галерея, valid_queries).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import JOB, read_meta, eval_scores, rerank_prepare, rerank_finalize

K1S = [4, 6, 8, 10, 12, 16, 20, 26]
K2S = [1, 2, 3, 4, 6, 8]
LAMS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.8]
K1S_FAST = [4, 6, 8, 12, 20]
K2S_FAST = [1, 2, 3, 4, 6]
LAMS_FAST = [0.15, 0.3, 0.45, 0.6]


def run_grid(label, q, g, qm, gm, k1s=K1S, k2s=K2S, lams=LAMS, out_dir=None):
    t_start = time.perf_counter()
    points = []
    for k1 in k1s:
        original_dist, initial_rank, V = rerank_prepare(q, g, k1)
        for k2 in k2s:
            if k2 > k1:
                continue
            dists = rerank_finalize(original_dist, initial_rank, V, len(q), k2, lams)
            for lam, dist in dists.items():
                m = eval_scores(-dist, qm, gm)
                points.append({"k1": k1, "k2": k2, "lam": lam,
                               "mAP": m["mAP"], "Rank-1": m["Rank-1"],
                               "mAP@10": m["mAP@10"]})
        print(f"[{label}] k1={k1} готово, {time.perf_counter()-t_start:.0f} c",
              flush=True)
    best = max(points, key=lambda p: p["mAP"])
    result = {"label": label, "n_points": len(points), "best": best,
              "elapsed_s": round(time.perf_counter() - t_start, 1),
              "points": points}
    if out_dir is None:
        out_dir = JOB / "out"
    (out_dir / f"grid_{label}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[{label}] best: {json.dumps(best)}", flush=True)
    return result


if __name__ == "__main__":
    label, qp, gp, qmp, gmp = sys.argv[1:6]
    fast = "--fast" in sys.argv
    q, g = np.load(qp), np.load(gp)
    qm, gm = read_meta(Path(qmp)), read_meta(Path(gmp))
    if fast:
        run_grid(label, q, g, qm, gm, K1S_FAST, K2S_FAST, LAMS_FAST)
    else:
        run_grid(label, q, g, qm, gm)
