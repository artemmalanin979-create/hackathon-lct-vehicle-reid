#!/usr/bin/env python3
"""Шаг 6b: контрольная 2-fold схема из брифа — вал пополам по идентичностям.

Подбор (быстрая сетка) на половине A -> объявление на половине B, и наоборот.
Обе половины — самостоятельные подзадачи (запросы и галерея только своих
идентичностей). Разница «оптимум на подборе − на отложенной» = цена переподгонки
на статистически одинаковых данных. Выход: out/halves.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (JOB, SPLIT, read_meta, eval_embeddings, eval_scores,
                    rerank, rerank_prepare, rerank_finalize)
from s04_grid import run_grid, K1S_FAST, K2S_FAST, LAMS_FAST

SEED = 20260915

qm = read_meta(SPLIT / "val_query.csv")
gm = read_meta(SPLIT / "val_gallery.csv")
q = np.load(JOB / "out/val_query_208.npy")
g = np.load(JOB / "out/val_gallery_208.npy")

paired_ids = sorted({r["vehicle_id"] for r in gm}, key=int)
refusal_ids = sorted({r["vehicle_id"] for r in qm if r["has_mate"] == "0"}, key=int)
rng = np.random.default_rng(SEED)
rng.shuffle(paired_ids)
rng.shuffle(refusal_ids)
halves = {}
for name, pi, ri in [("A", paired_ids[::2], refusal_ids[::2]),
                     ("B", paired_ids[1::2], refusal_ids[1::2])]:
    ids = set(pi) | set(ri)
    qi = [i for i, r in enumerate(qm) if r["vehicle_id"] in ids]
    gi = [i for i, r in enumerate(gm) if r["vehicle_id"] in set(pi)]
    halves[name] = {
        "qm": [qm[i] for i in qi], "gm": [gm[i] for i in gi],
        "q": q[qi], "g": g[gi],
        "n_ids_paired": len(pi), "n_ids_refusal": len(ri),
    }
    print(f"половина {name}: {len(qi)} запросов / {len(gi)} галерея "
          f"({len(pi)} paired + {len(ri)} refusal ид.)", flush=True)

out = {"seed": SEED, "folds": {}}
for tune_on, test_on in [("A", "B"), ("B", "A")]:
    t, v = halves[tune_on], halves[test_on]
    grid = run_grid(f"valhalf_{tune_on}", t["q"], t["g"], t["qm"], t["gm"],
                    K1S_FAST, K2S_FAST, LAMS_FAST)
    best = grid["best"]
    dist, _ = rerank(v["q"], v["g"], best["k1"], best["k2"], best["lam"])
    held = eval_scores(-dist, v["qm"], v["gm"])
    base_t = eval_embeddings(t["q"], t["g"], t["qm"], t["gm"])
    base_v = eval_embeddings(v["q"], v["g"], v["qm"], v["gm"])
    dist_p, _ = rerank(v["q"], v["g"], 20, 6, 0.3)
    paper_v = eval_scores(-dist_p, v["qm"], v["gm"])
    out["folds"][f"tune{tune_on}_test{test_on}"] = {
        "sizes": {"tune_q": len(t["qm"]), "tune_g": len(t["gm"]),
                  "test_q": len(v["qm"]), "test_g": len(v["gm"])},
        "best_params": {k: best[k] for k in ("k1", "k2", "lam")},
        "base_on_tune_half": base_t, "base_on_test_half": base_v,
        "tuned_mAP_on_tune_half": best["mAP"], "tuned_R1_on_tune_half": best["Rank-1"],
        "tuned_on_test_half": held, "paper_params_on_test_half": paper_v,
    }
    print(json.dumps(out["folds"][f"tune{tune_on}_test{test_on}"], indent=2), flush=True)

(JOB / "out/halves.json").write_text(json.dumps(out, indent=2) + "\n")
