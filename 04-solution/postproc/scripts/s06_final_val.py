#!/usr/bin/env python3
"""Шаг 6: финальные честные числа на нетронутом вал-сплите.

Все параметры/комбинации зафиксированы по протоколу подбора (train_fit):
  rerank-only:  (6,3,0.3)                        [grid_tune_base]
  TTA:          208+208f+256                     [tta_tune]
  TTA+rerank:   (10,3,0.45)                      [grid_tune_tta]
  TTA2:         208+256 (дешёвый, 2 прохода)     [tta_tune, 2-й по mAP среди 2-проходных]
  TTA2+rerank:  (10,3,0.3)                       [grid_tune_tta2]
  плюс безподборные: rerank(20,6,0.3) и TTA+rerank(20,6,0.3).
Выход: out/final_val.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import JOB, SPLIT, read_meta, eval_scores, eval_embeddings, rerank

qm = read_meta(SPLIT / "val_query.csv")
gm = read_meta(SPLIT / "val_gallery.csv")


def load(part, tag):
    return np.load(JOB / f"out/val_{part}_{tag}.npy")


configs = {
    "base": ("208", None),
    "rerank_paper": ("208", (20, 6, 0.3)),
    "rerank_tuned": ("208", (6, 3, 0.3)),
    "tta3": ("TTA", None),
    "tta3_rerank_tuned": ("TTA", (10, 3, 0.45)),
    "tta3_rerank_paper": ("TTA", (20, 6, 0.3)),
    "tta2": ("TTA2", None),
    "tta2_rerank_tuned": ("TTA2", (10, 3, 0.3)),
}

out = {}
for name, (tag, params) in configs.items():
    q, g = load("query", tag), load("gallery", tag)
    if params is None:
        m = eval_embeddings(q, g, qm, gm)
        rr_s = 0.0
    else:
        dist, rr_s = rerank(q, g, *params)
        m = eval_scores(-dist, qm, gm)
    out[name] = {"vectors": tag, "rerank_params": params,
                 "rerank_seconds": round(rr_s, 2), **m}
    print(f"{name}: mAP={m['mAP']:.5f} R1={m['Rank-1']:.5f} mAP@10={m['mAP@10']:.5f}",
          flush=True)

(JOB / "out/final_val.json").write_text(json.dumps(out, indent=2) + "\n")
