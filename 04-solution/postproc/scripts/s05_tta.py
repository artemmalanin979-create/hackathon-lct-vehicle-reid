#!/usr/bin/env python3
"""Шаг 5: TTA-комбинации на наборе (tune либо val).

Для каждой комбинации: среднее L2-нормированных векторов вариантов -> L2 -> косинус ->
метрики контура. Выход: out/tta_<set>.json. Выбор комбинации делается ТОЛЬКО по tune.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import JOB, SPLIT, read_meta, eval_embeddings, average_embeddings

COMBOS = [
    ["208"],
    ["208", "208f"],
    ["208", "256"],
    ["208", "288"],
    ["256", "256f"],
    ["208", "256", "288"],
    ["208", "208f", "256"],
    ["208", "208f", "256", "256f"],
    ["208", "256", "288", "208f"],          # комбинация из разбора
    ["208", "208f", "256", "256f", "288", "288f"],
]


def load(setname, part, tag):
    return np.load(JOB / f"out/{setname}_{part}_{tag}.npy")


def main(setname):
    if setname == "val":
        qm = read_meta(SPLIT / "val_query.csv")
        gm = read_meta(SPLIT / "val_gallery.csv")
    else:
        qm = read_meta(JOB / "tune/tune_query.csv")
        gm = read_meta(JOB / "tune/tune_gallery.csv")
    out = []
    for combo in COMBOS:
        t0 = time.perf_counter()
        q = average_embeddings([load(setname, "query", t) for t in combo])
        g = average_embeddings([load(setname, "gallery", t) for t in combo])
        m = eval_embeddings(q, g, qm, gm)
        out.append({"combo": "+".join(combo), "passes": len(combo), **m})
        print(f"[{setname}] {'+'.join(combo)}: mAP={m['mAP']:.5f} R1={m['Rank-1']:.5f} "
              f"({time.perf_counter()-t0:.1f} c)", flush=True)
    best = max(out, key=lambda p: p["mAP"])
    (JOB / f"out/tta_{setname}.json").write_text(
        json.dumps({"set": setname, "best": best, "runs": out}, indent=2) + "\n")
    print(f"[{setname}] best: {json.dumps(best)}")


if __name__ == "__main__":
    main(sys.argv[1])
