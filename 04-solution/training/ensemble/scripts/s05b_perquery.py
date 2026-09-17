#!/usr/bin/env python3
"""Пер-запросная картина: у скольких запросов AP вырос / упал у ансамбля против базы+rr."""
import json
from pathlib import Path
import numpy as np
OUT = Path.home() / "lct-reid/jobs/job_43/out"
aps = np.load(OUT / "s04_aps.npz"); r1 = np.load(OUT / "s04_rank1.npz")
ref = aps["osnet|rr"]; ref1 = r1["osnet|rr"]
res = {}
for key in ("ainv2|rr", "cat_w1.0|rr", "avg|rr", "avg|cos", "ainv2|cos"):
    d = aps[key] - ref; d1 = r1[key] - ref1
    res[key] = {"ap_up": int((d > 1e-12).sum()), "ap_down": int((d < -1e-12).sum()), "ap_same": int((np.abs(d) <= 1e-12).sum()),
                "mean_gain_when_up": float(d[d > 1e-12].mean()) if (d > 1e-12).any() else None,
                "mean_loss_when_down": float(d[d < -1e-12].mean()) if (d < -1e-12).any() else None,
                "rank1_won": int((d1 > 0).sum()), "rank1_lost": int((d1 < 0).sum()),
                "ap_lt_0.1_ref": int((ref < 0.1).sum()), "ap_lt_0.1_new": int((aps[key] < 0.1).sum())}
    print(key, res[key])
(OUT / "s05b_perquery.json").write_text(json.dumps(res, indent=2) + "\n")
