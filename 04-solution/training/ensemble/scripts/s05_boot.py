#!/usr/bin/env python3
"""job_43, шаг 4: парный бутстрэп по per-query AP (и Rank-1) контура, 4000 ресэмплов,
seed 20260916 — тот же приём, что training/attempt-2/boot.py. Опора: osnet|rr (0,694)."""
import json, sys
from pathlib import Path
import numpy as np

OUT = Path.home() / "lct-reid/jobs/job_43/out"
B, SEED = 4000, 20260916
aps = np.load(OUT / "s04_aps.npz"); r1s = np.load(OUT / "s04_rank1.npz")

def paired(a, b):
    assert len(a) == len(b)
    n = len(a)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(B, n))
    da = a[idx].mean(1) - b[idx].mean(1)
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "delta": float(a.mean() - b.mean()),
            "ci95": [float(np.quantile(da, 0.025)), float(np.quantile(da, 0.975))],
            "p_two_sided": float(2 * min((da <= 0).mean(), (da >= 0).mean())),
            "sd_delta": float(da.std(ddof=1)), "n_queries": int(n), "B": B, "seed": SEED}

res = {"ref": "osnet|rr", "sigma_brief": 0.0134, "mAP": {}, "Rank-1": {}, "mAP_vs_osnet_cos": {}}
ref_ap, ref_r1 = aps["osnet|rr"], r1s["osnet|rr"]
for key in aps.files:
    if key == "osnet|rr":
        continue
    res["mAP"][key] = paired(aps[key], ref_ap)
    res["Rank-1"][key] = paired(r1s[key], ref_r1)
    if key.endswith("|cos") and key != "osnet|cos":
        res["mAP_vs_osnet_cos"][key] = paired(aps[key], aps["osnet|cos"])
(OUT / "s05_boot.json").write_text(json.dumps(res, indent=2) + "\n")
print(f"{'конфигурация':16s} {'ΔmAP':>8s} {'95% ДИ':>20s} {'p':>6s} | {'ΔR1':>8s} {'95% ДИ':>20s} {'p':>6s}")
for key in aps.files:
    if key == "osnet|rr":
        continue
    m, r = res["mAP"][key], res["Rank-1"][key]
    print(f"{key:16s} {m['delta']:+.4f} [{m['ci95'][0]:+.4f}; {m['ci95'][1]:+.4f}] {m['p_two_sided']:.3f} | "
          f"{r['delta']:+.4f} [{r['ci95'][0]:+.4f}; {r['ci95'][1]:+.4f}] {r['p_two_sided']:.3f}")
