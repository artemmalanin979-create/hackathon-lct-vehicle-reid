#!/usr/bin/env python3
"""Парный бутстрэп по значениям AP, которые вернул контур reid_metrics.py.
Ничего не пересчитывает: берёт per-query AP из aps_*.npz, сохранённых eval_table.py."""
import sys, json
import numpy as np

def load(path, key):
    z = np.load(path)
    return z[key]

def main():
    pairs = json.loads(sys.argv[1])
    B = 4000
    out = {}
    for name, (fa, ka, fb, kb) in pairs.items():
        a, b = load(fa, ka), load(fb, kb)
        assert len(a) == len(b), (len(a), len(b))
        n = len(a)
        rng = np.random.default_rng(20260916)
        idx = rng.integers(0, n, size=(B, n))
        da = a[idx].mean(1) - b[idx].mean(1)
        out[name] = {"mAP_a": float(a.mean()), "mAP_b": float(b.mean()),
                     "delta": float(a.mean() - b.mean()),
                     "ci95": [float(np.quantile(da, 0.025)), float(np.quantile(da, 0.975))],
                     "p_two_sided": float(2 * min((da <= 0).mean(), (da >= 0).mean())),
                     "sd_delta": float(da.std(ddof=1)), "n_queries": n}
    print(json.dumps(out, indent=2, ensure_ascii=False))

main()
