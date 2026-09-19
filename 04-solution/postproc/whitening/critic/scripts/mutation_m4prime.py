#!/usr/bin/env python3
"""job_49: M4' — ансамбль с дисбалансом норм (хранимые векторы уже unit-norm,
поэтому «без L2-нормировки» в литеральном виде — no-op; см. mutations.py).
Здесь одна модель доминирует: e = l2(s*o + a), s = 2 и 10, по обе стороны."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recalc_d1 import (l2n, read_meta, learn_whitening, apply_whitening,
                       kreciprocal, contour, RHO, A2, J42, J45, SPLIT, OUT)


def main():
    qm, gm = read_meta(SPLIT / "val_query.csv"), read_meta(SPLIT / "val_gallery.csv")
    tf = read_meta(SPLIT / "train_fit.csv")
    vq = {t: l2n(np.load(A2 / f"val_query_{t}.npy")) for t in ("osnet", "ainv2")}
    vg = {t: l2n(np.load(A2 / f"val_gallery_{t}.npy")) for t in ("osnet", "ainv2")}
    Xt = {"osnet": l2n(np.load(J42 / "train_fit_osnet.npy")),
          "ainv2": l2n(np.load(J45 / "train_fit_ainv2.npy"))}
    vid_t = np.array([r["vehicle_id"] for r in tf]); cam_t = np.array([r["camera_id"] for r in tf])
    lw = learn_whitening(l2n(Xt["osnet"] + Xt["ainv2"]), vid_t, cam_t, RHO)

    res = {}
    for s in (2.0, 10.0):
        for side in ("osnet_x", "ainv2_x"):
            if side == "osnet_x":
                eq, eg = l2n(s * vq["osnet"] + vq["ainv2"]), l2n(s * vg["osnet"] + vg["ainv2"])
            else:
                eq, eg = l2n(vq["osnet"] + s * vq["ainv2"]), l2n(vg["osnet"] + s * vg["ainv2"])
            dq, dg = apply_whitening(eq, lw), apply_whitening(eg, lw)
            r = contour(-kreciprocal(dq, dg), qm, gm)
            tag = f"M4prime:{side}{s:g}"
            print(f"{tag:24s} mAP={r['mAP']:.4f} R1={r['Rank-1']:.4f}", flush=True)
            res[tag] = {"mAP": r["mAP"], "Rank-1": r["Rank-1"]}
    json.dump(res, open(OUT / "mutations_m4prime.json", "w"), indent=1)


if __name__ == "__main__":
    main()
