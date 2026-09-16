#!/usr/bin/env python3
"""Сводные таблицы: осмысленность карты, фон против центра, зона пластины.

Все интервалы — бутстрэп по случаям (10 000 ресэмплов, seed 20260916),
парные сравнения считаются по разностям внутри случая.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

JOB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JOB / "scripts"))
import occl  # noqa: E402

RNG = np.random.default_rng(20260916)
B = 10000


def ci(x, f=np.mean):
    x = np.asarray(x, float)
    if x.size == 0:
        return (float("nan"),) * 3
    idx = RNG.integers(0, x.size, (B, x.size))
    bs = f(x[idx], axis=1)
    return float(f(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def sign_p(x):
    """Двусторонний знаковый тест: доля бутстрэпа с другим знаком среднего."""
    m, lo, hi = ci(x)
    return float(min(1.0, 2 * min((np.asarray(x) <= 0).mean(), (np.asarray(x) >= 0).mean())))


S = {s: json.load(open(JOB / f"out/stats_{s}.json"))["rows"] for s in ("cases", "ref")}
M = {s: np.load(JOB / f"out/maps_{s}.npz") for s in ("cases", "ref")}
PL = {s: json.load(open(JOB / f"out/plate_{s}.json"))["rows"] for s in ("cases", "ref")}

out = {}

# ---------- A. агрегаты по видам случаев -------------------------------------
kinds = {}
for st in ("cases", "ref"):
    for k, r in enumerate(S[st]):
        kinds.setdefault(r["kind"], []).append((st, k, r))

tabA = []
for kind, items in kinds.items():
    def col(key, side="q"):
        return [r[side][key] for _, _, r in items]
    row = {"kind": kind, "n": len(items),
           "cos_med": float(np.median([r["cos"] for _, _, r in items]))}
    for key in ("mean_drop", "max_drop", "top10_share",
                "border_mean", "center_mean", "border_minus_center",
                "ring1_minus_center", "neg_share"):
        row[key] = float(np.median(col(key)))
    row["corr_null_med"] = float(np.median([r["corr_null"] for _, _, r in items]))
    alt = [r["corr_alt"] for _, _, r in items if "corr_alt" in r]
    row["corr_alt_med"] = float(np.median(alt)) if alt else None
    row["cos_null_med"] = float(np.median([r["cos_null"] for _, _, r in items]))
    tabA.append(row)
out["by_kind"] = tabA

# ---------- B. фон против центра (обе стороны всех пар) ----------------------
bmc, r1mc = [], []
for st in ("cases", "ref"):
    for r in S[st]:
        for side in ("q", "g"):
            bmc.append(r[side]["border_minus_center"])
            r1mc.append(r[side]["ring1_minus_center"])
m, lo, hi = ci(bmc)
out["border_minus_center"] = {"n": len(bmc), "mean": m, "ci": [lo, hi],
                              "median": float(np.median(bmc)),
                              "share_positive": float(np.mean(np.array(bmc) > 0))}
m, lo, hi = ci(r1mc)
out["ring1_minus_center"] = {"n": len(r1mc), "mean": m, "ci": [lo, hi],
                             "median": float(np.median(r1mc)),
                             "share_positive": float(np.mean(np.array(r1mc) > 0))}

# ---------- C. зона пластины --------------------------------------------------
pl_rows = [r for st in ("cases", "ref") for r in PL[st] if r.get("found")]
CTL = ["shift_ring", "mirror_ring", "side_ring", "rand1_ring", "rand2_ring"]
d_plate = np.array([r["delta"]["plate_ring"] for r in pl_rows])
d_ctl = np.array([[r["delta"][c] for c in CTL] for r in pl_rows])
diff = d_plate - d_ctl.mean(axis=1)
m, lo, hi = ci(diff)
out["plate"] = {
    "n_crops": len(PL["cases"]) + len(PL["ref"]),
    "n_found": len(pl_rows),
    "box_area_frac_med": float(np.median([r["box_area_frac"] for r in pl_rows])),
    "delta_plate_mean": float(d_plate.mean()),
    "delta_ctl_mean": float(d_ctl.mean()),
    "paired_diff_mean": m, "paired_diff_ci": [lo, hi],
    "paired_diff_median": float(np.median(diff)),
    "share_plate_hotter": float(np.mean(diff > 0)),
    "p_sign": sign_p(diff),
    "per_control": {},
}
for j, c in enumerate(CTL):
    d = d_plate - d_ctl[:, j]
    m2, lo2, hi2 = ci(d)
    out["plate"]["per_control"][c] = {"mean_delta_control": float(d_ctl[:, j].mean()),
                                      "paired_diff": m2, "ci": [lo2, hi2]}
dp = np.array([r["delta"]["platepad_ring"] for r in pl_rows])
ds = np.array([r["delta"]["shiftpad_ring"] for r in pl_rows])
m2, lo2, hi2 = ci(dp - ds)
out["plate"]["padded"] = {"delta_plate": float(dp.mean()), "delta_shift": float(ds.mean()),
                          "paired_diff": m2, "ci": [lo2, hi2]}
# чтение скользящей карты в зоне пластины
mp = np.array([r["map_plate_mean"] for r in pl_rows])
ma = np.array([r["map_all_mean"] for r in pl_rows])
ms = np.array([r["map_shift_mean"] for r in pl_rows])
m3, lo3, hi3 = ci(mp - ma)
m4, lo4, hi4 = ci(mp - ms)
out["plate"]["map_reading"] = {
    "map_plate_mean": float(mp.mean()), "map_all_mean": float(ma.mean()),
    "map_shift_mean": float(ms.mean()),
    "plate_minus_all": m3, "plate_minus_all_ci": [lo3, hi3],
    "plate_minus_shift": m4, "plate_minus_shift_ci": [lo4, hi4],
    "share_plate_above_map": float(np.mean(mp > ma)),
    "n_windows_plate_med": float(np.median([r["map_n_windows_plate"] for r in pl_rows])),
}

# ---------- D. пара-специфичность и негативный контроль ----------------------
cn = np.array([r["corr_null"] for st in ("cases", "ref") for r in S[st]])
m, lo, hi = ci(cn)
out["corr_null"] = {"n": len(cn), "mean": m, "ci": [lo, hi],
                    "median": float(np.median(cn))}
ca = np.array([r["corr_alt"] for r in S["cases"] if "corr_alt" in r])
if ca.size:
    m, lo, hi = ci(ca)
    out["corr_alt"] = {"n": len(ca), "mean": m, "ci": [lo, hi],
                       "median": float(np.median(ca))}
# концентрация: карта пары против null-карты того же запроса
t_pair, t_null = [], []
for st in ("cases", "ref"):
    for r in S[st]:
        t_pair.append(r["q"]["top10_share"])
        t_null.append(r["null"]["top10_share"])
m, lo, hi = ci(np.array(t_pair) - np.array(t_null))
out["top10_pair_minus_null"] = {"pair": float(np.mean(t_pair)),
                                "null": float(np.mean(t_null)),
                                "diff": m, "ci": [lo, hi]}
mp_ = np.array([r["q"]["max_drop"] for st in ("cases", "ref") for r in S[st]])
mn_ = np.array([r["null"]["max_drop"] for st in ("cases", "ref") for r in S[st]])
m, lo, hi = ci(mp_ - mn_)
out["maxdrop_pair_minus_null"] = {"pair": float(mp_.mean()), "null": float(mn_.mean()),
                                  "diff": m, "ci": [lo, hi]}

# ---------- E. вертикальный профиль карты ------------------------------------
prof = {}
for st in ("cases", "ref"):
    for k, r in enumerate(S[st]):
        for side, key in (("q", f"{k}_wq"), ("g", f"{k}_wg")):
            w = M[st][key]
            p = w.mean(axis=1)
            prof.setdefault(r["kind"], []).append(p)
out["row_profile"] = {k: np.mean(v, axis=0).round(5).tolist() for k, v in prof.items()}
allp = np.concatenate([np.array(v) for v in prof.values()])
out["row_profile_all"] = allp.mean(axis=0).round(5).tolist()

(JOB / "out/aggregate.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in out.items() if k not in ("by_kind", "row_profile")},
                 ensure_ascii=False, indent=1))
print()
for r in tabA:
    print(f"{r['kind']:12s} n={r['n']:3d} cos={r['cos_med']:+.3f} "
          f"mean_drop={r['mean_drop']:+.4f} max={r['max_drop']:+.4f} "
          f"top10={r['top10_share']:.3f} b-c={r['border_minus_center']:+.4f} "
          f"corr_null={r['corr_null_med']:+.2f} corr_alt="
          f"{'-' if r['corr_alt_med'] is None else format(r['corr_alt_med'], '+.2f')}")
