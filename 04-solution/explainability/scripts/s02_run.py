#!/usr/bin/env python3
"""Прогон карт влияния по отобранным случаям + опорная группа обычных совпадений.

Для каждого случая считаются:
  wq  — карта запроса против кандидата,
  wg  — карта кандидата против запроса,
  wq_alt — карта запроса против альтернативного кандидата (если задан):
           для двойников это верная пара, для ошибок из выборки — верная пара.
           Нужна для проверки пара-специфичности: если карта та же, объяснение
           не про пару, а про картинку.
  wq_null — карта запроса против СЛУЧАЙНОГО несвязанного кандидата (seed фикс.).
           Негативный контроль метода.

Выход: out/maps_<set>.npz (сетки 11x11) и out/stats_<set>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

JOB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JOB / "scripts"))
import occl  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--set", default="cases", choices=["cases", "ref"])
ap.add_argument("--n-ref", type=int, default=30)
ap.add_argument("--seed", type=int, default=20260916)
ap.add_argument("--mode", default="ring")
args = ap.parse_args()

qm, gm = occl.read_split("val_query"), occl.read_split("val_gallery")
gv = np.array([int(r["vehicle_id"]) for r in gm])
rng = np.random.default_rng(args.seed)

if args.set == "cases":
    items = json.load(open(JOB / "out/cases.json"))["cases"]
else:
    ref = json.load(open(JOB / "out/ref_cases.json"))["cases"]
    items = ref[: args.n_ref]

maps, rows = {}, []
t_all = time.perf_counter()
for k, c in enumerate(items):
    qi, gi = c["qi"], c["gi"]
    t0 = time.perf_counter()
    qim, gim = occl.net_input(qm[qi]), occl.net_input(gm[gi])
    r = occl.pair_maps(qim, gim, mode=args.mode)
    rec = {**c, "cos": r["cos"],
           "q": occl.stats(r["wq"]), "g": occl.stats(r["wg"])}
    maps[f"{k}_wq"], maps[f"{k}_wg"] = r["wq"], r["wg"]

    if c.get("gi_alt") is not None:
        aim = occl.net_input(gm[c["gi_alt"]])
        ra = occl.pair_maps(qim, aim, mode=args.mode, sides=("q",))
        maps[f"{k}_wq_alt"] = ra["wq"]
        rec["cos_alt"] = ra["cos"]
        rec["corr_alt"] = float(np.corrcoef(r["wq"].ravel(), ra["wq"].ravel())[0, 1])
        rec["alt"] = occl.stats(ra["wq"])

    # негативный контроль: случайный кандидат другой машины
    pool = np.flatnonzero(gv != int(qm[qi]["vehicle_id"]))
    ni = int(pool[rng.integers(len(pool))])
    rn = occl.pair_maps(qim, occl.net_input(gm[ni]), mode=args.mode, sides=("q",))
    maps[f"{k}_wq_null"] = rn["wq"]
    rec["null_gi"] = ni
    rec["cos_null"] = rn["cos"]
    rec["corr_null"] = float(np.corrcoef(r["wq"].ravel(), rn["wq"].ravel())[0, 1])
    rec["null"] = occl.stats(rn["wq"])
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    rows.append(rec)
    print(f"[{k+1}/{len(items)}] {c['kind']:12s} q{qi} g{gi} cos={r['cos']:.3f} "
          f"{rec['seconds']:.1f}s", flush=True)

np.savez_compressed(JOB / f"out/maps_{args.set}.npz", **maps)
(JOB / f"out/stats_{args.set}.json").write_text(
    json.dumps({"mode": args.mode, "win": occl.WIN, "stride": occl.STRIDE,
                "grid": occl.NG, "seed": args.seed,
                "total_seconds": round(time.perf_counter() - t_all, 1),
                "rows": rows}, ensure_ascii=False, indent=1))
print("всего", round(time.perf_counter() - t_all, 1), "с")
