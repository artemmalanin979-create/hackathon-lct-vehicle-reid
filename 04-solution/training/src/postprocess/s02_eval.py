#!/usr/bin/env python3
"""job_48, шаг 2: оценка новой модели и ансамблей контуром 04-solution/eval.

Метрики — ТОЛЬКО контуром reid_metrics.py (market, presence), KR (6,3,0,3) канонический,
whitening learn_lw/apply_lw дословно job_42/job_45, парный бутстрэп lib42/lib45
(B=4000, seed 20260916). Эталоны: d1_w1.0=0.7621 (champion), osnet=0.6937 — их per-query
AP берутся из job_45/out/s02_perquery.npz (тот же контур, тот же протокол).

Конфигурации (векторы val, whitening учится на train_fit — osnet/ainv2/j48):
  j48        — одиночная новая модель
  c_j48      — l2(OSNet + j48)
  d1_j48     — Lw_ens(osnet+j48) -> l2(OSNet + j48)     (ансамбль с OSNet, whitening)
  d3_j48     — Lw_cat -> [OSNet, j48]                   (конкат, whitening)
  c3         — l2(OSNet + ain_v2 + j48)                 (ансамбль с d1-составом)
  d1_3       — Lw_ens3 -> l2(OSNet + ain_v2 + j48)      (ансамбль с d1, whitening)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HOME = Path.home() / "lct-reid"
REPO = HOME / "repo"
JOB = HOME / "jobs" / "job_48"
OUT = JOB / "out"
A2 = REPO / "04-solution" / "training" / "attempt-2" / "out"
J45 = HOME / "jobs" / "job_45"
SPLIT = REPO / "04-solution" / "split" / "files"

sys.path.insert(0, str(REPO / "04-solution" / "eval"))
sys.path.insert(0, str(REPO / "04-solution" / "postproc" / "scripts"))
sys.path.insert(0, str(J45 / "scripts"))
import lib45  # noqa: E402
from lib45 import (l2n, load_val, load_train, run_eval, eval_config,  # noqa: E402
                   learn_lw, apply_lw, lw_stats, paired_bootstrap, dump_json,
                   read_meta, scores_from_embeddings)

KR = (6, 3, 0.3)


def fresh_lw(name, Xt, vid, cam):
    f = OUT / f"lw_{name}_rho0.5_f64.npz"
    if f.exists():
        z = np.load(f)
        return {"m": z["m"], "P": z["P"]}
    lw = learn_lw(np.asarray(Xt, np.float64), vid, cam, 0.5)
    np.savez(f, P=lw["P"], m=lw["m"])
    dump_json(lw_stats(lw) | {"name": name, "dim": int(Xt.shape[1])}, OUT / f"lw_{name}_stats.json")
    return {"m": lw["m"], "P": lw["P"]}


def main():
    vec, qm, gm = load_val()                      # osnet, ainv2 val vectors
    q48 = l2n(np.load(OUT / "val_query_j48.npy"))
    g48 = l2n(np.load(OUT / "val_gallery_j48.npy"))
    assert (OUT / "val_query_j48.ids").read_text().split() == [r["image_id"] for r in qm]
    assert (OUT / "val_gallery_j48.ids").read_text().split() == [r["image_id"] for r in gm]

    X, vid, cam, ids = load_train()               # train_fit: osnet + ainv2 vectors
    t48 = l2n(np.load(OUT / "train_fit_j48.npy"))
    assert (OUT / "train_fit_j48.ids").read_text().split() == ids

    pq_ref = np.load(J45 / "out" / "s02_perquery.npz")
    ap_d1_ref, ap_osnet_ref = pq_ref["d1_w1.0|kr|ap"], pq_ref["a|kr|ap"]
    assert abs(float(ap_d1_ref.mean()) - 0.7621) < 5e-4, float(ap_d1_ref.mean())

    results, perquery = {}, {}

    def one(tag, q, g, refs):
        out, pq = eval_config(q, g, qm, gm, tag)
        results[tag] = out
        perquery[f"{tag}|kr|ap"] = pq["kr"][0]
        perquery[f"{tag}|kr|r1"] = pq["kr"][1]
        if "cos" in pq:
            perquery[f"{tag}|cos|ap"] = pq["cos"][0]
        for rtag, ref in refs.items():
            results[tag][f"boot_vs_{rtag}"] = paired_bootstrap(pq["kr"][0], ref)
        return out

    o_v = (vec["osnet"][0], vec["osnet"][1])
    a_v = (vec["ainv2"][0], vec["ainv2"][1])

    # --- эталон d1_w1.0 тут же, для сверки 0.7621 и per-query контроля ---
    Xens = l2n(X["osnet"] + X["ainv2"])
    lw_ens45 = lib45.get_lw("ens_w1.0", X, vid, cam)[0]
    d1_v = (apply_lw(l2n(o_v[0] + a_v[0]), lw_ens45), apply_lw(l2n(o_v[1] + a_v[1]), lw_ens45))
    out_d1, pq_d1 = eval_config(*d1_v, qm, gm, "d1_w1.0(repro)")
    results["d1_w1.0(repro)"] = out_d1
    perquery["d1_w1.0(repro)|kr|ap"] = pq_d1["kr"][0]
    assert abs(out_d1["kr"]["mAP"] - 0.7621) < 5e-4, out_d1["kr"]["mAP"]

    # --- эталон osnet (repro 0.6937) ---
    out_os, pq_os = eval_config(*o_v, qm, gm, "osnet(repro)")
    results["osnet(repro)"] = out_os
    perquery["osnet(repro)|kr|ap"] = pq_os["kr"][0]
    assert abs(out_os["kr"]["mAP"] - 0.6937) < 5e-4, out_os["kr"]["mAP"]

    # --- 1. одиночная j48 ---
    one("j48", q48, g48, {"d1": ap_d1_ref, "osnet": ap_osnet_ref})

    # --- 2. ансамбль с OSNet: среднее ---
    q_c, g_c = l2n(o_v[0] + q48), l2n(o_v[1] + g48)
    one("c_j48", q_c, g_c, {"d1": ap_d1_ref})

    # --- 3. d1-style: whitening поверх среднего с OSNet ---
    lw_j48 = fresh_lw("ens_j48", l2n(X["osnet"] + t48), vid, cam)
    one("d1_j48", apply_lw(q_c, lw_j48), apply_lw(g_c, lw_j48), {"d1": ap_d1_ref})

    # --- 4. конкат с OSNet + whitening ---
    lw_cat = fresh_lw("cat_j48", np.concatenate([X["osnet"], t48], 1), vid, cam)
    one("d3_j48", apply_lw(np.concatenate([o_v[0], q48], 1), lw_cat),
        apply_lw(np.concatenate([o_v[1], g48], 1), lw_cat), {"d1": ap_d1_ref})

    # --- 5. тройной средний (OSNet + ain_v2 + j48) ---
    q_3, g_3 = l2n(o_v[0] + a_v[0] + q48), l2n(o_v[1] + a_v[1] + g48)
    one("c3", q_3, g_3, {"d1": ap_d1_ref})

    # --- 6. тройной + whitening ---
    lw_3 = fresh_lw("ens3", l2n(X["osnet"] + X["ainv2"] + t48), vid, cam)
    one("d1_3", apply_lw(q_3, lw_3), apply_lw(g_3, lw_3), {"d1": ap_d1_ref})

    np.savez(OUT / "s02_perquery_j48.npz", **perquery)
    dump_json(results, OUT / "s02_metrics.json")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
