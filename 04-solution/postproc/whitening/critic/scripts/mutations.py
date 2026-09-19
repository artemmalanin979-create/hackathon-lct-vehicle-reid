#!/usr/bin/env python3
"""job_49: мутации конвейера d1. Свой код (recalc_d1), метрики — контуром eval.

Мутации:
 M1 whitening ПОСЛЕ KR: окрестности KR из ансамблевого пространства,
    дистанционный член финальной смеси — из отбеленного пространства
    (whitening входит в пайплайн после шага KR).
 M2 whitening без центрирования: v = l2(e @ P.T)
 M3 KR(1,1,0) вместо (6,3,0.3)
 M4 ансамбль без L2-нормировки одной из моделей (raw OSNet / raw ain_v2)
 M5 случайная матрица P той же формы (seed 49) вместо обученной
 M6 whitening, обученный на val (утечка) вместо train_fit
 M7 без исключения same-camera (все камеры разные)
 M8 перестановка усреднения и whitening (= per-model whitening, потом среднее; аналог d2)
Референсы из их таблицы: a (OSNet), c_w1.0 (ансамбль без whitening), d2_w1.0.
Плюс: перепроверка бутстрэпа Δ против эталона a по stored per-query AP.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recalc_d1 import (l2n, read_meta, learn_whitening, apply_whitening,
                       kreciprocal, contour, RHO,
                       HOME, REPO, S, EVAL, SPLIT, A2, J42, J45, OUT)


def krec_parts(q, g, k1, k2):
    """KR с возвратом (J, Dn) — для мутации M1."""
    nq, ng = len(q), len(g)
    X = l2n(np.vstack([np.asarray(q, float), np.asarray(g, float)]))
    D = 1.0 - np.clip(X @ X.T, -1.0, 1.0)
    Dn = (D / D.max(axis=0)).T
    R = np.argsort(Dn, axis=1)
    N = nq + ng
    V = np.zeros_like(Dn)
    for i in range(N):
        from recalc_d1 import _krec
        kR = _krec(R, i, k1)
        exp = [kR]
        half = int(round(k1 / 2.0))
        for c in kR:
            cR = _krec(R, c, half)
            if np.intersect1d(cR, kR).size > 2.0 / 3 * cR.size:
                exp.append(cR)
        exp = np.unique(np.concatenate(exp))
        w = np.exp(-Dn[i, exp])
        V[i, exp] = w / w.sum()
    if k2 != 1:
        V = np.vstack([V[R[i, :k2]].mean(axis=0) for i in range(N)])
    inv = [np.flatnonzero(V[:, j]) for j in range(N)]
    J = np.zeros((nq, N))
    for i in range(nq):
        acc = np.zeros(N)
        for j in np.flatnonzero(V[i]):
            imgs = inv[j]
            acc[imgs] += np.minimum(V[i, j], V[imgs, j])
        J[i] = 1.0 - acc / (2.0 - acc)
    return J, Dn


def summarize(tag, res):
    print(f"  {tag:34s} mAP={res['mAP']:.4f} R1={res['Rank-1']:.4f} "
          f"R5={res['Rank-5']:.4f} valid={res['valid']}", flush=True)
    return {"mAP": res["mAP"], "Rank-1": res["Rank-1"], "Rank-5": res["Rank-5"],
            "mINP": res["mINP"], "valid": res["valid"]}


def main():
    t0 = time.perf_counter()
    qm, gm = read_meta(SPLIT / "val_query.csv"), read_meta(SPLIT / "val_gallery.csv")
    tf = read_meta(SPLIT / "train_fit.csv")
    vq = {t: l2n(np.load(A2 / f"val_query_{t}.npy")) for t in ("osnet", "ainv2")}
    vg = {t: l2n(np.load(A2 / f"val_gallery_{t}.npy")) for t in ("osnet", "ainv2")}
    vq_raw = {t: np.load(A2 / f"val_query_{t}.npy").astype(np.float64) for t in ("osnet", "ainv2")}
    vg_raw = {t: np.load(A2 / f"val_gallery_{t}.npy").astype(np.float64) for t in ("osnet", "ainv2")}
    Xt = {"osnet": l2n(np.load(J42 / "train_fit_osnet.npy")),
          "ainv2": l2n(np.load(J45 / "train_fit_ainv2.npy"))}
    vid_t = np.array([r["vehicle_id"] for r in tf]); cam_t = np.array([r["camera_id"] for r in tf])
    vid_v = np.array([r["vehicle_id"] for r in qm] + [r["vehicle_id"] for r in gm])
    cam_v = np.array([r["camera_id"] for r in qm] + [r["camera_id"] for r in gm])

    res = {}

    # базовые блоки
    Et = l2n(Xt["osnet"] + Xt["ainv2"])
    lw = learn_whitening(Et, vid_t, cam_t, RHO)
    eq, eg = l2n(vq["osnet"] + vq["ainv2"]), l2n(vg["osnet"] + vg["ainv2"])
    dq, dg = apply_whitening(eq, lw), apply_whitening(eg, lw)

    # референсы: a (OSNet), c_w1.0 (ансамбль без whitening)
    print("[референсы]")
    res["a_osnet_kr"] = summarize("a: OSNet + KR", contour(-kreciprocal(vq["osnet"], vg["osnet"]), qm, gm))
    res["c_ens_nolw_kr"] = summarize("c: ансамбль, без whitening",
                                     contour(-kreciprocal(eq, eg), qm, gm))
    # канон d1
    res["d1_canon_kr"] = summarize("d1 канон ( whitening -> KR )", contour(-kreciprocal(dq, dg), qm, gm))

    # M1: whitening ПОСЛЕ KR — окрестности J из ансамбля, дист. член из whitening
    print("[M1 whitening после KR]")
    J_e, Dn_e = krec_parts(eq, eg, 6, 3)
    Xw = l2n(np.vstack([dq, dg])); Dw = 1.0 - np.clip(Xw @ Xw.T, -1.0, 1.0)
    Dn_w = (Dw / Dw.max(axis=0)).T
    nq = len(dq)
    mix = (1 - 0.3) * J_e + 0.3 * Dn_w[:nq]
    res["M1_wh_after_kr"] = summarize("M1: J(ансамбль)+D(whitened)",
                                      contour(-mix[:, nq:], qm, gm))

    # M2: whitening без центрирования
    print("[M2 без центрирования]")
    dq2, dg2 = apply_whitening(eq, lw, center=False), apply_whitening(eg, lw, center=False)
    res["M2_no_center"] = summarize("M2: l2(e @ P.T)", contour(-kreciprocal(dq2, dg2), qm, gm))

    # M3: KR(1,1,0)
    print("[M3 KR(1,1,0)]")
    res["M3_kr110"] = summarize("M3: KR k1=1,k2=1,lam=0",
                                contour(-kreciprocal(dq, dg, k1=1, k2=1, lam=0.0), qm, gm))

    # M4: ансамбль без L2-нормировки одной из моделей
    print("[M4 ансамбль без l2 одной модели]")
    eq4a, eg4a = l2n(vq_raw["osnet"] + vq["ainv2"]), l2n(vg_raw["osnet"] + vg["ainv2"])
    dq4a, dg4a = apply_whitening(eq4a, lw), apply_whitening(eg4a, lw)
    res["M4_osnet_raw"] = summarize("M4: l2(osnet_raw + a)",
                                    contour(-kreciprocal(dq4a, dg4a), qm, gm))
    eq4b, eg4b = l2n(vq["osnet"] + vq_raw["ainv2"]), l2n(vg["osnet"] + vg_raw["ainv2"])
    dq4b, dg4b = apply_whitening(eq4b, lw), apply_whitening(eg4b, lw)
    res["M4_ainv2_raw"] = summarize("M4: l2(o + ainv2_raw)",
                                    contour(-kreciprocal(dq4b, dg4b), qm, gm))

    # M5: случайная P той же формы (с обученным m)
    print("[M5 случайная P]")
    rng = np.random.default_rng(49)
    lw_rand = {"m": lw["m"], "P": rng.standard_normal(lw["P"].shape)}
    dq5, dg5 = apply_whitening(eq, lw_rand), apply_whitening(eg, lw_rand)
    res["M5_random_P"] = summarize("M5: P ~ N(0,1) случайная",
                                   contour(-kreciprocal(dq5, dg5), qm, gm))

    # M6: whitening, обученный на val (утечка)
    print("[M6 whitening на val — утечка]")
    Ev = l2n(np.vstack([eq, eg]))
    lw_val = learn_whitening(Ev, vid_v, cam_v, RHO)
    dq6, dg6 = apply_whitening(eq, lw_val), apply_whitening(eg, lw_val)
    res["M6_lw_on_val"] = summarize("M6: whitening обучен на val",
                                    contour(-kreciprocal(dq6, dg6), qm, gm))

    # M7: без исключения same-camera (все камеры уникальны)
    print("[M7 без исключения same-camera]")
    r7 = contour(-kreciprocal(dq, dg), qm, gm)
    # пересчёт с фейковыми камерами: оборачиваем contour
    from reid_metrics import evaluate
    dist = kreciprocal(dq, dg)
    res7 = evaluate(-dist, query_ids=[r["vehicle_id"] for r in qm],
                    gallery_ids=[r["vehicle_id"] for r in gm],
                    query_cameras=[f"q{i}" for i in range(len(qm))],
                    gallery_cameras=[f"g{j}" for j in range(len(gm))],
                    known_absent=np.array([r.get("has_mate", "1") == "0" for r in qm]),
                    threshold=0.0, camera_policy="market", refusal_mode="presence")
    f7 = res7["ranking_full_gallery"]
    res["M7_no_cam_excl"] = summarize("M7: без искл. same-camera",
                                      {"mAP": f7["mAP"], "Rank-1": f7["Rank-1"],
                                       "Rank-5": f7["Rank-5"], "mINP": f7["mINP"],
                                       "valid": f7["num_valid_queries"]})

    # M8: перестановка усреднения и whitening (= per-model whitening, потом среднее)
    print("[M8 whitening до усреднения (per-model)]")
    lw_o = learn_whitening(Xt["osnet"], vid_t, cam_t, RHO)
    lw_a = learn_whitening(Xt["ainv2"], vid_t, cam_t, RHO)
    zo = np.load(J42 / "lw_P_rho0.5.npy").astype(np.float64)
    mo = np.load(J42 / "lw_m_rho0.5.npy").astype(np.float64)
    print(f"  сверка lw_osnet: max|ΔP|={np.abs(lw_o['P']-zo).max():.2e} "
          f"max|Δm|={np.abs(lw_o['m']-mo).max():.2e}")
    za = np.load(J45 / "lw_ainv2_rho0.5_f64.npz")
    print(f"  сверка lw_ainv2: max|ΔP|={np.abs(lw_a['P']-za['P']).max():.2e} "
          f"max|Δm|={np.abs(lw_a['m']-za['m']).max():.2e}")
    dq8 = l2n(apply_whitening(vq["osnet"], lw_o) + apply_whitening(vq["ainv2"], lw_a))
    dg8 = l2n(apply_whitening(vg["osnet"], lw_o) + apply_whitening(vg["ainv2"], lw_a))
    res["M8_lw_then_avg"] = summarize("M8: l2(w(o)+w(a)) [=d2]",
                                      contour(-kreciprocal(dq8, dg8), qm, gm))

    # бутстрэп d1 против эталона a по их per-query AP (проверка Δ+0,068 [+0,052;+0,084])
    pq = np.load(J45 / "s02_perquery.npz")
    a_ap, d1_ap = pq["a|kr|ap"], pq["d1_w1.0|kr|ap"]
    B, seed = 4000, 20260916
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a_ap), size=(B, len(a_ap)))
    dd = d1_ap[idx].mean(1) - a_ap[idx].mean(1)
    boot = {"delta": float(d1_ap.mean() - a_ap.mean()),
            "ci95": [float(np.quantile(dd, 0.025)), float(np.quantile(dd, 0.975))],
            "p_two_sided": float(2 * min((dd <= 0).mean(), (dd >= 0).mean()))}
    print(f"[бутстрэп d1-a] Δ={boot['delta']:+.4f} CI={boot['ci95']} p={boot['p_two_sided']}")
    res["bootstrap_d1_minus_a"] = boot

    json.dump(res, open(OUT / "mutations.json", "w"), indent=1)
    print(f"total {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
