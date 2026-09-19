#!/usr/bin/env python3
"""job_49: НЕЗАВИСИМЫЙ пересчёт конфигурации d1 (свой код, без скриптов job_45b).

Конвейер d1 (как заявлено):
  o = l2(osnet), a = l2(ainv2)
  e = l2(o + a)                      # ансамбль
  v = l2((e - m) @ P.T)              # whitening rho=0.5, обучен на train_fit
  метрики: cosine и k-reciprocal (k1=6, k2=3, lambda=0.3), контур 04-solution/eval

Всё считается своими реализациями (whitening-обучение, k-reciprocal),
метрики — только импортом контура reid_metrics.evaluate (это измеритель).
Сверка: s02_configs.json (mAP/Rank-1) и s02_perquery.npz (per-query AP/Rank-1).
"""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path
import numpy as np

HOME = Path.home() / "lct-reid"
REPO = HOME / "repo"
S = REPO / "04-solution"
EVAL = S / "eval"
SPLIT = S / "split/files"
A2 = S / "training/attempt-2/out"
J42 = HOME / "jobs/job_42/out"
J45 = HOME / "jobs/job_45/out"
OUT = HOME / "jobs/job_49/out"
sys.path.insert(0, str(EVAL))
from reid_metrics import evaluate, scores_from_embeddings  # noqa: E402  (только измеритель)

RHO = 0.5
K1, K2, LAM = 6, 3, 0.3


def l2n(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def read_meta(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


# ---------- whitening (моя реализация) ----------
def learn_whitening(X, ids, cams, rho):
    """Среднее m; ковариация C разностей пар «тот же id, другая камера»;
    shrinkage Cr = (1-rho) C + rho tr(C)/d I; P = inv(chol(Cr))."""
    X = np.asarray(X, dtype=np.float64)
    m = X.mean(axis=0)
    byid = {}
    for i, v in enumerate(ids):
        byid.setdefault(v, []).append(i)
    ii, jj = [], []
    for idxs in byid.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                if cams[idxs[a]] != cams[idxs[b]]:
                    ii.append(idxs[a]); jj.append(idxs[b])
    D = X[ii] - X[jj]
    C = D.T @ D / len(ii)
    Cr = (1 - rho) * C + rho * np.trace(C) / X.shape[1] * np.eye(X.shape[1])
    L = np.linalg.cholesky(Cr)
    P = np.linalg.inv(L)
    return {"m": m, "P": P, "n_pairs": len(ii)}


def apply_whitening(X, lw, center=True):
    X = np.asarray(X, dtype=np.float64)
    if center:
        X = X - lw["m"]
    return l2n(X @ lw["P"].T)


# ---------- k-reciprocal re-ranking (моя реализация, Zhong et al. 2017) ----------
def _krec(R, i, k):
    fwd = R[i, :k + 1]
    bwd = R[fwd, :k + 1]
    return fwd[np.where(bwd == i)[0]]


def kreciprocal(q, g, k1=K1, k2=K2, lam=LAM):
    """q, g — L2-нормированные векторы. Возвращает матрицу дистанций Q x G."""
    nq, ng = len(q), len(g)
    X = np.vstack([np.asarray(q, float), np.asarray(g, float)])
    X = l2n(X)
    D = 1.0 - np.clip(X @ X.T, -1.0, 1.0)
    Dn = (D / D.max(axis=0)).T            # нормировка на макс по столбцу, затем транспонирование
    R = np.argsort(Dn, axis=1)
    N = nq + ng
    V = np.zeros_like(Dn)
    for i in range(N):
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
    orig = Dn[:nq]
    return ((1 - lam) * J + lam * orig)[:, nq:]


# ---------- контур (их измеритель, импорт без изменений) ----------
def contour(scores, qm, gm):
    res = evaluate(
        scores,
        query_ids=[r["vehicle_id"] for r in qm],
        gallery_ids=[r["vehicle_id"] for r in gm],
        query_cameras=[r["camera_id"] for r in qm],
        gallery_cameras=[r["camera_id"] for r in gm],
        known_absent=np.array([r.get("has_mate", "1") == "0" for r in qm]),
        threshold=0.0, camera_policy="market", refusal_mode="presence",
    )
    full = res["ranking_full_gallery"]
    known = [row for row in res["per_query"] if row["status"] == "known"]
    ap = np.array([row["ap"] for row in known])
    r1 = np.array([float(row["rank1"]) for row in known])
    return {"mAP": full["mAP"], "Rank-1": full["Rank-1"], "Rank-5": full["Rank-5"],
            "mINP": full["mINP"], "valid": full["num_valid_queries"],
            "ap": ap, "r1": r1}


def main():
    t0 = time.perf_counter()
    qm, gm = read_meta(SPLIT / "val_query.csv"), read_meta(SPLIT / "val_gallery.csv")
    tf = read_meta(SPLIT / "train_fit.csv")

    # идентичность порядка строк — по .ids-файлам (общий .ids без тега = osnet)
    assert (A2 / "val_query.ids").read_text().split() == [r["image_id"] for r in qm]
    assert (A2 / "val_gallery.ids").read_text().split() == [r["image_id"] for r in gm]
    for tag in ("ainv2",):
        assert (A2 / f"val_query_{tag}.ids").read_text().split() == [r["image_id"] for r in qm]
        assert (A2 / f"val_gallery_{tag}.ids").read_text().split() == [r["image_id"] for r in gm]
    assert (J42 / "train_fit_osnet.ids").read_text().split() == [r["image_id"] for r in tf]
    assert (J45 / "train_fit_ainv2.ids").read_text().split() == [r["image_id"] for r in tf]

    vq = {t: l2n(np.load(A2 / f"val_query_{t}.npy")) for t in ("osnet", "ainv2")}
    vg = {t: l2n(np.load(A2 / f"val_gallery_{t}.npy")) for t in ("osnet", "ainv2")}
    Xt = {"osnet": l2n(np.load(J42 / "train_fit_osnet.npy")),
          "ainv2": l2n(np.load(J45 / "train_fit_ainv2.npy"))}
    vid = np.array([r["vehicle_id"] for r in tf])
    cam = np.array([r["camera_id"] for r in tf])

    # --- whitening: учим сами на train_fit (ансамбль), сверяем с сохранёнными матрицами ---
    Et = l2n(Xt["osnet"] + Xt["ainv2"])
    lw = learn_whitening(Et, vid, cam, RHO)
    z = np.load(J45 / "lw_ens_w1.0_rho0.5_f64.npz")
    dP = np.abs(lw["P"] - z["P"]).max()
    dm = np.abs(lw["m"] - z["m"]).max()
    print(f"[whitening] n_pairs={lw['n_pairs']} max|P_mine-P_stored|={dP:.3e} "
          f"max|m_mine-m_stored|={dm:.3e}")

    # --- d1 векторы ---
    eq, eg = l2n(vq["osnet"] + vq["ainv2"]), l2n(vg["osnet"] + vg["ainv2"])
    dq, dg = apply_whitening(eq, lw), apply_whitening(eg, lw)

    # --- cosine ---
    rc = contour(scores_from_embeddings(dq, dg, metric="cosine"), qm, gm)
    print(f"[d1 cos] mAP={rc['mAP']:.6f} R1={rc['Rank-1']:.6f} R5={rc['Rank-5']:.6f} "
          f"mINP={rc['mINP']:.6f} valid={rc['valid']}")

    # --- k-reciprocal (своя реализация) ---
    t1 = time.perf_counter()
    dist = kreciprocal(dq, dg)
    rk = contour(-dist, qm, gm)
    print(f"[d1 kr6,3,0.3] mAP={rk['mAP']:.6f} R1={rk['Rank-1']:.6f} R5={rk['Rank-5']:.6f} "
          f"mINP={rk['mINP']:.6f} valid={rk['valid']} ({time.perf_counter()-t1:.1f}s)")

    # --- сверка с их результатами ---
    s02 = json.load(open(J45 / "s02_configs.json"))
    ref = s02["table"]["d1_w1.0"]
    print(f"[ref d1 cos] mAP={ref['cos']['mAP']:.6f} R1={ref['cos']['Rank-1']:.6f}")
    print(f"[ref d1 kr ] mAP={ref['kr']['mAP']:.6f} R1={ref['kr']['Rank-1']:.6f}")
    pq = np.load(J45 / "s02_perquery.npz")
    ap_ref, r1_ref = pq["d1_w1.0|kr|ap"], pq["d1_w1.0|kr|r1"]
    print(f"[perquery kr] max|dAP|={np.abs(rk['ap']-ap_ref).max():.3e} "
          f"max|dR1|={np.abs(rk['r1']-r1_ref).max():.3e} n={len(ap_ref)}")
    apc_ref, r1c_ref = pq["d1_w1.0|cos|ap"], pq["d1_w1.0|cos|r1"]
    print(f"[perquery cos] max|dAP|={np.abs(rc['ap']-apc_ref).max():.3e} "
          f"max|dR1|={np.abs(rc['r1']-r1c_ref).max():.3e}")

    np.savez(OUT / "recalc_d1.npz", ap_kr=rk["ap"], r1_kr=rk["r1"], ap_cos=rc["ap"], r1_cos=rc["r1"])
    json.dump({"my_cos": {k: rc[k] for k in ("mAP", "Rank-1", "Rank-5", "mINP", "valid")},
               "my_kr": {k: rk[k] for k in ("mAP", "Rank-1", "Rank-5", "mINP", "valid")},
               "ref_cos": ref["cos"], "ref_kr": ref["kr"],
               "P_match_max_abs_diff": float(dP), "m_match_max_abs_diff": float(dm),
               "n_pairs": lw["n_pairs"], "seconds": time.perf_counter() - t0},
              open(OUT / "recalc_d1.json", "w"), indent=1)
    print(f"total {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
