"""Шаг 2: discriminative whitening (Lw, Radenović et al. TPAMI 2018, §3.5 / cirtorch whiten.py) на
замороженных векторах OSNet-AIN. Учится ТОЛЬКО на train_fit: m = среднее всех train_fit-векторов,
C = mean_{(i,j)} (x_i-x_j)(x_i-x_j)^T по всем парам same-ID & РАЗНЫЕ камеры,
Cρ = (1-ρ)C + ρ·tr(C)/512·I, P = chol(Cρ)^-1, x -> P(x-m), затем L2. Полные 512 измерений (поворот на
собственные векторы полной ковариации ортогонален и косинус не меняет — опущен).
KR(6,3,0.3) как есть поверх. ρ∈{0.1,0.5} зафиксированы брифом. Выбор ρ: whitening на train_fit БЕЗ
идентичностей tune-протокола -> AP@10 на tune-протоколе; финал на val — whitening на всём train_fit."""
import json, time
import numpy as np
from lib42 import *

RHOS = (0.1, 0.5)


def learn_lw(X, vid, cam, rho):
    X = np.asarray(X, np.float64); m = X.mean(0)
    by = {}
    for i, v in enumerate(vid):
        by.setdefault(v, []).append(i)
    ii, jj = [], []
    for idx in by.values():
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if cam[idx[a]] != cam[idx[b]]:
                    ii.append(idx[a]); jj.append(idx[b])
    D = X[ii] - X[jj]
    C = D.T @ D / len(ii)
    Cr = (1 - rho) * C + rho * np.trace(C) / X.shape[1] * np.eye(X.shape[1])
    L = np.linalg.cholesky(Cr)
    P = np.linalg.inv(L)
    ev = np.linalg.eigvalsh(C)
    return {"m": m, "P": P, "n_pairs": len(ii), "n_ids_with_pairs": int(sum(1 for idx in by.values() if len(set(cam[k] for k in idx)) > 1)),
            "C_trace": float(np.trace(C)), "C_eig_min": float(ev[0]), "C_eig_max": float(ev[-1]),
            "C_cond": float(ev[-1] / max(ev[0], 1e-300)), "Cr_cond": float(np.linalg.cond(Cr))}


def apply_lw(X, lw):
    return l2n((np.asarray(X, np.float64) - lw["m"]) @ lw["P"].T)


def eval_config(q, g, qm, gm, tag, noexcl=False):
    cos = scores_from_embeddings(q, g, metric="cosine"); dist, _ = rerank(q, g, *KR)
    res = {"cos": run_eval(cos, qm, gm), "kr": run_eval(-dist, qm, gm)}
    out = {n: summary(r) for n, r in res.items()}
    if noexcl:
        out["kr_no_camera_exclusion"] = summary(run_eval(-dist, qm, gm, fake_cameras=True))
        out["camera_gap_kr"] = out["kr_no_camera_exclusion"]["mAP"] - out["kr"]["mAP"]
    for n in ("cos", "kr"):
        print(f"[{tag}] {n}: mAP={out[n]['mAP']:.4f} R1={out[n]['Rank-1']:.4f} AP@10={out[n]['mAP@10(R)']:.4f}", flush=True)
    return out, {n: per_query(r) for n, r in res.items()}


tf = read_meta(SPLIT / "train_fit.csv")
ids = (OUT / "train_fit_osnet.ids").read_text().split(); assert ids == [r["image_id"] for r in tf]
X = np.load(OUT / "train_fit_osnet.npy").astype(np.float64)
vid = np.array([r["vehicle_id"] for r in tf]); cam = np.array([r["camera_id"] for r in tf])
pos = {iid: i for i, iid in enumerate(ids)}

# ---- выбор ρ на tune-протоколе (whitening без tune-идентичностей)
tq_m, tg_m = read_meta(TUNE / "tune_query.csv"), read_meta(TUNE / "tune_gallery.csv")
tune_ids = {r["vehicle_id"] for r in tq_m} | {r["vehicle_id"] for r in tg_m}
keep = np.array([v not in tune_ids for v in vid])
tq = X[[pos[r["image_id"]] for r in tq_m]]; tg = X[[pos[r["image_id"]] for r in tg_m]]
tune = {"n_train_rows_without_tune_ids": int(keep.sum()), "n_tune_ids": len(tune_ids)}
tune["baseline"], _ = eval_config(tq, tg, tq_m, tg_m, "tune base")
for rho in RHOS:
    lw = learn_lw(X[keep], vid[keep], cam[keep], rho)
    tune[f"rho{rho}"], _ = eval_config(apply_lw(tq, lw), apply_lw(tg, lw), tq_m, tg_m, f"tune rho={rho}")
    tune[f"rho{rho}"]["lw_stats"] = {k: v for k, v in lw.items() if k not in ("m", "P")}
selected = max(RHOS, key=lambda r: tune[f"rho{r}"]["kr"]["mAP@10(R)"])
print("SELECTED rho on tune by AP@10 (KR):", selected, flush=True)

# ---- val: whitening на всём train_fit
q, g, qm, gm = load_val()
val = {}
val["baseline"], pq_base = eval_config(q, g, qm, gm, "val base", noexcl=True)
assert abs(val["baseline"]["kr"]["mAP"] - BASELINE_MAP) < 1e-9
known = pq_base["kr"]["known"]; vidq = np.array([r["vehicle_id"] for r in qm])[known]
boot, pqs = {}, {}
for rho in RHOS:
    t0 = time.perf_counter(); lw = learn_lw(X, vid, cam, rho); tl = time.perf_counter() - t0
    val[f"rho{rho}"], pq = eval_config(apply_lw(q, lw), apply_lw(g, lw), qm, gm, f"val rho={rho}", noexcl=True)
    val[f"rho{rho}"]["lw_stats"] = {k: v for k, v in lw.items() if k not in ("m", "P")}; val[f"rho{rho}"]["learn_seconds"] = round(tl, 2)
    val[f"rho{rho}"]["step0"] = step0_numbers(pq["kr"], f"lw rho={rho} + KR")
    pqs[rho] = pq
    boot[f"rho{rho}"] = {
        "kr_mAP_vs_KR": paired_bootstrap(pq["kr"]["ap"][known], pq_base["kr"]["ap"][known]),
        "kr_mAP_vs_KR_grouped_by_id": paired_bootstrap(pq["kr"]["ap"][known], pq_base["kr"]["ap"][known], groups=vidq),
        "kr_AP10_vs_KR": paired_bootstrap(pq["kr"]["ap10_R_f"][known], pq_base["kr"]["ap10_R_f"][known]),
        "cos_mAP_vs_cosine": paired_bootstrap(pq["cos"]["ap"][known], pq_base["cos"]["ap"][known])}
    np.save(OUT / f"lw_P_rho{rho}.npy", lw["P"].astype(np.float32)); np.save(OUT / f"lw_m_rho{rho}.npy", lw["m"].astype(np.float32))
out = {"rhos": RHOS, "tune": tune, "selected_rho_on_tune": selected, "val": val, "bootstrap_val": boot}
dump_json(out, OUT / "step2_whitening.json")
np.savez(OUT / "perquery_step2.npz", known=known, ap_kr_base=pq_base["kr"]["ap"],
         **{f"ap_kr_rho{r}": pqs[r]["kr"]["ap"] for r in RHOS}, **{f"ap_cos_rho{r}": pqs[r]["cos"]["ap"] for r in RHOS})
for rho in RHOS:
    b = boot[f"rho{rho}"]["kr_mAP_vs_KR"]
    print(f"VAL rho={rho}: Lw+KR mAP {val[f'rho{rho}']['kr']['mAP']:.4f} vs KR {BASELINE_MAP:.4f}: delta {b['delta']:+.4f} CI {b['ci95']} p={b['p_two_sided']}")
