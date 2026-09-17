"""Шаг 1: diffusion вместо k-reciprocal (замена, не добавка).
Граф: взаимный kNN галереи (Iscen et al., CVPR 2017), веса cos^3 (γ=3 фиксирован заранее по статье),
S = D^-1/2 A D^-1/2. Запрос прикрепляется вектором y (cos^3 к kq=3 ближайшим), f0 = y,
f <- α S f + (1-α) y, ровно T=25 итераций (restart на исходный вектор запроса).
Ранжир: по f убыв., ничьи и нули — по исходному косинусу. Другие запросы в граф не входят.
Сетка (зафиксирована до результата): k∈{5,10,20}, α∈{0.8,0.95}; kq=3. Выбор варианта —
по AP@10 на tune-протоколе (train_fit-идентичности, как выбирались параметры KR); затем val vs KR."""
import json, sys, time
import numpy as np
from lib42 import *

GAMMA, KQ, T = 3, 3, 25
GRID = [(k, a) for k in (5, 10, 20) for a in (0.8, 0.95)]


def build_S(G, k):
    Gn = l2n(G); C = np.clip(Gn @ Gn.T, 0, 1); np.fill_diagonal(C, 0)
    nn = np.argsort(-C, axis=1, kind="stable")[:, :k]
    M = np.zeros_like(C, dtype=bool)
    rows = np.repeat(np.arange(len(G)), k); M[rows, nn.ravel()] = True
    mutual = M & M.T
    A = np.where(mutual, C ** GAMMA, 0.0)
    d = A.sum(1); dis = np.where(d > 0, 1 / np.sqrt(np.where(d > 0, d, 1)), 0.0)
    S = dis[:, None] * A * dis[None, :]
    return S, Gn, {"edges": int(mutual.sum() // 2), "isolated": int((d == 0).sum()),
                   "mean_degree": float(mutual.sum(1).mean())}


def diffuse(Q, G, k, alpha, kq=KQ, T=T):
    S, Gn, info = build_S(G, k)
    Qn = l2n(Q); cos = np.clip(Qn @ Gn.T, -1, 1)
    top = np.argsort(-cos, axis=1, kind="stable")[:, :kq]
    Y = np.zeros_like(cos); r = np.arange(len(Q))[:, None]
    Y[r, top] = np.clip(cos[r, top], 0, 1) ** GAMMA
    F = Y.copy()
    for t in range(T):
        Fn = alpha * (F @ S.T) + (1 - alpha) * Y  # S симметрична; F @ S = (S F^T)^T
        delta = np.abs(Fn - F).max(); F = Fn
    # лексикографический ранжир: f убыв., затем cos убыв. -> score = -позиция (точно, без ε)
    scores = np.empty_like(F)
    for i in range(len(Q)):
        order = np.lexsort((-cos[i], -F[i]))
        scores[i, order] = -np.arange(len(order), dtype=np.float64)
    info.update(last_delta=float(delta), reached_mean=float((F > 0).sum(1).mean()))
    return scores, info


def evaluate_set(q, g, qm, gm, tag, with_noexcl=False):
    cos = scores_from_embeddings(q, g, metric="cosine")
    dist_kr, _ = rerank(q, g, *KR)
    runs = {"cosine": cos, "kr": -dist_kr}
    infos = {}
    for k, a in GRID:
        t0 = time.perf_counter(); sc, info = diffuse(q, g, k, a); info["seconds"] = round(time.perf_counter() - t0, 2)
        runs[f"diff_k{k}_a{a}"] = sc; infos[f"diff_k{k}_a{a}"] = info
    res = {name: run_eval(sc, qm, gm) for name, sc in runs.items()}
    out = {"summary": {n: summary(r) for n, r in res.items()}, "graph": infos}
    pq = {n: per_query(r) for n, r in res.items()}
    if with_noexcl:
        out["summary_no_camera_exclusion"] = {n: summary(run_eval(sc, qm, gm, fake_cameras=True)) for n, sc in runs.items()}
        out["camera_gap"] = {n: out["summary_no_camera_exclusion"][n]["mAP"] - out["summary"][n]["mAP"] for n in runs}
    for n in out["summary"]:
        print(f"[{tag}] {n:16s} mAP={out['summary'][n]['mAP']:.4f} R1={out['summary'][n]['Rank-1']:.4f} "
              f"AP@10={out['summary'][n]['mAP@10(R)']:.4f} mAP1110={out['summary'][n]['mAP_1110']:.4f}", flush=True)
    return out, pq, runs


# ---- tune-протокол (train_fit) — только для ВЫБОРА варианта
tq_m, tg_m = read_meta(TUNE / "tune_query.csv"), read_meta(TUNE / "tune_gallery.csv")
tf_ids = (OUT / "train_fit_osnet.ids").read_text().split()
tf_emb = np.load(OUT / "train_fit_osnet.npy"); pos = {iid: i for i, iid in enumerate(tf_ids)}
tq = tf_emb[[pos[r["image_id"]] for r in tq_m]]; tg = tf_emb[[pos[r["image_id"]] for r in tg_m]]
tune_out, tune_pq, _ = evaluate_set(tq, tg, tq_m, tg_m, "tune")
cands = {n: tune_out["summary"][n]["mAP@10(R)"] for n in tune_out["summary"] if n.startswith("diff_")}
selected = max(cands, key=cands.get)
print("SELECTED on tune by AP@10:", selected, cands[selected], "| KR on tune:", tune_out["summary"]["kr"]["mAP@10(R)"], flush=True)

# ---- val
q, g, qm, gm = load_val()
val_out, val_pq, _ = evaluate_set(q, g, qm, gm, "val", with_noexcl=True)
known = val_pq["kr"]["known"]; vid = np.array([r["vehicle_id"] for r in qm])[known]
boot = {}
for n in val_pq:
    if n.startswith("diff_"):
        boot[n] = {"mAP_vs_KR": paired_bootstrap(val_pq[n]["ap"][known], val_pq["kr"]["ap"][known]),
                   "mAP_vs_KR_grouped_by_id": paired_bootstrap(val_pq[n]["ap"][known], val_pq["kr"]["ap"][known], groups=vid),
                   "AP10_vs_KR": paired_bootstrap(val_pq[n]["ap10_R_f"][known], val_pq["kr"]["ap10_R_f"][known]),
                   "mAP_vs_cosine": paired_bootstrap(val_pq[n]["ap"][known], val_pq["cosine"]["ap"][known])}
step0_sel = step0_numbers(val_pq[selected], selected)
out = {"grid": GRID, "gamma": GAMMA, "kq": KQ, "T": T, "tune": tune_out, "selected_on_tune": selected,
       "tune_candidates_AP10": cands, "val": val_out, "bootstrap_val": boot,
       "selected_val_step0": step0_sel}
dump_json(out, OUT / "step1_diffusion.json")
np.savez(OUT / "perquery_step1.npz", known=known, **{f"ap_{n}": val_pq[n]["ap"] for n in val_pq},
         **{f"first_{n}": val_pq[n]["first"] for n in val_pq})
b = boot[selected]["mAP_vs_KR"]
print(f"VAL selected {selected}: mAP {val_out['summary'][selected]['mAP']:.4f} vs KR {val_out['summary']['kr']['mAP']:.4f}; "
      f"delta {b['delta']:+.4f} CI {b['ci95']} p={b['p_two_sided']}")
