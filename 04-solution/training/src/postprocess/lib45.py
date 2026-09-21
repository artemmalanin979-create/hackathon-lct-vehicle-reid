"""job_45: общая обвязка. Метрики — ТОЛЬКО контуром 04-solution/eval/reid_metrics.py (импорт без
изменений); KR — 04-solution/postproc/scripts/common.rerank (канон, тот же, что дал 0,694);
whitening — learn_lw/apply_lw дословно из job_42/s2_whitening.py; бутстрэп — из job_42/lib42.py."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

HOME = Path.home() / "lct-reid"; REPO = HOME / "repo"; S = REPO / "04-solution"
EVAL = S / "eval"; SPLIT = S / "split/files"; A2 = S / "training/attempt-2/out"; TUNE = S / "postproc/tune"
J42 = HOME / "jobs/job_42/out"; J43 = HOME / "jobs/job_43/out"
JOB = HOME / "jobs/job_45"; OUT = JOB / "out"
sys.path.insert(0, str(EVAL)); sys.path.insert(0, str(S / "postproc/scripts"))
from reid_metrics import evaluate, scores_from_embeddings  # noqa: E402
from common import rerank  # noqa: E402

KR = (6, 3, 0.3)
BASELINE_MAP = 0.6936584724586873
RHO = 0.5
B_BOOT, SEED_BOOT = 4000, 20260916


def read_meta(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def l2n(m):
    m = np.asarray(m, dtype=np.float64)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def load_val():
    qm, gm = read_meta(SPLIT / "val_query.csv"), read_meta(SPLIT / "val_gallery.csv")
    assert (A2 / "val_query.ids").read_text().split() == [r["image_id"] for r in qm]
    assert (A2 / "val_gallery.ids").read_text().split() == [r["image_id"] for r in gm]
    for tag in ("osnet", "ainv2"):
        assert (A2 / f"val_query_{tag}.ids").read_text().split() == [r["image_id"] for r in qm] if (A2 / f"val_query_{tag}.ids").exists() else True
    vec = {tag: (l2n(np.load(A2 / f"val_query_{tag}.npy")), l2n(np.load(A2 / f"val_gallery_{tag}.npy")))
           for tag in ("osnet", "ainv2")}
    return vec, qm, gm


def load_train():
    """train_fit: векторы OSNet (job_42, crops_208 -> ORT) и ain_v2 (job_45/s01), vid/cam из train_fit.csv."""
    tf = read_meta(SPLIT / "train_fit.csv")
    ids = [r["image_id"] for r in tf]
    assert (J42 / "train_fit_osnet.ids").read_text().split() == ids
    assert (OUT / "train_fit_ainv2.ids").read_text().split() == ids
    X = {"osnet": l2n(np.load(J42 / "train_fit_osnet.npy")), "ainv2": l2n(np.load(OUT / "train_fit_ainv2.npy"))}
    vid = np.array([r["vehicle_id"] for r in tf]); cam = np.array([r["camera_id"] for r in tf])
    return X, vid, cam, ids


def run_eval(scores, qm, gm, *, fake_cameras=False, threshold=0.0):
    """Контур: market, presence. fake_cameras=True — ничего не исключается (разрыв по камере)."""
    qc = [f"q:{i}" for i in range(len(qm))] if fake_cameras else [r["camera_id"] for r in qm]
    gc = [f"g:{i}" for i in range(len(gm))] if fake_cameras else [r["camera_id"] for r in gm]
    return evaluate(scores, query_ids=[r["vehicle_id"] for r in qm], gallery_ids=[r["vehicle_id"] for r in gm],
                    query_cameras=qc, gallery_cameras=gc,
                    known_absent=np.array([r.get("has_mate", "1") == "0" for r in qm]),
                    threshold=threshold, camera_policy="market", refusal_mode="presence")


def summary(res):
    f, t = res["ranking_full_gallery"], res["ranking_top_k"]
    return {"mAP": f["mAP"], "Rank-1": f["Rank-1"], "Rank-5": f["Rank-5"], "mINP": f["mINP"],
            "mAP@10": t["mAP"], "valid": f["num_valid_queries"],
            "mAP_1110": f["mAP_by_query_average"]["all_queries_zero"]}


def per_query(res):
    known = [row for row in res["per_query"] if row["status"] == "known"]
    return np.array([row["ap"] for row in known]), np.array([float(row["rank1"]) for row in known])


def eval_config(q, g, qm, gm, tag, *, cos=True, noexcl=True):
    """cos + KR (market) + KR без исключения камеры; per-query AP/Rank-1 для KR и cos."""
    out, pq = {}, {}
    if cos:
        r = run_eval(scores_from_embeddings(q, g, metric="cosine"), qm, gm)
        out["cos"] = summary(r); pq["cos"] = per_query(r)
    dist, secs = rerank(q, g, *KR)
    r = run_eval(-dist, qm, gm); out["kr"] = summary(r); out["kr"]["rerank_seconds"] = round(secs, 2); pq["kr"] = per_query(r)
    if noexcl:
        out["kr_nocam"] = summary(run_eval(-dist, qm, gm, fake_cameras=True))
        out["camera_gap_kr"] = {k: out["kr_nocam"][k] - out["kr"][k] for k in ("mAP", "Rank-1")}
        if cos:
            out["cos_nocam"] = summary(run_eval(scores_from_embeddings(q, g, metric="cosine"), qm, gm, fake_cameras=True))
            out["camera_gap_cos"] = {k: out["cos_nocam"][k] - out["cos"][k] for k in ("mAP", "Rank-1")}
    print(f"[{tag}] " + (f"cos mAP={out['cos']['mAP']:.4f} R1={out['cos']['Rank-1']:.4f} | " if cos else "") +
          f"KR mAP={out['kr']['mAP']:.4f} R1={out['kr']['Rank-1']:.4f} R5={out['kr']['Rank-5']:.4f} mINP={out['kr']['mINP']:.4f} "
          f"AP@10={out['kr']['mAP@10']:.4f}" + (f" | gap(KR) {out['camera_gap_kr']['mAP']:+.4f}" if noexcl else ""), flush=True)
    return out, pq


# ---- whitening: дословно job_42/s2_whitening.py (learn_lw / apply_lw)
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


def lw_stats(lw):
    return {k: v for k, v in lw.items() if k not in ("m", "P")}


def load_lw_osnet():
    """Готовая матрица job_42 (ρ=0,5, учёна на всём train_fit), как она пойдёт в решение (float32)."""
    return {"m": np.load(J42 / "lw_m_rho0.5.npy").astype(np.float64), "P": np.load(J42 / "lw_P_rho0.5.npy").astype(np.float64)}


# ---- бутстрэп: дословно job_42/lib42.py
def paired_bootstrap(a, b, B=B_BOOT, seed=SEED_BOOT, groups=None):
    """a - b по среднему per-query значению; парный бутстрэп по запросам (или по группам-ID)."""
    a = np.asarray(a, float); b = np.asarray(b, float); n = len(a)
    assert len(b) == n
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, n, size=(B, n))
        da = a[idx].mean(1) - b[idx].mean(1)
    else:
        g = np.asarray(groups); ug = np.unique(g); members = [np.flatnonzero(g == u) for u in ug]
        da = np.empty(B)
        for t in range(B):
            pick = rng.integers(0, len(ug), size=len(ug))
            sel = np.concatenate([members[p] for p in pick])
            da[t] = a[sel].mean() - b[sel].mean()
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "delta": float(a.mean() - b.mean()),
            "ci95": [float(np.quantile(da, 0.025)), float(np.quantile(da, 0.975))],
            "p_two_sided": float(2 * min((da <= 0).mean(), (da >= 0).mean())),
            "sd_delta": float(da.std(ddof=1)), "n": int(n), "B": B, "seed": seed,
            "grouped_by_id": groups is not None}


def dump_json(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n")


# ---- конфигурации (одно место для s02/s03/s04): матрицы кэшируются в OUT/*_rho0.5_f64.npz
def get_lw(name, X=None, vid=None, cam=None):
    """Lw ρ=0,5 по имени: 'ainv2' | 'ens_w{w}' | 'cat_w{w}'; учится на train_fit один раз, потом читается."""
    f = OUT / f"lw_{name}_rho0.5_f64.npz"
    if f.exists():
        z = np.load(f); return {"m": z["m"], "P": z["P"]}, json.load(open(OUT / f"lw_{name}_stats.json"))
    assert X is not None
    if name == "ainv2":
        Xt = X["ainv2"]
    elif name.startswith("ens_w"):
        Xt = l2n(X["osnet"] + float(name[5:]) * X["ainv2"])
    elif name.startswith("cat_w"):
        Xt = np.concatenate([X["osnet"], float(name[5:]) * X["ainv2"]], 1)
    else:
        raise ValueError(name)
    import time; t0 = time.perf_counter(); lw = learn_lw(Xt, vid, cam, RHO)
    st = lw_stats(lw) | {"learn_s": round(time.perf_counter() - t0, 1), "dim": int(Xt.shape[1])}
    np.savez(f, P=lw["P"], m=lw["m"]); json.dump(st, open(OUT / f"lw_{name}_stats.json", "w"), indent=1)
    np.save(OUT / f"lw_{name}_P_rho0.5.npy", lw["P"].astype(np.float32)); np.save(OUT / f"lw_{name}_m_rho0.5.npy", lw["m"].astype(np.float32))
    return {"m": lw["m"], "P": lw["P"]}, st


def config_vectors(name, o, a, X=None, vid=None, cam=None):
    """Векторы конфигурации из (OSNet o, ain_v2 a) — одна формула для val, tune и вариантов абляции."""
    if name in ("a", "osnet"): return o
    if name == "ainv2": return a
    if name == "b": return apply_lw(o, load_lw_osnet())
    kind, w = name.split("_w"); w = float(w)
    if kind == "c": return l2n(o + w * a)
    if kind == "d1": return apply_lw(l2n(o + w * a), get_lw(f"ens_w{w}", X, vid, cam)[0])
    if kind == "d2": return l2n(apply_lw(o, load_lw_osnet()) + w * apply_lw(a, get_lw("ainv2", X, vid, cam)[0]))
    if kind == "d3": return apply_lw(np.concatenate([o, w * a], 1), get_lw(f"cat_w{w}", X, vid, cam)[0])
    raise ValueError(name)


CONFIG_NOTES = {"a": "OSNet (эталон 0,6937)", "b": "Lw_osnet(job_42, ρ=0,5) -> OSNet",
                "c_w1.0": "ансамбль l2(OSNet + ain_v2)", "c_w0.5": "ансамбль l2(OSNet + 0,5·ain_v2)",
                "d1_w1.0": "Lw_ens(train_fit) -> l2(OSNet + ain_v2)", "d1_w0.5": "Lw_ens(train_fit) -> l2(OSNet + 0,5·ain_v2)",
                "d2_w1.0": "l2(Lw_osnet(OSNet) + Lw_ainv2(ain_v2))", "d2_w0.5": "l2(Lw_osnet(OSNet) + 0,5·Lw_ainv2(ain_v2))",
                "d3_w1.0": "Lw_cat(train_fit, 1024) -> [OSNet, ain_v2]", "d3_w0.5": "Lw_cat(train_fit, 1024) -> [OSNet, 0,5·ain_v2]"}
CONFIG_ORDER = ["a", "b", "c_w1.0", "c_w0.5", "d1_w1.0", "d1_w0.5", "d2_w1.0", "d2_w0.5", "d3_w1.0", "d3_w0.5"]
