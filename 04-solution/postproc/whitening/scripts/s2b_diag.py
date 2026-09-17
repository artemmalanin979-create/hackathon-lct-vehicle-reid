"""Диагностика к шагу 2 (не подбор; кандидат остаётся ρ=0.5, выбранный на tune):
(a) разложение выигрыша: только центрирование (ρ=1 ⇒ P ∝ I: x -> (x-m)/‖x-m‖) и whitening ρ=0.5 БЕЗ центрирования;
(b) слабый прокси-контроль номера: тот же Lw(ρ=0.5)+KR на старых векторах baseline/out val_*_mask30 / _gray /
    _mask30gray (нижние 30 % кропа серым / обесцвечивание; это НЕ честная абляция пластины — см. plate-ablation/REPORT.md,
    у 38,8 % кропов пластина целиком выше линии 30 %). Вопрос: сохраняется ли Δ(Lw+KR − KR) при закрытой нижней трети?
(c) стоимость преобразования на 1860 векторов."""
import json, time
import numpy as np
from lib42 import *
src = open("s2_whitening.py").read().split("tf = read_meta(")[0]; ns = {}; exec(compile(src, "s2_head", "exec"), ns)
learn_lw, apply_lw = ns["learn_lw"], ns["apply_lw"]

tf = read_meta(SPLIT / "train_fit.csv"); X = np.load(OUT / "train_fit_osnet.npy").astype(np.float64)
vid = np.array([r["vehicle_id"] for r in tf]); cam = np.array([r["camera_id"] for r in tf])
q, g, qm, gm = load_val(); z = np.load(OUT / "perquery_step0.npz"); known = z["known"]; ap_kr = z["ap_kr"]
out = {}


def run_cfg(qq, gg, tag, ref_ap=ap_kr, qmm=qm, gmm=gm):
    dist, _ = rerank(qq, gg, *KR); res = run_eval(-dist, qmm, gmm); pq = per_query(res)
    s = summary(res); s["boot_vs_ref"] = paired_bootstrap(pq["ap"][known], ref_ap[known])
    s["first"] = step0_numbers(pq, tag)["first_correct_position"]
    print(f"[{tag}] mAP={s['mAP']:.4f} R1={s['Rank-1']:.4f} AP@10={s['mAP@10(R)']:.4f} d_vs_ref={s['boot_vs_ref']['delta']:+.4f} CI={np.round(s['boot_vs_ref']['ci95'],4).tolist()} p={s['boot_vs_ref']['p_two_sided']}", flush=True)
    return s, pq


# (a)
lw1 = learn_lw(X, vid, cam, 1.0); assert np.allclose(lw1["P"], lw1["P"][0, 0] * np.eye(512))
out["center_only_rho1"], _ = run_cfg(apply_lw(q, lw1), apply_lw(g, lw1), "centering only (rho=1) + KR")
lw5 = learn_lw(X, vid, cam, 0.5); lw5nc = dict(lw5, m=np.zeros(512))
out["rho0.5_no_centering"], _ = run_cfg(apply_lw(q, lw5nc), apply_lw(g, lw5nc), "Lw rho=0.5 without centering + KR")
out["rho0.5_full"], pq5 = run_cfg(apply_lw(q, lw5), apply_lw(g, lw5), "Lw rho=0.5 full + KR")
# (b) прокси-контроль номера
out["plate_proxy"] = {}
for v in ("mask30", "gray", "mask30gray"):
    qv = np.load(BASE_OUT / f"val_query_{v}.npy"); gv = np.load(BASE_OUT / f"val_gallery_{v}.npy")
    base_s, base_pq = run_cfg(qv, gv, f"{v}: KR only")
    lw_s, lw_pq = run_cfg(apply_lw(qv, lw5), apply_lw(gv, lw5), f"{v}: Lw0.5+KR", ref_ap=base_pq["ap"])
    out["plate_proxy"][v] = {"kr_only": {k: base_s[k] for k in ("mAP", "Rank-1", "mAP@10(R)")},
                             "lw_kr": {k: lw_s[k] for k in ("mAP", "Rank-1", "mAP@10(R)")},
                             "delta_lw_minus_kr_on_masked": lw_s["boot_vs_ref"]}
# (c) стоимость
t0 = time.perf_counter()
for _ in range(20):
    apply_lw(np.concatenate([q, g]), lw5)
out["transform_cost"] = {"ms_per_1860_vectors_float64": round(1000 * (time.perf_counter() - t0) / 20, 2),
                         "matrix_bytes_fp32": int(512 * 512 * 4 + 512 * 4)}
dump_json(out, OUT / "step2b_diag.json"); print(json.dumps(out["transform_cost"]))
