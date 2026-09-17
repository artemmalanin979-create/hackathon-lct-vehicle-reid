"""Диагностика к шагу 1 (НЕ кандидат, без подбора): тот же diffusion, но граф общий — все 1110 запросов +
750 галереи как узлы (как у KR, который считается на объединении). Параметры = выбранные на tune
(k=20, α=0.8, γ=3, T=25); restart — на собственный узел запроса (one-hot). Вопрос: проигрыш KR из-за
того, что gallery-only граф не видит структуры запросов, или из-за самого механизма?"""
import json, time
import numpy as np
from lib42 import *
src = open("s1_diffusion.py").read().split("# ---- tune-протокол")[0]; ns = {}; exec(compile(src, "s1_head", "exec"), ns)
K, ALPHA, T = 20, 0.8, 25


def diffuse_joint(Q, G, k=K, alpha=ALPHA, T=T):
    X = np.concatenate([l2n(Q), l2n(G)]); nq = len(Q)
    S, Xn, info = ns["build_S"](X, k)
    cos = np.clip(Xn[:nq] @ Xn[nq:].T, -1, 1)
    Y = np.zeros((nq, len(X))); Y[np.arange(nq), np.arange(nq)] = 1.0
    F = Y.copy()
    for t in range(T):
        F = alpha * (F @ S.T) + (1 - alpha) * Y
    Fg = F[:, nq:]
    scores = np.empty_like(Fg)
    for i in range(nq):
        order = np.lexsort((-cos[i], -Fg[i])); scores[i, order] = -np.arange(len(order), dtype=np.float64)
    return scores, info


out = {"params": {"k": K, "alpha": ALPHA, "T": T, "gamma": ns["GAMMA"], "restart": "one-hot own node", "graph": "queries+gallery"}}
tq_m, tg_m = read_meta(TUNE / "tune_query.csv"), read_meta(TUNE / "tune_gallery.csv")
ids = (OUT / "train_fit_osnet.ids").read_text().split(); emb = np.load(OUT / "train_fit_osnet.npy"); pos = {v: i for i, v in enumerate(ids)}
tq = emb[[pos[r["image_id"]] for r in tq_m]]; tg = emb[[pos[r["image_id"]] for r in tg_m]]
sc, info = diffuse_joint(tq, tg); out["tune"] = {"summary": summary(run_eval(sc, tq_m, tg_m)), "graph": info}
q, g, qm, gm = load_val()
sc, info = diffuse_joint(q, g); res = run_eval(sc, qm, gm); pq = per_query(res)
out["val"] = {"summary": summary(res), "graph": info, "camera_gap": summary(run_eval(sc, qm, gm, fake_cameras=True))["mAP"] - summary(res)["mAP"]}
z = np.load(OUT / "perquery_step0.npz"); known = z["known"]
out["bootstrap_val"] = {"mAP_vs_KR": paired_bootstrap(pq["ap"][known], z["ap_kr"][known]),
                        "mAP_vs_cosine": paired_bootstrap(pq["ap"][known], z["ap_cos"][known])}
out["val"]["step0"] = step0_numbers(pq, "joint diffusion")
dump_json(out, OUT / "step1b_diffusion_joint.json")
print(json.dumps({k: out[k] for k in ("tune", "val")}, ensure_ascii=False)[:1500]); b = out["bootstrap_val"]["mAP_vs_KR"]
print(f"JOINT val mAP {out['val']['summary']['mAP']:.4f} vs KR: {b['delta']:+.4f} CI {b['ci95']} p={b['p_two_sided']}")
