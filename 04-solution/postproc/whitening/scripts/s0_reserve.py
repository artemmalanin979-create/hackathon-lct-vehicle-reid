"""Шаг 0: где лежит резерв. Текущая система = OSNet-AIN (baseline/out/val_*.npy) + KR(6,3,0.3).
Всё через контур; выход out/step0.json + out/perquery_step0.npz."""
import time, json
import numpy as np
from lib42 import *

q, g, qm, gm = load_val()
cos = scores_from_embeddings(q, g, metric="cosine")
t0 = time.perf_counter(); dist_kr, secs = rerank(q, g, *KR); print("KR seconds", round(secs, 2))
res = {"cosine": run_eval(cos, qm, gm), "kr": run_eval(-dist_kr, qm, gm)}
res_noexcl = {"cosine": run_eval(cos, qm, gm, fake_cameras=True),
              "kr": run_eval(-dist_kr, qm, gm, fake_cameras=True)}
out = {"summary": {k: summary(v) for k, v in res.items()},
       "summary_no_camera_exclusion": {k: summary(v) for k, v in res_noexcl.items()},
       "camera_gap": {k: res_noexcl[k]["ranking_full_gallery"]["mAP"] - res[k]["ranking_full_gallery"]["mAP"]
                      for k in res}}
assert abs(out["summary"]["kr"]["mAP"] - BASELINE_MAP) < 1e-9, out["summary"]["kr"]["mAP"]
print(json.dumps(out, indent=1, ensure_ascii=False))
pq = {k: per_query(v) for k, v in res.items()}
out["step0"] = {k: step0_numbers(pq[k], k) for k in pq}
# σ оценки mAP (бутстрэп по 832 запросам) — сверка с σ = 0,0134 из брифа
out["sigma_mAP_bootstrap"] = {k: sd_of_mean(pq[k]["ap"][pq[k]["known"]]) for k in pq}
out["kr_minus_cosine_bootstrap"] = paired_bootstrap(pq["kr"]["ap"][pq["kr"]["known"]],
                                                    pq["cosine"]["ap"][pq["cosine"]["known"]])
dump_json(out, OUT / "step0.json")
np.savez(OUT / "perquery_step0.npz",
         known=pq["kr"]["known"], R=pq["kr"]["R"],
         ap_kr=pq["kr"]["ap"], first_kr=pq["kr"]["first"],
         ap_cos=pq["cosine"]["ap"], first_cos=pq["cosine"]["first"],
         ap10R_kr=pq["kr"]["ap10_R_f"], ap10R_cos=pq["cosine"]["ap10_R_f"],
         vehicle_id=np.array([r["vehicle_id"] for r in qm]))
print(json.dumps(out["step0"]["kr"], indent=1, ensure_ascii=False))
print("sigma", out["sigma_mAP_bootstrap"], "kr-cos", out["kr_minus_cosine_bootstrap"])
