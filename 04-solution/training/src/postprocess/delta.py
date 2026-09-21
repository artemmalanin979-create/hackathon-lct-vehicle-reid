#!/usr/bin/env python3
"""job_55: дельта метрик d1_j48 от float32-конверсии whitening (P, m).

Контур — только 04-solution/eval/reid_metrics.py (market, presence, KR 6,3,0.3
из postproc/scripts). Эталон — f64 P,m из job_48 (KR mAP 0.7740915539438481).
Сравниваем три варианта применения whitening к одним и тем же векторам
l2n(OSNet+j48) валидации:
  A f64 P,m, матmul f64     — эталон job_48 (контроль сходимости)
  B f32 P,m, матmul f64     — чистая дельта конверсии хранения
  C f32 P,m, матmul f32     — как в сервисе (app/core/model.py)
"""
import csv, json, sys
from pathlib import Path
import numpy as np

HOME = Path.home() / "lct-reid"
REPO = HOME / "repo"
A2 = REPO / "04-solution" / "training" / "attempt-2" / "out"
J48 = HOME / "jobs" / "job_48" / "out"
SPLIT = REPO / "04-solution" / "split" / "files"
sys.path.insert(0, str(REPO / "04-solution" / "eval"))
sys.path.insert(0, str(REPO / "04-solution" / "postproc" / "scripts"))
sys.path.insert(0, str(REPO / "04-solution" / "service"))
from reid_metrics import evaluate  # noqa: E402
from app.core.rerank import rerank_distances as _unused  # noqa: E402

def l2n(m):
    m = np.asarray(m, np.float64)
    return m / np.linalg.norm(m, axis=1, keepdims=True)

def kr_scores(q, g, k1=6, k2=3, lam=0.3):
    """k-reciprocal (Zhong) — дословно app/core/rerank.py: уверенность 1-d."""
    sys.path.insert(0, str(REPO / "04-solution" / "service"))
    from app.core.rerank import rerank_distances, distances_to_scores
    return distances_to_scores(rerank_distances(q, g, k1, k2, lam))

def read_meta(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))

qm = read_meta(SPLIT / "val_query.csv"); gm = read_meta(SPLIT / "val_gallery.csv")
q_ids = [r["image_id"] for r in qm]; g_ids = [r["image_id"] for r in gm]
assert (J48 / "val_query_j48.ids").read_text().split() == q_ids
assert (J48 / "val_gallery_j48.ids").read_text().split() == g_ids

q = l2n(np.load(A2 / "val_query_osnet.npy") + np.load(J48 / "val_query_j48.npy"))
g = l2n(np.load(A2 / "val_gallery_osnet.npy") + np.load(J48 / "val_gallery_j48.npy"))

z64 = np.load(J48 / "lw_ens_j48_rho0.5_f64.npz"); P64, m64 = z64["P"], z64["m"]
z32 = np.load(REPO / "04-solution" / "service" / "model" / "lw_ens_j48_rho0.5.npz")
P32, m32 = z32["P"], z32["m"]
assert P32.dtype == np.float32 and m32.dtype == np.float32

def apply_lw(x, P, m, f64):
    if f64:
        return l2n((np.asarray(x, np.float64) - m.astype(np.float64)) @ P.astype(np.float64).T)
    xf = (x.astype(np.float32) - m) @ P.T
    return l2n(xf)

kwargs = dict(query_ids=[r["vehicle_id"] for r in qm], gallery_ids=[r["vehicle_id"] for r in gm],
              query_cameras=[r["camera_id"] for r in qm], gallery_cameras=[r["camera_id"] for r in gm],
              known_absent=np.array([r["has_mate"] == "0" for r in qm]),
              camera_policy="market", refusal_mode="presence")

out = {}
for tag, P, m, f64 in (("A_f64_ref", P64, m64, True), ("B_f32_storage", P32, m32, True),
                       ("C_f32_service", P32, m32, False)):
    qw, gw = apply_lw(q, P, m, f64), apply_lw(g, P, m, f64)
    s = kr_scores(qw, gw)
    r = evaluate(s, threshold=-1e9, **kwargs)["ranking_full_gallery"]
    c = evaluate(np.clip(qw @ gw.T, -1, 1), threshold=-1e9, **kwargs)["ranking_full_gallery"]
    out[tag] = {"kr_mAP": r["mAP"], "kr_Rank1": r["Rank-1"], "cos_mAP": c["mAP"], "cos_Rank1": c["Rank-1"]}
    print(tag, json.dumps(out[tag]), flush=True)

ref = out["A_f64_ref"]
for tag in ("B_f32_storage", "C_f32_service"):
    out[tag]["d_kr_mAP"] = out[tag]["kr_mAP"] - ref["kr_mAP"]
    out[tag]["d_cos_mAP"] = out[tag]["cos_mAP"] - ref["cos_mAP"]
out["max_abs_emb_diff_B"] = float(np.abs(apply_lw(q, P32, m32, True) - apply_lw(q, P64, m64, True)).max())
Path(HOME / "jobs" / "job_55" / "out" / "f32_delta.json").write_text(json.dumps(out, indent=2) + "\n")
print("DONE")
