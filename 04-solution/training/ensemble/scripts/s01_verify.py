#!/usr/bin/env python3
"""job_43: сверка эталонных векторов между собой (то, что достижимо без кадров)."""
import csv, json
from pathlib import Path
import numpy as np

HOME = Path.home() / "lct-reid"; REPO = HOME / "repo"; S = REPO / "04-solution"
OUT = HOME / "jobs/job_43/out"

def load(p): return np.load(p, allow_pickle=False)
def maxdiff(a, b): return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
def ids(p): return Path(p).read_text().split()
def csv_ids(p):
    with open(p, newline="") as f: return [r["image_id"] for r in csv.DictReader(f)]

res = {}
# (A) тестовый набор: artifacts-final vs baseline/out/test_*.npy vs baseline/artifacts
fin = load(S / "service/artifacts-final/embeddings.npy")
bq, bg = load(S / "baseline/out/test_query.npy"), load(S / "baseline/out/test_gallery.npy")
res["A_test"] = {
    "artifacts_final_shape": list(fin.shape), "dtype": str(fin.dtype),
    "max_abs_diff_vs_baseline_out_test": maxdiff(fin, np.concatenate([bq, bg])),
    "max_abs_diff_vs_baseline_artifacts": maxdiff(fin, load(S / "baseline/artifacts/embeddings.npy")),
    "row_norm_min": float(np.linalg.norm(fin.astype(np.float64), axis=1).min()),
    "row_norm_max": float(np.linalg.norm(fin.astype(np.float64), axis=1).max()),
}
# (B) val-сплит: OSNet из трёх независимых прогонов
A2 = S / "training/attempt-2/out"
vq_ref, vg_ref = csv_ids(S / "split/files/val_query.csv"), csv_ids(S / "split/files/val_gallery.csv")
res["B_val_ids"] = {
    "attempt2_query_ids_eq_csv": ids(A2 / "val_query.ids") == vq_ref,
    "attempt2_gallery_ids_eq_csv": ids(A2 / "val_gallery.ids") == vg_ref,
    "attempt2_ainv2_query_ids_eq_csv": ids(A2 / "val_query_ainv2.ids") == vq_ref,
    "attempt2_ainv1_query_ids_eq_csv": ids(A2 / "val_query_ainv1.ids") == vq_ref,
    "baseline_query_ids_eq_csv": ids(S / "baseline/out/val_query.ids") == vq_ref,
    "postproc_query_ids_eq_csv": ids(S / "postproc/out/val_query.ids") == vq_ref,
    "baseline_gallery_ids_eq_csv": ids(S / "baseline/out/val_gallery.ids") == vg_ref,
}
pairs = {
    "attempt2_osnet_vs_baseline": (A2 / "val_query_osnet.npy", S / "baseline/out/val_query.npy",
                                   A2 / "val_gallery_osnet.npy", S / "baseline/out/val_gallery.npy"),
    "attempt2_osnet_vs_postproc208": (A2 / "val_query_osnet.npy", S / "postproc/out/val_query_208.npy",
                                      A2 / "val_gallery_osnet.npy", S / "postproc/out/val_gallery_208.npy"),
    "baseline_vs_postproc": (S / "baseline/out/val_query.npy", S / "postproc/out/val_query.npy",
                             S / "baseline/out/val_gallery.npy", S / "postproc/out/val_gallery.npy"),
}
res["B_val_osnet"] = {k: {"query": maxdiff(load(a), load(b)), "gallery": maxdiff(load(c), load(d))}
                      for k, (a, b, c, d) in pairs.items()}
# нормы и попарные косинусы моделей на одних и тех же изображениях val
def l2(m): m = m.astype(np.float64); return m / np.linalg.norm(m, axis=1, keepdims=True)
V = {t: (load(A2 / f"val_query_{t}.npy"), load(A2 / f"val_gallery_{t}.npy")) for t in ("osnet", "ainv1", "ainv2")}
res["C_val_models"] = {}
for t, (q, g) in V.items():
    m = np.concatenate([q, g]).astype(np.float64)
    nr = np.linalg.norm(m, axis=1)
    res["C_val_models"][t] = {"shape": [list(q.shape), list(g.shape)], "norm_min": float(nr.min()), "norm_max": float(nr.max())}
def cos_same(t1, t2):
    a = l2(np.concatenate(V[t1])); b = l2(np.concatenate(V[t2]))
    c = np.sum(a * b, axis=1); return {"mean": float(c.mean()), "min": float(c.min()), "p10": float(np.quantile(c, .1))}
res["C_val_models"]["cos_osnet_ainv2_same_image"] = cos_same("osnet", "ainv2")
res["C_val_models"]["cos_osnet_ainv1_same_image"] = cos_same("osnet", "ainv1")
(OUT / "s01_verify.json").write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps(res, indent=2))
