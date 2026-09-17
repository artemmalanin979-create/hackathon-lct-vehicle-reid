#!/usr/bin/env python3
"""job_43, шаги 2–3: полная таблица (3 модели + ансамбли) x (косинус, переранжирование)
x (market / без исключения камеры). Метрики — только контуром 04-solution/eval/reid_metrics.py,
переранжирование — канонический порт из 04-solution/postproc/scripts/common.py (импорт).
"""
import csv, json, sys, time
from pathlib import Path
import numpy as np

HOME = Path.home() / "lct-reid"; REPO = HOME / "repo"; S = REPO / "04-solution"
OUT = HOME / "jobs/job_43/out"
sys.path.insert(0, str(S / "eval")); sys.path.insert(0, str(S / "postproc/scripts"))
from reid_metrics import evaluate, scores_from_embeddings  # noqa
from common import rerank  # noqa

A2 = S / "training/attempt-2/out"
K1, K2, LAM = 6, 3, 0.3

def rd(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))
qm, gm = rd(S / "split/files/val_query.csv"), rd(S / "split/files/val_gallery.csv")
assert (A2 / "val_query.ids").read_text().split() == [r["image_id"] for r in qm]
assert (A2 / "val_gallery.ids").read_text().split() == [r["image_id"] for r in gm]

def run(scores, *, fake_cameras=False):
    qc = [f"q:{i}" for i in range(len(qm))] if fake_cameras else [r["camera_id"] for r in qm]
    gc = [f"g:{i}" for i in range(len(gm))] if fake_cameras else [r["camera_id"] for r in gm]
    r = evaluate(scores, query_ids=[r["vehicle_id"] for r in qm], gallery_ids=[r["vehicle_id"] for r in gm],
                 query_cameras=qc, gallery_cameras=gc,
                 known_absent=np.array([r["has_mate"] == "0" for r in qm]),
                 threshold=0.0, camera_policy="market", refusal_mode="presence")
    f = r["ranking_full_gallery"]; t = r["ranking_top_k"]
    known = [row for row in r["per_query"] if row["status"] == "known"]
    return ({"mAP": f["mAP"], "Rank-1": f["Rank-1"], "Rank-5": f["Rank-5"], "mINP": f["mINP"],
             "valid_queries": f["num_valid_queries"], "mAP@10": t["mAP"]},
            np.array([row["ap"] for row in known]), np.array([row["rank1"] for row in known]))

def l2(m):
    m = m.astype(np.float64); return m / np.linalg.norm(m, axis=1, keepdims=True)

def load(tag):
    return l2(np.load(A2 / f"val_query_{tag}.npy")), l2(np.load(A2 / f"val_gallery_{tag}.npy"))

base = load("osnet"); v1 = load("ainv1"); v2 = load("ainv2")
configs = {"osnet": base, "ainv1": v1, "ainv2": v2}
for w in (0.25, 0.5, 1.0):
    configs[f"cat_w{w}"] = tuple(np.concatenate([b, w * a], axis=1) for b, a in zip(base, v2))
configs["avg"] = tuple(l2(b + a) for b, a in zip(base, v2))
for w in (0.25, 0.5):   # дополнительно: взвешенное среднее (не требовалось, для полноты)
    configs[f"avg_w{w}"] = tuple(l2(b + w * a) for b, a in zip(base, v2))

table, aps, r1s = {}, {}, {}
for name, (q, g) in configs.items():
    t0 = time.perf_counter()
    sc = scores_from_embeddings(q, g, metric="cosine")
    m, ap, r1 = run(sc); table[f"{name}|cos|market"] = m; aps[f"{name}|cos"] = ap; r1s[f"{name}|cos"] = r1
    m2, _, _ = run(sc, fake_cameras=True); table[f"{name}|cos|nocam"] = m2
    d, secs = rerank(q, g, K1, K2, LAM)
    m, ap, r1 = run(-d); m["rerank_seconds"] = round(secs, 2)
    table[f"{name}|rr|market"] = m; aps[f"{name}|rr"] = ap; r1s[f"{name}|rr"] = r1
    m2, _, _ = run(-d, fake_cameras=True); table[f"{name}|rr|nocam"] = m2
    for sc_name in ("cos", "rr"):
        a, b = table[f"{name}|{sc_name}|market"], table[f"{name}|{sc_name}|nocam"]
        table[f"{name}|{sc_name}|gap"] = {k: b[k] - a[k] for k in ("mAP", "Rank-1", "Rank-5", "mINP")}
    tc, tr = table[f"{name}|cos|market"], table[f"{name}|rr|market"]
    print(f"[{name}] dim={q.shape[1]} cos mAP={tc['mAP']:.4f} R1={tc['Rank-1']:.4f} | "
          f"rr mAP={tr['mAP']:.4f} R1={tr['Rank-1']:.4f} | "
          f"gap(cos) {table[name+'|cos|gap']['mAP']:+.4f} gap(rr) {table[name+'|rr|gap']['mAP']:+.4f} | "
          f"{time.perf_counter()-t0:.1f}s", flush=True)

(OUT / "s04_table.json").write_text(json.dumps(table, indent=2) + "\n")
np.savez(OUT / "s04_aps.npz", **aps); np.savez(OUT / "s04_rank1.npz", **r1s)
print("DONE")
