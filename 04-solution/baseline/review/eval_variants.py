#!/usr/bin/env python3
"""Метрики контуром для вариантов препроцессинга (те же веса, обучения нет)."""
import csv, json, sys, itertools
from pathlib import Path
import numpy as np
JOB = Path(__file__).resolve().parent.parent
EMB = JOB/"work"/"emb"
SPLIT = Path("/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/split/files")
sys.path.insert(0, "/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/eval")
from reid_metrics import evaluate, scores_from_embeddings
def rd(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))
qm, gm = rd(SPLIT/"val_query.csv"), rd(SPLIT/"val_gallery.csv")
QV=[r["vehicle_id"] for r in qm]; GV=[r["vehicle_id"] for r in gm]
QC=[r["camera_id"] for r in qm]; GC=[r["camera_id"] for r in gm]
ABS=np.array([r["has_mate"]=="0" for r in qm])
def l2(x): return x/np.linalg.norm(x,axis=1,keepdims=True)
def load(v): return l2(np.load(EMB/f"val_query_{v}.npy").astype(np.float64)), l2(np.load(EMB/f"val_gallery_{v}.npy").astype(np.float64))
rows=[]
def report(name, q, g):
    res = evaluate(scores_from_embeddings(q,g), query_ids=QV, gallery_ids=GV, query_cameras=QC,
                   gallery_cameras=GC, known_absent=ABS, threshold=0.0,
                   camera_policy="market", refusal_mode="presence")
    f=res["ranking_full_gallery"]; t=res["ranking_top_k"]
    r=dict(name=name, mAP=f["mAP"], R1=f["Rank-1"], R5=f["Rank-5"], mAP10=t["mAP"], mINP=f["mINP"])
    print(f"{name:32s} mAP={r['mAP']:.4f} R1={r['R1']:.4f} R5={r['R5']:.4f} mAP@10={r['mAP10']:.4f}", flush=True)
    rows.append(r); return q,g
V = {v: load(v) for v in ["base208","bgr208","flip208","res256","res288","letterbox208"]}
for v in ["base208","bgr208","flip208","res256","res288","letterbox208"]:
    report(v, *V[v])
# flip-TTA: среднее base и flip
report("TTA base+flip", l2(V["base208"][0]+V["flip208"][0]), l2(V["base208"][1]+V["flip208"][1]))
# мультимасштаб
report("TTA 208+256", l2(V["base208"][0]+V["res256"][0]), l2(V["base208"][1]+V["res256"][1]))
report("TTA 208+256+288", l2(V["base208"][0]+V["res256"][0]+V["res288"][0]), l2(V["base208"][1]+V["res256"][1]+V["res288"][1]))
report("TTA 208+256+288+flip", l2(V["base208"][0]+V["res256"][0]+V["res288"][0]+V["flip208"][0]),
                               l2(V["base208"][1]+V["res256"][1]+V["res288"][1]+V["flip208"][1]))
report("TTA 208+flip+letterbox", l2(V["base208"][0]+V["flip208"][0]+V["letterbox208"][0]),
                                 l2(V["base208"][1]+V["flip208"][1]+V["letterbox208"][1]))
json.dump(rows, open(JOB/"work/variants.json","w"), indent=1)
