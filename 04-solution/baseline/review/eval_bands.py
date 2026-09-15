#!/usr/bin/env python3
"""Контроль абляции: mAP при закрашивании 30% полосы снизу / сверху / посередине."""
import csv, sys
from pathlib import Path
import numpy as np
JOB=Path(__file__).resolve().parent.parent
SPLIT=Path("/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/split/files")
sys.path.insert(0,"/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/eval")
from reid_metrics import evaluate, scores_from_embeddings
def rd(p): return list(csv.DictReader(open(p,newline="")))
qm,gm=rd(SPLIT/"val_query.csv"),rd(SPLIT/"val_gallery.csv")
QV=[r["vehicle_id"] for r in qm];GV=[r["vehicle_id"] for r in gm]
QC=[r["camera_id"] for r in qm];GC=[r["camera_id"] for r in gm]
ABS=np.array([r["has_mate"]=="0" for r in qm])
def l2(x): return x/np.linalg.norm(x,axis=1,keepdims=True)
def rep(name,q,g):
    r=evaluate(scores_from_embeddings(q,g),query_ids=QV,gallery_ids=GV,query_cameras=QC,
               gallery_cameras=GC,known_absent=ABS,threshold=0.34921352213815304,
               camera_policy="market",refusal_mode="presence")
    f=r["ranking_full_gallery"]
    print(f"{name:28s} mAP={f['mAP']:.4f}  Δ={f['mAP']-0.6565692724907637:+.4f}   R1={f['Rank-1']:.4f}  TNR={r['refusal']['tnr']:.4f}")
rep("без вмешательства", l2(np.load(JOB/"subject/out/val_query.npy").astype(np.float64)),
                         l2(np.load(JOB/"subject/out/val_gallery.npy").astype(np.float64)))
for v,label in [("btm30","низ 30% (их абляция)"),("top30","ВЕРХ 30% (контроль)"),("mid30","СЕРЕДИНА 30% (контроль)")]:
    rep(label, l2(np.load(JOB/f"work/emb/val_query_{v}.npy").astype(np.float64)),
               l2(np.load(JOB/f"work/emb/val_gallery_{v}.npy").astype(np.float64)))
