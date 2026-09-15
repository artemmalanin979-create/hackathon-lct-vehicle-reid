#!/usr/bin/env python3
"""Компромисс F1/TNR по отфильтрованным top-1 + точность принятых пар."""
import csv
from pathlib import Path
import numpy as np
JOB=Path(__file__).resolve().parent.parent
SPLIT=Path("/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/split/files")
def rd(p): return list(csv.DictReader(open(p,newline="")))
qm,gm=rd(SPLIT/"val_query.csv"),rd(SPLIT/"val_gallery.csv")
qv=np.array([r["vehicle_id"] for r in qm]); gv=np.array([r["vehicle_id"] for r in gm])
qc=np.array([r["camera_id"] for r in qm]); gc=np.array([r["camera_id"] for r in gm])
mate=np.array([r["has_mate"]=="1" for r in qm])
def l2(x): return x/np.linalg.norm(x,axis=1,keepdims=True)
S=l2(np.load(JOB/"subject/out/val_query.npy").astype(np.float64))@l2(np.load(JOB/"subject/out/val_gallery.npy").astype(np.float64)).T
same_id=(gv[None,:]==qv[:,None]); keep=~(same_id&(gc[None,:]==qc[:,None]))
top1=np.array([S[i,keep[i]].max() for i in range(len(qm))])
def stat(t):
    acc=top1>=t; tp=int((acc&mate).sum()); fp=int((acc&~mate).sum()); fn=int(mate.sum())-tp
    return 2*tp/(2*tp+fp+fn), int((~acc&~mate).sum())/int((~mate).sum()), tp/max(tp+fp,1), tp/(tp+fn), acc.mean()
print("порог   F1      TNR     prec    recall  доля_принятых")
for t in [0.34921352213815304,0.40,0.45,0.4803,0.50,0.55,0.60]:
    print(("%.4f  "+"%.4f  "*5)%((t,)+stat(t)))
for t in (0.34921352213815304,):
    for tag,msk in [("market",(S>=t)&keep),("без фильтра",(S>=t))]:
        n=int(msk.sum()); tp=int((msk&same_id).sum())
        print(f"пары {tag} при t={t:.4f}: принято {n}, верных {tp}, точность {tp/n:.4f}")
E=np.load(JOB/"subject/artifacts/embeddings.npy").astype(np.float64)
St=l2(E[:1110])@l2(E[1110:]).T
t=0.34921352213815304
print("ТЕСТ: отказов %d/1110 (%.3f%%); val без фильтра: %d/1110 (%.3f%%)"
      % (int((St.max(1)<t).sum()),(St.max(1)<t).mean()*100,int((S.max(1)<t).sum()),(S.max(1)<t).mean()*100))
