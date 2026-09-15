#!/usr/bin/env python3
"""Свой пересчёт порога/F1/TNR + перенос порога val->test."""
import os
import csv, json, sys
from pathlib import Path
import numpy as np
JOB = Path(__file__).resolve().parent.parent
REPO = Path(__file__).resolve().parents[3]  # корень репозитория
DATA = Path(os.environ.get("REID_DATA_DIR", REPO / "data"))
SPLIT = REPO / "04-solution/split/files"
def rd(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))
qm, gm = rd(SPLIT/"val_query.csv"), rd(SPLIT/"val_gallery.csv")
qv=np.array([r["vehicle_id"] for r in qm]); gv=np.array([r["vehicle_id"] for r in gm])
qc=np.array([r["camera_id"] for r in qm]); gc=np.array([r["camera_id"] for r in gm])
mate=np.array([r["has_mate"]=="1" for r in qm])
def l2(x): return x/np.linalg.norm(x,axis=1,keepdims=True)
q=l2(np.load(JOB/"subject/out/val_query.npy").astype(np.float64))
g=l2(np.load(JOB/"subject/out/val_gallery.npy").astype(np.float64))
S=q@g.T
# top-1 с исключением (как при выборе порога) и без (как на тесте)
top1_f=np.empty(len(qm)); top1_u=S.max(1)
for i in range(len(qm)):
    keep=~((gv==qv[i])&(gc==qc[i])); top1_f[i]=S[i,keep].max()
T=0.34921352213815304
def prf(s, pos, t):
    acc=s>=t; tp=int((acc&pos).sum()); fp=int((acc&~pos).sum()); fn=int(pos.sum())-tp
    tn=int((~acc&~pos).sum())
    return dict(F1=2*tp/(2*tp+fp+fn), TNR=tn/int((~pos).sum()), prec=tp/(tp+fp),
                rec=tp/(tp+fn), refused=int((~acc).sum()))
# свой argmax F1 по всем кандидатам-порогам
cands=np.unique(top1_f)
f1s=np.array([prf(top1_f,mate,t)["F1"] for t in cands])
best=cands[int(np.argmax(f1s))]
print("МОЙ argmax F1 порог = %.17g  F1=%.17g" % (best, f1s.max()))
print("  их t* = %.17g  (совпадает: %s)" % (T, np.isclose(best,T,rtol=0,atol=0)))
print("  при их t*:", json.dumps(prf(top1_f,mate,T)))
print("\nval, БЕЗ исключения камеры (режим теста), тот же t*:", json.dumps(prf(top1_u,mate,T)))
print("  доля отказов val(с искл.) = %.4f, val(без искл.) = %.4f"
      % (prf(top1_f,mate,T)["refused"]/len(qm), prf(top1_u,mate,T)["refused"]/len(qm)))
# тест
E=np.load(JOB/"subject/artifacts/embeddings.npy").astype(np.float64)
tq=l2(E[:1110]); tg=l2(E[1110:])
t1=(tq@tg.T).max(1)
print("  доля отказов на ТЕСТЕ  = %.4f (%d из 1110)" % ((t1<T).mean(), int((t1<T).sum())))
def q_(a): return " ".join("%s=%.3f"%(p,np.percentile(a,p)) for p in (1,5,10,25,50,75,95))
print("\nраспределение top-1 cos:")
print("  val без искл. :", q_(top1_u))
print("  val с искл.   :", q_(top1_f))
print("  test          :", q_(t1))
print("  val без искл., подмножество has_mate=0 :", q_(top1_u[~mate]))
print("  val без искл., подмножество has_mate=1 :", q_(top1_u[mate]))
# что было бы при оптимальном пороге на НЕфильтрованных событиях
cands=np.unique(top1_u); f1s=np.array([prf(top1_u,mate,t)["F1"] for t in cands])
b2=cands[int(np.argmax(f1s))]
print("\nargmax F1 по НЕфильтрованным top-1 (протокол теста): t=%.6f F1=%.4f" % (b2, f1s.max()),
      json.dumps(prf(top1_u,mate,b2)))
# порог под максимум сбалансированной точности / под TNR>=0.5
for target in (0.5, 0.8):
    t = np.quantile(top1_u[~mate], target)
    print("  порог для TNR=%.1f на val(без искл.): t=%.4f -> %s" % (target, t, json.dumps(prf(top1_u,mate,t))))
