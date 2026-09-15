#!/usr/bin/env python3
"""Полное независимое переизвлечение всех 1860 векторов теста (кроп numpy-срезом)."""
import os
import csv, time
from pathlib import Path
import numpy as np, onnxruntime as ort
from PIL import Image
JOB=Path(__file__).resolve().parent.parent
REPO = Path(__file__).resolve().parents[3]  # корень репозитория
DATA=Path(os.environ.get("REID_DATA_DIR", REPO / "data"))
def rd(p): return [(r["image_id"],int(r["x"]),int(r["y"]),int(r["w"]),int(r["h"]))
                   for r in csv.DictReader(open(p,newline=""))]
rows=rd(DATA/"test_query.csv")+rd(DATA/"test_gallery.csv")
o=ort.SessionOptions(); o.log_severity_level=3
sess=ort.InferenceSession(str(JOB/"subject/osnet_ain_x1_0_vehicle_reid.onnx"),sess_options=o,providers=["CPUExecutionProvider"])
INP=sess.get_inputs()[0].name
out=np.empty((len(rows),512),np.float32)
for s in range(0,len(rows),24):
    bt=[]
    for iid,x,y,w,h in rows[s:s+24]:
        a=np.asarray(Image.open(DATA/"images"/f"{iid}.jpg").convert("RGB"))[y:y+h, x:x+w]
        bt.append(np.asarray(Image.fromarray(a).resize((208,208),Image.BILINEAR),dtype=np.float32).transpose(2,0,1))
    out[s:s+len(bt)]=sess.run(None,{INP:np.stack(bt)})[0]
m=out.astype(np.float64); m/=np.linalg.norm(m,axis=1,keepdims=True)
E=np.load(JOB/"subject/artifacts/embeddings.npy").astype(np.float64); E/=np.linalg.norm(E,axis=1,keepdims=True)
cos=np.sum(m*E,axis=1)
print("min cos=%.9f  max|diff|=%.2e  строк с cos<0.9999: %d"%(cos.min(),np.abs(m-E).max(),int((cos<0.9999).sum())))
rng=np.random.default_rng(5); idx=rng.choice(len(rows),200,replace=False)
print("ближайшая строка = своя:", int(((E@m[idx].T).argmax(0)==idx).sum()), "/200")
