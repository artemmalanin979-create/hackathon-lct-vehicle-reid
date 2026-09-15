#!/usr/bin/env python3
"""Контроль абляции: закрываем серым 30% полосу СВЕРХУ и ПО СЕРЕДИНЕ — той же площади,
тем же цветом 127, всё остальное идентично subject/scripts/extract_embeddings.py."""
import csv, time
from pathlib import Path
import numpy as np, onnxruntime as ort
from PIL import Image
JOB=Path(__file__).resolve().parent.parent
DATA=Path("/home/artem/projects/hackathon-lct-vehicle-reid/data")
SPLIT=Path("/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/split/files")
OUT=JOB/"work"/"emb"
o=ort.SessionOptions(); o.log_severity_level=3
sess=ort.InferenceSession(str(JOB/"subject/osnet_ain_x1_0_vehicle_reid.onnx"),sess_options=o,providers=["CPUExecutionProvider"])
INP=sess.get_inputs()[0].name
def rd(p):
    return [(r["image_id"],int(r["x"]),int(r["y"]),int(r["w"]),int(r["h"]))
            for r in csv.DictReader(open(p,newline=""))]
VAR=["btm30","top30","mid30"]
def bands(crop):
    a=np.asarray(crop); H=a.shape[0]; b=int(round(H*0.3))
    out={}
    for v in VAR:
        z=a.copy()
        if v=="btm30": z[-b:]=127
        elif v=="top30": z[:b]=127
        else:
            s=(H-b)//2; z[s:s+b]=127
        out[v]=np.asarray(Image.fromarray(z).resize((208,208),Image.BILINEAR),dtype=np.float32).transpose(2,0,1)
    return out
def run(name, rows, batch=24):
    acc={v:np.empty((len(rows),512),np.float32) for v in VAR}
    t0=time.perf_counter()
    for s in range(0,len(rows),batch):
        ch=rows[s:s+batch]; per={v:[] for v in VAR}
        for iid,x,y,w,h in ch:
            with Image.open(DATA/"images"/f"{iid}.jpg") as im:
                c=im.convert("RGB").crop((x,y,x+w,y+h))
            for v,a in bands(c).items(): per[v].append(a)
        for v in VAR: acc[v][s:s+len(ch)]=sess.run(None,{INP:np.stack(per[v])})[0]
    for v in VAR:
        m=acc[v].astype(np.float64); m/=np.linalg.norm(m,axis=1,keepdims=True)
        np.save(OUT/f"{name}_{v}.npy",m.astype(np.float32))
    print(f"{name} за {time.perf_counter()-t0:.0f}s",flush=True)
run("val_query", rd(SPLIT/"val_query.csv")); run("val_gallery", rd(SPLIT/"val_gallery.csv"))
# контроль: btm30 должен совпасть с их val_*_mask30.npy
for nm,ref in [("val_query","val_query_mask30.npy"),("val_gallery","val_gallery_mask30.npy")]:
    a=np.load(OUT/f"{nm}_btm30.npy").astype(np.float64); b=np.load(JOB/"subject/out"/ref).astype(np.float64)
    print(f"{nm}: btm30 vs их {ref}: min cos={np.sum(a*b,axis=1).min():.8f} max|diff|={np.abs(a-b).max():.1e}")
