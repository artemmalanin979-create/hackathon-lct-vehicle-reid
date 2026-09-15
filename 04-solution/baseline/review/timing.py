#!/usr/bin/env python3
"""Повтор замера батч=1 + разложение по стадиям + холодный/тёплый кеш страниц."""
import csv, os, sys, time, json
from pathlib import Path
import numpy as np, onnxruntime as ort
from PIL import Image
JOB=Path(__file__).resolve().parent.parent
DATA=Path("/home/artem/projects/hackathon-lct-vehicle-reid/data")
SPLIT=Path("/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/split/files")
rows=[(r["image_id"],int(r["x"]),int(r["y"]),int(r["w"]),int(r["h"]))
      for r in csv.DictReader(open(SPLIT/"val_query.csv",newline=""))][:200]
o=ort.SessionOptions(); o.log_severity_level=3
sess=ort.InferenceSession(str(JOB/"subject/osnet_ain_x1_0_vehicle_reid.onnx"),sess_options=o,providers=["CPUExecutionProvider"])
INP=sess.get_inputs()[0].name
print("intra_op_num_threads по умолчанию:", o.intra_op_num_threads, "| CPU:", os.cpu_count(),
      "| loadavg:", os.getloadavg())
def crop(r):
    iid,x,y,w,h=r
    with Image.open(DATA/"images"/f"{iid}.jpg") as im:
        c=im.convert("RGB").crop((x,y,x+w,y+h))
    return np.asarray(c.resize((208,208),Image.BILINEAR),dtype=np.float32).transpose(2,0,1)
def evict():
    for r in rows:
        fd=os.open(DATA/"images"/f"{r[0]}.jpg", os.O_RDONLY)
        try: os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
        finally: os.close(fd)
def full(rs):
    t=time.perf_counter()
    for r in rs: sess.run(None,{INP:crop(r)[None]})
    return (time.perf_counter()-t)/len(rs)*1000
# прогрев сессии
for r in rows[:5]: sess.run(None,{INP:crop(r)[None]})
print("\n-- батч 1, полный конвейер (мс/объект) --")
print("  тёплый кеш, прогон 1: %.1f" % full(rows))
print("  тёплый кеш, прогон 2: %.1f" % full(rows))
evict(); print("  ХОЛОДНЫЙ кеш:       %.1f" % full(rows))
evict(); print("  ХОЛОДНЫЙ кеш ещё раз: %.1f" % full(rows))
print("\n-- разложение (тёплый кеш) --")
t=time.perf_counter()
for r in rows:
    with Image.open(DATA/"images"/f"{r[0]}.jpg") as im: im.convert("RGB")
dec=(time.perf_counter()-t)/len(rows)*1000
t=time.perf_counter(); arrs=[crop(r) for r in rows]; pre=(time.perf_counter()-t)/len(rows)*1000
t=time.perf_counter()
for a in arrs: sess.run(None,{INP:a[None]})
net=(time.perf_counter()-t)/len(rows)*1000
print("  только декод кадра 1920x1080: %.1f | декод+кроп+resize: %.1f | только сеть (батч1): %.1f"%(dec,pre,net))
t=time.perf_counter()
for s in range(0,200,32): sess.run(None,{INP:np.stack(arrs[s:s+32])})
print("  только сеть, батч 32: %.1f мс/объект"%((time.perf_counter()-t)/200*1000))
