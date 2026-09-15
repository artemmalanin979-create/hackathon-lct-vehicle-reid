#!/usr/bin/env python3
"""Честность применения сплита: те ли поля, те ли идентичности, тот ли порядок."""
import csv
from pathlib import Path
DATA=Path("/home/artem/projects/hackathon-lct-vehicle-reid/data")
SPLIT=Path("/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/split/files")
JOB=Path(__file__).resolve().parent.parent
def rd(p): return list(csv.DictReader(open(p,newline="")))
tr={r["image_id"]:r for r in rd(DATA/"train.csv")}
q,g,fit=rd(SPLIT/"val_query.csv"),rd(SPLIT/"val_gallery.csv"),rd(SPLIT/"train_fit.csv")
for rows,name in [(q,"val_query"),(g,"val_gallery"),(fit,"train_fit")]:
    bad=[r["image_id"] for r in rows if r["image_id"] not in tr or any(
        tr[r["image_id"]][k]!=r[k] for k in ("x","y","w","h","vehicle_id","camera_id"))]
    print(f"{name}: расхождений с train.csv = {len(bad)}")
qi={r['image_id'] for r in q}; gi={r['image_id'] for r in g}; fi={r['image_id'] for r in fit}
qv={r['vehicle_id'] for r in q}; gv={r['vehicle_id'] for r in g}; fv={r['vehicle_id'] for r in fit}
print("q∩g",len(qi&gi),"q∩fit",len(qi&fi),"g∩fit",len(gi&fi),"| (qv∪gv)∩fv",len((qv|gv)&fv))
print("has_mate противоречий:",sum(1 for r in q if (r['vehicle_id'] in gv)!=(r['has_mate']=='1')))
gby={}
for r in g: gby.setdefault(r['vehicle_id'],[]).append(r['camera_id'])
print("has_mate=1 без кросс-камерного mate:",sum(1 for r in q if r['has_mate']=='1' and all(c==r['camera_id'] for c in gby[r['vehicle_id']])))
for nm,p in [("val_query",SPLIT/"val_query.csv"),("val_gallery",SPLIT/"val_gallery.csv"),
             ("test_query",DATA/"test_query.csv"),("test_gallery",DATA/"test_gallery.csv")]:
    print(f"out/{nm}.ids == порядок {p.name}:",
          (JOB/f"subject/out/{nm}.ids").read_text().split()==[r["image_id"] for r in rd(p)])
