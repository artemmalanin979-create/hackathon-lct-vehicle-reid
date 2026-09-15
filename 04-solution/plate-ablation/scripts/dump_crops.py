import csv, sys, random
from pathlib import Path
from PIL import Image
DATA=Path("/home/artem/projects/hackathon-lct-vehicle-reid/data/images")
SPLIT=Path("/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/split/files")
def rd(p): return list(csv.DictReader(open(p,newline="")))
rows=[(r,"q") for r in rd(SPLIT/"val_query.csv")]+[(r,"g") for r in rd(SPLIT/"val_gallery.csv")]
print(len(rows))
import numpy as np
ws=[int(r["w"]) for r,_ in rows]; hs=[int(r["h"]) for r,_ in rows]
print("w: min",min(ws),"med",int(np.median(ws)),"max",max(ws))
print("h: min",min(hs),"med",int(np.median(hs)),"max",max(hs))
random.seed(0)
for i,(r,tag) in enumerate(random.sample(rows,4)):
    with Image.open(DATA/f"{r['image_id']}.jpg") as im:
        c=im.convert("RGB").crop((int(r["x"]),int(r["y"]),int(r["x"])+int(r["w"]),int(r["y"])+int(r["h"])))
    c.save(f"pics/peek_{i}.png")
    print(i,tag,r["image_id"],c.size)
