"""Насколько заливка сдвигает статистики входа сети (первый узел — InstanceNormalization).

Для 300 кропов val сравниваем поканальные mean/std тензора 208x208 до и после
вмешательства. Чем меньше сдвиг, тем меньше вмешательство «видно» нормировке и тем
чище измеряется потеря именно информации, а не сдвиг распределения.
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, 'work')
from mask_ops import fill, controls
DATA = Path("/home/artem/projects/hackathon-lct-vehicle-reid/data")
SPLIT = Path("/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/split/files")
boxes = json.load(open("work/boxes.json"))["val_query"]
rows = [(r["image_id"], int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))
        for r in csv.DictReader(open(SPLIT/"val_query.csv", newline=""))]
rng = np.random.default_rng(7); sel = rng.choice(len(rows), 300, replace=False)

def inp(a):
    return np.asarray(Image.fromarray(a).resize((208,208), Image.BILINEAR), np.float64)

acc = {k: [] for k in ("ring", "gray127", "desat")}
area = []
for i in sel:
    b = boxes[i]
    if not b: continue
    iid, x, y, w, h = rows[i]
    with Image.open(DATA/"images"/f"{iid}.jpg") as im:
        a = np.asarray(im.convert("RGB").crop((x, y, x+w, y+h)))
    box = tuple(b[:4]); area.append(box[2]*box[3]/(w*h))
    t0 = inp(a); m0, s0 = t0.mean((0,1)), t0.std((0,1))
    for mode in acc:
        t = inp(fill(a, box, mode)); m, s = t.mean((0,1)), t.std((0,1))
        acc[mode].append([np.abs(m-m0).mean(), np.abs(s-s0).mean()])
out = {"n": len(area), "доля площади кропа под маской (медиана)": round(float(np.median(area)), 4)}
for mode, v in acc.items():
    v = np.array(v)
    out[mode] = {"сдвиг канального среднего, уровней (медиана)": round(float(np.median(v[:,0])), 3),
                 "сдвиг канального std, уровней (медиана)": round(float(np.median(v[:,1])), 3)}
json.dump(out, open("out/fill_stats.json","w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
