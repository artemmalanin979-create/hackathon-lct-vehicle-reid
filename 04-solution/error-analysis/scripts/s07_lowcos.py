#!/usr/bin/env python3
"""Все запросы с cos(лучшая верная пара) < 0.20: сверка «одна ли это машина».

Кроп берётся с запасом 35% вокруг bbox, сам bbox обведён — видно, не стоит ли
рамка на соседней машине.
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sheets_lib import build_sheet, P

JOB = Path(__file__).resolve().parent.parent
SPLIT = P/"04-solution/split/files"; IMG = P/"data/images"


def meta(p):
    rows = list(csv.DictReader(open(p, newline="")))
    for r in rows:
        for k in ("x", "y", "w", "h", "vehicle_id", "camera_id"):
            r[k] = int(r[k])
    return rows


qm, gm = meta(SPLIT/"val_query.csv"), meta(SPLIT/"val_gallery.csv")


def boxed(row, ctx=0.35):
    x, y, w, h = row["x"], row["y"], row["w"], row["h"]
    with Image.open(IMG/f"{row['image_id']}.jpg") as im:
        im = im.convert("RGB"); W, H = im.size
        dx, dy = int(w*ctx), int(h*ctx)
        x0, y0 = max(0, x-dx), max(0, y-dy)
        x1, y1 = min(W, x+w+dx), min(H, y+h+dy)
        c = im.crop((x0, y0, x1, y1))
    ImageDraw.Draw(c).rectangle([x-x0, y-y0, x-x0+w, y-y0+h], outline=(255, 40, 40), width=5)
    return c


pb = np.load(JOB/'out/perquery_base.npz', allow_pickle=True)
fq = np.load(JOB/'out/feats_query.npz', allow_pickle=True)
fg = np.load(JOB/'out/feats_gallery.npz', allow_pickle=True)
S = pb['score_matrix']; idx = np.flatnonzero(pb['status'] == 'known')
qv, gv = fq['vehicle_id'], fg['vehicle_id']; qc, gc = fq['camera_id'], fg['camera_id']
sel = []
for i in idx:
    m = np.flatnonzero((gv == qv[i]) & (gc != qc[i]))
    j = int(m[np.argmax(S[i, m])])
    if S[i, j] < 0.20:
        sel.append((int(i), j, float(S[i, j])))
sel.sort(key=lambda t: t[2])
print("запросов с cos лучшей пары < 0.20:", len(sel))
rows = [[(boxed(qm[i]), f"#{n} ЗАПРОС v{qv[i]} cam{qc[i]}"),
         (boxed(gm[j]), f"его пара cam{gc[j]}, cos {c:.3f}\nранг верного {int(pb['first_rank'][i])}")]
        for n, (i, j, c) in enumerate(sel, 1)]
for s in range(0, len(rows), 9):
    p = JOB/f"sheets/lowcos_{s//9+1:02d}.jpg"
    build_sheet(rows[s:s+9], ["ЗАПРОС (bbox обведён, кроп с запасом)", "«ВЕРНАЯ ПАРА» ИЗ ГАЛЕРЕИ"],
                f"Все {len(sel)} запросов с cos к своей паре < 0.20 — сверка разметки (лист {s//9+1})",
                p, border_by=[(40, 90, 200), (20, 140, 60)], cw=330, ch=240)
    print(p.name)
(JOB/'out/lowcos_cases.json').write_text(json.dumps(
    [{"n": n, "qi": i, "mate": j, "cos": c, "q_vid": int(qv[i]), "q_cam": int(qc[i]),
      "mate_cam": int(gc[j]), "rank": int(pb['first_rank'][i]),
      "q_img": qm[i]["image_id"], "g_img": gm[j]["image_id"]}
     for n, (i, j, c) in enumerate(sel, 1)], indent=2, ensure_ascii=False)+"\n")
