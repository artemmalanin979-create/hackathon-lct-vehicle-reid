#!/usr/bin/env python3
"""Контрольные картинки: (1) bbox на кадре, (2) кроп как его видит сеть + линия 70%,
(3) кроп после --mask-bottom 0.3."""
import os
import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
JOB=Path(__file__).resolve().parent.parent
REPO = Path(__file__).resolve().parents[3]  # корень репозитория
DATA=Path(os.environ.get("REID_DATA_DIR", REPO / "data"))
SPLIT=REPO / "04-solution/split/files"
rows=list(csv.DictReader(open(SPLIT/"val_query.csv",newline="")))
rng=np.random.default_rng(7)
pick=[0]+sorted(rng.choice(len(rows),11,replace=False).tolist())
# 1) кадры с bbox
W,H=480,270
sheet=Image.new("RGB",(W*4,H*3),(20,20,20)); d=ImageDraw.Draw(sheet)
for n,i in enumerate(pick[:12]):
    r=rows[i]; x,y,w,h=int(r["x"]),int(r["y"]),int(r["w"]),int(r["h"])
    im=Image.open(DATA/"images"/f"{r['image_id']}.jpg").convert("RGB")
    sx,sy=W/im.width,H/im.height
    im=im.resize((W,H),Image.BILINEAR)
    dd=ImageDraw.Draw(im); dd.rectangle([x*sx,y*sy,(x+w)*sx,(y+h)*sy],outline=(255,0,0),width=3)
    dd.text((4,4),f"{r['image_id']} {im.width}x{im.height} bbox {x},{y},{w},{h} cam={r['camera_id']}",fill=(255,255,0))
    sheet.paste(im,((n%4)*W,(n//4)*H))
sheet.save(JOB/"work/pics/frames_bbox.png")
# 2/3) кропы 208 + линия 70% и mask30
def crop_of(r, mask=False):
    x,y,w,h=int(r["x"]),int(r["y"]),int(r["w"]),int(r["h"])
    with Image.open(DATA/"images"/f"{r['image_id']}.jpg") as im:
        c=im.convert("RGB").crop((x,y,x+w,y+h))
    if mask:
        a=np.asarray(c).copy(); b=int(round(a.shape[0]*0.3)); a[-b:,:,:]=127; c=Image.fromarray(a)
    return c.resize((208,208),Image.BILINEAR)
for tag,mask in [("crops208",False),("crops208_mask30",True)]:
    S=208; sheet=Image.new("RGB",(S*6,S*4),(20,20,20))
    for n,i in enumerate(pick[:24] if len(pick)>=24 else (pick*3)[:24]):
        c=crop_of(rows[i],mask); dd=ImageDraw.Draw(c)
        if not mask: dd.line([(0,int(S*0.7)),(S,int(S*0.7))],fill=(255,0,0),width=2)
        dd.text((3,3),rows[i]["image_id"][:14],fill=(255,255,0))
        sheet.paste(c,((n%6)*S,(n//6)*S))
    sheet.save(JOB/f"work/pics/{tag}.png")
print("готово", [p.name for p in sorted((JOB/'work/pics').iterdir())])
