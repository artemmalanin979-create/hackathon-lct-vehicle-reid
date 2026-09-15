"""Контрольные картинки: что именно закрашивается — пластина и контроли равной площади."""
import csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
FONT = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", 14)
sys.path.insert(0, 'work')
from mask_ops import fill, controls
from extract_variants import pad_box
DATA = Path("/home/artem/projects/hackathon-lct-vehicle-reid/data")
SPLIT = Path("/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/split/files")
boxes = json.load(open("work/boxes.json"))
rows = [(r["image_id"], int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))
        for r in csv.DictReader(open(SPLIT / "val_query.csv", newline=""))]

# 1) лист «схема контроля»: на 8 кропах — пластина (красный) и 4 контроля (цветные)
sel = [3, 57, 131, 204, 333, 502, 677, 908]
COLS = {"plate": (255,60,60), "shift": (60,200,255), "mirror": (255,220,60),
        "side": (120,255,120), "rand1": (255,120,255)}
TW, TH = 380, 300
sheet = Image.new("RGB", (TW*4, TH*2), (12,12,12)); D = ImageDraw.Draw(sheet)
for k, i in enumerate(sel):
    iid, x, y, w, h = rows[i]
    with Image.open(DATA/"images"/f"{iid}.jpg") as im:
        c = im.convert("RGB").crop((x, y, x+w, y+h))
    b = boxes["val_query"][i]
    d = ImageDraw.Draw(c); lw = max(2, w//220)
    if b:
        box = tuple(b[:4]); ctl = controls(box, (h, w), 1000+i)
        d.rectangle([box[0], box[1], box[0]+box[2], box[1]+box[3]], outline=COLS["plate"], width=lw+1)
        for nm in ("shift", "mirror", "side", "rand1"):
            cb = ctl[nm]
            d.rectangle([cb[0], cb[1], cb[0]+cb[2], cb[1]+cb[3]], outline=COLS[nm], width=lw)
    s = min(TW/c.width, (TH-18)/c.height)
    c = c.resize((int(c.width*s), int(c.height*s)), Image.BILINEAR)
    ox, oy = (k % 4)*TW, (k//4)*TH
    sheet.paste(c, (ox, oy+18))
    D.text((ox+3, oy+3), f"#{i}  красн=пластина, гол=сдвиг, жёлт=зеркало, зел=вбок, роз=случ.", fill=(255,255,255), font=FONT)
sheet.save("pics/control_scheme.png")

# 2) лист «как выглядит вход сети» для 4 кропов x 5 вариантов
VAR = [("base","без вмешательства"), ("plate_ring","пластина, заливка кольцом"),
       ("shift_ring","контроль-сдвиг, заливка кольцом"),
       ("plate_gray127","пластина, серый 127"), ("plate_desat","пластина, обесцвечена")]
sel2 = [3, 131, 333, 908]
CW = 208
sheet2 = Image.new("RGB", (CW*len(VAR)+10, (CW+22)*len(sel2)+6), (12,12,12))
D2 = ImageDraw.Draw(sheet2)
for r, i in enumerate(sel2):
    iid, x, y, w, h = rows[i]
    with Image.open(DATA/"images"/f"{iid}.jpg") as im:
        a = np.asarray(im.convert("RGB").crop((x, y, x+w, y+h)))
    b = boxes["val_query"][i]
    box = tuple(b[:4]) if b else None
    ctl = controls(box, (h, w), 1000+i) if box else None
    for cc, (v, title) in enumerate(VAR):
        img = a
        if box:
            if v == "plate_ring": img = fill(a, box, "ring")
            elif v == "shift_ring": img = fill(a, ctl["shift"], "ring")
            elif v == "plate_gray127": img = fill(a, box, "gray127")
            elif v == "plate_desat": img = fill(a, box, "desat")
        t = Image.fromarray(img).resize((CW, CW), Image.BILINEAR)
        ox, oy = cc*CW+5, r*(CW+22)+22
        sheet2.paste(t, (ox, oy))
        if r == 0: D2.text((ox+2, 4), title, fill=(255,255,0), font=FONT)
sheet2.save("pics/inputs_variants.png")
print("ok")
