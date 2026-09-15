"""Перевод ручной разметки из координат листа в координаты кропа + лист проверки."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
FONT = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", 14)
DATA = Path("/home/artem/projects/hackathon-lct-vehicle-reid/data")
meta = json.load(open("work/label_meta.json"))
disp = json.load(open("work/labels_display.json"))
out = []
for m in meta:
    w, h = m["w"], m["h"]
    s = min(1020 / w, 800 / h, 2.5)
    d = disp[str(m["idx"])]
    box = None if d is None else [round(d[0]/s), round(d[1]/s), round(d[2]/s), round(d[3]/s)]
    out.append(dict(m, plate=box, scale=s))
json.dump(out, open("work/labels.json", "w"), indent=1)
# лист проверки: 12 кропов на лист, бокс нарисован
per, cols = 12, 4
TW, TH = 330, 270
for sheet in range(3):
    img = Image.new("RGB", (TW*cols, TH*3), (15,15,15)); dd = ImageDraw.Draw(img)
    for k in range(per):
        i = sheet*per + k
        if i >= len(out): break
        r = out[i]
        with Image.open(DATA/"images"/f"{r['image_id']}.jpg") as im:
            c = im.convert("RGB").crop((r["x"], r["y"], r["x"]+r["w"], r["y"]+r["h"]))
        dr = ImageDraw.Draw(c)
        if r["plate"]:
            x,y,bw,bh = r["plate"]
            dr.rectangle([x,y,x+bw,y+bh], outline=(255,0,0), width=max(2, r["w"]//250))
        sc = min(TW/c.width, (TH-16)/c.height)
        c = c.resize((int(c.width*sc), int(c.height*sc)), Image.BILINEAR)
        ox, oy = (k%cols)*TW, (k//cols)*TH
        img.paste(c, (ox, oy+16))
        dd.text((ox+3, oy+3), f"#{i} {'нет' if not r['plate'] else str(r['plate'])}", fill=(255,255,0), font=FONT)
    img.save(f"pics/verify_labels_{sheet}.png")
print("ok", sum(1 for r in out if r["plate"]), "из", len(out))
