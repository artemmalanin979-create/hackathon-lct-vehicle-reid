#!/usr/bin/env python3
"""Полнокадровая сверка: запрос и его верная пара с нарисованными bbox."""
import csv, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
P = Path("/home/artem/projects/hackathon-lct-vehicle-reid")
D = P/"data/images"; SPLIT = P/"04-solution/split/files"
JOB = Path(__file__).resolve().parent.parent
F = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", 13)


def meta(p):
    rows = list(csv.DictReader(open(p, newline="")))
    for r in rows:
        for k in ("x", "y", "w", "h", "vehicle_id", "camera_id"):
            r[k] = int(r[k])
    return rows


qm, gm = meta(SPLIT/"val_query.csv"), meta(SPLIT/"val_gallery.csv")


def panel(row, cap, W=700):
    im = Image.open(D/f"{row['image_id']}.jpg").convert("RGB")
    ImageDraw.Draw(im).rectangle([row["x"], row["y"], row["x"]+row["w"], row["y"]+row["h"]],
                                 outline=(255, 40, 40), width=6)
    im = im.resize((W, int(W*im.height/im.width)), Image.LANCZOS)
    return im, cap


def grid(panels, out, title, cols=2):
    w, h = panels[0][0].size
    rows = (len(panels)+cols-1)//cols
    sheet = Image.new("RGB", (cols*(w+10)+10, 24+rows*(h+26)+6), (245, 245, 247))
    d = ImageDraw.Draw(sheet)
    d.text((10, 5), title, font=F, fill=(10, 10, 10))
    for n, (im, cap) in enumerate(panels):
        x, y = 10+(n % cols)*(w+10), 24+(n//cols)*(h+26)
        sheet.paste(im, (x, y)); d.text((x, y+h+4), cap, font=F, fill=(10, 10, 10))
    sheet.save(out, quality=88)
    return out


if __name__ == "__main__":
    import json
    m = {x["case"]: x for x in json.load(open(JOB/"out/error_sample_meta.json"))}
    for grp, cases in [("A", [11, 13]), ("B", [15, 48])]:
        ps = []
        for c in cases:
            x = m[c]
            ps.append(panel(qm[x["qi"]], f"кейс #{c}: ЗАПРОС v{x['q_vid']} cam{x['q_cam']}"))
            ps.append(panel(gm[x["mate"]], f"кейс #{c}: «ВЕРНЫЙ ОТВЕТ» v{x['q_vid']} cam{x['mate_cam']}, ранг {x['rank']}"))
        grid(ps, JOB/f"sheets/label_check_{grp}.jpg",
             "Сверка разметки: один ли это автомобиль (bbox красным)")
        print(f"sheets/label_check_{grp}.jpg")
