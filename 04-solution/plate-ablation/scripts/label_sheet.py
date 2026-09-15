"""Листы для ручной разметки: кроп в исходном разрешении с координатной сеткой."""
import csv, json, random, sys
from pathlib import Path
from PIL import Image, ImageDraw
DATA = Path("/home/artem/projects/hackathon-lct-vehicle-reid/data")
SPLIT = Path("/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/split/files")
OUT = Path("pics/label"); OUT.mkdir(parents=True, exist_ok=True)

def rd(p): return list(csv.DictReader(open(p, newline="")))
rows = [dict(r, src="q") for r in rd(SPLIT / "val_query.csv")] + \
       [dict(r, src="g") for r in rd(SPLIT / "val_gallery.csv")]
random.seed(20260915)
sample = random.sample(rows, 36)
meta = []
MAXW, MAXH = 1020, 800
for i, r in enumerate(sample):
    x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
    with Image.open(DATA / "images" / f"{r['image_id']}.jpg") as im:
        crop = im.convert("RGB").crop((x, y, x + w, y + h))
    s = min(MAXW / w, MAXH / h, 2.5)
    vis = crop.resize((int(w * s), int(h * s)), Image.LANCZOS).convert("RGB")
    d = ImageDraw.Draw(vis)
    step = 20 if max(w, h) <= 700 else 50
    big = step * 5
    for gx in range(0, w + 1, step):
        X = gx * s
        d.line([X, 0, X, vis.height], fill=(255, 0, 0) if gx % big == 0 else (0, 255, 0), width=1)
    for gy in range(0, h + 1, step):
        Y = gy * s
        d.line([0, Y, vis.width, Y], fill=(255, 0, 0) if gy % big == 0 else (0, 255, 0), width=1)
    for gx in range(0, w + 1, big):
        d.text((gx * s + 2, 2), str(gx), fill=(255, 255, 0))
    for gy in range(big, h + 1, big):
        d.text((2, gy * s + 2), str(gy), fill=(255, 255, 0))
    d.text((vis.width - 150, 2), f"#{i} {w}x{h} шаг{step}", fill=(255, 255, 0))
    vis.save(OUT / f"lab_{i:02d}.png")
    meta.append(dict(idx=i, image_id=r["image_id"], x=x, y=y, w=w, h=h, src=r["src"]))
json.dump(meta, open("work/label_meta.json", "w"), indent=1)
print("готово", len(meta))
