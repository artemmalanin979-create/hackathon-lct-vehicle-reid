"""Проявочный лист: кропы из TRAIN (dev-выборка, отдельно от val) с боксами детектора."""
import os
import csv, random, sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plate_detect import detect
REPO = Path(__file__).resolve().parents[3]  # корень репозитория
DATA = Path(os.environ.get("REID_DATA_DIR", REPO / "data"))
rows = list(csv.DictReader(open(DATA / "train.csv", newline="")))
random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
sample = random.sample(rows, 12)
TW, TH = 300, 260
sheet = Image.new("RGB", (TW * 4, TH * 3), (20, 20, 20))
d0 = ImageDraw.Draw(sheet)
t0 = time.perf_counter()
for i, r in enumerate(sample):
    x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
    with Image.open(DATA / "images" / f"{r['image_id']}.jpg") as im:
        crop = im.convert("RGB").crop((x, y, x + w, y + h))
    arr = np.asarray(crop)
    t1 = time.perf_counter()
    boxes = detect(arr, topn=2)
    dt = time.perf_counter() - t1
    vis = crop.copy(); dr = ImageDraw.Draw(vis)
    for j, (sc, bx, by, bw, bh, f) in enumerate(boxes):
        col = (255, 0, 0) if j == 0 else (255, 200, 0)
        dr.rectangle([bx, by, bx + bw, by + bh], outline=col, width=max(2, w // 200))
    s = min(TW / vis.width, (TH - 18) / vis.height)
    vis = vis.resize((max(1, int(vis.width * s)), max(1, int(vis.height * s))), Image.BILINEAR)
    ox, oy = (i % 4) * TW, (i // 4) * TH
    sheet.paste(vis, (ox, oy + 18))
    lab = f"#{i} {dt*1000:.0f}ms"
    if boxes:
        f = boxes[0][5]
        lab += f" s={boxes[0][0]:.1f} d={f[chr(39)+chr(39)] if False else f['dens']:.2f} nx={f['nx']} ny={f['ny']} {f['w']}x{f['h']}"
    d0.text((ox + 4, oy + 4), lab, fill=(255, 255, 255))
sheet.save(sys.argv[2] if len(sys.argv) > 2 else "pics/dev_sheet.png")
print(f"total {time.perf_counter()-t0:.1f}s")
