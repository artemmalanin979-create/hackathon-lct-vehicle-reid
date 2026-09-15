#!/usr/bin/env python3
"""s07a: контактные листы для визуальной проверки (в пределах лимита <=100 файлов):
 - sheet_plates_*.png: 48 кропов ТС (train/query/gallery + тёмные + крупные) — проверка
   размытия номерных знаков;
 - sheet_pass_low.png / sheet_pass_high.png: тройки кадров одного (vehicle_id,camera_id)
   с низким / высоким полнокадровым сходством — что такое «один проезд» на деле;
 - sheet_firstpair.png: первые строки test_query и test_gallery (кадры целиком + кропы).
Память: по одному файлу. Запуск: python3 s07a_sheets.py"""
import csv, json, os
import numpy as np
from PIL import Image, ImageDraw

DATA = "/home/artem/projects/hackathon-lct-vehicle-reid/data"
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)

files = json.load(open(os.path.join(HERE, "scan_files.json")))
idx = {os.path.splitext(f)[0]: i for i, f in enumerate(files)}
N = len(files)
split = np.full(N, -1, np.int8); vid = np.full(N, -1, np.int32); cam = np.full(N, -1, np.int32)
bbox = np.zeros((N, 4), np.int32)
for fn, sname in [("train.csv", 0), ("test_query.csv", 1), ("test_gallery.csv", 2)]:
    for r in csv.DictReader(open(os.path.join(DATA, fn))):
        i = idx[r["image_id"]]
        split[i] = sname
        bbox[i] = [int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])]
        if sname == 0:
            vid[i] = int(r["vehicle_id"]); cam[i] = int(r["camera_id"])
SPL = ["tr", "q", "g"]
T = np.load(os.path.join(HERE, "thumbs32.npy")).astype(np.float32)
bright = T.mean(1)

def load(i):
    with Image.open(os.path.join(DATA, "images", files[i])) as im:
        return im.convert("RGB").copy()

def crop_of(i, maxw=320, maxh=240):
    im = load(i)
    x, y, w, h = bbox[i]
    c = im.crop((x, y, x+w, y+h))
    s = min(maxw/c.width, maxh/c.height)
    return c.resize((max(1, int(c.width*s)), max(1, int(c.height*s))), Image.BILINEAR)

# --- выбор 48 кропов ---
pick = []
for s, n in [(0, 16), (1, 8), (2, 8)]:
    pick += list(rng.choice(np.where(split == s)[0], n, replace=False))
area = bbox[:, 2].astype(np.int64) * bbox[:, 3]
pick += list(np.argsort(bright)[:8])            # самые тёмные кадры
pick += list(np.argsort(-area)[:8])             # самые крупные объекты
pick = list(dict.fromkeys(int(i) for i in pick))[:48]
json.dump([files[i] for i in pick], open(os.path.join(HERE, "sheet_plates_picks.json"), "w"))

CW, CH, PAD = 340, 265, 6
for sh in range(4):
    grid = Image.new("RGB", (4*CW, 3*CH), (28, 28, 28))
    d = ImageDraw.Draw(grid)
    for k, i in enumerate(pick[sh*12:(sh+1)*12]):
        c = crop_of(i)
        cx, cy = (k % 4)*CW+PAD, (k//4)*CH+PAD
        grid.paste(c, (cx, cy))
        d.text((cx+2, cy+CH-22), f"{k+sh*12}:{SPL[split[i]]} {files[i][:8]} {bbox[i][2]}x{bbox[i][3]} b{int(bright[i])}",
               fill=(255, 220, 0))
    grid.save(os.path.join(HERE, f"sheet_plates_{sh}.png"))

# --- тройки одного (vid,cam): низкое vs высокое сходство ---
V = T - T.mean(1, keepdims=True)
V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
groups = {}
for i in np.where(split == 0)[0]:
    groups.setdefault((int(vid[i]), int(cam[i])), []).append(i)
trip = {k: g for k, g in groups.items() if len(g) == 3}
scored = []
for k, g in trip.items():
    s = min(float(V[a] @ V[b]) for ai, a in enumerate(g) for b in g[ai+1:])
    scored.append((s, k, g))
scored.sort()
sel = {"low": scored[:4], "high": scored[-4:]}
for name, rows in sel.items():
    grid = Image.new("RGB", (3*406, 4*234), (28, 28, 28))
    d = ImageDraw.Draw(grid)
    for r, (s, key, g) in enumerate(rows):
        for c, i in enumerate(sorted(g)):
            im = load(i)
            dd = ImageDraw.Draw(im)
            x, y, w, h = bbox[i]
            dd.rectangle([x, y, x+w, y+h], outline=(255, 40, 40), width=6)
            im = im.resize((400, 225), Image.BILINEAR)
            grid.paste(im, (c*406+3, r*234+3))
        d.text((6, r*234+6), f"vid{key[0]} cam{key[1]} min_cos={s:.2f}", fill=(255, 220, 0))
    grid.save(os.path.join(HERE, f"sheet_pass_{name}.png"))
json.dump({n: [{"vid": k[0], "cam": k[1], "min_cos": round(s, 3), "files": [files[i] for i in g]}
               for s, k, g in rows] for n, rows in sel.items()},
          open(os.path.join(HERE, "sheet_pass_picks.json"), "w"), indent=1)

# --- пара первых строк ---
q0 = idx["486dd80d22f94a5496b3e670b19399a7"]; g0 = idx["df7ea50171af48df81584ab7e8107d15"]
grid = Image.new("RGB", (2*646, 363+330), (28, 28, 28))
d = ImageDraw.Draw(grid)
for c, i in enumerate([q0, g0]):
    im = load(i)
    dd = ImageDraw.Draw(im)
    x, y, w, h = bbox[i]
    dd.rectangle([x, y, x+w, y+h], outline=(255, 40, 40), width=6)
    grid.paste(im.resize((640, 360), Image.BILINEAR), (c*646+3, 3))
    cr = crop_of(i, 420, 320)
    grid.paste(cr, (c*646+3, 366))
    d.text((c*646+6, 340), f"{SPL[split[i]]} {files[i]}", fill=(255, 220, 0))
grid.save(os.path.join(HERE, "sheet_firstpair.png"))
print("sheets done")
