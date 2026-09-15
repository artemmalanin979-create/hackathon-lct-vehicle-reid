#!/usr/bin/env python3
"""s10: проверка обещания ТЗ 5.4 (идентичности train и test не пересекаются) на
единственно доступном материале: near-dup пары train<->test (ff cos32>=0.90) и
парковочные пары (IoU>=0.8). Если в такой паре размечена ТА ЖЕ машина (crop64>=0.8)
=> машина теста присутствует в train => нарушение обещания.
Выход: out_s10.json + sheet_traintest.png. Запуск: python3 s10_train_test_overlap.py"""
import csv, json, os
import numpy as np
from PIL import Image, ImageDraw

DATA = "/home/artem/projects/hackathon-lct-vehicle-reid/data"
HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1920, 1080

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

T = np.load(os.path.join(HERE, "thumbs32.npy")).astype(np.float32)
V = T - T.mean(1, keepdims=True)
V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)

def iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2 = min(b1[0]+b1[2], b2[0]+b2[2]); y2 = min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0, x2-x1)*max(0, y2-y1)
    return inter/(b1[2]*b1[3]+b2[2]*b2[3]-inter)

tr = np.where(split == 0)[0]; te = np.where(split != 0)[0]
S = V[tr] @ V[te].T
IO = np.zeros_like(S)
pairs = []
r, c = np.nonzero(S >= 0.90)
for a, b in zip(r, c):
    pairs.append((int(tr[a]), int(te[b]), float(S[a, b]), "A_moment"))
# парковочные: IoU>=0.8 при ff<0.90
bbt = bbox[tr].astype(np.float32); bbe = bbox[te].astype(np.float32)
x1 = np.maximum(bbt[:, 0:1], bbe[None, :, 0].squeeze(0)) if False else None
ax1, ay1 = bbt[:, 0:1], bbt[:, 1:2]; ax2, ay2 = ax1+bbt[:, 2:3], ay1+bbt[:, 3:4]
bx1, by1, bx2, by2 = bbe[:, 0], bbe[:, 1], bbe[:, 0]+bbe[:, 2], bbe[:, 1]+bbe[:, 3]
ix = np.maximum(0, np.minimum(ax2, bx2[None, :])-np.maximum(ax1, bx1[None, :]))
iy = np.maximum(0, np.minimum(ay2, by2[None, :])-np.maximum(ay1, by1[None, :]))
inter = ix*iy
IOU = inter/(bbt[:, 2:3]*bbt[:, 3:4]+(bbe[:, 2]*bbe[:, 3])[None, :]-inter)
r, c = np.nonzero((IOU >= 0.8) & (S < 0.90))
for a, b in zip(r, c):
    pairs.append((int(tr[a]), int(te[b]), float(S[a, b]), "B_parked"))

cache = {}
def crop64(i):
    if i not in cache:
        if len(cache) > 600: cache.clear()
        with Image.open(os.path.join(DATA, "images", files[i])) as im:
            im.draft("L", (960, 540))
            g = im.convert("L")
            sx, sy = g.width/W, g.height/H
            x, y, w, h = bbox[i]
            cc = g.crop((int(x*sx), int(y*sy), int(np.ceil((x+w)*sx)), int(np.ceil((y+h)*sy))))
            v = np.asarray(cc.resize((64, 64), Image.BILINEAR), np.float32).ravel()
            v -= v.mean(); v /= (np.linalg.norm(v)+1e-9)
            cache[i] = v
    return cache[i]

res = []
for a, b, s, kind in pairs:
    cc = float(crop64(a) @ crop64(b))
    res.append({"train_file": files[a], "train_vid": int(vid[a]), "train_cam": int(cam[a]),
                "test_file": files[b], "test_split": ["", "query", "gallery"][split[b]],
                "ff": round(s, 3), "iou": round(iou(bbox[a], bbox[b]), 3),
                "crop": round(cc, 3), "kind": kind})
hits = [x for x in res if x["crop"] >= 0.8]
out = {"n_pairs_checked": len(res),
       "n_A_moment": sum(1 for x in res if x["kind"] == "A_moment"),
       "n_B_parked": sum(1 for x in res if x["kind"] == "B_parked"),
       "n_same_vehicle_like": len(hits),
       "n_test_frames_affected": len(set(x["test_file"] for x in hits)),
       "affected_train_vids": sorted(set(x["train_vid"] for x in hits)),
       "hits": sorted(hits, key=lambda x: -x["crop"])}
with open(os.path.join(HERE, "out_s10.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

# лист: топ до 12 хитов
sel = out["hits"][:12]
if sel:
    grid = Image.new("RGB", (4*330, 3*270), (28, 28, 28))
    d = ImageDraw.Draw(grid)
    for m, x in enumerate(sel):
        for t, (fn, key) in enumerate([(x["train_file"], "tr"), (x["test_file"], x["test_split"][:1])]):
            i = idx[os.path.splitext(fn)[0]]
            with Image.open(os.path.join(DATA, "images", files[i])) as im:
                xx, yy, ww, hh = bbox[i]
                c = im.convert("RGB").crop((xx, yy, xx+ww, yy+hh)).resize((158, 158), Image.BILINEAR)
            grid.paste(c, ((m % 4)*330+4 + t*162, (m//4)*270+4))
        d.text(((m % 4)*330+4, (m//4)*270+240),
               f"vid{x['train_vid']} cam{x['train_cam']} vs {x['test_split']} crop{x['crop']:.2f} {x['kind'][:1]}",
               fill=(255, 220, 0))
    grid.save(os.path.join(HERE, "sheet_traintest.png"))
print(json.dumps({k: out[k] for k in out if k != "hits"}, ensure_ascii=False, indent=1))
print("top hits:", json.dumps(out["hits"][:6], ensure_ascii=False, indent=1))
