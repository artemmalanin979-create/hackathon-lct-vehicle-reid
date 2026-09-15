#!/usr/bin/env python3
"""s05b: у пар query<->gallery «тот же момент/место» — та же ли МАШИНА размечена?
Пары: (A) полнокадровый cos32>=0.90 (тот же всплеск кадров);
      (B) bbox IoU>=0.80 при cos32<0.90 (кандидаты «припаркована, переснята позже»).
Сходство объектов: кроп bbox из полукадра (draft 1/2), grayscale 64x64,
zero-mean/unit-norm, косинус. Калибровка порога на train-парах с известным vehicle_id:
позитив = same vid & cos32>=0.90, негатив = diff vid & cos32>=0.80 (та же сцена, другая машина).
Выход: out_s05b.json, qg_flagged_pairs.csv, sheet_qg_*.png. Память: по одному файлу."""
import csv, json, os
import numpy as np
from PIL import Image, ImageDraw

DATA = "/home/artem/projects/hackathon-lct-vehicle-reid/data"
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)
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

def iou_mat(A, B):
    ax1, ay1 = A[:, 0:1], A[:, 1:2]; ax2, ay2 = ax1 + A[:, 2:3], ay1 + A[:, 3:4]
    bx1, by1 = B[:, 0], B[:, 1]; bx2, by2 = bx1 + B[:, 2], by1 + B[:, 3]
    ix = np.maximum(0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    iy = np.maximum(0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = ix * iy
    return inter / (A[:, 2:3]*A[:, 3:4] + (B[:, 2]*B[:, 3])[None, :] - inter)

qi = np.where(split == 1)[0]; gi = np.where(split == 2)[0]
S = V[qi] @ V[gi].T
IOU = iou_mat(bbox[qi], bbox[gi])
A_pairs = [(int(qi[a]), int(gi[b]), float(S[a, b]), float(IOU[a, b]))
           for a, b in zip(*np.nonzero(S >= 0.90))]
B_pairs = [(int(qi[a]), int(gi[b]), float(S[a, b]), float(IOU[a, b]))
           for a, b in zip(*np.nonzero((IOU >= 0.80) & (S < 0.90)))]

cache = {}
def crop64(i):
    if i not in cache:
        if len(cache) > 500: cache.clear()
        with Image.open(os.path.join(DATA, "images", files[i])) as im:
            im.draft("L", (960, 540))
            g = im.convert("L")
            sx, sy = g.width / W, g.height / H
            x, y, w, h = bbox[i]
            c = g.crop((int(x*sx), int(y*sy), int(np.ceil((x+w)*sx)), int(np.ceil((y+h)*sy))))
            v = np.asarray(c.resize((64, 64), Image.BILINEAR), np.float32).ravel()
            v -= v.mean(); v /= (np.linalg.norm(v) + 1e-9)
            cache[i] = v
    return cache[i]

def crop_cos(pairs):
    return np.array([float(crop64(a) @ crop64(b)) for a, b, *_ in pairs])

# --- калибровка на train ---
tr = np.where(split == 0)[0]
pos, neg = [], []
Btr = 512
for i0 in range(0, len(tr), Btr):
    Str = V[tr[i0:i0+Btr]] @ V[tr].T
    r, c = np.nonzero(Str >= 0.80)
    for a, b in zip(r, c):
        ia, ib = int(tr[i0+a]), int(tr[b])
        if ia >= ib: continue
        if vid[ia] == vid[ib] and Str[a, b] >= 0.90:
            pos.append((ia, ib))
        elif vid[ia] != vid[ib]:
            neg.append((ia, ib))
rng.shuffle(pos)
pos = pos[:300]
pcos = crop_cos(pos); ncos = crop_cos(neg)
qs = lambda x: {p: round(float(np.quantile(x, q)), 3) for p, q in
                [("p05", .05), ("p50", .5), ("p95", .95)]} if len(x) else None
out = {"calib_pos_same_vid_n": len(pos), "calib_neg_diff_vid_n": len(neg),
       "calib_pos_cropcos": qs(pcos), "calib_neg_cropcos": qs(ncos)}
thr = 0.80
out["crop_threshold_used"] = thr
out["calib_pos_share_ge_thr"] = round(float((pcos >= thr).mean()), 3)
out["calib_neg_share_ge_thr"] = round(float((ncos >= thr).mean()), 3)

# --- применение к q-g парам ---
acos = crop_cos(A_pairs); bcos = crop_cos(B_pairs)
out["A_same_moment_pairs"] = {"n": len(A_pairs), "cropcos": qs(acos),
    "n_same_vehicle_like": int((acos >= thr).sum()),
    "n_queries_with_same_vehicle_like": len(set(a for (a, b, s, i), cc in zip(A_pairs, acos) if cc >= thr))}
out["B_parked_pairs_iou0.8_cos_lt_0.9"] = {"n": len(B_pairs), "cropcos": qs(bcos),
    "n_same_vehicle_like": int((bcos >= thr).sum()),
    "n_queries_with_same_vehicle_like": len(set(a for (a, b, s, i), cc in zip(B_pairs, bcos) if cc >= thr))}
qA = set(a for (a, b, s, i), cc in zip(A_pairs, acos) if cc >= thr)
qB = set(a for (a, b, s, i), cc in zip(B_pairs, bcos) if cc >= thr)
out["queries_with_any_same_vehicle_same_place_candidate"] = len(qA | qB)

with open(os.path.join(HERE, "qg_flagged_pairs.csv"), "w", newline="") as f:
    wcsv = csv.writer(f)
    wcsv.writerow(["query_image_id", "gallery_image_id", "kind", "fullframe_cos32", "bbox_iou", "crop_cos64"])
    for (a, b, s, i), cc in zip(A_pairs, acos):
        wcsv.writerow([os.path.splitext(files[a])[0], os.path.splitext(files[b])[0], "A_same_moment", round(s, 4), round(i, 4), round(cc, 4)])
    for (a, b, s, i), cc in zip(B_pairs, bcos):
        wcsv.writerow([os.path.splitext(files[a])[0], os.path.splitext(files[b])[0], "B_parked_reobs", round(s, 4), round(i, 4), round(cc, 4)])

# --- контрольные листы: по 12 пар каждого типа (кропы рядом) ---
def sheet(pairs, ccos, name):
    if not len(pairs): return
    k = np.linspace(0, len(pairs)-1, min(12, len(pairs))).astype(int)
    seen, sel = set(), []
    for j in np.argsort([-pairs[i][2] for i in range(len(pairs))]):
        if len(sel) >= 12: break
        if pairs[j][0] in seen: continue
        seen.add(pairs[j][0]); sel.append(j)
    grid = Image.new("RGB", (4*330, 3*270), (28, 28, 28))
    d = ImageDraw.Draw(grid)
    for m, j in enumerate(sel):
        a, b, s, i = pairs[j]
        cell_x, cell_y = (m % 4)*330+4, (m//4)*270+4
        for t, ii in enumerate([a, b]):
            with Image.open(os.path.join(DATA, "images", files[ii])) as im:
                x, y, w, h = bbox[ii]
                c = im.convert("RGB").crop((x, y, x+w, y+h)).resize((158, 158*h//w if w else 158), Image.BILINEAR)
                c = c.crop((0, 0, 158, min(c.height, 230)))
                grid.paste(c, (cell_x + t*162, cell_y))
        d.text((cell_x, cell_y+238), f"ff{s:.2f} iou{i:.2f} crop{ccos[j]:.2f}", fill=(255, 220, 0))
    grid.save(os.path.join(HERE, f"sheet_qg_{name}.png"))
sheet(A_pairs, acos, "A_same_moment")
sheet(B_pairs, bcos, "B_parked")

with open(os.path.join(HERE, "out_s05b.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
