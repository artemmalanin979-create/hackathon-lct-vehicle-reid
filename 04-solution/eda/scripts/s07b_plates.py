#!/usr/bin/env python3
"""s07b: номерные знаки — автолокализация размытой пластины и проверка однородности.
Идея: размытая пластина = гладкое (низкий градиент) прямоугольное пятно с пропорциями
номера (~4.6:1) в нижних 2/3 кропа, обычно светлее окружения.
 1) на 400 кропах (48 из s07a + 352 случайных) ищем лучшее «гладкое окно» -> метрики;
 2) рисуем боксы на 48 контрольных кропах (sheet_plates_boxed_*.png) для ручной проверки;
 3) обратный скан «нестёртых номеров»: окна с ВЫСОКОЙ плотностью границ и пропорциями
    номера; топ-12 -> sheet_suspect_text.png (ручная проверка);
 4) однородность: остаточная градиентная энергия внутри найденных боксов, день vs ночь.
Выход: out_s07.json + листы. Запуск: python3 s07b_plates.py"""
import csv, json, os
import numpy as np
from PIL import Image, ImageDraw

DATA = "/home/artem/projects/hackathon-lct-vehicle-reid/data"
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1)
W, H = 1920, 1080

files = json.load(open(os.path.join(HERE, "scan_files.json")))
idx = {os.path.splitext(f)[0]: i for i, f in enumerate(files)}
N = len(files)
split = np.full(N, -1, np.int8)
bbox = np.zeros((N, 4), np.int32)
for fn, sname in [("train.csv", 0), ("test_query.csv", 1), ("test_gallery.csv", 2)]:
    for r in csv.DictReader(open(os.path.join(DATA, fn))):
        i = idx[r["image_id"]]
        split[i] = sname
        bbox[i] = [int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])]

picks48 = [idx[os.path.splitext(f)[0]] for f in json.load(open(os.path.join(HERE, "sheet_plates_picks.json")))]
others = rng.choice(np.setdiff1d(np.arange(N), picks48), 352, replace=False)
sample = picks48 + [int(i) for i in others]

def crop_gray(i, target_w=256):
    with Image.open(os.path.join(DATA, "images", files[i])) as im:
        x, y, w, h = bbox[i]
        # draft на подходящий масштаб
        im.draft("L", (max(64, int(W*target_w/max(1, w))//2), 1))
        g = im.convert("L")
        sx, sy = g.width/W, g.height/H
        c = g.crop((int(x*sx), int(y*sy), int(np.ceil((x+w)*sx)), int(np.ceil((y+h)*sy))))
        th = max(24, int(target_w*h/w))
        return np.asarray(c.resize((target_w, th), Image.BILINEAR), np.float32)

def integral(a):
    s = np.zeros((a.shape[0]+1, a.shape[1]+1), np.float64)
    s[1:, 1:] = np.cumsum(np.cumsum(a, 0), 1)
    return s
def win_sum(S, y0, x0, hh, ww):
    return S[y0+hh, x0+ww] - S[y0, x0+ww] - S[y0+hh, x0] + S[y0, x0]

def analyze(i):
    g = crop_gray(i)
    hh, ww = g.shape
    gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    gy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
    grad = gx + gy
    Sg, Sb = integral(grad), integral(g)
    y_lo = int(hh*0.30)
    mg = grad[y_lo:].mean() + 1e-6
    mb = g[y_lo:].mean()
    best_s, best_t = None, None
    for fw in (0.10, 0.14, 0.18, 0.24, 0.30, 0.38):
        w_ = max(12, int(ww*fw)); h_ = max(5, int(w_/4.6))
        if y_lo+h_ >= hh: continue
        step = max(2, w_//8)
        for y0 in range(y_lo, hh-h_, step):
            for x0 in range(0, ww-w_, step):
                a = w_*h_
                wg = win_sum(Sg, y0, x0, h_, w_)/a
                wb = win_sum(Sb, y0, x0, h_, w_)/a
                smooth = (mg - wg)/mg + 0.25*max(0.0, (wb-mb))/(mb+1e-6)
                text = wg/mg if wb > mb*0.7 else 0
                cand_s = (smooth, (x0, y0, w_, h_), wg, wb)
                cand_t = (text, (x0, y0, w_, h_), wg, wb)
                if best_s is None or cand_s[0] > best_s[0]: best_s = cand_s
                if best_t is None or cand_t[0] > best_t[0]: best_t = cand_t
    return g, best_s, best_t, mg, mb

res = []
for i in sample:
    g, bs, bt, mg, mb = analyze(i)
    res.append({"i": i, "file": files[i], "split": int(split[i]),
                "crop_bright": round(float(mb), 1), "crop_grad": round(float(mg), 2),
                "smooth_score": round(float(bs[0]), 3), "smooth_box": bs[1],
                "smooth_grad": round(float(bs[2]), 2), "smooth_bright": round(float(bs[3]), 1),
                "grad_ratio_in_box": round(float(bs[2]/mg), 3),
                "text_score": round(float(bt[0]), 3), "text_box": bt[1],
                "hw": list(g.shape)})

out = {"n_sampled": len(res)}
gr = np.array([r["grad_ratio_in_box"] for r in res])
br = np.array([r["crop_bright"] for r in res])
qs = lambda x: {p: round(float(np.quantile(x, q)), 3) for p, q in
                [("p10", .1), ("p50", .5), ("p90", .9)]}
out["grad_ratio_in_smooth_box"] = {"all": qs(gr),
    "day_bright_ge_80": qs(gr[br >= 80]), "night_bright_lt_50": qs(gr[br < 50]),
    "n_day": int((br >= 80).sum()), "n_night": int((br < 50).sum())}
ts = np.array([r["text_score"] for r in res])
out["text_score_quantiles"] = qs(ts)
out["n_text_score_ge_3"] = int((ts >= 3).sum())

# --- листы: 48 с боксами ---
def draw_cell(i, box, maxw=320, maxh=240):
    with Image.open(os.path.join(DATA, "images", files[i])) as im:
        x, y, w, h = bbox[i]
        c = im.convert("RGB").crop((x, y, x+w, y+h))
    r = next(rr for rr in res if rr["i"] == i)
    ch, cw = r["hw"]
    c = c.resize((cw, ch), Image.BILINEAR)
    d = ImageDraw.Draw(c)
    bx, by, bw, bh = box
    d.rectangle([bx, by, bx+bw, by+bh], outline=(255, 0, 0), width=2)
    s = min(maxw/cw, maxh/ch, 1.5)
    return c.resize((int(cw*s), int(ch*s)), Image.BILINEAR)

CW, CH, PAD = 340, 265, 6
for sh in range(4):
    grid = Image.new("RGB", (4*CW, 3*CH), (28, 28, 28))
    d = ImageDraw.Draw(grid)
    for k, i in enumerate(picks48[sh*12:(sh+1)*12]):
        r = next(rr for rr in res if rr["i"] == i)
        c = draw_cell(i, r["smooth_box"])
        cx, cy = (k % 4)*CW+PAD, (k//4)*CH+PAD
        grid.paste(c, (cx, cy))
        d.text((cx+2, cy+CH-22), f"{k+sh*12}: ratio{r['grad_ratio_in_box']:.2f} b{int(r['crop_bright'])}",
               fill=(255, 220, 0))
    grid.save(os.path.join(HERE, f"sheet_plates_boxed_{sh}.png"))

# --- топ-12 подозрительных «текстовых» окон ---
top = sorted(res, key=lambda r: -r["text_score"])[:12]
grid = Image.new("RGB", (4*CW, 3*CH), (28, 28, 28))
d = ImageDraw.Draw(grid)
for k, r in enumerate(top):
    c = draw_cell(r["i"], r["text_box"])
    cx, cy = (k % 4)*CW+PAD, (k//4)*CH+PAD
    grid.paste(c, (cx, cy))
    d.text((cx+2, cy+CH-22), f"txt{r['text_score']:.1f} {r['file'][:8]} {['tr','q','g'][r['split']]}",
           fill=(255, 220, 0))
grid.save(os.path.join(HERE, "sheet_suspect_text.png"))
out["top_text_files"] = [{"file": r["file"], "text_score": r["text_score"]} for r in top]

with open(os.path.join(HERE, "out_s07.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
