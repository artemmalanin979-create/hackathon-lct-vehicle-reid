#!/usr/bin/env python3
"""s08: сложность данных — площади bbox, доля тёмных кадров, крайние случаи.
Числа по CSV + яркость по thumbs32. Выход: out_s08.json. Запуск: python3 s08_complexity.py"""
import csv, json, os
import numpy as np

DATA = "/home/artem/projects/hackathon-lct-vehicle-reid/data"
HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1920, 1080
FRAME = W * H

files = json.load(open(os.path.join(HERE, "scan_files.json")))
idx = {os.path.splitext(f)[0]: i for i, f in enumerate(files)}
T = np.load(os.path.join(HERE, "thumbs32.npy")).astype(np.float32)
bright_all = T.mean(1)

out = {}
qs = lambda x, r=1: {p: round(float(np.quantile(x, q)), r) for p, q in
                     [("min", 0), ("p10", .1), ("p50", .5), ("p90", .9), ("max", 1.0)]}
for fn, key in [("train.csv", "train"), ("test_query.csv", "query"), ("test_gallery.csv", "gallery")]:
    rows = list(csv.DictReader(open(os.path.join(DATA, fn))))
    b = np.array([[int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])] for r in rows])
    ii = np.array([idx[r["image_id"]] for r in rows])
    area = b[:, 2].astype(np.int64) * b[:, 3]
    ar = b[:, 2] / b[:, 3]
    br = bright_all[ii]
    touch = ((b[:, 0] == 0) | (b[:, 1] == 0) | (b[:, 0]+b[:, 2] >= W) | (b[:, 1]+b[:, 3] >= H))
    out[key] = {
        "n": len(rows),
        "bbox_area_px": qs(area, 0),
        "bbox_area_share_of_frame_pct": qs(area/FRAME*100, 2),
        "share_area_lt_2pct_frame": round(float((area < 0.02*FRAME).mean()), 4),
        "share_area_lt_5pct_frame": round(float((area < 0.05*FRAME).mean()), 4),
        "share_area_gt_25pct_frame": round(float((area > 0.25*FRAME).mean()), 4),
        "min_w": int(b[:, 2].min()), "min_h": int(b[:, 3].min()),
        "aspect_w_over_h": qs(ar, 2),
        "n_aspect_gt_2.5": int((ar > 2.5).sum()), "n_aspect_lt_0.5": int((ar < 0.5).sum()),
        "n_touch_frame_edge": int(touch.sum()),
        "share_touch_frame_edge": round(float(touch.mean()), 3),
        "frame_brightness": qs(br),
        "share_dark_lt_60": round(float((br < 60).mean()), 4),
        "share_dark_lt_40": round(float((br < 40).mean()), 4)}
with open(os.path.join(HERE, "out_s08.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
