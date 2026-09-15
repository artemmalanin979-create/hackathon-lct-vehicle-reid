#!/usr/bin/env python3
"""s01: точные колонки CSV, типы, диапазоны, счётчики, распределения.
Запуск: python3 s01_csv_stats.py  -> печатает JSON и пишет out_s01.json рядом."""
import csv, json, os, statistics as st
from collections import Counter

DATA = "/home/artem/projects/hackathon-lct-vehicle-reid/data"
HERE = os.path.dirname(os.path.abspath(__file__))

def read_csv(name):
    path = os.path.join(DATA, name)
    with open(path, "rb") as f:
        raw_first = f.readline()
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return raw_first.decode("utf-8", "replace").rstrip("\r\n"), rows[0], rows[1:]

def col_profile(header, rows):
    prof = {}
    for i, name in enumerate(header):
        vals = [r[i] for r in rows]
        is_int = all(v.lstrip("-").isdigit() for v in vals)
        p = {"type": "int" if is_int else "str", "n_unique": len(set(vals))}
        if is_int:
            iv = [int(v) for v in vals]
            p.update(min=min(iv), max=max(iv))
        else:
            p.update(sample=vals[0], minlen=min(map(len, vals)), maxlen=max(map(len, vals)))
        prof[name] = p
    return prof

out = {}
sets = {}
for name in ["train.csv", "test_query.csv", "test_gallery.csv"]:
    raw, header, rows = read_csv(name)
    ids = [r[header.index("image_id")] for r in rows]
    dup_rows = len(rows) - len({tuple(r) for r in rows})
    dup_ids = len(ids) - len(set(ids))
    sets[name] = set(ids)
    out[name] = {
        "raw_header_line": raw, "columns": header, "n_rows": len(rows),
        "dup_full_rows": dup_rows, "dup_image_id": dup_ids,
        "profile": col_profile(header, rows),
    }
    out[name]["objects_per_image_max"] = max(Counter(ids).values())

# пересечения image_id между файлами
names = list(sets)
out["image_id_overlaps"] = {f"{a} & {b}": len(sets[a] & sets[b])
                            for i, a in enumerate(names) for b in names[i+1:]}

# распределения по train
_, header, rows = read_csv("train.csv")
vi = header.index("vehicle_id"); ci = header.index("camera_id")
per_vid = Counter(r[vi] for r in rows)
cnts = sorted(per_vid.values())
out["train_identities"] = {
    "n_identities": len(per_vid),
    "frames_per_identity": {"min": cnts[0], "median": st.median(cnts), "max": cnts[-1],
                            "hist": dict(sorted(Counter(cnts).items()))},
}
per_cam = Counter(r[ci] for r in rows)
cam_cnts = sorted(per_cam.values(), reverse=True)
out["train_cameras"] = {
    "n_cameras": len(per_cam),
    "cam_id_range": [min(map(int, per_cam)), max(map(int, per_cam))],
    "frames_per_camera_top10": cam_cnts[:10],
    "frames_per_camera": {"min": cam_cnts[-1], "median": st.median(cam_cnts), "max": cam_cnts[0]},
    "by_camera": {k: per_cam[k] for k in sorted(per_cam, key=lambda x: -per_cam[x])},
}
# камер на идентичность
vid2cams = {}
for r in rows:
    vid2cams.setdefault(r[vi], set()).add(r[ci])
ncams = sorted(len(s) for s in vid2cams.values())
out["train_cams_per_identity"] = {"min": ncams[0], "median": st.median(ncams), "max": ncams[-1],
                                  "hist": dict(sorted(Counter(ncams).items()))}
# сколько (vehicle_id, camera_id) пар имеют >1 кадра (повторные кадры той же машины той же камерой)
vidcam = Counter((r[vi], r[ci]) for r in rows)
rep = Counter(v for v in vidcam.values())
out["train_frames_per_vid_cam_pair_hist"] = dict(sorted(rep.items()))

with open(os.path.join(HERE, "out_s01.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:6000])
