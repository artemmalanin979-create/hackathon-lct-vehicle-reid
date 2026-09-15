#!/usr/bin/env python3
"""Шаг 2: протокол ПОДБОРА из train_fit (идентичности не пересекаются с вал — свойство P1 сплита).

Зеркалит численную структуру вал-протокола (цели: галерея ~750 = 1 кадр на
(vehicle, camera) у 295 идентичностей; 832 известных запроса ≤3 на идентичность;
278 отказных запросов ≤4 на идентичность от 74 идентичностей без галереи).
Серийная логика (cos32-дубли) НЕ воспроизводится — для подбора параметров
пост-обработки достаточно совпадения размеров и камерной структуры; отличие
зафиксировано в отчёте как ограничение.

Детерминизм: numpy.default_rng(20260915), сортированные списки. Протокол построен
ДО каких-либо замеров метрик на нём и по результатам не корректировался.
Пишет tune/tune_query.csv (+has_mate), tune/tune_gallery.csv, tune/manifest.json.
"""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import JOB, SPLIT, read_meta

SEED = 20260915
N_GALLERY_IDS = 295
N_REFUSAL_IDS = 74
N_KNOWN_Q = 832
N_REFUSAL_Q = 278
MAX_Q_PER_ID = 3
MAX_REF_Q_PER_ID = 4

rng = np.random.default_rng(SEED)
rows = read_meta(SPLIT / "train_fit.csv")
by_vid = defaultdict(list)
for r in rows:
    by_vid[r["vehicle_id"]].append(r)
vids = sorted(by_vid, key=int)

chosen = rng.choice(len(vids), size=N_GALLERY_IDS + N_REFUSAL_IDS, replace=False)
paired_ids = [vids[i] for i in sorted(chosen[:N_GALLERY_IDS])]
refusal_ids = [vids[i] for i in sorted(chosen[N_GALLERY_IDS:])]

gallery, known_q = [], []
for vid in paired_ids:
    frames = sorted(by_vid[vid], key=lambda r: r["image_id"])
    by_cam = defaultdict(list)
    for r in frames:
        by_cam[r["camera_id"]].append(r)
    leftovers = []
    for cam in sorted(by_cam):
        pick = rng.integers(len(by_cam[cam]))
        gallery.append(by_cam[cam][pick])
        leftovers += [r for i, r in enumerate(by_cam[cam]) if i != pick]
    if leftovers:
        take = min(len(leftovers), MAX_Q_PER_ID)
        idx = rng.choice(len(leftovers), size=take, replace=False)
        known_q += [leftovers[i] for i in sorted(idx)]

refusal_q = []
for vid in refusal_ids:
    frames = sorted(by_vid[vid], key=lambda r: r["image_id"])
    take = min(len(frames), MAX_REF_Q_PER_ID)
    idx = rng.choice(len(frames), size=take, replace=False)
    refusal_q += [frames[i] for i in sorted(idx)]

# подрезка до целевых чисел (равномерно случайно, детерминированно)
if len(known_q) > N_KNOWN_Q:
    keep = sorted(rng.choice(len(known_q), size=N_KNOWN_Q, replace=False))
    known_q = [known_q[i] for i in keep]
if len(refusal_q) > N_REFUSAL_Q:
    keep = sorted(rng.choice(len(refusal_q), size=N_REFUSAL_Q, replace=False))
    refusal_q = [refusal_q[i] for i in keep]

# проверки конструкции
g_ids = {r["vehicle_id"] for r in gallery}
g_cams = defaultdict(set)
for r in gallery:
    g_cams[r["vehicle_id"]].add(r["camera_id"])
g_imgs = {r["image_id"] for r in gallery}
for r in known_q:
    assert r["vehicle_id"] in g_ids and r["image_id"] not in g_imgs
    assert g_cams[r["vehicle_id"]] - {r["camera_id"]}, "нет кросс-камерной пары"
for r in refusal_q:
    assert r["vehicle_id"] not in g_ids

tune = JOB / "tune"
tune.mkdir(exist_ok=True)
qcols = ["image_id", "x", "y", "w", "h", "vehicle_id", "camera_id", "has_mate"]
gcols = qcols[:-1]
with open(tune / "tune_query.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(qcols)
    for r in known_q:
        w.writerow([r[c] for c in gcols] + ["1"])
    for r in refusal_q:
        w.writerow([r[c] for c in gcols] + ["0"])
with open(tune / "tune_gallery.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(gcols)
    for r in gallery:
        w.writerow([r[c] for c in gcols])

sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
       for p in [tune / "tune_query.csv", tune / "tune_gallery.csv"]}
manifest = {
    "seed": SEED, "source": "train_fit.csv (идентичности ∩ вал = ∅)",
    "gallery_frames": len(gallery), "gallery_ids": len(g_ids),
    "known_queries": len(known_q), "known_query_ids": len({r["vehicle_id"] for r in known_q}),
    "refusal_queries": len(refusal_q), "refusal_ids": len({r["vehicle_id"] for r in refusal_q}),
    "sha256": sha,
}
(tune / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(manifest, indent=2, ensure_ascii=False))
