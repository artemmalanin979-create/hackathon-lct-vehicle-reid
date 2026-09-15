"""Общая инфраструктура: пути, загрузка CSV, миниатюры 32x32 (рецепт EDA s02),
серии (компоненты cos>=0.90), md5-пары скопированных кадров.

Память: миниатюры 11416x1024 uint8 ~ 12 МБ; полных матриц "все на все" нет —
косинусы считаются только внутри малых подмножеств (<=2000 кадров) блоками.
"""
import csv
import hashlib
import json
import os
from multiprocessing import Pool

import numpy as np

DATA = "/home/artem/projects/hackathon-lct-vehicle-reid/data"
IMAGES = os.path.join(DATA, "images")
JOB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(JOB, "out")
SPLIT_DIR = os.path.join(JOB, "split")

SERIES_THRESHOLD = 0.90  # порог "тот же проезд" — откалиброван в EDA s05

# 9 пар скопированных кадров внутри train + 2 пары с участием теста (EDA s03,
# перепроверяются побайтово в check_split.py и пересчитываются при --full-scan).
MD5_PAIRS_TRAIN = [
    ("0fed09a1fc044696832778ed10534131", "a39cfa2e577248ed88bd4a5bc355890e"),
    ("84dabe36f1314d0fa54d4c854c844664", "8f531ff8c8ea4805a085b21bca63d41f"),
    ("1b4ab30599ea41178b775b81f5f6265c", "502870478e7b4ecdb8e068d07e1156dd"),
    ("aab87e0ce41b44d3bbbc800c6afb228f", "e9f66f92006248b486e7e75bd714a05d"),
    ("64a9bec33f0e4b21b1f9a0c8f8504741", "6d813a171e564ec4a47e9e034c95846e"),
    ("4b90daade9334addb76bf838887e1430", "6394c5b6fb4749d285faf3d8a8b6d6b4"),
    ("b7f109546f3e447fb87515f35d202127", "cb659b378e8d42d6b6287a7264298db3"),
    ("25fe4ed7ba5b433b9ae84795ea909729", "d0ab089f96254e589345c8469d982a7a"),
    ("32bc324bbfb445fe8761dde16ba7759c", "ba950758834d4f26b7df5dd4832b6797"),
]
MD5_PAIRS_MIXED = [  # (train_id, test_id) и (query_id, gallery_id) — вне нашего сплита
    ("8f5f7f037b824441b1b29e7906ef2e12", "f3fddb55448a46778b5d807642cceb71"),
    ("8d4f78a13e3e4f569761273ba212d220", "8dfe01febd46475ab5aef97033716ce1"),
]


def read_csv(name):
    with open(os.path.join(DATA, name), newline="") as f:
        return list(csv.DictReader(f))


def load_train():
    rows = read_csv("train.csv")
    for r in rows:
        r["vehicle_id"] = int(r["vehicle_id"])
        r["camera_id"] = int(r["camera_id"])
    return rows


def _thumb_one(image_id):
    from PIL import Image  # импорт в воркере
    im = Image.open(os.path.join(IMAGES, image_id + ".jpg"))
    im.draft("L", (160, 160))
    g = im.convert("L").resize((32, 32), Image.BILINEAR)
    return np.asarray(g, dtype=np.uint8).reshape(-1)


def thumbs_for(image_ids, cache_name):
    """Миниатюры 32x32 (uint8, N x 1024) с кэшем в out/. Порядок = image_ids."""
    os.makedirs(OUT, exist_ok=True)
    npy = os.path.join(OUT, cache_name + ".npy")
    ids_json = os.path.join(OUT, cache_name + "_ids.json")
    if os.path.exists(npy) and os.path.exists(ids_json):
        with open(ids_json) as f:
            cached = json.load(f)
        if cached == list(image_ids):
            return np.load(npy)
    with Pool(8) as p:
        arrs = p.map(_thumb_one, image_ids, chunksize=64)
    T = np.stack(arrs)
    np.save(npy, T)
    with open(ids_json, "w") as f:
        json.dump(list(image_ids), f)
    return T


def normed(T):
    V = T.astype(np.float32)
    V = V - V.mean(1, keepdims=True)
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)


def components(V, threshold=SERIES_THRESHOLD, block=512):
    """Компоненты связности графа cos>=threshold на N<=~2000 векторах.
    Блочно, без матрицы NxN в float64 разом (N здесь мало, но дисциплина та же)."""
    n = len(V)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for s in range(0, n, block):
        S = V[s:s + block] @ V.T  # block x n, float32
        for i in range(S.shape[0]):
            gi = s + i
            for gj in np.nonzero(S[i, gi + 1:] >= threshold)[0]:
                union(gi, int(gi + 1 + gj))
    labels = np.array([find(i) for i in range(n)])
    return labels


def series_stats(V):
    labels = components(V)
    n_series = len(set(labels.tolist()))
    return {"frames": len(V), "series": n_series,
            "frames_per_series": round(len(V) / n_series, 4) if n_series else None}


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
