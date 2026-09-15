#!/usr/bin/env python3
"""s04: осмысленность camera_id в train.csv.
Признак кадра = миниатюра 32x32 с замазанной областью bbox (фокус на фон/сцену),
нормированная (zero-mean, unit-norm). Метрики:
 - интра- vs интер-камерное сходство (косинус);
 - классификация камеры ближайшим центроидом (2-fold чёт/нечет), точность vs случайная 1/96;
 - согласованность сцены внутри каждой камеры (в т.ч. крупных 90 и 89);
 - топ похожих ПАР РАЗНЫХ камер (кандидаты «одна сцена — разные camera_id» => сессии?);
 - яркость по камерам (день/ночь внутри одной камеры);
 - геометрия bbox по камерам.
Требует s02. Выход: out_s04.json. Запуск: python3 s04_camera.py"""
from pathlib import Path
import csv, json, os
import numpy as np

REPO = Path(__file__).resolve().parents[3]  # корень репозитория
DATA = str(Path(os.environ.get("REID_DATA_DIR", REPO / "data")))
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)

files = json.load(open(os.path.join(HERE, "scan_files.json")))
idx = {os.path.splitext(f)[0]: i for i, f in enumerate(files)}
thumbs = np.load(os.path.join(HERE, "thumbs32.npy")).astype(np.float32).reshape(-1, 32, 32)

rows = list(csv.DictReader(open(os.path.join(DATA, "train.csv"))))
W, H = 1920, 1080
cam = np.array([int(r["camera_id"]) for r in rows])
vid = np.array([int(r["vehicle_id"]) for r in rows])
ti = np.array([idx[r["image_id"]] for r in rows])
bx = np.array([[int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])] for r in rows])

# фон: замазываем bbox средним по кадру
F = thumbs[ti].copy()
bright = F.reshape(len(F), -1).mean(1)
for k in range(len(F)):
    x, y, w, h = bx[k]
    x0, y0 = int(x * 32 / W), int(y * 32 / H)
    x1, y1 = int(np.ceil((x + w) * 32 / W)), int(np.ceil((y + h) * 32 / H))
    F[k, y0:y1, x0:x1] = F[k].mean()
V = F.reshape(len(F), -1)
V = V - V.mean(1, keepdims=True)
V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)

out = {}
cams = np.unique(cam)

# 1) интра/интер сходство (сэмплы пар)
def mean_pair_sim(idxs, n=2000):
    if len(idxs) < 2: return None
    a = rng.choice(idxs, n); b = rng.choice(idxs, n)
    m = a != b
    return float((V[a[m]] * V[b[m]]).sum(1).mean())
intra = {int(c): mean_pair_sim(np.where(cam == c)[0]) for c in cams}
a = rng.choice(len(V), 4000); b = rng.choice(len(V), 4000)
m = cam[a] != cam[b]
inter = float((V[a[m]] * V[b[m]]).sum(1).mean())
vals = [v for v in intra.values() if v is not None]
out["intra_camera_cosine"] = {"mean_over_cameras": round(float(np.mean(vals)), 3),
                              "min": round(min(vals), 3), "max": round(max(vals), 3),
                              "cam90": round(intra[90], 3), "cam89": round(intra[89], 3)}
out["inter_camera_cosine_mean"] = round(inter, 3)
out["intra_low_cameras"] = {int(c): round(v, 3) for c, v in sorted(intra.items(), key=lambda x: (x[1] is None, x[1]))[:8] if v is not None}

# 2) классификация камеры ближайшим центроидом, 2-fold чёт/нечет
def centroids(mask):
    C = np.zeros((len(cams), V.shape[1]), np.float32)
    for j, c in enumerate(cams):
        w = mask & (cam == c)
        if w.sum(): C[j] = V[w].mean(0)
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    return C
acc, per_cam_acc = [], {}
for fold in (0, 1):
    tr = (np.arange(len(V)) % 2) == fold
    te = ~tr
    C = centroids(tr)
    pred = cams[np.argmax(V[te] @ C.T, 1)]
    acc.append(float((pred == cam[te]).mean()))
    for c in (90, 89):
        m2 = cam[te] == c
        per_cam_acc.setdefault(c, []).append(float((pred[m2] == c).mean()))
out["nearest_centroid_camera_accuracy"] = {"2fold_mean": round(float(np.mean(acc)), 3),
    "chance": round(1 / len(cams), 4),
    "cam90": round(float(np.mean(per_cam_acc[90])), 3), "cam89": round(float(np.mean(per_cam_acc[89])), 3)}

# 3) похожие пары разных камер (центроиды всей выборки)
C = centroids(np.ones(len(V), bool))
S = C @ C.T
np.fill_diagonal(S, -1)
pairs = []
for j in range(len(cams)):
    for k in range(j + 1, len(cams)):
        pairs.append((float(S[j, k]), int(cams[j]), int(cams[k])))
pairs.sort(reverse=True)
top = pairs[:12]
def shared_ids(c1, c2):
    return int(len(set(vid[cam == c1]) & set(vid[cam == c2])))
out["top_similar_camera_pairs"] = [
    {"cams": [c1, c2], "centroid_cos": round(s, 3), "shared_vehicle_ids": shared_ids(c1, c2),
     "sizes": [int((cam == c1).sum()), int((cam == c2).sum())]} for s, c1, c2 in top]
out["camera_pair_centroid_cos_quantiles"] = {q: round(float(np.quantile([p[0] for p in pairs], qq)), 3)
    for q, qq in [("p50", .5), ("p90", .9), ("p99", .99), ("max", 1.0)]}

# 4) яркость по камерам: доля тёмных кадров (mean<60) на камеру
dark = bright < 60
out["dark_frames_train_pct"] = round(float(dark.mean()) * 100, 2)
bycam_dark = {int(c): round(float(dark[cam == c].mean()), 3) for c in cams}
out["cameras_with_day_and_night"] = int(sum(1 for c in cams if 0.05 < bycam_dark[int(c)] < 0.95))
out["dark_share_cam90_cam89"] = [bycam_dark[90], bycam_dark[89]]

# 5) геометрия bbox по камерам: средний центр и разброс
geo = {}
for c in (90, 89, 43, 82, 6):
    m3 = cam == c
    cx = (bx[m3, 0] + bx[m3, 2] / 2) / W
    cy = (bx[m3, 1] + bx[m3, 3] / 2) / H
    geo[int(c)] = {"cx_mean": round(float(cx.mean()), 2), "cx_std": round(float(cx.std()), 2),
                   "cy_mean": round(float(cy.mean()), 2), "cy_std": round(float(cy.std()), 2),
                   "area_mean_pct": round(float((bx[m3, 2] * bx[m3, 3]).mean() / (W * H)) * 100, 1)}
out["bbox_geometry_sample_cams"] = geo

with open(os.path.join(HERE, "out_s04.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
