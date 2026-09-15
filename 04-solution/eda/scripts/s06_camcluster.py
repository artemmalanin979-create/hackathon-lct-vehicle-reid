#!/usr/bin/env python3
"""s06: суррогат «точки наблюдения» без меток и его честная оценка на train.
Признак сцены = миниатюра 32x32 с замазанным bbox (как в s04), zero-mean/unit-norm.
Группировка = компоненты связности графа "cos >= t" (union-find), t из сетки.
Оценка на train против истинного camera_id: парные precision/recall/F1, purity.
Применение лучшего t к тесту (query+gallery): кластеры, покрытие query кандидатами
gallery из той же точки; заодно — неоднородность камер 90/89 и слияния разных camera_id.
Память: блочно. Выход: out_s06.json. Запуск: python3 s06_camcluster.py"""
from pathlib import Path
import csv, json, os
import numpy as np

REPO = Path(__file__).resolve().parents[3]  # корень репозитория
DATA = str(Path(os.environ.get("REID_DATA_DIR", REPO / "data")))
HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1920, 1080
GRID = [0.45, 0.55, 0.65, 0.75, 0.85]
EDGE_CAP = 30_000_000

files = json.load(open(os.path.join(HERE, "scan_files.json")))
idx = {os.path.splitext(f)[0]: i for i, f in enumerate(files)}
N = len(files)
split = np.full(N, -1, np.int8)
vid = np.full(N, -1, np.int32); cam = np.full(N, -1, np.int32)
bbox = np.zeros((N, 4), np.int32)
for fn, sname in [("train.csv", 0), ("test_query.csv", 1), ("test_gallery.csv", 2)]:
    for r in csv.DictReader(open(os.path.join(DATA, fn))):
        i = idx[r["image_id"]]
        split[i] = sname
        bbox[i] = [int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])]
        if sname == 0:
            vid[i] = int(r["vehicle_id"]); cam[i] = int(r["camera_id"])

T = np.load(os.path.join(HERE, "thumbs32.npy")).astype(np.float32).reshape(-1, 32, 32)
F = T.copy()
for k in range(N):
    x, y, w, h = bbox[k]
    x0, y0 = int(x*32/W), int(y*32/H)
    x1, y1 = int(np.ceil((x+w)*32/W)), int(np.ceil((y+h)*32/H))
    F[k, y0:y1, x0:x1] = F[k].mean()
V = F.reshape(N, -1)
V = V - V.mean(1, keepdims=True)
V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)

def edges_above(sub_idx, tmin):
    """пары (i<j, cos>=tmin) внутри подмножества sub_idx; возвращает локальные индексы"""
    Vs = V[sub_idx]
    n = len(sub_idx)
    ei, ej, es = [], [], []
    tot = 0
    B = 512
    for i0 in range(0, n, B):
        S = Vs[i0:i0+B] @ Vs.T
        r, c = np.nonzero(S >= tmin)
        keep = (r + i0) < c
        r, c = r[keep], c[keep]
        ei.append((r+i0).astype(np.int32)); ej.append(c.astype(np.int32))
        es.append(S[r, c].astype(np.float32))
        tot += len(r)
        if tot > EDGE_CAP:
            raise SystemExit("edge cap exceeded")
    return np.concatenate(ei), np.concatenate(ej), np.concatenate(es)

def components(n, a, b):
    parent = np.arange(n)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in zip(a, b):
        ri, rj = find(i), find(j)
        if ri != rj: parent[ri] = rj
    lab = np.array([find(i) for i in range(n)])
    _, lab = np.unique(lab, return_inverse=True)
    return lab

def pair_metrics(pred, truth):
    """парные precision/recall/F1 + purity по таблице сопряжённости"""
    key = pred.astype(np.int64) * (truth.max()+1) + truth
    _, cell = np.unique(key, return_counts=True)
    tp = float((cell*(cell-1)//2).sum())
    _, pc = np.unique(pred, return_counts=True)
    _, tc = np.unique(truth, return_counts=True)
    ppos = float((pc*(pc-1)//2).sum()); tpos = float((tc*(tc-1)//2).sum())
    prec = tp/ppos if ppos else 1.0; rec = tp/tpos
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    # purity: доля кадров, попавших в «мажоритарную камеру» своего кластера
    pur = 0
    for p in np.unique(pred):
        m = pred == p
        _, c = np.unique(truth[m], return_counts=True)
        pur += c.max()
    return {"pairwise_precision": round(prec, 3), "pairwise_recall": round(rec, 3),
            "pairwise_f1": round(f1, 3), "purity": round(pur/len(truth), 3),
            "n_clusters": int(pred.max()+1),
            "n_clusters_ge2": int((np.bincount(pred) >= 2).sum())}

out = {}

# --- оценка на train ---
tr = np.where(split == 0)[0]
a, b, s = edges_above(tr, min(GRID))
out["train_edges_ge_min_t"] = int(len(a))
res = {}
for t in GRID:
    m = s >= t
    lab = components(len(tr), a[m], b[m])
    r = pair_metrics(lab, cam[tr])
    res[str(t)] = r
out["train_grouping_vs_camera_id"] = res
best_t = max(GRID, key=lambda t: res[str(t)]["pairwise_f1"])
out["best_t_by_f1"] = best_t

# лучший t: детали ошибок
m = s >= best_t
lab = components(len(tr), a[m], b[m])
# кластеры, объединяющие >=2 камер с >=3 кадрами каждая (кандидаты «сессии одной точки»)
merged = []
for p in np.unique(lab):
    mm = lab == p
    cs, cc = np.unique(cam[tr][mm], return_counts=True)
    big = [(int(c), int(n)) for c, n in zip(cs, cc) if n >= 3]
    if len(big) >= 2:
        merged.append({"cluster_size": int(mm.sum()), "cameras": big})
merged.sort(key=lambda d: -d["cluster_size"])
out["clusters_merging_cameras_at_best_t"] = {"n": len(merged), "top": merged[:12]}
# неоднородность крупных камер: сколько кластеров (>=3 кадров) внутри cam90/cam89
for c in (90, 89, 43, 82):
    mm = cam[tr] == c
    ll = lab[mm]
    _, cnt = np.unique(ll, return_counts=True)
    out.setdefault("scene_clusters_inside_camera", {})[str(c)] = {
        "frames": int(mm.sum()), "clusters_ge3": int((cnt >= 3).sum()),
        "clusters_ge2": int((cnt >= 2).sum()), "singletons": int((cnt == 1).sum()),
        "largest": int(cnt.max())}

# согласованность: у одной (vid,cam)-группы кадры в одном кластере? (сан-чек проездов)
grp = {}
for k, i in enumerate(tr):
    grp.setdefault((vid[i], cam[i]), []).append(k)
same_cluster = sum(1 for g in grp.values() if len(g) > 1 and len(set(lab[g])) == 1)
multi = sum(1 for g in grp.values() if len(g) > 1)
out["same_pass_groups_in_one_cluster_at_best_t"] = {
    "groups": multi, "fully_in_one_cluster": same_cluster,
    "share": round(same_cluster/multi, 3)}

# --- применение к тесту ---
te = np.where(split != 0)[0]
a2, b2, s2 = edges_above(te, best_t)
lab2 = components(len(te), a2[s2 >= best_t], b2[s2 >= best_t])
spl2 = split[te]
cnt = np.bincount(lab2)
out["test_clustering_at_best_t"] = {
    "n_frames": len(te), "n_clusters": int(lab2.max()+1),
    "n_clusters_ge2": int((cnt >= 2).sum()), "largest": int(cnt.max()),
    "singletons": int((cnt == 1).sum())}
# для каждого query: кандидаты gallery в том же кластере
qmask = spl2 == 1
qlab = lab2[qmask]
gal_in = np.bincount(lab2[spl2 == 2], minlength=lab2.max()+1)
q_with_g = int((gal_in[qlab] > 0).sum())
out["queries_with_gallery_in_same_scene_cluster"] = {
    "n": q_with_g, "of": int(qmask.sum()), "share": round(q_with_g/qmask.sum(), 3)}
# размер «своей» галерейной группы для таких query
gsizes = gal_in[qlab][gal_in[qlab] > 0]
out["gallery_candidates_in_same_cluster_per_query"] = {
    "p50": float(np.median(gsizes)) if len(gsizes) else None,
    "max": int(gsizes.max()) if len(gsizes) else None}

# --- оценка числа «проездов» (для п.8): компоненты full-frame cos>=0.90 в тесте ---
Traw = np.load(os.path.join(HERE, "thumbs32.npy")).astype(np.float32)
Vr = Traw - Traw.mean(1, keepdims=True)
Vr /= (np.linalg.norm(Vr, axis=1, keepdims=True) + 1e-9)
def comps_pass(sub_idx, t=0.90):
    Vs = Vr[sub_idx]
    n = len(sub_idx)
    ei, ej = [], []
    B = 512
    for i0 in range(0, n, B):
        S = Vs[i0:i0+B] @ Vs.T
        r, c = np.nonzero(S >= t)
        keep = (r+i0) < c
        ei.append((r[keep]+i0).astype(np.int32)); ej.append(c[keep].astype(np.int32))
    lab = components(n, np.concatenate(ei), np.concatenate(ej))
    return int(lab.max()+1)
out["pass_groups_fullframe_0.90"] = {
    "query": comps_pass(np.where(split == 1)[0]),
    "gallery": comps_pass(np.where(split == 2)[0]),
    "test_all": comps_pass(te),
    "train": comps_pass(tr)}

with open(os.path.join(HERE, "out_s06.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
