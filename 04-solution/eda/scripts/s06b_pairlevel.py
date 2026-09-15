#!/usr/bin/env python3
"""s06b: точная парная оценка «same camera?» по сходству сцены (без транзитивного
замыкания, чтобы исключить эффект цепочек) + кластеризация mutual-kNN.
Фичи из thumbs32: raw (замазан bbox), grad (градиентная карта), concat.
 1) точный проход по всем C(9556,2)=45.7M парам train: для сетки порогов считаем
    precision/recall предсказания "одна камера" (+ отдельно "same pass" fullframe>=0.90 OR scene>=t);
 2) mutual-kNN граф (k=5) + порог, компоненты, парные P/R/purity vs camera_id;
 3) применение лучшей связки к тесту.
Выход: out_s06b.json. Запуск: python3 s06b_pairlevel.py"""
import csv, json, os
import numpy as np

DATA = "/home/artem/projects/hackathon-lct-vehicle-reid/data"
HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1920, 1080

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
Fm = T.copy()
for k in range(N):
    x, y, w, h = bbox[k]
    x0, y0 = int(x*32/W), int(y*32/H)
    x1, y1 = int(np.ceil((x+w)*32/W)), int(np.ceil((y+h)*32/H))
    Fm[k, y0:y1, x0:x1] = Fm[k].mean()

def norm(M):
    M = M.reshape(len(M), -1).astype(np.float32)
    M = M - M.mean(1, keepdims=True)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)

gx = np.diff(Fm, axis=2, prepend=Fm[:, :, :1])
gy = np.diff(Fm, axis=1, prepend=Fm[:, :1, :])
G = np.sqrt(gx**2 + gy**2)
feats = {"raw": norm(Fm), "grad": norm(G),
         "concat": norm(np.concatenate([norm(Fm), norm(G)], 1))}
# full-frame (без маски) для канала "same pass"
Traw = np.load(os.path.join(HERE, "thumbs32.npy")).astype(np.float32)
Vff = norm(Traw)

out = {}
tr = np.where(split == 0)[0]
ctr = cam[tr]
n_tr = len(tr)
tot_pairs = n_tr*(n_tr-1)//2
same_cam_pairs = int(sum(c*(c-1)//2 for c in np.bincount(ctr)))
out["train_pairs_total"] = tot_pairs
out["train_same_camera_pairs"] = same_cam_pairs

TH = [0.5, 0.6, 0.7, 0.8, 0.9]
B = 512
res = {}
for name, V in feats.items():
    Vt = V[tr]; Vf = Vff[tr]
    cnt_ge = np.zeros(len(TH), np.int64); cnt_ge_sc = np.zeros(len(TH), np.int64)
    cnt_or = np.zeros(len(TH), np.int64); cnt_or_sc = np.zeros(len(TH), np.int64)
    for i0 in range(0, n_tr, B):
        S = Vt[i0:i0+B] @ Vt.T
        Sf = Vf[i0:i0+B] @ Vf.T
        samec = (ctr[i0:i0+B, None] == ctr[None, :])
        iu = np.arange(i0, min(i0+B, n_tr))[:, None] < np.arange(n_tr)[None, :]
        ff = Sf >= 0.90
        for ti, t in enumerate(TH):
            m = (S >= t) & iu
            cnt_ge[ti] += int(m.sum()); cnt_ge_sc[ti] += int((m & samec).sum())
            mo = (m | (ff & iu))
            cnt_or[ti] += int(mo.sum()); cnt_or_sc[ti] += int((mo & samec).sum())
    res[name] = {str(t): {
        "pairs_flagged": int(cnt_ge[ti]),
        "precision_same_cam": round(cnt_ge_sc[ti]/cnt_ge[ti], 3) if cnt_ge[ti] else None,
        "recall_same_cam": round(cnt_ge_sc[ti]/same_cam_pairs, 4),
        "with_fullframe_or": {
            "pairs_flagged": int(cnt_or[ti]),
            "precision": round(cnt_or_sc[ti]/cnt_or[ti], 3) if cnt_or[ti] else None,
            "recall": round(cnt_or_sc[ti]/same_cam_pairs, 4)}} for ti, t in enumerate(TH)}
out["pair_level_same_camera"] = res

# --- mutual-kNN кластеризация ---
def knn_graph(V, sub, k=5, tmin=0.5):
    Vs = V[sub]; n = len(sub)
    nbr = np.zeros((n, k), np.int32); nbs = np.zeros((n, k), np.float32)
    for i0 in range(0, n, B):
        S = Vs[i0:i0+B] @ Vs.T
        for r in range(S.shape[0]):
            S[r, i0+r] = -2
        part = np.argpartition(-S, k, axis=1)[:, :k]
        nbr[i0:i0+B] = part
        nbs[i0:i0+B] = np.take_along_axis(S, part, 1)
    edges = set()
    for i in range(n):
        for j, s in zip(nbr[i], nbs[i]):
            if s >= tmin and i in nbr[j] and i < j:
                edges.add((i, int(j)))
            elif s >= tmin and i in nbr[j] and j < i:
                edges.add((int(j), i))
    return list(edges)

def components(n, edges):
    parent = np.arange(n)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj: parent[ri] = rj
    lab = np.array([find(i) for i in range(n)])
    _, lab = np.unique(lab, return_inverse=True)
    return lab

def pair_metrics(pred, truth):
    key = pred.astype(np.int64)*(truth.max()+1) + truth
    _, cell = np.unique(key, return_counts=True)
    tp = float((cell*(cell-1)//2).sum())
    _, pc = np.unique(pred, return_counts=True)
    _, tc = np.unique(truth, return_counts=True)
    ppos = float((pc*(pc-1)//2).sum()); tpos = float((tc*(tc-1)//2).sum())
    prec = tp/ppos if ppos else 1.0; rec = tp/tpos
    pur = 0
    for p in np.unique(pred):
        m = pred == p
        _, c = np.unique(truth[m], return_counts=True)
        pur += c.max()
    return {"pairwise_precision": round(prec, 3), "pairwise_recall": round(rec, 3),
            "pairwise_f1": round(2*prec*rec/(prec+rec), 3) if prec+rec else 0,
            "purity": round(pur/len(truth), 3), "n_clusters": int(pred.max()+1),
            "largest": int(np.bincount(pred).max())}

mres = {}
for name in ["raw", "grad", "concat"]:
    for tmin in [0.55, 0.65, 0.75]:
        edges = knn_graph(feats[name], tr, k=5, tmin=tmin)
        lab = components(n_tr, edges)
        mres[f"{name}_t{tmin}"] = pair_metrics(lab, ctr)
out["mutual_knn_train"] = mres

with open(os.path.join(HERE, "out_s06b.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
