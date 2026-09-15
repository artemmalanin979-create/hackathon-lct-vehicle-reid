#!/usr/bin/env python3
"""s06d: финальный суррогат «той же точки наблюдения» и его честная оценка.
Сигналы: grad96 (градиентная карта миниатюры 96x54 с замазанным bbox, косинус),
ff32 (полнокадровый косинус 32x32), IoU bbox, crop64 (кроп объекта, выборочно).
 1) точные P/R «same camera» на всех парах train для grad96 и IoU;
 2) на всех same-vid парах train: recall инструмента на same-cam (то, что надо исключать)
    и ложные срабатывания на diff-cam (истинные кросс-камерные матчи, которые жалко терять);
    crop64 — на подвыборке;
 3) mutual-kNN кластеризация grad96 на train (метрики vs camera_id) и на тесте;
 4) итоговые флаги для теста уже посчитаны в s05b (qg_flagged_pairs.csv).
Выход: out_s06d.json, test_scene_clusters.csv. Запуск: python3 s06d_surrogate.py"""
import csv, json, os
import numpy as np
from PIL import Image

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

# --- фичи ---
T32 = np.load(os.path.join(HERE, "thumbs32.npy")).astype(np.float32)
Vff = T32 - T32.mean(1, keepdims=True)
Vff /= (np.linalg.norm(Vff, axis=1, keepdims=True) + 1e-9)
del T32

T96 = np.load(os.path.join(HERE, "thumbs96.npy")).astype(np.float32).reshape(-1, 54, 96)
for k in range(N):
    x, y, w, h = bbox[k]
    x0, y0 = int(x*96/W), int(y*54/H)
    x1, y1 = int(np.ceil((x+w)*96/W)), int(np.ceil((y+h)*54/H))
    T96[k, y0:y1, x0:x1] = T96[k].mean()
gx = np.diff(T96, axis=2, prepend=T96[:, :, :1])
gy = np.diff(T96, axis=1, prepend=T96[:, :1, :])
G = np.sqrt(gx*gx + gy*gy).reshape(N, -1)
del gx, gy, T96
G -= G.mean(1, keepdims=True)
G /= (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)

out = {}
tr = np.where(split == 0)[0]
ctr = cam[tr]; n_tr = len(tr)
same_cam_pairs = int(sum(c*(c-1)//2 for c in np.bincount(ctr)))

def iou_block(A, B):
    ax1, ay1 = A[:, 0:1].astype(np.float32), A[:, 1:2].astype(np.float32)
    ax2, ay2 = ax1 + A[:, 2:3], ay1 + A[:, 3:4]
    bx1, by1 = B[:, 0].astype(np.float32), B[:, 1].astype(np.float32)
    bx2, by2 = bx1 + B[:, 2], by1 + B[:, 3]
    ix = np.maximum(0, np.minimum(ax2, bx2[None, :]) - np.maximum(ax1, bx1[None, :]))
    iy = np.maximum(0, np.minimum(ay2, by2[None, :]) - np.maximum(ay1, by1[None, :]))
    inter = ix*iy
    return inter / (A[:, 2:3].astype(np.float32)*A[:, 3:4] + (B[:, 2]*B[:, 3]).astype(np.float32)[None, :] - inter)

TH_G = [0.3, 0.4, 0.5, 0.6, 0.7]
TH_I = [0.7, 0.8, 0.9, 0.95]
B = 384
gt = G[tr]; vf = Vff[tr]; bb = bbox[tr]
cg = np.zeros((len(TH_G), 2), np.int64)   # [flagged, flagged&same_cam]
ci = np.zeros((len(TH_I), 2), np.int64)
cc_or = np.zeros((len(TH_G), 2), np.int64)  # grad>=t OR ff>=0.90
for i0 in range(0, n_tr, B):
    Sg = gt[i0:i0+B] @ gt.T
    Sf = vf[i0:i0+B] @ vf.T
    Io = iou_block(bb[i0:i0+B], bb)
    samec = ctr[i0:i0+B, None] == ctr[None, :]
    iu = np.arange(i0, min(i0+B, n_tr))[:, None] < np.arange(n_tr)[None, :]
    ff = (Sf >= 0.90) & iu
    for ti, t in enumerate(TH_G):
        m = (Sg >= t) & iu
        cg[ti] += [m.sum(), (m & samec).sum()]
        mo = m | ff
        cc_or[ti] += [mo.sum(), (mo & samec).sum()]
    for ti, t in enumerate(TH_I):
        m = (Io >= t) & iu
        ci[ti] += [m.sum(), (m & samec).sum()]
out["pairlevel_grad96"] = {str(t): {"flagged": int(cg[ti][0]),
    "precision_same_cam": round(cg[ti][1]/cg[ti][0], 3) if cg[ti][0] else None,
    "recall_same_cam": round(cg[ti][1]/same_cam_pairs, 4)} for ti, t in enumerate(TH_G)}
out["pairlevel_grad96_or_ff32"] = {str(t): {"flagged": int(cc_or[ti][0]),
    "precision_same_cam": round(cc_or[ti][1]/cc_or[ti][0], 3) if cc_or[ti][0] else None,
    "recall_same_cam": round(cc_or[ti][1]/same_cam_pairs, 4)} for ti, t in enumerate(TH_G)}
out["pairlevel_iou_alone"] = {str(t): {"flagged": int(ci[ti][0]),
    "precision_same_cam": round(ci[ti][1]/ci[ti][0], 3) if ci[ti][0] else None,
    "recall_same_cam": round(ci[ti][1]/same_cam_pairs, 4)} for ti, t in enumerate(TH_I)}

# --- 2) same-vid пары: цель и цена инструмента ---
byvid = {}
for i in tr:
    byvid.setdefault(int(vid[i]), []).append(int(i))
sv_pairs = [(a, b) for g in byvid.values() for k, a in enumerate(g) for b in g[k+1:]]
pa = np.array([p[0] for p in sv_pairs]); pb = np.array([p[1] for p in sv_pairs])
samec = cam[pa] == cam[pb]
ffv = (Vff[pa]*Vff[pb]).sum(1)
ggv = (G[pa]*G[pb]).sum(1)
def iou_vec(A, Bv):
    x1 = np.maximum(A[:, 0], Bv[:, 0]); y1 = np.maximum(A[:, 1], Bv[:, 1])
    x2 = np.minimum(A[:, 0]+A[:, 2], Bv[:, 0]+Bv[:, 2]); y2 = np.minimum(A[:, 1]+A[:, 3], Bv[:, 1]+Bv[:, 3])
    inter = np.maximum(0, x2-x1)*np.maximum(0, y2-y1)
    return inter/(A[:, 2]*A[:, 3]+Bv[:, 2]*Bv[:, 3]-inter)
iouv = iou_vec(bbox[pa].astype(np.float32), bbox[pb].astype(np.float32))
GT, FT = 0.5, 0.90   # рабочая точка: grad96>=0.5, ff32>=0.90 (высокая точность по п.1)
flag_no_crop = (ggv >= GT) | (ffv >= FT)
out["same_vid_pairs"] = {"n_total": len(sv_pairs), "n_same_cam": int(samec.sum()),
    "n_diff_cam": int((~samec).sum())}
out["instrument_no_crop@grad0.5_or_ff0.90"] = {
    "recall_on_same_cam_same_vid": round(float(flag_no_crop[samec].mean()), 3),
    "false_flag_on_diff_cam_same_vid": round(float(flag_no_crop[~samec].mean()), 4)}
# компоненты по отдельности
out["signal_rates_same_vid"] = {
    "same_cam": {"ff32_ge_0.90": round(float((ffv[samec] >= .90).mean()), 3),
                 "grad96_ge_0.5": round(float((ggv[samec] >= .5).mean()), 3),
                 "iou_ge_0.8": round(float((iouv[samec] >= .8).mean()), 3)},
    "diff_cam": {"ff32_ge_0.90": round(float((ffv[~samec] >= .90).mean()), 4),
                 "grad96_ge_0.5": round(float((ggv[~samec] >= .5).mean()), 4),
                 "iou_ge_0.8": round(float((iouv[~samec] >= .8).mean()), 4)}}

# crop64 на подвыборке same-vid пар с IoU>=0.8 (вклад «парковочного» сигнала)
cache = {}
def crop64(i):
    if i not in cache:
        if len(cache) > 600: cache.clear()
        with Image.open(os.path.join(DATA, "images", files[i])) as im:
            im.draft("L", (960, 540))
            g = im.convert("L")
            sx, sy = g.width/W, g.height/H
            x, y, w, h = bbox[i]
            c = g.crop((int(x*sx), int(y*sy), int(np.ceil((x+w)*sx)), int(np.ceil((y+h)*sy))))
            v = np.asarray(c.resize((64, 64), Image.BILINEAR), np.float32).ravel()
            v -= v.mean(); v /= (np.linalg.norm(v)+1e-9)
            cache[i] = v
    return cache[i]
mi = np.where((iouv >= 0.8) & ~flag_no_crop)[0]   # что добавит crop-сигнал сверх остальных
sel_sc = rng.permutation(mi[samec[mi]])[:400]
sel_dc = rng.permutation(mi[~samec[mi]])[:400]
cs_sc = np.array([float(crop64(pa[k]) @ crop64(pb[k])) for k in sel_sc])
cs_dc = np.array([float(crop64(pa[k]) @ crop64(pb[k])) for k in sel_dc])
out["crop_term_sample"] = {
    "same_cam_iou0.8_not_yet_flagged_n": int((samec & (iouv >= 0.8) & ~flag_no_crop).sum()),
    "sample_n": len(sel_sc), "share_crop_ge_0.8": round(float((cs_sc >= 0.8).mean()), 3) if len(sel_sc) else None,
    "diff_cam_iou0.8_not_yet_flagged_n": int((~samec & (iouv >= 0.8) & ~flag_no_crop).sum()),
    "sample_dc_n": len(sel_dc), "share_dc_crop_ge_0.8": round(float((cs_dc >= 0.8).mean()), 3) if len(sel_dc) else None}
# полный инструмент (crop-часть по выборочной оценке)
p_sc = float((cs_sc >= 0.8).mean()) if len(sel_sc) else 0.0
p_dc = float((cs_dc >= 0.8).mean()) if len(sel_dc) else 0.0
add_sc = p_sc * float((samec & (iouv >= 0.8) & ~flag_no_crop).sum()) / max(1, samec.sum())
add_dc = p_dc * float((~samec & (iouv >= 0.8) & ~flag_no_crop).sum()) / max(1, (~samec).sum())
out["instrument_full_estimate"] = {
    "recall_on_same_cam_same_vid": round(float(flag_no_crop[samec].mean()) + add_sc, 3),
    "false_flag_on_diff_cam_same_vid": round(float(flag_no_crop[~samec].mean()) + add_dc, 4)}

# --- 3) mutual-kNN кластеризация grad96 ---
def knn_graph(V, sub, k=5, tmin=0.35):
    Vs = V[sub]; n = len(sub)
    nbr = np.zeros((n, k), np.int32); nbs = np.zeros((n, k), np.float32)
    for i0 in range(0, n, B):
        S = Vs[i0:i0+B] @ Vs.T
        for r in range(S.shape[0]):
            S[r, i0+r] = -2
        part = np.argpartition(-S, k, axis=1)[:, :k]
        nbr[i0:i0+B] = part; nbs[i0:i0+B] = np.take_along_axis(S, part, 1)
    edges = []
    nbrsets = [set(map(int, nbr[i])) for i in range(n)]
    for i in range(n):
        for j, s in zip(nbr[i], nbs[i]):
            j = int(j)
            if s >= tmin and i < j and i in nbrsets[j]:
                edges.append((i, j))
            elif s >= tmin and j < i and i in nbrsets[j]:
                edges.append((j, i))
    return list(set(edges))
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
    key = pred.astype(np.int64)*(truth.max()+1)+truth
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
            "purity": round(pur/len(truth), 3), "n_clusters": int(pred.max()+1),
            "largest": int(np.bincount(pred).max())}
mres = {}
for tmin in [0.3, 0.4, 0.5]:
    lab = components(n_tr, knn_graph(G, tr, 5, tmin))
    mres[f"t{tmin}"] = pair_metrics(lab, ctr)
out["mutual_knn_grad96_train"] = mres

# тест: кластеры при t=0.4
te = np.where(split != 0)[0]
lab_te = components(len(te), knn_graph(G, te, 5, 0.4))
cnt = np.bincount(lab_te)
spl2 = split[te]
gal_in = np.bincount(lab_te[spl2 == 2], minlength=lab_te.max()+1)
qlab = lab_te[spl2 == 1]
out["test_clusters_grad96_t0.4"] = {
    "n_clusters": int(lab_te.max()+1), "n_ge2": int((cnt >= 2).sum()),
    "largest": int(cnt.max()), "singletons": int((cnt == 1).sum()),
    "queries_with_gallery_same_cluster": int((gal_in[qlab] > 0).sum())}
with open(os.path.join(HERE, "test_scene_clusters.csv"), "w", newline="") as f:
    wcsv = csv.writer(f)
    wcsv.writerow(["image_id", "split", "scene_cluster"])
    for k, i in enumerate(te):
        wcsv.writerow([os.path.splitext(files[i])[0], ["", "query", "gallery"][split[i]], int(lab_te[k])])

with open(os.path.join(HERE, "out_s06d.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
