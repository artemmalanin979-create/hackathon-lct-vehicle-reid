#!/usr/bin/env python3
"""s05: почти-дубликаты кадров (кадры одного проезда) внутри и между выборками.
Признак кадра = полная миниатюра 32x32 (thumbs32.npy), zero-mean/unit-norm, косинус.
 - калибровка порога на train: пары same(vid,cam) [один проезд] vs same-cam-diff-vid vs diff-cam;
 - блочный проход все-на-все (11416^2), сбор пар cos>=0.90, счётчики по категориям выборок;
 - компоненты связности при основном пороге;
 - для каждого query: есть ли в gallery кандидат "тот же кадр/тот же проезд";
 - проверка пары первых строк test_query/test_gallery;
 - совпадения bbox с точностью +-5 px у разных image_id;
 - верификация кросс-сплитовых пар в среднем разрешении (240x135): доля изменившихся
   пикселей вне объединения bbox (same-pass ~ мало, другой проезд ~ много).
Память: блочно, пиковое ~ десятки МБ. Выход: out_s05.json. Запуск: python3 s05_neardup.py"""
import csv, json, os
import numpy as np
from PIL import Image

DATA = "/home/artem/projects/hackathon-lct-vehicle-reid/data"
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)
W, H = 1920, 1080
COLLECT_T = 0.90
GRID = [0.90, 0.95, 0.98, 0.995]
EDGE_CAP = 20_000_000

files = json.load(open(os.path.join(HERE, "scan_files.json")))
idx = {os.path.splitext(f)[0]: i for i, f in enumerate(files)}
N = len(files)
md5 = {}
for line in open(os.path.join(HERE, "scan_meta.jsonl")):
    d = json.loads(line)
    md5[os.path.splitext(d["f"])[0]] = d["md5"]

split = np.full(N, -1, np.int8)  # 0 train, 1 query, 2 gallery
SPL = ["train", "query", "gallery"]
vid = np.full(N, -1, np.int32); cam = np.full(N, -1, np.int32)
bbox = np.zeros((N, 4), np.int32)
row_order = {}
for si, (fn, sname) in enumerate([("train.csv", 0), ("test_query.csv", 1), ("test_gallery.csv", 2)]):
    for rn, r in enumerate(csv.DictReader(open(os.path.join(DATA, fn)))):
        i = idx[r["image_id"]]
        split[i] = sname
        bbox[i] = [int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])]
        row_order[i] = rn
        if sname == 0:
            vid[i] = int(r["vehicle_id"]); cam[i] = int(r["camera_id"])
assert (split >= 0).all()

T = np.load(os.path.join(HERE, "thumbs32.npy")).astype(np.float32)
V = T - T.mean(1, keepdims=True)
V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)

out = {}

# --- 1) калибровка на train ---
groups = {}
for i in np.where(split == 0)[0]:
    groups.setdefault((vid[i], cam[i]), []).append(i)
same_pass = [(a, b) for g in groups.values() if len(g) > 1 for k, a in enumerate(g) for b in g[k+1:]]
sp = np.array([float(V[a] @ V[b]) for a, b in same_pass])
tr_idx = np.where(split == 0)[0]
def sample_pairs(cond, n):
    got = []
    while len(got) < n:
        a = rng.choice(tr_idx, n * 2); b = rng.choice(tr_idx, n * 2)
        m = (a != b) & cond(a, b)
        got.extend(zip(a[m], b[m]))
    return got[:n]
sc = np.array([float(V[a] @ V[b]) for a, b in
               sample_pairs(lambda a, b: (cam[a] == cam[b]) & (vid[a] != vid[b]), 30000)])
dc = np.array([float(V[a] @ V[b]) for a, b in
               sample_pairs(lambda a, b: cam[a] != cam[b], 30000)])
qs = lambda x: {p: round(float(np.quantile(x, q)), 3) for p, q in
                [("p50", .5), ("p90", .9), ("p99", .99), ("p999", .999), ("max", 1.0)]}
out["calibration_fullframe_cos"] = {
    "same_vid_cam_pairs_n": len(sp), "same_vid_cam": qs(sp),
    "same_vid_cam_share_ge": {str(t): round(float((sp >= t).mean()), 3) for t in GRID},
    "same_cam_diff_vid_sample": qs(sc),
    "same_cam_diff_vid_share_ge": {str(t): round(float((sc >= t).mean()), 4) for t in GRID},
    "diff_cam_sample": qs(dc),
    "diff_cam_share_ge": {str(t): round(float((dc >= t).mean()), 5) for t in GRID}}

# --- 2) блочный все-на-все, сбор пар cos>=COLLECT_T ---
ei, ej, es = [], [], []
n_edges = 0
B = 512
for i0 in range(0, N, B):
    S = V[i0:i0+B] @ V.T
    r, c = np.nonzero(S >= COLLECT_T)
    keep = (r + i0) < c
    r, c = r[keep], c[keep]
    ei.append((r + i0).astype(np.int32)); ej.append(c.astype(np.int32))
    es.append(S[r, c].astype(np.float32))
    n_edges += len(r)
    if n_edges > EDGE_CAP:
        raise SystemExit("edge cap exceeded, raise COLLECT_T")
ei = np.concatenate(ei); ej = np.concatenate(ej); es = np.concatenate(es)
out["pairs_ge_collect_threshold"] = int(len(ei))

cat = split[ei] * 3 + split[ej]  # категории (упорядочены i<j по индексу файла, не по сплиту)
def cat_name(a, b):
    x, y = sorted([a, b])
    return f"{SPL[x]}+{SPL[y]}"
counts = {}
for t in GRID:
    m = es >= t
    cc = {}
    for a in range(3):
        for b in range(3):
            mm = m & (split[ei] == a) & (split[ej] == b)
            if mm.sum():
                cc[cat_name(a, b)] = cc.get(cat_name(a, b), 0) + int(mm.sum())
    # для train+train: разбивка по истине
    mt = m & (split[ei] == 0) & (split[ej] == 0)
    tt = {"same_vid_cam": int((mt & (vid[ei] == vid[ej]) & (cam[ei] == cam[ej])).sum()),
          "same_cam_diff_vid": int((mt & (cam[ei] == cam[ej]) & (vid[ei] != vid[ej])).sum()),
          "diff_cam_same_vid": int((mt & (cam[ei] != cam[ej]) & (vid[ei] == vid[ej])).sum()),
          "diff_cam_diff_vid": int((mt & (cam[ei] != cam[ej]) & (vid[ei] != vid[ej])).sum())}
    counts[str(t)] = {"by_split_pair": cc, "train_train_breakdown": tt}
out["neardup_pair_counts"] = counts

# --- 3) компоненты при основном пороге 0.98 ---
MAIN_T = 0.98
parent = np.arange(N)
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for a, b in zip(ei[es >= MAIN_T], ej[es >= MAIN_T]):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb
comp = {}
for i in range(N):
    comp.setdefault(find(i), []).append(i)
multi = [v for v in comp.values() if len(v) > 1]
sizes = sorted((len(v) for v in multi), reverse=True)
out["components_at_0.98"] = {
    "n_multi_groups": len(multi), "n_frames_in_multi": int(sum(sizes)),
    "size_hist": {str(s): sizes.count(s) for s in sorted(set(sizes))},
    "groups_with_mixed_splits": int(sum(1 for v in multi if len(set(split[v])) > 1)),
    "mixed_split_patterns": {}}
pat = {}
for v in multi:
    ss = sorted(set(int(s) for s in split[v]))
    if len(ss) > 1:
        k = "+".join(SPL[s] for s in ss)
        pat[k] = pat.get(k, 0) + 1
out["components_at_0.98"]["mixed_split_patterns"] = pat

# --- 4) query -> gallery: тот же кадр / тот же проезд ---
def iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2 = min(b1[0]+b1[2], b2[0]+b2[2]); y2 = min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    return inter / (b1[2]*b1[3] + b2[2]*b2[3] - inter)
qg = {}
m = ((split[ei] == 1) & (split[ej] == 2)) | ((split[ei] == 2) & (split[ej] == 1))
for a, b, s in zip(ei[m], ej[m], es[m]):
    q, g = (a, b) if split[a] == 1 else (b, a)
    qg.setdefault(int(q), []).append((float(s), int(g)))
for t in GRID:
    nq = sum(1 for v in qg.values() if any(s >= t for s, _ in v))
    out.setdefault("queries_with_gallery_neardup", {})[str(t)] = nq
same_frame_qg = [(q, g) for q, v in qg.items() for s, g in v if md5[os.path.splitext(files[q])[0]] == md5[os.path.splitext(files[g])[0]]]
out["queries_with_gallery_same_md5"] = len(set(q for q, _ in same_frame_qg))

# --- 5) пара первых строк query/gallery ---
q0 = [i for i, r in row_order.items() if split[i] == 1 and r == 0][0]
g0 = [i for i, r in row_order.items() if split[i] == 2 and r == 0][0]
out["first_rows_pair"] = {
    "query_file": files[q0], "gallery_file": files[g0],
    "fullframe_cos": round(float(V[q0] @ V[g0]), 4),
    "bbox_iou": round(iou(bbox[q0], bbox[g0]), 3),
    "same_md5": md5[os.path.splitext(files[q0])[0]] == md5[os.path.splitext(files[g0])[0]]}

# --- 6) bbox-совпадения +-5px у разных image_id ---
order = np.argsort(bbox[:, 0], kind="stable")
bb = bbox[order]
hits = []
for k in range(N):
    x = bb[k, 0]; k2 = k + 1
    while k2 < N and bb[k2, 0] - x <= 5:
        if (abs(bb[k, 1]-bb[k2, 1]) <= 5 and abs(bb[k, 2]-bb[k2, 2]) <= 5
                and abs(bb[k, 3]-bb[k2, 3]) <= 5):
            hits.append((int(order[k]), int(order[k2])))
        k2 += 1
bb_cnt = {}
sim_of_hits = np.array([float(V[a] @ V[b]) for a, b in hits]) if hits else np.array([])
for a, b in hits:
    k = cat_name(split[a], split[b])
    bb_cnt[k] = bb_cnt.get(k, 0) + 1
out["bbox_match_pm5px"] = {"n_pairs": len(hits), "by_split_pair": bb_cnt,
    "share_with_fullframe_cos_ge_0.98": round(float((sim_of_hits >= 0.98).mean()), 3) if len(hits) else None,
    "share_with_fullframe_cos_ge_0.90": round(float((sim_of_hits >= 0.90).mean()), 3) if len(hits) else None}

# --- 7) верификация кросс-сплитовых пар (cos>=0.95) в 240x135 ---
cross = [(int(a), int(b), float(s)) for a, b, s in zip(ei, ej, es)
         if s >= 0.95 and split[a] != split[b]]
cross.sort(key=lambda x: -x[2])
cross = cross[:300]
cache = {}
def midres(i):
    if i not in cache:
        if len(cache) > 380: cache.clear()
        with Image.open(os.path.join(DATA, "images", files[i])) as im:
            cache[i] = np.asarray(im.convert("L").resize((240, 135), Image.BILINEAR), np.float32)
    return cache[i]
ver = []
for a, b, s in cross:
    A, B_ = midres(a), midres(b)
    diff = np.abs(A - B_) > 15
    mask = np.zeros((135, 240), bool)
    for i in (a, b):
        x, y, w, h = bbox[i]
        mask[int(y*135/H):int(np.ceil((y+h)*135/H)), int(x*240/W):int(np.ceil((x+w)*240/W))] = True
    outside = float(diff[~mask].mean()) if (~mask).sum() else None
    ver.append({"pair": [files[a], files[b]], "splits": [SPL[split[a]], SPL[split[b]]],
                "cos32": round(s, 3), "chg_outside_bbox": round(outside, 4),
                "bbox_iou": round(iou(bbox[a], bbox[b]), 3)})
chg = np.array([v["chg_outside_bbox"] for v in ver]) if ver else np.array([])
out["cross_split_verification_midres"] = {
    "n_pairs_checked": len(ver), "chg_outside_bbox_quantiles": qs(chg) if len(ver) else None,
    "n_same_pass_like_chg_le_0.05": int((chg <= 0.05).sum()) if len(ver) else 0,
    "pairs": ver[:40]}

with open(os.path.join(HERE, "out_s05.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in out if k != "cross_split_verification_midres"}, ensure_ascii=False, indent=1))
print("verification pairs:", out["cross_split_verification_midres"]["n_pairs_checked"],
      "chg_q:", out["cross_split_verification_midres"]["chg_outside_bbox_quantiles"],
      "same-pass-like:", out["cross_split_verification_midres"]["n_same_pass_like_chg_le_0.05"])
