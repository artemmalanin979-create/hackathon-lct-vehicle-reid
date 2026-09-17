#!/usr/bin/env python3
"""job_43, шаг 1-bis: проверки на своих ONNX-векторах train_fit (crops_208.npy).

(1) dev-срез обучения (make_dev_split из attempt-2/reid_train4.py, SEED 20260916, 100 идентичностей):
    контур должен дать dev mAP ≈ 0,71358 (база), ≈ 0,7558 (ain_v2, эпоха 15), ≈ 0,5584 (ain_v1, эпоха 5) —
    проверка тождества ONNX-весов на узле тем моделям, чьи числа в отчёте.
(2) tune-протокол postproc (s02_make_tune_protocol.py, seed 20260915; sha256 ↔ tune/manifest.json):
    база должна дать ≈ 0,5877 / 0,5421 (косинус) и 0,6360 / 0,6082 (rr 6,3,0,3) — проверка препроцессинга
    кропов (crops_208 = JPEG q95 + двойной resize, ожидается малое отклонение от кадрового пути).
(3) тот же протокол для ain_v1 / ain_v2 / ансамбля: метрики на обучающих идентичностях (диагностика запоминания).
"""
import csv, hashlib, io, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HOME = Path.home() / "lct-reid"; REPO = HOME / "repo"; S = REPO / "04-solution"
JOB = HOME / "jobs/job_43"; OUT = JOB / "out"
sys.path.insert(0, str(S / "eval")); sys.path.insert(0, str(S / "postproc/scripts"))
from reid_metrics import evaluate, scores_from_embeddings  # noqa
from common import rerank  # noqa

def l2(m):
    m = m.astype(np.float64); return m / np.linalg.norm(m, axis=1, keepdims=True)
E = {t: l2(np.load(OUT / f"emb_train_{t}.npy")) for t in ("osnet", "ainv1", "ainv2")}
E["avg"] = l2(E["osnet"] + E["ainv2"])
vids = np.load(OUT / "train_meta_vehicle_id.npy"); cams = np.load(OUT / "train_meta_camera_id.npy")
ids = (OUT / "train_fit.ids").read_text().split()
res = {}

# ---------- (1) dev-срез: дословно make_dev_split из reid_train4.py
def make_dev_split(vids, cams, n_dev_ids, seed):
    uniq = np.unique(vids)
    rng = np.random.default_rng(seed)
    dev_ids = set(uniq[rng.permutation(len(uniq))[:n_dev_ids]].tolist())
    is_dev = np.array([v in dev_ids for v in vids])
    train_idx = np.flatnonzero(~is_dev)
    q_idx, g_idx = [], []
    for v in sorted(dev_ids):
        rows = np.flatnonzero(vids == v)
        g_here, q_here = [], []
        for c in np.unique(cams[rows]):
            grp = rows[cams[rows] == c]
            g_here.append(int(grp[0])); q_here.extend(int(i) for i in grp[1:])
        if not q_here and len(g_here) >= 2:
            q_here.append(g_here.pop())
        q_idx.extend(q_here); g_idx.extend(g_here)
    return train_idx, np.array(sorted(q_idx)), np.array(sorted(g_idx))

train_idx, q_idx, g_idx = make_dev_split(vids, cams, 100, 20260916)
res["dev_split"] = {"n_train": int(len(train_idx)), "n_query": int(len(q_idx)), "n_gallery": int(len(g_idx)),
                    "expected": {"osnet": 0.71358, "ainv2": 0.7558, "ainv1": 0.5584, "n_query": 353, "n_gallery": 241}}
def dev_metrics(emb):
    q, g = emb[q_idx], emb[g_idx]
    r = evaluate(scores_from_embeddings(q, g, metric="cosine"),
                 query_ids=[int(vids[i]) for i in q_idx], gallery_ids=[int(vids[i]) for i in g_idx],
                 query_cameras=[int(cams[i]) for i in q_idx], gallery_cameras=[int(cams[i]) for i in g_idx],
                 known_absent=np.zeros(len(q_idx), dtype=bool), threshold=0.0, camera_policy="market", refusal_mode="presence")
    f = r["ranking_full_gallery"]
    return {"mAP": round(f["mAP"], 5), "Rank-1": round(f["Rank-1"], 5), "n": f["num_valid_queries"]}
for t in ("osnet", "ainv1", "ainv2", "avg"):
    res["dev_split"][t] = dev_metrics(E[t])
    print(f"[dev] {t:6s} {res['dev_split'][t]}", flush=True)

# ---------- (2) tune-протокол: дословно s02_make_tune_protocol.py (в свой каталог)
def read_meta(path):
    with open(path, newline="") as f: return list(csv.DictReader(f))
SEED = 20260915; N_GALLERY_IDS = 295; N_REFUSAL_IDS = 74; N_KNOWN_Q = 832; N_REFUSAL_Q = 278; MAX_Q_PER_ID = 3; MAX_REF_Q_PER_ID = 4
rng = np.random.default_rng(SEED)
rows = read_meta(S / "split/files/train_fit.csv")
by_vid = defaultdict(list)
for r in rows: by_vid[r["vehicle_id"]].append(r)
vids_s = sorted(by_vid, key=int)
chosen = rng.choice(len(vids_s), size=N_GALLERY_IDS + N_REFUSAL_IDS, replace=False)
paired_ids = [vids_s[i] for i in sorted(chosen[:N_GALLERY_IDS])]
refusal_ids = [vids_s[i] for i in sorted(chosen[N_GALLERY_IDS:])]
gallery, known_q = [], []
for vid in paired_ids:
    frames = sorted(by_vid[vid], key=lambda r: r["image_id"])
    by_cam = defaultdict(list)
    for r in frames: by_cam[r["camera_id"]].append(r)
    leftovers = []
    for cam in sorted(by_cam):
        pick = rng.integers(len(by_cam[cam]))
        gallery.append(by_cam[cam][pick]); leftovers += [r for i, r in enumerate(by_cam[cam]) if i != pick]
    if leftovers:
        take = min(len(leftovers), MAX_Q_PER_ID); idx = rng.choice(len(leftovers), size=take, replace=False)
        known_q += [leftovers[i] for i in sorted(idx)]
refusal_q = []
for vid in refusal_ids:
    frames = sorted(by_vid[vid], key=lambda r: r["image_id"])
    take = min(len(frames), MAX_REF_Q_PER_ID); idx = rng.choice(len(frames), size=take, replace=False)
    refusal_q += [frames[i] for i in sorted(idx)]
if len(known_q) > N_KNOWN_Q:
    keep = sorted(rng.choice(len(known_q), size=N_KNOWN_Q, replace=False)); known_q = [known_q[i] for i in keep]
if len(refusal_q) > N_REFUSAL_Q:
    keep = sorted(rng.choice(len(refusal_q), size=N_REFUSAL_Q, replace=False)); refusal_q = [refusal_q[i] for i in keep]
tune = JOB / "tune"; tune.mkdir(exist_ok=True)
qcols = ["image_id", "x", "y", "w", "h", "vehicle_id", "camera_id", "has_mate"]; gcols = qcols[:-1]
with open(tune / "tune_query.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(qcols)
    for r in known_q: w.writerow([r[c] for c in gcols] + ["1"])
    for r in refusal_q: w.writerow([r[c] for c in gcols] + ["0"])
with open(tune / "tune_gallery.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(gcols)
    for r in gallery: w.writerow([r[c] for c in gcols])
sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [tune / "tune_query.csv", tune / "tune_gallery.csv"]}
manifest = json.loads((S / "postproc/tune/manifest.json").read_text())
res["tune_protocol"] = {"sha256_matches_manifest": sha == manifest["sha256"], "sha256": sha,
                        "n_query": len(known_q) + len(refusal_q), "n_gallery": len(gallery),
                        "expected_osnet": {"cos": [0.5876891496603117, 0.5420673076923077], "rr": [0.6360220173715281, 0.6081730769230769]}}
print("[tune] sha256 == manifest:", sha == manifest["sha256"], flush=True)

qm = read_meta(tune / "tune_query.csv"); gm = read_meta(tune / "tune_gallery.csv")
pos = {iid: i for i, iid in enumerate(ids)}
qi = np.array([pos[r["image_id"]] for r in qm]); gi = np.array([pos[r["image_id"]] for r in gm])
def run(scores, fake_cameras=False):
    qc = [f"q:{i}" for i in range(len(qm))] if fake_cameras else [r["camera_id"] for r in qm]
    gc = [f"g:{i}" for i in range(len(gm))] if fake_cameras else [r["camera_id"] for r in gm]
    r = evaluate(scores, query_ids=[r["vehicle_id"] for r in qm], gallery_ids=[r["vehicle_id"] for r in gm],
                 query_cameras=qc, gallery_cameras=gc, known_absent=np.array([r["has_mate"] == "0" for r in qm]),
                 threshold=0.0, camera_policy="market", refusal_mode="presence")
    f = r["ranking_full_gallery"]
    return {"mAP": f["mAP"], "Rank-1": f["Rank-1"], "Rank-5": f["Rank-5"], "mINP": f["mINP"], "valid_queries": f["num_valid_queries"]}
for t in ("osnet", "ainv1", "ainv2", "avg"):
    q, g = E[t][qi], E[t][gi]
    sc = scores_from_embeddings(q, g, metric="cosine")
    d, _ = rerank(q, g, 6, 3, 0.3)
    res["tune_protocol"][t] = {"cos": run(sc), "cos_nocam": run(sc, True), "rr": run(-d), "rr_nocam": run(-d, True)}
    c, rr = res["tune_protocol"][t]["cos"], res["tune_protocol"][t]["rr"]
    print(f"[tune] {t:6s} cos {c['mAP']:.4f}/{c['Rank-1']:.4f}  rr {rr['mAP']:.4f}/{rr['Rank-1']:.4f}  "
          f"gap(cos) {res['tune_protocol'][t]['cos_nocam']['mAP']-c['mAP']:+.4f}", flush=True)
# dev-идентичности внутри tune-протокола (для ain_v2 они не были в обучении)
dev_set = set(int(v) for v in np.unique(vids[q_idx]))
res["tune_protocol"]["tune_identities_in_dev"] = int(sum(int(r["vehicle_id"]) in dev_set for r in gm))
# ---------- (4) честный подбор веса ансамбля на dev-срезе (100 идентичностей, в обучение ain_v2 не входили)
def dev_metrics_scores(q, g, rr=False):
    sc = -rerank(q, g, 6, 3, 0.3)[0] if rr else scores_from_embeddings(q, g, metric="cosine")
    r = evaluate(sc, query_ids=[int(vids[i]) for i in q_idx], gallery_ids=[int(vids[i]) for i in g_idx],
                 query_cameras=[int(cams[i]) for i in q_idx], gallery_cameras=[int(cams[i]) for i in g_idx],
                 known_absent=np.zeros(len(q_idx), dtype=bool), threshold=0.0, camera_policy="market", refusal_mode="presence")
    f = r["ranking_full_gallery"]
    return {"mAP": f["mAP"], "Rank-1": f["Rank-1"]}
res["dev_weight_sweep"] = {}
for w in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
    row = {}
    for kind in ("avg", "cat"):
        if kind == "avg":
            emb = l2(E["osnet"] + w * E["ainv2"])
        else:
            emb = np.concatenate([E["osnet"], w * E["ainv2"]], axis=1)
        q, g = emb[q_idx], emb[g_idx]
        row[kind] = {"cos": dev_metrics_scores(q, g), "rr": dev_metrics_scores(q, g, rr=True)}
    res["dev_weight_sweep"][str(w)] = row
    print(f"[dev-sweep] w={w:<4} avg cos {row['avg']['cos']['mAP']:.4f} rr {row['avg']['rr']['mAP']:.4f} | "
          f"cat cos {row['cat']['cos']['mAP']:.4f} rr {row['cat']['rr']['mAP']:.4f}", flush=True)
q, g = E["ainv2"][q_idx], E["ainv2"][g_idx]
res["dev_weight_sweep"]["ainv2_only"] = {"cos": dev_metrics_scores(q, g), "rr": dev_metrics_scores(q, g, rr=True)}
print("[dev-sweep] ain_v2 only:", res["dev_weight_sweep"]["ainv2_only"], flush=True)
(OUT / "s03_train_checks.json").write_text(json.dumps(res, indent=2) + "\n")
print("DONE")
