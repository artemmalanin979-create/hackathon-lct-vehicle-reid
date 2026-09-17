#!/usr/bin/env python3
"""job_43: ain_v1 / ain_v2 на подмножестве crops_208 (dev-срез ∪ tune-протокол) — узел разделён с
чужим процессом, полный прогон 7248×2 не окупается: дальше эти векторы нужны только на этих строках.
Остальные строки заполняются NaN (чтобы случайное использование было заметно)."""
import csv, json, sys, time, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import onnxruntime as ort

HOME = Path.home() / "lct-reid"; REPO = HOME / "repo"; S = REPO / "04-solution"
JOB = HOME / "jobs/job_43"; OUT = JOB / "out"
MODELS = {"ainv1": HOME / "weights/ain_v1_s1.onnx", "ainv2": HOME / "weights/ain_v2_s2.onnx"}
BATCH, THREADS = 32, 4

vids = np.load(OUT / "train_meta_vehicle_id.npy"); cams = np.load(OUT / "train_meta_camera_id.npy")
ids = (OUT / "train_fit.ids").read_text().split()

# dev-срез (reid_train4.py::make_dev_split, SEED 20260916, 100 ид.)
uniq = np.unique(vids); rng = np.random.default_rng(20260916)
dev_ids = set(uniq[rng.permutation(len(uniq))[:100]].tolist())
dev_rows = np.flatnonzero(np.array([v in dev_ids for v in vids]))

# tune-протокол (s02_make_tune_protocol.py, seed 20260915) — только image_id
def read_meta(path):
    with open(path, newline="") as f: return list(csv.DictReader(f))
SEED = 20260915; N_GALLERY_IDS = 295; N_REFUSAL_IDS = 74; N_KNOWN_Q = 832; N_REFUSAL_Q = 278; MAX_Q_PER_ID = 3; MAX_REF_Q_PER_ID = 4
rng = np.random.default_rng(SEED)
rows = read_meta(S / "split/files/train_fit.csv")
by_vid = defaultdict(list)
for r in rows: by_vid[r["vehicle_id"]].append(r)
vids_s = sorted(by_vid, key=int)
chosen = rng.choice(len(vids_s), size=N_GALLERY_IDS + N_REFUSAL_IDS, replace=False)
paired_ids = [vids_s[i] for i in sorted(chosen[:N_GALLERY_IDS])]; refusal_ids = [vids_s[i] for i in sorted(chosen[N_GALLERY_IDS:])]
gallery, known_q = [], []
for vid in paired_ids:
    frames = sorted(by_vid[vid], key=lambda r: r["image_id"]); by_cam = defaultdict(list)
    for r in frames: by_cam[r["camera_id"]].append(r)
    leftovers = []
    for cam in sorted(by_cam):
        pick = rng.integers(len(by_cam[cam])); gallery.append(by_cam[cam][pick]); leftovers += [r for i, r in enumerate(by_cam[cam]) if i != pick]
    if leftovers:
        take = min(len(leftovers), MAX_Q_PER_ID); idx = rng.choice(len(leftovers), size=take, replace=False); known_q += [leftovers[i] for i in sorted(idx)]
refusal_q = []
for vid in refusal_ids:
    frames = sorted(by_vid[vid], key=lambda r: r["image_id"]); take = min(len(frames), MAX_REF_Q_PER_ID)
    idx = rng.choice(len(frames), size=take, replace=False); refusal_q += [frames[i] for i in sorted(idx)]
if len(known_q) > N_KNOWN_Q:
    keep = sorted(rng.choice(len(known_q), size=N_KNOWN_Q, replace=False)); known_q = [known_q[i] for i in keep]
if len(refusal_q) > N_REFUSAL_Q:
    keep = sorted(rng.choice(len(refusal_q), size=N_REFUSAL_Q, replace=False)); refusal_q = [refusal_q[i] for i in keep]
pos = {iid: i for i, iid in enumerate(ids)}
tune_rows = np.array([pos[r["image_id"]] for r in known_q + refusal_q + gallery])
subset = np.unique(np.concatenate([dev_rows, tune_rows]))
np.save(OUT / "subset_idx.npy", subset)
print(json.dumps({"dev_rows": int(len(dev_rows)), "tune_rows": int(len(tune_rows)), "subset": int(len(subset))}), flush=True)

crops = np.load(HOME / "data/crops_208.npy", mmap_mode="r")
info = {}
for tag, path in MODELS.items():
    o = ort.SessionOptions(); o.log_severity_level = 3; o.intra_op_num_threads = THREADS; o.inter_op_num_threads = 1
    s = ort.InferenceSession(str(path), sess_options=o, providers=["CPUExecutionProvider"])
    inp = s.get_inputs()[0]; dim = int(s.get_outputs()[0].shape[-1])
    emb = np.full((crops.shape[0], dim), np.nan, dtype=np.float32)
    t0 = time.perf_counter()
    for st in range(0, len(subset), BATCH):
        rows_i = subset[st:st + BATCH]
        batch = np.asarray(crops[rows_i], dtype=np.float32).transpose(0, 3, 1, 2)
        (e,) = s.run(None, {inp.name: batch})
        emb[rows_i] = e
        if st % (BATCH * 20) == 0:
            print(f"[{tag}] {st}/{len(subset)} {time.perf_counter()-t0:.0f}s", flush=True)
    el = time.perf_counter() - t0
    m = emb[subset].astype(np.float64); m /= np.linalg.norm(m, axis=1, keepdims=True); emb[subset] = m.astype(np.float32)
    np.save(OUT / f"emb_train_{tag}.npy", emb)
    info[tag] = {"model": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": int(len(subset)),
                 "elapsed_s": round(el, 1), "img_per_s": round(len(subset) / el, 1)}
    print(json.dumps({tag: info[tag]}), flush=True)
(OUT / "s02b_extract_info.json").write_text(json.dumps(info, indent=2) + "\n")
print("DONE", flush=True)
