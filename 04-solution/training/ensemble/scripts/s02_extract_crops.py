#!/usr/bin/env python3
"""job_43, шаг 1 (адаптированный): векторы трёх моделей на data/crops_208.npy.

crops_208.npy на узле = 7248 кропов train_fit (не 11 416), см. journal.md.
Препроцессинг как в baseline/scripts/extract_embeddings.py: uint8 RGB HWC -> float32
0..255 -> NCHW -> ONNX -> L2 (float64) -> float32. Ограничение: 4 потока ORT.
"""
import io, json, os, sys, time, hashlib
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image

HOME = Path.home() / "lct-reid"
REPO = HOME / "repo"
JOB = HOME / "jobs" / "job_43"
OUT = JOB / "out"
MODELS = {
    "osnet": REPO / "04-solution/service/model/osnet_ain_x1_0_vehicle_reid.onnx",
    "ainv1": HOME / "weights/ain_v1_s1.onnx",
    "ainv2": HOME / "weights/ain_v2_s2.onnx",
}
BATCH = 32
THREADS = 4

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def session(path):
    o = ort.SessionOptions()
    o.log_severity_level = 3
    o.intra_op_num_threads = THREADS
    o.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=o, providers=["CPUExecutionProvider"])

def main():
    crops = np.load(HOME / "data/crops_208.npy", mmap_mode="r")
    z = np.load(HOME / "data/train_crops.npz", allow_pickle=False)
    n = crops.shape[0]
    assert crops.shape == (n, 208, 208, 3) and crops.dtype == np.uint8, crops.shape
    info = {"n": int(n), "crops_shape": list(crops.shape)}

    # порядок = train_fit.csv?
    import csv
    with open(REPO / "04-solution/split/files/train_fit.csv", newline="") as f:
        tf = list(csv.DictReader(f))
    ids_csv = [r["image_id"] for r in tf]
    ids_npz = z["image_id"].tolist()
    info["order_equals_train_fit_csv"] = bool(ids_csv == ids_npz)
    info["vehicle_id_equals_csv"] = bool([int(r["vehicle_id"]) for r in tf] == z["vehicle_id"].tolist())
    info["camera_id_equals_csv"] = bool([int(r["camera_id"]) for r in tf] == z["camera_id"].tolist())
    np.save(OUT / "train_meta_vehicle_id.npy", z["vehicle_id"])
    np.save(OUT / "train_meta_camera_id.npy", z["camera_id"])
    (OUT / "train_fit.ids").write_text("".join(i + "\n" for i in ids_npz))

    # crops_208 воспроизводится из train_crops.npz (decode JPEG -> resize 208 BILINEAR)?
    data, off = z["data"], z["offsets"]
    maxdiff, meandiff = 0, 0.0
    K = 64
    for i in range(K):
        with Image.open(io.BytesIO(data[off[i]:off[i + 1]].tobytes())) as im:
            im = im.convert("RGB").resize((208, 208), Image.BILINEAR)
            a = np.asarray(im).astype(np.int16)
        d = np.abs(a - crops[i].astype(np.int16))
        maxdiff = max(maxdiff, int(d.max())); meandiff += float(d.mean()) / K
    info["crops_vs_npz_first64"] = {"max_abs_diff": maxdiff, "mean_abs_diff": meandiff}
    print(json.dumps(info), flush=True)

    for tag, path in MODELS.items():
        s = session(path)
        inp = s.get_inputs()[0]; outp = s.get_outputs()[0]
        dim = int(outp.shape[-1])
        emb = np.empty((n, dim), dtype=np.float32)
        t0 = time.perf_counter()
        for st in range(0, n, BATCH):
            batch = np.asarray(crops[st:st + BATCH], dtype=np.float32).transpose(0, 3, 1, 2)
            (e,) = s.run(None, {inp.name: batch})
            emb[st:st + len(batch)] = e
            if st % (BATCH * 40) == 0:
                print(f"[{tag}] {st}/{n} {time.perf_counter()-t0:.0f}s", flush=True)
        el = time.perf_counter() - t0
        raw_norms = np.linalg.norm(emb.astype(np.float64), axis=1)
        m = emb.astype(np.float64); m /= np.linalg.norm(m, axis=1, keepdims=True)
        np.save(OUT / f"emb_train_{tag}.npy", m.astype(np.float32))
        rec = {"model": str(path), "sha256": sha(path), "input": [inp.name, inp.shape],
               "output": [outp.name, outp.shape], "elapsed_s": round(el, 1),
               "img_per_s": round(n / el, 1), "raw_norm_min": float(raw_norms.min()),
               "raw_norm_max": float(raw_norms.max())}
        info[tag] = rec
        print(json.dumps({tag: rec}), flush=True)
        del s
    (OUT / "s02_extract_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
