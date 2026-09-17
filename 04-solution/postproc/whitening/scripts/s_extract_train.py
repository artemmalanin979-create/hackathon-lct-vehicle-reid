"""train_fit-эмбеддинги базовой OSNet-AIN из ~/lct-reid/data/crops_208.npy (7248×208×208×3 uint8).
Препроцессинг = baseline/scripts/extract_embeddings.py: RGB uint8 0..255 -> float32 -> NCHW -> ONNX -> L2.
Порядок crops_208 проверен против train_crops.npz (image_id явно = train_fit.csv): совпадение побайтно.
Узел без AVX2 и с чужим процессом на тех же ядрах: 1 поток ORT на процесс, без spin-wait, 2 шарда.
usage: s_extract_train.py <shard> <n_shards> | merge"""
import io, json, time, sys
from pathlib import Path
import numpy as np
from lib42 import REPO, SPLIT, OUT, read_meta

DATA = Path.home() / "lct-reid/data"
MODEL = REPO / "04-solution/service/model/osnet_ain_x1_0_vehicle_reid.onnx"
tf = read_meta(SPLIT / "train_fit.csv")

if sys.argv[1] == "merge":
    parts = sorted(OUT.glob("train_fit_osnet_part*.npy"))
    out = np.concatenate([np.load(p) for p in parts]); assert out.shape == (len(tf), 512), out.shape
    out = (out.astype(np.float64) / np.linalg.norm(out.astype(np.float64), axis=1, keepdims=True)).astype(np.float32)
    np.save(OUT / "train_fit_osnet.npy", out)
    (OUT / "train_fit_osnet.ids").write_text("".join(r["image_id"] + "\n" for r in tf))
    meta = [json.load(open(p)) for p in sorted(OUT.glob("train_fit_extract_part*.json"))]
    json.dump({"rows": int(out.shape[0]), "parts": meta, "model": MODEL.name,
               "note": "crops_208 = train_crops.npz(256,JPEG q95) -> resize 208 BILINEAR; не прямой resize из кадра"},
              open(OUT / "train_fit_extract.json", "w"), indent=1, ensure_ascii=False)
    print("merged", out.shape); sys.exit(0)

import onnxruntime as ort
from PIL import Image
shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
crops = np.load(DATA / "crops_208.npy", mmap_mode="r")
assert crops.shape == (len(tf), 208, 208, 3), crops.shape
if shard == 0:
    z = np.load(DATA / "train_crops.npz")
    assert [str(x) for x in z["image_id"]] == [r["image_id"] for r in tf]
    data, off = z["data"], z["offsets"]; rng = np.random.default_rng(0); same, other = [], []
    for i in rng.choice(len(tf), 24, replace=False):
        im = Image.open(io.BytesIO(data[off[i]:off[i + 1]].tobytes())).convert("RGB").resize((208, 208), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32)
        same.append(np.abs(a - crops[i].astype(np.float32)).mean()); other.append(np.abs(a - crops[(i + 1) % len(tf)].astype(np.float32)).mean())
    print("order check: same", float(np.max(same)), "next", float(np.min(other)), flush=True)
    assert np.max(same) < 8 and np.min(other) > np.max(same)
lo, hi = (len(tf) * shard) // n_shards, (len(tf) * (shard + 1)) // n_shards
o = ort.SessionOptions(); o.log_severity_level = 3; o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
o.add_session_config_entry("session.intra_op.allow_spinning", "0")
sess = ort.InferenceSession(str(MODEL), sess_options=o, providers=["CPUExecutionProvider"]); inp = sess.get_inputs()[0].name
out = np.empty((hi - lo, 512), np.float32); t0 = time.perf_counter(); B = 8
for s in range(lo, hi, B):
    e = min(s + B, hi)
    batch = np.ascontiguousarray(crops[s:e]).astype(np.float32).transpose(0, 3, 1, 2)
    out[s - lo:e - lo] = sess.run(None, {inp: batch})[0]
    if (s - lo) % 400 == 0:
        print(f"shard {shard}: {s - lo}/{hi - lo} {time.perf_counter() - t0:.0f}s", flush=True)
el = time.perf_counter() - t0
np.save(OUT / f"train_fit_osnet_part{shard}.npy", out)
json.dump({"shard": shard, "lo": lo, "hi": hi, "elapsed_s": round(el, 1), "per_img_ms": round(1000 * el / (hi - lo), 1)},
          open(OUT / f"train_fit_extract_part{shard}.json", "w"))
print("done shard", shard, el, flush=True)
