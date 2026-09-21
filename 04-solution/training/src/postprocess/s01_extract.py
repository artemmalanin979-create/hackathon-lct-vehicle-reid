#!/usr/bin/env python3
"""job_48, шаг 1: векторы новой модели (job48_combined_s*.onnx) на валидации и train_fit.

Препроцессинг дословно baseline/scripts/extract_embeddings.py (bbox->RGB->208 BILINEAR
->0..255 NCHW->ONNX->L2) и job_45/s01 (train_fit из crops_208.npy). Контроль качества:
косинус новых векторов против векторов исходного OSNet на тех же кропах должен быть
высоким (дообучение LP-FT не должно было разрушить признак полностью); нижняя граница
фиксируется до просмотра метрик.

Выход: out/val_query_j48.npy, out/val_gallery_j48.npy, out/train_fit_j48.npy (+ids, info).
"""
import csv, json, sys, time, hashlib
from pathlib import Path
import numpy as np
import onnxruntime as ort

HOME = Path.home() / "lct-reid"
REPO = HOME / "repo"
JOB = HOME / "jobs" / "job_48"
OUT = JOB / "out"
MODEL = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(JOB.glob("job48_combined_s*.onnx"))[0]
BATCH, THREADS = 16, 1  # 1 поток: хост общий, многопоточный ORT под контенцией медленнее (job_44)


def session(path):
    o = ort.SessionOptions(); o.log_severity_level = 3
    o.intra_op_num_threads = THREADS; o.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=o, providers=["CPUExecutionProvider"])


def embed_rows(s, rows, images_dir):
    """rows: (image_id,x,y,w,h) — кроп с кадра, как в baseline extract_embeddings."""
    inp = s.get_inputs()[0].name
    dim = int(s.get_outputs()[0].shape[-1])
    emb = np.empty((len(rows), dim), np.float32)
    from PIL import Image
    t0 = time.perf_counter()
    buf, at = [], 0
    def flush():
        nonlocal at
        if not buf:
            return
        batch = np.stack(buf).astype(np.float32)
        (e,) = s.run(None, {inp: batch})
        emb[at:at + len(buf)] = e
        at += len(buf)
        buf.clear()
    for i, (iid, x, y, w, h) in enumerate(rows):
        with Image.open(images_dir / f"{iid}.jpg") as im:
            crop = im.convert("RGB").crop((x, y, x + w, y + h)).resize((208, 208), Image.BILINEAR)
        buf.append(np.asarray(crop, dtype=np.float32).transpose(2, 0, 1))
        if len(buf) == BATCH:
            flush()
        if i % 500 == 0:
            print(f"{i}/{len(rows)} {time.perf_counter()-t0:.0f}s", flush=True)
    flush()
    m = emb.astype(np.float64)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    assert (n > 0).all()
    return (m / n).astype(np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = session(MODEL)
    sha = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    info = {"model": str(MODEL), "sha256": sha, "threads": THREADS}
    print(json.dumps({k: v for k, v in info.items()}), flush=True)

    images = HOME / "data" / "images"
    for split in ("val_query", "val_gallery"):
        with open(REPO / "04-solution" / "split" / "files" / f"{split}.csv", newline="") as f:
            rows = [(r["image_id"], int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))
                    for r in csv.DictReader(f)]
        t0 = time.perf_counter()
        emb = embed_rows(s, rows, images)
        el = time.perf_counter() - t0
        np.save(OUT / f"{split}_j48.npy", emb)
        (OUT / f"{split}_j48.ids").write_text("".join(r[0] + "\n" for r in rows))
        info[split] = {"rows": len(rows), "elapsed_s": round(el, 1), "img_per_s": round(len(rows) / el, 2)}
        print(json.dumps({split: info[split]}), flush=True)

    # train_fit: те же кропы, что job_42/45 (crops_208.npy, порядок train_fit.csv)
    ids = (HOME / "jobs" / "job_43" / "out" / "train_fit.ids").read_text().split()
    crops = np.load(HOME / "data" / "crops_208.npy", mmap_mode="r")
    assert crops.shape[0] == len(ids)
    inp = s.get_inputs()[0].name
    emb = np.empty((len(ids), 512), np.float32)
    t0 = time.perf_counter()
    for st in range(0, len(ids), BATCH):
        batch = np.asarray(crops[st:st + BATCH], dtype=np.float32).transpose(0, 3, 1, 2)
        (e,) = s.run(None, {inp: batch})
        emb[st:st + len(batch)] = e
        if st % (BATCH * 50) == 0:
            print(f"train_fit {st}/{len(ids)} {time.perf_counter()-t0:.0f}s", flush=True)
    el = time.perf_counter() - t0
    m = emb.astype(np.float64); m /= np.linalg.norm(m, axis=1, keepdims=True)
    np.save(OUT / "train_fit_j48.npy", m.astype(np.float32))
    (OUT / "train_fit_j48.ids").write_text("".join(i + "\n" for i in ids))
    info["train_fit"] = {"rows": len(ids), "elapsed_s": round(el, 1), "img_per_s": round(len(ids) / el, 2)}

    # контроль близости к исходному OSNet (train_fit, тот же препроцессинг)
    old = np.load(HOME / "jobs" / "job_42" / "out" / "train_fit_osnet.npy").astype(np.float64)
    old /= np.linalg.norm(old, axis=1, keepdims=True)
    cos = float((m * old).sum(1).mean())
    info["cos_to_osnet_train_fit"] = cos
    (OUT / "s01_extract_info.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({"cos_to_osnet_train_fit": cos}), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
