#!/usr/bin/env python3
"""Шаг 3: извлечение векторов для TTA-вариантов (размер входа x отражение).

Препроцессинг побайтово повторяет baseline/scripts/extract_embeddings.py
(кроп bbox -> RGB -> resize SxS PIL bilinear -> float32 0..255 -> NCHW -> L2),
плюс два параметра: размер входа S и горизонтальное отражение ПОСЛЕ resize.
Проверка эквивалентности при S=208 без отражения — сверка с val_query.npy,
полученным их скриптом (max|diff| печатается).

Выход: out/{set}_{part}_{S}{f}.npy; тайминг каждого варианта — out/s03_timings.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import JOB, SPLIT, DATA, read_meta, l2norm

MODEL = JOB / "model/osnet_ain_x1_0_vehicle_reid.onnx"
IMAGES = DATA / "images"
BATCH = 16


def make_session():
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(MODEL), sess_options=opts,
                                providers=["CPUExecutionProvider"])


def load_crop(row, size, flip):
    with Image.open(IMAGES / f"{row['image_id']}.jpg") as im:
        im = im.convert("RGB")
        crop = im.crop((int(row["x"]), int(row["y"]),
                        int(row["x"]) + int(row["w"]), int(row["y"]) + int(row["h"])))
    crop = crop.resize((size, size), Image.BILINEAR)
    arr = np.asarray(crop, dtype=np.float32)
    if flip:
        arr = arr[:, ::-1, :]
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def extract(session, rows, size, flip):
    input_name = session.get_inputs()[0].name
    out = np.empty((len(rows), 512), dtype=np.float32)
    t0 = time.perf_counter()
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        batch = np.stack([load_crop(r, size, flip) for r in chunk])
        (emb,) = session.run(None, {input_name: batch})
        out[start:start + len(chunk)] = emb
    dt = time.perf_counter() - t0
    return l2norm(out.astype(np.float64)).astype(np.float32), dt


def main():
    session = make_session()
    sets = {
        ("val", "query"): read_meta(SPLIT / "val_query.csv"),
        ("val", "gallery"): read_meta(SPLIT / "val_gallery.csv"),
        ("tune", "query"): read_meta(JOB / "tune/tune_query.csv"),
        ("tune", "gallery"): read_meta(JOB / "tune/tune_gallery.csv"),
    }
    variants = [(208, False), (208, True), (256, False), (256, True),
                (288, False), (288, True)]
    timings = {}
    for (sname, part), rows in sets.items():
        ids_path = JOB / f"out/{sname}_{part}.ids"
        if not ids_path.exists():
            ids_path.write_text("".join(r["image_id"] + "\n" for r in rows))
        for size, flip in variants:
            tag = f"{size}{'f' if flip else ''}"
            path = JOB / f"out/{sname}_{part}_{tag}.npy"
            if sname == "val" and tag == "208":
                # уже извлечено их скриптом; сверим свой препроцессинг
                ref = np.load(JOB / f"out/val_{part}.npy")
                mine, dt = extract(session, rows, size, flip)
                print(f"val_{part}_208 сверка со скриптом бейзлайна: "
                      f"max|diff|={np.abs(mine - ref).max():.2e}", flush=True)
                np.save(path, ref)  # каноническими остаются их вектора
            elif path.exists():
                continue
            else:
                emb, dt = extract(session, rows, size, flip)
                np.save(path, emb)
            timings[f"{sname}_{part}_{tag}"] = {
                "rows": len(rows), "elapsed_s": round(dt, 2),
                "ms_per_obj_batch16": round(1000 * dt / len(rows), 2)}
            print(f"{sname}_{part}_{tag}: {len(rows)} строк за {dt:.1f} с", flush=True)
    (JOB / "out/s03_timings.json").write_text(json.dumps(timings, indent=2) + "\n")


if __name__ == "__main__":
    main()
