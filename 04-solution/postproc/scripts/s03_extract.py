#!/usr/bin/env python3
"""Шаг 3: извлечение векторов для TTA-вариантов (размер входа x отражение).

Препроцессинг побайтово повторяет baseline/scripts/extract_embeddings.py
(кроп bbox -> RGB -> resize SxS PIL bilinear -> float32 0..255 -> NCHW -> L2),
плюс два параметра: размер входа S и горизонтальное отражение ПОСЛЕ resize.
Проверка эквивалентности при S=208 без отражения — сверка с val_query.npy,
полученным их скриптом (max|diff| печатается).

Выход: out/{set}_{part}_{S}{f}.npy; тайминг каждого варианта — out/s03_timings.json.

По умолчанию извлекаются оба набора и все шесть вариантов (как в отчёте). Для
частичного прогона: --sets val --variants 208,208f,256 (столько нужно шагу 6).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

if __package__:
    from .inputs import JOB, REPO, SPLIT, DATA, require_dataset, require_files, BASE_HINT
else:
    from inputs import JOB, REPO, SPLIT, DATA, require_dataset, require_files, BASE_HINT

# Веса решения лежат в service/model (их кладёт service/model/fetch_model.sh).
MODEL = Path(os.environ.get(
    "REID_MODEL_PATH", REPO / "04-solution/service/model/osnet_ain_x1_0_vehicle_reid.onnx"))
IMAGES = DATA / "images"
BATCH = 16


def make_session():
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(MODEL), sess_options=opts,
                                providers=["CPUExecutionProvider"])


def load_crop(row, size, flip):
    import numpy as np
    from PIL import Image

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
    import numpy as np
    if __package__:
        from .common import l2norm
    else:
        from common import l2norm

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


ALL_VARIANTS = [(208, False), (208, True), (256, False), (256, True),
                (288, False), (288, True)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default="val,tune", help="наборы через запятую: val,tune")
    ap.add_argument("--variants", default=",".join(f"{s}{'f' if f else ''}" for s, f in ALL_VARIANTS),
                    help="варианты входа через запятую, например 208,208f,256")
    args = ap.parse_args()
    wanted_sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    wanted_tags = [t.strip() for t in args.variants.split(",") if t.strip()]

    csv_paths = {(sname, part): (SPLIT / f"val_{part}.csv" if sname == "val"
                                else JOB / f"tune/tune_{part}.csv")
                 for sname in ("val", "tune") if sname in wanted_sets
                 for part in ("query", "gallery")}
    require_files(csv_paths.values(), hint=BASE_HINT)
    # Cached variants are skipped below: their source images are not required.
    active_tags = [f"{size}{'f' if flip else ''}" for size, flip in ALL_VARIANTS
                   if f"{size}{'f' if flip else ''}" in wanted_tags]
    needed_csvs = [path for (sname, part), path in csv_paths.items()
                   if any((sname == "val" and tag == "208")
                          or not (JOB / f"out/{sname}_{part}_{tag}.npy").exists()
                          for tag in active_tags)]
    require_dataset(needed_csvs, IMAGES, suffixes=(".jpg",))
    if "val" in wanted_sets and "208" in active_tags:
        require_files([JOB / f"out/val_{part}.npy" for part in ("query", "gallery")],
                      hint=BASE_HINT)
    require_files([MODEL], hint="Веса — service/model/fetch_model.sh; проверьте REID_MODEL_PATH.")

    import numpy as np
    if __package__:
        from .common import read_meta
    else:
        from common import read_meta

    session = make_session()
    all_sets = {
        ("val", "query"): lambda: read_meta(SPLIT / "val_query.csv"),
        ("val", "gallery"): lambda: read_meta(SPLIT / "val_gallery.csv"),
        ("tune", "query"): lambda: read_meta(JOB / "tune/tune_query.csv"),
        ("tune", "gallery"): lambda: read_meta(JOB / "tune/tune_gallery.csv"),
    }
    sets = {key: make() for key, make in all_sets.items() if key[0] in wanted_sets}
    variants = [(size, flip) for size, flip in ALL_VARIANTS
                if f"{size}{'f' if flip else ''}" in wanted_tags]
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
                ref_path = JOB / f"out/val_{part}.npy"
                if not ref_path.is_file():
                    raise SystemExit(
                        f"нет {ref_path}: это базовые векторы бейзлайна. Получить их — "
                        "scripts/extract_embeddings.py бейзлайна (см. postproc/README.md, шаг 1)")
                ref = np.load(ref_path)
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
    if wanted_sets == ["val", "tune"] and len(variants) == len(ALL_VARIANTS):
        (JOB / "out/s03_timings.json").write_text(json.dumps(timings, indent=2) + "\n")
    else:  # частичный прогон не должен затирать тайминги полного
        print("частичный прогон: out/s03_timings.json не перезаписан", flush=True)


if __name__ == "__main__":
    main()
