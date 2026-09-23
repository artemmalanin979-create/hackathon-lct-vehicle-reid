#!/usr/bin/env python3
"""Извлечение L2-нормированных векторов OSNet-AIN (vehicle-reid-0001) по CSV с bbox.

Вход: CSV с колонками image_id,x,y,w,h (лишние колонки игнорируются; порядок строк
файла сохраняется в выходном массиве). Кроп по bbox -> RGB -> resize 208x208
(PIL bilinear) -> float32 0..255 -> NCHW -> ONNX -> L2-нормировка.

Препроцессинг по карточке OMZ vehicle-reid-0001 (тег 2022.1.0): original model ждёт
RGB 1x3x208x208; конверсия в IR делает только --reverse_input_channels (без mean/scale);
accuracy-check.yml: resize use_pillow BILINEAR без нормализации. Первый узел графа —
InstanceNormalization на входе, поэтому по-канальная аффинная нормализация не нужна
(и не влияет на выход).

Память: изображения читаются потоково, в памяти держится один батч (по умолчанию 16
кропов = ~4 МБ) и выходной массив N x 512 float32.

Абляции для проверки независимости от номерной пластины:
  --mask-bottom F   залить нижнюю долю F кропа серым (127,127,127) ДО resize
  --grayscale       обесцветить кроп (L -> RGB) ДО resize
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

if __package__:
    from .inputs import require_dataset, require_files
else:
    from inputs import require_dataset, require_files

INPUT_SIZE = 208


def read_rows(csv_path: Path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_id", "x", "y", "w", "h"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{csv_path}: нет колонок {sorted(missing)}")
        return [
            (r["image_id"], int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))
            for r in reader
        ]


def load_crop(images_dir: Path, row, mask_bottom: float, grayscale: bool) -> np.ndarray:
    import numpy as np
    from PIL import Image

    image_id, x, y, w, h = row
    with Image.open(images_dir / f"{image_id}.jpg") as im:
        im = im.convert("RGB")
        crop = im.crop((x, y, x + w, y + h))
    if grayscale:
        crop = crop.convert("L").convert("RGB")
    if mask_bottom > 0:
        arr = np.asarray(crop)
        band = int(round(arr.shape[0] * mask_bottom))
        if band > 0:
            arr = arr.copy()
            arr[-band:, :, :] = 127
        crop = Image.fromarray(arr)
    crop = crop.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    return np.asarray(crop, dtype=np.float32).transpose(2, 0, 1)  # C,H,W; RGB; 0..255


def make_session(model_path: Path, threads: int) -> ort.InferenceSession:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # заглушить INFO/WARNING про неиспользуемые инициализаторы
    if threads:
        opts.intra_op_num_threads = threads
    return ort.InferenceSession(
        str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
    )


def l2norm(m: np.ndarray) -> np.ndarray:
    import numpy as np

    return m / np.linalg.norm(m, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="выход .npy float32 N x 512")
    ap.add_argument("--ids-out", type=Path, default=None,
                    help="текстовый файл image_id по строкам выхода (для сверки порядка)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=0, help="intra-op потоки ORT (0=default)")
    ap.add_argument("--mask-bottom", type=float, default=0.0)
    ap.add_argument("--grayscale", action="store_true")
    ap.add_argument("--timing", type=Path, default=None, help="куда писать JSON замера")
    ap.add_argument("--limit", type=int, default=0, help="обработать только первые N строк")
    args = ap.parse_args()
    require_dataset([args.csv], args.images_dir, suffixes=(".jpg",), limit=args.limit)
    require_files([args.model], hint="Укажите --model; веса — service/model/fetch_model.sh.")

    import numpy as np

    rows = read_rows(args.csv)
    if args.limit:
        rows = rows[: args.limit]
    session = make_session(args.model, args.threads)
    input_name = session.get_inputs()[0].name

    out = np.empty((len(rows), 512), dtype=np.float32)
    t_start = time.perf_counter()
    for start in range(0, len(rows), args.batch):
        chunk = rows[start : start + args.batch]
        batch = np.stack(
            [load_crop(args.images_dir, r, args.mask_bottom, args.grayscale) for r in chunk]
        )
        (emb,) = session.run(None, {input_name: batch})
        out[start : start + len(chunk)] = emb
    elapsed = time.perf_counter() - t_start

    out = l2norm(out.astype(np.float64)).astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out)
    if args.ids_out:
        args.ids_out.write_text("".join(r[0] + "\n" for r in rows))

    stats = {
        "csv": str(args.csv), "rows": len(rows), "batch": args.batch,
        "threads": args.threads, "mask_bottom": args.mask_bottom,
        "grayscale": args.grayscale, "elapsed_s": round(elapsed, 3),
        "throughput_obj_per_s": round(len(rows) / elapsed, 2) if elapsed else None,
    }
    if args.timing:
        args.timing.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
