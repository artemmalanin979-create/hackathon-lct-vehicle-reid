#!/usr/bin/env python3
"""Нарезка кропов train_fit для отправки на удалённый узел.

Кадр 1920x1080 + bbox -> RGB кроп -> resize 256x256 (PIL BILINEAR) -> JPEG q95.
Всё пакуется в один .npz: конкатенированные байты JPEG + смещения + метки.
На узле CPU занят чужой работой, поэтому там кадры не декодируются: приезжают
уже вырезанные кропы, они один раз распаковываются в uint8-массив в ОЗУ и дальше
идут на GPU напрямую.

ТОЛЬКО train_fit. val_query/val_gallery/val_unused не трогаются принципиально.
"""
from __future__ import annotations

import csv
import io
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/home/artem/projects/hackathon-lct-vehicle-reid")
DATA = ROOT / "data" / "images"
SPLIT = ROOT / "04-solution" / "split" / "files"
JOB = Path(__file__).resolve().parent.parent
SIZE = 256
QUALITY = 95


def rows():
    with open(SPLIT / "train_fit.csv", newline="") as f:
        return list(csv.DictReader(f))


def work(args):
    i, r = args
    iid = r["image_id"]
    x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
    with Image.open(DATA / f"{iid}.jpg") as im:
        crop = im.convert("RGB").crop((x, y, x + w, y + h))
    crop = crop.resize((SIZE, SIZE), Image.BILINEAR)
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=QUALITY, subsampling=0)
    return i, buf.getvalue()


def main():
    rs = rows()
    assert len(rs) == 7248, len(rs)
    t0 = time.perf_counter()
    with Pool(10) as p:
        res = p.map(work, list(enumerate(rs)), chunksize=32)
    res.sort()
    blobs = [b for _, b in res]

    offsets = np.zeros(len(blobs) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(b) for b in blobs])
    data = np.frombuffer(b"".join(blobs), dtype=np.uint8)

    vids = np.array([int(r["vehicle_id"]) for r in rs], dtype=np.int64)
    cams = np.array([int(r["camera_id"]) for r in rs], dtype=np.int64)
    iids = np.array([r["image_id"] for r in rs])

    out = JOB / "work" / "train_crops.npz"
    np.savez(out, data=data, offsets=offsets, vehicle_id=vids, camera_id=cams,
             image_id=iids, size=np.array([SIZE]), quality=np.array([QUALITY]))
    mb = out.stat().st_size / 1e6
    print(f"{len(blobs)} кропов {SIZE}x{SIZE} JPEG q{QUALITY} -> {out} "
          f"({mb:.1f} МБ, {time.perf_counter()-t0:.0f}s); "
          f"идентичностей {len(set(vids.tolist()))}, камер {len(set(cams.tolist()))}")


if __name__ == "__main__":
    main()
