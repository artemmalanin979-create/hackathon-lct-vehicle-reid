"""Детекции пластины на всех 1860 кропах val (query+gallery), кэш в JSON."""
import argparse
import os
import csv, json, sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plate_detect import detect
REPO = Path(__file__).resolve().parents[3]  # корень репозитория
DATA = Path(os.environ.get("REID_DATA_DIR", REPO / "data"))
SPLIT = REPO / "04-solution/split/files"

def rows(name):
    return [(r["image_id"], int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))
            for r in csv.DictReader(open(SPLIT / f"{name}.csv", newline=""))]

def work(args):
    i, (iid, x, y, w, h) = args
    with Image.open(DATA / "images" / f"{iid}.jpg") as im:
        a = np.asarray(im.convert("RGB").crop((x, y, x + w, y + h)))
    d = detect(a, topn=1)
    if not d:
        return i, None
    s, bx, by, bw, bh, f = d[0]
    return i, [int(bx), int(by), int(bw), int(bh), round(float(s), 2),
               round(float(f["dens"]), 3), round(float(f["std"]), 1)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("work/boxes.json"),
                    help="куда писать кэш детекций (по умолчанию work/boxes.json)")
    ap.add_argument("--limit", type=int, default=0,
                    help="обработать только первые N кропов каждого набора (0 — все); "
                         "неполный кэш годится только для проверки запуска")
    ap.add_argument("--procs", type=int, default=6, help="процессов в пуле")
    args = ap.parse_args()

    out = {}
    for name in ("val_query", "val_gallery"):
        rs = rows(name)
        if args.limit:
            rs = rs[:args.limit]
        t0 = time.perf_counter()
        with Pool(args.procs) as p:
            res = p.map(work, list(enumerate(rs)), chunksize=8)
        res.sort()
        out[name] = [b for _, b in res]
        found = sum(1 for b in out[name] if b)
        print(f"{name}: {len(rs)} кропов, бокс найден у {found}, "
              f"{time.perf_counter()-t0:.0f}s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print(f"записано: {args.out}", flush=True)
