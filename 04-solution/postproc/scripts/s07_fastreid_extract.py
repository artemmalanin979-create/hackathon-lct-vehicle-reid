#!/usr/bin/env python3
"""Шаг 7: вектора fast-reid SBS(R50-ibn) VeRi (Apache-2.0) на CPU.

Запускать интерпретатором fastreid_job/venv/bin/python (torch 2.14 cpu).
Препроцессинг — как у fastreid на тесте: кроп bbox -> RGB -> Resize (256,256)
PIL BICUBIC (interpolation=3) -> float32 0..255 CHW; нормализация mean/std —
внутри модели (Baseline.preprocess_image). Инференс eval: эмбеддинг 2048-d
(EmbeddingHead, NECK_FEAT=after). Выход L2-нормируется.

Классификатор в чекпойнте не используется на инференсе; несовпадение его ключей
(heads.classifier.weight -> heads.weight) на выход не влияет — проверено load-репортом.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

JOB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JOB / "fastreid_job/fast-reid"))
from fastreid.config import get_cfg  # noqa: E402
from fastreid.modeling.meta_arch import build_model  # noqa: E402

IMAGES = Path("/home/artem/projects/hackathon-lct-vehicle-reid/data/images")
SIZE = 256


def build():
    cfg = get_cfg()
    cfg.merge_from_file(str(JOB / "fastreid_job/fast-reid/configs/VeRi/sbs_R50-ibn.yml"))
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.DEVICE = "cpu"
    model = build_model(cfg)
    model.eval()
    ckpt = torch.load(JOB / "fastreid_job/veri_sbs_R50-ibn.pth",
                      map_location="cpu", weights_only=False)
    res = model.load_state_dict(ckpt["model"], strict=False)
    assert res.missing_keys == ["heads.weight"], res.missing_keys
    model_only = JOB / "fastreid_job/veri_sbs_R50-ibn.model_only.pth"
    if not model_only.exists():
        torch.save(ckpt["model"], model_only)
    return model


def load_crop(row):
    with Image.open(IMAGES / f"{row['image_id']}.jpg") as im:
        im = im.convert("RGB")
        crop = im.crop((int(row["x"]), int(row["y"]),
                        int(row["x"]) + int(row["w"]), int(row["y"]) + int(row["h"])))
    crop = crop.resize((SIZE, SIZE), Image.BICUBIC)
    return np.asarray(crop, dtype=np.float32).transpose(2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timing", type=Path, default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    model = build()

    out = np.empty((len(rows), 2048), dtype=np.float32)
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(rows), args.batch):
            chunk = rows[start:start + args.batch]
            batch = torch.from_numpy(np.stack([load_crop(r) for r in chunk]))
            emb = model(batch)  # мутирует batch (sub_/div_) — batch дальше не нужен
            out[start:start + len(chunk)] = emb.numpy()
            if start % (args.batch * 25) == 0:
                done = start + len(chunk)
                print(f"{args.csv.name}: {done}/{len(rows)} "
                      f"({(time.perf_counter()-t0)/done:.2f} с/объект)", flush=True)
    elapsed = time.perf_counter() - t0

    out = out.astype(np.float64)
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out.astype(np.float32))
    stats = {"csv": str(args.csv), "rows": len(rows), "batch": args.batch,
             "threads": args.threads or torch.get_num_threads(),
             "elapsed_s": round(elapsed, 1),
             "s_per_obj": round(elapsed / len(rows), 3)}
    if args.timing:
        args.timing.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
