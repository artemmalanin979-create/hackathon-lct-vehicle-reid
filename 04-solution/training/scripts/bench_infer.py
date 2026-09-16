#!/usr/bin/env python3
"""Время инференса на CPU: полный конвейер (кадр+bbox -> вектор) и только сеть.

Оба замера делаются в одной сессии на одной машине, поэтому сравнение
«новая модель против OSNet» честное, даже если абсолютные числа зависят от загрузки.
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image

DATA = Path("/home/artem/projects/hackathon-lct-vehicle-reid/data/images")
SPLIT = Path("/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/split/files")


def sess_of(p):
    o = ort.SessionOptions(); o.log_severity_level = 3
    return ort.InferenceSession(str(p), sess_options=o, providers=["CPUExecutionProvider"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="имя=путь.onnx")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = [(r["image_id"], int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))
            for r in csv.DictReader(open(SPLIT / "val_query.csv", newline=""))][:args.n]
    res = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        s = sess_of(path)
        inp = s.get_inputs()[0].name
        size = s.get_inputs()[0].shape[-1]
        size = 208 if not isinstance(size, int) else size

        # 1) полный конвейер, батч 1
        t0 = time.perf_counter()
        for r in rows:
            with Image.open(DATA / f"{r[0]}.jpg") as im:
                c = im.convert("RGB").crop((r[1], r[2], r[1] + r[3], r[2] + r[4]))
            a = np.asarray(c.resize((size, size), Image.BILINEAR),
                           dtype=np.float32).transpose(2, 0, 1)[None]
            s.run(None, {inp: a})
        full1 = (time.perf_counter() - t0) / len(rows) * 1e3

        # 2) только сеть (батч 1 и 32) на уже подготовленных тензорах
        pre = []
        for r in rows[:32]:
            with Image.open(DATA / f"{r[0]}.jpg") as im:
                c = im.convert("RGB").crop((r[1], r[2], r[1] + r[3], r[2] + r[4]))
            pre.append(np.asarray(c.resize((size, size), Image.BILINEAR),
                                  dtype=np.float32).transpose(2, 0, 1))
        one = pre[0][None]
        for _ in range(3):
            s.run(None, {inp: one})
        t0 = time.perf_counter()
        for _ in range(30):
            s.run(None, {inp: one})
        net1 = (time.perf_counter() - t0) / 30 * 1e3

        b32 = np.stack(pre)
        s.run(None, {inp: b32})
        t0 = time.perf_counter()
        for _ in range(5):
            s.run(None, {inp: b32})
        net32 = (time.perf_counter() - t0) / 5 / 32 * 1e3

        res[name] = {"input": int(size), "onnx_bytes": Path(path).stat().st_size,
                     "full_pipeline_b1_ms": round(full1, 1),
                     "net_only_b1_ms": round(net1, 1),
                     "net_only_b32_ms_per_obj": round(net32, 1)}
        print(name, json.dumps(res[name]), flush=True)
    args.out.write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
