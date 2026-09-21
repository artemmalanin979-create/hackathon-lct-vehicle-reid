#!/usr/bin/env python3
"""Приёмка сборки: имена state_dict должны совпасть с инициализаторами ONNX,
а выход torch — с выходом ONNX на боевых кропах.

Запускается и локально (если есть torch), и на узле. Числа — единственное
доказательство правильности сборки.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from osnet_ain import OSNetAIN  # noqa: E402


def load_state(npz_path):
    z = np.load(npz_path)
    return {k: torch.from_numpy(np.asarray(z[k])) for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--probe", default=None, help="npz с x (N,3,H,W float32 0..255) и emb_onnx")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save", default=None, help="куда положить torch-веса (.pt)")
    a = ap.parse_args()

    sd = load_state(a.npz)
    model = OSNetAIN()
    own = model.state_dict()
    miss = sorted(set(own) - set(sd))
    extra = sorted(set(sd) - set(own))
    shape_bad = [k for k in set(own) & set(sd) if tuple(own[k].shape) != tuple(sd[k].shape)]
    rep = {"n_onnx": len(sd), "n_model": len(own), "missing_in_onnx": miss,
           "extra_in_onnx": extra, "shape_mismatch": shape_bad}
    print(json.dumps({k: (v if not isinstance(v, list) else v[:10]) for k, v in rep.items()},
                     indent=1, ensure_ascii=False))
    if miss or extra or shape_bad:
        raise SystemExit("СБОРКА НЕВЕРНА: имена/формы не сошлись")
    model.load_state_dict(sd, strict=True)
    model.eval()

    if a.probe:
        z = np.load(a.probe)
        x = torch.from_numpy(z["x"].astype(np.float32))
        with torch.no_grad():
            y = model(x).numpy()
        yn = y / np.linalg.norm(y, axis=1, keepdims=True)
        ref = z["emb_onnx"]
        refn = ref / np.linalg.norm(ref, axis=1, keepdims=True)
        cos = (yn * refn).sum(1)
        rep["probe_n"] = int(len(cos))
        rep["max_abs_diff"] = float(np.abs(yn - refn).max())
        rep["min_cos"] = float(cos.min())
        print(f"проб {len(cos)}: min cos = {cos.min():.9f}, max|Δ| = {rep['max_abs_diff']:.3e}")
        if cos.min() < 0.9999:
            raise SystemExit("СБОРКА НЕВЕРНА: косинус < 0,9999")
    if a.save:
        torch.save(model.state_dict(), a.save)
        print("сохранено:", a.save)
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
