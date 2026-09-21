"""Экспорт дообученного OSNet-AIN в ONNX — тем же контрактом, что у vehicle-reid-0001.

Вход `input`: RGB 0..255, NCHW, динамический батч (нормировку делает input_IN внутри
графа). Выход `output`: вектор 512 БЕЗ L2-нормировки — ровно как у исходного ONNX,
чтобы скрипт извлечения векторов сервиса работал без единой правки.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"D:\lct-reid")
sys.path.insert(0, str(ROOT))
from osnet_ain import OSNetAIN  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--probe", default=str(ROOT / "probe64.npz"))
ap.add_argument("--size", type=int, default=208)
a = ap.parse_args()

sd = torch.load(a.weights, map_location="cpu", weights_only=True)
sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
sd = {k: v for k, v in sd.items() if not k.startswith("classifier.")}
model = OSNetAIN()
model.load_state_dict(sd, strict=True)
model.eval()

z = np.load(a.probe)
x = torch.from_numpy(z["x"].astype(np.float32))
with torch.no_grad():
    ref = model(x).numpy()

dummy = torch.zeros(1, 3, a.size, a.size)
torch.onnx.export(model, dummy, a.out, opset_version=13,
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})

p = Path(a.out)
meta = {"weights": a.weights, "onnx": a.out, "bytes": p.stat().st_size,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "opset": 13, "input": [1, 3, a.size, a.size], "dim": int(ref.shape[1])}
np.save(p.with_suffix(".torch_probe.npy"), ref.astype(np.float32))
Path(a.out + ".json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta, indent=1), flush=True)
