"""Экспорт обученной модели в ONNX (сервис работает через onnxruntime).

В граф зашиты нормировка ImageNet и L2-нормировка выхода: на вход подаётся
RGB 0..255 NCHW 1x3x208x208 — ровно то же, что ждёт базовый экстрактор векторов.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(r"D:\lct-reid")
sys.path.insert(0, str(ROOT))
from reid_train import ReIDNet  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="r50_208")
ap.add_argument("--ckpt", default="final.pt")
ap.add_argument("--size", type=int, default=208)
ap.add_argument("--arch", default="resnet50")
ap.add_argument("--last-stride", type=int, default=2)
ap.add_argument("--opset", type=int, default=13)
a = ap.parse_args()

run_dir = ROOT / "runs" / a.run
st = torch.load(run_dir / a.ckpt, map_location="cpu", weights_only=False)
model = ReIDNet(st["num_classes"], a.arch, a.last_stride, None)
model.load_state_dict(st["model"])
model.eval()

dummy = torch.randint(0, 256, (1, 3, a.size, a.size)).float()
out = run_dir / f"{a.run}.onnx"
torch.onnx.export(
    model, dummy, str(out), input_names=["input"], output_names=["embedding"],
    dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
    opset_version=a.opset, do_constant_folding=True,
)

# контрольные значения: те же входы прогоняются torch-ом, чтобы локально сверить с ORT
torch.manual_seed(0)
probe = torch.randint(0, 256, (4, 3, a.size, a.size)).float()
with torch.no_grad():
    ref = model(probe).numpy()
np.savez(run_dir / f"{a.run}_probe.npz", inputs=probe.numpy(), torch_out=ref)
meta = {"onnx": str(out), "bytes": out.stat().st_size, "epoch": st["epoch"],
        "num_classes": st["num_classes"], "dim": int(ref.shape[1]), "size": a.size,
        "opset": a.opset,
        "weights_pt_bytes": (run_dir / a.ckpt).stat().st_size}
(run_dir / f"{a.run}_export.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta, indent=1))
