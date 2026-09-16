"""Прогон torch-модели на РЕАЛЬНЫХ кропах валидации (приезжают как uint8 npz).

Нужен, чтобы сверка «ONNX даёт те же векторы, что torch» делалась на боевых входах,
а не только на случайном шуме.
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(r"D:\lct-reid")
sys.path.insert(0, str(ROOT))
from reid_train import ReIDNet  # noqa: E402

ckpt = ROOT / "runs" / sys.argv[1] / sys.argv[2]
probe = ROOT / "data" / "val_probe64.npz"
st = torch.load(ckpt, map_location="cpu", weights_only=False)
model = ReIDNet(st["num_classes"], "resnet50", 2, None)
model.load_state_dict(st["model"])
model.eval()
z = np.load(probe)
x = torch.from_numpy(z["crops"]).permute(0, 3, 1, 2).float()
with torch.no_grad():
    out = model(x).numpy()
np.save(ROOT / "runs" / sys.argv[1] / "val_probe64_torch.npy", out)
print("torch probe:", out.shape, float(np.linalg.norm(out, axis=1).mean()))
