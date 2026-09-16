"""Где уходит время эпохи: чистый шаг сети против аугментации."""
import sys, time, json
import numpy as np, torch
sys.path.insert(0, r"D:\lct-reid")
from reid_train import ReIDNet, gpu_augment, batch_hard_triplet
import torch.nn as nn

torch.backends.cudnn.benchmark = True
dev = torch.device("cuda")
m = ReIDNet(1071, "resnet50", 2, None).to(dev).train()
opt = torch.optim.Adam(m.parameters(), lr=3.5e-4)
ce = nn.CrossEntropyLoss(label_smoothing=0.1)
rng = np.random.default_rng(0)
arr = np.random.randint(0, 255, (256, 208, 208, 3), dtype=np.uint8)
lab = torch.arange(32, device=dev) // 4

def step(aug):
    idx = rng.integers(0, 256, 32)
    x = torch.from_numpy(arr[idx]).to(dev).permute(0, 3, 1, 2).float()
    if aug:
        x = gpu_augment(x, rng)
    f, lg = m.forward_train(x)
    loss = ce(lg, lab) + batch_hard_triplet(f, lab, 0.3)[0]
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

for aug in (False, True):
    for _ in range(4):
        step(aug)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(15):
        step(aug)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 15
    print(json.dumps({"aug": aug, "step_ms": round(dt * 1e3, 1),
                      "epoch_s_207": round(dt * 207, 1)}), flush=True)

# сама аугментация отдельно
x = torch.randn(32, 3, 208, 208, device=dev)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(15):
    gpu_augment(x, rng)
torch.cuda.synchronize()
print(json.dumps({"aug_only_ms": round((time.perf_counter() - t0) / 15 * 1e3, 1)}))
