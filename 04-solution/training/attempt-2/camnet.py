"""Камерный штраф VOC-ReID (§7.7 рецепта): ОТДЕЛЬНАЯ сеть распознавания камеры.

Объединять её со стволом re-id нельзя — ствол тогда обязан кодировать камеру, то есть
ровно то, от чего мы избавляемся. Поэтому маленькая сеть с нуля, 96 классов камер.
На инференсе camera_id не нужен: сеть сама предсказывает камерный вектор, его схожесть
вычитается из матрицы схожести re-id.

Вход тот же, что у основной модели: RGB 0..255 NCHW 208x208 (нормировка внутри графа).
Выход: камерный вектор 128 без L2-нормировки (нормирует потребитель).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(r"D:\lct-reid")
SEED = 20260916


class CamNet(nn.Module):
    """~0,4 M параметров. Сцену/точку съёмки распознавать легко, разрешение не нужно:
    вход 208x208 сразу уполовинивается, дальше пять ступеней со stride 2.

    Depthwise-свёрток нет намеренно: на Maxwell их cuDNN-ядра медленные, и первая же
    версия сети словила TDR («the launch timed out and was terminated») — карта
    обслуживает дисплей, ядро длиннее ~2 с драйвер убивает.
    """

    def __init__(self, dim=128, num_classes=0):
        super().__init__()
        self.register_buffer("mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))
        ch = [3, 24, 48, 64, 96, 128]
        blocks = [nn.AvgPool2d(2)]
        for i in range(5):
            blocks += [nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(ch[i + 1]), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*blocks)
        self.proj = nn.Linear(ch[-1], dim)
        self.bn = nn.BatchNorm1d(dim)
        self.dim = dim
        self.classifier = nn.Linear(dim, num_classes, bias=False) if num_classes else None

    def forward(self, x):
        x = (x - self.mean) / self.std
        y = F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)
        return self.bn(self.proj(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="camnet")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--size", type=int, default=208)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.1)
    a = ap.parse_args()

    run_dir = ROOT / "runs" / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    z = np.load(ROOT / "data" / "train_crops.npz", allow_pickle=False)
    cams = z["camera_id"].astype(np.int64)
    arr = np.load(ROOT / "data" / f"crops_{a.size}.npy")
    uniq = sorted(set(cams.tolist()))
    c2l = {c: i for i, c in enumerate(uniq)}
    lab = np.array([c2l[int(c)] for c in cams])

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(lab))
    n_val = int(len(lab) * a.val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    model = CamNet(a.dim, num_classes=len(uniq)).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=5e-4, foreach=False)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    cfg = {"run": a.run, "epochs": a.epochs, "classes": len(uniq), "params": n_par,
           "batch": a.batch, "lr": a.lr, "dim": a.dim,
           "n_train": len(tr_idx), "n_val": len(val_idx)}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    print(json.dumps(cfg), flush=True)

    log = run_dir / "log.jsonl"
    for ep in range(a.epochs):
        model.train()
        scale = 0.5 * (1 + np.cos(np.pi * ep / a.epochs))
        for g in opt.param_groups:
            g["lr"] = a.lr * max(scale, 0.02)
        order = rng.permutation(tr_idx)
        t0, tot, seen = time.perf_counter(), 0.0, 0
        for s in range(0, len(order) - a.batch + 1, a.batch):
            idx = np.sort(order[s:s + a.batch])
            x = torch.from_numpy(arr[idx]).to(dev).permute(0, 3, 1, 2).float()
            flip = torch.rand(len(idx), device=dev) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(3), x)
            y = torch.from_numpy(lab[idx]).to(dev)
            out = model.classifier(model(x))
            loss = ce(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            seen += len(idx)
        model.eval()
        correct = 0
        with torch.no_grad():
            for s in range(0, len(val_idx), 64):
                idx = np.sort(val_idx[s:s + 64])
                x = torch.from_numpy(arr[idx]).to(dev).permute(0, 3, 1, 2).float()
                pred = model.classifier(model(x)).argmax(1).cpu().numpy()
                correct += int((pred == lab[idx]).sum())
        rec = {"epoch": ep + 1, "lr": round(a.lr * max(scale, 0.02), 7),
               "loss": round(tot / seen, 4), "val_top1": round(correct / len(val_idx), 4),
               "sec": round(time.perf_counter() - t0, 1)}
        with open(log, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        tmp = run_dir / "ckpt.tmp"
        torch.save({"model": model.state_dict(), "epoch": ep + 1, "classes": len(uniq)}, tmp)
        os.replace(tmp, run_dir / "ckpt.pt")

    torch.save(model.state_dict(), run_dir / "camnet.pt")
    # экспорт ONNX (без классификатора): вход RGB 0..255, выход вектор dim
    exp = CamNet(a.dim)
    exp.load_state_dict({k: v for k, v in model.state_dict().items()
                         if not k.startswith("classifier.")}, strict=True)
    exp.eval()
    torch.onnx.export(exp.cpu(), torch.zeros(1, 3, a.size, a.size),
                      str(run_dir / "camnet.onnx"), opset_version=13,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
    print("ГОТОВО: камерная сеть обучена и экспортирована", flush=True)


if __name__ == "__main__":
    main()
