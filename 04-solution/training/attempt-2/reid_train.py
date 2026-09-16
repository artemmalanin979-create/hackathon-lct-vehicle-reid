"""Дообучение ReID-модели на train_fit. Запускается на удалённом узле (Windows, 1 GPU).

Рецепт — классический сильный бейзлайн повторной идентификации (Luo et al., "Bag of
Tricks"), с отступлениями, вынужденными железом (см. REPORT.md):
  backbone ResNet-50 (ImageNet) -> GeM p=3 -> BNNeck -> ID-loss (label smoothing 0.1)
  + batch-hard triplet (margin 0.3), Adam 3.5e-4, warmup 5 эпох, косинусный спад,
  random erasing 0.5, pad-10-and-crop, горизонтальное отражение.

Особенности под чужую занятую машину:
  * данные приезжают уже вырезанными (JPEG 256x256), один раз распаковываются в ОЗУ;
  * НИКАКИХ многопоточных загрузчиков: батч набирается индексацией массива в ОЗУ,
    вся аугментация делается на GPU;
  * выполнение режется на куски (--max-seconds) с полным чекпойнтом каждую эпоху,
    чтобы одна ssh-сессия была короткой и ничего не терялось.

Метрика для внутреннего мониторинга считается ТЕМ ЖЕ контуром reid_metrics.py,
своего расчёта метрик нет нигде.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

ROOT = Path(r"D:\lct-reid")
sys.path.insert(0, str(ROOT))
from reid_metrics import evaluate, scores_from_embeddings  # noqa: E402

SEED = 20260915
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- данные

def load_crops(npz_path: Path, cache: Path, size: int):
    """JPEG -> uint8 [N,size,size,3] в ОЗУ. Декодирование один раз, потом кэш на диске."""
    z = np.load(npz_path, allow_pickle=False)
    vids, cams = z["vehicle_id"], z["camera_id"]
    if cache.exists():
        arr = np.load(cache, mmap_mode=None)
        assert arr.shape == (len(vids), size, size, 3), arr.shape
        return arr, vids, cams
    from PIL import Image
    data, off = z["data"], z["offsets"]
    n = len(off) - 1
    arr = np.empty((n, size, size, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    for i in range(n):
        with Image.open(io.BytesIO(data[off[i]:off[i + 1]].tobytes())) as im:
            im = im.convert("RGB")
            if im.size != (size, size):
                im = im.resize((size, size), Image.BILINEAR)
            arr[i] = np.asarray(im)
    print(f"декодировано {n} кропов за {time.perf_counter()-t0:.0f}s -> {cache}", flush=True)
    np.save(cache, arr)
    return arr, vids, cams


def make_dev_split(vids, cams, n_dev_ids: int):
    """n_dev_ids идентичностей train_fit откладываются для честного мониторинга.

    Протокол копирует устройство нашего вала: галерея — 1 кадр с каждой камеры
    идентичности, запросы — остальные её кадры; совпадения со своей камеры
    исключаются камерной политикой контура.
    """
    uniq = np.unique(vids)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(uniq))
    dev_ids = set(uniq[perm[:n_dev_ids]].tolist())
    is_dev = np.array([v in dev_ids for v in vids])
    train_idx = np.flatnonzero(~is_dev)

    q_idx, g_idx = [], []
    for v in sorted(dev_ids):
        rows = np.flatnonzero(vids == v)
        g_here, q_here = [], []
        for c in np.unique(cams[rows]):
            grp = rows[cams[rows] == c]
            g_here.append(int(grp[0]))
            q_here.extend(int(i) for i in grp[1:])
        if not q_here and len(g_here) >= 2:
            q_here.append(g_here.pop())         # у ид. по одному кадру на камеру
        q_idx.extend(q_here)
        g_idx.extend(g_here)
    return train_idx, np.array(sorted(q_idx)), np.array(sorted(g_idx))


class PKSampler:
    """P идентичностей x K кадров. Кадры одной идентичности берутся с РАЗНЫХ камер,
    пока камеры есть: иначе самым «похожим» положительным примером оказывается
    соседний кадр той же серии, и сеть учится запоминать фон точки съёмки."""

    def __init__(self, vids, cams, idx, P, K, seed):
        self.P, self.K = P, K
        self.rng = np.random.default_rng(seed)
        self.by_id = {}
        for i in idx:
            self.by_id.setdefault(int(vids[i]), []).append(int(i))
        self.ids = sorted(self.by_id)
        self.cam_of = {int(i): int(cams[i]) for i in idx}
        self.n_batches = max(1, len(idx) // (P * K))

    def pick(self, pid):
        pool = self.by_id[pid]
        by_cam = {}
        for i in pool:
            by_cam.setdefault(self.cam_of[i], []).append(i)
        cams = list(by_cam)
        self.rng.shuffle(cams)
        out = []
        while len(out) < self.K:
            progressed = False
            for c in cams:
                if by_cam[c]:
                    out.append(by_cam[c].pop(self.rng.integers(len(by_cam[c]))))
                    progressed = True
                    if len(out) == self.K:
                        break
            if not progressed:                  # кадров меньше K -> с повторением
                out.extend(self.rng.choice(pool, self.K - len(out), replace=True).tolist())
        return out[: self.K]

    def epoch(self):
        order = self.rng.permutation(self.ids)
        batches = []
        for b in range(self.n_batches):
            sel = order[(b * self.P) % len(order): (b * self.P) % len(order) + self.P]
            if len(sel) < self.P:
                sel = np.concatenate([sel, self.rng.choice(self.ids, self.P - len(sel), False)])
            idx = []
            for pid in sel:
                idx.extend(self.pick(int(pid)))
            batches.append(np.array(idx))
        return batches


# ---------------------------------------------------------------- аугментация на GPU

def gpu_augment(x, rng, pad=10, re_prob=0.5):
    """x: float32 N,3,H,W в 0..255 на GPU. Отражение + pad-and-crop + random erasing.

    Вся случайность рождается на CPU одним numpy-генератором: тянуть скаляры с GPU
    через .item() — это синхронизация на каждый вызов, а CPU узла и без нас занят.
    """
    n, _, h, w = x.shape
    flip = torch.from_numpy(rng.random(n) < 0.5).to(x.device)
    x = torch.where(flip[:, None, None, None], x.flip(3), x)

    xp = F.pad(x, (pad, pad, pad, pad))
    ox = rng.integers(0, 2 * pad + 1, n)
    oy = rng.integers(0, 2 * pad + 1, n)
    out = torch.stack([xp[i, :, oy[i]:oy[i] + h, ox[i]:ox[i] + w] for i in range(n)])

    fill = torch.tensor(IMAGENET_MEAN, device=x.device).view(3, 1, 1) * 255.0
    for i in range(n):
        if rng.random() >= re_prob:
            continue
        for _ in range(10):
            area = h * w * rng.uniform(0.02, 0.4)
            ar = rng.uniform(0.3, 3.33)
            eh, ew = int(round((area * ar) ** 0.5)), int(round((area / ar) ** 0.5))
            if eh < h and ew < w:
                y0 = int(rng.integers(0, h - eh))
                x0 = int(rng.integers(0, w - ew))
                out[i, :, y0:y0 + eh, x0:x0 + ew] = fill
                break
    return out


# ---------------------------------------------------------------- модель

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p, self.eps = p, eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            (x.size(-2), x.size(-1))).pow(1.0 / self.p)


class ReIDNet(nn.Module):
    """Вход — RGB 0..255 NCHW: нормировка внутри графа, чтобы ONNX был самодостаточным."""

    def __init__(self, num_classes, arch="resnet50", last_stride=2, w_path=None):
        super().__init__()
        net = getattr(torchvision.models, arch)(weights=None)
        if w_path:
            sd = torch.load(w_path, map_location="cpu", weights_only=True)
            missing = net.load_state_dict(sd, strict=False)
            print("загрузка ImageNet:", missing, flush=True)
        if last_stride == 1:
            net.layer4[0].conv2.stride = (1, 1)
            net.layer4[0].downsample[0].stride = (1, 1)
        self.backbone = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                      net.layer1, net.layer2, net.layer3, net.layer4)
        self.dim = net.fc.in_features
        self.pool = GeM()
        self.bnneck = nn.BatchNorm1d(self.dim)
        self.bnneck.bias.requires_grad_(False)
        nn.init.constant_(self.bnneck.weight, 1.0)
        nn.init.constant_(self.bnneck.bias, 0.0)
        self.classifier = nn.Linear(self.dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) * 255.0)

    def features(self, x):
        x = (x - self.mean) / self.std
        return self.pool(self.backbone(x)).flatten(1)

    def forward(self, x):
        """Инференс: L2-нормированный вектор (то, что уедет в ONNX)."""
        return F.normalize(self.bnneck(self.features(x)), dim=1)

    def forward_train(self, x):
        f = self.features(x)
        return f, self.classifier(self.bnneck(f))


def batch_hard_triplet(feat, labels, margin=0.3):
    d = torch.cdist(feat, feat, p=2)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=feat.device)
    pos = d.masked_fill(~same | eye, -1.0).max(1).values
    neg = d.masked_fill(same, float("inf")).min(1).values
    return F.relu(pos - neg + margin).mean(), (neg > pos).float().mean()


# ---------------------------------------------------------------- прогон векторов

@torch.no_grad()
def embed(model, arr, idx, device, bs=48):
    model.eval()
    out = np.empty((len(idx), model.dim), dtype=np.float32)
    for s in range(0, len(idx), bs):
        chunk = idx[s:s + bs]
        x = torch.from_numpy(arr[chunk]).to(device).permute(0, 3, 1, 2).float()
        out[s:s + len(chunk)] = model(x).cpu().numpy()
    model.train()
    return out


def dev_metrics(model, arr, vids, cams, q_idx, g_idx, device):
    q = embed(model, arr, q_idx, device)
    g = embed(model, arr, g_idx, device)
    res = evaluate(
        scores_from_embeddings(q, g, metric="cosine"),
        query_ids=[int(vids[i]) for i in q_idx],
        gallery_ids=[int(vids[i]) for i in g_idx],
        query_cameras=[int(cams[i]) for i in q_idx],
        gallery_cameras=[int(cams[i]) for i in g_idx],
        known_absent=np.zeros(len(q_idx), dtype=bool),
        threshold=0.0, camera_policy="market", refusal_mode="presence",
    )
    f = res["ranking_full_gallery"]
    return {"mAP": f["mAP"], "Rank-1": f["Rank-1"], "n": f["num_valid_queries"]}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r50")
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--P", type=int, default=8)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3.5e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--n-dev-ids", type=int, default=100)
    ap.add_argument("--last-stride", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=480.0)
    args = ap.parse_args()

    run_dir = ROOT / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    arr, vids, cams = load_crops(ROOT / "data" / "train_crops.npz",
                                 ROOT / "data" / f"crops_{args.size}.npy", args.size)
    train_idx, q_idx, g_idx = make_dev_split(vids, cams, args.n_dev_ids)
    tr_ids = sorted({int(v) for v in vids[train_idx]})
    id2lab = {v: i for i, v in enumerate(tr_ids)}
    labels_all = np.array([id2lab.get(int(v), -1) for v in vids])

    model = ReIDNet(len(tr_ids), args.arch, args.last_stride,
                    str(ROOT / "payload" / "resnet50-imagenet.pth")
                    if args.arch == "resnet50" else None).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    sampler = PKSampler(vids, cams, train_idx, args.P, args.K, SEED)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    aug_rng = np.random.default_rng(SEED + 7)

    ckpt = run_dir / "ckpt.pt"
    start_epoch = 0
    if ckpt.exists():
        st = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        start_epoch = st["epoch"]
        sampler.rng = np.random.default_rng(SEED + start_epoch)
        aug_rng = np.random.default_rng(SEED + 7 + start_epoch)
        print(f"продолжение с эпохи {start_epoch}", flush=True)
    else:
        (run_dir / "config.json").write_text(json.dumps(vars(args), indent=1))
        print(f"train: {len(train_idx)} кадров / {len(tr_ids)} ид.; "
              f"dev: {len(q_idx)} запросов / {len(g_idx)} галерея / "
              f"{args.n_dev_ids} ид.", flush=True)

    log = run_dir / "log.jsonl"
    t_start = time.perf_counter()
    for ep in range(start_epoch, args.epochs):
        if ep < args.warmup:
            lr = args.lr * (0.1 + 0.9 * ep / max(1, args.warmup))
        else:
            prog = (ep - args.warmup) / max(1, args.epochs - args.warmup)
            lr = args.lr * 0.5 * (1 + np.cos(np.pi * prog))
        for gparam in opt.param_groups:
            gparam["lr"] = lr

        batches = sampler.epoch()
        t0 = time.perf_counter()
        agg = np.zeros(4)
        for bi, idx in enumerate(batches):
            x = torch.from_numpy(arr[idx]).to(device, non_blocking=True)
            x = x.permute(0, 3, 1, 2).float()
            x = gpu_augment(x, aug_rng)
            y = torch.from_numpy(labels_all[idx]).to(device)
            f, logits = model.forward_train(x)
            l_id = ce(logits, y)
            l_tri, frac = batch_hard_triplet(f, y, args.margin)
            loss = l_id + l_tri
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            agg += [loss.item(), l_id.item(), l_tri.item(), frac.item()]
        agg /= len(batches)
        rec = {"epoch": ep + 1, "lr": round(lr, 7), "loss": round(agg[0], 4),
               "id": round(agg[1], 4), "tri": round(agg[2], 4),
               "tri_ok": round(agg[3], 4), "sec": round(time.perf_counter() - t0, 1)}
        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs:
            rec["dev"] = dev_metrics(model, arr, vids, cams, q_idx, g_idx, device)
        with open(log, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep + 1, "num_classes": len(tr_ids)}, ckpt)
        if ep + 1 == args.epochs:
            torch.save({"model": model.state_dict(), "epoch": ep + 1,
                        "num_classes": len(tr_ids), "args": vars(args)},
                       run_dir / "final.pt")
            print("ГОТОВО: обучение завершено", flush=True)
            break
        if time.perf_counter() - t_start > args.max_seconds:
            print(f"ПАУЗА: сделано до эпохи {ep+1}/{args.epochs}, чекпойнт сохранён",
                  flush=True)
            break


if __name__ == "__main__":
    main()
