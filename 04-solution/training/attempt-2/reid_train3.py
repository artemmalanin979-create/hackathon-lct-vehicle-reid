"""Уточняющий заход (прогон 2) — то же, что reid_train2.py, с двумя правками.

  ГЛАВНОЕ (самый весомый параметр — скорость обучения): в прогоне 1 замерено по эпохам,
  что dev mAP монотонно падает с ростом LR: 3,5e-5 -> +0,019; 1,925e-4 -> -0,004;
  3,5e-4 -> -0,150 (CH при этом рос 3,12 -> 3,91). Рабочий режим — LR <= 3,5e-5.

  ВТОРОЕ (исправление реализации под текст рецепта, не новый параметр): §7.6 предписывает
  на этапе 1 обучать «только BNNeck + классификатор». В прогоне 1 на этапе 1 обучался ещё
  и `fc.*.0` — Linear 512->256, который является частью ВЕКТОРА, а не головы. То есть
  «линейный пробинг» менял само представление; при случайно инициализированном
  классификаторе (1071 класс, стартовый CE = ln 1071 = 6,98) это его и разрушило.
  Флагом --freeze-fc этап 1 становится настоящим LP: `fc` заморожен целиком (веса и
  статистики BN), меняется только классификатор, вектор по построению не может испортиться.

--- ниже исходная шапка reid_train2.py ---

Вторая попытка дообучения — по исправленному рецепту (`РЕЦЕПТ.md`).

Отличия от провалившегося прогона (`remote/reid_train.py`), по пунктам рецепта:

  §7.1 старт  — OSNet-AIN `vehicle-reid-0001` (веса вынуты из ONNX), а не ImageNet-R50;
  §7.3 сэмплер— батч C камер x P' машин(этой камеры) x K кадров; у каждой машины
                в батче есть кадр «камеры-хозяйки» и кадр другой камеры;
  §7.3 лосс   — MCNL: d(поз) < d(нег с ДРУГОЙ камеры) < d(нег со СВОЕЙ камеры),
                m1 = m2 = 0,1. Позитив выбирается только кросс-камерный;
  §7.5 REA    — СНЯТ (в прошлом прогоне был p=0,5);
  §7.5 MixStyle — после stem (стадия conv2) и после mid (стадия conv3), только там;
  §7.6 расписание — три этапа: LP -> частичная разморозка -> полная, с воротами;
  §8.1 прокси — pseudo-F (Calinski-Harabasz) по камерам считается каждую эпоху:
                если растёт две валидации подряд, прогон надо убивать.

BNNeck и label smoothing в прошлом прогоне уже были — бесплатных пунктов тут нет;
голова OSNet-AIN (Linear 512->256 + BatchNorm1d, две штуки) сама по себе BNNeck:
триплет считается до BN, ID-лосс — после, вектор на инференсе — после.

Метрика мониторинга считается ТЕМ ЖЕ контуром `reid_metrics.py`, своего расчёта нет.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(r"D:\lct-reid")
sys.path.insert(0, str(ROOT))
from reid_metrics import evaluate, scores_from_embeddings  # noqa: E402
from osnet_ain import OSNetAIN  # noqa: E402

SEED = 20260916
FREEZE_FC = False   # см. шапку: --freeze-fc делает этап 1 настоящим LP


# ------------------------------------------------------------------ данные

def load_crops(size: int):
    z = np.load(ROOT / "data" / "train_crops.npz", allow_pickle=False)
    vids, cams = z["vehicle_id"], z["camera_id"]
    arr = np.load(ROOT / "data" / f"crops_{size}.npy", mmap_mode=None)
    assert arr.shape == (len(vids), size, size, 3), arr.shape
    return arr, vids, cams


def make_dev_split(vids, cams, n_dev_ids: int, seed: int):
    """n_dev_ids идентичностей откладываются для мониторинга. Устройство протокола
    копирует наш вал: галерея — 1 кадр с каждой камеры, запросы — остальные."""
    uniq = np.unique(vids)
    rng = np.random.default_rng(seed)
    dev_ids = set(uniq[rng.permutation(len(uniq))[:n_dev_ids]].tolist())
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
            q_here.append(g_here.pop())
        q_idx.extend(q_here)
        g_idx.extend(g_here)
    return train_idx, np.array(sorted(q_idx)), np.array(sorted(g_idx))


class CameraSampler:
    """§7.3: C камер -> P' машин, снятых этой камерой -> K кадров (>=1 с этой камеры,
    >=1 с любой другой). Так у каждого якоря в батче есть и внутрикамерные негативы
    (другие машины той же камеры), и кросс-камерный позитив."""

    def __init__(self, vids, cams, idx, C, P, K, seed, own_k=2):
        self.C, self.P, self.K = C, P, K
        self.own_k = max(1, min(own_k, K - 1))
        self.rng = np.random.default_rng(seed)
        self.cam_of = {int(i): int(cams[i]) for i in idx}
        self.vid_of = {int(i): int(vids[i]) for i in idx}
        self.by_id = {}
        self.by_id_cam = {}
        for i in idx:
            i = int(i)
            self.by_id.setdefault(self.vid_of[i], []).append(i)
            self.by_id_cam.setdefault((self.vid_of[i], self.cam_of[i]), []).append(i)
        # машины, у которых есть кадр с камеры c И хотя бы одна другая камера
        self.ids_of_cam = {}
        for (v, c), lst in self.by_id_cam.items():
            n_cams = len({self.cam_of[i] for i in self.by_id[v]})
            if n_cams >= 2:
                self.ids_of_cam.setdefault(c, []).append(v)
        self.cams = sorted(k for k, v in self.ids_of_cam.items() if len(v) >= P)
        self.n_batches = max(1, len(idx) // (C * P * K))

    def pick(self, vid, cam):
        """K кадров машины vid: >=1 с камеры cam, >=1 с другой, дальше — по кругу."""
        own = list(self.by_id_cam[(vid, cam)])
        other = [i for i in self.by_id[vid] if self.cam_of[i] != cam]
        self.rng.shuffle(own)
        self.rng.shuffle(other)
        n_own = min(self.own_k, len(own))
        out = own[:n_own] + [other[0]]
        rest = own[n_own:] + other[1:]
        self.rng.shuffle(rest)
        out += rest[: self.K - len(out)]
        while len(out) < self.K:
            out.append(int(self.rng.choice(self.by_id[vid])))
        return out[: self.K]

    def epoch(self):
        batches = []
        for _ in range(self.n_batches):
            cams = self.rng.choice(self.cams, min(self.C, len(self.cams)), replace=False)
            idx = []
            for c in cams:
                pool = self.ids_of_cam[int(c)]
                sel = self.rng.choice(pool, self.P, replace=len(pool) < self.P)
                for v in sel:
                    idx.extend(self.pick(int(v), int(c)))
            batches.append(np.array(idx, dtype=np.int64))
        return batches


# ------------------------------------------------------- аугментация на GPU

def gpu_augment(x, rng, pad=10):
    """Отражение + pad-and-crop. Random Erasing СНЯТ (§7.5 рецепта)."""
    n, _, h, w = x.shape
    flip = torch.from_numpy(rng.random(n) < 0.5).to(x.device)
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    xp = F.pad(x, (pad, pad, pad, pad))
    ox = rng.integers(0, 2 * pad + 1, n)
    oy = rng.integers(0, 2 * pad + 1, n)
    return torch.stack([xp[i, :, oy[i]:oy[i] + h, ox[i]:ox[i] + w] for i in range(n)])


def mixstyle(x, p=0.5, alpha=0.1, eps=1e-6, gen=None):
    """MixStyle (arXiv:2104.02008): перемешивание статистик стиля внутри батча.
    Метки доменов не нужны — авторы меряют случайную перестановку не хуже."""
    if torch.rand(1, generator=gen, device="cpu").item() > p:
        return x
    b = x.size(0)
    mu = x.mean(dim=[2, 3], keepdim=True)
    sig = (x.var(dim=[2, 3], keepdim=True) + eps).sqrt()
    xn = (x - mu) / sig
    lam = torch.distributions.Beta(alpha, alpha).sample((b, 1, 1, 1)).to(x.device)
    perm = torch.randperm(b, generator=gen).to(x.device)
    mu_m = mu * lam + mu[perm] * (1 - lam)
    sig_m = sig * lam + sig[perm] * (1 - lam)
    return xn * sig_m + mu_m


# ------------------------------------------------------------------ лоссы

def mcnl_loss(feat, labels, cams, m1=0.1, m2=0.1):
    """MCNL (AAAI'20): d(поз) + m1 < d(нег с ДРУГОЙ камеры) + m2 < d(нег со СВОЕЙ).

    Позитив — сложнейший (самый далёкий) и обязательно кросс-камерный (§7.3);
    оба негатива — сложнейшие (самые близкие) в своей группе.
    Возвращает (loss, доля якорей с выполненным порядком, доля пригодных якорей).
    """
    d = torch.cdist(feat, feat, p=2)
    n = len(labels)
    eye = torch.eye(n, dtype=torch.bool, device=feat.device)
    same_id = labels[:, None] == labels[None, :]
    same_cam = cams[:, None] == cams[None, :]

    pos_cross = same_id & ~eye & ~same_cam
    pos_any = same_id & ~eye
    has_cross = pos_cross.any(1)
    pos_mask = torch.where(has_cross[:, None], pos_cross, pos_any)
    d_pos = d.masked_fill(~pos_mask, -1.0).max(1).values

    neg_other = (~same_id) & (~same_cam)
    neg_same = (~same_id) & same_cam
    inf = torch.tensor(float("inf"), device=feat.device)
    d_no = torch.where(neg_other, d, inf).min(1).values
    d_ns = torch.where(neg_same, d, inf).min(1).values

    has_pos, has_no, has_ns = pos_mask.any(1), neg_other.any(1), neg_same.any(1)
    m_a = has_pos & has_no          # d(поз) < d(нег с другой камеры)
    m_b = has_no & has_ns           # d(нег с другой) < d(нег со своей)
    z = feat.sum() * 0.0
    loss = z
    if m_a.any():
        loss = loss + F.relu(d_pos[m_a] - d_no[m_a] + m1).mean()
    if m_b.any():
        loss = loss + F.relu(d_no[m_b] - d_ns[m_b] + m2).mean()
    full = m_a & m_b
    order = (((d_pos < d_no) & (d_no < d_ns))[full].float().mean()
             if full.any() else z)
    return loss, order, full.float().mean()


# ---------------------------------------------------------- прогон векторов

@torch.no_grad()
def embed(model, arr, idx, device, bs=32):
    model.eval()
    out = np.empty((len(idx), model.dim), dtype=np.float32)
    for s in range(0, len(idx), bs):
        chunk = idx[s:s + bs]
        x = torch.from_numpy(arr[chunk]).to(device).permute(0, 3, 1, 2).float()
        e = model(x)
        out[s:s + len(chunk)] = F.normalize(e, dim=1).cpu().numpy()
    return out


def calinski_harabasz(x, lab):
    """pseudo-F по камерам (§8.1). Матожидание при независимости = 1,0."""
    x = np.asarray(x, dtype=np.float64)
    n, _ = x.shape
    uniq = np.unique(lab)
    k = len(uniq)
    if k < 2 or n <= k:
        return float("nan")
    mean = x.mean(0)
    bss = wss = 0.0
    for c in uniq:
        g = x[lab == c]
        mg = g.mean(0)
        bss += len(g) * float(((mg - mean) ** 2).sum())
        wss += float(((g - mg) ** 2).sum())
    if wss <= 0:
        return float("inf")
    return (bss / (k - 1)) / (wss / (n - k))


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
    allv = np.vstack([q, g])
    allc = np.concatenate([cams[q_idx], cams[g_idx]])
    return {"mAP": round(f["mAP"], 5), "Rank-1": round(f["Rank-1"], 5),
            "CH": round(calinski_harabasz(allv, allc), 4), "n": f["num_valid_queries"]}


# -------------------------------------------------------------- заморозка

def set_stage(model, stage):
    """Что обучается на каждом этапе (§7.6). Замороженные модули переводятся в
    .eval() — так делает и fast-reid: BN в них тоже заморожен."""
    groups = {
        1: [],                                  # только головы
        2: ["conv4", "conv5"],                  # + верхние блоки
        3: ["conv2", "conv3", "conv4", "conv5", "conv1", "input_IN", "pool2", "pool3"],
    }[stage]
    head = ["classifier"] if (FREEZE_FC and stage == 1) else ["fc", "classifier"]
    for name, mod in model.named_children():
        train_it = name in groups or name in head
        for p in mod.parameters():
            p.requires_grad_(train_it)
        mod.train(train_it)
    return groups


def param_groups(model, stage, lr):
    head, back = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if name.startswith(("fc.", "classifier.")) else back).append(p)  # noqa: E501
    if stage == 1:
        return [{"params": head, "lr": lr, "tag": "head"}]
    back_mult = 0.1 if stage == 2 else 1.0
    return [{"params": head, "lr": lr, "tag": "head"},
            {"params": back, "lr": lr * back_mult, "tag": "backbone"}]


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ain_v1")
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--size", type=int, default=208)
    ap.add_argument("--C", type=int, default=4)
    ap.add_argument("--P", type=int, default=4)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--own-k", type=int, default=2, help="кадров с камеры-хозяйки в K")
    ap.add_argument("--lr", type=float, default=3.5e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--m1", type=float, default=0.1)
    ap.add_argument("--m2", type=float, default=0.1)
    ap.add_argument("--w-id", type=float, default=1.0)
    ap.add_argument("--mixstyle-p", type=float, default=0.5)
    ap.add_argument("--n-dev-ids", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--max-batches", type=int, default=0, help="дымовой прогон: сколько батчей")
    ap.add_argument("--init", default="osnet_ain_start.pt")
    ap.add_argument("--resume-from", default=None, help="stageN.pt предыдущего этапа")
    ap.add_argument("--freeze-fc", action="store_true",
                    help="этап 1: заморозить fc (Linear+BN вектора), учить только классификатор")
    args = ap.parse_args()
    global FREEZE_FC
    FREEZE_FC = bool(args.freeze_fc)

    run_dir = ROOT / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(SEED + args.stage)
    gen = torch.Generator().manual_seed(SEED + args.stage)

    arr, vids, cams = load_crops(args.size)
    train_idx, q_idx, g_idx = make_dev_split(vids, cams, args.n_dev_ids, SEED)
    tr_ids = sorted({int(v) for v in vids[train_idx]})
    id2lab = {v: i for i, v in enumerate(tr_ids)}
    labels_all = np.array([id2lab.get(int(v), -1) for v in vids])

    model = OSNetAIN(num_classes=len(tr_ids)).to(device)
    ckpt = run_dir / f"stage{args.stage}_ckpt.pt"
    start_epoch = 0
    if ckpt.exists():
        st = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        start_epoch = st["epoch"]
        print(f"продолжение этапа {args.stage} с эпохи {start_epoch}", flush=True)
    else:
        src = run_dir / args.resume_from if args.resume_from else ROOT / args.init
        sd = torch.load(src, map_location="cpu", weights_only=True)
        sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"старт этапа {args.stage} из {src}; нет в файле: {list(missing)[:4]}...; "
              f"лишних: {list(unexpected)[:4]}", flush=True)

    groups = set_stage(model, args.stage)
    opt = torch.optim.Adam(param_groups(model, args.stage, args.lr),
                           weight_decay=args.wd, foreach=False)
    if ckpt.exists():
        opt.load_state_dict(st["opt"])
    sampler = CameraSampler(vids, cams, train_idx, args.C, args.P, args.K,
                            SEED + 100 * args.stage + start_epoch, args.own_k)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    aug_rng = np.random.default_rng(SEED + 7 + 100 * args.stage + start_epoch)

    if start_epoch == 0:
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        cfg = {**vars(args), "trainable_params": n_tr, "unfrozen": groups,
               "batch": args.C * args.P * args.K, "n_train_ids": len(tr_ids),
               "n_train_frames": len(train_idx), "n_cams_usable": len(sampler.cams),
               "batches_per_epoch": sampler.n_batches}
        (run_dir / f"stage{args.stage}_config.json").write_text(json.dumps(cfg, indent=1))
        print(json.dumps(cfg, ensure_ascii=False), flush=True)
        rec0 = {"stage": args.stage, "epoch": 0,
                "dev": dev_metrics(model, arr, vids, cams, q_idx, g_idx, device)}
        set_stage(model, args.stage)
        with open(run_dir / f"stage{args.stage}_log.jsonl", "a") as fh:
            fh.write(json.dumps(rec0) + "\n")
        print(json.dumps(rec0), flush=True)

    log = run_dir / f"stage{args.stage}_log.jsonl"
    base_lrs = [g["lr"] for g in opt.param_groups]
    for ep in range(start_epoch, args.epochs):
        if ep < args.warmup:
            scale = 0.1 + 0.9 * ep / max(1, args.warmup)
        else:
            prog = (ep - args.warmup) / max(1, args.epochs - args.warmup)
            scale = 0.5 * (1 + np.cos(np.pi * prog))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * scale

        batches = sampler.epoch()
        if args.max_batches:
            batches = batches[: args.max_batches]
        t0 = time.perf_counter()
        agg = np.zeros(5)
        for idx in batches:
            x = torch.from_numpy(arr[idx]).to(device, non_blocking=True)
            x = x.permute(0, 3, 1, 2).float()
            x = gpu_augment(x, aug_rng)
            y = torch.from_numpy(labels_all[idx]).to(device)
            c = torch.from_numpy(cams[idx].astype(np.int64)).to(device)

            frozen = args.stage < 3
            with torch.set_grad_enabled(not frozen):
                h = model.stem(x)
                h = mixstyle(h, p=args.mixstyle_p, gen=gen)
                h = model.mid(h)
                h = mixstyle(h, p=args.mixstyle_p, gen=gen)
            if frozen and args.stage == 2:
                h = h.detach().requires_grad_(False)
            if args.stage == 1:
                with torch.no_grad():
                    fmap = model.top(h)
                g_feat = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
            else:
                fmap = model.top(h)
                g_feat = F.adaptive_avg_pool2d(fmap, 1).flatten(1)

            pre = torch.cat([f[0](g_feat) for f in model.fc], dim=1)   # до BN — триплет
            emb = torch.cat([f(g_feat) for f in model.fc], dim=1)      # после BN — ID
            logits = model.classifier(emb)
            l_id = ce(logits, y)
            l_tri, order, usable = mcnl_loss(pre, y, c, args.m1, args.m2)
            loss = args.w_id * l_id + l_tri
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            agg += [loss.item(), l_id.item(), l_tri.item(), order.item(), usable.item()]
        agg /= len(batches)

        rec = {"stage": args.stage, "epoch": ep + 1, "lr": round(base_lrs[0] * scale, 8),
               "loss": round(agg[0], 4), "id": round(agg[1], 4), "mcnl": round(agg[2], 4),
               "order_ok": round(agg[3], 4), "usable": round(agg[4], 4),
               "sec": round(time.perf_counter() - t0, 1),
               "gpu_MiB": round(torch.cuda.max_memory_allocated() / 2 ** 20)}
        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs:
            rec["dev"] = dev_metrics(model, arr, vids, cams, q_idx, g_idx, device)
            set_stage(model, args.stage)
        with open(log, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

        tmp = run_dir / f"stage{args.stage}_ckpt.tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep + 1, "num_classes": len(tr_ids)}, tmp)
        os.replace(tmp, ckpt)

    torch.save(model.state_dict(), run_dir / f"stage{args.stage}.pt")
    print(f"ГОТОВО: этап {args.stage} завершён ({args.epochs} эпох)", flush=True)


if __name__ == "__main__":
    main()
