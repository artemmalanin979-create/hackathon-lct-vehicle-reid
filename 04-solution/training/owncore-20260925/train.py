"""Two-stage training of the standalone pixel-to-descriptor vehicle Re-ID core.

The frozen protocol supplies all hyperparameters and input hashes. Only the
historical own dev100 split selects a checkpoint; reused validation is never
read by this trainer.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # Windows python312._pth omits the script directory.
from core import CrossViewCore
from dataset import CameraAwarePKSampler, CropDataset, load_combined

def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_hash(path: str | Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"hash mismatch for {path}: {actual} != {expected}")
    return actual


def require_vram_below(peak_bytes: int, limit_gib: float = 3.6) -> None:
    if peak_bytes >= limit_gib * 1024**3:
        raise RuntimeError(f"peak reserved VRAM {peak_bytes} reached {limit_gib} GiB limit")


def require_rss_below(peak_bytes: int, limit_mib: int) -> None:
    if peak_bytes >= int(limit_mib) * 1024**2:
        raise RuntimeError(f"peak RSS {peak_bytes} reached {limit_mib} MiB limit")


def peak_rss_bytes() -> int:
    """OS peak working set, including native Torch allocations (not tracemalloc)."""
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        if not ok:
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)
    with Path("/proc/self/status").open(encoding="ascii") as stream:
        for line in stream:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    raise OSError("/proc/self/status lacks VmHWM")


def select_protocol_device(requested: str | None, cuda_available: bool | None = None) -> torch.device:
    """Never silently switch the hardware fixed before training."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        available = torch.cuda.is_available() if cuda_available is None else cuda_available
        if not available:
            raise RuntimeError("protocol requires CUDA, but CUDA is unavailable")
        return torch.device("cuda")
    raise ValueError("protocol training.device must be exactly 'cpu' or 'cuda'")


def require_protocol_memory(stats: dict, device: torch.device, cfg: dict) -> None:
    if device.type == "cuda":
        require_vram_below(stats["peak_vram_bytes"])
    else:
        require_rss_below(stats["peak_rss_bytes"], cfg["peak_rss_limit_mib"])


def load_evaluator(path: str | Path, expected_sha256: str, scope_sha256: str):
    """Load the exact archived evaluator plus its required sibling offline."""
    path = Path(path)
    checked_hash(path, expected_sha256)
    checked_hash(path.with_name("scope_metrics.py"), scope_sha256)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("owncore_reid_metrics", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_training_stage(model: CrossViewCore, stage: int) -> None:
    if stage not in (1, 2):
        raise ValueError("training stage must be 1 or 2")
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(stage == 2)
    if stage == 1:
        # Frozen ImageNet BN statistics must not drift during linear probing.
        model.backbone.eval()
    elif model.training:
        model.backbone.train()


def cross_camera_batch_hard(descriptor: torch.Tensor, ids: torch.Tensor,
                            cameras: torch.Tensor, margin: float) -> torch.Tensor:
    """Hardest different-camera positive versus nearest different-ID negative."""
    z = F.normalize(descriptor.float(), p=2, dim=1)
    distance = 1.0 - z @ z.T
    same_id = ids[:, None] == ids[None, :]
    positive = same_id & (cameras[:, None] != cameras[None, :])
    negative = ~same_id
    valid = positive.any(dim=1) & negative.any(dim=1)
    if not bool(valid.any()):
        return z.sum() * 0.0
    hardest_positive = distance.masked_fill(~positive, -torch.inf).max(dim=1).values
    hardest_negative = distance.masked_fill(~negative, torch.inf).min(dim=1).values
    return F.relu(hardest_positive[valid] - hardest_negative[valid] + margin).mean()


def augment(images: torch.Tensor) -> torch.Tensor:
    """Small label-independent RGB changes, applied to fit batches only."""
    count = images.shape[0]
    flip = torch.rand(count, device=images.device) < 0.5
    images = torch.where(flip[:, None, None, None], images.flip(-1), images)
    brightness = 0.9 + 0.2 * torch.rand(count, 1, 1, 1, device=images.device)
    contrast = 0.9 + 0.2 * torch.rand(count, 1, 1, 1, device=images.device)
    means = images.mean(dim=(2, 3), keepdim=True)
    return ((images - means) * contrast + means).mul(brightness).clamp(0.0, 255.0)


def train_epoch(model: CrossViewCore, loader: DataLoader, optimizer: torch.optim.Optimizer,
                device: torch.device, margin: float, metric_weight: float,
                smoothing: float, wall_deadline: float) -> dict:
    model.train()
    if not any(parameter.requires_grad for parameter in model.backbone.parameters()):
        model.backbone.eval()
    sums = {"loss": 0.0, "classification": 0.0, "triplet": 0.0}
    start = time.monotonic()
    batches = 0
    for images, labels, cameras, _ in loader:
        if time.monotonic() > wall_deadline:
            raise TimeoutError("frozen wall limit exceeded")
        images = augment(images.to(device, non_blocking=True))
        labels, cameras = labels.to(device), cameras.to(device)
        optimizer.zero_grad(set_to_none=True)
        descriptor = model(images)
        ce = F.cross_entropy(model.classify(descriptor), labels, label_smoothing=smoothing)
        triplet = cross_camera_batch_hard(descriptor, labels, cameras, margin)
        loss = ce + metric_weight * triplet
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite training loss")
        loss.backward()
        if not all(bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()
                   if parameter.grad is not None):
            raise FloatingPointError("nonfinite training gradient")
        optimizer.step()
        for key, value in (("loss", loss), ("classification", ce), ("triplet", triplet)):
            sums[key] += float(value.detach())
        batches += 1
    if batches == 0:
        raise ValueError("zero training batches")
    return {key: value / batches for key, value in sums.items()} | {
        "batches": batches, "seconds": time.monotonic() - start,
        "peak_rss_bytes": peak_rss_bytes(),
        "peak_vram_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "peak_tensor_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }


@torch.no_grad()
def extract(model: CrossViewCore, dataset: CropDataset, batch: int,
            device: torch.device) -> np.ndarray:
    model.eval()
    rows = []
    for images, _, _, _ in DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0):
        rows.append(model(images.to(device)).cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(rows, axis=0)


def dev_cosine_map(model: CrossViewCore, data, evaluator, batch: int,
                   device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    q, g = data.dev_query_indices, data.dev_gallery_indices
    query = extract(model, CropDataset(data, q), batch, device)
    gallery = extract(model, CropDataset(data, g), batch, device)
    q_ids, g_ids = data.vehicle_ids[q], data.vehicle_ids[g]
    scores = evaluator.scores_from_embeddings(query, gallery, metric="cosine")
    report = evaluator.evaluate(scores, query_ids=q_ids, gallery_ids=g_ids,
                      query_cameras=data.camera_ids[q], gallery_cameras=data.camera_ids[g],
                      known_absent=~np.isin(q_ids, g_ids), threshold=0.0,
                      camera_policy="market", refusal_mode="presence")
    value = float(report["ranking_full_gallery"]["mAP"])
    if not math.isfinite(value):
        raise FloatingPointError("nonfinite dev cosine mAP")
    return value, query, gallery


def _optimizer(model: CrossViewCore, cfg: dict, stage: int) -> torch.optim.Optimizer:
    heads = [*model.global_head.parameters(), *model.top_head.parameters(),
             *model.bottom_head.parameters(), *model.classifier.parameters()]
    if stage == 1:
        groups = [{"params": heads, "lr": float(cfg["stage1_head_lr"])}]
    else:
        groups = [{"params": model.backbone.parameters(), "lr": float(cfg["stage2_backbone_lr"])},
                  {"params": heads, "lr": float(cfg["stage2_head_lr"])}]
    return torch.optim.AdamW(groups, weight_decay=float(cfg["weight_decay"]))


def smoke_one_batch(model: CrossViewCore, loader: DataLoader, cfg: dict,
                    device: torch.device, wall_deadline: float) -> dict:
    """Exercise one real P8K4 batch, including unfrozen backbone backward/update."""
    model.train()
    set_training_stage(model, 2)
    optimizer = _optimizer(model, cfg, 2)
    batch = next(iter(loader))
    _, labels, cameras, rows = batch
    if len(rows) != 32 or len(torch.unique(labels)) != 8:
        raise ValueError("smoke batch is not P8K4")
    cross_camera_anchors = sum(bool(torch.unique(cameras[labels == label]).numel() >= 2)
                               for label in labels)
    if cross_camera_anchors == 0:
        raise ValueError("smoke batch has no cross-camera positive")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stats = train_epoch(model, [batch], optimizer, device, margin=0.3,
                        metric_weight=0.5, smoothing=0.1, wall_deadline=wall_deadline)
    named = {"backbone": model.backbone[0][0].weight,
             "global": model.global_head.weight,
             "top": model.top_head.weight,
             "bottom": model.bottom_head.weight,
             "classifier": model.classifier.weight}
    gradients = {name: bool(parameter.grad is not None and parameter.grad.abs().sum() > 0)
                 for name, parameter in named.items()}
    if not all(gradients.values()):
        raise RuntimeError(f"empty gradient in smoke: {gradients}")
    require_protocol_memory(stats, device, cfg)
    return {"status": "PASS", "stats": stats, "gradients": gradients,
            "distinct_id_count": int(torch.unique(labels).numel()),
            "cross_camera_anchor_count": cross_camera_anchors,
            "sample_rows": [int(row) for row in rows]}


def make_checkpoint(model_state: dict, num_ids: int, hashes: dict,
                    best_epoch: int, best_map: float) -> dict:
    required = ("protocol", "imagenet", "core_py", "dataset_py", "train_py")
    missing = set(required) - set(hashes)
    if missing:
        raise ValueError(f"missing checkpoint provenance hashes: {sorted(missing)}")
    return {"model_state_dict": model_state, "num_ids": num_ids,
            "protocol_sha256": hashes["protocol"],
            "model_config": {"backbone": "mobilenet_v3_small", "descriptor_dim": 512,
                             "input_size": 208},
            "best_epoch": best_epoch, "dev_cosine_mAP": best_map,
            "imagenet_sha256": hashes["imagenet"],
            "source_sha256": {key: hashes[key] for key in ("core_py", "dataset_py", "train_py")}}


def _atomic_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_best_checkpoint(out: Path, payload: dict) -> dict:
    """Persist a recoverable best while the run remains explicitly PARTIAL."""
    with tempfile.NamedTemporaryFile(mode="wb", dir=out, prefix=".checkpoint.",
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, out / "checkpoint.pt")
    finally:
        temporary.unlink(missing_ok=True)
    status = {"status": "PARTIAL", "best_epoch": payload["best_epoch"],
              "dev_cosine_mAP": payload["dev_cosine_mAP"],
              "protocol_sha256": payload["protocol_sha256"],
              "checkpoint_sha256": sha256(out / "checkpoint.pt")}
    _atomic_json(out / "checkpoint-status.json", status)
    return status


def probe_stage2_batch(model: CrossViewCore, loader: DataLoader,
                       device: torch.device) -> dict:
    """Time an unfrozen backward pass without changing weights or BN statistics."""
    images, labels, cameras, _ = next(iter(loader))
    images, labels, cameras = images.to(device), labels.to(device), cameras.to(device)
    model.eval()
    set_training_stage(model, 2)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.monotonic()
    descriptor = model(images)
    loss = F.cross_entropy(model.classify(descriptor), labels, label_smoothing=0.1)
    loss = loss + 0.5 * cross_camera_batch_hard(descriptor, labels, cameras, 0.3)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite stage-2 timing probe loss")
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - start
    peak = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    model.zero_grad(set_to_none=True)
    return {"seconds": elapsed, "peak_vram_bytes": peak, "peak_rss_bytes": peak_rss_bytes()}


def projected_runtime(elapsed: float, stage1_epoch_seconds: float,
                      stage2_probe_seconds: float, steps_per_epoch: int,
                      remaining_stage2_epochs: int) -> float:
    """Conservative estimate from measured stage-1 throughput and stage-2 backward."""
    stage1_batch = stage1_epoch_seconds / steps_per_epoch
    stage2_batch = max(stage1_batch, stage2_probe_seconds)
    return elapsed + remaining_stage2_epochs * steps_per_epoch * stage2_batch * 1.25


def synthetic_smoke() -> dict:
    torch.manual_seed(1)
    model = CrossViewCore(num_ids=8)
    images = torch.randint(0, 256, (8, 3, 64, 64), dtype=torch.int32).float()
    ids = torch.arange(4).repeat_interleave(2)
    cameras = torch.tensor([0, 1] * 4)
    evidence = []
    for stage in (1, 2):
        model.train()
        set_training_stage(model, stage)
        descriptor = model(images)
        loss = F.cross_entropy(model.classify(descriptor), ids) + 0.5 * cross_camera_batch_hard(
            descriptor, ids, cameras, 0.3)
        loss.backward()
        backbone_grad = any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                            for p in model.backbone.parameters())
        if not bool(torch.isfinite(loss)) or backbone_grad != (stage == 2):
            raise AssertionError("synthetic smoke failed loss or stage-specific backbone gradient")
        evidence.append({"stage": stage, "loss": float(loss.detach()),
                         "backbone_gradient": backbone_grad,
                         "descriptor_shape": list(descriptor.shape),
                         "mean_norm": float(descriptor.norm(dim=1).mean().detach())})
        model.zero_grad(set_to_none=True)
    return {"status": "PASS", "device": "cpu", "stages": evidence}


def run(args: argparse.Namespace) -> dict:
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    evaluator = load_evaluator(args.evaluator,
                               protocol["evaluation"]["evaluator_sha256"],
                               protocol["evaluation"]["evaluator_scope_dependency_sha256"])
    if args.synthetic_smoke:
        return synthetic_smoke() | {"protocol_sha256": sha256(protocol_path),
                                    "evaluator_sha256": sha256(args.evaluator)}
    if protocol["status"] != "frozen_before_training" or protocol["training"]["attempts"] != 1:
        raise ValueError("training requires the frozen single-attempt protocol")
    cfg = protocol["training"]
    if (cfg["batch_size"] != 32 or
            not cfg["sampler"].startswith("8 vehicle IDs x 4 frames") or
            cfg["stage1_epochs"] != 2 or cfg["stage2_epochs"] != 12):
        raise ValueError("trainer supports only frozen P8K4 batch32 protocol")
    device = select_protocol_device(cfg.get("device"))
    if device.type == "cpu":
        threads = int(cfg["threads"])
        if threads < 1 or int(cfg["peak_rss_limit_mib"]) < 1:
            raise ValueError("CPU threads and peak RSS limit must be positive")
        torch.set_num_threads(threads)
    hashes = {"protocol": sha256(protocol_path), "evaluator": sha256(args.evaluator),
              "scope_metrics": sha256(Path(args.evaluator).with_name("scope_metrics.py")),
              "core_py": sha256(Path(__file__).with_name("core.py")),
              "dataset_py": sha256(Path(__file__).with_name("dataset.py")),
              "train_py": sha256(__file__)}
    for name, path, expected in (
        ("train_npz", args.data_npz, protocol["data"]["train_source_sha256"]),
        ("train_crops", args.data_raw, protocol["data"]["train_crops_sha256"]),
        ("split", args.split, protocol["data"]["split_sha256"]),
        ("imagenet", args.imagenet, protocol["architecture"]["weights_sha256"]),
    ):
        hashes[name] = checked_hash(path, expected)
    data = load_combined(args.data_npz, args.data_raw, args.split)
    if (len(data.train_indices) != protocol["data"]["fit_rows_expected"] or
            len(data.dev_indices) != protocol["data"]["dev_rows_expected"] or
            len(np.unique(data.dev_vehicle_ids)) != protocol["data"]["dev_ids_expected"]):
        raise ValueError("split cardinalities differ from frozen protocol")
    if len(data.dev_query_indices) == 0 or len(data.dev_gallery_indices) == 0:
        raise ValueError("dev query/gallery cannot be empty")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "inputs.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = CrossViewCore(len(data.label_to_vehicle)).to(device)
    model.load_imagenet(args.imagenet, hashes["imagenet"])
    sampler = CameraAwarePKSampler(data.train_vehicle_ids, data.train_camera_ids,
                                   np.arange(len(data.train_indices)), p=8, k=4,
                                   batches=int(cfg["steps_per_epoch"]), seed=seed)
    loader = DataLoader(CropDataset(data, data.train_indices, labels=True),
                        batch_sampler=sampler, num_workers=0, pin_memory=device.type == "cuda")
    moscow = timezone(timedelta(hours=3))
    stop_at = datetime.fromisoformat(cfg["stop_no_later_than_moscow"]).replace(tzinfo=moscow)
    seconds_until_stop = (stop_at - datetime.now(timezone.utc)).total_seconds()
    wall_budget = min(float(cfg["wall_limit_seconds"]), seconds_until_stop)
    if wall_budget <= 0:
        raise TimeoutError("frozen Moscow stop deadline has passed")
    run_start = time.monotonic()
    wall_deadline = run_start + wall_budget
    if args.smoke_only:
        smoke = smoke_one_batch(model, loader, cfg, device, wall_deadline)
        smoke.update(protocol_sha256=hashes["protocol"], input_sha256=hashes,
                     device=str(device), torch=torch.__version__)
        (out / "smoke.json").write_text(json.dumps(smoke, indent=2) + "\n", encoding="utf-8")
        return smoke
    rows: list[dict] = []
    best_map, best_epoch, best_state, best_q, best_g = -math.inf, -1, None, None, None
    epoch_index = 0
    for stage, count in ((1, int(cfg["stage1_epochs"])), (2, int(cfg["stage2_epochs"]))):
        model.train()
        set_training_stage(model, stage)
        optimizer = _optimizer(model, cfg, stage)
        for local_epoch in range(count):
            sampler.set_epoch(epoch_index)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            stats = train_epoch(model, loader, optimizer, device, margin=0.3,
                                metric_weight=0.5, smoothing=0.1, wall_deadline=wall_deadline)
            record = {"stage": stage, "stage_epoch": local_epoch + 1,
                      "global_epoch": epoch_index + 1, **stats}
            try:
                require_protocol_memory(stats, device, cfg)
            except RuntimeError as exc:
                record.update(status="FAIL", reason=str(exc))
                with (out / "epochs.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                raise
            if stage == 2:
                dev_map, query, gallery = dev_cosine_map(model, data, evaluator, batch=32, device=device)
                record["dev_cosine_mAP"] = dev_map
                if dev_map > best_map:
                    best_map, best_epoch = dev_map, epoch_index + 1
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_q, best_g = query, gallery
                    save_best_checkpoint(out, make_checkpoint(
                        best_state, len(data.label_to_vehicle), hashes, best_epoch, best_map))
            rows.append(record)
            with (out / "epochs.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            epoch_index += 1
        if stage == 1:
            # The pilot is a real gate, not an after-the-fact diagnostic.
            reasons = []
            if rows[1]["loss"] >= rows[0]["loss"]:
                reasons.append("stage-1 loss did not decrease")
            for row in rows[:2]:
                try:
                    require_protocol_memory(row, device, cfg)
                except RuntimeError as exc:
                    reasons.append(str(exc))
            probe = None
            projected_seconds = None
            if not reasons:
                probe = probe_stage2_batch(model, loader, device)
                try:
                    require_protocol_memory(probe, device, cfg)
                except RuntimeError as exc:
                    reasons.append(str(exc))
            if not reasons:
                projected_seconds = projected_runtime(
                    time.monotonic() - run_start,
                    sum(row["seconds"] for row in rows[:2]) / 2.0,
                    probe["seconds"], int(cfg["steps_per_epoch"]), int(cfg["stage2_epochs"]))
                if projected_seconds > wall_budget:
                    reasons.append("projected 14-epoch runtime exceeds frozen wall/deadline budget")
            pilot = {"status": "FAIL" if reasons else "PASS", "reasons": reasons,
                     "device": device.type, "stage1": rows[:2], "stage2_probe": probe,
                     "projected_total_seconds": projected_seconds,
                     "wall_budget_seconds": wall_budget,
                     "projection_safety_factor": 1.25}
            (out / "pilot.json").write_text(json.dumps(pilot, indent=2) + "\n", encoding="utf-8")
            if reasons:
                raise RuntimeError(f"frozen pilot gate failed: {reasons}")
    if best_state is None:
        raise RuntimeError("no stage-2 checkpoint selected")
    checkpoint_path = out / "checkpoint.pt"
    if not checkpoint_path.is_file():
        raise RuntimeError("selected best checkpoint was not persisted")
    q, g = data.dev_query_indices, data.dev_gallery_indices
    np.savez(out / "dev_embeddings.npz", query_embeddings=best_q, gallery_embeddings=best_g,
             query_indices=q, gallery_indices=g,
             query_vehicle_ids=data.vehicle_ids[q], gallery_vehicle_ids=data.vehicle_ids[g],
             query_camera_ids=data.camera_ids[q], gallery_camera_ids=data.camera_ids[g],
             known_absent=~np.isin(data.vehicle_ids[q], data.vehicle_ids[g]),
             protocol_sha256=np.array(hashes["protocol"]))
    summary = {"status": "PASS", "protocol_sha256": hashes["protocol"],
               "checkpoint_sha256": sha256(checkpoint_path),
               "dev_embeddings_sha256": sha256(out / "dev_embeddings.npz"),
               "best_epoch": best_epoch, "dev_cosine_mAP": best_map,
               "num_ids": len(data.label_to_vehicle), "device": str(device),
               "torch": torch.__version__, "numpy": np.__version__,
               "input_sha256": hashes, "epochs": rows}
    (out / "training.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _atomic_json(out / "checkpoint-status.json",
                 {"status": "COMPLETE", "best_epoch": best_epoch,
                  "dev_cosine_mAP": best_map, "protocol_sha256": hashes["protocol"],
                  "checkpoint_sha256": summary["checkpoint_sha256"],
                  "training_sha256": sha256(out / "training.json")})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true",
                        help="one real protocol-device P8K4 fit batch, new output directory, no epochs")
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--evaluator", type=Path, required=True,
                        help="Archived reid_metrics.py with sibling scope_metrics.py")
    parser.add_argument("--data-npz", type=Path)
    parser.add_argument("--data-raw", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--imagenet", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not args.synthetic_smoke and any(getattr(args, key) is None
                                        for key in ("data_npz", "data_raw", "split", "imagenet", "out")):
        parser.error("--data-npz, --data-raw, --split, --imagenet and --out are required for training")
    result = run(args)
    print(json.dumps({key: value for key, value in result.items() if key != "epochs"}, sort_keys=True))


if __name__ == "__main__":
    main()
