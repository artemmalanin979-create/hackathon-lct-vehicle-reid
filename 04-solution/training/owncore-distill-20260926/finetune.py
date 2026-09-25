#!/usr/bin/env python3
"""Bounded teacher-distillation continuation of the standalone CrossViewCore.

Only own fit or verified combined fit enters gradients. The teacher is a frozen
target file and is never part of the exported image encoder. Dev100 selects the
checkpoint; this trainer has no validation-image or validation-label input.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "owncore-20260925"
sys.path.insert(0, str(BASE))
from core import CrossViewCore  # noqa: E402
from dataset import CameraAwarePKSampler, load_combined  # noqa: E402
from train import (  # noqa: E402
    _atomic_json, augment, checked_hash, cross_camera_batch_hard,
    dev_cosine_map, load_evaluator, peak_rss_bytes, require_rss_below,
    save_best_checkpoint, set_training_stage, sha256,
)

MODEL_CONFIG = {"backbone": "mobilenet_v3_small", "descriptor_dim": 512,
                "input_size": 208}
TEACHER_FIELDS = {"row_indices", "target", "vehicle_id", "camera_id",
                  "image_id", "P", "m"}


def load_teacher_targets(path: Path, expected_sha256: str,
                         fit_indices: np.ndarray, metadata: dict) -> np.ndarray:
    """Refuse any teacher target that is not aligned with every exact fit row."""
    checked_hash(path, expected_sha256)
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    if fit_indices.ndim != 1 or len(fit_indices) == 0 or not np.all(np.diff(fit_indices) > 0):
        raise ValueError("fit row_indices must be nonempty, unique and ascending")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != TEACHER_FIELDS:
            raise ValueError(f"teacher fields differ from frozen schema: {sorted(archive.files)}")
        rows = archive["row_indices"]
        if rows.dtype != np.int64 or not np.array_equal(rows, fit_indices):
            raise ValueError("teacher row_indices differ from exact ascending fit rows")
        for key in ("vehicle_id", "camera_id", "image_id"):
            actual = archive[key]
            source = np.asarray(metadata[key])
            if source.ndim != 1 or len(source) <= int(fit_indices[-1]):
                raise ValueError(f"source {key} metadata is incomplete")
            if actual.shape != (len(rows),) or not np.array_equal(actual, source[rows]):
                raise ValueError(f"teacher {key} differs from source rows")
        target = archive["target"]
        P, m = archive["P"], archive["m"]
        if target.dtype != np.float32 or target.shape != (len(rows), 512):
            raise ValueError("teacher target must be float32 fit_rows x 512")
        if P.dtype != np.float64 or P.shape != (512, 512) or m.dtype != np.float64 or m.shape != (512,):
            raise ValueError("teacher P/m must be float64 512x512/512")
        if not (np.isfinite(target).all() and np.isfinite(P).all() and np.isfinite(m).all()):
            raise ValueError("teacher targets or whitening P/m are not finite")
        norms = np.linalg.norm(target, axis=1)
        if not np.all(np.abs(norms - 1.0) <= 1e-4):
            raise ValueError("teacher targets must have unit L2 norm")
        return np.array(target, copy=True)


def load_warm_start(path: Path, expected_sha256: str, expected_num_ids: int) -> CrossViewCore:
    """Hash and strictly reload a compatible checkpoint before any optimizer exists."""
    checked_hash(path, expected_sha256)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("num_ids") != expected_num_ids:
        raise ValueError("warm-start checkpoint num_ids differs from own classifier")
    if payload.get("model_config") != MODEL_CONFIG:
        raise ValueError("warm-start checkpoint model_config differs from CrossViewCore")
    if not isinstance(payload.get("protocol_sha256"), str) or len(payload["protocol_sha256"]) != 64:
        raise ValueError("warm-start checkpoint lacks source protocol SHA-256")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("warm-start checkpoint lacks model_state_dict")
    model = CrossViewCore(expected_num_ids)
    model.load_state_dict(state, strict=True)
    return model


class DistillDataset(Dataset):
    """Each training position retains its raw row, own CE label and teacher vector."""

    def __init__(self, data, targets: np.ndarray, own_fit_ids: np.ndarray):
        self.data = data
        self.targets = np.asarray(targets, dtype=np.float32)
        self.own_fit_ids = np.asarray(own_fit_ids, dtype=np.int64)
        if self.targets.shape != (len(data.train_indices), 512):
            raise ValueError("teacher target rows are not aligned with fit positions")
        if len(self.own_fit_ids) < 2 or not np.all(np.diff(self.own_fit_ids) > 0):
            raise ValueError("own classifier IDs must be sorted and unique")
        vids = data.vehicle_ids[data.train_indices]
        self.class_labels = np.full(len(vids), -1, dtype=np.int64)
        own_positions = np.flatnonzero(vids < 100000)
        candidates = np.searchsorted(self.own_fit_ids, vids[own_positions])
        if np.any(candidates >= len(self.own_fit_ids)) or not np.array_equal(
                self.own_fit_ids[candidates], vids[own_positions]):
            raise ValueError("own fit identity missing from classifier")
        self.class_labels[own_positions] = candidates

    def __len__(self) -> int:
        return len(self.data.train_indices)

    def __getitem__(self, position: int):
        row = int(self.data.train_indices[position])
        image = torch.from_numpy(np.array(self.data.crops[row], copy=True)).permute(2, 0, 1).float()
        return (image, int(self.class_labels[position]), int(self.data.vehicle_ids[row]),
                int(self.data.camera_ids[row]), row,
                torch.from_numpy(np.array(self.targets[position], copy=True)))


def classification_on_own(logits: torch.Tensor, own_labels: torch.Tensor,
                          smoothing: float) -> tuple[torch.Tensor, int]:
    own = own_labels >= 0
    count = int(own.sum().item())
    if count == 0:
        return logits.sum() * 0.0, 0
    return F.cross_entropy(logits[own], own_labels[own], label_smoothing=smoothing), count


def balanced_relation_loss(student: torch.Tensor, teacher: torch.Tensor,
                           vehicle_ids: torch.Tensor, cameras: torch.Tensor,
                           ) -> tuple[torch.Tensor, int, int]:
    """Give cross-camera positives and different-ID negatives equal loss weight."""
    student = F.normalize(student.float(), p=2, dim=1)
    teacher = F.normalize(teacher.float(), p=2, dim=1)
    same = vehicle_ids[:, None] == vehicle_ids[None, :]
    upper = torch.triu(torch.ones_like(same, dtype=torch.bool), diagonal=1)
    positive = upper & same & (cameras[:, None] != cameras[None, :])
    negative = upper & ~same
    n_positive = int(positive.sum().item())
    n_negative = int(negative.sum().item())
    if not n_positive:
        raise ValueError("batch has no cross-camera positive for relation distillation")
    if not n_negative:
        raise ValueError("batch has no different-ID negative for relation distillation")
    squared = (student @ student.T - teacher @ teacher.T).square()
    return 0.5 * (squared[positive].mean() + squared[negative].mean()), n_positive, n_negative


def projected_total_seconds(*, elapsed: float, first_epoch_seconds: float,
                            stage2_batch_seconds: float, steps_per_epoch: int,
                            remaining_epochs: int, safety_factor: float = 1.25) -> float:
    if min(elapsed, first_epoch_seconds, stage2_batch_seconds) < 0 or steps_per_epoch < 1 or remaining_epochs < 0:
        raise ValueError("pilot times and counts must be nonnegative")
    seconds_per_batch = max(first_epoch_seconds / steps_per_epoch, stage2_batch_seconds)
    return elapsed + remaining_epochs * steps_per_epoch * seconds_per_batch * safety_factor


def make_distilled_checkpoint(model: CrossViewCore, *, protocol_sha: str,
                              source_sha: str, teacher_sha: str, code_sha: str,
                              best_epoch: int, best_map: float) -> dict:
    return {
        "model_state_dict": {key: value.detach().cpu().clone()
                             for key, value in model.state_dict().items()},
        "num_ids": model.num_ids, "protocol_sha256": protocol_sha,
        "model_config": MODEL_CONFIG, "best_epoch": int(best_epoch),
        "dev_cosine_mAP": float(best_map),
        "source_checkpoint_sha256": source_sha,
        "teacher_targets_sha256": teacher_sha,
        "source_sha256": {"finetune_py": code_sha},
    }


def _validate_protocol(protocol: dict) -> dict:
    if protocol.get("status") != "frozen_before_training" or protocol.get("version") != 2:
        raise ValueError("distillation requires frozen version-2 protocol")
    cfg = protocol["training"]
    attempt = cfg["this_attempt"]
    if (cfg["attempts"] != 2 or attempt not in (1, 2) or
            cfg["batch_size"] != 32 or cfg["steps_per_epoch"] != 600 or
            (cfg["stage1_epochs"], cfg["stage2_epochs"]) !=
            ((1, 11) if attempt == 1 else (0, 8)) or
            cfg["device"] != "cpu" or cfg["threads"] != 4):
        raise ValueError("trainer disagrees with frozen attempt schedule")
    fixed = {
        "seed": 20260926,
        "sampler": "8 vehicle IDs x 4 frames, cross-camera positives required",
        "optimizer": "AdamW",
        "weight_decay": 0.0001,
        "head_classifier_lr": 0.0002,
        "backbone_lr": 5e-5,
        "scheduler": "cosine anneal within each stage to eta_min=0.000001",
        "peak_rss_limit_mib": 4096,
        "wall_limit_seconds": 7200,
        "stop_no_later_than_moscow": "2026-09-27 00:00:00",
        "augmentation": "fit-only horizontal flip p=0.5 and brightness/contrast 0.9..1.1, same as prior run",
    }
    for key, expected in fixed.items():
        if cfg.get(key) != expected:
            raise ValueError(f"trainer disagrees with frozen {key}")
    losses = (("classification_loss", "CE", 1.0),
              ("metric_loss", "cross_camera_batch_hard_triplet", 0.5),
              ("embedding_kd", "1 - dot(L2 student,L2 teacher)", 2.0))
    for key, kind, weight in losses:
        if cfg[key]["type"] != kind or cfg[key]["weight"] != weight:
            raise ValueError(f"trainer disagrees with frozen {key}")
    if (cfg["classification_loss"]["label_smoothing"] != 0.1 or
            cfg["metric_loss"]["margin"] != 0.3 or
            cfg["relation_kd"]["type"] !=
            "0.5*(mean_squared_cross_camera_positive + mean_squared_different_ID_negative) of batch Gram similarities" or
            cfg["relation_kd"]["weight"] != 0.5):
        raise ValueError("trainer disagrees with frozen loss parameters")
    return cfg


def _optimizer(model: CrossViewCore, cfg: dict, stage: int) -> torch.optim.Optimizer:
    heads = [*model.global_head.parameters(), *model.top_head.parameters(),
             *model.bottom_head.parameters(), *model.classifier.parameters()]
    groups = [{"params": heads, "lr": float(cfg["head_classifier_lr"])}]
    if stage == 2:
        groups.insert(0, {"params": model.backbone.parameters(),
                          "lr": float(cfg["backbone_lr"])})
    return torch.optim.AdamW(groups, weight_decay=float(cfg["weight_decay"]))


def _one_step(model: CrossViewCore, batch: tuple, optimizer: torch.optim.Optimizer,
              cfg: dict, device: torch.device, *, augment_input: bool = True) -> dict:
    images, own_labels, vehicle_ids, cameras, _, targets = batch
    images = images.to(device)
    if augment_input:
        images = augment(images)
    own_labels = own_labels.to(device)
    vehicle_ids, cameras = vehicle_ids.to(device), cameras.to(device)
    targets = targets.to(device)
    optimizer.zero_grad(set_to_none=True)
    descriptor = model(images)
    ce, own_count = classification_on_own(model.classify(descriptor), own_labels,
                                          smoothing=float(cfg["classification_loss"]["label_smoothing"]))
    triplet = cross_camera_batch_hard(descriptor, vehicle_ids, cameras,
                                      margin=float(cfg["metric_loss"]["margin"]))
    kd = 1.0 - (F.normalize(descriptor.float(), dim=1) *
                F.normalize(targets.float(), dim=1)).sum(dim=1).mean()
    relation, positive_count, negative_count = balanced_relation_loss(
        descriptor, targets, vehicle_ids, cameras)
    loss = (ce * float(cfg["classification_loss"]["weight"]) +
            triplet * float(cfg["metric_loss"]["weight"]) +
            kd * float(cfg["embedding_kd"]["weight"]) +
            relation * float(cfg["relation_kd"]["weight"]))
    if not all(bool(torch.isfinite(item)) for item in (ce, triplet, kd, relation, loss)):
        raise FloatingPointError("nonfinite distillation loss")
    loss.backward()
    if not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):
        raise FloatingPointError("nonfinite distillation gradient")
    optimizer.step()
    return {"loss": float(loss.detach()), "classification": float(ce.detach()),
            "triplet": float(triplet.detach()), "embedding_kd": float(kd.detach()),
            "relation_kd": float(relation.detach()), "own_ce_rows": own_count,
            "cross_camera_positive_pairs": positive_count,
            "different_id_negative_pairs": negative_count}


def smoke_one_real_batch(model: CrossViewCore, loader: DataLoader, cfg: dict,
                         device: torch.device) -> dict:
    """Mutate only a disposable warm-start instance; full run reloads source."""
    model.train()
    set_training_stage(model, 2)
    optimizer = _optimizer(model, cfg, 2)
    batch = None
    for candidate in loader:
        if (len(candidate[0]) == 32 and int(torch.unique(candidate[2]).numel()) == 8 and
                bool((candidate[1] >= 0).any())):
            batch = candidate
            break
    if batch is None:
        raise ValueError("smoke could not obtain real P8K4 batch with own CE rows")
    watched = {"backbone": model.backbone[0][0].weight,
               "global": model.global_head.weight,
               "top": model.top_head.weight,
               "bottom": model.bottom_head.weight,
               "classifier": model.classifier.weight}
    before = {key: value.detach().clone() for key, value in watched.items()}
    started = time.monotonic()
    stats = _one_step(model, batch, optimizer, cfg, device)
    stats["seconds"] = time.monotonic() - started
    stats["peak_rss_bytes"] = peak_rss_bytes()
    gradients = {key: bool(value.grad is not None and torch.isfinite(value.grad).all() and
                           value.grad.abs().sum() > 0) for key, value in watched.items()}
    updates = {key: bool(torch.any(value.detach() != before[key]))
               for key, value in watched.items()}
    if not all(gradients.values()) or not all(updates.values()):
        raise RuntimeError(f"smoke missing gradients/updates: {gradients}, {updates}")
    require_rss_below(stats["peak_rss_bytes"], int(cfg["peak_rss_limit_mib"]))
    return {"status": "PASS", "stats": stats, "gradients": gradients,
            "updates": updates, "batch_rows": [int(x) for x in batch[4]]}


def _train_epoch(model: CrossViewCore, loader: DataLoader,
                 optimizer: torch.optim.Optimizer, scheduler, cfg: dict,
                 device: torch.device, wall_deadline: float) -> dict:
    model.train()
    if not any(p.requires_grad for p in model.backbone.parameters()):
        model.backbone.eval()
    started = time.monotonic()
    sums = {key: 0.0 for key in ("loss", "classification", "triplet",
                                "embedding_kd", "relation_kd")}
    own_rows = positive_pairs = negative_pairs = count = 0
    for batch in loader:
        if time.monotonic() >= wall_deadline:
            raise TimeoutError("frozen training wall/Moscow deadline exceeded")
        record = _one_step(model, batch, optimizer, cfg, device)
        scheduler.step()
        for key in sums:
            sums[key] += record[key]
        own_rows += record["own_ce_rows"]
        positive_pairs += record["cross_camera_positive_pairs"]
        negative_pairs += record["different_id_negative_pairs"]
        count += 1
    if count != int(cfg["steps_per_epoch"]):
        raise ValueError(f"epoch executed {count} rather than frozen {cfg['steps_per_epoch']} batches")
    return {key: value / count for key, value in sums.items()} | {
        "batches": count, "own_ce_rows": own_rows,
        "cross_camera_positive_pairs": positive_pairs,
        "different_id_negative_pairs": negative_pairs,
        "seconds": time.monotonic() - started,
        "peak_rss_bytes": peak_rss_bytes(),
        "learning_rates_final": [group["lr"] for group in optimizer.param_groups],
    }


def _check_source_manifest(attempt: int, cfg: dict, args: argparse.Namespace,
                           hashes: dict) -> None:
    if attempt == 1:
        if args.source_checkpoint_sha256 != cfg["source_checkpoint_sha256"]:
            raise ValueError("source checkpoint SHA differs from frozen attempt-1 protocol")
        return
    if args.source_training_json is None:
        raise ValueError("combined attempt requires --source-training-json from attempt 1")
    manifest = json.loads(args.source_training_json.read_text(encoding="utf-8"))
    hashes["source_training_json"] = sha256(args.source_training_json)
    if (manifest.get("status") != "PASS" or
            manifest.get("checkpoint_sha256") != args.source_checkpoint_sha256 or
            manifest.get("protocol_sha256") != sha256(HERE / "protocol-own.json")):
        raise ValueError("combined warm start differs from completed attempt-1 manifest")


def run(args: argparse.Namespace) -> dict:
    protocol_path = Path(args.protocol)
    checked_hash(protocol_path, args.protocol_sha256)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    cfg = _validate_protocol(protocol)
    attempt = cfg["this_attempt"]
    device = torch.device("cpu")
    torch.set_num_threads(int(cfg["threads"]))
    hashes = {"protocol": args.protocol_sha256, "finetune_py": sha256(__file__),
              "core_py": sha256(BASE / "core.py"),
              "dataset_py": sha256(BASE / "dataset.py"),
              "source_checkpoint": checked_hash(args.source_checkpoint,
                                                 args.source_checkpoint_sha256),
              "data_npz": checked_hash(args.data_npz, protocol["data"]["train_source_sha256"]),
              "data_raw": checked_hash(args.data_raw, protocol["data"]["train_crops_sha256"]),
              "split": checked_hash(args.split, protocol["data"]["split_sha256"]),
              "teacher_targets": checked_hash(args.teacher_targets, args.teacher_targets_sha256)}
    _check_source_manifest(attempt, cfg, args, hashes)
    evaluator = load_evaluator(args.evaluator,
                               protocol["evaluation"]["evaluator_sha256"],
                               protocol["evaluation"]["evaluator_scope_dependency_sha256"])
    hashes["evaluator"] = sha256(args.evaluator)
    hashes["scope_metrics"] = sha256(args.evaluator.with_name("scope_metrics.py"))
    data = load_combined(args.data_npz, args.data_raw, args.split)
    if (len(data.train_indices) != protocol["data"]["fit_rows_expected"] or
            len(data.dev_indices) != protocol["data"]["dev_rows_expected"] or
            len(np.unique(data.dev_vehicle_ids)) != protocol["data"]["dev_ids_expected"]):
        raise ValueError("fit/dev cardinalities differ from frozen protocol")
    with np.load(args.data_npz, allow_pickle=False) as meta_archive:
        metadata = {key: np.array(meta_archive[key], copy=True)
                    for key in ("vehicle_id", "camera_id", "image_id")}
    targets = load_teacher_targets(args.teacher_targets, args.teacher_targets_sha256,
                                   data.train_indices, metadata)
    own_fit_ids = np.unique(data.vehicle_ids[data.train_indices][
        data.vehicle_ids[data.train_indices] < 100000])
    if len(own_fit_ids) != 1071:
        raise ValueError(f"own classifier requires 1071 fit IDs, got {len(own_fit_ids)}")
    dataset = DistillDataset(data, targets, own_fit_ids)
    sampler = CameraAwarePKSampler(data.train_vehicle_ids, data.train_camera_ids,
                                   np.arange(len(dataset)), p=8, k=4,
                                   batches=int(cfg["steps_per_epoch"]), seed=int(cfg["seed"]))
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    moscow = timezone(timedelta(hours=3))
    stop_at = datetime.fromisoformat(cfg["stop_no_later_than_moscow"]).replace(tzinfo=moscow)
    wall_budget = min(float(cfg["wall_limit_seconds"]),
                      (stop_at - datetime.now(timezone.utc)).total_seconds())
    if wall_budget <= 0:
        raise TimeoutError("frozen Moscow stop deadline has passed")
    if args.out.exists():
        raise FileExistsError("refusing to overwrite training output directory")
    args.out.mkdir(parents=True)
    _atomic_json(args.out / "inputs.json", hashes)
    run_start = time.monotonic()
    wall_deadline = run_start + wall_budget
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        disposable = load_warm_start(args.source_checkpoint,
                                     args.source_checkpoint_sha256, len(own_fit_ids)).to(device)
        smoke = smoke_one_real_batch(disposable, loader, cfg, device)
        smoke.update(protocol_sha256=args.protocol_sha256,
                     source_checkpoint_sha256=args.source_checkpoint_sha256,
                     teacher_targets_sha256=args.teacher_targets_sha256)
        _atomic_json(args.out / "smoke.json", smoke)
        del disposable
        gc.collect()
        if args.smoke_only:
            return smoke
        model = load_warm_start(args.source_checkpoint,
                                args.source_checkpoint_sha256, len(own_fit_ids)).to(device)
        epochs: list[dict] = []
        best_map, best_epoch, best_q, best_g = -math.inf, -1, None, None
        global_epoch = 0
        for stage, n_epochs in ((1, int(cfg["stage1_epochs"])),
                                (2, int(cfg["stage2_epochs"]))):
            if n_epochs == 0:
                continue
            model.train()
            set_training_stage(model, stage)
            optimizer = _optimizer(model, cfg, stage)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs * int(cfg["steps_per_epoch"]), eta_min=1e-6)
            for local_epoch in range(n_epochs):
                sampler.set_epoch(global_epoch)
                stats = _train_epoch(model, loader, optimizer, scheduler, cfg,
                                     device, wall_deadline)
                require_rss_below(stats["peak_rss_bytes"], int(cfg["peak_rss_limit_mib"]))
                global_epoch += 1
                dev_map, query, gallery = dev_cosine_map(model, data, evaluator,
                                                          batch=32, device=device)
                row = {"global_epoch": global_epoch, "stage": stage,
                       "stage_epoch": local_epoch + 1, "dev_cosine_mAP": dev_map, **stats}
                if dev_map > best_map:
                    best_map, best_epoch, best_q, best_g = dev_map, global_epoch, query, gallery
                    payload = make_distilled_checkpoint(
                        model, protocol_sha=args.protocol_sha256,
                        source_sha=args.source_checkpoint_sha256,
                        teacher_sha=args.teacher_targets_sha256,
                        code_sha=hashes["finetune_py"], best_epoch=best_epoch, best_map=best_map)
                    save_best_checkpoint(args.out, payload)
                    row["selected_best"] = True
                else:
                    row["selected_best"] = False
                epochs.append(row)
                with (args.out / "epochs.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                print(json.dumps(row, sort_keys=True), flush=True)
                if global_epoch == 1:
                    remaining = int(cfg["stage1_epochs"] + cfg["stage2_epochs"]) - 1
                    projected = projected_total_seconds(
                        elapsed=time.monotonic() - run_start,
                        first_epoch_seconds=stats["seconds"],
                        stage2_batch_seconds=smoke["stats"]["seconds"],
                        steps_per_epoch=int(cfg["steps_per_epoch"]),
                        remaining_epochs=remaining)
                    pilot = {"status": "PASS" if projected <= wall_budget else "FAIL",
                             "projected_total_seconds": projected,
                             "wall_budget_seconds": wall_budget,
                             "peak_rss_bytes": stats["peak_rss_bytes"],
                             "stage2_smoke_batch_seconds": smoke["stats"]["seconds"],
                             "first_epoch_seconds": stats["seconds"],
                             "remaining_epochs": remaining, "projection_safety_factor": 1.25}
                    _atomic_json(args.out / "pilot.json", pilot)
                    if pilot["status"] != "PASS":
                        raise RuntimeError("frozen first-epoch runtime pilot exceeded wall budget")
        if best_epoch < 1 or best_q is None or best_g is None:
            raise RuntimeError("no dev-selected checkpoint after complete training")
        q, g = data.dev_query_indices, data.dev_gallery_indices
        np.savez(args.out / "dev_embeddings.npz",
                 query_embeddings=best_q, gallery_embeddings=best_g,
                 query_indices=q, gallery_indices=g,
                 query_vehicle_ids=data.vehicle_ids[q], gallery_vehicle_ids=data.vehicle_ids[g],
                 query_camera_ids=data.camera_ids[q], gallery_camera_ids=data.camera_ids[g],
                 known_absent=~np.isin(data.vehicle_ids[q], data.vehicle_ids[g]),
                 protocol_sha256=np.array(args.protocol_sha256))
        checkpoint_path = args.out / "checkpoint.pt"
        summary = {"status": "PASS", "protocol_sha256": args.protocol_sha256,
                   "checkpoint_sha256": sha256(checkpoint_path),
                   "dev_embeddings_sha256": sha256(args.out / "dev_embeddings.npz"),
                   "best_epoch": best_epoch, "dev_cosine_mAP": best_map,
                   "num_ids": len(own_fit_ids), "device": str(device),
                   "torch": torch.__version__, "numpy": np.__version__,
                   "input_sha256": hashes, "epochs": epochs}
        _atomic_json(args.out / "training.json", summary)
        _atomic_json(args.out / "checkpoint-status.json",
                     {"status": "COMPLETE", "protocol_sha256": args.protocol_sha256,
                      "checkpoint_sha256": summary["checkpoint_sha256"],
                      "training_sha256": sha256(args.out / "training.json"),
                      "best_epoch": best_epoch, "dev_cosine_mAP": best_map})
        return summary
    except Exception as error:
        _atomic_json(args.out / "failure.json",
                     {"status": "PARTIAL", "reason": f"{type(error).__name__}: {error}",
                      "protocol_sha256": args.protocol_sha256,
                      "source_checkpoint_sha256": args.source_checkpoint_sha256,
                      "teacher_targets_sha256": args.teacher_targets_sha256})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint-sha256", required=True)
    parser.add_argument("--source-training-json", type=Path,
                        help="required for combined: completed attempt-1 training.json")
    parser.add_argument("--data-npz", type=Path, required=True)
    parser.add_argument("--data-raw", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--teacher-targets", type=Path, required=True)
    parser.add_argument("--teacher-targets-sha256", required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({key: value for key, value in report.items() if key != "epochs"},
                     sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
