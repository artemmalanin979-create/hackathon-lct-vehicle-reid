"""Checked combined crops and camera-aware P x K batches for owncore training."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class CombinedData:
    crops: np.ndarray
    vehicle_ids: np.ndarray
    camera_ids: np.ndarray
    train_indices: np.ndarray
    dev_indices: np.ndarray
    dev_query_indices: np.ndarray
    dev_gallery_indices: np.ndarray
    label_to_vehicle: np.ndarray

    @property
    def train_vehicle_ids(self) -> np.ndarray:
        return self.vehicle_ids[self.train_indices]

    @property
    def train_camera_ids(self) -> np.ndarray:
        return self.camera_ids[self.train_indices]

    @property
    def dev_vehicle_ids(self) -> np.ndarray:
        return self.vehicle_ids[self.dev_indices]


def _unique_indices(split: dict, name: str, upper: int) -> np.ndarray:
    values = np.asarray(split[name], dtype=np.int64)
    if values.ndim != 1 or np.any(values < 0) or np.any(values >= upper):
        raise ValueError(f"{name} out of range")
    if len(np.unique(values)) != len(values):
        raise ValueError(f"{name} contains duplicate rows")
    return values


def load_combined(npz_path: str | Path, raw_path: str | Path, split_path: str | Path) -> CombinedData:
    """Use fixed own dev100; all external identities remain in the fit partition."""
    with np.load(npz_path, allow_pickle=False) as meta:
        vids = np.asarray(meta["vehicle_id"], dtype=np.int64)
        cams = np.asarray(meta["camera_id"], dtype=np.int64)
    if vids.ndim != 1 or cams.shape != vids.shape:
        raise ValueError("vehicle_id and camera_id must be aligned vectors")
    crops = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    if crops.shape != (len(vids), 208, 208, 3) or crops.dtype != np.uint8:
        raise ValueError("raw crops must be aligned uint8 NHWC 208x208 RGB")
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    fit_own = _unique_indices(split, "fit_indices", len(vids))
    dev = _unique_indices(split, "dev_indices", len(vids))
    if np.intersect1d(fit_own, dev).size:
        raise ValueError("fit/dev row overlap")
    own = np.flatnonzero(vids < 100000)
    if not np.array_equal(np.sort(np.concatenate((fit_own, dev))), own):
        raise ValueError("fit/dev split does not partition own rows")
    fit_ids = np.asarray(split["fit_ids"], dtype=np.int64)
    dev_ids = np.asarray(split["dev_ids"], dtype=np.int64)
    if not np.array_equal(np.unique(vids[fit_own]), np.unique(fit_ids)):
        raise ValueError("fit identity list differs from fit rows")
    if not np.array_equal(np.unique(vids[dev]), np.unique(dev_ids)):
        raise ValueError("dev identity list differs from dev rows")
    if np.intersect1d(fit_ids, dev_ids).size:
        raise ValueError("fit/dev vehicle_id overlap")
    external = np.flatnonzero(vids >= 100000)
    train = np.sort(np.concatenate((fit_own, external)))
    if np.intersect1d(vids[train], dev_ids).size:
        raise ValueError("dev identity leaked into train")
    q = _unique_indices(split, "dev_query_indices", len(vids))
    g = _unique_indices(split, "dev_gallery_indices", len(vids))
    if not np.isin(q, dev).all() or not np.isin(g, dev).all():
        raise ValueError("dev query/gallery contains non-dev row")
    if np.intersect1d(q, g).size:
        raise ValueError("dev query/gallery row overlap")
    if not np.array_equal(np.sort(np.concatenate((q, g))), np.sort(dev)):
        raise ValueError("dev query/gallery does not partition dev rows")
    return CombinedData(crops, vids, cams, train, dev, q, g, np.unique(vids[train]))


class CropDataset(Dataset):
    def __init__(self, data: CombinedData, indices: np.ndarray, labels: bool = False):
        self.data, self.indices, self.labels = data, np.asarray(indices, dtype=np.int64), labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        row = int(self.indices[position])
        # Copy is needed because memmap pages are read-only and torch requires writable storage.
        image = torch.from_numpy(np.array(self.data.crops[row], copy=True)).permute(2, 0, 1).float()
        vehicle_id = int(self.data.vehicle_ids[row])
        target = int(np.searchsorted(self.data.label_to_vehicle, vehicle_id)) if self.labels else vehicle_id
        return image, target, int(self.data.camera_ids[row]), row


class CameraAwarePKSampler(Sampler[list[int]]):
    """P distinct IDs; multi-camera IDs get cross-camera pairs, all IDs get CE."""

    def __init__(self, vids: np.ndarray, cams: np.ndarray, indices: np.ndarray,
                 p: int = 8, k: int = 4, batches: int = 600, seed: int = 20260925):
        self.vids, self.cams = np.asarray(vids), np.asarray(cams)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.p, self.k, self.batches, self.seed, self.epoch = p, k, batches, seed, 0
        if self.vids.shape != self.cams.shape or self.vids.ndim != 1:
            raise ValueError("vids/cams must be aligned vectors")
        if p < 2 or k < 2 or batches < 1:
            raise ValueError("P>=2, K>=2 and batches>=1 are required")
        if np.any(self.indices < 0) or np.any(self.indices >= len(self.vids)):
            raise ValueError("sampler indices out of range")
        self.by_id_cam: dict[int, dict[int, list[int]]] = {}
        for index in self.indices:
            vid, cam = int(self.vids[index]), int(self.cams[index])
            self.by_id_cam.setdefault(vid, {}).setdefault(cam, []).append(int(index))
        self.all_ids = np.array(sorted(self.by_id_cam))
        self.multicam_ids = np.array(sorted(vid for vid, by_cam in self.by_id_cam.items()
                                             if len(by_cam) >= 2))
        if len(self.all_ids) < p or len(self.multicam_ids) == 0:
            raise ValueError("need P identities and at least one multi-camera identity")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.batches):
            batch: list[int] = []
            selected = rng.choice(self.all_ids, self.p, replace=False)
            if not np.isin(selected, self.multicam_ids).any():
                # Ensure a real cross-camera anchor is present for the metric loss.
                selected[0] = rng.choice(self.multicam_ids)
            for vid in selected:
                by_cam = self.by_id_cam[int(vid)]
                if len(by_cam) >= 2:
                    two_cams = rng.choice(np.array(list(by_cam)), 2, replace=False)
                    chosen = [int(rng.choice(by_cam[int(cam)])) for cam in two_cams]
                else:
                    chosen = []
                pool = [row for rows in by_cam.values() for row in rows if row not in chosen]
                extra = self.k - len(chosen)
                if extra:
                    if len(pool) >= extra:
                        chosen.extend(int(x) for x in rng.choice(pool, extra, replace=False))
                    else:
                        chosen.extend(pool)
                        all_rows = [row for rows in by_cam.values() for row in rows]
                        chosen.extend(int(x) for x in rng.choice(all_rows, extra - len(pool), replace=True))
                batch.extend(chosen)
            rng.shuffle(batch)
            yield batch
