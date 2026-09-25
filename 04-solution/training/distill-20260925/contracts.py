"""Fail closed before labels or features can reach training/evaluation."""
import hashlib
from pathlib import Path

import numpy as np


def checked_hash(path, expected):
    with Path(path).open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"hash mismatch: {path}: {actual} != {expected}")
    return actual


def checked_features(x, actual_ids, expected_ids):
    x = np.asarray(x)
    if list(actual_ids) != list(expected_ids):
        raise ValueError("feature row order differs from metadata")
    if x.ndim != 2 or len(x) != len(expected_ids) or not x.shape[1]:
        raise ValueError("feature dimensions differ from metadata")
    if not np.isfinite(x).all() or np.any(np.linalg.norm(x, axis=1) == 0):
        raise ValueError("nonfinite or zero feature")
    return x


def check_disjoint(*groups):
    sets = [set(map(int, g)) for g in groups]
    for i, a in enumerate(sets):
        for b in sets[i + 1:]:
            if a & b:
                raise ValueError(f"identity overlap: {sorted(a & b)[:10]}")


def grouped_split(vids, n_dev=100, seed=20260916):
    vids = np.asarray(vids)
    unique = np.unique(vids)
    dev_ids = unique[np.random.default_rng(seed).permutation(len(unique))[:n_dev]]
    mask = np.isin(vids, dev_ids)
    return np.flatnonzero(~mask), np.flatnonzero(mask)
