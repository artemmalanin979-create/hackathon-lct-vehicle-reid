#!/usr/bin/env python3
"""Склейка кусков векторов вариантов и L2-нормировка (куски пишет plate_variants.py)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plate_variants import VARIANTS

JOB = Path(__file__).resolve().parent.parent
SIZES = {"val_query": 1110, "val_gallery": 750}

tag = sys.argv[1]
d = JOB / "work" / f"emb_{tag}"
for split, n in SIZES.items():
    for v in VARIANTS:
        parts = sorted(d.glob(f"{split}_{v}_*.npy"))
        m = np.concatenate([np.load(p) for p in parts]).astype(np.float64)
        assert len(m) == n, f"{split}/{v}: {len(m)} != {n} (кусков {len(parts)})"
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        np.save(d / f"{split}_{v}.npy", m.astype(np.float32))
    print(f"{split}: {len(VARIANTS)} вариантов x {n} — склеено")
