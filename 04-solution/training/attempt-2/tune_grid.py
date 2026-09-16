#!/usr/bin/env python3
"""Подбор параметров переранжирования на протоколе из train_fit (не на валидации).

Сетка и код перебора — готовые, из 04-solution/postproc/scripts/s04_grid.py (импорт).
Результат кладётся в out/ этого задания; в репозиторий ничего не пишется.
"""
import sys
from pathlib import Path
import numpy as np

JOB = Path(__file__).resolve().parent.parent
REPO = Path("/home/artem/projects/hackathon-lct-vehicle-reid")
sys.path.insert(0, str(REPO / "04-solution/postproc/scripts"))
from common import read_meta          # noqa: E402
from s04_grid import run_grid         # noqa: E402

tag = sys.argv[1]
q = np.load(JOB / "out" / f"tune_query_{tag}.npy")
g = np.load(JOB / "out" / f"tune_gallery_{tag}.npy")
qm = read_meta(JOB / "tune" / "tune_query.csv")
gm = read_meta(JOB / "tune" / "tune_gallery.csv")
run_grid(f"tune_{tag}", q, g, qm, gm, out_dir=JOB / "out")
