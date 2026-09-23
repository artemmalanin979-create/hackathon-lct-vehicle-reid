#!/usr/bin/env python3
"""Шаг 5b: собрать векторы выбранных TTA-комбинаций в отдельные файлы.

Шаг 5 (s05_tta.py) только оценивает комбинации, файлов не пишет, а шаг 6
(s06_final_val.py) читает готовые out/{set}_{part}_{TTA,TTA2}.npy — в
репозитории этого звена не было, и s06 падал на отсутствующем файле. Здесь оно
и восстановлено: усреднение ровно тех комбинаций, которые выбраны по протоколу
подбора (out/tta_{set}.json, набор tune) и названы в докстроке шага 6:

  TTA  = 208 + 208f + 256   (лучшая трёхпроходная на tune)
  TTA2 = 208 + 256          (лучшая двухпроходная на tune)

Усреднение — общая функция average_embeddings (L2 -> среднее -> L2), та же, что
считала числа шага 5. Новых величин здесь не появляется.

    s05b_make_tta_vectors.py [--sets val,tune]
"""
import argparse
import sys
from pathlib import Path

from inputs import JOB, require_files, TTA_HINT

COMBOS = {"TTA": ["208", "208f", "256"], "TTA2": ["208", "256"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default="val", help="наборы через запятую: val,tune")
    args = ap.parse_args()
    setnames = [s.strip() for s in args.sets.split(",") if s.strip()]
    require_files([JOB / f"out/{setname}_{part}_{tag}.npy"
                   for setname in setnames for part in ("query", "gallery")
                   for tag in {t for combo in COMBOS.values() for t in combo}], hint=TTA_HINT)

    import numpy as np
    from common import average_embeddings

    for setname in (s.strip() for s in args.sets.split(",") if s.strip()):
        for part in ("query", "gallery"):
            for tag, combo in COMBOS.items():
                sources = [JOB / f"out/{setname}_{part}_{t}.npy" for t in combo]
                out = JOB / f"out/{setname}_{part}_{tag}.npy"
                np.save(out, average_embeddings([np.load(p) for p in sources]))
                print(f"{out.name}: {'+'.join(combo)}", flush=True)


if __name__ == "__main__":
    main()
