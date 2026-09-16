#!/usr/bin/env python3
"""Сколько стоит точное объяснение: замер по частям.

Отдельно меряем базовый граф, граф с выведенной картой признаков и саму арифметику
разложения, чтобы честно ответить, годится ли это для живого показа.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

JOB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JOB / "scripts"))
import occl  # noqa: E402
import contrib  # noqa: E402

qm, gm = occl.read_split("val_query"), occl.read_split("val_gallery")
imgs = [occl.net_input(qm[i]) for i in (395, 608, 413, 158, 320)]
x1 = np.stack([imgs[0].astype(np.float32).transpose(2, 0, 1)])
x2 = np.stack([a.astype(np.float32).transpose(2, 0, 1) for a in imgs[:2]])

o = ort.SessionOptions()
o.log_severity_level = 3
base = ort.InferenceSession(str(occl.MODEL), sess_options=o,
                            providers=["CPUExecutionProvider"])
contrib._prepare()
patched = contrib._S
res = {}


def bench(sess, x, n=40):
    """Машина занята соседними процессами (load average ~12 на 12 ядрах), поэтому
    кроме медианы отдаём минимум: он ближе к стоимости на свободном CPU."""
    nm = sess.get_inputs()[0].name
    sess.run(None, {nm: x})
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, {nm: x})
        ts.append((time.perf_counter() - t0) * 1000)
    return {"median": round(float(np.median(ts)), 2), "min": round(float(np.min(ts)), 2)}


res["ms_base_b1"] = bench(base, x1)
res["ms_base_b2"] = bench(base, x2)
res["ms_patched_b1"] = bench(patched, x1)
res["ms_patched_b2"] = bench(patched, x2)
res["load_average"] = [round(v, 2) for v in __import__("os").getloadavg()]

# арифметика разложения: A(512xC) @ z(C x HW) и скалярные произведения
A, c = contrib._AC
_, feat, _ = contrib.forward(imgs[:1])
z = feat[0].astype(np.float64).reshape(512, -1)
f = np.random.rand(512)
ts = []
for _ in range(200):
    t0 = time.perf_counter()
    ((f @ A) @ z)
    ts.append((time.perf_counter() - t0) * 1000)
res["ms_decomposition_math"] = {"median": round(float(np.median(ts)), 3),
                                "min": round(float(np.min(ts)), 3)}

# полный цикл «объяснить показанного кандидата» (кропы уже загружены)
ts = []
for _ in range(40):
    t0 = time.perf_counter()
    contrib.pair_contrib(imgs[0], imgs[1])
    ts.append((time.perf_counter() - t0) * 1000)
res["ms_pair_contrib_full"] = {"median": round(float(np.median(ts)), 1),
                               "min": round(float(np.min(ts)), 1)}

# то же, но вектор кандидата уже в индексе: считаем карту только для запроса
ts = []
for _ in range(40):
    t0 = time.perf_counter()
    out, ft, _ = contrib.forward(imgs[:1])
    v = out.astype(np.float64)[0]
    nrm = np.linalg.norm(v)
    zz = ft[0].astype(np.float64).reshape(512, -1)
    ((f @ A) @ zz) / (zz.shape[1] * nrm)
    ts.append((time.perf_counter() - t0) * 1000)
res["ms_query_side_only"] = {"median": round(float(np.median(ts)), 1),
                             "min": round(float(np.min(ts)), 1)}

res["ms_crop_load"] = round(np.median(
    [(lambda t: t)(0) for _ in range(1)]) if False else 0, 1)
t0 = time.perf_counter()
for i in (395, 608, 413, 158, 320):
    occl.net_input(qm[i])
res["ms_crop_load"] = round((time.perf_counter() - t0) / 5 * 1000, 1)

(JOB / "out/contrib_speed.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(json.dumps(res, ensure_ascii=False, indent=1))
