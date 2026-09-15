#!/usr/bin/env python3
"""Отдельная проверка порядка строк в artifacts/embeddings.npy (не «глазами»).

Что проверяется:
  1. Форма (1860, 512), dtype float32, все значения конечны, L2-нормы ~1.
  2. Порядок image_id извлечения совпадает с порядком строк test_query.csv,
     затем test_gallery.csv (файлы .ids писались в момент извлечения).
  3. Независимая пересборка: для K контрольных позиций (включая границы блоков
     0, 1109, 1110, 1859 и детерминированную случайную выборку) вектор
     пересчитывается ЗАНОВО из CSV-строки (свежая сессия, батч=1) и сравнивается
     с соответствующей строкой embeddings.npy: cos >= 0.9999. Дополнительно
     ищется ближайшая строка всего файла — должна быть сама позиция (совпадения
     допускаются только при побайтово одинаковых кропах — репортится).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

JOB = Path(__file__).resolve().parent.parent
DATA = Path("/home/artem/projects/hackathon-lct-vehicle-reid/data")
sys.path.insert(0, str(JOB / "scripts"))
from extract_embeddings import load_crop, make_session, l2norm  # noqa: E402

K_RANDOM = 20
BOUNDARY = [0, 1109, 1110, 1859]


def read_rows(path: Path):
    with open(path, newline="") as f:
        return [(r["image_id"], int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))
                for r in csv.DictReader(f)]


def main():
    emb = np.load(JOB / "artifacts" / "embeddings.npy")
    q_rows = read_rows(DATA / "test_query.csv")
    g_rows = read_rows(DATA / "test_gallery.csv")
    all_rows = q_rows + g_rows

    report = {"shape_ok": emb.shape == (len(all_rows), 512),
              "dtype": str(emb.dtype), "finite": bool(np.isfinite(emb).all()),
              "l2_norm_max_dev": float(np.abs(np.linalg.norm(emb, axis=1) - 1).max())}

    ids_run = (JOB / "out" / "test_query.ids").read_text().split() + \
              (JOB / "out" / "test_gallery.ids").read_text().split()
    report["ids_order_ok"] = ids_run == [r[0] for r in all_rows]

    rng = np.random.default_rng(20260915)
    picks = sorted(set(BOUNDARY) | set(rng.choice(len(all_rows), K_RANDOM, replace=False).tolist()))
    sess = make_session(JOB / "model" / "osnet_ain_x1_0_vehicle_reid.onnx", 0)
    inp = sess.get_inputs()[0].name

    checks, min_cos = [], 1.0
    for idx in picks:
        crop = load_crop(DATA / "images", all_rows[idx], 0.0, False)[None]
        vec = l2norm(sess.run(None, {inp: crop})[0].astype(np.float64))[0]
        cos_all = emb.astype(np.float64) @ vec
        cos_here = float(cos_all[idx])
        nearest = int(np.argmax(cos_all))
        ok = cos_here >= 0.9999 and (nearest == idx or np.isclose(cos_all[nearest], cos_here))
        checks.append({"idx": idx, "image_id": all_rows[idx][0], "cos_at_idx": round(cos_here, 6),
                       "nearest_idx": nearest, "ok": bool(ok)})
        min_cos = min(min_cos, cos_here)

    report["recompute_positions"] = picks
    report["recompute_min_cos_at_idx"] = round(min_cos, 6)
    report["recompute_all_ok"] = all(c["ok"] for c in checks)
    report["failed"] = [c for c in checks if not c["ok"]]
    report["VERDICT"] = ("OK" if report["shape_ok"] and report["finite"]
                         and report["ids_order_ok"] and report["recompute_all_ok"] else "FAIL")
    (JOB / "out" / "order_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("shape_ok", "ids_order_ok", "recompute_min_cos_at_idx",
                       "recompute_all_ok", "VERDICT")}))


if __name__ == "__main__":
    main()
