#!/usr/bin/env python3
"""Reproduce d1_j48 from organizer images and retain historical research anchors.

The old all-query KR numbers below are historical, not the current independent-query
submission metric. Run the organizer's evaluate.py separately as documented in
SOLUTION.md; it needs pandas, which is intentionally absent from the service image.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from check_inputs import REPO, check_inputs, print_result, sha256


def file_records(directory):
    return {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(directory.iterdir()) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("val", "test", "all"), default="val")
    parser.add_argument("--repo-dir", type=Path, default=REPO)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    repo, data, out = args.repo_dir.resolve(), args.data_dir.resolve(), args.out_dir.resolve()
    # This check runs before numpy, ONNX Runtime or any model is imported.
    inputs = check_inputs(repo, data, args.mode)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / f"inputs-{args.mode}.json"
    report_path.write_text(json.dumps(inputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_result(inputs)
    if not inputs["ok"]:
        print(f"Полный список: {report_path}", file=sys.stderr)
        return 2
    if args.check_only:
        return 0
    # Reject user overrides that would silently change the published configuration.
    # MODEL_PATH/MODEL2_PATH/WHITENING_PATH are set by the shipped Dockerfile;
    # Embedder checks their full SHA-256. Only numeric/mode overrides are rejected.
    overrides = [name for name in os.environ if name.startswith("REID_")]
    if overrides:
        parser.error("Для эталонного прогона уберите переопределения: " + ", ".join(sorted(overrides)))
    modes = ("val", "test") if args.mode == "all" else (args.mode,)
    for name in modes:
        csv_dir = repo / "04-solution/split/files" if name == "val" else data
        query, gallery = csv_dir / f"{name}_query.csv", csv_dir / f"{name}_gallery.csv"
        subprocess.run([sys.executable, "-B", "-m", "app.batch",
                        "--images-dir", str(data / "images"), "--query", str(query),
                        "--gallery", str(gallery), "--out-dir", str(out / name),
                        "--threads", "2", "--batch", "32"],
                       cwd=repo / "04-solution/service", check=True)
    if args.mode == "test":
        report = {"input_verification": inputs, "files": file_records(out / "test"),
                  "batch": json.loads((out / "test/run_info.json").read_text()),
                  "note": "Ground truth for test is not supplied; no test quality metrics are computed."}
        (out / "test-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print("TEST REPRODUCED", json.dumps(report["files"]))
        return 0
    return validation(repo, data, out, modes, inputs)


def validation(repo, data, out, modes, inputs):
    import csv
    import hashlib
    import json
    import os
    import platform
    import subprocess
    import sys
    import time
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, str(repo / "04-solution/service"))
    sys.path.insert(0, str(repo / "04-solution/eval"))
    sys.path.insert(0, str(repo / "04-solution/postproc/scripts"))
    from reid_metrics import evaluate, scores_from_embeddings
    from common import rerank
    from app.core import config
    from app.core.model import Embedder
    from app.core.preprocess import read_rows, load_crop

    split = repo / "04-solution/split/files"
    def rows(path):
        with path.open(newline="") as f:
            return list(csv.DictReader(f))

    qm, gm = rows(split / "val_query.csv"), rows(split / "val_gallery.csv")
    manifest = json.loads((split / "manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((split / name).read_bytes()).hexdigest() == digest, name
    assert not {r["vehicle_id"] for r in rows(split / "train_fit.csv")} & {r["vehicle_id"] for r in qm + gm}

    # Якоря калибровки 20.09.2026 (конфигурация d1_j48): рабочие пороги и черновой
    # argmax-F1 из сохранённого прогона tools/calibrate_threshold.py.
    calib_dir = repo / "04-solution/service/calib-d1_j48"
    headline = json.loads((calib_dir / "headline.json").read_text())
    summary = json.loads((calib_dir / "summary.json").read_text())
    tau = float(headline["cosine_rule_reproduced"]["threshold"])   # рабочий t_cos
    rr_tau = float(headline["rerank_threshold"])                   # рабочий t_rr
    old_tau = float(summary["cosine_market_presence"]["selected"]["best_f1"]["threshold"])

    emb = np.load(out / "val/embeddings.npy", allow_pickle=False)
    assert emb.shape == (len(qm) + len(gm), 512) and emb.dtype == np.float32
    assert np.isfinite(emb).all()
    q, g = emb[:len(qm)], emb[len(qm):]
    assert np.max(np.abs(np.linalg.norm(emb.astype(np.float64), axis=1) - 1)) < 1e-6
    scores = scores_from_embeddings(q, g, metric="cosine")
    metadata = dict(query_ids=[r["vehicle_id"] for r in qm], gallery_ids=[r["vehicle_id"] for r in gm],
        query_cameras=[r["camera_id"] for r in qm], gallery_cameras=[r["camera_id"] for r in gm],
        known_absent=np.array([r["has_mate"] == "0" for r in qm]), camera_policy="market", refusal_mode="presence")

    def assess(matrix, threshold):
        r = evaluate(matrix, threshold=threshold, **metadata)
        return {"ranking": r["ranking_full_gallery"], "top10": r["ranking_top_k"],
                "top10_by_filter_order": r["ranking_top_k_by_filter_order"],
                "refusal": {k: v for k, v in r["refusal"].items() if k != "pr_curve"}, "counts": r["counts"]}

    base = assess(scores, tau)
    distance, seconds = rerank(q, g, 6, 3, 0.3)
    reranked = assess(1 - distance, 0.)
    report = {
        "input_verification": inputs,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "onnxruntime": __import__("onnxruntime").__version__, "threads": 2, "batch": 32},
        "model": {"bytes": config.MODEL_PATH.stat().st_size,
                  "sha256": hashlib.sha256(config.MODEL_PATH.read_bytes()).hexdigest()},
        "model2": {"bytes": config.MODEL2_PATH.stat().st_size,
                   "sha256": hashlib.sha256(config.MODEL2_PATH.read_bytes()).hexdigest()},
        "whitening": {"bytes": config.WHITENING_PATH.stat().st_size,
                      "sha256": hashlib.sha256(config.WHITENING_PATH.read_bytes()).hexdigest()},
        "split": {"query": len(qm), "gallery": len(gm), "known": int(sum(r["has_mate"] == "1" for r in qm)),
                  "unknown": int(sum(r["has_mate"] == "0" for r in qm))},
        "base": base, "old_threshold": assess(scores, old_tau), "reranked": reranked,
        "rerank_seconds": seconds,
        "delta_mAP": reranked["ranking"]["mAP"] - base["ranking"]["mAP"],
        "service_config": {k: getattr(config, k) for k in dir(config) if k.startswith(("DEFAULT_THRESHOLD", "RERANK"))},
        "batch": {name: json.loads((out / name / "run_info.json").read_text()) for name in modes},
    }
    # Перекалибровка обоих порогов — тем же правилом и в той же записи, что у
    # инструмента, которым посчитаны рабочие пороги (service/tools/calibrate_threshold.py,
    # он же в refusal/scripts/analyze.py и training/ensemble/scripts/s06_refusal.py):
    # максимум от min(TNR, F1 при долях отказных 0.10/0.25/0.40), при равенстве —
    # больший F1, затем больший TNR. Тай-брейк здесь — часть правила, а не оформление:
    # на косинусе минимум даёт именно TNR, а TNR постоянен на участках сетки без
    # «неизвестных» событий, поэтому максимум достигается на плато; его концы
    # (0.5141977 и 0.5145892, tp 566 против 565) расходятся на 3.9e-04 — в 400 раз
    # больше допуска сверки ниже.
    def recalibrate(matrix):
        """Операционные точки контура -> порог. Сетка кандидатов — каждое различное
        значение уверенности top-1; сентинел «ничего не принято» (threshold None)
        пропускается: recall 0 даёт F1 0, то есть цель 0, и выиграть он не может."""
        known, unknown = report["split"]["known"], report["split"]["unknown"]
        curve = evaluate(matrix, threshold=0., **metadata)["refusal"]["pr_curve"]
        rows = []
        for t, tp, fp in zip(curve["thresholds"], curve["tp"], curve["fp"]):
            if t is None:
                continue
            recall, fpr = tp / known, fp / unknown
            f1s = [2*(1-p)*recall / ((1-p)*(1+recall)+p*fpr) for p in (.10, .25, .40)]
            rows.append({"threshold": float(t), "objective": min(1-fpr, *f1s),
                         "f1_at_priors_0.10_0.25_0.40": f1s, "f1": 2*tp / (tp + known + fp),
                         "tnr": 1-fpr, "tp": tp, "fp": fp})
        return max(rows, key=lambda r: (r["objective"], r["f1"], r["tnr"], r["threshold"]))

    best = recalibrate(scores)
    report["calibration"] = dict(best, historical_threshold=tau,
                                 threshold_abs_delta=abs(best["threshold"] - tau))
    rr_best = recalibrate(1 - distance)
    report["rerank_calibration"] = dict(rr_best, historical_threshold=rr_tau,
                                        threshold_abs_delta=abs(rr_best["threshold"] - rr_tau))
    report["rerank_refusal"] = assess(1-distance, rr_best["threshold"])["refusal"]
    report["rerank_at_service_threshold"] = assess(
        1-distance, getattr(config, "DEFAULT_THRESHOLD_RERANK", rr_best["threshold"]))["refusal"]
    # Check model-to-row order by independently inferring boundary rows at batch 1.
    model = Embedder(threads=2)
    meta_rows = read_rows(split / "val_query.csv") + read_rows(split / "val_gallery.csv")
    report["row_order"] = {}
    for index in (0, len(qm)-1, len(qm), len(meta_rows)-1):
        vector = model.embed_one(load_crop(data / "images", meta_rows[index]))
        own = float(scores_from_embeddings(vector[None], emb[index:index+1])[0, 0])
        assert own > 1 - 1e-10
        report["row_order"][str(index)] = own
    # Informational speed, warm model, complete decode/crop/inference path.
    timings = []
    for _ in range(2):
        start = time.perf_counter()
        model.embed_rows(data / "images", meta_rows[:32], batch_size=1)
        timings.append((time.perf_counter()-start)*1000/32)
    report["current_batch1_ms_per_object"] = timings
    report["files"] = {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                       for name in modes for p in (out / name).iterdir() if p.is_file()}
    # Контрольная сверка с эталоном контура (конфигурация d1_j48, f64-конвейер
    # job_48). Допуск 1e-6: это в 3 порядка жёстче различимого, при этом честно для
    # пересчёта на другом CPU/потоках (последние биты float32-инференса плавают).
    reference = json.loads((repo / "04-solution/training/combined/s02_metrics.json").read_text())["d1_j48"]
    for name, block, key in (("base", base, "cos"), ("rerank_tuned", reranked, "kr")):
        for metric in ("mAP", "Rank-1", "Rank-5"):
            assert abs(block["ranking"][metric] - reference[key][metric]) < 1e-6, (name, metric)
    # И перекалибровка тем же правилом сходится к сохранённым порогам 20.09. Допуск
    # тот же 1e-6 и по той же причине: порог — это значение уверенности, оно едет
    # вместе с последними битами float32-инференса. Измерено: векторы этого прогона
    # расходятся с векторами прогона калибровки (job_55, другой хост и 4 потока) до
    # 1.2e-07, что даёт |Δ| порога 1.0e-08 на косинусе и 2.7e-09 на переранжировании.
    for scale, got, want in (("cos", best["threshold"], tau), ("rr", rr_best["threshold"], rr_tau)):
        assert abs(got - want) < 1e-6, (scale, got, want, abs(got - want))
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    # Обе конфигурации печатаются рядом и подписанными: одни и те же имена метрик
    # у косинуса и у переранжирования значат разные числа, и §6 приводит обе строки.
    for label, ranking, threshold, refusal in (
            ("cosine  (t_cos; HTTP API и batch --no-rerank; metrics.json -> base)",
             base["ranking"], best["threshold"], base["refusal"]),
            ("historical all-query KR (не режим сдачи; metrics.json -> rerank_refusal)",
             reranked["ranking"], rr_best["threshold"], report["rerank_refusal"])):
        print("REPRODUCED", label, json.dumps({
            "mAP": ranking["mAP"], "Rank-1": ranking["Rank-1"], "threshold": threshold,
            "F1": refusal["f1"], "TNR": refusal["tnr"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
