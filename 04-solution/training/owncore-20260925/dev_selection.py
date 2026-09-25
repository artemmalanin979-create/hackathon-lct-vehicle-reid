#!/usr/bin/env python3
"""Freeze fusion weight and refusal thresholds using historical own-ID dev only.

This script never opens the external validation CSV or its labels. Baseline and
new core receive the same verified uint8 RGB crops, converted once per batch.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

from evaluate import (DEFAULT_REPO, HERE, _evaluation_modules, _validate_embeddings,
                      evaluate_matrix, fuse_embeddings, require_sha256,
                      score_matrix)


def choose_weight(grid: list[tuple[float, float]]) -> float:
    if [float(weight) for weight, _ in grid] != [0.25, 0.5, 0.75]:
        raise ValueError("fusion weight grid differs from frozen protocol")
    if any(not math.isfinite(float(value)) for _, value in grid):
        raise ValueError("nonfinite dev fusion mAP")
    return float(max(grid, key=lambda pair: (pair[1], -pair[0]))[0])


def calibrate_threshold(repo: Path, scores: np.ndarray, qm: list[dict],
                        gm: list[dict]) -> float:
    """Maximize dev presence F1, breaking exact ties toward higher threshold."""
    _, result = evaluate_matrix(repo, scores, qm, gm, threshold=None)
    curve = result["refusal"]["pr_curve"]
    candidates = []
    for threshold, precision, recall in zip(curve["thresholds"],
                                            curve["precision"], curve["recall"]):
        if threshold is None or precision is None or recall is None:
            continue
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        candidates.append((f1, float(threshold)))
    if not candidates:
        raise ValueError("dev has no finite refusal thresholds")
    return max(candidates)[1]


def _frozen_dev_indices(repo: Path, protocol: dict, count: int) -> tuple[np.ndarray, np.ndarray]:
    split_path = repo / protocol["data"]["split_file"]
    require_sha256(split_path, protocol["data"]["split_sha256"])
    split = json.loads(split_path.read_text(encoding="utf-8"))
    q, g, dev = (np.asarray(split[key], dtype=np.int64) for key in (
        "dev_query_indices", "dev_gallery_indices", "dev_indices"))
    if any(np.any(x < 0) or np.any(x >= count) or len(np.unique(x)) != len(x)
           for x in (q, g, dev)):
        raise ValueError("invalid dev row indices")
    if len(dev) != protocol["data"]["dev_rows_expected"] or not np.array_equal(
        np.sort(np.concatenate((q, g))), np.sort(dev)):
        raise ValueError("dev query/gallery no longer partition frozen dev")
    return q, g


def select_on_dev(repo: Path, protocol_path: Path, protocol_sha256: str,
                  metadata: Path, raw_crops: Path, model: Path, model_sha256: str,
                  out: Path, batch_size: int = 16) -> dict:
    if out.exists():
        raise FileExistsError("refusing to overwrite dev selection")
    require_sha256(protocol_path, protocol_sha256)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_training":
        raise ValueError("protocol was not frozen before training")
    require_sha256(metadata, protocol["data"]["train_source_sha256"])
    require_sha256(raw_crops, protocol["data"]["train_crops_sha256"])
    require_sha256(model, model_sha256)
    ev = protocol["evaluation"]
    for key, relative in (
        ("evaluator_sha256", "04-solution/eval/reid_metrics.py"),
        ("reranker_sha256", "04-solution/service/app/core/rerank.py"),
        ("preprocess_sha256", "04-solution/service/app/core/preprocess.py"),
        ("baseline_model_a_sha256", "04-solution/service/model/osnet_ain_x1_0_vehicle_reid.onnx"),
        ("baseline_model_b_sha256", "04-solution/service/model/osnet_ain_combined_v1.onnx"),
        ("baseline_whitening_sha256", "04-solution/service/model/lw_ens_j48_rho0.5.npz"),
    ):
        require_sha256(repo / relative, ev[key])
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    with np.load(metadata, allow_pickle=False) as meta:
        vids = np.asarray(meta["vehicle_id"], dtype=np.int64)
        cams = np.asarray(meta["camera_id"], dtype=np.int64)
    crops = np.load(raw_crops, mmap_mode="r", allow_pickle=False)
    if crops.shape != (len(vids), 208, 208, 3) or crops.dtype != np.uint8 or cams.shape != vids.shape:
        raise ValueError("metadata and uint8 NHWC crop rows are not aligned")
    q, g = _frozen_dev_indices(repo, protocol, len(crops))
    if len(np.unique(vids[np.concatenate((q, g))])) != protocol["data"]["dev_ids_expected"]:
        raise ValueError("historical dev identity count changed")
    _evaluation_modules(repo)
    from app.core.model import Embedder
    import onnxruntime as ort
    baseline = Embedder(threads=2)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    core = ort.InferenceSession(str(model), sess_options=options,
                                providers=["CPUExecutionProvider"])
    order = np.concatenate((q, g))
    base_vectors = np.empty((len(order), 512), dtype=np.float32)
    core_vectors = np.empty_like(base_vectors)
    for begin in range(0, len(order), batch_size):
        positions = order[begin:begin + batch_size]
        tensor = np.ascontiguousarray(crops[positions].transpose(0, 3, 1, 2), dtype=np.float32)
        base_vectors[begin:begin + len(positions)] = _validate_embeddings(
            baseline.embed_tensors(tensor), len(positions))
        core_vectors[begin:begin + len(positions)] = _validate_embeddings(
            core.run(None, {core.get_inputs()[0].name: tensor})[0], len(positions))
    gallery_ids = set(vids[g].tolist())
    qm = [{"vehicle_id": str(vids[i]), "camera_id": str(cams[i]),
           "has_mate": "1" if int(vids[i]) in gallery_ids else "0"} for i in q]
    gm = [{"vehicle_id": str(vids[i]), "camera_id": str(cams[i])} for i in g]
    nquery = len(q)
    fusion_grid = []
    for weight in (0.25, 0.5, 0.75):
        fused = fuse_embeddings(base_vectors, core_vectors, weight)
        matrix = score_matrix(repo, fused[:nquery], fused[nquery:], "cosine")
        metrics, _ = evaluate_matrix(repo, matrix, qm, gm, threshold=None)
        fusion_grid.append((weight, metrics["mAP"]))
    weight = choose_weight(fusion_grid)
    descriptors = {"baseline": base_vectors, "core": core_vectors,
                   "fusion": fuse_embeddings(base_vectors, core_vectors, weight)}
    thresholds, maps = {}, {}
    for name, vectors in descriptors.items():
        thresholds[name], maps[name] = {}, {}
        for mode in ("cosine", "KR"):
            scores = score_matrix(repo, vectors[:nquery], vectors[nquery:], mode)
            thresholds[name][mode] = calibrate_threshold(repo, scores, qm, gm)
            metrics, _ = evaluate_matrix(repo, scores, qm, gm, thresholds[name][mode])
            maps[name][mode] = metrics["mAP"]
    selected = {
        "selection_split": "dev", "status": "PASS", "protocol_sha256": protocol_sha256,
        "model_sha256": model_sha256, "fusion_core_weight": weight,
        "fusion_dev_cosine_grid": [{"core_weight": w, "mAP": m} for w, m in fusion_grid],
        "thresholds": thresholds, "dev_mAP": maps,
        "dev_query": len(q), "dev_gallery": len(g),
        "metadata_sha256": protocol["data"]["train_source_sha256"],
        "raw_crops_sha256": protocol["data"]["train_crops_sha256"],
        "split_sha256": protocol["data"]["split_sha256"],
        "release_whitening_sha256": ev["baseline_whitening_sha256"],
        "release_thresholds_separate_from_dev": {"cosine": 0.5141976914190476,
                                                 "KR": 0.5282812306342437},
        "fusion": "L2 per branch, sqrt-weight concatenation, one KR on all 1024-d vectors",
        "limitation": "release whitening historically saw dev100; dev is not an untouched baseline holdout",
        "runtime": {"python": sys.version, "numpy": np.__version__,
                    "onnxruntime": ort.__version__, "platform": platform.platform()},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selected, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--metadata", type=Path, required=True, help="combined_train.npz")
    parser.add_argument("--raw-crops", type=Path, required=True, help="combined_crops_208.npy")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True, help="new selection.json")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    report = select_on_dev(args.repo.resolve(), args.protocol, args.protocol_sha256,
                           args.metadata, args.raw_crops, args.model, args.model_sha256,
                           args.out, args.batch_size)
    print(json.dumps({"status": report["status"], "fusion_core_weight": report["fusion_core_weight"],
                      "thresholds": report["thresholds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
