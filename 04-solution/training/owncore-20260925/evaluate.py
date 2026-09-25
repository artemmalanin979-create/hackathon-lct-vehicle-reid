#!/usr/bin/env python3
"""Compare a standalone image encoder with the released ensemble on real val JPGs.

No cached features are accepted. The released baseline is freshly extracted and
must reproduce its published cosine and KR mAP before candidate scores are read.
The repeatedly used validation is diagnostic, not an untouched holdout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[2]
EXPECTED_BASELINE = {"cosine": 0.7316093422310448, "KR": 0.7740915539438481}
RELEASE_THRESHOLDS = {"cosine": 0.5141976914190476, "KR": 0.5282812306342437}


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_sha256(path: Path, expected: str) -> str:
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("expected SHA-256 must be 64 lowercase hexadecimal characters")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}: {actual} != {expected}")
    return actual


def read_dev_thresholds(path: Path, expected_sha256: str) -> dict:
    """Accept only immutable dev selection; never derive refusal on val labels."""
    require_sha256(path, expected_sha256)
    selection = json.loads(Path(path).read_text(encoding="utf-8"))
    if selection.get("selection_split") != "dev":
        raise ValueError("threshold selection must come from dev")
    thresholds = selection.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("dev selection is missing thresholds")
    for mode in ("cosine", "KR"):
        value = thresholds.get(mode)
        if isinstance(value, dict):
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"invalid dev {mode} threshold")
    return thresholds


def read_dev_selection(path: Path, expected_sha256: str) -> dict:
    require_sha256(path, expected_sha256)
    selected = json.loads(Path(path).read_text(encoding="utf-8"))
    if selected.get("selection_split") != "dev":
        raise ValueError("fusion and threshold selection must come from dev")
    thresholds = selected.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("dev selection is missing thresholds")
    for model in ("baseline", "core", "fusion"):
        block = thresholds.get(model)
        if not isinstance(block, dict):
            raise ValueError(f"dev selection is missing {model} thresholds")
        for mode in ("cosine", "KR"):
            value = block.get(mode)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"invalid {model}/{mode} dev threshold")
    weight = selected.get("fusion_core_weight")
    if weight not in (0.25, 0.5, 0.75):
        raise ValueError("fusion weight was not in the frozen dev grid")
    return selected


def _evaluation_modules(repo: Path):
    eval_dir = str(repo / "04-solution/eval")
    service_dir = str(repo / "04-solution/service")
    for directory in (eval_dir, service_dir):
        if directory not in sys.path:
            sys.path.insert(0, directory)
    from reid_metrics import evaluate, scores_from_embeddings
    from app.core.rerank import rerank_scores
    return evaluate, scores_from_embeddings, rerank_scores


def score_matrix(repo: Path, query: np.ndarray, gallery: np.ndarray, mode: str) -> np.ndarray:
    _, cosine, kr = _evaluation_modules(repo)
    if mode == "cosine":
        return cosine(query, gallery)
    if mode == "KR":
        return kr(query, gallery, 6, 3, 0.3)
    raise ValueError(f"unknown ranking mode: {mode}")


def evaluate_matrix(repo: Path, scores: np.ndarray, query_meta: list[dict],
                    gallery_meta: list[dict], threshold: float | None) -> tuple[dict, dict]:
    evaluate, _, _ = _evaluation_modules(repo)
    if scores.shape != (len(query_meta), len(gallery_meta)):
        raise ValueError("score rows/columns are not aligned with frozen query/gallery CSV")
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite retrieval scores")
    result = evaluate(
        scores, query_ids=[r["vehicle_id"] for r in query_meta],
        gallery_ids=[r["vehicle_id"] for r in gallery_meta],
        query_cameras=[r["camera_id"] for r in query_meta],
        gallery_cameras=[r["camera_id"] for r in gallery_meta],
        known_absent=[r["has_mate"] == "0" for r in query_meta],
        threshold=float("inf") if threshold is None else float(threshold),
        camera_policy="market", refusal_mode="presence", ap_method="step",
    )
    ranking, refusal = result["ranking_full_gallery"], result["refusal"]
    summary = {
        "mAP": ranking["mAP"], "Rank-1": ranking["Rank-1"],
        "Rank-5": ranking["Rank-5"], "mINP": ranking["mINP"],
        "F1": "NOT MEASURED" if threshold is None else refusal["f1"],
        "TNR": "NOT MEASURED" if threshold is None else refusal["tnr"],
        "threshold": "NOT MEASURED" if threshold is None else float(threshold),
        "counts": result["counts"],
    }
    return summary, result


def score_and_summarize(repo: Path, query: np.ndarray, gallery: np.ndarray,
                        query_meta: list[dict], gallery_meta: list[dict], *,
                        mode: str, threshold: float | None) -> dict:
    summary, _ = evaluate_matrix(repo, score_matrix(repo, query, gallery, mode),
                                 query_meta, gallery_meta, threshold)
    return summary


def require_reproduced_baseline(observed: dict, expected: dict | None = None,
                                tolerance: float = 1e-6) -> None:
    frozen = EXPECTED_BASELINE if expected is None else expected
    for mode in ("cosine", "KR"):
        actual = observed[mode]["mAP"]
        if actual is None or abs(actual - frozen[mode]) > tolerance:
            raise AssertionError(f"baseline {mode} mAP not reproduced: {actual} != {frozen[mode]}")
    rank1 = observed["KR"].get("Rank-1")
    if rank1 is not None and abs(rank1 - 0.7307692307692307) > tolerance:
        raise AssertionError(f"baseline KR Rank-1 not reproduced: {rank1}")


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate_frozen_inputs(repo: Path, data_dir: Path, protocol: dict) -> dict:
    """Check CSV, every val JPG and implementation before any model inference."""
    ev, dat = protocol["evaluation"], protocol["data"]
    for key, relative in (
        ("evaluator_sha256", "04-solution/eval/reid_metrics.py"),
        ("reranker_sha256", "04-solution/service/app/core/rerank.py"),
        ("preprocess_sha256", "04-solution/service/app/core/preprocess.py"),
        ("baseline_model_a_sha256", "04-solution/service/model/osnet_ain_x1_0_vehicle_reid.onnx"),
        ("baseline_model_b_sha256", "04-solution/service/model/osnet_ain_combined_v1.onnx"),
        ("baseline_whitening_sha256", "04-solution/service/model/lw_ens_j48_rho0.5.npz"),
    ):
        require_sha256(repo / relative, ev[key])
    split = repo / "04-solution/split/files"
    require_sha256(split / "val_query.csv", dat["val_query_csv_sha256"])
    require_sha256(split / "val_gallery.csv", dat["val_gallery_csv_sha256"])
    local_manifest_path = repo / dat["val_images_manifest_local"]
    require_sha256(local_manifest_path, dat["val_images_manifest_sha256"])
    image_manifest = json.loads(local_manifest_path.read_text(encoding="utf-8"))
    if image_manifest["count"] != dat["val_images_count"]:
        raise ValueError("frozen val image count changed")
    image_ids = [r["image_id"] for r in _read_csv(split / "val_query.csv") +
                 _read_csv(split / "val_gallery.csv")]
    if set(image_ids) != {entry["image_id"] for entry in image_manifest["files"]}:
        raise ValueError("val image IDs do not match frozen CSV rows")
    reproduce_dir = str(repo / "04-solution/reproduce")
    if reproduce_dir not in sys.path:
        sys.path.insert(0, reproduce_dir)
    from check_inputs import check_inputs
    result = check_inputs(repo, data_dir, "val")
    if not result["ok"]:
        raise ValueError("val input image/CSV hash verification failed: " + json.dumps({
            key: result[key][:3] for key in ("missing", "mismatched", "unreadable")}, ensure_ascii=False))
    return result


def _validate_embeddings(vectors: np.ndarray, rows: int) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.shape != (rows, 512) or not np.isfinite(array).all():
        raise ValueError(f"model returned invalid descriptor matrix {array.shape}")
    if np.max(np.abs(np.linalg.norm(array.astype(np.float64), axis=1) - 1)) > 1e-4:
        raise ValueError("model returned non-unit descriptors")
    return array


def extract_images(repo: Path, images_dir: Path, rows: list, baseline=None,
                   candidate=None, batch_size: int = 16) -> np.ndarray:
    """Fresh image/bbox extraction in exact CSV order, never from an old NPY."""
    _evaluation_modules(repo)
    from app.core.preprocess import load_crop
    if (baseline is None) == (candidate is None):
        raise ValueError("select exactly one image encoder")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output = np.empty((len(rows), 512), dtype=np.float32)
    for begin in range(0, len(rows), batch_size):
        crops = np.stack([load_crop(images_dir, row) for row in rows[begin:begin + batch_size]])
        if baseline is not None:
            vectors = baseline.embed_tensors(crops)
        else:
            vectors = candidate.run(None, {candidate.get_inputs()[0].name: crops})[0]
        output[begin:begin + len(crops)] = _validate_embeddings(vectors, len(crops))
    return output


def paired_bootstrap(candidate: dict, baseline: dict, query_meta: list[dict], *,
                     seed: int = 20260925, repeats: int = 4000) -> dict:
    left, right = candidate["per_query"], baseline["per_query"]
    valid = [i for i, (a, b) in enumerate(zip(left, right))
             if a["status"] == b["status"] == "known"]
    ids = np.array([query_meta[i]["vehicle_id"] for i in valid])
    differences = np.array([left[i]["ap"] - right[i]["ap"] for i in valid], dtype=np.float64)
    unique = sorted(set(ids.tolist()))
    sums = np.array([differences[ids == identity].sum() for identity in unique])
    counts = np.array([(ids == identity).sum() for identity in unique])
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(unique), size=(repeats, len(unique)))
    values = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    p = min(1., 2 * min((np.count_nonzero(values <= 0) + 1) / (repeats + 1),
                        (np.count_nonzero(values >= 0) + 1) / (repeats + 1)))
    return {"delta_mAP": float(differences.mean()),
            "ci95": np.quantile(values, [.025, .975]).tolist(), "p_two_sided": float(p),
            "queries": len(valid), "vehicle_ids": len(unique), "repeats": repeats, "seed": seed}


def holm_adjusted(p_values: list[float]) -> list[float]:
    """Familywise adjustment for the four preregistered mAP comparisons."""
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in p_values):
        raise ValueError("Holm inputs must be finite probabilities")
    ordered = sorted(range(len(p_values)), key=lambda index: p_values[index])
    result = [0.] * len(ordered)
    maximum = 0.
    for rank, index in enumerate(ordered):
        maximum = max(maximum, min(1., (len(ordered) - rank) * p_values[index]))
        result[index] = maximum
    return result


def fuse_embeddings(baseline: np.ndarray, core: np.ndarray, core_weight: float) -> np.ndarray:
    """1024-d research descriptor; KR must see these vectors together once."""
    if core_weight not in (0.25, 0.5, 0.75) or baseline.shape != core.shape:
        raise ValueError("fusion requires aligned vectors and a frozen dev-grid weight")
    a, b = np.asarray(baseline, np.float64), np.asarray(core, np.float64)
    if np.any(np.linalg.norm(a, axis=1) == 0) or np.any(np.linalg.norm(b, axis=1) == 0):
        raise ValueError("fusion of zero descriptors is undefined")
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return np.concatenate((np.sqrt(1 - core_weight) * a,
                           np.sqrt(core_weight) * b), axis=1).astype(np.float32)


def _camera_gap(repo: Path, scores: np.ndarray, qm: list[dict], gm: list[dict],
                true_map: float) -> float:
    # Give every query/gallery a distinct camera. This diagnostic restores
    # same-camera positives while keeping identity and score rows unchanged.
    qcopy = [dict(row, camera_id=f"q:{i}") for i, row in enumerate(qm)]
    gcopy = [dict(row, camera_id=f"g:{i}") for i, row in enumerate(gm)]
    summary, _ = evaluate_matrix(repo, scores, qcopy, gcopy, threshold=None)
    return float(summary["mAP"] - true_map)


def compare_arrays(repo: Path, baseline: np.ndarray, core: np.ndarray, qm: list[dict],
                   gm: list[dict], *, selection: dict | None) -> dict:
    nquery, ngallery = len(qm), len(gm)
    if baseline.shape != core.shape or baseline.shape != (nquery + ngallery, 512):
        raise ValueError("model rows or 512-dimensional interface mismatch")
    base_scores = {mode: score_matrix(repo, baseline[:nquery], baseline[nquery:], mode)
                   for mode in ("cosine", "KR")}
    output, raw = {name: {} for name in ("baseline", "core", "fusion")}, {}
    for mode in ("cosine", "KR"):
        dev_baseline = selection["thresholds"]["baseline"][mode] if selection else RELEASE_THRESHOLDS[mode]
        summary, result = evaluate_matrix(repo, base_scores[mode], qm, gm, dev_baseline)
        output["baseline"][mode] = summary
        raw[("baseline", mode)] = result
    require_reproduced_baseline(output["baseline"])
    core_scores = {mode: score_matrix(repo, core[:nquery], core[nquery:], mode)
                   for mode in ("cosine", "KR")}
    for mode in ("cosine", "KR"):
        threshold = selection["thresholds"]["core"][mode] if selection else None
        summary, result = evaluate_matrix(repo, core_scores[mode], qm, gm, threshold)
        output["core"][mode] = summary
        raw[("core", mode)] = result
    fusion_scores = {}
    if selection is not None:
        fused = fuse_embeddings(baseline, core, selection["fusion_core_weight"])
        for mode in ("cosine", "KR"):
            scores = score_matrix(repo, fused[:nquery], fused[nquery:], mode)
            fusion_scores[mode] = scores
            summary, result = evaluate_matrix(repo, scores, qm, gm,
                                              selection["thresholds"]["fusion"][mode])
            output["fusion"][mode] = summary
            raw[("fusion", mode)] = result
    else:
        output["fusion"] = "NOT MEASURED: no dev-selected weight"
    for name in ("baseline", "core", "fusion"):
        if name == "fusion" and selection is None:
            continue
        for mode in ("cosine", "KR"):
            scores = base_scores[mode] if name == "baseline" else core_scores[mode] if name == "core" else fusion_scores[mode]
            output[name][mode]["same_camera_gap_mAP"] = _camera_gap(
                repo, scores, qm, gm, output[name][mode]["mAP"])
            if name != "baseline":
                output[name][mode]["paired_vs_baseline"] = paired_bootstrap(
                    raw[(name, mode)], raw[("baseline", mode)], qm)
    if selection is not None:
        family = [(name, mode) for name in ("core", "fusion")
                  for mode in ("cosine", "KR")]
        adjusted = holm_adjusted([output[name][mode]["paired_vs_baseline"]["p_two_sided"]
                                  for name, mode in family])
        for (name, mode), value in zip(family, adjusted):
            output[name][mode]["paired_vs_baseline"]["p_Holm_four_tests"] = value
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--model", type=Path, help="Standalone ONNX core")
    parser.add_argument("--model-sha256")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--baseline-dir", type=Path,
                        help="Reuse this run's hash-checked fresh baseline vectors")
    parser.add_argument("--dev-selection", type=Path)
    parser.add_argument("--dev-selection-sha256")
    parser.add_argument("--out", type=Path, required=True, help="New results directory")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.dev_selection is None and args.dev_selection_sha256 is not None or (
        args.dev_selection is not None and args.dev_selection_sha256 is None):
        parser.error("dev selection path and SHA-256 must be supplied together")
    if args.baseline_only and (args.model or args.baseline_dir or args.dev_selection):
        parser.error("baseline-only takes no candidate or previous baseline")
    if not args.baseline_only and (not args.model or not args.model_sha256):
        parser.error("comparison needs standalone ONNX model and SHA-256")
    repo, data_dir = args.repo.resolve(), args.data_dir.resolve()
    require_sha256(args.protocol, args.protocol_sha256)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_training":
        raise ValueError("protocol was not frozen before training")
    if args.model:
        require_sha256(args.model, args.model_sha256)
    selection = read_dev_selection(args.dev_selection, args.dev_selection_sha256) if args.dev_selection else None
    if selection is not None and (selection.get("model_sha256") != args.model_sha256 or
                                  selection.get("protocol_sha256") != args.protocol_sha256):
        raise ValueError("dev selection was made for another model/protocol")
    args.out.mkdir(parents=True, exist_ok=False)
    input_report = validate_frozen_inputs(repo, data_dir, protocol)
    (args.out / "input-check.json").write_text(json.dumps(input_report, indent=2) + "\n")
    _evaluation_modules(repo)
    from app.core.model import Embedder
    from app.core.preprocess import read_rows
    import onnxruntime as ort
    split = repo / "04-solution/split/files"
    qmeta, gmeta = _read_csv(split / "val_query.csv"), _read_csv(split / "val_gallery.csv")
    qrows, grows = read_rows(split / "val_query.csv"), read_rows(split / "val_gallery.csv")
    if [len(qrows), len(grows)] != protocol["data"]["val_rows_expected"]:
        raise ValueError("frozen validation row count changed")
    if [r.image_id for r in qrows + grows] != [r["image_id"] for r in qmeta + gmeta]:
        raise ValueError("model and evaluator CSV orders differ")
    if args.baseline_dir:
        prior = json.loads((args.baseline_dir / "baseline-report.json").read_text(encoding="utf-8"))
        if (prior["protocol_sha256"] != args.protocol_sha256 or
                prior["val_manifest_sha256"] != protocol["data"]["val_images_manifest_sha256"] or
                prior["status"] != "PASS"):
            raise ValueError("prior fresh baseline provenance differs")
        prior_vectors = args.baseline_dir / "baseline-fresh.npy"
        require_sha256(prior_vectors, prior["baseline_fresh_sha256"])
        baseline = _validate_embeddings(np.load(prior_vectors, allow_pickle=False),
                                        len(qrows) + len(grows))
    else:
        baseline_runner = Embedder(threads=2)
        baseline = extract_images(repo, data_dir / "images", qrows + grows,
                                  baseline=baseline_runner, batch_size=args.batch_size)
    # Refuse to compare a new model if this machine no longer reproduces release.
    baseline_metrics = {mode: score_and_summarize(
        repo, baseline[:len(qrows)], baseline[len(qrows):], qmeta, gmeta,
        mode=mode, threshold=RELEASE_THRESHOLDS[mode]) for mode in ("cosine", "KR")}
    np.save(args.out / "baseline-fresh.npy", baseline, allow_pickle=False)
    require_reproduced_baseline(baseline_metrics)
    baseline_record = {
        "status": "PASS", "protocol_sha256": args.protocol_sha256,
        "val_manifest_sha256": protocol["data"]["val_images_manifest_sha256"],
        "baseline_fresh_sha256": sha256(args.out / "baseline-fresh.npy"),
        "metrics": baseline_metrics, "verified_input_files": input_report["required_files"],
        "source": "fresh image/bbox extraction" if not args.baseline_dir else
                  f"hash-checked fresh baseline from {args.baseline_dir}",
    }
    (args.out / "baseline-report.json").write_text(json.dumps(
        baseline_record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.baseline_only:
        print(json.dumps(baseline_record, ensure_ascii=False))
        return 0
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    candidate = ort.InferenceSession(str(args.model), sess_options=options,
                                     providers=["CPUExecutionProvider"])
    core = extract_images(repo, data_dir / "images", qrows + grows,
                          candidate=candidate, batch_size=args.batch_size)
    np.save(args.out / "core-fresh.npy", core, allow_pickle=False)
    report = {
        "status": "PASS", "scope": "reused val1110x750; not an untouched holdout",
        "metrics": compare_arrays(repo, baseline, core, qmeta, gmeta, selection=selection),
        "inputs": {"protocol_sha256": args.protocol_sha256, "model_sha256": args.model_sha256,
                   "dev_selection_sha256": args.dev_selection_sha256 or "NOT MEASURED",
                   "val_manifest_sha256": protocol["data"]["val_images_manifest_sha256"],
                   "verified_files": input_report["required_files"]},
        "runtime": {"python": sys.version, "numpy": np.__version__,
                    "onnxruntime": ort.__version__, "platform": platform.platform()},
        "baseline_fresh_sha256": sha256(args.out / "baseline-fresh.npy"),
        "core_fresh_sha256": sha256(args.out / "core-fresh.npy"),
        "fusion": "research-only 1024-d concatenation; cosine is weighted pairwise score, KR runs once on all concatenated vectors",
        "refusal": "dev-frozen thresholds only; no dev selection means NOT MEASURED",
    }
    (args.out / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False,
                                                       indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "PASS", "metrics": report["metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
