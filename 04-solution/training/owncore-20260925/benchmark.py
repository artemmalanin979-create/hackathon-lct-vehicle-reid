#!/usr/bin/env python3
"""Paired CPU batch-1 tensor-to-descriptor benchmark of release/core/fusion.

The same 40 verified real validation crops and two ORT threads are used for all
three modes. JPEG decode, crop/resize, search and KR are outside the timer.
Cold start is session construction plus first inference in a fresh subprocess.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from evaluate import (DEFAULT_REPO, HERE, _evaluation_modules, _read_csv,
                      _validate_embeddings, fuse_embeddings, require_sha256,
                      sha256, validate_frozen_inputs)


MODES = ("baseline", "core", "fusion")


def paired_order(sample_index: int) -> tuple[str, str, str]:
    return MODES if sample_index % 2 == 0 else tuple(reversed(MODES))


def _runner(repo: Path, model: Path, core_weight: float,
            needed_modes: tuple[str, ...] = MODES):
    _evaluation_modules(repo)
    from app.core.model import Embedder
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    needs_baseline = any(mode in ("baseline", "fusion") for mode in needed_modes)
    needs_core = any(mode in ("core", "fusion") for mode in needed_modes)
    baseline = Embedder(threads=2) if needs_baseline else None
    core = (ort.InferenceSession(str(model), sess_options=options,
                                 providers=["CPUExecutionProvider"])
            if needs_core else None)
    input_name = core.get_inputs()[0].name if core is not None else None

    def run(mode: str, tensor: np.ndarray) -> np.ndarray:
        if mode == "baseline":
            return _validate_embeddings(baseline.embed_tensors(tensor), len(tensor))
        if mode == "core":
            return _validate_embeddings(core.run(None, {input_name: tensor})[0], len(tensor))
        if mode == "fusion":
            b = _validate_embeddings(baseline.embed_tensors(tensor), len(tensor))
            c = _validate_embeddings(core.run(None, {input_name: tensor})[0], len(tensor))
            fused = fuse_embeddings(b, c, core_weight)
            if fused.shape != (len(tensor), 1024) or not np.isfinite(fused).all():
                raise ValueError("fusion descriptor is invalid")
            return fused
        raise ValueError(f"unknown benchmark mode: {mode}")

    return run


def _cold(repo: Path, model: Path, model_sha256: str, sample: Path,
          mode: str, weight: float) -> dict:
    require_sha256(model, model_sha256)
    tensor = np.load(sample, allow_pickle=False)
    if tensor.shape != (1, 3, 208, 208) or tensor.dtype != np.float32:
        raise ValueError("cold sample must be one RGB float32 NCHW crop")
    start = time.perf_counter()
    run = _runner(repo, model, weight, (mode,))
    output = run(mode, tensor)
    duration = time.perf_counter() - start
    return {"cold_start_seconds": duration,
            "fresh_process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "output_dimensions": output.shape[1]}


def _limits(measured: dict, protocol: dict) -> dict:
    limits = protocol["benchmark"]["release_performance_limits"]
    base = measured["baseline"]
    checked = {}
    for mode in ("core", "fusion"):
        row = measured[mode]
        checked[mode] = {
            "p95": row["batch1_p95_ms"] <= limits["batch1_p95_max_ratio_to_same_run_baseline"] * base["batch1_p95_ms"],
            "FPS": row["FPS_batch1"] >= limits["batch1_FPS_min_ratio_to_same_run_baseline"] * base["FPS_batch1"],
            "cold_start": row["cold_start_seconds"] <= limits["cold_start_max_ratio_to_same_run_baseline"] * base["cold_start_seconds"],
            "peak_RSS": row["fresh_process_peak_rss_mib"] <= limits["peak_RSS_max_ratio_to_same_run_baseline"] * base["fresh_process_peak_rss_mib"],
            "weights": row["weights_bytes"] <= limits["standalone_weight_bytes_max" if mode == "core" else "fusion_weight_bytes_max"],
            "task_weights": row["weights_bytes"] <= limits["task_weight_bytes_max"],
        }
    return checked


def benchmark(repo: Path, data_dir: Path, protocol_path: Path, protocol_sha256: str,
              model: Path, model_sha256: str, selection_path: Path,
              selection_sha256: str, out: Path) -> dict:
    import onnxruntime as ort
    if out.exists():
        raise FileExistsError("refusing to overwrite benchmark results")
    require_sha256(protocol_path, protocol_sha256)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require_sha256(model, model_sha256)
    require_sha256(selection_path, selection_sha256)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (selection.get("selection_split") != "dev" or
            selection.get("model_sha256") != model_sha256 or
            selection.get("protocol_sha256") != protocol_sha256):
        raise ValueError("dev selection/model/protocol mismatch")
    weight = selection["fusion_core_weight"]
    input_report = validate_frozen_inputs(repo, data_dir, protocol)
    _evaluation_modules(repo)
    from app.core.preprocess import read_rows, load_crop
    rows = read_rows(repo / "04-solution/split/files/val_query.csv")[:45]
    if len(rows) != 45:
        raise ValueError("need frozen first 45 val query crops")
    tensors = [load_crop(data_dir / "images", row)[None] for row in rows]
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / "same45-crops.npy", np.concatenate(tensors), allow_pickle=False)
    np.save(out / "cold-crop.npy", tensors[0], allow_pickle=False)
    cold = {}
    for mode in MODES:
        command = [sys.executable, "-B", __file__, "--repo", str(repo),
                   "--model", str(model), "--model-sha256", model_sha256,
                   "--cold-mode", mode, "--cold-sample", str(out / "cold-crop.npy"),
                   "--fusion-weight", str(weight)]
        result = subprocess.run(command, text=True, capture_output=True, check=True,
                                timeout=180, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        cold[mode] = json.loads(result.stdout)
    run = _runner(repo, model, weight)
    for tensor in tensors[:5]:
        for mode in MODES:
            run(mode, tensor)
    samples = {mode: [] for mode in MODES}
    for index, tensor in enumerate(tensors[5:]):
        for mode in paired_order(index):
            start = time.perf_counter()
            output = run(mode, tensor)
            elapsed = time.perf_counter() - start
            if output.shape[0] != 1:
                raise ValueError("batch-1 inference produced another batch size")
            samples[mode].append(elapsed)
    model_dir = repo / "04-solution/service/model"
    base_bytes = sum((model_dir / name).stat().st_size for name in (
        "osnet_ain_x1_0_vehicle_reid.onnx", "osnet_ain_combined_v1.onnx",
        "lw_ens_j48_rho0.5.npz"))
    measured = {}
    for mode in MODES:
        values = np.asarray(samples[mode], dtype=np.float64)
        size = base_bytes if mode == "baseline" else model.stat().st_size if mode == "core" else base_bytes + model.stat().st_size
        dimensions = 1024 if mode == "fusion" else 512
        measured[mode] = {
            "batch1_p50_ms": float(np.quantile(values, .5) * 1000),
            "batch1_p95_ms": float(np.quantile(values, .95) * 1000),
            "FPS_batch1": float(1 / values.mean()),
            "samples_seconds": values.tolist(), "cold_start_seconds": cold[mode]["cold_start_seconds"],
            "fresh_process_peak_rss_mib": cold[mode]["fresh_process_peak_rss_mib"],
            "weights_bytes": size, "embedding_dimensions": dimensions,
            "gallery750_raw_index_bytes": 750 * dimensions * 4,
            "VRAM": "NOT APPLICABLE: CPUExecutionProvider",
        }
    checks = _limits(measured, protocol)
    report = {
        "status": "PASS", "measurements": measured, "limit_checks": checks,
        "threads": 2, "batch": 1, "warmup": 5, "samples": 40,
        "input": "frozen val query CSV first45; first5 warmup, next40 timed; all JPG and CSV SHA-256 checked",
        "same45_crops_sha256": sha256(out / "same45-crops.npy"),
        "boundary": "preprocessed tensor to normalized descriptor; no JPEG decode/API/Qdrant/KR",
        "cold_boundary": "session construction plus first inference in fresh process, Python import excluded",
        "fusion": "both image encoders run once plus 1024-d sqrt-weight concatenation",
        "scope": "same workstation; alternating paired order limits but does not eliminate contention",
        "source": {"protocol_sha256": protocol_sha256, "model_sha256": model_sha256,
                   "dev_selection_sha256": selection_sha256,
                   "verified_input_files": input_report["required_files"]},
        "runtime": {"onnxruntime": ort.__version__, "numpy": np.__version__,
                    "python": sys.version, "platform": platform.platform()},
        "concurrent_process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    (out / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False,
                                                  indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--dev-selection", type=Path)
    parser.add_argument("--dev-selection-sha256")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cold-mode", choices=MODES)
    parser.add_argument("--cold-sample", type=Path)
    parser.add_argument("--fusion-weight", type=float)
    args = parser.parse_args()
    if args.cold_mode:
        if args.cold_sample is None or args.fusion_weight is None:
            parser.error("cold mode needs sample and fusion weight")
        print(json.dumps(_cold(args.repo.resolve(), args.model, args.model_sha256,
                               args.cold_sample, args.cold_mode, args.fusion_weight)))
        return
    if not all((args.data_dir, args.protocol_sha256, args.dev_selection,
                args.dev_selection_sha256, args.out)):
        parser.error("benchmark needs data-dir, protocol SHA, dev selection/SHA and new out")
    report = benchmark(args.repo.resolve(), args.data_dir.resolve(), args.protocol,
                       args.protocol_sha256, args.model, args.model_sha256,
                       args.dev_selection, args.dev_selection_sha256, args.out)
    print(json.dumps({"status": "PASS", "measurements": {
        mode: {key: value for key, value in row.items() if key != "samples_seconds"}
        for mode, row in report["measurements"].items()},
        "limit_checks": report["limit_checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
