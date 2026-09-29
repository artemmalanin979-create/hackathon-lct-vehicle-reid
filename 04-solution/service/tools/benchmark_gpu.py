#!/usr/bin/env python3
"""Full-frame extraction benchmark matching the organizer's timing boundary.

Default protocol: 50 batch-1 warmups, 300 separately synchronized batch-1
measurements, then at least 10 seconds at each batch size 1/8/16/32. Each
timed call uses Embedder.embed_rows(), including JPEG read/decode, BBox crop,
preprocessing, both ONNX forwards, whitening, final L2 and output validation.
Gallery search, re-ranking, CSV parsing, weight loading and input preflight are
outside the timers. CPU runs require --allow-cpu and are labeled as smoke runs.
"""
from __future__ import annotations

import argparse
import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import numpy as np

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))

from app.core import config  # noqa: E402
from app.core.model import Embedder  # noqa: E402
from app.core.preprocess import (BBoxRow, crop_problems, read_rows,
                                 resolve_image_path)  # noqa: E402


BATCH_SIZES = (1, 8, 16, 32)
SCOPE = "disk_jpeg_bbox_preprocess_two_onnx_whitening_l2"


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("нужно целое число > 0") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("нужно целое число > 0")
    return parsed


def positive_seconds(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("нужно конечное число секунд > 0") from exc
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("нужно конечное число секунд > 0")
    return parsed


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _cuda_synchronizer():
    """Use CUDA runtime directly; do not add torch just for synchronization."""
    candidates = ["libcudart.so.12", "libcudart.so.13", "libcudart.so"]
    for root in sys.path:
        if root:
            candidates.extend(str(p) for p in
                              (Path(root) / "nvidia/cuda_runtime/lib").glob("libcudart.so*"))
    errors = []
    for candidate in dict.fromkeys(candidates):
        try:
            runtime = ctypes.CDLL(candidate)
            synchronize = runtime.cudaDeviceSynchronize
            synchronize.argtypes = []
            synchronize.restype = ctypes.c_int
            break
        except (OSError, AttributeError) as exc:
            errors.append(f"{candidate}: {exc}")
    else:
        raise RuntimeError("CUDA активна, но cudaDeviceSynchronize недоступен; "
                           "замер без синхронизации запрещён: " + "; ".join(errors))

    def sync() -> None:
        code = synchronize()
        if code != 0:
            raise RuntimeError(f"cudaDeviceSynchronize вернул код {code}")

    return sync


def _batch(rows: list[BBoxRow], cursor: int, size: int) -> tuple[list[BBoxRow], int]:
    selected = [rows[(cursor + offset) % len(rows)] for offset in range(size)]
    return selected, cursor + size


def _hardware() -> dict:
    result = {"platform": platform.platform(), "cpu_count": os.cpu_count(),
              "gpu": "NOT MEASURED"}
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=5, check=True)
        result["gpu"] = gpu.stdout.strip() or "NOT MEASURED"
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    return result


def run(args: argparse.Namespace) -> dict:
    if args.out.exists():
        raise SystemExit(f"файл результата уже существует: {args.out}")
    rows = read_rows(args.csv)
    if not rows:
        raise SystemExit("CSV не содержит кадров")
    problems = crop_problems(args.images_dir, rows, args.csv)
    if problems:
        raise SystemExit(f"входные кадры не прошли проверку ({len(problems)}): "
                         + "; ".join(problems[:3]))
    # Freeze input bytes outside the measured cycle. Later verify every frame
    # actually timed, so a changed JPEG cannot inherit an old input hash.
    path_by_id = {row.image_id: resolve_image_path(args.images_dir, row.image_id).resolve()
                  for row in rows}
    input_hashes_before = {path: file_sha256(path) for path in set(path_by_id.values())}

    load_start = time.perf_counter()
    embedder = Embedder(threads=args.threads)
    load_seconds = time.perf_counter() - load_start
    backend = embedder.inference_backend
    gpu_active = backend["active_device"] == "cuda"
    if not gpu_active and not args.allow_cpu:
        raise SystemExit("CUDA не активна; для проверочного CPU-прогона добавьте --allow-cpu. "
                         f"Фактический backend: {backend}")
    sync = _cuda_synchronizer() if gpu_active else lambda: None

    cursor = 0
    for _ in range(args.warmup):
        selected, cursor = _batch(rows, cursor, 1)
        embedder.embed_rows(args.images_dir, selected, batch_size=1)
        sync()

    timed_ranges = []
    latency_start_cursor = cursor
    latency_ms = []
    for _ in range(args.latency_runs):
        selected, cursor = _batch(rows, cursor, 1)
        sync()
        start = time.perf_counter_ns()
        embedder.embed_rows(args.images_dir, selected, batch_size=1)
        sync()
        latency_ms.append((time.perf_counter_ns() - start) / 1e6)
    timed_ranges.append((latency_start_cursor, cursor))

    throughput = {}
    throughput_details = {}
    for size in BATCH_SIZES:
        for _ in range(3):
            selected, cursor = _batch(rows, cursor, size)
            embedder.embed_rows(args.images_dir, selected, batch_size=size)
            sync()
        sync()
        start = time.perf_counter()
        throughput_start_cursor = cursor
        frames = 0
        while True:
            selected, cursor = _batch(rows, cursor, size)
            embedder.embed_rows(args.images_dir, selected, batch_size=size)
            sync()
            frames += size
            elapsed = time.perf_counter() - start
            if elapsed >= args.throughput_seconds:
                break
        timed_ranges.append((throughput_start_cursor, cursor))
        throughput[str(size)] = frames / elapsed
        throughput_details[str(size)] = {"frames": frames, "elapsed_s": elapsed}

    first = [rows[0]]
    repeat_a = embedder.embed_rows(args.images_dir, first, batch_size=1)
    sync()
    repeat_b = embedder.embed_rows(args.images_dir, first, batch_size=1)
    sync()
    first_image = resolve_image_path(args.images_dir, rows[0].image_id)
    timed_indices = set()
    for begin, end in timed_ranges:
        if end - begin >= len(rows):
            timed_indices.update(range(len(rows)))
        else:
            timed_indices.update(index % len(rows) for index in range(begin, end))
    timed_paths = sorted({path_by_id[rows[index].image_id] for index in timed_indices})
    timed_images = []
    for path in timed_paths:
        digest = file_sha256(path)
        if digest != input_hashes_before[path]:
            raise RuntimeError(f"кадр изменился во время замера: {path}")
        timed_images.append({"path": str(path), "bytes": path.stat().st_size,
                             "sha256": digest})
    weight_paths = (config.MODEL_PATH, config.MODEL2_PATH, config.WHITENING_PATH)
    weights = [{"path": str(path), "bytes": path.stat().st_size,
                "sha256": file_sha256(path)} for path in weight_paths]
    counts_and_gpu = (gpu_active and args.warmup == 50 and args.latency_runs == 300
                      and args.throughput_seconds >= 10)
    source_sha_valid = bool(re.fullmatch(r"[0-9a-f]{40}", args.source_sha))
    peak_vram_mib = "NOT MEASURED"  # Snapshots from nvidia-smi are not a true peak.
    protocol_ready = (counts_and_gpu and source_sha_valid and len(timed_images) >= 32
                      and isinstance(peak_vram_mib, (int, float)))
    report = {
        "scope": SCOPE,
        "reference": "official LCT case-7 Q&A, 2026-09-29, performance answer",
        "measurement_counts_and_gpu": counts_and_gpu,
        "official_counts_and_gpu": protocol_ready,
        "protocol_ready": protocol_ready,
        "protocol_limitations": [
            reason for condition, reason in (
                (counts_and_gpu, "GPU или число/длительность замеров не соответствуют ответу организатора"),
                (source_sha_valid, "полный 40-значный source SHA не указан"),
                (len(timed_images) >= 32, "использовано меньше 32 разных файлов кадров"),
                (isinstance(peak_vram_mib, (int, float)), "точный пиковый VRAM не измерен"),
            ) if not condition
        ],
        "peak_vram_mib": peak_vram_mib,
        "run_kind": "gpu_candidate" if gpu_active else "cpu_smoke_only",
        "inference_backend": backend,
        "cuda_synchronization": ("cudaDeviceSynchronize before/after each latency sample "
                                 "and after each throughput batch" if gpu_active else
                                 "not applicable: CPU"),
        "warmup_b1": args.warmup,
        "latency_b1": {
            "samples_ms": latency_ms,
            "median_ms": float(np.median(latency_ms)),
            "p95_ms": float(np.percentile(latency_ms, 95)),
        },
        "throughput_min_seconds_per_batch": args.throughput_seconds,
        "throughput_fps_by_batch": throughput,
        "throughput_details": throughput_details,
        "best_fps": max(throughput.values()),
        "weight_load_s": load_seconds,
        "determinism_first_frame": {
            "bitwise_equal": bool(np.array_equal(repeat_a, repeat_b)),
            "max_abs_delta": float(np.max(np.abs(repeat_a - repeat_b))),
        },
        "inputs": {
            "csv": str(args.csv), "csv_sha256": file_sha256(args.csv),
            "images_dir": str(args.images_dir), "rows": len(rows),
            "first_image": str(first_image),
            "first_image_sha256": file_sha256(first_image),
            "distinct_timed_images": len(timed_images),
            "timed_images": timed_images,
            "model_sha256": config.MODEL_SHA256,
            "model2_sha256": config.MODEL2_SHA256,
            "whitening_sha256": config.WHITENING_SHA256,
        },
        "weights": {"files": weights,
                    "total_bytes": sum(item["bytes"] for item in weights)},
        "toolchain": {"python": platform.python_version(), "numpy": np.__version__,
                      "onnxruntime": __import__("onnxruntime").__version__,
                      "source_sha": args.source_sha},
        "hardware": _hardware(),
        "threads": args.threads,
        "timing_note": "Warmed page cache may affect JPEG I/O; this is a local "
                       "measurement, not the organizer's score.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as target:
        json.dump(report, target, ensure_ascii=False, indent=2)
        target.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True,
                        help="CSV image_id,x,y,w,h; кадры повторно читаются с диска")
    parser.add_argument("--out", type=Path, required=True,
                        help="новый JSON файл результата; существующий не перезаписывается")
    parser.add_argument("--threads", type=positive_int, default=1,
                        help="число intra-op потоков ORT; по умолчанию 1")
    parser.add_argument("--warmup", type=positive_int, default=50)
    parser.add_argument("--latency-runs", type=positive_int, default=300)
    parser.add_argument("--throughput-seconds", type=positive_seconds, default=10.0)
    parser.add_argument("--allow-cpu", action="store_true",
                        help="разрешить только проверочный CPU-прогон, не GPU-результат")
    parser.add_argument("--source-sha", default=os.environ.get("SOURCE_SHA", "NOT PROVIDED"))
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"run_kind": result["run_kind"],
                      "median_ms": result["latency_b1"]["median_ms"],
                      "best_fps": result["best_fps"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
