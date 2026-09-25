#!/usr/bin/env python3
"""Create fit-only, row-checked teacher descriptors for CrossViewCore distillation.

The two release ONNX files are used offline only. New training never sees the
historical dev rows through this artifact; inference uses one student model.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[2]
CSV_RELATIVE = Path("04-solution/split/files/train_fit.csv")
LIB_RELATIVE = Path("04-solution/training/src/postprocess/lib45.py")
MODEL_A_RELATIVE = Path("04-solution/service/model/osnet_ain_x1_0_vehicle_reid.onnx")
MODEL_B_RELATIVE = Path("04-solution/service/model/osnet_ain_combined_v1.onnx")
NPZ_KEYS = frozenset(("row_indices", "target", "vehicle_id", "camera_id",
                      "image_id", "P", "m"))


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_sha256(path: Path, expected: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("expected SHA-256 must be lowercase 64-hex")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")
    return actual


def array_sha256(values: np.ndarray) -> str:
    """Raw C-order bytes; dtype/shape are separately recorded in the manifest."""
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    vectors = np.asarray(values, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 512 or not np.isfinite(vectors).all():
        raise ValueError("teacher vectors must be finite N x 512")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("zero or nonfinite teacher vector")
    return vectors / norms


def load_historical_whitening(source: Path, expected_sha256: str) -> dict:
    """Execute only hash-pinned historical l2n/learn_lw/apply_lw function ASTs."""
    require_sha256(source, expected_sha256)
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    names = ("l2n", "learn_lw", "apply_lw")
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in names]
    if [node.name for node in nodes] != list(names):
        raise ValueError("hash-pinned lib45 lacks historical whitening functions")
    scope = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    return {name: scope[name] for name in names}


@dataclass(frozen=True)
class DatasetRows:
    crops: np.ndarray
    row_indices: np.ndarray
    vehicle_id: np.ndarray
    camera_id: np.ndarray
    image_id: np.ndarray
    own_count: int
    fit_own_count: int
    split: dict


def _indices(split: dict, key: str, upper: int) -> np.ndarray:
    values = np.asarray(split[key], dtype=np.int64)
    if values.ndim != 1 or np.any(values < 0) or np.any(values >= upper):
        raise ValueError(f"{key} out of range")
    if len(np.unique(values)) != len(values):
        raise ValueError(f"{key} contains duplicate rows")
    return values


def _metadata(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as archive:
        images = np.asarray(archive["image_id"])
        vehicles = np.asarray(archive["vehicle_id"], dtype=np.int64)
        cameras = np.asarray(archive["camera_id"], dtype=np.int64)
        sources = np.asarray(archive["source"]) if "source" in archive else None
    if (images.ndim != 1 or vehicles.shape != images.shape or cameras.shape != images.shape
            or images.dtype.kind != "U" or (sources is not None and sources.shape != images.shape)):
        raise ValueError("metadata image/vehicle/camera/source rows are malformed")
    if len(np.unique(images)) != len(images):
        raise ValueError("duplicate image_id in metadata")
    return images, vehicles, cameras, sources


def _assert_csv_order(csv_path: Path, own_images: np.ndarray,
                      own_vehicles: np.ndarray, own_cameras: np.ndarray) -> None:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(own_images):
        raise ValueError("CSV row count differs from own metadata")
    for index, row in enumerate(rows):
        if (row["image_id"] != own_images[index]
                or int(row["vehicle_id"]) != int(own_vehicles[index])
                or int(row["camera_id"]) != int(own_cameras[index])):
            raise ValueError(f"CSV row order/identity disagrees with own metadata at {index}")


def verify_dataset(repo: Path, protocol: dict, metadata: Path, raw: Path, *,
                   own_protocol: dict | None = None, own_metadata: Path | None = None,
                   own_raw: Path | None = None) -> DatasetRows:
    """Verify all inputs, identity partition, and original row order before inference."""
    data = protocol["data"]
    require_sha256(metadata, data["train_source_sha256"])
    require_sha256(raw, data["train_crops_sha256"])
    split_file = repo / data["split_file"]
    require_sha256(split_file, data["split_sha256"])
    csv_file = repo / CSV_RELATIVE
    require_sha256(csv_file, protocol["teacher"]["train_fit_csv_sha256"])
    split = json.loads(split_file.read_text(encoding="utf-8"))
    image, vehicle, camera, source = _metadata(metadata)
    crops = np.load(raw, mmap_mode="r", allow_pickle=False)
    if crops.shape != (len(image), 208, 208, 3) or crops.dtype != np.uint8:
        raise ValueError("raw crops must align with metadata as uint8 NHWC 208")

    combined = own_protocol is not None
    if combined:
        if own_metadata is None or own_raw is None:
            raise ValueError("combined targets require own metadata and raw crops")
        require_sha256(own_metadata, own_protocol["data"]["train_source_sha256"])
        require_sha256(own_raw, own_protocol["data"]["train_crops_sha256"])
        own_image, own_vehicle, own_camera, _ = _metadata(own_metadata)
        own_crops = np.load(own_raw, mmap_mode="r", allow_pickle=False)
        own_count = len(own_image)
        if (own_crops.shape != (own_count, 208, 208, 3) or own_crops.dtype != np.uint8
                or own_count >= len(image)):
            raise ValueError("own prefix shape invalid for combined input")
        if (not np.array_equal(image[:own_count], own_image)
                or not np.array_equal(vehicle[:own_count], own_vehicle)
                or not np.array_equal(camera[:own_count], own_camera)
                or source is None or not np.all(source[:own_count] == "own")
                or np.any(source[own_count:] == "own")
                or np.any(vehicle[own_count:] < 100000)):
            raise ValueError("combined metadata prefix differs from own rows")
        for begin in range(0, own_count, 64):
            end = min(begin + 64, own_count)
            if not np.array_equal(crops[begin:end], own_crops[begin:end]):
                raise ValueError(f"combined raw prefix differs from own crops at {begin}")
    else:
        own_count = len(image)
        own_image, own_vehicle, own_camera = image, vehicle, camera
        if source is not None and not np.all(source == "own"):
            raise ValueError("own metadata contains foreign source rows")
    _assert_csv_order(csv_file, own_image, own_vehicle, own_camera)

    fit_own = _indices(split, "fit_indices", own_count)
    dev = _indices(split, "dev_indices", own_count)
    if (len(fit_own) != len(split["fit_indices"])
            or len(dev) != data["dev_rows_expected"]
            or len(np.unique(own_vehicle[dev])) != data["dev_ids_expected"]):
        raise ValueError("frozen fit/dev counts changed")
    if np.intersect1d(fit_own, dev).size or not np.array_equal(
            np.sort(np.concatenate((fit_own, dev))), np.arange(own_count)):
        raise ValueError("fit/dev rows do not partition own data")
    if (not np.array_equal(np.unique(own_vehicle[fit_own]),
                           np.sort(np.asarray(split["fit_ids"], dtype=np.int64)))
            or not np.array_equal(np.unique(own_vehicle[dev]),
                                   np.sort(np.asarray(split["dev_ids"], dtype=np.int64)))
            or np.intersect1d(own_vehicle[fit_own], own_vehicle[dev]).size):
        raise ValueError("fit and dev vehicle ID partition changed")
    if combined:
        indices = np.sort(np.concatenate((fit_own, np.arange(own_count, len(image)))))
    else:
        indices = np.sort(fit_own)
    if len(indices) != data["fit_rows_expected"]:
        raise ValueError("fit row count differs from frozen protocol")
    if np.intersect1d(vehicle[indices], own_vehicle[dev]).size:
        raise ValueError("dev identity leaked into teacher targets")
    return DatasetRows(crops, indices, vehicle[indices], camera[indices], image[indices],
                       own_count, len(fit_own), split)


def _session(path: Path, threads: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options,
                                providers=["CPUExecutionProvider"])


def extract_branches(rows: DatasetRows, model_a: Path, model_b: Path,
                     batch_size: int, threads: int) -> tuple[np.ndarray, np.ndarray]:
    if batch_size < 1 or threads < 1:
        raise ValueError("batch_size and threads must be positive")
    session_a, session_b = _session(model_a, threads), _session(model_b, threads)
    input_a, input_b = session_a.get_inputs()[0].name, session_b.get_inputs()[0].name
    raw_a = np.empty((len(rows.row_indices), 512), dtype=np.float32)
    raw_b = np.empty_like(raw_a)
    for begin in range(0, len(rows.row_indices), batch_size):
        indices = rows.row_indices[begin:begin + batch_size]
        tensor = np.ascontiguousarray(
            rows.crops[indices].transpose(0, 3, 1, 2), dtype=np.float32)
        for session, name, output in ((session_a, input_a, raw_a),
                                      (session_b, input_b, raw_b)):
            value = np.asarray(session.run(None, {name: tensor})[0])
            if (value.dtype != np.float32 or value.shape != (len(indices), 512)
                    or not np.isfinite(value).all()):
                raise ValueError("ONNX teacher branch returned invalid features")
            output[begin:begin + len(indices)] = value
        if begin % (batch_size * 50) == 0:
            print(f"teacher rows {begin}/{len(rows.row_indices)}", flush=True)
    return raw_a, raw_b


def _whitening_digest(protocol: dict) -> str:
    source = protocol["teacher"]["whitening_source"]
    match = re.search(r"SHA256 ([0-9a-f]{64})", source)
    if not match:
        raise ValueError("protocol has no frozen lib45 SHA-256")
    return match.group(1)


def verify_teacher_compatibility(own: dict, combined: dict) -> None:
    """Keep attempt 2 targets in precisely the attempt 1 teacher coordinates."""
    if own["data"]["split_sha256"] != combined["data"]["split_sha256"]:
        raise ValueError("teacher split differs between own and combined protocols")
    for key in ("model_a_sha256", "model_b_sha256", "train_fit_csv_sha256"):
        if own["teacher"][key] != combined["teacher"][key]:
            raise ValueError(f"teacher {key} differs between attempts")
    if _whitening_digest(own) != _whitening_digest(combined):
        raise ValueError("teacher whitening source differs between attempts")
    if own["teacher"].get("geometry") != combined["teacher"].get("geometry"):
        raise ValueError("teacher geometry differs between attempts")


def _checked_own_teacher(path: Path, expected_sha256: str, rows: DatasetRows,
                         own_protocol_sha256: str) -> dict:
    require_sha256(path, expected_sha256)
    manifest_path = Path(str(path) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("mode") != "own" or manifest.get("npz_sha256") != expected_sha256
            or manifest.get("protocol_sha256") != own_protocol_sha256):
        raise ValueError("own teacher manifest does not match frozen artifact/protocol")
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != NPZ_KEYS:
            raise ValueError("own teacher NPZ schema differs")
        artifact = {key: np.array(z[key], copy=True) for key in NPZ_KEYS}
    n = rows.fit_own_count
    if (artifact["row_indices"].dtype != np.int64
            or not np.array_equal(artifact["row_indices"], rows.row_indices[:n])
            or artifact["target"].dtype != np.float32
            or artifact["target"].shape != (n, 512)
            or artifact["vehicle_id"].dtype != np.int64
            or not np.array_equal(artifact["vehicle_id"], rows.vehicle_id[:n])
            or artifact["camera_id"].dtype != np.int64
            or not np.array_equal(artifact["camera_id"], rows.camera_id[:n])
            or not np.array_equal(artifact["image_id"], rows.image_id[:n])
            or artifact["P"].dtype != np.float64 or artifact["P"].shape != (512, 512)
            or artifact["m"].dtype != np.float64 or artifact["m"].shape != (512,)):
        raise ValueError("own teacher rows/geometry do not align with combined prefix")
    normalize_rows(artifact["target"])
    if np.max(np.abs(np.linalg.norm(artifact["target"], axis=1) - 1)) > 1e-5:
        raise ValueError("own teacher targets are not L2-normalized")
    if not np.isfinite(artifact["P"]).all() or not np.isfinite(artifact["m"]).all():
        raise ValueError("own teacher whitening contains NaN or infinity")
    for key in ("target", "P", "m"):
        if manifest.get("array_sha256", {}).get(key) != array_sha256(artifact[key]):
            raise ValueError(f"own teacher manifest {key} hash disagrees")
    return artifact


def create_targets(repo: Path, protocol_path: Path, protocol_sha256: str,
                   metadata: Path, raw: Path, output: Path, *, mode: str,
                   own_protocol_path: Path | None = None,
                   own_protocol_sha256: str | None = None,
                   own_metadata: Path | None = None, own_raw: Path | None = None,
                   own_targets: Path | None = None,
                   own_targets_sha256: str | None = None,
                   model_a: Path | None = None, model_b: Path | None = None,
                   batch_size: int = 16, threads: int = 2) -> dict:
    """Write a new NPZ and manifest after all hash and row checks pass."""
    if mode not in ("own", "combined"):
        raise ValueError("mode must be own or combined")
    manifest_path = Path(str(output) + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite teacher target or manifest")
    require_sha256(protocol_path, protocol_sha256)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_training":
        raise ValueError("teacher protocol was not frozen")
    if protocol["training"]["this_attempt"] != (1 if mode == "own" else 2):
        raise ValueError("teacher mode and protocol attempt disagree")
    own_protocol = None
    if mode == "combined":
        if (own_protocol_path is None or own_protocol_sha256 is None
                or own_metadata is None or own_raw is None
                or own_targets is None or own_targets_sha256 is None):
            raise ValueError("combined mode needs own protocol, inputs and teacher SHA")
        require_sha256(own_protocol_path, own_protocol_sha256)
        own_protocol = json.loads(own_protocol_path.read_text(encoding="utf-8"))
        if own_protocol.get("status") != "frozen_before_training":
            raise ValueError("own teacher protocol was not frozen")
        verify_teacher_compatibility(own_protocol, protocol)
    rows = verify_dataset(repo, protocol, metadata, raw, own_protocol=own_protocol,
                          own_metadata=own_metadata, own_raw=own_raw)
    model_a = model_a or repo / MODEL_A_RELATIVE
    model_b = model_b or repo / MODEL_B_RELATIVE
    for path, expected in ((model_a, protocol["teacher"]["model_a_sha256"]),
                           (model_b, protocol["teacher"]["model_b_sha256"])):
        require_sha256(path, expected)
    lib_path = repo / LIB_RELATIVE
    lib_sha = _whitening_digest(protocol)
    require_sha256(lib_path, lib_sha)
    if mode == "own":
        history = load_historical_whitening(lib_path, lib_sha)
        own_artifact = None
    else:
        own_artifact = _checked_own_teacher(own_targets, own_targets_sha256, rows,
                                             own_protocol_sha256)

    raw_a, raw_b = extract_branches(rows, model_a, model_b, batch_size, threads)
    if raw_a.shape != raw_b.shape or raw_a.shape != (len(rows.row_indices), 512):
        raise ValueError("teacher branch dimensions differ")
    fused = normalize_rows(normalize_rows(raw_a) + normalize_rows(raw_b))
    if mode == "own":
        lw = history["learn_lw"](fused, rows.vehicle_id, rows.camera_id, 0.5)
        if (lw["P"].shape != (512, 512) or lw["m"].shape != (512,)
                or lw["n_pairs"] != rows.split.get("teacher_fit_pair_count", lw["n_pairs"])):
            raise ValueError("historical whitening shape/pair count changed")
        P, m = np.asarray(lw["P"], np.float64), np.asarray(lw["m"], np.float64)
        target64 = history["apply_lw"](fused, lw)
        whitening_stats = {key: value for key, value in lw.items() if key not in ("P", "m")}
        parity = None
    else:
        P, m = own_artifact["P"], own_artifact["m"]
        target64 = normalize_rows((fused - m) @ P.T)
        parity = float(np.max(np.abs(target64[:rows.fit_own_count].astype(np.float32)
                                      - own_artifact["target"])))
        if not math.isfinite(parity) or parity > 1e-6:
            raise ValueError(f"combined own teacher target parity failed: {parity}")
        whitening_stats = None
    target = np.asarray(target64, dtype=np.float32)
    normalize_rows(target)
    if np.max(np.abs(np.linalg.norm(target, axis=1) - 1)) > 1e-5:
        raise ValueError("final teacher target is not L2-normalized")
    artifact = {"row_indices": np.asarray(rows.row_indices, dtype=np.int64),
                "target": target,
                "vehicle_id": np.asarray(rows.vehicle_id, dtype=np.int64),
                "camera_id": np.asarray(rows.camera_id, dtype=np.int64),
                "image_id": np.asarray(rows.image_id),
                "P": P, "m": m}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez(stream, **artifact)
    result = {
        "status": "PASS", "mode": mode, "npz_sha256": sha256(output),
        "protocol_sha256": protocol_sha256,
        "row_count": len(rows.row_indices), "own_fit_rows": rows.fit_own_count,
        "vehicle_ids": len(np.unique(rows.vehicle_id)),
        "vehicle_id_values": np.unique(rows.vehicle_id).astype(int).tolist(),
        "dev_rows_excluded": protocol["data"]["dev_rows_expected"],
        "input_sha256": {
            "metadata": protocol["data"]["train_source_sha256"],
            "raw_crops": protocol["data"]["train_crops_sha256"],
            "split": protocol["data"]["split_sha256"],
            "train_fit_csv": protocol["teacher"]["train_fit_csv_sha256"],
            "model_a": protocol["teacher"]["model_a_sha256"],
            "model_b": protocol["teacher"]["model_b_sha256"],
            "lib45": lib_sha,
            "teacher_targets_py": sha256(Path(__file__)),
            **({"own_metadata": own_protocol["data"]["train_source_sha256"],
                "own_raw": own_protocol["data"]["train_crops_sha256"],
                "own_targets": own_targets_sha256,
                "own_protocol": own_protocol_sha256} if mode == "combined" else {})},
        "array_sha256": {key: array_sha256(value) for key, value in artifact.items()},
        "array_schema": {key: {"dtype": str(value.dtype), "shape": list(value.shape)}
                         for key, value in artifact.items()},
        "whitening_stats": whitening_stats,
        "max_own_target_parity": parity,
        "runtime": {"python": sys.version, "numpy": np.__version__,
                    "platform": platform.platform()},
        "limitation": "Teacher backbone and warm-start had historical dev100 selection exposure; this artifact adds no direct dev targets.",
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("own", "combined"), required=True)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--raw-crops", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-a", type=Path)
    parser.add_argument("--model-b", type=Path)
    parser.add_argument("--own-protocol", type=Path)
    parser.add_argument("--own-protocol-sha256")
    parser.add_argument("--own-metadata", type=Path)
    parser.add_argument("--own-raw", type=Path)
    parser.add_argument("--own-targets", type=Path)
    parser.add_argument("--own-targets-sha256")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    report = create_targets(args.repo.resolve(), args.protocol, args.protocol_sha256,
                            args.metadata, args.raw_crops, args.out, mode=args.mode,
                            own_protocol_path=args.own_protocol,
                            own_protocol_sha256=args.own_protocol_sha256,
                            own_metadata=args.own_metadata, own_raw=args.own_raw,
                            own_targets=args.own_targets,
                            own_targets_sha256=args.own_targets_sha256,
                            model_a=args.model_a, model_b=args.model_b,
                            batch_size=args.batch_size, threads=args.threads)
    print(json.dumps({"status": report["status"], "mode": report["mode"],
                      "rows": report["row_count"], "sha256": report["npz_sha256"]}))


if __name__ == "__main__":
    main()
