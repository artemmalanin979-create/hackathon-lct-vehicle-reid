#!/usr/bin/env python3
"""Strict checkpoint -> one standalone ONNX image encoder, verified on real crops."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np

from evaluate import HERE, DEFAULT_REPO, require_sha256, sha256


def checkpoint_contract(payload: object, protocol_sha256: str) -> tuple[int, dict]:
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise ValueError("checkpoint must contain a model_state_dict")
    if payload.get("protocol_sha256") != protocol_sha256:
        raise ValueError("checkpoint protocol SHA-256 mismatch")
    num_ids = payload.get("num_ids")
    if isinstance(num_ids, bool) or not isinstance(num_ids, int) or num_ids < 2:
        raise ValueError("checkpoint has invalid num_ids")
    expected_config = {"backbone": "mobilenet_v3_small", "descriptor_dim": 512,
                       "input_size": 208}
    if payload.get("model_config") != expected_config:
        raise ValueError("checkpoint model_config disagrees with frozen architecture")
    return num_ids, payload["model_state_dict"]


def _real_samples(repo: Path, raw_crops: Path, protocol: dict) -> np.ndarray:
    require_sha256(raw_crops, protocol["data"]["train_crops_sha256"])
    split = repo / protocol["data"]["split_file"]
    require_sha256(split, protocol["data"]["split_sha256"])
    indices = json.loads(split.read_text(encoding="utf-8"))["dev_query_indices"][:8]
    if len(indices) != 8:
        raise ValueError("need eight frozen real dev crops for export parity")
    crops = np.load(raw_crops, mmap_mode="r", allow_pickle=False)
    if crops.ndim != 4 or crops.shape[1:] != (208, 208, 3) or crops.dtype != np.uint8:
        raise ValueError("real crop array is not uint8 NHWC 208x208")
    if max(indices) >= len(crops):
        raise ValueError("split dev sample outside crop array")
    return np.ascontiguousarray(crops[indices].transpose(0, 3, 1, 2), dtype=np.float32)


def export_checkpoint(checkpoint: Path, checkpoint_sha256: str, protocol_path: Path,
                      protocol_sha256: str, raw_crops: Path, out: Path,
                      manifest: Path, repo: Path = DEFAULT_REPO) -> dict:
    import onnx
    import onnxruntime as ort
    import torch
    from core import CrossViewCore

    if out.exists() or manifest.exists():
        raise FileExistsError("refusing to overwrite ONNX or export manifest")
    require_sha256(protocol_path, protocol_sha256)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_training":
        raise ValueError("protocol was not frozen before training")
    require_sha256(checkpoint, checkpoint_sha256)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    num_ids, state = checkpoint_contract(payload, protocol_sha256)
    torch.set_num_threads(2)
    model = CrossViewCore(num_ids)
    model.load_state_dict(state, strict=True)
    model.eval()
    real = _real_samples(repo, raw_crops, protocol)
    with torch.inference_mode():
        torch_descriptors = model(torch.from_numpy(real)).cpu().numpy()
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="owncore-export-", dir=out.parent) as temporary:
        candidate = Path(temporary) / "model.onnx"
        torch.onnx.export(model, torch.from_numpy(real[:1]), str(candidate),
                          input_names=["images"], output_names=["descriptor"],
                          dynamic_axes={"images": {0: "batch"},
                                        "descriptor": {0: "batch"}},
                          opset_version=17, dynamo=False)
        onnx.checker.check_model(onnx.load(str(candidate)))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(candidate), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        onnx_descriptors = session.run(None, {session.get_inputs()[0].name: real})[0]
        if onnx_descriptors.shape != (8, 512) or not np.isfinite(onnx_descriptors).all():
            raise ValueError("ONNX output is not finite Nx512")
        max_abs = float(np.max(np.abs(torch_descriptors - onnx_descriptors)))
        unit_error = float(np.max(np.abs(np.linalg.norm(onnx_descriptors, axis=1) - 1)))
        if max_abs > 1e-5 or unit_error > 1e-5:
            raise AssertionError(f"Torch/ONNX descriptor parity failed: max_abs={max_abs}, unit={unit_error}")
        os.replace(candidate, out)
    report = {
        "status": "PASS", "checkpoint_sha256": checkpoint_sha256,
        "protocol_sha256": protocol_sha256, "onnx_sha256": sha256(out),
        "onnx_bytes": out.stat().st_size, "checkpoint_num_ids": num_ids,
        "input": "8 frozen real dev crops; RGB float32 NCHW 0..255, 208x208",
        "train_crops_sha256": protocol["data"]["train_crops_sha256"],
        "split_sha256": protocol["data"]["split_sha256"],
        "torch_onnx_max_abs": max_abs, "onnx_unit_norm_max_error": unit_error,
        "opset": 17, "torch": torch.__version__, "onnx": onnx.__version__,
        "onnxruntime": ort.__version__, "numpy": np.__version__,
        "python": sys.version, "platform": platform.platform(),
        "scope": "real dev crop embedding parity; validation ranking/refusal parity is a separate evaluation check",
    }
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--raw-crops", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    report = export_checkpoint(args.checkpoint, args.checkpoint_sha256,
                               args.protocol, args.protocol_sha256, args.raw_crops,
                               args.out, args.manifest, args.repo.resolve())
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
