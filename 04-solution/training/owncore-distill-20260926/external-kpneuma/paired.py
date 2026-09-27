"""Re-extract both frozen models and bootstrap paired AP deltas on KPNEUMA."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OWN_RUN_SHA = "8f9904133c079cec86a6cb06a999b2939f08c80017938ec9860dcd471eb9df15"
PREFLIGHT_SHA = "b2302b8a23e7518f4f72f4361234bf9a58e7cd5f5c8e63d36c19a9c58ccc1359"
OWN_RESULTS_SHA = "ab26bee61a9e8998931d2783353011b2fad2fff7d4761c81198697794f674594"
RELEASE_FILES = {
    "04-solution/service/model/osnet_ain_x1_0_vehicle_reid.onnx": "4aaad3e5db648618b0df3d2ff21c61323985ff9e50194c3d2edd4fb87c92d91f",
    "04-solution/service/model/osnet_ain_combined_v1.onnx": "b1ba5021275b34079a1653608bdfd215fc9404306dc909852e4cdfa03402efb2",
    "04-solution/service/model/lw_ens_j48_rho0.5.npz": "eb4433ffd5e38d3751d5cb04e234090060a83be6d1720b47bcf2fa1274b3c5b5",
    "04-solution/service/app/core/model.py": "f9e035ea2c330537f858b199215e00bd94932ae7a2c78ee7634de22dca6d6d35",
    "04-solution/service/app/core/config.py": "54d85b96bc7c8672a4de282c77a644ad9dd8ac96a1cc0c7dfd7dabd8af71a767",
}


def own_runner():
    spec = importlib.util.spec_from_file_location("owncore_external_frozen", HERE / "run.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen CrossViewCore runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_summary(actual: dict, prior: dict, label: str) -> None:
    if actual["valid_queries"] != 318:
        raise AssertionError(f"{label}: skipped a valid query")
    for key in ("mAP", "Rank-1", "Rank-5", "mINP"):
        if abs(actual[key] - prior[key]) > 1e-9:
            raise AssertionError(f"{label}: rerun changed {key}: {actual[key]} != {prior[key]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path,
                        default=ROOT / "04-solution/eval/external-kpneuma-20260928/results.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite existing report: {args.out}")
    own = own_runner()
    own.require_hash(HERE / "run.py", OWN_RUN_SHA)
    own.require_hash(HERE / "preflight.json", PREFLIGHT_SHA)
    own.require_hash(HERE / "results.json", OWN_RESULTS_SHA)
    for relative, expected in RELEASE_FILES.items():
        own.require_hash(ROOT / relative, expected)
    load_part, score_summary, ci, load_crop, cosine_scores, rerank_scores = own.verified_modules()
    checked, data = own.preflight(args, load_part)
    if checked != json.loads((HERE / "preflight.json").read_text(encoding="utf-8")):
        raise ValueError("inputs differ from saved preflight")
    qrows, qids, qvlds, grows, gids, gvlds, reference = data
    own_reference = json.loads((HERE / "results.json").read_text(encoding="utf-8"))

    from app.core.model import Embedder

    started = time.perf_counter()
    baseline = Embedder(threads=1)
    baseline_q = baseline.embed_rows(args.dataset_root / "images_query", qrows, batch_size=16)
    baseline_g = baseline.embed_rows(args.dataset_root / "images_gallery", grows, batch_size=16)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.model), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    core_q = own.embed_rows(session, args.dataset_root / "images_query", qrows, load_crop)
    core_g = own.embed_rows(session, args.dataset_root / "images_gallery", grows, load_crop)

    vectors = {"release": (baseline_q, baseline_g), "core": (core_q, core_g)}
    summaries: dict[str, dict] = {"release": {}, "core": {}}
    aps: dict[str, dict] = {"release": {}, "core": {}}
    for name, (query, gallery) in vectors.items():
        for mode in ("cosine", "rerank"):
            scores = (cosine_scores(query, gallery) if mode == "cosine" else
                      rerank_scores(query, gallery, 6, 3, 0.3))
            summary, per_query_ap = score_summary(scores, qids, gids, qvlds, gvlds)
            prior = (reference if name == "release" else own_reference["ranking"])[mode]
            require_summary(summary, prior, f"{name}/{mode}")
            summaries[name][mode] = summary
            aps[name][mode] = per_query_ap

    draws = np.random.default_rng(own.SEED).integers(
        0, 318, size=(own.BOOTSTRAP_REPEATS, 318))
    deltas = {}
    for mode in ("cosine", "rerank"):
        difference = aps["core"][mode] - aps["release"][mode]
        deltas[mode] = {"core_minus_release_mAP": float(difference.mean()),
                        "paired_bootstrap_95": ci(difference, draws),
                        "core_better_queries": int((difference > 0).sum()),
                        "core_worse_queries": int((difference < 0).sum()),
                        "same_AP_queries": int((difference == 0).sum())}

    report = {
        "status": "PASS", "source_sha": own.SOURCE_SHA,
        "model_sha256": own.MODEL_SHA, "checkpoint_sha256": own.CHECKPOINT_SHA,
        "release_model_files_sha256": RELEASE_FILES,
        "dataset_archive_sha256": own.ARCHIVE_SHA,
        "ordered_labels_sha256": checked["ordered_labels_sha256"],
        "counts": {"query": 318, "gallery": 318, "valid": 318, "skipped": 0},
        "protocol": {"metric": "market full-gallery AP", "rerank": [6, 3, 0.3],
                     "bootstrap": "paired query/ID resampling", "seed": own.SEED,
                     "repeats": own.BOOTSTRAP_REPEATS},
        "source_parity": "re-extracted release and core features; all mAP/Rank-1/Rank-5/mINP match saved results within 1e-9",
        "summary": summaries, "paired_delta": deltas,
        "per_query_AP_in_filename_order": {
            name: {mode: [float(value) for value in values]
                   for mode, values in blocks.items()}
            for name, blocks in aps.items()
        },
        "elapsed_seconds": time.perf_counter() - started,
        "timing_scope": "not a speed benchmark: models ran sequentially and were not interleaved",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({"status": "PASS", "paired_delta": deltas,
                      "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
