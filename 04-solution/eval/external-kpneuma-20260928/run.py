"""Frozen, external KPNEUMA retrieval check using the release model and evaluator.

The dataset stays outside Git. See PROTOCOL.md before interpreting any number.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "04-solution/eval"))
sys.path.insert(0, str(ROOT / "04-solution/service"))

import numpy as np
import onnxruntime
from PIL import Image, __version__ as pillow_version

from reid_metrics import evaluate
from app.core import config
from app.core.model import Embedder
from app.core.preprocess import BBoxRow
from app.core.ranking import cosine_scores
from app.core.rerank import rerank_scores


EXPECTED = {
    "images_query": (318, "9c2f87f21ce2db2087ce904b0e3d1066ab31e1e60886e00390b8607d6d23d7ae"),
    "images_gallery": (318, "19a071fee864e02c9f3eda15873979d35b2f469fb63c2cee04a6ef8f9d4d3ea1"),
}
ARCHIVE_SHA256 = "c55c90b1b2e87e86991d47e18a52ce39b7839f2a17a8142be90320a4b9821332"
RELEASE_SHA = "ed028e82702ee7d80904b5a49cd5cc5c23b5505c"
SEED = 20260928
BOOTSTRAP_REPEATS = 2000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_part(root: Path, part: str) -> tuple[list[BBoxRow], list[str], list[str]]:
    directory = root / part
    files = sorted(directory.glob("*.png"))
    expected_count, expected_digest = EXPECTED[part]
    if len(files) != expected_count or len(list(directory.iterdir())) != expected_count:
        raise ValueError(f"{part}: expected exactly {expected_count} PNG files")
    digest = hashlib.sha256()
    rows, ids, vlds = [], [], []
    for path in files:
        fields = path.stem.split("_")
        if len(fields) != 3 or not all(field.isdigit() for field in fields):
            raise ValueError(f"bad KPNEUMA image name: {path.name}")
        vehicle_id, vld, _frame = fields
        digest.update(path.name.encode() + b"\0" + bytes.fromhex(sha256(path)))
        with Image.open(path) as image:
            image.load()
            width, height = image.size
        if width < 1 or height < 1:
            raise ValueError(f"empty image: {path.name}")
        rows.append(BBoxRow(path.name, 0, 0, width, height))
        ids.append(vehicle_id)
        vlds.append(vld)
    if digest.hexdigest() != expected_digest:
        raise ValueError(f"{part}: content/name hash differs from frozen protocol")
    return rows, ids, vlds


def score_summary(scores: np.ndarray, qids: list[str], gids: list[str],
                  qvlds: list[str], gvlds: list[str]) -> tuple[dict, np.ndarray]:
    result = evaluate(
        scores, qids, gids, qvlds, gvlds,
        known_absent=np.zeros(len(qids), dtype=bool),
        threshold=-1e9, camera_policy="market", refusal_mode="presence",
    )
    ranking = result["ranking_full_gallery"]
    aps = np.array([row["ap"] for row in result["per_query"]], dtype=np.float64)
    return {
        "mAP": ranking["mAP"], "Rank-1": ranking["Rank-1"],
        "Rank-5": ranking["Rank-5"], "mINP": ranking["mINP"],
        "valid_queries": ranking["num_valid_queries"],
    }, aps


def ci(values: np.ndarray, sample_indices: np.ndarray) -> list[float]:
    means = values[sample_indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="directory containing images_query and images_gallery")
    parser.add_argument("--archive", type=Path, required=True,
                        help="original downloaded data.tar.gz")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"will not overwrite existing report: {args.out}")
    if sha256(args.archive) != ARCHIVE_SHA256:
        raise ValueError("archive hash differs from frozen protocol")
    if (config.RERANK_K1, config.RERANK_K2, config.RERANK_LAMBDA) != (6, 3, 0.3):
        raise ValueError("rerank parameters differ from frozen release")

    qrows, qids, qvlds = load_part(args.dataset_root, "images_query")
    grows, gids, gvlds = load_part(args.dataset_root, "images_gallery")
    if len(set(qids)) != len(qids) or len(set(gids)) != len(gids):
        raise ValueError("the frozen test requires one query and gallery item per ID")
    if set(qids) != set(gids) or any(qvlds[i] == gvlds[gids.index(pid)]
                                    for i, pid in enumerate(qids)):
        raise ValueError("each query must have a gallery mate from another VLD")

    # Oracle and negative controls validate the label/protocol interpretation.
    perfect = np.equal.outer(qids, gids).astype(np.float64)
    random = np.random.default_rng(SEED)
    noise_scores = cosine_scores(random.standard_normal((len(qids), 512)),
                                 random.standard_normal((len(gids), 512)))
    perfect_result, _ = score_summary(perfect, qids, gids, qvlds, gvlds)
    random_result, _ = score_summary(noise_scores, qids, gids, qvlds, gvlds)
    if perfect_result["mAP"] != 1.0 or perfect_result["valid_queries"] != len(qids):
        raise AssertionError("positive control failed")

    start = time.perf_counter()
    embedder = Embedder(threads=1)
    query_emb = embedder.embed_rows(args.dataset_root / "images_query", qrows, batch_size=16)
    gallery_emb = embedder.embed_rows(args.dataset_root / "images_gallery", grows, batch_size=16)
    embed_seconds = time.perf_counter() - start
    started_rank = time.perf_counter()
    cosine, cosine_ap = score_summary(cosine_scores(query_emb, gallery_emb),
                                      qids, gids, qvlds, gvlds)
    rerank, rerank_ap = score_summary(rerank_scores(query_emb, gallery_emb, 6, 3, 0.3),
                                      qids, gids, qvlds, gvlds)
    rank_seconds = time.perf_counter() - started_rank
    draws = np.random.default_rng(SEED).integers(0, len(qids),
                                                  size=(BOOTSTRAP_REPEATS, len(qids)))
    cosine["mAP_bootstrap_95"] = ci(cosine_ap, draws)
    rerank["mAP_bootstrap_95"] = ci(rerank_ap, draws)
    delta = rerank_ap - cosine_ap

    report = {
        "source_sha": RELEASE_SHA,
        "dataset": "KPNEUMA test query/gallery, virtual loop detectors",
        "dataset_archive_sha256": ARCHIVE_SHA256,
        "counts": {"query": len(qids), "gallery": len(gids), "valid_cross_vld": len(qids)},
        "protocol": {"camera_policy": "market", "query_average": "valid_queries",
                     "ranking_scope": "full_gallery", "one_positive_per_query": True,
                     "bbox": "whole pre-cropped PNG", "input_size": config.INPUT_SIZE,
                     "batch_size": 16, "onnx_threads": 1, "rerank": [6, 3, 0.3],
                     "bootstrap_repeats": BOOTSTRAP_REPEATS, "seed": SEED},
        "models": {"osnet_sha256": config.MODEL_SHA256,
                   "combined_sha256": config.MODEL2_SHA256,
                   "whitening_sha256": config.WHITENING_SHA256},
        "controls": {"perfect_labels": perfect_result, "random_vectors": random_result},
        "cosine": cosine, "rerank": rerank,
        "delta_rerank_minus_cosine": {"mAP": float(delta.mean()),
                                       "mAP_bootstrap_95": ci(delta, draws)},
        "timing_seconds": {"embed": embed_seconds, "ranking_and_metrics": rank_seconds},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pillow": pillow_version, "onnxruntime": onnxruntime.__version__,
                    "cpu_count": os.cpu_count()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"cosine": cosine, "rerank": rerank,
                      "controls": report["controls"], "timing_seconds": report["timing_seconds"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
