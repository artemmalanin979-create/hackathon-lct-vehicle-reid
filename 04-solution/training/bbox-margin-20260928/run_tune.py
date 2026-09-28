"""Frozen ±3% bbox pilot on train-fit tune IDs; no val access or release mutation."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "04-solution/eval"))
sys.path.insert(0, str(ROOT / "04-solution/service"))

import numpy as np
import onnxruntime
from PIL import Image, __version__ as pillow_version

from reid_metrics import evaluate
from app.core import config
from app.core.model import Embedder
from app.core.preprocess import BBoxRow, crop_to_input, load_crop
from app.core.ranking import cosine_scores
from app.core.rerank import rerank_scores
from app.core.validation import validate_embeddings

DATA = Path("/home/artem/projects/hackathon-lct-vehicle-reid/data/images")
INPUTS = {
    "query": (
        "04-solution/postproc/tune/tune_query.csv",
        "17b03ff171a82a42fb3733b293cd8c1be7f111076613d2c0c8041564b201dce6",
        "798f4af4728073d6dc4937c8552b770602b7f6347ad7159b90c18ed16a3beea8",
    ),
    "gallery": (
        "04-solution/postproc/tune/tune_gallery.csv",
        "224fe2e21ddcc991b8a88a2a2e95d36d9ea3ac0e8e58439d9da2fe5175bcd518",
        "9554594bc6a39f742653e604025e8fcd05d03a72990b12612d1d59bb3de89358",
    ),
}
WEIGHTS = {
    "osnet": (config.MODEL_PATH, config.MODEL_SHA256),
    "combined": (config.MODEL2_PATH, config.MODEL2_SHA256),
    "whitening": (config.WHITENING_PATH, config.WHITENING_SHA256),
}
VARIANTS = ("original", "inward_3pct", "outward_3pct")
SEED = 20260928
N_BOOT = 2000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_meta(part: str) -> list[dict[str, str]]:
    rel, csv_sha, image_sha = INPUTS[part]
    path = ROOT / rel
    if sha256(path) != csv_sha:
        raise ValueError(f"{rel}: CSV hash mismatch")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    digest = hashlib.sha256()
    for row in rows:
        path = DATA / (row["image_id"] + ".jpg")
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(path.name.encode() + b"\0" + bytes.fromhex(sha256(path)))
        if int(row["w"]) <= 0 or int(row["h"]) <= 0:
            raise ValueError(f"nonpositive bbox in {part}: {row['image_id']}")
    if digest.hexdigest() != image_sha:
        raise ValueError(f"{part}: image hash mismatch")
    return rows


def transformed_bbox(row: dict[str, str], variant: str) -> BBoxRow:
    x, y, w, h = (int(row[key]) for key in ("x", "y", "w", "h"))
    if variant != "original":
        dx = math.floor(0.03 * w + 0.5)
        dy = math.floor(0.03 * h + 0.5)
        if variant == "inward_3pct":
            x, y, w, h = x + dx, y + dy, w - 2 * dx, h - 2 * dy
        elif variant == "outward_3pct":
            x, y, w, h = x - dx, y - dy, w + 2 * dx, h + 2 * dy
        else:
            raise ValueError(variant)
    if w <= 0 or h <= 0:
        raise ValueError(f"{variant}: nonpositive transformed bbox for {row['image_id']}")
    return BBoxRow(row["image_id"], x, y, w, h)


def extract(embedder: Embedder, rows: list[dict[str, str]], variant: str) -> tuple[np.ndarray, float]:
    out = np.full((len(rows), config.EMBEDDING_DIM), np.nan, dtype=np.float32)
    start = time.perf_counter()
    for first in range(0, len(rows), 16):
        chunk = rows[first:first + 16]
        tensors = []
        for row in chunk:
            box = transformed_bbox(row, variant)
            with Image.open(DATA / (box.image_id + ".jpg")) as image:
                image.load()
                tensor = crop_to_input(image, box.x, box.y, box.w, box.h)
            tensors.append(tensor)
        features = embedder.embed_tensors(np.stack(tensors))
        validate_embeddings(features, rows=len(chunk))
        out[first:first + len(chunk)] = features
    validate_embeddings(out, rows=len(rows))
    return out, time.perf_counter() - start


def summary(scores: np.ndarray, qm: list[dict[str, str]], gm: list[dict[str, str]]) -> tuple[dict, np.ndarray]:
    result = evaluate(
        scores,
        [r["vehicle_id"] for r in qm], [r["vehicle_id"] for r in gm],
        [r["camera_id"] for r in qm], [r["camera_id"] for r in gm],
        known_absent=np.array([r["has_mate"] == "0" for r in qm]),
        threshold=-1e9, camera_policy="market", refusal_mode="presence",
    )
    rank = result["ranking_full_gallery"]
    aps = np.array([np.nan if r["ap"] is None else r["ap"]
                    for r in result["per_query"]], dtype=np.float64)
    return {
        "mAP": rank["mAP"], "Rank-1": rank["Rank-1"],
        "Rank-5": rank["Rank-5"], "mINP": rank["mINP"],
        "valid_queries": rank["num_valid_queries"],
    }, aps


def cluster_ci(delta: np.ndarray, vehicle_ids: list[str]) -> list[float]:
    valid = np.isfinite(delta)
    all_ids = np.asarray(vehicle_ids)[valid]
    vals = delta[valid]
    groups = sorted(set(all_ids))
    group_sum = np.array([vals[all_ids == pid].sum() for pid in groups])
    group_count = np.array([(all_ids == pid).sum() for pid in groups])
    draws = np.random.default_rng(SEED).integers(0, len(groups), size=(N_BOOT, len(groups)))
    means = group_sum[draws].sum(axis=1) / group_count[draws].sum(axis=1)
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    if config.INPUT_SIZE != 208 or (config.RERANK_K1, config.RERANK_K2, config.RERANK_LAMBDA) != (6, 3, 0.3):
        raise ValueError("release input/rerank constants changed")
    for name, (path, expected) in WEIGHTS.items():
        if sha256(path) != expected:
            raise ValueError(f"{name}: weight hash mismatch")
    qm, gm = load_meta("query"), load_meta("gallery")
    qids = np.array([r["vehicle_id"] for r in qm])
    gids = np.array([r["vehicle_id"] for r in gm])
    absent = np.array([r["has_mate"] == "0" for r in qm])
    if len(qm) != 1110 or len(gm) != 732 or absent.sum() != 278 or not np.array_equal(absent, ~np.isin(qids, gids)):
        raise ValueError("tune composition changed")
    # Check the original variant against the release crop routine on a real frame.
    first = transformed_bbox(qm[0], "original")
    with Image.open(DATA / (first.image_id + ".jpg")) as image:
        if not np.array_equal(crop_to_input(image, first.x, first.y, first.w, first.h),
                              load_crop(DATA, first)):
            raise AssertionError("original crop differs from release")
    perfect_scores = np.equal.outer(qids, gids).astype(np.float64)
    random = np.random.default_rng(SEED)
    random_scores = cosine_scores(random.standard_normal((len(qm), 512)),
                                  random.standard_normal((len(gm), 512)))
    perfect, _ = summary(perfect_scores, qm, gm)
    negative, _ = summary(random_scores, qm, gm)
    if perfect["mAP"] != 1.0 or perfect["valid_queries"] != 832 or not (0 < negative["mAP"] < 0.2):
        raise AssertionError("positive/negative evaluator control failed")

    embedder = Embedder(threads=1)
    variants: dict[str, dict] = {}
    raw_aps: dict[str, np.ndarray] = {}
    start_all = time.perf_counter()
    for variant in VARIANTS:
        q, q_time = extract(embedder, qm, variant)
        g, g_time = extract(embedder, gm, variant)
        cosine, _ = summary(cosine_scores(q, g), qm, gm)
        kr, aps = summary(rerank_scores(q, g, 6, 3, 0.3), qm, gm)
        variants[variant] = {"cosine": cosine, "KR": kr,
                             "embed_seconds": q_time + g_time}
        raw_aps[variant] = aps
        print(f"{variant}: cosine={cosine['mAP']:.6f} KR={kr['mAP']:.6f}"
              f" embed={q_time + g_time:.2f}s", flush=True)

    candidates = []
    for variant in VARIANTS[1:]:
        diff = raw_aps[variant] - raw_aps["original"]
        delta = float(np.nanmean(diff))
        interval = cluster_ci(diff, qids.tolist())
        gate = delta >= 0.015 and interval[0] > 0
        variants[variant]["delta_KR_mAP"] = delta
        variants[variant]["delta_KR_mAP_cluster_bootstrap_95"] = interval
        variants[variant]["tune_gate"] = gate
        if gate:
            candidates.append(variant)
    selected = max(candidates, key=lambda v: (variants[v]["delta_KR_mAP"], v == "inward_3pct")) if candidates else None
    result = {
        "source_sha": "a0404df", "protocol_sha256": sha256(HERE / "PROTOCOL.md"),
        "script_sha256": sha256(Path(__file__)),
        "stage": "tune", "selected_for_val": selected,
        "counts": {"query": len(qm), "gallery": len(gm), "known": int((~absent).sum()),
                   "absent": int(absent.sum())},
        "controls": {"perfect_labels": perfect, "random_vectors": negative},
        "variants": variants,
        "runtime_seconds": time.perf_counter() - start_all,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pillow": pillow_version, "onnxruntime": onnxruntime.__version__,
                    "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                    "omp_threads": os.environ.get("OMP_NUM_THREADS")},
    }
    target = HERE / "tune_results.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"selected_for_val": selected,
                      "deltas": {v: variants[v].get("delta_KR_mAP") for v in VARIANTS[1:]}}))


if __name__ == "__main__":
    main()
