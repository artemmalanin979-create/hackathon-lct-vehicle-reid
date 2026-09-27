"""Frozen KPNEUMA retrieval check of the standalone combined CrossViewCore ONNX."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import __version__ as pillow_version


ROOT = Path(__file__).resolve().parents[4]
EXTERNAL = ROOT / "04-solution/eval/external-kpneuma-20260928"
MODEL_SHA = "4cb47b31cf2abf3eabb9fdf30f8de5bc4f06cfe2306670a01fe1bc9e95961df6"
CHECKPOINT_SHA = "30d98c3f4b8c4a99421dc4feb4f835f8e467b00f306bbe78ba3ab5b8cfd51467"
ARCHIVE_SHA = "c55c90b1b2e87e86991d47e18a52ce39b7839f2a17a8142be90320a4b9821332"
REFERENCE_SHA = "6ed7bb8d8f87daab1f9e08697fcefa8d8ec8ca3dcab22559ff5f2a59b4e2f0ab"
SOURCE_SHA = "ed028e82702ee7d80904b5a49cd5cc5c23b5505c"
SEED = 20260928
BOOTSTRAP_REPEATS = 2000
SOURCES = {
    "04-solution/service/app/core/preprocess.py": "4a6a203feb059d1626ab3a683dc505956e36c3280d746a96a0abd593238af686",
    "04-solution/service/app/core/ranking.py": "92aebe6c65dd54b9375abbb56c26cafd4e6344c8ef5c27a638dde4d10c19c6b9",
    "04-solution/service/app/core/rerank.py": "ac946451b2e2ae901e4597667b4b5bfe9a77d85d94caf2e369636cee876feb76",
    "04-solution/eval/reid_metrics.py": "79fba7051bc1fd6e856230f6a256334196c7ae7f14767ad85175aaeabe37ffec",
    "04-solution/eval/scope_metrics.py": "6dd489984a531ad2d07b58b1e963d61f015e5d4ee780187de90b424b71e463df",
    "04-solution/eval/external-kpneuma-20260928/run.py": "138558c7c4bb711ead6ddad7b74bfa4f0dcfed40ab033181ec71592a7365490a",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def verified_modules():
    for relative, expected in SOURCES.items():
        require_hash(ROOT / relative, expected)
    sys.path.insert(0, str(EXTERNAL))
    sys.path.insert(0, str(ROOT / "04-solution/service"))
    sys.path.insert(0, str(ROOT / "04-solution/eval"))
    from run import load_part, score_summary, ci
    from app.core.preprocess import load_crop
    from app.core.ranking import cosine_scores
    from app.core.rerank import rerank_scores
    return load_part, score_summary, ci, load_crop, cosine_scores, rerank_scores


def preflight(args, load_part) -> tuple[dict, tuple]:
    require_hash(args.archive, ARCHIVE_SHA)
    require_hash(args.model, MODEL_SHA)
    require_hash(args.checkpoint, CHECKPOINT_SHA)
    require_hash(args.reference_results, REFERENCE_SHA)
    qrows, qids, qvlds = load_part(args.dataset_root, "images_query")
    grows, gids, gvlds = load_part(args.dataset_root, "images_gallery")
    if len(qrows) != 318 or len(grows) != 318:
        raise ValueError("KPNEUMA query/gallery count differs from 318/318")
    if len(set(qids)) != 318 or len(set(gids)) != 318 or set(qids) != set(gids):
        raise ValueError("KPNEUMA IDs are missing or duplicated")
    mates = {pid: gvlds[index] for index, pid in enumerate(gids)}
    if any(qvld == mates[pid] for pid, qvld in zip(qids, qvlds)):
        raise ValueError("a KPNEUMA query mate uses the same VLD")
    reference = json.loads(args.reference_results.read_text(encoding="utf-8"))
    if reference.get("source_sha") != SOURCE_SHA or reference.get("counts") != {
        "query": 318, "gallery": 318, "valid_cross_vld": 318
    }:
        raise ValueError("release reference has unexpected source or dataset counts")
    label_blob = json.dumps({"query": list(zip(qids, qvlds)),
                             "gallery": list(zip(gids, gvlds))},
                            separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    result = {
        "status": "PASS", "source_sha": SOURCE_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA, "onnx_sha256": MODEL_SHA,
        "archive_sha256": ARCHIVE_SHA, "reference_results_sha256": REFERENCE_SHA,
        "query_count": 318, "gallery_count": 318,
        "unique_query_ids": 318, "unique_gallery_ids": 318,
        "valid_cross_vld": 318, "same_vld_mates": 0,
        "ordered_labels_sha256": hashlib.sha256(label_blob).hexdigest(),
        "code_sha256": SOURCES,
        "preprocess": "PIL RGB, full PNG bbox, bilinear 208x208, float32 NCHW 0..255",
    }
    return result, (qrows, qids, qvlds, grows, gids, gvlds, reference)


def embed_rows(session: ort.InferenceSession, directory: Path, rows: list,
               load_crop) -> np.ndarray:
    desc = np.empty((len(rows), 512), dtype=np.float32)
    input_name = session.get_inputs()[0].name
    for first in range(0, len(rows), 16):
        batch = np.stack([load_crop(directory, row) for row in rows[first:first + 16]])
        output = session.run(None, {input_name: batch})[0]
        if output.shape != (len(batch), 512) or not np.isfinite(output).all():
            raise ValueError(f"invalid ONNX descriptor output shape/values: {output.shape}")
        norms = np.linalg.norm(output.astype(np.float64), axis=1)
        if np.max(np.abs(norms - 1.0)) > 1e-4:
            raise ValueError("ONNX descriptors are not unit vectors")
        desc[first:first + len(batch)] = output
    return desc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, default=EXTERNAL / "results.json")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight", type=Path,
                        help="required saved preflight JSON for an inference run")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite existing report: {args.out}")
    load_part, score_summary, ci, load_crop, cosine_scores, rerank_scores = verified_modules()
    checked, data = preflight(args, load_part)
    qrows, qids, qvlds, grows, gids, gvlds, reference = data
    if args.preflight_only:
        report = checked
    else:
        if args.preflight is None:
            parser.error("inference requires --preflight from an earlier preflight-only run")
        prior = json.loads(args.preflight.read_text(encoding="utf-8"))
        if prior != checked:
            raise ValueError("saved preflight and current inputs disagree")
        perfect = np.equal.outer(qids, gids).astype(np.float64)
        rng = np.random.default_rng(SEED)
        random_scores = cosine_scores(rng.standard_normal((318, 512)),
                                      rng.standard_normal((318, 512)))
        positive, _ = score_summary(perfect, qids, gids, qvlds, gvlds)
        negative, _ = score_summary(random_scores, qids, gids, qvlds, gvlds)
        if positive["mAP"] != 1.0 or positive["valid_queries"] != 318:
            raise AssertionError("perfect-ranking positive control failed")
        if abs(negative["mAP"] - reference["controls"]["random_vectors"]["mAP"]) > 1e-12:
            raise AssertionError("fixed random negative control differs from release evaluation")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(args.model), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        model_input = session.get_inputs()[0]
        model_output = session.get_outputs()[0]
        if model_input.type != "tensor(float)" or model_input.shape[1:] != [3, 208, 208] or \
           model_output.shape[-1] != 512:
            raise ValueError("ONNX input/output contract differs from frozen 208x208 → 512D")
        start = time.perf_counter()
        query = embed_rows(session, args.dataset_root / "images_query", qrows, load_crop)
        gallery = embed_rows(session, args.dataset_root / "images_gallery", grows, load_crop)
        embed_seconds = time.perf_counter() - start
        start = time.perf_counter()
        cosine, cosine_ap = score_summary(cosine_scores(query, gallery),
                                           qids, gids, qvlds, gvlds)
        rerank, rerank_ap = score_summary(rerank_scores(query, gallery, 6, 3, 0.3),
                                           qids, gids, qvlds, gvlds)
        rank_seconds = time.perf_counter() - start
        if cosine["valid_queries"] != 318 or rerank["valid_queries"] != 318:
            raise AssertionError("an evaluation mode skipped a valid query")
        draws = np.random.default_rng(SEED).integers(0, 318,
                                                     size=(BOOTSTRAP_REPEATS, 318))
        cosine["mAP_bootstrap_95"] = ci(cosine_ap, draws)
        rerank["mAP_bootstrap_95"] = ci(rerank_ap, draws)
        delta = rerank_ap - cosine_ap
        report = {
            "status": "PASS", "preflight": checked,
            "preflight_sha256": sha256(args.preflight),
            "protocol_sha256": sha256(Path(__file__).with_name("PROTOCOL.md")),
            "model": "combined CrossViewCore, one image encoder, no teacher at inference",
            "ranking": {"cosine": cosine, "rerank": rerank},
            "controls": {"perfect_labels": positive, "random_vectors": negative},
            "rerank_minus_cosine": {"mAP": float(delta.mean()),
                                      "mAP_bootstrap_95": ci(delta, draws)},
            "release_reference": {
                "cosine_mAP": reference["cosine"]["mAP"],
                "rerank_mAP": reference["rerank"]["mAP"],
                "own_minus_release_cosine_mAP": cosine["mAP"] - reference["cosine"]["mAP"],
                "own_minus_release_rerank_mAP": rerank["mAP"] - reference["rerank"]["mAP"],
            },
            "timing_seconds": {"decode_resize_embed": embed_seconds,
                               "ranking_and_metrics": rank_seconds},
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "pillow": pillow_version, "onnxruntime": ort.__version__,
                        "cpu_count": os.cpu_count(), "onnx_threads": 1,
                        "batch_size": 16},
            "limitations": "KPNEUMA virtual detectors from aerial video; not LCT closed test. "
                           "No absent-identity cases; refusal F1/TNR not measured.",
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps(report if args.preflight_only else {
        "status": report["status"], "ranking": report["ranking"],
        "controls": report["controls"], "release_reference": report["release_reference"],
        "timing_seconds": report["timing_seconds"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
