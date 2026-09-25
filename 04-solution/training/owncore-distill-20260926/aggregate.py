"""Apply the predeclared Holm correction across two completed attempts.

The single-attempt evaluator prints a four-comparison correction for local
diagnostics. It must not be reported as the eight-comparison family result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
EXPECTED_BASELINE = {"cosine": 0.7316093422310448, "KR": 0.7740915539438481}
EXPECTED_PROTOCOL = {
    "own": ("2943648c367ba6594e4b6c9d779d9eff4df1ceb3e902fef5d79d05dff1739d4f", 1),
    "combined": ("648f60b5e3a2a0004bf33aab2d432eba0e751b26393ff83f66ff84f7239458ac", 2),
}


MODES = ("cosine", "KR")
VARIANTS = ("core", "fusion")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def holm(raw: list[float]) -> list[float]:
    """Return Holm-adjusted two-sided p values in original order."""
    if not raw or any(not math.isfinite(p) or p < 0 or p > 1 for p in raw):
        raise ValueError("raw p values must be finite probabilities")
    count = len(raw)
    order = sorted(range(count), key=lambda index: (raw[index], index))
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * raw[index]))
        adjusted[index] = running
    return adjusted


def comparisons(attempt: str, report: dict) -> list[dict]:
    if report.get("status") != "PASS" or not isinstance(report.get("metrics"), dict):
        raise ValueError(f"{attempt} evaluation is not a completed PASS")
    metrics = report["metrics"]
    rows: list[dict] = []
    for variant in VARIANTS:
        for mode in MODES:
            baseline = metrics["baseline"][mode]
            candidate = metrics[variant][mode]
            paired = candidate["paired_vs_baseline"]
            raw_p = float(paired["p_two_sided"])
            delta = float(candidate["mAP"]) - float(baseline["mAP"])
            if not math.isclose(delta, float(paired["delta_mAP"]), rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"{attempt}/{variant}/{mode}: reported delta disagrees with mAP")
            if int(paired["repeats"]) != 4000 or int(paired["seed"]) != 20260925:
                raise ValueError(f"{attempt}/{variant}/{mode}: bootstrap plan changed")
            if int(paired["queries"]) != 832 or int(paired["vehicle_ids"]) != 284:
                raise ValueError(f"{attempt}/{variant}/{mode}: paired evaluation population changed")
            rows.append({"attempt": attempt, "variant": variant, "mode": mode,
                         "baseline_mAP": float(baseline["mAP"]),
                         "candidate_mAP": float(candidate["mAP"]),
                         "delta_mAP": delta, "ci95": paired["ci95"],
                         "p_two_sided_raw": raw_p})
    return rows


def aggregate(own: dict, combined: dict) -> dict:
    if own is combined:
        raise ValueError("the two attempts must be distinct reports")
    rows = comparisons("own", own) + comparisons("combined", combined)
    if len(rows) != 8:
        raise AssertionError("predeclared family must contain exactly eight comparisons")
    for variant in VARIANTS:
        for mode in MODES:
            first = own["metrics"]["baseline"][mode]["mAP"]
            second = combined["metrics"]["baseline"][mode]["mAP"]
            if not math.isclose(float(first), float(second), rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"baseline changed between attempts in {mode}")
            if not math.isclose(float(first), EXPECTED_BASELINE[mode], rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"baseline differs from frozen release in {mode}")
    adjusted = holm([row["p_two_sided_raw"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["p_Holm_eight_tests"] = value
    return {"status": "CALCULATED", "family_size": 8,
            "family": "own/combined × core/fusion × cosine/KR; two-sided paired mAP deltas",
            "comparisons": rows,
            "warning": "Local p_Holm_four_tests inside each evaluation.json is not family-wide."}


def require_protocol(report: dict, expected_sha256: str, attempt: str) -> None:
    actual = report.get("inputs", {}).get("protocol_sha256")
    if actual != expected_sha256:
        raise ValueError(f"{attempt} evaluation protocol SHA-256 mismatch")


def require_role_protocol(path: Path, expected_sha256: str, attempt: str) -> dict:
    role_sha, role_attempt = EXPECTED_PROTOCOL[attempt]
    if expected_sha256 != role_sha or sha256(path) != role_sha:
        raise ValueError(f"{attempt} protocol differs from predeclared role SHA-256")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if (protocol.get("status") != "frozen_before_training" or
            protocol.get("training", {}).get("this_attempt") != role_attempt):
        raise ValueError(f"{attempt} protocol role/attempt is not frozen")
    return protocol


def require_distinct_attempts(own: dict, combined: dict,
                              own_sha256: str, combined_sha256: str) -> None:
    for key, first, second in (
        ("evaluation", own_sha256, combined_sha256),
        ("model", own.get("inputs", {}).get("model_sha256"),
         combined.get("inputs", {}).get("model_sha256")),
        ("checkpoint", own.get("inputs", {}).get("checkpoint_sha256"),
         combined.get("inputs", {}).get("checkpoint_sha256")),
    ):
        if first is None or second is None or first == second:
            raise ValueError(f"two distinct completed {key} attempts are required")


def verify_reported_metrics(report: dict, fresh: dict, attempt: str) -> None:
    """A published p value is evidence only when it matches a fresh bootstrap."""
    reported = report.get("metrics")
    if not isinstance(reported, dict):
        raise ValueError(f"{attempt} evaluation lacks metrics")
    for variant in ("baseline", *VARIANTS):
        for mode in MODES:
            old, new = reported[variant][mode], fresh[variant][mode]
            for key in ("mAP", "Rank-1", "Rank-5", "mINP"):
                if not math.isclose(float(old[key]), float(new[key]), rel_tol=0, abs_tol=1e-10):
                    raise ValueError(f"{attempt}/{variant}/{mode}: {key} disagrees with fresh vectors")
            if variant == "baseline":
                continue
            earlier, current = old["paired_vs_baseline"], new["paired_vs_baseline"]
            for key in ("delta_mAP", "p_two_sided"):
                if not math.isclose(float(earlier[key]), float(current[key]), rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"{attempt}/{variant}/{mode}: {key} disagrees with fresh bootstrap")
            if not np.allclose(earlier["ci95"], current["ci95"], atol=1e-12, rtol=0):
                raise ValueError(f"{attempt}/{variant}/{mode}: CI disagrees with fresh bootstrap")


def recompute_from_attested_vectors(repo: Path, data_dir: Path, evaluation_path: Path,
                                    report: dict, protocol_path: Path,
                                    protocol_sha256: str, selection_path: Path,
                                    *, attempt: str) -> dict:
    """Re-evaluate row-ordered, hash-bound vectors and redo paired bootstrap."""
    import sys

    previous = str(HERE.parent / "owncore-20260925")
    if previous not in sys.path:
        sys.path.insert(0, previous)
    from evaluate import (_read_csv, _validate_embeddings, compare_arrays,
                          read_dev_selection, require_sha256, validate_frozen_inputs)

    protocol = require_role_protocol(protocol_path, protocol_sha256, attempt)
    require_protocol(report, protocol_sha256, attempt)
    inputs = report.get("inputs", {})
    if report.get("status") != "PASS" or report.get("scope") != "reused val1110x750; not an untouched holdout":
        raise ValueError(f"{attempt} evaluation status/scope is not accepted")
    if inputs.get("val_manifest_sha256") != protocol["data"]["val_images_manifest_sha256"]:
        raise ValueError(f"{attempt} val manifest differs from frozen protocol")
    selection_sha = inputs.get("dev_selection_sha256")
    selection = read_dev_selection(selection_path, selection_sha)
    if (selection.get("protocol_sha256") != protocol_sha256 or
            selection.get("model_sha256") != inputs.get("model_sha256")):
        raise ValueError(f"{attempt} dev selection model/protocol differs")
    input_check = validate_frozen_inputs(repo, data_dir, protocol)
    if inputs.get("verified_files") != input_check["required_files"]:
        raise ValueError(f"{attempt} verified val file count differs")
    split = repo / "04-solution/split/files"
    qmeta, gmeta = _read_csv(split / "val_query.csv"), _read_csv(split / "val_gallery.csv")
    if [len(qmeta), len(gmeta)] != protocol["data"]["val_rows_expected"]:
        raise ValueError(f"{attempt} val query/gallery rows differ")
    total = len(qmeta) + len(gmeta)
    directory = evaluation_path.parent
    baseline_path, core_path = directory / "baseline-fresh.npy", directory / "core-fresh.npy"
    require_sha256(baseline_path, report["baseline_fresh_sha256"])
    require_sha256(core_path, report["core_fresh_sha256"])
    baseline = _validate_embeddings(np.load(baseline_path, allow_pickle=False), total)
    core = _validate_embeddings(np.load(core_path, allow_pickle=False), total)
    fresh = compare_arrays(repo, baseline, core, qmeta, gmeta, selection=selection)
    verify_reported_metrics(report, fresh, attempt)
    return {"status": "PASS", "metrics": fresh, "inputs": inputs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--own-evaluation", type=Path, required=True)
    parser.add_argument("--own-sha256", required=True)
    parser.add_argument("--combined-evaluation", type=Path, required=True)
    parser.add_argument("--combined-sha256", required=True)
    parser.add_argument("--own-protocol-sha256", required=True)
    parser.add_argument("--combined-protocol-sha256", required=True)
    parser.add_argument("--own-protocol", type=Path, required=True)
    parser.add_argument("--combined-protocol", type=Path, required=True)
    parser.add_argument("--own-selection", type=Path, required=True)
    parser.add_argument("--combined-selection", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite aggregate result")
    for name, path, expected in (("own", args.own_evaluation, args.own_sha256),
                                 ("combined", args.combined_evaluation, args.combined_sha256)):
        if sha256(path) != expected:
            raise ValueError(f"{name} evaluation SHA-256 mismatch")
    own = json.loads(args.own_evaluation.read_text(encoding="utf-8"))
    combined = json.loads(args.combined_evaluation.read_text(encoding="utf-8"))
    require_distinct_attempts(own, combined, args.own_sha256, args.combined_sha256)
    repo, data_dir = args.repo.resolve(), args.data_dir.resolve()
    fresh_own = recompute_from_attested_vectors(
        repo, data_dir, args.own_evaluation, own, args.own_protocol,
        args.own_protocol_sha256, args.own_selection, attempt="own")
    fresh_combined = recompute_from_attested_vectors(
        repo, data_dir, args.combined_evaluation, combined, args.combined_protocol,
        args.combined_protocol_sha256, args.combined_selection, attempt="combined")
    result = aggregate(fresh_own, fresh_combined)
    result["status"] = "PASS"
    result["inputs"] = {"own_evaluation_sha256": args.own_sha256,
                        "combined_evaluation_sha256": args.combined_sha256,
                        "own_protocol_sha256": args.own_protocol_sha256,
                        "combined_protocol_sha256": args.combined_protocol_sha256}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"status": result["status"], "family_size": result["family_size"]}))


if __name__ == "__main__":
    main()
