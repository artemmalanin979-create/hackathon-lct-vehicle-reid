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
    rows = comparisons("own", own) + comparisons("combined", combined)
    if len(rows) != 8:
        raise AssertionError("predeclared family must contain exactly eight comparisons")
    for variant in VARIANTS:
        for mode in MODES:
            first = own["metrics"]["baseline"][mode]["mAP"]
            second = combined["metrics"]["baseline"][mode]["mAP"]
            if not math.isclose(float(first), float(second), rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"baseline changed between attempts in {mode}")
    adjusted = holm([row["p_two_sided_raw"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["p_Holm_eight_tests"] = value
    return {"status": "PASS", "family_size": 8,
            "family": "own/combined × core/fusion × cosine/KR; two-sided paired mAP deltas",
            "comparisons": rows,
            "warning": "Local p_Holm_four_tests inside each evaluation.json is not family-wide."}


def require_protocol(report: dict, expected_sha256: str, attempt: str) -> None:
    actual = report.get("inputs", {}).get("protocol_sha256")
    if actual != expected_sha256:
        raise ValueError(f"{attempt} evaluation protocol SHA-256 mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--own-evaluation", type=Path, required=True)
    parser.add_argument("--own-sha256", required=True)
    parser.add_argument("--combined-evaluation", type=Path, required=True)
    parser.add_argument("--combined-sha256", required=True)
    parser.add_argument("--own-protocol-sha256", required=True)
    parser.add_argument("--combined-protocol-sha256", required=True)
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
    require_protocol(own, args.own_protocol_sha256, "own")
    require_protocol(combined, args.combined_protocol_sha256, "combined")
    result = aggregate(own, combined)
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
