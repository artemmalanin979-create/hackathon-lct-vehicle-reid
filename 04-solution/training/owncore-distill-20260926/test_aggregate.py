"""Hand-checked multiplicity and comparison-population boundaries."""
import copy
import unittest
from pathlib import Path

from aggregate import (EXPECTED_BASELINE, EXPECTED_PROTOCOL, aggregate, holm,
                       require_distinct_attempts, require_protocol,
                       require_role_protocol, verify_reported_metrics)


def report(raw_p: list[float], deltas: list[float]) -> dict:
    metrics = {"baseline": {}, "core": {}, "fusion": {}}
    index = 0
    for variant in ("core", "fusion"):
        for mode in ("cosine", "KR"):
            metrics["baseline"][mode] = {"mAP": EXPECTED_BASELINE[mode],
                                         "Rank-1": 0.6, "Rank-5": 0.8, "mINP": 0.4,
                                         "F1": 0.75, "TNR": 0.5, "threshold": 0.4}
            metrics[variant][mode] = {
                "mAP": EXPECTED_BASELINE[mode] + deltas[index],
                "Rank-1": 0.6, "Rank-5": 0.8, "mINP": 0.4,
                "F1": 0.7, "TNR": 0.6, "threshold": 0.3,
                "paired_vs_baseline": {"delta_mAP": deltas[index],
                                       "ci95": [deltas[index] - 0.02, deltas[index] + 0.02],
                                       "p_two_sided": raw_p[index], "repeats": 4000,
                                       "seed": 20260925, "queries": 832, "vehicle_ids": 284,
                                       "p_Holm_four_tests": min(1, 4 * raw_p[index])},
            }
            index += 1
    return {"status": "PASS", "metrics": metrics}


class AggregateContracts(unittest.TestCase):
    def test_holm_uses_all_eight_raw_p_not_local_four_adjustments(self):
        own = report([0.01, 0.03, 0.04, 0.9], [0.1, 0.2, -0.1, 0])
        combined = report([0.02, 0.2, 0.4, 0.7], [0.15, 0.01, -0.05, 0])
        result = aggregate(own, combined)
        values = [row["p_Holm_eight_tests"] for row in result["comparisons"]]
        self.assertEqual(result["family_size"], 8)
        self.assertEqual(result["status"], "CALCULATED")
        self.assertAlmostEqual(values[0], 0.08)  # Eight hypotheses, not four.
        self.assertAlmostEqual(values[4], 0.14)
        self.assertGreater(values[1], own["metrics"]["core"]["KR"]
                           ["paired_vs_baseline"]["p_Holm_four_tests"])

    def test_holm_hand_example_preserves_original_order(self):
        self.assertEqual(holm([0.01, 0.04, 0.03, 0.9]), [0.04, 0.09, 0.09, 0.9])
        with self.assertRaises(ValueError):
            holm([float("nan"), 0.1])

    def test_rejects_changed_baseline_and_population(self):
        own = report([0.1] * 4, [0.1] * 4)
        combined = report([0.1] * 4, [0.1] * 4)
        changed = copy.deepcopy(combined)
        changed["metrics"]["baseline"]["KR"]["mAP"] = EXPECTED_BASELINE["KR"] + 0.01
        for variant in ("core", "fusion"):
            changed["metrics"][variant]["KR"]["paired_vs_baseline"]["delta_mAP"] = 0.09
        with self.assertRaisesRegex(ValueError, "baseline changed"):
            aggregate(own, changed)
        changed = copy.deepcopy(combined)
        changed["metrics"]["core"]["KR"]["paired_vs_baseline"]["queries"] = 831
        with self.assertRaisesRegex(ValueError, "population changed"):
            aggregate(own, changed)

    def test_rejects_stale_protocol_before_aggregation(self):
        candidate = report([0.1] * 4, [0.1] * 4)
        candidate["inputs"] = {"protocol_sha256": "correct"}
        require_protocol(candidate, "correct", "own")
        with self.assertRaisesRegex(ValueError, "protocol SHA-256 mismatch"):
            require_protocol(candidate, "stale", "own")

    def test_rejects_same_wrong_baseline_in_both_attempts(self):
        own = report([0.1] * 4, [0.01] * 4)
        combined = copy.deepcopy(own)
        for value in (own, combined):
            for mode in ("cosine", "KR"):
                value["metrics"]["baseline"][mode]["mAP"] = 0.5
                for variant in ("core", "fusion"):
                    value["metrics"][variant][mode]["mAP"] = 0.51
        with self.assertRaisesRegex(ValueError, "frozen release"):
            aggregate(own, combined)

    def test_p_value_must_match_recomputed_bootstrap(self):
        fresh = report([0.04, 0.08, 0.2, 0.4], [0.01] * 4)
        reported = copy.deepcopy(fresh)
        verify_reported_metrics(reported, fresh["metrics"], "own")
        reported["metrics"]["core"]["KR"]["paired_vs_baseline"]["p_two_sided"] = 0.0005
        with self.assertRaisesRegex(ValueError, "fresh bootstrap"):
            verify_reported_metrics(reported, fresh["metrics"], "own")

    def test_refusal_and_threshold_must_match_fresh_vectors(self):
        fresh = report([0.04, 0.08, 0.2, 0.4], [0.01] * 4)
        verify_reported_metrics(copy.deepcopy(fresh), fresh["metrics"], "own")
        for key, false_value in (("F1", 1.0), ("TNR", 1.0), ("threshold", 0.9)):
            with self.subTest(key=key):
                reported = copy.deepcopy(fresh)
                reported["metrics"]["core"]["KR"][key] = false_value
                with self.assertRaisesRegex(ValueError, key):
                    verify_reported_metrics(reported, fresh["metrics"], "own")

    def test_attempt_roles_are_bound_to_two_frozen_protocol_files(self):
        folder = Path(__file__).resolve().parent
        own_path, combined_path = folder / "protocol-own.json", folder / "protocol-combined.json"
        own_sha, combined_sha = EXPECTED_PROTOCOL["own"][0], EXPECTED_PROTOCOL["combined"][0]
        self.assertEqual(require_role_protocol(own_path, own_sha, "own")["training"]["this_attempt"], 1)
        self.assertEqual(require_role_protocol(combined_path, combined_sha, "combined")["training"]["this_attempt"], 2)
        with self.assertRaisesRegex(ValueError, "predeclared role"):
            require_role_protocol(own_path, own_sha, "combined")

    def test_rejects_duplicate_attempt_artifacts(self):
        own = report([0.1] * 4, [0.01] * 4)
        own["inputs"] = {"model_sha256": "model-a", "checkpoint_sha256": "checkpoint-a"}
        with self.assertRaisesRegex(ValueError, "evaluation"):
            require_distinct_attempts(own, own, "same", "same")
        second = copy.deepcopy(own)
        with self.assertRaisesRegex(ValueError, "model"):
            require_distinct_attempts(own, second, "eval-a", "eval-b")
        second["inputs"]["model_sha256"] = "model-b"
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            require_distinct_attempts(own, second, "eval-a", "eval-b")


if __name__ == "__main__":
    unittest.main()
