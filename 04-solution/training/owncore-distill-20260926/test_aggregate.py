"""Hand-checked multiplicity and comparison-population boundaries."""
import copy
import unittest

from aggregate import aggregate, holm, require_protocol


def report(raw_p: list[float], deltas: list[float]) -> dict:
    metrics = {"baseline": {}, "core": {}, "fusion": {}}
    index = 0
    for variant in ("core", "fusion"):
        for mode in ("cosine", "KR"):
            metrics["baseline"][mode] = {"mAP": 0.5}
            metrics[variant][mode] = {
                "mAP": 0.5 + deltas[index],
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
        changed["metrics"]["baseline"]["KR"]["mAP"] = 0.51
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


if __name__ == "__main__":
    unittest.main()
