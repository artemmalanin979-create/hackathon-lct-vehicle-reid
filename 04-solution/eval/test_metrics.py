"""Analytic oracles plus invariants. No network, sklearn, torch or pytest."""
import itertools
import math
import unittest
from fractions import Fraction

import numpy as np

import reid_metrics as metrics


def run(scores, qids, gids, qcams=None, gcams=None, absent=None, **options):
    scores = np.asarray(scores, dtype=float)
    nq, ng = scores.shape
    defaults = dict(threshold=.5, camera_policy="market", refusal_mode="top1")
    defaults.update(options)
    return metrics.evaluate(
        scores, qids, gids, [0] * nq if qcams is None else qcams,
        [1] * ng if gcams is None else gcams,
        [False] * nq if absent is None else absent, **defaults,
    )


def ranked(pattern, **options):
    n = len(pattern)
    return run([list(range(n, 0, -1))], [1],
               [1 if x == "P" else 100 + i for i, x in enumerate(pattern)],
               absent=["P" not in pattern], **options)


def open_fixture(**options):
    return run(
        [[.9, .1, .05, 0], [0, .7, .1, .8], [0, .05, .15, .2],
         [0, .02, .03, .85], [0, .02, .03, .1]],
        ["A", "B", "C", "D", "E"], ["A", "B", "C", "X"],
        absent=[False, False, False, True, True], **options,
    )


class MetricTests(unittest.TestCase):
    def ranking_equal(self, result, ap, r1, r5, inp):
        for key, value in [("mAP", ap), ("Rank-1", r1), ("Rank-5", r5), ("mINP", inp)]:
            if value is None:
                self.assertIsNone(result["ranking"][key], key)
            else:
                self.assertAlmostEqual(result["ranking"][key], float(value), places=12, msg=key)

    def test_H01_single_perfect(self):
        self.ranking_equal(ranked("P"), 1, 1, 1, 1)

    def test_H02_single_rank2_step(self):
        self.ranking_equal(ranked("NP"), Fraction(1, 2), 0, 1, Fraction(1, 2))

    def test_H02_single_rank2_matlab(self):
        self.ranking_equal(ranked("NP", ap_method="market_matlab"), Fraction(1, 4), 0, 1, Fraction(1, 2))

    def test_H03_multiple_matches(self):
        self.ranking_equal(ranked("NPNPP"), Fraction(8, 15), 0, 1, Fraction(3, 5))
        self.ranking_equal(ranked("NPNPP", ap_method="market_matlab"), Fraction(73, 180), 0, 1, Fraction(3, 5))

    def test_H04_cmc_first_inp_last(self):
        self.ranking_equal(ranked("PNNNNP"), Fraction(2, 3), 1, 1, Fraction(1, 3))
        self.ranking_equal(ranked("PNNNNP", ap_method="market_matlab"), Fraction(19, 30), 1, 1, Fraction(1, 3))

    def test_H05_rank5_boundary(self):
        self.ranking_equal(ranked("NNNNNP"), Fraction(1, 6), 0, 0, Fraction(1, 6))
        self.ranking_equal(ranked("NNNNP"), Fraction(1, 5), 0, 1, Fraction(1, 5))

    def test_H06_remove_same_id_camera(self):
        result = run([[3, 2, 1]], [1], [1, 2, 1], gcams=[0, 1, 1], include_rankings=True)
        self.ranking_equal(result, Fraction(1, 2), 0, 1, Fraction(1, 2))
        self.assertEqual(result["per_query"][0]["ranking"], [1, 2])

    def test_H07_macro_queries_not_positives(self):
        result = run([[4, 3, 2, 1], [4, 2, 1, 3]], [10, 20], [10, 20, 20, 99])
        self.ranking_equal(result, Fraction(17, 24), Fraction(1, 2), 1, Fraction(3, 4))

    def test_query_weight_not_identity_weight(self):
        result = run([[2, 1], [2, 1], [2, 1]], [1, 1, 2], [1, 2])
        self.ranking_equal(result, Fraction(5, 6), Fraction(2, 3), 1, Fraction(5, 6))

    def test_H08_invalid_queries_excluded(self):
        result = run([[4, 2, 1], [.8, .4, .1], [.2, 3, .1]],
                     [10, 30, 20], [10, 20, 99], gcams=[1, 0, 1],
                     absent=[False, True, False])
        self.ranking_equal(result, 1, 1, 1, 1)
        self.assertEqual([q["status"] for q in result["per_query"]],
                         ["known", "unknown", "filtered_positive"])
        self.assertEqual(result["counts"]["refusal_queries"], 2)
        self.assertEqual(result["counts"]["unknown_queries"], 1)
        self.assertEqual(result["refusal"]["tnr"], 0)

    def test_H09_camera_policy_distinction(self):
        args = ([[2, 1]], [1], [2, 1])
        self.ranking_equal(run(*args, gcams=[0, 1]), Fraction(1, 2), 0, 1, Fraction(1, 2))
        self.ranking_equal(run(*args, gcams=[0, 1], camera_policy="all_same_camera"), 1, 1, 1, 1)

    def test_H10_ties_column_order(self):
        a = run([[1, 1]], [1], [1, 2])
        b = run([[1, 1]], [1], [2, 1])
        self.ranking_equal(a, 1, 1, 1, 1)
        self.ranking_equal(b, Fraction(1, 2), 0, 1, Fraction(1, 2))
        worst = run([[1] * 6], [1], [2, 3, 4, 5, 6, 1])
        self.ranking_equal(worst, Fraction(1, 6), 0, 0, Fraction(1, 6))

    def test_ties_fixed_keys_survive_permutation(self):
        a = run([[1, 1]], [1], [1, 2], gallery_keys=["b", "a"])
        b = run([[1, 1]], [1], [2, 1], gallery_keys=["a", "b"])
        self.ranking_equal(a, Fraction(1, 2), 0, 1, Fraction(1, 2))
        self.assertEqual(a["ranking"], b["ranking"])

    def test_H11_empty_gallery(self):
        result = run(np.empty((1, 0)), [1], [], absent=[True], threshold=-np.inf)
        self.ranking_equal(result, None, None, None, None)
        self.assertEqual(result["refusal"]["tnr"], 1)
        self.assertIsNone(result["refusal"]["auc_pr"])
        self.assertIsNone(result["refusal"]["precision"])
        self.assertFalse(result["per_query"][0]["accepted"])

    def test_empty_queries(self):
        result = run(np.empty((0, 2)), [], [1, 2], absent=[])
        self.ranking_equal(result, None, None, None, None)
        self.assertIsNone(result["refusal"]["tnr"])
        self.assertEqual(result["counts"]["queries"], 0)

    def test_all_positives_filtered(self):
        result = run([[1]], [1], [1], gcams=[0])
        self.ranking_equal(result, None, None, None, None)
        self.assertEqual(result["counts"]["filtered_queries"], 1)
        self.assertIsNone(result["refusal"]["tnr"])
        self.assertEqual(result["refusal"]["fn"], 0)

    def test_absent_query_survives_for_refusal(self):
        result = run([[.9]], [1], [2], absent=[True])
        self.ranking_equal(result, None, None, None, None)
        self.assertEqual(result["refusal"]["fp"], 1)
        self.assertEqual(result["refusal"]["tnr"], 0)
        self.assertIsNone(result["refusal"]["recall"])

    def test_H12_presence_confusion(self):
        ref = open_fixture(refusal_mode="presence")["refusal"]
        self.assertEqual([ref[x] for x in ["tp", "fp", "tn", "fn"]], [2, 1, 1, 1])
        for name in ["precision", "recall", "f1"]:
            self.assertAlmostEqual(ref[name], 2 / 3)
        self.assertEqual(ref["tnr"], .5)
        self.assertAlmostEqual(ref["ap_pr_step"], 29 / 36)
        self.assertAlmostEqual(ref["auc_pr"], 55 / 72)

    def test_H12_identity_confusion_and_recall_ceiling(self):
        ref = open_fixture()["refusal"]
        self.assertEqual([ref[x] for x in ["tp", "fp", "tn_unknown", "fn"]], [1, 2, 1, 2])
        for name in ["precision", "recall", "f1", "auc_pr", "ap_pr_step"]:
            self.assertAlmostEqual(ref[name], 1 / 3)
        self.assertEqual(ref["tnr"], .5)
        self.assertAlmostEqual(ref["pr_curve"]["max_recall"], 1 / 3)

    def test_H13_wrong_identity_not_tp(self):
        args = ([[.7, .9], [.8, .6]], [1, 2], [1, 2])
        self.assertEqual(run(*args, refusal_mode="presence")["refusal"]["f1"], 1)
        ref = run(*args)["refusal"]
        self.assertEqual((ref["tp"], ref["fp"], ref["fn"], ref["f1"]), (0, 2, 2, 0))
        self.assertEqual(ref["auc_pr"], 0)

    def test_H14_pr_ties_are_threshold_groups(self):
        for gids in [[1, 2], [2, 1]]:
            ref = run([[1, 1]], [1], gids, refusal_mode="pairwise")["refusal"]
            self.assertEqual(ref["ap_pr_step"], .5)
            self.assertEqual(ref["auc_pr"], .75)
            self.assertEqual(ref["pr_curve"]["thresholds"], [None, 1.0])
            self.assertEqual(ref["pr_curve"]["recall"], [0.0, 1.0])

    def test_H15_pairwise_counts(self):
        ref = run([[.9, .8, .1]], [1], [1, 2, 1], refusal_mode="pairwise")["refusal"]
        self.assertEqual([ref[x] for x in ["tp", "fp", "fn"]], [1, 1, 1])
        self.assertEqual(ref["f1"], .5)
        self.assertAlmostEqual(ref["ap_pr_step"], 5 / 6)
        self.assertAlmostEqual(ref["auc_pr"], 19 / 24)

    def test_tnr_is_per_unknown_query_not_pair(self):
        ref = run([[.9, .8], [.2, .1]], [3, 4], [1, 2],
                  absent=[True, True], refusal_mode="pairwise")["refusal"]
        self.assertEqual(ref["fp"], 2)
        self.assertEqual(ref["fp_unknown"], 1)
        self.assertEqual(ref["tn_unknown"], 1)
        self.assertEqual(ref["tnr"], .5)

    def test_threshold_inclusive(self):
        self.assertEqual(run([[.5]], [1], [1])["refusal"]["tp"], 1)
        self.assertEqual(run([[.5]], [1], [1], score_kind="distance")["refusal"]["tp"], 1)

    def test_reject_all_and_accept_all(self):
        a = open_fixture(threshold=np.inf)["refusal"]
        b = open_fixture(threshold=-np.inf)["refusal"]
        self.assertEqual((a["tp"], a["fp"], a["fn"], a["tnr"], a["f1"]), (0, 0, 3, 1, 0))
        self.assertIsNone(a["precision"])
        self.assertEqual((b["tp"], b["fp"], b["fn"], b["tnr"]), (1, 4, 2, 0))
        self.assertEqual(a["auc_pr"], b["auc_pr"])

    def test_distance_and_similarity_agree(self):
        scores = np.array([[.8, .9, .1], [.2, .3, .4]])
        a = run(scores, [1, 3], [1, 2, 1], absent=[False, True])
        b = run(-scores, [1, 3], [1, 2, 1], absent=[False, True],
                threshold=-.5, score_kind="distance")
        self.assertEqual(a["ranking"], b["ranking"])
        for key in ["tp", "fp", "fn", "f1", "tnr", "auc_pr", "ap_pr_step"]:
            self.assertEqual(a["refusal"][key], b["refusal"][key])

    def test_junk_frame_and_explicit_exclusions(self):
        result = run([[4, 3, 2, 1]], [1], [99, 98, 97, 1], gcams=[1, 0, 1, 1],
                     gallery_junk=[True, False, False, False],
                     query_frames=[7], gallery_frames=[8, 7, 8, 9],
                     exclude_mask=[[False, False, True, False]], include_rankings=True)
        self.ranking_equal(result, 1, 1, 1, 1)
        self.assertEqual(result["per_query"][0]["ranking"], [3])

    def test_frame_number_is_scoped_by_camera(self):
        result = run([[1]], [1], [1], query_frames=[7], gallery_frames=[7])
        self.ranking_equal(result, 1, 1, 1, 1)

    def test_masked_items_have_zero_influence(self):
        a = run([[2, 1]], [1], [2, 1])
        b = run([[999, 2, 1]], [1], [999, 2, 1], gallery_junk=[True, False, False])
        self.assertEqual(a["ranking"], b["ranking"])
        self.assertEqual(a["refusal"], b["refusal"])

    def test_camera_exclusion_applies_before_threshold(self):
        ref = run([[.99, .2]], [1], [1, 1], gcams=[0, 1])["refusal"]
        self.assertEqual((ref["tp"], ref["fn"]), (0, 1))

    def test_ranking_invariant_under_monotone_score_transform(self):
        scores = np.array([[.8, .9, .1], [.2, .3, .4]])
        a = run(scores, [1, 3], [1, 2, 1], absent=[False, True])
        b = run(7 * scores + 10, [1, 3], [1, 2, 1], absent=[False, True], threshold=13.5)
        self.assertEqual(a["ranking"], b["ranking"])
        for name in ["tp", "fp", "f1", "tnr", "auc_pr"]:
            self.assertEqual(a["refusal"][name], b["refusal"][name])

    def test_no_magic_junk_id(self):
        self.ranking_equal(run([[2, 1]], [1], [-1, 1]), Fraction(1, 2), 0, 1, Fraction(1, 2))
        self.ranking_equal(run([[2, 1]], [1], [-1, 1], gallery_junk=[True, False]), 1, 1, 1, 1)

    def test_validation_contradictory_flags(self):
        with self.assertRaises(ValueError):
            run([[1]], [1], [1], absent=[True])
        with self.assertRaises(ValueError):
            run([[1]], [1], [2], absent=[False])
        with self.assertRaises(ValueError):
            run([[1]], [-1], [-1], absent=[True])

    def test_validation_shapes_ids_and_nonfinite(self):
        for value in [np.nan, np.inf, -np.inf]:
            with self.assertRaises(ValueError):
                run([[value]], [1], [1])
        for options in [dict(absent=[0]), dict(gallery_keys=[1, 1]),
                        dict(gallery_junk=[1, 0]), dict(query_frames=[1]),
                        dict(camera_policy="none"), dict(refusal_mode="automatic"),
                        dict(threshold=float("nan")), dict(exclude_mask=[[False]])]:
            with self.assertRaises(ValueError):
                run([[1, 0]], [1], [1, 2], **options)
        with self.assertRaises(ValueError):
            run([[1]], [None], [1])
        with self.assertRaises(ValueError):
            run([[1]], [1, 2], [1])

    def test_embeddings(self):
        q, g = [[1, 0]], [[1, 0], [0, 1], [-1, 0]]
        np.testing.assert_allclose(metrics.scores_from_embeddings(q, g), [[1, 0, -1]])
        np.testing.assert_allclose(metrics.scores_from_embeddings(q, g, "negative_l2"),
                                   [[0, -math.sqrt(2), -2]])
        np.testing.assert_allclose(metrics.scores_from_embeddings([[1e300, 0]], [[2e300, 0]]), [[1]])
        with self.assertRaises(ValueError):
            metrics.scores_from_embeddings([[0, 0]], g)

    def test_train_test_disjoint(self):
        self.assertEqual(metrics.validate_identity_split([1, 2], [3, 4])["overlap"], 0)
        with self.assertRaises(ValueError):
            metrics.validate_identity_split([1, 2], [2, 3])

    def test_exhaustive_small_random_levels(self):
        # Combinatorial expectation is derived in ORACLES.md, independently of code.
        for n in range(1, 8):
            harmonic = sum((Fraction(1, r) for r in range(1, n + 1)), Fraction())
            for m in range(1, n + 1):
                values = []
                for positions in itertools.combinations(range(n), m):
                    pattern = ["P" if i in positions else "N" for i in range(n)]
                    values.append(ranked(pattern)["ranking"])
                expected_ap = Fraction(1) if n == 1 else harmonic / n + (m - 1) * (n - harmonic) / (n * (n - 1))
                expected_inp = sum((Fraction(m, r) * math.comb(r - 1, m - 1)
                                    for r in range(m, n + 1)), Fraction()) / math.comb(n, m)
                for key, expected in [("mAP", expected_ap), ("mINP", expected_inp),
                                      ("Rank-1", Fraction(m, n)),
                                      ("Rank-5", 1 - Fraction(math.comb(n - m, min(5, n)), math.comb(n, min(5, n))))]:
                    self.assertAlmostEqual(sum(v[key] for v in values) / len(values), float(expected), places=12)


class BlindSpotTests(unittest.TestCase):
    def test_keys_three_cycle_and_permutations(self):
        from oracle_checks import check_legacy
        for perm in itertools.permutations(range(4)):
            for kind, sign in [("similarity", 1), ("distance", -1)]:
                gids = [8, 8, 7, 7]
                keys = ["b", "d", "a", "c"]
                result = check_legacy(metrics, [[sign] * 4], [7],
                                      [gids[j] for j in perm], [0], [1] * 4, [False],
                                      gallery_keys=[keys[j] for j in perm], score_kind=kind)
                self.assertAlmostEqual(result["ranking"]["mAP"], 5 / 6, places=12)

    def test_keys_combined_with_every_filter(self):
        from oracle_checks import check_legacy
        filters = [dict(gallery_junk=[False, True, False]),
                   dict(exclude_mask=[[False, True, False]]),
                   dict(query_frames=[7], gallery_frames=[8, 7, 8])]
        for options in filters:
            for kind, sign in [("similarity", 1), ("distance", -1)]:
                check_legacy(metrics, [[5 * sign, 9 * sign, sign]], [7], [8, 7, 7],
                             [0], [1, 0 if "query_frames" in options else 1, 1], [False],
                             gallery_keys=["c", "a", "b"], score_kind=kind, **options)
        for camera in ["market", "all_same_camera"]:
            check_legacy(metrics, [[5, 9, 1]], [7], [8, 7, 7], [0], [1, 0, 1], [False],
                         gallery_keys=[30, 10, 20], camera_policy=camera)

    def test_distance_pr_curve_all_fields(self):
        from oracle_checks import check_legacy
        for mode in ["presence", "top1", "pairwise"]:
            result = check_legacy(metrics, [[1., 3.], [1., 2.], [2., 2.]],
                                  [1, 9, 2], [1, 2], [0] * 3, [1, 1],
                                  [False, True, False], threshold=2.,
                                  score_kind="distance", refusal_mode=mode)
            self.assertTrue(all(t is None or t > 0
                                for t in result["refusal"]["pr_curve"]["thresholds"]))

    def test_distance_top_score_original_units(self):
        result = run([[1., 3.]], [1], [1, 2], threshold=2., score_kind="distance")
        self.assertEqual(result["per_query"][0]["top_score"], 1.)

    def test_audit_counts_and_one_based_ranks(self):
        from oracle_checks import check_legacy
        result = check_legacy(metrics, [[9, 4, 3, 2, 1]], [1], [1, 2, 1, 3, 1],
                              [0], [0, 1, 1, 1, 1], [False])
        row = result["per_query"][0]
        self.assertEqual((row["eligible_count"], row["num_relevant"]), (4, 2))
        self.assertEqual(row["positive_ranks"], [2, 4])
        self.assertEqual(result["counts"]["relevant_pairs"], 2)

    def test_unknown_threshold_equality_all_modes(self):
        from oracle_checks import check_legacy
        for mode in ["presence", "top1", "pairwise"]:
            for kind in ["similarity", "distance"]:
                result = check_legacy(metrics, [[.5]], [9], [1], [0], [1], [True],
                                      refusal_mode=mode, score_kind=kind)
                self.assertTrue(result["per_query"][0]["accepted"])
                self.assertEqual(result["refusal"]["tnr"], 0.)
                self.assertEqual(result["refusal"]["fp_unknown"], 1)

    def test_float64_near_float32_ranking(self):
        for kind, values in [("similarity", [.6, .6 + 2e-8]),
                             ("distance", [.6 + 2e-8, .6])]:
            result = run([values], [1], [1, 2], score_kind=kind)
            self.assertEqual(result["ranking"]["mAP"], .5)
            self.assertEqual(result["ranking"]["Rank-1"], 0.)

    def test_float64_near_float32_unknown_threshold(self):
        for kind, threshold in [("similarity", .5 + 6e-9), ("distance", .5 - 6e-9)]:
            result = run([[.5]], [9], [1], absent=[True], threshold=threshold, score_kind=kind)
            self.assertFalse(result["per_query"][0]["accepted"])
            self.assertEqual(result["refusal"]["tnr"], 1.)

    def test_float32_scores_are_honest_exact_ties(self):
        values = np.array([[.6, .6 + 2e-8]], dtype=np.float32)
        self.assertEqual(values[0, 0], values[0, 1])
        result = run(values, [1], [1, 2], include_rankings=True)
        self.assertEqual(result["per_query"][0]["ranking"], [0, 1])
        self.assertEqual(result["ranking"]["mAP"], 1.)

    def test_full_output_fraction_oracle(self):
        from oracle_checks import differential_all_fields
        self.assertEqual(differential_all_fields(metrics, trials=120)["mismatches"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
