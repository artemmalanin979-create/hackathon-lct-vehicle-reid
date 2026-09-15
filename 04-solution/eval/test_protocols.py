"""Definition-based tests of truncation, convention branches and precision."""
from decimal import Decimal, localcontext
import itertools
import unittest

import numpy as np

import reid_metrics as metrics
from oracle_checks import project
from protocol_oracle import expected_extensions
from review.oracle import diff


def evaluate(scores, qids, gids, **options):
    raw = np.asarray(scores)
    kw = dict(threshold=.5, camera_policy="market", refusal_mode="top1",
              include_rankings=True)
    kw.update(options)
    qcams = kw.pop("qcams", [0] * len(qids))
    gcams = kw.pop("gcams", [1] * len(gids))
    absent = [q not in gids for q in qids]
    result = metrics.evaluate(raw, qids, gids, qcams, gcams, absent, **kw)
    return result


def check_extensions(scores, qids, gids, qcams, gcams, **options):
    raw = np.asarray(scores).reshape(len(qids), len(gids))
    absent = [q not in gids for q in qids]
    kw = dict(threshold=.5, camera_policy="market", refusal_mode="top1")
    kw.update(options)
    expected = expected_extensions(raw.tolist(), qids, gids, qcams, gcams, absent, **kw)
    implementation_kw = dict(kw)
    if kw.get("exclude_mask") is not None:
        implementation_kw["exclude_mask"] = np.asarray(kw["exclude_mask"], dtype=bool).reshape(raw.shape)
    actual = metrics.evaluate(raw, qids, gids, qcams, gcams, absent, **implementation_kw)
    differences = diff(project(actual, expected), expected, atol=1e-12)
    if differences:
        raise AssertionError(differences[:12])
    return actual


def differential_extensions(trials=1800, seed=20260917):
    rng = np.random.default_rng(seed)
    for trial in range(trials):
        nq, ng = int(rng.integers(0, 5)), int(rng.integers(0, 10))
        gids, qids = rng.integers(0, 4, ng).tolist(), rng.integers(0, 6, nq).tolist()
        raw = rng.integers(0, 5, (nq, ng)).astype(float) / 4
        if trial % 5 == 0:
            raw = .6 + raw * 2e-8
        qcams, gcams = rng.integers(0, 3, nq).tolist(), rng.integers(0, 3, ng).tolist()
        threshold = float(raw.flat[0]) if raw.size else .5
        try:
            check_extensions(
                raw, qids, gids, qcams, gcams,
                threshold=threshold, top_k=int(rng.integers(1, 13)), rank_ks=[1, 2, 5, 12],
                ap_method=["step", "market_matlab"][trial % 2],
                score_kind=["similarity", "distance"][(trial // 2) % 2],
                camera_policy=["market", "all_same_camera"][(trial // 4) % 2],
                refusal_mode=["presence", "top1", "pairwise"][(trial // 8) % 3],
                gallery_keys=rng.permutation(ng).tolist(),
                gallery_junk=(rng.random(ng) < .15).tolist(),
                exclude_mask=(rng.random((nq, ng)) < .15).tolist(),
                query_frames=rng.integers(0, 3, nq).tolist(),
                gallery_frames=rng.integers(0, 3, ng).tolist(),
                query_average=["valid_queries", "all_queries_zero"][trial % 2],
                ap_denominator=["all_gallery_positives", "retrieved_positives"][(trial // 2) % 2],
                incomplete_inp=["undefined_if_incomplete", "zero_if_incomplete"][(trial // 4) % 2],
                rank_beyond_k=["undefined", "clamp_to_k"][(trial // 8) % 2],
                filtered_positive_policy=["drop", "as_unknown"][(trial // 16) % 2],
                truncation_order=["top_k_then_filter", "filter_then_top_k"][(trial // 32) % 2],
                include_rankings=bool(trial % 2))
        except AssertionError as exc:
            raise AssertionError(f"seed={seed}, trial={trial}: {exc}") from exc
    return dict(trials=trials, seed=seed, mismatches=0,
                scope="All new metric/count/PR/list fields, all convention branches, exact Fraction oracle")


class ProtocolTests(unittest.TestCase):
    def test_rank12_is_missing_from_default_submission(self):
        result = evaluate([list(range(12, 0, -1))], [1], [2] * 11 + [1], rank_ks=[1, 5, 12])
        full, short = result["ranking_full_gallery"], result["ranking_top_k"]
        self.assertAlmostEqual(full["mAP"], 1 / 12, places=12)
        self.assertEqual(short["mAP"], 0.)
        self.assertEqual(short["num_valid_queries"], 1)
        self.assertIsNone(short["mINP"])
        self.assertEqual(short["mINP_by_incomplete_policy"]["zero_if_incomplete"], 0.)
        self.assertEqual(result["per_query_top_k"][0]["ranking"], list(range(10)))
        self.assertIsNone(short["Rank-12"])
        self.assertEqual(short["ranks_by_beyond_k_policy"]["clamp_to_k"]["Rank-12"], 0.)
        self.assertEqual(full["Rank-12"], 1.)

    def test_ap_denominators_are_both_reported(self):
        for method, expected in [("step", .25), ("market_matlab", .125)]:
            result = evaluate([[4, 3, 2, 1]], [1], [2, 1, 2, 1], top_k=2, ap_method=method)
            short = result["ranking_top_k"]
            branches = short["mAP_by_ap_denominator_and_query_average"]
            self.assertEqual(branches["all_gallery_positives"]["valid_queries"], expected)
            self.assertEqual(branches["retrieved_positives"]["valid_queries"], expected * 2)
            chosen = evaluate([[4, 3, 2, 1]], [1], [2, 1, 2, 1], top_k=2,
                              ap_method=method, ap_denominator="retrieved_positives")
            self.assertEqual(chosen["ranking_top_k"]["mAP"], expected * 2)

    def test_missed_positive_query_stays_in_denominator(self):
        result = evaluate([[2, 1], [1, 2]], [1, 1], [1, 2], top_k=1)
        short = result["ranking_top_k"]
        self.assertEqual(short["mAP"], .5)
        self.assertEqual(short["mAP_denominator"], 2)
        self.assertEqual(short["num_incomplete_queries"], 1)
        self.assertEqual(short["num_complete_queries"], 1)
        self.assertIsNone(short["mINP"])
        self.assertEqual(short["mINP_by_incomplete_policy"]["zero_if_incomplete"], .5)

    def test_query_average_switch_and_empty_denominators(self):
        result = evaluate([[1], [1]], [1, 9], [1], query_average="all_queries_zero")
        for key in ["ranking_full_gallery", "ranking_top_k"]:
            self.assertEqual(result[key]["mAP"], .5)
            self.assertEqual(result[key]["mAP_denominator"], 2)
        self.assertEqual(result["ranking_full_gallery"]["mAP_by_query_average"],
                         {"valid_queries": 1., "all_queries_zero": .5})
        result = evaluate([[1]], [9], [1], query_average="all_queries_zero")
        self.assertEqual(result["ranking_top_k"]["mAP"], 0.)
        result = evaluate(np.empty((0, 1)), [], [1], query_average="all_queries_zero")
        self.assertIsNone(result["ranking_top_k"]["mAP"])

    def test_filtered_query_refusal_policies(self):
        for mode in ["presence", "top1", "pairwise"]:
            result = evaluate([[1]], [1], [1], gcams=[0], refusal_mode=mode,
                              filtered_positive_policy="as_unknown")
            branches = result["refusal_by_filtered_positive_policy"]
            self.assertIsNone(branches["drop"]["refusal"]["tnr"])
            self.assertEqual(branches["drop"]["counts"]["unknown_queries"], 0)
            self.assertEqual(branches["as_unknown"]["refusal"]["tnr"], 1.)
            self.assertEqual(result["counts"]["unknown_queries"], 1)
            self.assertEqual(result["per_query"][0]["status"], "filtered_positive")
            self.assertEqual(result["refusal"]["fn"], 0)
            result = evaluate([[.1, .9], [.9, .1]], [9, 1], [1, 2], gcams=[0, 1], refusal_mode=mode)
            branches = result["refusal_by_filtered_positive_policy"]
            self.assertEqual(branches["drop"]["refusal"]["tnr"], 0.)
            self.assertEqual(branches["as_unknown"]["refusal"]["tnr"], .5)

    def test_filter_order_changes_submitted_list(self):
        result = evaluate([[9, 8, 1]], [1], [1, 2, 1], gcams=[0, 1, 1], top_k=2)
        branches = result["ranking_top_k_by_filter_order"]
        self.assertEqual(branches["top_k_then_filter"]["mAP"], 0.)
        self.assertEqual(branches["filter_then_top_k"]["mAP"], .5)
        self.assertEqual(result["per_query_top_k"][0]["submitted_ranking_before_filter"], [0, 1])
        self.assertEqual(result["per_query_top_k"][0]["ranking"], [1])
        switched = evaluate([[9, 8, 1]], [1], [1, 2, 1], gcams=[0, 1, 1], top_k=2,
                            truncation_order="filter_then_top_k")
        self.assertEqual(switched["ranking_top_k"]["mAP"], .5)

    def test_top_k_ties_and_keys_at_boundary(self):
        for permutation in itertools.permutations(range(3)):
            gids, keys = [2, 1, 2], ["c", "b", "a"]
            result = evaluate([[1] * 3], [1], [gids[j] for j in permutation], top_k=1,
                              gallery_keys=[keys[j] for j in permutation])
            self.assertEqual(result["ranking_top_k"]["mAP"], 0.)
            self.assertEqual(result["ranking_full_gallery"]["mAP"], .5)

    def test_large_k_recovers_full_metrics(self):
        result = evaluate([[9, 4, 3, 2, 1]], [1], [1, 2, 1, 3, 1],
                          gcams=[0, 1, 1, 1, 1], top_k=20)
        for name in ["mAP", "Rank-1", "Rank-5", "mINP"]:
            self.assertEqual(result["ranking_top_k"][name], result["ranking_full_gallery"][name])

    def test_protocol_defaults_are_explicit(self):
        result = evaluate([[1]], [1], [1])
        expected = dict(top_k=10, query_average="valid_queries", filtered_positive_policy="drop",
                        ap_denominator="all_gallery_positives", incomplete_inp="undefined_if_incomplete",
                        rank_beyond_k="undefined", truncation_order="top_k_then_filter",
                        score_precision="float64_no_downcast", threshold_precision="float64_no_downcast",
                        tie_equality="exact_represented_value", ranking_alias="ranking_full_gallery",
                        refusal_scope="full_eligible_gallery")
        for key, value in expected.items():
            self.assertEqual(result["protocol"][key], value)
        self.assertEqual(result["ranking_full_gallery"]["scope"], "full_gallery")
        self.assertEqual(result["ranking_top_k"]["scope"], "top_k")

    def test_float32_embeddings_accumulate_in_float64(self):
        q = np.array([[1., 0.]], dtype=np.float32)
        g = np.array([[.6, .8], [.6, np.nextafter(np.float32(.8), np.float32(0))]], dtype=np.float32)
        scores = metrics.scores_from_embeddings(q, g)
        self.assertEqual(scores.dtype, np.dtype("float64"))
        with localcontext() as context:
            context.prec = 60
            expected = []
            for x, y in g:
                a, b = Decimal.from_float(float(x)), Decimal.from_float(float(y))
                expected.append(float(a / (a * a + b * b).sqrt()))
        np.testing.assert_allclose(scores[0], expected, rtol=0, atol=2e-16)
        self.assertGreater(scores[0, 1], scores[0, 0])
        self.assertEqual(scores.astype(np.float32)[0, 0], scores.astype(np.float32)[0, 1])
        self.assertEqual(evaluate(scores, [1], [1, 2])["ranking"]["mAP"], .5)

    def test_invalid_new_switches_and_ranks(self):
        invalid = [dict(top_k=v) for v in [0, -1, True, 1.5]]
        invalid += [dict(rank_ks=v) for v in [[], [0], [False], [1.5], None, "5"]]
        invalid += [{key: "automatic"} for key in ["query_average", "ap_denominator",
                    "incomplete_inp", "rank_beyond_k", "truncation_order", "filtered_positive_policy"]]
        for options in invalid:
            with self.assertRaises(ValueError):
                evaluate([[1]], [1], [1], **options)

    def test_top_k_exhaustive_fraction_oracle(self):
        for n in range(1, 6):
            for pattern in itertools.product([False, True], repeat=n):
                if not any(pattern):
                    continue
                for k in range(1, n + 2):
                    for method in ["step", "market_matlab"]:
                        check_extensions([list(range(n, 0, -1))], [1],
                                         [1 if positive else 2 for positive in pattern], [0], [1] * n,
                                         top_k=k, ap_method=method, rank_ks=[1, 2, 5, 8],
                                         include_rankings=True)

    def test_new_fields_fraction_oracle(self):
        self.assertEqual(differential_extensions(trials=144)["mismatches"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
