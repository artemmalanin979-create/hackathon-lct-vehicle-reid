"""The streaming scorer may use gallery vectors and one query at a time only."""

import unittest

import numpy as np

from app.core import rerank


class IndependentRerankTests(unittest.TestCase):
    def test_other_queries_cannot_change_a_query_result(self):
        # A second query changes reciprocal neighborhoods in the legacy batch
        # algorithm. The fixed fixture therefore catches cross-query leakage.
        query = np.array([[1., 0.], [0., 1.]])
        gallery = np.array([
            [1., 0.], [.98, .2], [.5, .866], [0., 1.],
            [-.5, .866], [-1., 0.], [0., -1.],
        ])
        scorer = getattr(rerank, "rerank_scores_independent", rerank.rerank_scores)

        alone_first = scorer(query[:1], gallery, 3, 2, .3)[0]
        alone_second = scorer(query[1:], gallery, 3, 2, .3)[0]
        together = scorer(query, gallery, 3, 2, .3)
        reversed_rows = scorer(query[::-1], gallery, 3, 2, .3)

        np.testing.assert_array_equal(together[0], alone_first)
        np.testing.assert_array_equal(together[1], alone_second)
        np.testing.assert_array_equal(reversed_rows[0], alone_second)
        np.testing.assert_array_equal(reversed_rows[1], alone_first)
        self.assertGreater(alone_first[0], alone_first[-1])
        self.assertGreater(alone_second[3], alone_second[5])

    def test_empty_query_has_the_same_gallery_width(self):
        scorer = getattr(rerank, "rerank_scores_independent", rerank.rerank_scores)
        result = scorer(np.empty((0, 2)), np.eye(2), 3, 2, .3)
        self.assertEqual(result.shape, (0, 2))

    def test_gallery_outside_cosine_top50_cannot_change_reranked_top10(self):
        rng = np.random.default_rng(5)
        query = rng.normal(size=(1, 8))
        query /= np.linalg.norm(query, axis=1, keepdims=True)
        gallery = rng.normal(size=(60, 8))
        gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
        distant = -query
        cosine = (np.vstack([gallery, distant]) @ query[0])
        selected = np.argsort(-cosine, kind="stable")[:50]
        self.assertNotIn(60, selected)

        baseline = rerank.rerank_scores_independent(query, gallery, 6, 3, .3)[0]
        augmented = rerank.rerank_scores_independent(
            query, np.vstack([gallery, distant]), 6, 3, .3)[0]

        np.testing.assert_array_equal(
            np.argsort(-baseline, kind="stable")[:10],
            np.argsort(-augmented, kind="stable")[:10])
        np.testing.assert_allclose(baseline[selected], augmented[selected], rtol=0, atol=1e-12)
        self.assertGreater(np.min(baseline[selected]),
                           np.max(baseline[np.setdiff1d(np.arange(60), selected)]))


if __name__ == "__main__":
    unittest.main()
