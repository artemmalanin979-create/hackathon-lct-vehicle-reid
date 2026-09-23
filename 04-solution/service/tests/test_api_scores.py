"""The HTTP search path must not turn non-finite numbers into a refusal."""
import unittest
from unittest.mock import Mock, patch

import numpy as np
from fastapi import HTTPException

from app.api import main


class ApiScoreTests(unittest.TestCase):
    def setUp(self):
        self.store = Mock()
        self.store.reachable.return_value = True
        self.store.count.return_value = 1
        patcher = patch.dict(main.state, {"store": self.store}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_bad_score_is_a_visible_upstream_error(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad):
                self.store.search.return_value = [
                    {"gallery_id": "valid", "confidence": .9},
                    {"gallery_id": "invalid", "confidence": bad}]
                with self.assertRaises(HTTPException) as exc:
                    main._search_by_vector(np.ones(512), 10, .5)
                self.assertEqual(exc.exception.status_code, 502)
                self.assertIn("NaN", exc.exception.detail)

    def test_bad_threshold_is_rejected_before_search(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad), self.assertRaises(HTTPException) as exc:
                main._search_by_vector(np.ones(512), 10, bad)
            self.assertEqual(exc.exception.status_code, 422)
        self.store.search.assert_not_called()

    def test_finite_acceptance_and_refusal_are_unchanged(self):
        self.store.search.return_value = [{"gallery_id": "g", "confidence": .8}]
        response = main._search_by_vector(np.ones(512), 10, .8)
        self.assertFalse(response.refusal)
        self.assertEqual(response.candidates[0].gallery_id, "g")
        self.assertTrue(main._search_by_vector(np.ones(512), 10, .9).refusal)
        self.store.search.return_value = []
        self.assertTrue(main._search_by_vector(np.ones(512), 10, .5).refusal)


if __name__ == "__main__":
    unittest.main()
