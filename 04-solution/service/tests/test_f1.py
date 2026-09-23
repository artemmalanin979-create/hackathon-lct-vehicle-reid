"""F1: identical frames are matches; non-finite scores cannot become refusals."""
import contextlib
import io
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.core.ranking import accepted_candidates, ranked_indices
from app.core.rerank import rerank_scores
from app.core.submission import write_candidates, write_submission


class RerankTests(unittest.TestCase):
    def test_identical_singleton_is_a_match_without_numeric_warnings(self):
        vector = np.array([[1., 0., 0.]])
        with np.errstate(all="raise"):
            scores = rerank_scores(vector, vector, 6, 3, .3)
        np.testing.assert_array_equal(scores, [[1.]])
        self.assertEqual(accepted_candidates(scores[0], .5282812306342437), [0])

    def test_identical_population_has_finite_scores(self):
        vector = np.tile([1., 0., 0.], (20, 1))
        with np.errstate(all="raise"):
            scores = rerank_scores(vector[:3], vector[3:], 6, 3, .3)
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue((scores >= .5282812306342437).all())

    def test_ordinary_and_mixed_self_match_equal_canonical_algorithm(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "postproc/scripts"))
        from common import rerank
        vectors = np.random.default_rng(93).normal(size=(17, 32)).astype(np.float32)
        for mixed in (False, True):
            with self.subTest(mixed=mixed):
                gallery = vectors[3:].copy()
                if mixed:
                    gallery[0] = vectors[0]
                expected, _ = rerank(vectors[:3], gallery, 6, 3, .3)
                actual = rerank_scores(vectors[:3], gallery, 6, 3, .3)
                np.testing.assert_array_equal(actual, 1 - expected)

    def test_nonfinite_embeddings_are_rejected(self):
        for bad in (np.nan, np.inf, -np.inf):
            for in_query in (False, True):
                q, g = np.eye(2), np.eye(2)
                (q if in_query else g)[0, 0] = bad
                with self.subTest(bad=bad, in_query=in_query):
                    with self.assertRaisesRegex(ValueError, "NaN|конечн"):
                        rerank_scores(q, g, 6, 3, .3)


class ThresholdTests(unittest.TestCase):
    def test_nonfinite_cli_threshold_never_constructs_model(self):
        from app import batch
        from app.core import model

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (16, 16), (10, 20, 30)).save(root / "valid.png")
            query, gallery, out = root / "query.csv", root / "gallery.csv", root / "out"
            for csv_path in (query, gallery):
                csv_path.write_text("image_id,x,y,w,h\nvalid,0,0,16,16\n")
            args = ["batch", "--images-dir", str(root), "--query", str(query),
                    "--gallery", str(gallery), "--out-dir", str(out)]
            for mode in ("--rerank", "--no-rerank"):
                for value in ("nan", "inf", "-inf"):
                    stderr = io.StringIO()
                    with self.subTest(mode=mode, threshold=value), \
                         patch("sys.argv", [*args, mode, f"--threshold={value}"]), \
                         patch.object(model, "Embedder", side_effect=AssertionError(
                             "invalid threshold must fail before model construction")) as embedder, \
                         contextlib.redirect_stderr(stderr):
                        with self.assertRaises(SystemExit) as exc:
                            batch.main()
                        self.assertEqual(exc.exception.code, 2)
                        embedder.assert_not_called()
                    self.assertIn("threshold", stderr.getvalue())
                    self.assertNotIn("Traceback", stderr.getvalue())
                    self.assertFalse(out.exists())

    def test_bad_score_anywhere_cannot_be_silently_dropped(self):
        for bad in (np.nan, np.inf, -np.inf):
            for row in ([bad, .8], [.8, bad]):
                with self.subTest(row=row):
                    with self.assertRaisesRegex(ValueError, "NaN|конечн"):
                        accepted_candidates(np.array(row), .5)
                    with self.assertRaisesRegex(ValueError, "NaN|конечн"):
                        ranked_indices(np.array(row))

    def test_nonfinite_threshold_is_rejected_even_without_candidates(self):
        for bad in (np.nan, np.inf, -np.inf):
            for row in (np.array([.8, .2]), np.array([])):
                with self.subTest(bad=bad, row=row):
                    with self.assertRaisesRegex(ValueError, "threshold|порог"):
                        accepted_candidates(row, bad)

    def test_threshold_equality_ties_and_legitimate_refusal_are_unchanged(self):
        scores = np.array([.8, .4, .8])
        self.assertEqual(accepted_candidates(scores, .8), [0, 2])
        self.assertEqual(accepted_candidates(scores, np.nextafter(.8, np.inf)), [])
        self.assertEqual(accepted_candidates(scores, -1.), [0, 2, 1])
        self.assertEqual(accepted_candidates(scores, 1.1), [])

    def test_writers_reject_nan_before_touching_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.csv"
            for writer in (write_submission, write_candidates):
                path.write_text("previous complete result\n")
                extra = (.5,) if writer is write_candidates else ()
                with self.subTest(writer=writer.__name__):
                    with self.assertRaisesRegex(ValueError, "NaN|конечн"):
                        writer(path, ["q0", "q1"], ["g"], np.array([[.8], [np.nan]]), *extra)
                    self.assertEqual(path.read_text(), "previous complete result\n")

    def test_writer_rejects_nan_threshold_before_touching_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.csv"
            with self.assertRaisesRegex(ValueError, "threshold|порог"):
                write_candidates(path, ["q"], ["g"], np.array([[.8]]), np.nan)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
