"""The dev baseline must share the released descriptor geometry, not its teacher."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import experiment as e


class CapturedCalibration(Exception):
    pass


class SelectionGeometry(unittest.TestCase):
    def test_dev_baseline_scores_use_release_whitening(self):
        # Hand example: diagonal(2,1) maps a 45-degree vector to (2,1).
        # Its cosine to the x axis is 2/sqrt(5), not the teacher's 1/sqrt(2).
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            np.savez(out / "teacher_fit_only.npz", P=np.eye(2), m=np.zeros(2))
            np.savez(out / "release.npz", P=np.diag([2., 1.]), m=np.zeros(2))
            np.savez(out / "main.npz", dummy=np.ones(1))
            np.savez(out / "ablation.npz", dummy=np.ones(1))
            rows = e.l2(np.array([[1., 1.], [1., 0.]]))
            vectors = {"A_train_fit": rows, "B_train_fit": rows}
            data = (vectors, {}, None, None, None, None, [0], [1], [], [])
            captured = []

            def calibration(matrix, *_):
                captured.append(matrix)
                raise CapturedCalibration

            with patch.object(e, "head_numpy", return_value=rows), \
                 patch.object(e, "COSINE", side_effect=lambda q, g: q @ g.T, create=True), \
                 patch.object(e, "evaluate", return_value={"ranking_full_gallery": {"mAP": .5}}), \
                 patch.object(e, "calibrate", side_effect=calibration):
                with self.assertRaises(CapturedCalibration):
                    e.evaluate_run(SimpleNamespace(out=out),
                                   {"fusion": {"student_weights": [.25, .5, .75]}},
                                   {"whitening_release": out / "release.npz"}, data)
            self.assertEqual(len(captured), 1)
            self.assertAlmostEqual(float(captured[0][0, 0]), 2 / np.sqrt(5), places=6)
            self.assertGreater(abs(float(captured[0][0, 0]) - 1 / np.sqrt(2)), .1)


if __name__ == "__main__":
    unittest.main()
