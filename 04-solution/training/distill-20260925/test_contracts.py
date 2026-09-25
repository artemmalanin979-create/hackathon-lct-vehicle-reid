"""Behavioral guards: the corresponding removal must admit a corrupt input."""
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from contracts import checked_features, check_disjoint, checked_hash, grouped_split


class InputContracts(unittest.TestCase):
    def test_row_order_positive_and_permuted(self):
        x = np.eye(2, dtype=np.float32)
        np.testing.assert_array_equal(checked_features(x, ["a", "b"], ["a", "b"]), x)
        with self.assertRaisesRegex(ValueError, "order"):
            checked_features(x, ["b", "a"], ["a", "b"])

    def test_finite_and_nonzero_vectors(self):
        x = np.eye(2, dtype=np.float32)
        checked_features(x, ["a", "b"], ["a", "b"])
        for bad in (np.array([[np.nan, 1], [1, 0]]), np.zeros((2, 2))):
            with self.assertRaises(ValueError):
                checked_features(bad, ["a", "b"], ["a", "b"])

    def test_identity_leakage_and_positive_control(self):
        check_disjoint([1, 1, 2], [3, 4], [5])
        with self.assertRaisesRegex(ValueError, "overlap"):
            check_disjoint([1, 1, 2], [2, 4], [5])

    def test_hash_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "input"
            p.write_bytes(b"abc")
            good = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
            self.assertEqual(checked_hash(p, good), good)
            p.write_bytes(b"abd")
            with self.assertRaisesRegex(ValueError, "hash"):
                checked_hash(p, good)

    def test_whole_identity_split_and_reproducibility(self):
        vids = np.repeat(np.arange(120), 3)
        fit, dev = grouped_split(vids, n_dev=10)
        self.assertEqual(len(dev), 30)
        self.assertFalse(set(vids[fit]) & set(vids[dev]))
        fit2, dev2 = grouped_split(vids, n_dev=10)
        np.testing.assert_array_equal(dev, dev2)
        np.testing.assert_array_equal(fit, fit2)


if __name__ == "__main__":
    unittest.main()
