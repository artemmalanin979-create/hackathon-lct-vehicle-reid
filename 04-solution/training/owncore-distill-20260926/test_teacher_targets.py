"""Behavioral guards for fit-only, row-aligned teacher supervision."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from teacher_targets import (load_historical_whitening, normalize_rows,
                             verify_dataset, verify_teacher_compatibility)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TeacherDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        csv_path = self.repo / "04-solution/split/files/train_fit.csv"
        csv_path.parent.mkdir(parents=True)
        self.csv = csv_path
        self.rows = [("im0", 11, 1), ("im1", 11, 2),
                     ("im2", 12, 3), ("im3", 13, 4)]
        self.write_csv(self.rows)
        self.split_path = self.repo / "split.json"
        self.write_split([0, 1, 2], [3])
        self.own_npz = self.repo / "own.npz"
        self.own_raw = self.repo / "own.npy"
        self.write_own()
        self.own_protocol = self.protocol(self.own_npz, self.own_raw, 3)

    def write_csv(self, rows: list[tuple[str, int, int]]) -> None:
        with self.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_id", "x", "y", "w", "h", "vehicle_id", "camera_id"])
            for image_id, vehicle_id, camera_id in rows:
                writer.writerow([image_id, 0, 0, 10, 10, vehicle_id, camera_id])

    def write_split(self, fit: list[int], dev: list[int]) -> None:
        data = {"fit_indices": fit, "dev_indices": dev,
                "fit_ids": [11, 12], "dev_ids": [13],
                "dev_query_indices": dev, "dev_gallery_indices": []}
        self.split_path.write_text(json.dumps(data), encoding="utf-8")

    def write_own(self, image_ids: list[str] | None = None) -> None:
        ids = [row[0] for row in self.rows] if image_ids is None else image_ids
        np.savez(self.own_npz, image_id=np.array(ids),
                 vehicle_id=np.array([row[1] for row in self.rows], dtype=np.int64),
                 camera_id=np.array([row[2] for row in self.rows], dtype=np.int64))
        np.save(self.own_raw, np.zeros((4, 208, 208, 3), dtype=np.uint8))

    def protocol(self, metadata: Path, crops: Path, fit_count: int) -> dict:
        return {"data": {"train_source_sha256": digest(metadata),
                         "train_crops_sha256": digest(crops),
                         "split_file": "split.json", "split_sha256": digest(self.split_path),
                         "fit_rows_expected": fit_count, "dev_rows_expected": 1,
                         "dev_ids_expected": 1},
                "teacher": {"train_fit_csv_sha256": digest(self.csv),
                            "model_a_sha256": "a" * 64,
                            "model_b_sha256": "b" * 64,
                            "whitening_source": "lib45 SHA256 " + "8" * 64}}

    def test_own_fit_rows_are_ordered_and_dev_is_absent(self) -> None:
        rows = verify_dataset(self.repo, self.own_protocol, self.own_npz, self.own_raw)
        np.testing.assert_array_equal(rows.row_indices, [0, 1, 2])
        np.testing.assert_array_equal(rows.vehicle_id, [11, 11, 12])
        self.assertEqual(rows.image_id.tolist(), ["im0", "im1", "im2"])
        self.assertNotIn(13, rows.vehicle_id.tolist())

    def test_self_consistently_reordered_metadata_still_fails_csv_alignment(self) -> None:
        self.write_own(["im1", "im0", "im2", "im3"])
        protocol = self.protocol(self.own_npz, self.own_raw, 3)
        with self.assertRaisesRegex(ValueError, "CSV.*row|row.*CSV"):
            verify_dataset(self.repo, protocol, self.own_npz, self.own_raw)

    def test_self_consistently_rewritten_split_cannot_put_dev_in_fit(self) -> None:
        self.write_split([0, 1, 3], [2])
        protocol = self.protocol(self.own_npz, self.own_raw, 3)
        with self.assertRaisesRegex(ValueError, "fit.*ID|dev.*ID|identity"):
            verify_dataset(self.repo, protocol, self.own_npz, self.own_raw)

    def test_hash_mismatch_rejected_before_loading_rows(self) -> None:
        self.write_own(["tampered", "im1", "im2", "im3"])
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            verify_dataset(self.repo, self.own_protocol, self.own_npz, self.own_raw)

    def test_combined_prefix_and_all_foreign_fit_rows(self) -> None:
        combined_npz = self.repo / "combined.npz"
        combined_raw = self.repo / "combined.npy"
        with np.load(self.own_npz, allow_pickle=False) as z:
            own_ids = z["image_id"]
            own_vid = z["vehicle_id"]
            own_cam = z["camera_id"]
        np.savez(combined_npz, image_id=np.concatenate((own_ids, ["foreign"])),
                 vehicle_id=np.concatenate((own_vid, [200001])),
                 camera_id=np.concatenate((own_cam, [2000])),
                 source=np.array(["own"] * 4 + ["carla"]))
        np.save(combined_raw, np.zeros((5, 208, 208, 3), dtype=np.uint8))
        protocol = self.protocol(combined_npz, combined_raw, 4)
        rows = verify_dataset(self.repo, protocol, combined_npz, combined_raw,
                              own_protocol=self.own_protocol,
                              own_metadata=self.own_npz, own_raw=self.own_raw)
        np.testing.assert_array_equal(rows.row_indices, [0, 1, 2, 4])
        self.assertEqual(rows.own_count, 4)

    def test_combined_wrong_own_raw_prefix_rejected_even_with_valid_hashes(self) -> None:
        combined_npz = self.repo / "combined.npz"
        combined_raw = self.repo / "combined.npy"
        with np.load(self.own_npz, allow_pickle=False) as z:
            np.savez(combined_npz,
                     image_id=np.concatenate((z["image_id"], ["foreign"])),
                     vehicle_id=np.concatenate((z["vehicle_id"], [200001])),
                     camera_id=np.concatenate((z["camera_id"], [2000])),
                     source=np.array(["own"] * 4 + ["carla"]))
        wrong = np.zeros((5, 208, 208, 3), dtype=np.uint8)
        wrong[0, 0, 0, 0] = 1
        np.save(combined_raw, wrong)
        protocol = self.protocol(combined_npz, combined_raw, 4)
        with self.assertRaisesRegex(ValueError, "raw.*prefix"):
            verify_dataset(self.repo, protocol, combined_npz, combined_raw,
                           own_protocol=self.own_protocol,
                           own_metadata=self.own_npz, own_raw=self.own_raw)

    def test_combined_cannot_change_teacher_model_coordinate_system(self) -> None:
        combined = json.loads(json.dumps(self.own_protocol))
        verify_teacher_compatibility(self.own_protocol, combined)
        combined["teacher"]["model_b_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "teacher.*model_b"):
            verify_teacher_compatibility(self.own_protocol, combined)


class TeacherGeometryTests(unittest.TestCase):
    def test_normalize_rows_rejects_zero_and_nan(self) -> None:
        for bad in (np.zeros((1, 512), dtype=np.float32),
                    np.full((1, 512), np.nan, dtype=np.float32)):
            with self.subTest(bad=bad[0, 0]), self.assertRaises(ValueError):
                normalize_rows(bad)

    def test_historical_whitening_source_must_match_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "lib45.py"
            source.write_text("def learn_lw(X, vid, cam, rho):\n    return None\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_historical_whitening(source, "0" * 64)

    def test_hash_pinned_historical_whitening_uses_cross_camera_fit_pair(self) -> None:
        repo = Path(__file__).resolve().parents[3]
        source = repo / "04-solution/training/src/postprocess/lib45.py"
        functions = load_historical_whitening(
            source, "8a03c39489e463529ea5814753b6be435989fe14ea972d2848851bad09fb6983")
        vectors = np.zeros((3, 512), dtype=np.float64)
        vectors[0, 0] = vectors[1, 1] = vectors[2, 2] = 1
        matrix = functions["learn_lw"](vectors, np.array([10, 10, 20]),
                                        np.array([1, 2, 3]), 0.5)
        self.assertEqual(matrix["n_pairs"], 1)
        self.assertEqual(matrix["n_ids_with_pairs"], 1)
        np.testing.assert_allclose(matrix["m"][:3], [1 / 3] * 3)
        self.assertEqual(matrix["P"].shape, (512, 512))
        np.testing.assert_allclose(np.linalg.norm(functions["apply_lw"](vectors, matrix), axis=1),
                                   np.ones(3), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
