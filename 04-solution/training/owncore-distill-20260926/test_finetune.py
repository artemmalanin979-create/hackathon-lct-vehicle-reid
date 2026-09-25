"""Behavioral checks for the bounded teacher-distillation continuation."""
from __future__ import annotations

import hashlib
import copy
import json
from datetime import datetime, timezone
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "owncore-20260925"))

from finetune import (  # noqa: E402
    DistillDataset,
    balanced_relation_loss,
    classification_on_own,
    load_teacher_targets,
    load_warm_start,
    make_distilled_checkpoint,
    projected_total_seconds,
    verify_teacher_manifest,
    _check_source_manifest,
    require_time_remaining,
    _validate_protocol,
)
from core import CrossViewCore  # noqa: E402
from export import checkpoint_contract  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TeacherContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "targets.npz"
        self.indices = np.array([0, 2, 3], dtype=np.int64)
        self.meta = {"vehicle_id": np.array([11, 99, 12, 12], dtype=np.int64),
                     "camera_id": np.array([0, 3, 0, 1], dtype=np.int64),
                     "image_id": np.array(["a", "unused", "c", "d"])}

    def write(self, **changes):
        values = {
            "row_indices": self.indices,
            "target": np.tile(np.eye(1, 512, 0, dtype=np.float32), (3, 1)),
            "vehicle_id": self.meta["vehicle_id"][self.indices],
            "camera_id": self.meta["camera_id"][self.indices],
            "image_id": self.meta["image_id"][self.indices],
            "P": np.eye(512, dtype=np.float64),
            "m": np.zeros(512, dtype=np.float64),
        }
        values.update(changes)
        np.savez(self.path, **values)
        return _sha256(self.path)

    def test_exact_teacher_rows_are_accepted(self):
        digest = self.write()
        targets = load_teacher_targets(self.path, digest, self.indices, self.meta)
        self.assertEqual(targets.shape, (3, 512))
        self.assertEqual(targets.dtype, np.float32)
        np.testing.assert_array_equal(targets[:, 0], np.ones(3))

    def test_reordered_rows_or_identity_are_rejected(self):
        digest = self.write(row_indices=np.array([0, 3, 2], dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "row_indices"):
            load_teacher_targets(self.path, digest, self.indices, self.meta)
        digest = self.write(vehicle_id=np.array([11, 12, 777], dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "vehicle_id"):
            load_teacher_targets(self.path, digest, self.indices, self.meta)

    def test_nonunit_or_nonfinite_teacher_is_rejected(self):
        zeros = np.zeros((3, 512), dtype=np.float32)
        digest = self.write(target=zeros)
        with self.assertRaisesRegex(ValueError, "unit"):
            load_teacher_targets(self.path, digest, self.indices, self.meta)
        bad = np.tile(np.eye(1, 512, 0, dtype=np.float32), (3, 1))
        bad[2, 0] = np.nan
        digest = self.write(target=bad)
        with self.assertRaisesRegex(ValueError, "finite"):
            load_teacher_targets(self.path, digest, self.indices, self.meta)

    def test_hash_mismatch_is_rejected_before_targets_are_read(self):
        self.write()
        with self.assertRaisesRegex(ValueError, "hash"):
            load_teacher_targets(self.path, "0" * 64, self.indices, self.meta)

    def test_teacher_manifest_binds_artifact_and_release_model_hashes(self):
        digest = self.write()
        protocol_path = HERE / "protocol-own.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        manifest = self.valid_manifest(protocol, _sha256(protocol_path), digest, "own",
                                       target_path=self.path)
        manifest_path = Path(str(self.path) + ".manifest.json")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        verify_teacher_manifest(self.path, digest, _sha256(protocol_path), protocol, 3)
        for key, value in (("model_b", "0" * 64), ("raw_crops", "1" * 64),
                           ("train_fit_csv", "2" * 64)):
            corrupted = copy.deepcopy(manifest)
            corrupted["input_sha256"][key] = value
            manifest_path.write_text(json.dumps(corrupted), encoding="utf-8")
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                verify_teacher_manifest(self.path, digest, _sha256(protocol_path), protocol, 3)
        corrupted = copy.deepcopy(manifest)
        corrupted["protocol_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(corrupted), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "protocol_sha256"):
            verify_teacher_manifest(self.path, digest, _sha256(protocol_path), protocol, 3)
        corrupted = copy.deepcopy(manifest)
        corrupted["array_sha256"]["target"] = "0" * 64
        manifest_path.write_text(json.dumps(corrupted), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "target"):
            verify_teacher_manifest(self.path, digest, _sha256(protocol_path), protocol, 3)

    def test_combined_manifest_requires_own_target_and_own_protocol_hashes(self):
        digest = self.write()
        protocol_path = HERE / "protocol-combined.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        own_sha = "b" * 64
        manifest = self.valid_manifest(protocol, _sha256(protocol_path), digest,
                                       "combined", own_sha=own_sha,
                                       target_path=self.path)
        path = Path(str(self.path) + ".manifest.json")
        path.write_text(json.dumps(manifest), encoding="utf-8")
        verify_teacher_manifest(self.path, digest, _sha256(protocol_path), protocol, 3,
                                own_target_sha=own_sha)
        manifest["input_sha256"]["own_targets"] = "a" * 64
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "own_targets"):
            verify_teacher_manifest(self.path, digest, _sha256(protocol_path), protocol, 3,
                                    own_target_sha=own_sha)

    @staticmethod
    def valid_manifest(protocol, protocol_sha, digest, mode, own_sha=None,
                       target_path=None):
        teacher, data = protocol["teacher"], protocol["data"]
        lib_hash = teacher["whitening_source"].split("SHA256 ")[1]
        inputs = {"metadata": data["train_source_sha256"],
                  "raw_crops": data["train_crops_sha256"],
                  "split": data["split_sha256"],
                  "train_fit_csv": teacher["train_fit_csv_sha256"],
                  "model_a": teacher["model_a_sha256"],
                  "model_b": teacher["model_b_sha256"],
                  "lib45": lib_hash,
                  "teacher_targets_py": _sha256(HERE / "teacher_targets.py")}
        if mode == "combined":
            own = json.loads((HERE / "protocol-own.json").read_text(encoding="utf-8"))
            inputs.update(own_metadata=own["data"]["train_source_sha256"],
                          own_raw=own["data"]["train_crops_sha256"],
                          own_targets=own_sha,
                          own_protocol=_sha256(HERE / "protocol-own.json"))
        with np.load(target_path, allow_pickle=False) as archive:
            array_hashes = {key: hashlib.sha256(np.ascontiguousarray(archive[key]).tobytes()).hexdigest()
                            for key in ("target", "P", "m")}
        return {"status": "PASS", "mode": mode, "npz_sha256": digest,
                "protocol_sha256": protocol_sha, "row_count": 3,
                "input_sha256": inputs, "array_sha256": array_hashes}


class LossTests(unittest.TestCase):
    def test_relation_balances_cross_camera_positives_and_different_ids(self):
        student = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]], requires_grad=True)
        teacher = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        ids = torch.tensor([0, 0, 1, 1])
        cams = torch.tensor([0, 1, 0, 1])
        loss, positive, negative = balanced_relation_loss(student, teacher, ids, cams)
        self.assertAlmostEqual(float(loss.detach()), 0.75)
        self.assertEqual((positive, negative), (2, 4))
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(float(student.grad.abs().sum()), 0.0)

    def test_relation_refuses_batch_without_cross_camera_positive(self):
        z = F.normalize(torch.randn(4, 512), dim=1)
        with self.assertRaisesRegex(ValueError, "cross-camera"):
            balanced_relation_loss(z, z, torch.tensor([0, 0, 1, 1]),
                                   torch.tensor([0, 0, 1, 1]))

    def test_foreign_rows_do_not_enter_own_classifier_loss(self):
        logits = torch.tensor([[4., 0.], [0., 4.], [100., -100.]], requires_grad=True)
        labels = torch.tensor([0, 1, -1])
        actual, count = classification_on_own(logits, labels, smoothing=0.1)
        expected = F.cross_entropy(logits[:2], labels[:2], label_smoothing=0.1)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(float(actual.detach()), float(expected.detach()))
        actual.backward()
        self.assertEqual(float(logits.grad[2].abs().sum()), 0.0)

    def test_foreign_only_batch_has_zero_ce_with_valid_gradient(self):
        logits = torch.randn(3, 2, requires_grad=True)
        loss, count = classification_on_own(logits, torch.full((3,), -1), smoothing=0.1)
        self.assertEqual(count, 0)
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertEqual(float(logits.grad.abs().sum()), 0.0)


class TrainingBoundaryTests(unittest.TestCase):
    def test_combined_source_manifest_must_use_same_own_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.json"
            args = SimpleNamespace(source_training_json=path,
                                   source_checkpoint_sha256="a" * 64)
            manifest = {"status": "PASS", "checkpoint_sha256": "a" * 64,
                        "protocol_sha256": _sha256(HERE / "protocol-own.json"),
                        "input_sha256": {"teacher_targets": "b" * 64}}
            path.write_text(json.dumps(manifest), encoding="utf-8")
            hashes = {"own_teacher_targets": "b" * 64}
            _check_source_manifest(2, {}, args, hashes)
            manifest["input_sha256"]["teacher_targets"] = "c" * 64
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "teacher"):
                _check_source_manifest(2, {}, args, hashes)

    def test_deadline_recheck_catches_smoke_crossing_moscow_cutoff(self):
        cutoff = datetime(2026, 9, 26, 21, 0, tzinfo=timezone.utc)
        before = datetime(2026, 9, 26, 20, 59, 59, tzinfo=timezone.utc)
        after = datetime(2026, 9, 26, 21, 0, 1, tzinfo=timezone.utc)
        require_time_remaining(101.0, cutoff, now_monotonic=100.0, now_utc=before)
        with self.assertRaises(TimeoutError):
            require_time_remaining(101.0, cutoff, now_monotonic=100.5, now_utc=after)
        with self.assertRaises(TimeoutError):
            require_time_remaining(101.0, cutoff, now_monotonic=101.0, now_utc=before)

    def test_protocol_mutation_of_optimizer_or_loss_is_rejected(self):
        frozen = json.loads((HERE / "protocol-own.json").read_text(encoding="utf-8"))
        self.assertEqual(_validate_protocol(frozen)["this_attempt"], 1)
        for key, value in (("optimizer", "SGD"), ("scheduler", "constant"),
                           ("head_classifier_lr", 0.01)):
            changed = copy.deepcopy(frozen)
            changed["training"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "frozen"):
                _validate_protocol(changed)
        changed = copy.deepcopy(frozen)
        changed["training"]["metric_loss"]["margin"] = 1.0
        with self.assertRaisesRegex(ValueError, "frozen"):
            _validate_protocol(changed)

    def test_dataset_aligns_full_row_and_teacher_position_and_masks_foreign_ce(self):
        data = SimpleNamespace(
            crops=np.arange(4 * 8 * 8 * 3, dtype=np.uint8).reshape(4, 8, 8, 3),
            train_indices=np.array([0, 2, 3], dtype=np.int64),
            vehicle_ids=np.array([11, 999, 12, 100001], dtype=np.int64),
            camera_ids=np.array([0, 4, 1, 2], dtype=np.int64),
        )
        target = np.zeros((3, 512), dtype=np.float32)
        target[:, 0] = np.array([0.1, 0.2, 0.3])
        dataset = DistillDataset(data, target, np.array([11, 12], dtype=np.int64))
        image, cls, vid, cam, row, teacher = dataset[2]
        self.assertEqual(tuple(image.shape), (3, 8, 8))
        self.assertEqual((cls, vid, cam, row), (-1, 100001, 2, 3))
        self.assertAlmostEqual(float(teacher[0]), 0.3)
        self.assertEqual(dataset[1][1], 1)

    def test_warm_start_rejects_wrong_sha_and_incompatible_classifier(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pt"
            source = CrossViewCore(2)
            payload = {"model_state_dict": source.state_dict(), "num_ids": 2,
                       "protocol_sha256": "1" * 64,
                       "model_config": {"backbone": "mobilenet_v3_small",
                                        "descriptor_dim": 512, "input_size": 208}}
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "hash"):
                load_warm_start(path, "0" * 64, 2)
            with self.assertRaisesRegex(ValueError, "num_ids"):
                load_warm_start(path, _sha256(path), 3)
            loaded = load_warm_start(path, _sha256(path), 2)
            self.assertTrue(torch.equal(loaded.classifier.weight, source.classifier.weight))

    def test_distilled_checkpoint_loads_with_existing_onnx_export_contract(self):
        model = CrossViewCore(2)
        payload = make_distilled_checkpoint(model, protocol_sha="a" * 64,
                                            source_sha="b" * 64, teacher_sha="c" * 64,
                                            code_sha="d" * 64, best_epoch=1, best_map=0.4)
        num_ids, state = checkpoint_contract(payload, "a" * 64)
        self.assertEqual(num_ids, 2)
        CrossViewCore(num_ids).load_state_dict(state, strict=True)

    def test_pilot_projection_accounts_for_all_remaining_batches(self):
        self.assertEqual(projected_total_seconds(elapsed=100, first_epoch_seconds=200,
                         stage2_batch_seconds=0.5, steps_per_epoch=600,
                         remaining_epochs=11, safety_factor=1.25), 4225.0)


if __name__ == "__main__":
    unittest.main()
