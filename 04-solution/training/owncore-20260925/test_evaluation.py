"""Small, hand-checked contracts for the independent image-model comparison."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import evaluate as own_eval
import export as own_export
import benchmark as own_benchmark

selection_spec = importlib.util.spec_from_file_location(
    "owncore_selection", Path(__file__).with_name("dev_selection.py"))
own_select = importlib.util.module_from_spec(selection_spec)
selection_spec.loader.exec_module(own_select)


REPO = Path(__file__).resolve().parents[3]


class EvaluationContracts(unittest.TestCase):
    def test_wrong_checkpoint_hash_is_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.onnx"
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                own_eval.require_sha256(path, "0" * 64)

    def test_evaluator_sibling_scope_dependency_hash_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source = repo / "04-solution/eval"
            source.mkdir(parents=True)
            (source / "reid_metrics.py").write_bytes(b"main evaluator")
            sibling = source / "scope_metrics.py"
            sibling.write_bytes(b"ranking scopes")
            frozen = {"evaluator_sha256": hashlib.sha256(b"main evaluator").hexdigest(),
                      "evaluator_scope_dependency_sha256": hashlib.sha256(b"ranking scopes").hexdigest()}
            own_eval.require_evaluator_files(repo, frozen)
            sibling.write_bytes(b"silent scope mutation")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                own_eval.require_evaluator_files(repo, frozen)

    def test_only_hash_checked_dev_thresholds_are_used(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(json.dumps({"selection_split": "dev", "thresholds": {
                "cosine": 0.4, "KR": 0.5}}))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(own_eval.read_dev_thresholds(path, digest),
                             {"cosine": 0.4, "KR": 0.5})
            path.write_text(path.read_text().replace('"dev"', '"val"'))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                own_eval.read_dev_thresholds(path, digest)
            with self.assertRaisesRegex(ValueError, "dev"):
                own_eval.read_dev_thresholds(
                    path, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_same_camera_mate_is_excluded_from_cosine_ranking(self):
        # Query 7 has a perfect same-camera match, which must be junk. Its
        # other-camera mate ranks ahead of ID 8. Query 9 has no mate.
        query = np.array([[1., 0.], [0., -1.]], dtype=np.float32)
        gallery = np.array([[1., 0.], [.8, .6], [0., 1.]], dtype=np.float32)
        qm = [{"vehicle_id": "7", "camera_id": "a", "has_mate": "1"},
              {"vehicle_id": "9", "camera_id": "a", "has_mate": "0"}]
        gm = [{"vehicle_id": "7", "camera_id": "a"},
              {"vehicle_id": "7", "camera_id": "b"},
              {"vehicle_id": "8", "camera_id": "b"}]
        result = own_eval.score_and_summarize(REPO, query, gallery, qm, gm,
                                               mode="cosine", threshold=None)
        self.assertEqual(result["mAP"], 1.0)
        self.assertEqual(result["Rank-1"], 1.0)
        self.assertEqual(result["counts"]["known_queries"], 1)
        self.assertEqual(result["F1"], "NOT MEASURED")

    def test_fusion_keeps_both_independent_coordinate_spaces(self):
        baseline = np.array([[1., 0.]], dtype=np.float32)
        core = np.array([[0., 1.]], dtype=np.float32)
        fused = own_eval.fuse_embeddings(baseline, core, .25)
        np.testing.assert_allclose(fused, [[np.sqrt(.75), 0., 0., .5]], atol=1e-7)
        self.assertEqual(fused.shape, (1, 4))

    def test_baseline_gate_prevents_comparison_on_wrong_mAP(self):
        observed = {"cosine": {"mAP": .70}, "KR": {"mAP": .77, "Rank-1": .73}}
        with self.assertRaisesRegex(AssertionError, "baseline"):
            own_eval.require_reproduced_baseline(observed)

    def test_export_rejects_checkpoint_for_another_architecture(self):
        checkpoint = {"model_state_dict": {"weight": np.zeros(1)},
                      "num_ids": 5, "protocol_sha256": "1" * 64,
                      "model_config": {"backbone": "osnet", "descriptor_dim": 512,
                                       "input_size": 208}}
        with self.assertRaisesRegex(ValueError, "architecture"):
            own_export.checkpoint_contract(checkpoint, "1" * 64)

    def test_dev_threshold_maximizes_presence_f1(self):
        query = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
        gallery = np.array([[1., 0.], [0., -.5]], dtype=np.float32)
        qm = [{"vehicle_id": "1", "camera_id": "a", "has_mate": "1"},
              {"vehicle_id": "2", "camera_id": "a", "has_mate": "0"}]
        gm = [{"vehicle_id": "1", "camera_id": "b"},
              {"vehicle_id": "3", "camera_id": "b"}]
        scores = own_eval.score_matrix(REPO, query, gallery, "cosine")
        self.assertEqual(own_select.calibrate_threshold(REPO, scores, qm, gm), 1.0)

    def test_fusion_weight_tie_prefers_smaller_predeclared_weight(self):
        self.assertEqual(own_select.choose_weight([(0.25, .4), (0.5, .4), (0.75, .2)]), .25)

    def test_benchmark_reverses_paired_order_every_sample(self):
        self.assertEqual(own_benchmark.paired_order(0), ("baseline", "core", "fusion"))
        self.assertEqual(own_benchmark.paired_order(1), ("fusion", "core", "baseline"))

    def test_holm_adjustment_covers_all_four_declared_comparisons(self):
        self.assertEqual(own_eval.holm_adjusted([.01, .02, .3, .4]),
                         [.04, .06, .6, .6])


if __name__ == "__main__":
    unittest.main()
