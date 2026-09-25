"""Small behavioural checks for the independent image-to-descriptor core."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import CrossViewCore  # noqa: E402
from dataset import CameraAwarePKSampler, load_combined  # noqa: E402
from train import (cross_camera_batch_hard, make_checkpoint, peak_rss_bytes, projected_runtime,
                   require_rss_below, require_vram_below, select_protocol_device,
                   save_best_checkpoint, set_training_stage, smoke_one_batch,
                   train_epoch)  # noqa: E402


class CoreBehaviour(unittest.TestCase):
    def test_trainer_imports_own_core_in_isolated_python(self):
        script = Path(__file__).with_name("train.py")
        result = subprocess.run([sys.executable, "-I", str(script), "--help"],
                                capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--synthetic-smoke", result.stdout)

    def test_onnx_opset17_runs_dynamic_batch_with_torch_parity(self):
        import onnx
        import onnxruntime as ort

        model = CrossViewCore(num_ids=8).eval()
        images = torch.rand(2, 3, 208, 208, dtype=torch.float32).mul(255)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "core.onnx"
            torch.onnx.export(model, images[:1], path, opset_version=17,
                              input_names=["images"], output_names=["descriptor"],
                              dynamic_axes={"images": {0: "batch"}, "descriptor": {0: "batch"}},
                              dynamo=False)
            onnx.checker.check_model(onnx.load(path))
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            actual = session.run(None, {"images": images.numpy()})[0]
            with torch.inference_mode():
                expected = model(images).numpy()
        self.assertEqual(actual.shape, (2, 512))
        self.assertLess(float(np.max(np.abs(actual - expected))), 1e-5)
        self.assertTrue(np.allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-5))

    def test_image_to_normalized_512_and_gradients_reach_three_heads(self):
        torch.manual_seed(1)
        model = CrossViewCore(num_ids=8).train()
        x = torch.full((2, 3, 64, 64), 127.0, dtype=torch.float32)
        x[1, :, :32] = 220.0
        z = model(x)
        self.assertEqual(tuple(z.shape), (2, 512))
        self.assertTrue(torch.allclose(z.norm(dim=1), torch.ones(2), atol=1e-5))
        self.assertFalse(torch.allclose(z[0], z[1]))
        model.classify(z).square().sum().backward()
        for head in (model.global_head, model.top_head, model.bottom_head):
            self.assertIsNotNone(head.weight.grad)
            self.assertGreater(float(head.weight.grad.abs().sum()), 0.0)

    def test_stage_two_trains_backbone_from_image_gradients(self):
        model = CrossViewCore(num_ids=8).eval()
        set_training_stage(model, 1)
        self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))
        set_training_stage(model, 2)
        self.assertTrue(all(p.requires_grad for p in model.backbone.parameters()))
        model(torch.randn(2, 3, 64, 64).mul(20).add(128))[0, 0].backward()
        self.assertGreater(float(model.backbone[0][0].weight.grad.abs().sum()), 0.0)

    def test_imagenet_checkpoint_is_hash_checked_and_loads_features(self):
        from torchvision.models import mobilenet_v3_small

        source = mobilenet_v3_small(weights=None).state_dict()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mobilenet.pth"
            torch.save(source, path)
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            model = CrossViewCore(num_ids=8)
            with self.assertRaisesRegex(ValueError, "hash"):
                model.load_imagenet(path, "0" * 64)
            model.load_imagenet(path, sha)
            self.assertTrue(torch.equal(model.backbone[0][0].weight, source["features.0.0.weight"]))


class DataBehaviour(unittest.TestCase):
    @staticmethod
    def fixture(tmp: Path, bad_split: bool = False):
        vids = np.array([1, 1, 2, 2, 2, 2, 100001, 100001, 100001, 100001], dtype=np.int64)
        cams = np.array([1, 2, 1, 2, 1, 2, 1001, 1002, 1001, 1002], dtype=np.int64)
        npz, raw, split = tmp / "combined.npz", tmp / "raw.npy", tmp / "split.json"
        np.savez(npz, vehicle_id=vids, camera_id=cams)
        np.save(raw, np.zeros((len(vids), 208, 208, 3), dtype=np.uint8))
        split.write_text(json.dumps({"fit_indices": [0, 1, 2] if bad_split else [0, 1],
                                     "dev_indices": [2, 3, 4, 5],
                                     "dev_query_indices": [3, 5],
                                     "dev_gallery_indices": [2, 4],
                                     "fit_ids": [1], "dev_ids": [2]}), encoding="utf-8")
        return npz, raw, split

    def test_combined_fit_excludes_dev_and_keeps_foreign_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz, raw, split = self.fixture(Path(tmp))
            data = load_combined(npz, raw, split)
            self.assertEqual(set(data.train_vehicle_ids.tolist()), {1, 100001})
            self.assertEqual(set(data.dev_vehicle_ids.tolist()), {2})
            self.assertEqual(len(data.train_indices), 6)
            self.assertEqual(len(data.label_to_vehicle), 2)

    def test_split_containing_dev_row_in_fit_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz, raw, split = self.fixture(Path(tmp), bad_split=True)
            with self.assertRaisesRegex(ValueError, "fit|overlap|dev"):
                load_combined(npz, raw, split)

    def test_pk_sampler_gives_eight_ids_and_cross_camera_positive(self):
        vids = np.repeat(np.arange(8), 4)
        cams = np.tile([0, 0, 1, 1], 8)
        sampler = CameraAwarePKSampler(vids, cams, np.arange(32), p=8, k=4, batches=1, seed=17)
        batch = np.asarray(next(iter(sampler)))
        self.assertEqual(len(batch), 32)
        self.assertEqual(len(set(vids[batch])), 8)
        for vid in range(8):
            rows = batch[vids[batch] == vid]
            self.assertEqual(len(rows), 4)
            self.assertEqual(set(cams[rows]), {0, 1})

    def test_pk_sampler_includes_single_camera_id_for_ce_without_fake_positive(self):
        vids = np.repeat(np.arange(8), 4)
        cams = np.tile([0, 0, 1, 1], 8)
        cams[28:] = 0
        sampler = CameraAwarePKSampler(vids, cams, np.arange(32), p=8, k=4, batches=1, seed=17)
        batch = np.asarray(next(iter(sampler)))
        self.assertEqual(len(batch), 32)
        self.assertEqual(len(set(vids[batch])), 8)
        self.assertEqual(set(cams[batch][vids[batch] == 7]), {0})
        self.assertTrue(any(len(set(cams[batch][vids[batch] == v])) == 2 for v in range(7)))

    def test_pk_sampler_rejects_no_cross_camera_anchor(self):
        vids = np.repeat(np.arange(8), 4)
        cams = np.zeros(32, dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "multi.camera"):
            CameraAwarePKSampler(vids, cams, np.arange(32), p=8, k=4, batches=1, seed=17)


class LossBehaviour(unittest.TestCase):
    def test_device_selection_is_explicit_and_never_falls_back(self):
        self.assertEqual(select_protocol_device("cpu", cuda_available=True).type, "cpu")
        self.assertEqual(select_protocol_device("cuda", cuda_available=True).type, "cuda")
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            select_protocol_device("cuda", cuda_available=False)
        for absent_or_unknown in (None, "auto", "gpu"):
            with self.subTest(device=absent_or_unknown), self.assertRaisesRegex(ValueError, "device"):
                select_protocol_device(absent_or_unknown, cuda_available=True)

    def test_cpu_rss_limit_rejects_exact_boundary(self):
        mib = 1024**2
        require_rss_below(1536 * mib - 1, limit_mib=1536)
        with self.assertRaisesRegex(RuntimeError, "RSS"):
            require_rss_below(1536 * mib, limit_mib=1536)

    def test_native_peak_rss_is_observed(self):
        self.assertGreater(peak_rss_bytes(), 0)

    def test_cross_camera_batch_hard_penalizes_bad_order_and_backpropagates(self):
        z = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6], [0.8, -0.6]], requires_grad=True)
        ids = torch.tensor([1, 1, 2, 2])
        cams = torch.tensor([0, 1, 0, 1])
        loss = cross_camera_batch_hard(z, ids, cams, margin=0.2)
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertGreater(float(z.grad.abs().sum()), 0.0)
        no_cross_cam = cross_camera_batch_hard(z.detach(), ids, torch.zeros(4, dtype=torch.long), margin=0.2)
        self.assertEqual(float(no_cross_cam), 0.0)

    def test_real_training_step_updates_head_with_finite_loss(self):
        torch.manual_seed(3)
        model = CrossViewCore(num_ids=2)
        set_training_stage(model, 1)
        images = torch.randint(0, 256, (4, 3, 64, 64)).float()
        rows = [(images[i], i // 2, i % 2, i) for i in range(4)]
        loader = DataLoader(rows, batch_size=4)
        optimizer = torch.optim.AdamW(model.global_head.parameters(), lr=1e-3)
        before = model.global_head.weight.detach().clone()
        stats = train_epoch(model, loader, optimizer, torch.device("cpu"), margin=0.3,
                            metric_weight=0.5, smoothing=0.1,
                            wall_deadline=time.monotonic() + 30)
        self.assertEqual(stats["batches"], 1)
        self.assertTrue(np.isfinite(stats["loss"]))
        self.assertFalse(torch.equal(before, model.global_head.weight))

    def test_real_pk_smoke_checks_backbone_and_all_heads(self):
        torch.manual_seed(5)
        model = CrossViewCore(num_ids=8)
        images = torch.randint(0, 256, (32, 3, 64, 64)).float()
        rows = [(images[i], i // 4, i % 2, i) for i in range(32)]
        cfg = {"stage2_backbone_lr": 2e-4, "stage2_head_lr": 5e-4,
               "weight_decay": 1e-4, "peak_rss_limit_mib": 4096}
        result = smoke_one_batch(model, DataLoader(rows, batch_size=32), cfg,
                                 torch.device("cpu"), time.monotonic() + 30)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["cross_camera_anchor_count"], 32)
        self.assertTrue(all(result["gradients"].values()))
        self.assertIsNone(result["stats"]["peak_vram_bytes"])
        self.assertGreater(result["stats"]["peak_rss_bytes"], 0)

    def test_checkpoint_contains_required_source_hashes(self):
        hashes = {key: key * 3 for key in ("protocol", "imagenet", "core_py", "dataset_py", "train_py")}
        checkpoint = make_checkpoint({"weight": torch.ones(1)}, 8, hashes, 3, 0.4)
        self.assertEqual(checkpoint["protocol_sha256"], hashes["protocol"])
        self.assertEqual(checkpoint["source_sha256"],
                         {key: hashes[key] for key in ("core_py", "dataset_py", "train_py")})
        self.assertEqual(checkpoint["model_config"]["descriptor_dim"], 512)
        with self.assertRaisesRegex(ValueError, "train_py"):
            make_checkpoint({}, 8, {k: v for k, v in hashes.items() if k != "train_py"}, 3, 0.4)

    def test_best_checkpoint_is_recoverable_and_explicitly_partial(self):
        hashes = {key: "a" * 64 for key in ("protocol", "imagenet", "core_py", "dataset_py", "train_py")}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            first = make_checkpoint({"weight": torch.tensor([1.0])}, 8, hashes, 3, 0.4)
            status1 = save_best_checkpoint(out, first)
            self.assertEqual(status1["status"], "PARTIAL")
            self.assertEqual(status1["best_epoch"], 3)
            self.assertEqual(torch.load(out / "checkpoint.pt", weights_only=True)["best_epoch"], 3)
            second = make_checkpoint({"weight": torch.tensor([2.0])}, 8, hashes, 5, 0.5)
            status2 = save_best_checkpoint(out, second)
            self.assertNotEqual(status1["checkpoint_sha256"], status2["checkpoint_sha256"])
            self.assertEqual(torch.load(out / "checkpoint.pt", weights_only=True)["best_epoch"], 5)
            self.assertEqual(json.loads((out / "checkpoint-status.json").read_text())["status"], "PARTIAL")
            self.assertEqual(list(out.glob(".*.tmp")), [])

    def test_pilot_projection_accounts_for_slower_unfrozen_backward(self):
        # 2 remaining epochs x 10 steps x 20 s/step x 1.25 + 200 s elapsed.
        self.assertEqual(projected_runtime(200, 100, 20, 10, 2), 700)

    def test_vram_guard_rejects_limit_including_exact_boundary(self):
        limit = math.ceil(3.6 * 1024**3)
        require_vram_below(limit - 1)
        with self.assertRaisesRegex(RuntimeError, "VRAM"):
            require_vram_below(limit)


if __name__ == "__main__":
    unittest.main()
