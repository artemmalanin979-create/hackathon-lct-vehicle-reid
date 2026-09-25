"""Small behavioural checks for the independent image-to-descriptor core."""
from __future__ import annotations

import hashlib
import json
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
from train import (cross_camera_batch_hard, make_checkpoint, projected_runtime,
                   set_training_stage, smoke_one_batch, train_epoch)  # noqa: E402


class CoreBehaviour(unittest.TestCase):
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
               "weight_decay": 1e-4}
        result = smoke_one_batch(model, DataLoader(rows, batch_size=32), cfg,
                                 torch.device("cpu"), time.monotonic() + 30)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["cross_camera_anchor_count"], 32)
        self.assertTrue(all(result["gradients"].values()))

    def test_checkpoint_contains_required_source_hashes(self):
        hashes = {key: key * 3 for key in ("protocol", "imagenet", "core_py", "dataset_py", "train_py")}
        checkpoint = make_checkpoint({"weight": torch.ones(1)}, 8, hashes, 3, 0.4)
        self.assertEqual(checkpoint["protocol_sha256"], hashes["protocol"])
        self.assertEqual(checkpoint["source_sha256"],
                         {key: hashes[key] for key in ("core_py", "dataset_py", "train_py")})
        self.assertEqual(checkpoint["model_config"]["descriptor_dim"], 512)
        with self.assertRaisesRegex(ValueError, "train_py"):
            make_checkpoint({}, 8, {k: v for k, v in hashes.items() if k != "train_py"}, 3, 0.4)

    def test_pilot_projection_accounts_for_slower_unfrozen_backward(self):
        # 2 remaining epochs x 10 steps x 20 s/step x 1.25 + 200 s elapsed.
        self.assertEqual(projected_runtime(200, 100, 20, 10, 2), 700)


if __name__ == "__main__":
    unittest.main()
