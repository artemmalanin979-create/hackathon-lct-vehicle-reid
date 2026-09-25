"""Standalone RGB crop -> vehicle descriptor, independent of the release ensemble."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small


class CrossViewCore(nn.Module):
    """MobileNetV3-Small with learned global and two horizontal-local branches.

    Forward accepts RGB float32 NCHW in [0, 255] and returns one L2-normalized
    512-vector per image. The classifier is training-only and is never exported.
    """

    descriptor_dim = 512

    def __init__(self, num_ids: int):
        super().__init__()
        if num_ids < 2:
            raise ValueError("num_ids must be at least two")
        self.num_ids = int(num_ids)
        self.backbone = mobilenet_v3_small(weights=None).features
        channels = 576  # final feature channels of torchvision MobileNetV3-Small
        self.global_head = nn.Linear(channels, 256)
        self.top_head = nn.Linear(channels, 128)
        self.bottom_head = nn.Linear(channels, 128)
        self.classifier = nn.Linear(self.descriptor_dim, num_ids, bias=False)
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone((x / 255.0 - self.rgb_mean) / self.rgb_std)
        global_feature = F.adaptive_avg_pool2d(feature_map, (1, 1)).flatten(1)
        stripes = F.adaptive_avg_pool2d(feature_map, (2, 1))
        descriptor = torch.cat((
            self.global_head(global_feature),
            self.top_head(stripes[:, :, 0, 0]),
            self.bottom_head(stripes[:, :, 1, 0]),
        ), dim=1)
        return F.normalize(descriptor, p=2, dim=1, eps=1e-12)

    def classify(self, descriptor: torch.Tensor) -> torch.Tensor:
        return self.classifier(descriptor)

    def load_imagenet(self, path: str | Path, expected_sha256: str) -> str:
        """Load official torchvision state_dict from a separately verified file.

        No implicit download occurs. Only the features are transferred; our three
        projection heads and identity classifier retain their own initialization.
        """
        path = Path(path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise ValueError(f"ImageNet weight hash mismatch: {actual} != {expected_sha256}")
        source = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(source, dict):
            raise ValueError("ImageNet checkpoint must be a torchvision state_dict")
        features = {name.removeprefix("features."): weight
                    for name, weight in source.items() if name.startswith("features.")}
        self.backbone.load_state_dict(features, strict=True)
        return actual
