"""OSNet-AIN (`vehicle-reid-0001`) — самодостаточная сборка с именами параметров,
совпадающими с инициализаторами ONNX-файла OMZ.

Модуль переписан по эталону sovrasov/deep-person-reid (ветка `vehicle_reid`, MIT):
оставлено ровно то, что реально присутствует в экспортированном графе — без
attention-модулей, LCT-гейта и атрибутных голов, которых в 559 инициализаторах нет.
Правильность сборки доказывается не происхождением кода, а числом: выходы torch
обязаны совпасть с выходами ONNX (см. `code/build_from_onnx.py`).

Топология, вычитанная прямо из графа:
  input_IN (InstanceNorm2d(3, affine)) -> conv1 (Conv7x7/s2 + InstanceNorm2d(64,affine) + ReLU)
  -> MaxPool3x3/s2 -> conv2 [OSBlockINin, OSBlockINin] -> pool2 (Conv1x1+BN+ReLU, AvgPool2/2)
  -> conv3 [OSBlock, OSBlockINin] -> pool3 -> conv4 [OSBlockINin, OSBlock]
  -> conv5 (Conv1x1+BN+ReLU) -> GAP -> две головы fc.0/fc.1 (Linear 512->256 + BN1d)
  -> конкатенация 512. L2-нормировки в графе НЕТ (её делает потребитель).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0, IN=False):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=False)
        self.bn = nn.InstanceNorm2d(cout, affine=True) if IN else nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, cin, cout, stride=1, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.out_fn = nn.ReLU() if relu else None

    def forward(self, x):
        y = self.bn(self.conv(x))
        return self.out_fn(y) if self.out_fn is not None else y


class Conv1x1Linear(nn.Module):
    def __init__(self, cin, cout, stride=1, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(cout) if bn else None

    def forward(self, x):
        x = self.conv(x)
        return self.bn(x) if self.bn is not None else x


class LightConv3x3(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False, groups=cout)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class LightConvStream(nn.Module):
    def __init__(self, cin, cout, depth):
        super().__init__()
        layers = [LightConv3x3(cin, cout)]
        layers += [LightConv3x3(cout, cout) for _ in range(depth - 1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ChannelGate(nn.Module):
    def __init__(self, cin, reduction=16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(cin, cin // reduction, 1, bias=True, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(cin // reduction, cin, 1, bias=True, padding=0)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        y = self.gate_activation(self.fc2(self.relu(self.fc1(self.global_avgpool(x)))))
        return x * y


class _OSBase(nn.Module):
    """Общая часть OSBlock и OSBlockINin: 4 потока LightConvStream + канальный гейт."""

    def __init__(self, cin, cout, reduction=4, T=4, use_in=False):
        super().__init__()
        mid = cout // reduction
        self.conv1 = Conv1x1(cin, mid)
        self.conv2 = nn.ModuleList([LightConvStream(mid, mid, t) for t in range(1, T + 1)])
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, cout, bn=not use_in)
        self.downsample = Conv1x1Linear(cin, cout) if cin != cout else None
        self.IN = nn.InstanceNorm2d(cout, affine=True) if use_in else None

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        x1 = self.conv1(x)
        x2 = 0
        for stream in self.conv2:
            x2 = x2 + self.gate(stream(x1))
        x3 = self.conv3(x2)
        if self.IN is not None:
            x3 = self.IN(x3)
        return F.relu(x3 + identity)


class OSBlock(_OSBase):
    def __init__(self, cin, cout, **kw):
        super().__init__(cin, cout, use_in=False, **kw)


class OSBlockINin(_OSBase):
    def __init__(self, cin, cout, **kw):
        super().__init__(cin, cout, use_in=True, **kw)


class OSNetAIN(nn.Module):
    """`osnet_ain2_x1_0` из ветки vehicle_reid: input_IN=True, conv1_IN=True,
    две fc-головы по 256 (итоговый вектор 512)."""

    BLOCKS = ([OSBlockINin, OSBlockINin], [OSBlock, OSBlockINin], [OSBlockINin, OSBlock])
    CH = (64, 256, 384, 512)

    def __init__(self, feature_dim=256, n_heads=2, num_classes=0):
        super().__init__()
        c = self.CH
        self.input_IN = nn.InstanceNorm2d(3, affine=True)
        self.conv1 = ConvLayer(3, c[0], 7, stride=2, padding=3, IN=True)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Sequential(self.BLOCKS[0][0](c[0], c[1]), self.BLOCKS[0][1](c[1], c[1]))
        self.pool2 = nn.Sequential(Conv1x1(c[1], c[1]), nn.AvgPool2d(2, stride=2))
        self.conv3 = nn.Sequential(self.BLOCKS[1][0](c[1], c[2]), self.BLOCKS[1][1](c[2], c[2]))
        self.pool3 = nn.Sequential(Conv1x1(c[2], c[2]), nn.AvgPool2d(2, stride=2))
        self.conv4 = nn.Sequential(self.BLOCKS[2][0](c[2], c[3]), self.BLOCKS[2][1](c[3], c[3]))
        self.conv5 = Conv1x1(c[3], c[3])
        self.fc = nn.ModuleList([
            nn.Sequential(nn.Linear(c[3], feature_dim), nn.BatchNorm1d(feature_dim))
            for _ in range(n_heads)])
        self.dim = feature_dim * n_heads
        self.classifier = nn.Linear(self.dim, num_classes, bias=False) if num_classes else None
        if self.classifier is not None:
            nn.init.normal_(self.classifier.weight, std=0.001)

    # --- части, между которыми режется заморозка (LP-FT) -------------------
    def stem(self, x):
        y = self.input_IN(x)
        y = self.pool1(self.conv1(y))
        y = self.pool2(self.conv2(y))
        return y

    def mid(self, y):
        return self.pool3(self.conv3(y))

    def top(self, y):
        return self.conv5(self.conv4(y))

    def head(self, feat_map):
        g = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        return torch.cat([f(g) for f in self.fc], dim=1)

    def forward(self, x):
        """Вход — RGB 0..255 NCHW (нормировку делает input_IN, как в исходном графе)."""
        return self.head(self.top(self.mid(self.stem(x))))

    def forward_norm(self, x):
        return F.normalize(self.forward(x), dim=1)
