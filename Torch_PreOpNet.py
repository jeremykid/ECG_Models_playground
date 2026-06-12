"""
PreOpNet models adapted from the original EffNet-style implementation:
https://github.com/ecg-net/PreOpNet/blob/main/models.py

The source repository uses the Cedars-Sinai Academic Software License. The
underlying model is described in Ouyang et al., "Electrocardiographic Deep
Learning for Predicting Post-Procedural Mortality".

This module intentionally provides two wrappers around a shared ECG encoder:

- PreOpNetBackbone:
  Returns an embedding for downstream task heads. When tabular features are
  present, they are projected to ``5 * num_features`` before late fusion. This
  matches the fusion style used by the rest of our ECG model family.

- PreOpNetRawClinical:
  Keeps the paper-faithful raw-clinical late fusion pattern. Additional
  features are concatenated directly to the pooled ECG embedding before the
  final linear head.

The shared encoder is exposed as PreOpNetEncoder. The ``PreOpNet`` name points
to the raw-clinical wrapper for convenience when reproducing the original
end-to-end behavior.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        expansion: int,
        activation: type[nn.Module],
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.stride = stride
        expanded = in_channel * expansion
        self.conv1 = nn.Conv1d(in_channel, expanded, kernel_size=1)
        self.conv2 = nn.Conv1d(
            expanded,
            expanded,
            kernel_size=3,
            groups=expanded,
            padding=padding,
            stride=stride,
        )
        self.conv3 = nn.Conv1d(expanded, out_channel, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm1d(expanded)
        self.bn2 = nn.BatchNorm1d(expanded)
        self.dropout = nn.Dropout()
        self.act = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.conv1(x)))
        y = self.act(self.bn2(self.conv2(y)))
        y = self.conv3(y)
        if self.stride == 1:
            y = self.dropout(y)
            y = x + y
        return y


class MBConv(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channels: int,
        expansion: int,
        layers: int,
        activation: type[nn.Module] = nn.ReLU6,
        stride: int = 2,
    ) -> None:
        super().__init__()
        blocks = OrderedDict()
        for idx in range(0, layers - 1):
            blocks[f"block{idx}"] = Bottleneck(
                in_channel,
                in_channel,
                expansion,
                activation,
            )
        blocks[f"block{layers}"] = Bottleneck(
            in_channel,
            out_channels,
            expansion,
            activation,
            stride=stride,
        )
        self.stack = nn.Sequential(blocks)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stack(x)
        return self.bn(x)


class PreOpNetEncoder(nn.Module):
    """Shared ECG encoder adapted from the original EffNet backbone."""

    def __init__(
        self,
        depth: list[int] | tuple[int, ...] = (1, 2, 2, 3, 3, 3, 3),
        channels: list[int] | tuple[int, ...] = (32, 16, 24, 40, 80, 112, 192, 320, 1280),
        dilation: int = 1,
        stride: int = 2,
        expansion: int = 6,
        in_chans: int = 12,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.embedding_dim = channels[8]
        self.stage1 = nn.Conv1d(
            in_chans,
            channels[0],
            kernel_size=3,
            stride=stride,
            padding=1,
            dilation=dilation,
        )
        self.stage1_bn = nn.BatchNorm1d(channels[0])
        self.stage2 = MBConv(channels[0], channels[1], expansion, depth[0], stride=2)
        self.stage3 = MBConv(channels[1], channels[2], expansion, depth[1], stride=2)
        self.pool = nn.MaxPool1d(3, stride=1, padding=1)
        self.stage4 = MBConv(channels[2], channels[3], expansion, depth[2], stride=2)
        self.stage5 = MBConv(channels[3], channels[4], expansion, depth[3], stride=2)
        self.stage6 = MBConv(channels[4], channels[5], expansion, depth[4], stride=2)
        self.stage7 = MBConv(channels[5], channels[6], expansion, depth[5], stride=2)
        self.stage8 = MBConv(channels[6], channels[7], expansion, depth[6], stride=2)
        self.stage9 = nn.Conv1d(channels[7], channels[8], kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1_bn(self.stage1(x))
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.pool(x)
        x = self.stage7(x)
        x = self.stage8(x)
        x = self.stage9(x)
        x = self.act(self.avg_pool(x)[:, :, 0])
        x = self.dropout(x)
        return x


class PreOpNetBackbone(nn.Module):
    """
    Future-facing PreOpNet variant that returns an embedding.

    This wrapper keeps clinical features separate from the encoder. If tabular
    features are provided they are projected to ``5 * num_features`` before
    late fusion, matching the rest of our ECG backbones.
    """

    def __init__(
        self,
        num_features: int = 0,
        depth: list[int] | tuple[int, ...] = (1, 2, 2, 3, 3, 3, 3),
        channels: list[int] | tuple[int, ...] = (32, 16, 24, 40, 80, 112, 192, 320, 1280),
        dilation: int = 1,
        stride: int = 2,
        expansion: int = 6,
        in_chans: int = 12,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.encoder = PreOpNetEncoder(
            depth=depth,
            channels=channels,
            dilation=dilation,
            stride=stride,
            expansion=expansion,
            in_chans=in_chans,
            dropout_rate=dropout_rate,
        )
        self.ecg_embedding_dim = self.encoder.embedding_dim
        self.feature_proj_dim = 5 * num_features
        self.embedding_dim = self.ecg_embedding_dim + self.feature_proj_dim
        if num_features > 0:
            self.feature_proj = nn.Linear(num_features, self.feature_proj_dim)
        else:
            self.feature_proj = None

    def forward(self, inputs: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if self.num_features > 0:
            ecg, features = inputs
        else:
            ecg = inputs
            features = None

        ecg_embedding = self.encoder(ecg)
        if self.feature_proj is None:
            return ecg_embedding

        projected = self.feature_proj(features)
        return torch.cat((ecg_embedding, projected), dim=1)


class PreOpNetRawClinical(nn.Module):
    """
    Paper-faithful late fusion wrapper adapted from the original EffNet.

    Additional features are treated as a raw numeric vector and concatenated
    directly to the pooled ECG embedding before the final head.
    """

    def __init__(
        self,
        num_additional_features: int = 0,
        num_classes: int = 1,
        depth: list[int] | tuple[int, ...] = (1, 2, 2, 3, 3, 3, 3),
        channels: list[int] | tuple[int, ...] = (32, 16, 24, 40, 80, 112, 192, 320, 1280),
        dilation: int = 1,
        stride: int = 2,
        expansion: int = 6,
        in_chans: int = 12,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_additional_features = num_additional_features
        self.encoder = PreOpNetEncoder(
            depth=depth,
            channels=channels,
            dilation=dilation,
            stride=stride,
            expansion=expansion,
            in_chans=in_chans,
            dropout_rate=dropout_rate,
        )
        self.head = nn.Linear(self.encoder.embedding_dim + num_additional_features, num_classes)

    def forward(self, inputs: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if self.num_additional_features > 0:
            ecg, additional = inputs
        else:
            ecg = inputs
            additional = None

        embedding = self.encoder(ecg)
        if self.num_additional_features > 0:
            embedding = torch.cat((embedding, additional), dim=1)
        return self.head(embedding)


class PreOpNet(PreOpNetRawClinical):
    """Default PreOpNet wrapper using the paper-faithful raw clinical fusion."""


# Legacy alias kept for old references that still use the original name.
EffNet = PreOpNetRawClinical
