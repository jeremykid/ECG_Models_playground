"""Medformer-style ECG classifier.

Adapted from DeepECG-Kit's Medformer implementation:
https://github.com/stevenah/deepecg-kit/blob/main/deepecgkit/models/medformer.py

DeepECG-Kit is licensed under Apache-2.0. The underlying Medformer
architecture is from Wang et al., "Medformer: A Multi-Granularity Patching
Transformer for Medical Time-Series Classification", NeurIPS 2024.
"""

from __future__ import annotations

import torch
from torch import nn


def _resolve_alias(
    primary_value: int | None,
    alias_value: int | None,
    default: int,
    primary_name: str,
    alias_name: str,
) -> int:
    if primary_value is not None and alias_value is not None and primary_value != alias_value:
        raise ValueError(
            f"`{primary_name}` and `{alias_name}` were both provided with different values."
        )
    if primary_value is not None:
        return primary_value
    if alias_value is not None:
        return alias_value
    return default


class PatchEmbedding1D(nn.Module):
    """Converts a 1D signal into non-overlapping patch embeddings."""

    def __init__(self, input_channels: int, d_model: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv1d(input_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).transpose(1, 2)


class CrossChannelAttention(nn.Module):
    """Self-attention over a token sequence, kept for Medformer compatibility."""

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x, _ = self.attn(x, x, x)
        return self.norm(x + residual)


class IntraGranularityBlock(nn.Module):
    """Transformer encoder block within one patch granularity."""

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class InterGranularityAttention(nn.Module):
    """Fuses information across patch granularities."""

    def __init__(
        self,
        d_model: int,
        num_granularities: int,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.num_granularities = num_granularities

    def forward(self, granularity_features: list[torch.Tensor]) -> list[torch.Tensor]:
        pooled = [features.mean(dim=1) for features in granularity_features]
        stacked = torch.stack(pooled, dim=1)
        attn_out, _ = self.attn(stacked, stacked, stacked)
        attn_out = self.norm(attn_out + stacked)

        updated = []
        for index, features in enumerate(granularity_features):
            scale = attn_out[:, index].unsqueeze(1)
            updated.append(features + scale)
        return updated


class Medformer(nn.Module):
    """Multi-granularity patching Transformer for ECG classification.

    Args:
        input_channels: Number of ECG leads or channels.
        output_size: Number of output classes or targets.
        num_leads: Alias for ``input_channels``.
        num_classes: Alias for ``output_size``.
        d_model: Transformer embedding dimension.
        patch_sizes: Patch sizes for each granularity level.
        num_encoder_layers: Number of intra/inter granularity fusion layers.
        nhead: Number of attention heads.
        dim_feedforward: Transformer feedforward dimension.
        dropout_rate: Dropout probability.
        max_patches: Maximum supported patch tokens per granularity.
    """

    def __init__(
        self,
        input_channels: int | None = None,
        output_size: int | None = None,
        *,
        num_leads: int | None = None,
        num_classes: int | None = None,
        d_model: int = 128,
        patch_sizes: tuple[int, ...] = (10, 25, 50),
        num_encoder_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 256,
        dropout_rate: float = 0.1,
        max_patches: int = 500,
    ):
        super().__init__()
        input_channels = _resolve_alias(
            input_channels, num_leads, 1, "input_channels", "num_leads"
        )
        output_size = _resolve_alias(output_size, num_classes, 4, "output_size", "num_classes")

        self.patch_sizes = patch_sizes
        self.d_model = d_model
        num_granularities = len(patch_sizes)

        self.patch_embeddings = nn.ModuleList(
            [PatchEmbedding1D(input_channels, d_model, patch_size) for patch_size in patch_sizes]
        )
        self.pos_embeddings = nn.ParameterList(
            [nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02) for _ in patch_sizes]
        )
        self.intra_blocks = nn.ModuleList()
        self.inter_blocks = nn.ModuleList()
        for _ in range(num_encoder_layers):
            self.intra_blocks.append(
                nn.ModuleList(
                    [
                        IntraGranularityBlock(
                            d_model,
                            nhead,
                            dim_feedforward,
                            dropout_rate,
                        )
                        for _ in patch_sizes
                    ]
                )
            )
            self.inter_blocks.append(
                InterGranularityAttention(
                    d_model,
                    num_granularities,
                    min(4, nhead),
                    dropout_rate,
                )
            )

        self.dropout = nn.Dropout(dropout_rate)
        self._feature_dim = d_model * num_granularities
        self.classifier = nn.Linear(self._feature_dim, output_size)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        granularity_features = []
        for patch_embedding, pos_embedding in zip(self.patch_embeddings, self.pos_embeddings):
            patches = patch_embedding(x)
            num_patches = patches.size(1)
            if num_patches > pos_embedding.size(1):
                raise ValueError(
                    f"Input creates {num_patches} patches, but max_patches is "
                    f"{pos_embedding.size(1)}."
                )
            patches = patches + pos_embedding[:, :num_patches]
            granularity_features.append(patches)

        for intra_layer, inter_layer in zip(self.intra_blocks, self.inter_blocks):
            for index, intra_block in enumerate(intra_layer):
                granularity_features[index] = intra_block(granularity_features[index])
            granularity_features = inter_layer(granularity_features)

        pooled = [features.mean(dim=1) for features in granularity_features]
        return torch.cat(pooled, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "PatchEmbedding1D",
    "CrossChannelAttention",
    "IntraGranularityBlock",
    "InterGranularityAttention",
    "Medformer",
]
