"""Bidirectional Mamba-style ECG classifier.

Adapted from DeepECG-Kit's Mamba1D implementation:
https://github.com/stevenah/deepecg-kit/blob/main/deepecgkit/models/mamba1d.py

DeepECG-Kit is licensed under Apache-2.0. The underlying Mamba architecture is
from Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State
Spaces", arXiv:2312.00752, 2023.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
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


class SelectiveSSM(nn.Module):
    """Pure PyTorch selective state space layer.

    The layer follows the Mamba idea of making state space parameters depend on
    the current token. This implementation favors readability and portability
    over the fused kernels used in production Mamba packages.
    """

    def __init__(self, d_inner: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        self.conv1d = nn.Conv1d(
            d_inner,
            d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        a_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
        self.a_log = nn.Parameter(a_log.unsqueeze(0).expand(d_inner, -1))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = F.silu(x_conv).transpose(1, 2)

        projected = self.x_proj(x_conv)
        b_select = projected[:, :, : self.d_state]
        c_select = projected[:, :, self.d_state : self.d_state * 2]
        delta = projected[:, :, self.d_state * 2 :]
        delta = F.softplus(self.dt_proj(delta))

        a_coeff = -torch.exp(self.a_log)
        state = torch.zeros(
            batch_size,
            self.d_inner,
            self.d_state,
            device=x.device,
            dtype=x.dtype,
        )

        outputs = []
        for timestep in range(seq_len):
            dt = delta[:, timestep].unsqueeze(-1)
            a_bar = torch.exp(dt * a_coeff)
            b_bar = dt * b_select[:, timestep].unsqueeze(1)
            x_t = x_conv[:, timestep].unsqueeze(-1)

            state = a_bar * state + b_bar * x_t
            y_t = (state * c_select[:, timestep].unsqueeze(1)).sum(-1)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return y + x * self.D


class MambaBlock(nn.Module):
    """Single Mamba block with pre-norm, gating, SSM, and residual output."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expansion_factor: int = 2,
    ):
        super().__init__()
        d_inner = d_model * expansion_factor

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.ssm = SelectiveSSM(d_inner, d_state, d_conv)
        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x, gate = self.in_proj(x).chunk(2, dim=-1)
        x = self.ssm(x)
        x = x * F.silu(gate)
        x = self.out_proj(x)
        return x + residual


class BidirectionalMamba(nn.Module):
    """Runs Mamba blocks forward and backward, then fuses the two directions."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expansion_factor: int = 2,
    ):
        super().__init__()
        self.forward_block = MambaBlock(d_model, d_state, d_conv, expansion_factor)
        self.backward_block = MambaBlock(d_model, d_state, d_conv, expansion_factor)
        self.combine = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forward_out = self.forward_block(x)
        backward_out = self.backward_block(x.flip(1)).flip(1)
        return self.combine(torch.cat([forward_out, backward_out], dim=-1))


class Mamba1D(nn.Module):
    """Bidirectional Mamba-style model for ECG classification.

    Args:
        input_channels: Number of ECG leads or channels.
        output_size: Number of output classes or targets.
        num_leads: Alias for ``input_channels``.
        num_classes: Alias for ``output_size``.
        d_model: Token embedding dimension.
        d_state: State space dimension.
        d_conv: Local depthwise convolution width.
        expansion_factor: Inner dimension expansion factor.
        num_layers: Number of bidirectional Mamba layers.
        patch_size: Non-overlapping Conv1d patch size.
        dropout_rate: Classifier dropout probability.
        max_patches: Maximum supported patch tokens for positional embeddings.
    """

    def __init__(
        self,
        input_channels: int | None = None,
        output_size: int | None = None,
        *,
        num_leads: int | None = None,
        num_classes: int | None = None,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expansion_factor: int = 2,
        num_layers: int = 4,
        patch_size: int = 50,
        dropout_rate: float = 0.1,
        max_patches: int = 500,
    ):
        super().__init__()
        input_channels = _resolve_alias(
            input_channels, num_leads, 1, "input_channels", "num_leads"
        )
        output_size = _resolve_alias(output_size, num_classes, 4, "output_size", "num_classes")

        self.patch_embed = nn.Conv1d(
            input_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [
                BidirectionalMamba(d_model, d_state, d_conv, expansion_factor)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self._feature_dim = d_model
        self.classifier = nn.Linear(d_model, output_size)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).transpose(1, 2)
        num_patches = x.size(1)
        if num_patches > self.pos_embedding.size(1):
            raise ValueError(
                f"Input creates {num_patches} patches, but max_patches is "
                f"{self.pos_embedding.size(1)}."
            )
        x = x + self.pos_embedding[:, :num_patches]

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = ["SelectiveSSM", "MambaBlock", "BidirectionalMamba", "Mamba1D"]

