# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo
#
# Conv2dNormActivation follows torchvision.ops.misc (BSD-3-Clause):
#   https://github.com/pytorch/vision/blob/main/torchvision/ops/misc.py

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

__all__ = ['AdaptiveAvgPool2d', 'Conv2dNormActivation', 'adaptive_avg_pool2d']


class Conv2dNormActivation(nn.Sequential):
    """Conv2d, optional norm layer and optional activation, indexed 0, 1, 2 as in torchvision."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU,
        dilation: int = 1,
        inplace: bool | None = True,
        bias: bool | None = None,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer() if inplace is None else activation_layer(inplace=inplace))
        super().__init__(*layers)


def _pooling_matrix(size: int | torch.Tensor, bins: int, like: torch.Tensor) -> torch.Tensor:
    """Return the (bins, size) matrix whose rows average the windows of `F.adaptive_avg_pool2d`."""
    i = torch.arange(bins, device=like.device)
    start = torch.div(i * size, bins, rounding_mode='floor')
    end = torch.div((i + 1) * size + bins - 1, bins, rounding_mode='floor')

    j = torch.arange(size, device=like.device)
    mask = (j >= start[:, None]) & (j < end[:, None])
    return mask.to(like.dtype) / (end - start).to(like.dtype)[:, None]


def adaptive_avg_pool2d(x: torch.Tensor, output_size: int | tuple[int, int]) -> torch.Tensor:
    """Adaptive average pooling that exports to ONNX with dynamic height and width.

    Args:
        x: Input of shape (N, C, H, W).
        output_size: Number of bins, one value for both sides or (out_h, out_w).

    Returns:
        Tensor of shape (N, C, out_h, out_w).
    """
    out_h, out_w = (output_size, output_size) if isinstance(output_size, int) else output_size
    if out_h == 1 and out_w == 1:
        return x.mean((2, 3), keepdim=True)

    h, w = x.shape[2:]
    pool_h = _pooling_matrix(h, out_h, x)
    pool_w = _pooling_matrix(w, out_w, x)
    return torch.matmul(torch.matmul(pool_h, x), pool_w.transpose(0, 1))


class AdaptiveAvgPool2d(nn.Module):
    """`nn.AdaptiveAvgPool2d` built on `adaptive_avg_pool2d`."""

    def __init__(self, output_size: int) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return adaptive_avg_pool2d(x, self.output_size)
