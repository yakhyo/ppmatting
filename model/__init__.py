# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from typing import TYPE_CHECKING

from .ppmatting_onnx import PPMattingONNX, postprocess, preprocess

if TYPE_CHECKING:
    from torch import nn

__all__ = ['PPMattingONNX', 'load_model', 'postprocess', 'preprocess']


def load_model(name: str, weights: str) -> nn.Module:
    """Build `ppmatting_v1` or `ppmatting_v2`, load the state dict at `weights` and return it in eval mode."""
    import torch

    from .ppmatting import PPMatting, PPMattingV2

    if name == 'ppmatting_v1':
        model = PPMatting()
    elif name == 'ppmatting_v2':
        model = PPMattingV2()
    else:
        raise ValueError(f"unknown model {name!r}, expected 'ppmatting_v1' or 'ppmatting_v2'")
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    return model.eval()
