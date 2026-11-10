# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo
#
# Architectures ported from PaddleSeg Matting/ppmatting/models/backbone (Apache-2.0):
#   https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.10/Matting/ppmatting/models/backbone

from itertools import accumulate

import torch
import torch.nn.functional as F
from torch import nn

from model.common import Conv2dNormActivation

__all__ = ['HRNetW18', 'STDC1', 'pad_same_3x3_stride2']


# STDC1
class CatBottleneck(nn.Module):
    """STDC module: four chained convs whose outputs are concatenated."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        widths = [out_channels // 2, out_channels // 4, out_channels // 8, out_channels // 8]
        self.conv_list = nn.ModuleList(
            [Conv2dNormActivation(in_channels, widths[0], kernel_size=1)]
            + [Conv2dNormActivation(c_in, c_out) for c_in, c_out in zip(widths, widths[1:], strict=False)]
        )
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(widths[0], widths[0], 3, stride=2, padding=1, groups=widths[0], bias=False),
                nn.BatchNorm2d(widths[0]),
            )
            # Paddle's AvgPool2D defaults to exclusive=True, which leaves the padding out of the average.
            self.skip = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.conv_list[0](x)
        out = self.avd_layer(first) if self.stride == 2 else first
        outs = []
        for conv in self.conv_list[1:]:
            out = conv(out)
            outs.append(out)
        if self.stride == 2:
            first = self.skip(first)
        return torch.cat([first, *outs], dim=1)


class STDC1(nn.Module):
    """STDC1 backbone, returning features at strides 2, 4, 8, 16 and 32."""

    def __init__(self) -> None:
        super().__init__()
        self.feat_channels = [32, 64, 256, 512, 1024]
        blocks_per_stage = (2, 2, 2)

        channels = self.feat_channels
        layers = [
            Conv2dNormActivation(3, channels[0], stride=2),
            Conv2dNormActivation(channels[0], channels[1], stride=2),
        ]
        for stage, num_blocks in enumerate(blocks_per_stage):
            in_channels, out_channels = channels[stage + 1], channels[stage + 2]
            layers.append(CatBottleneck(in_channels, out_channels, stride=2))
            layers += [CatBottleneck(out_channels, out_channels, stride=1) for _ in range(num_blocks - 1)]
        self.features = nn.Sequential(*layers)
        # Outputs are taken after each stem conv and after the last block of each stage: (0, 1, 3, 5, 7).
        self.out_indices = (0, *accumulate(blocks_per_stage, initial=1))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_indices:
                feats.append(x)
        return feats


# HRNet-W18
def pad_same_3x3_stride2(x: torch.Tensor) -> torch.Tensor:
    """Zero-pad as Paddle's `padding='same'` does for a 3x3 stride-2 conv: `n % 2` before, 1 after."""
    h, w = x.shape[2:]
    return F.pad(x, (w % 2, 1, h % 2, 1))


class BottleneckBlock(nn.Module):
    """ResNet bottleneck block."""

    def __init__(self, in_channels: int, num_filters: int, downsample: bool) -> None:
        super().__init__()
        self.conv1 = Conv2dNormActivation(in_channels, num_filters, 1)
        self.conv2 = Conv2dNormActivation(num_filters, num_filters, 3)
        self.conv3 = Conv2dNormActivation(num_filters, num_filters * 4, 1, activation_layer=None)
        self.downsample = downsample
        if downsample:
            self.conv_down = Conv2dNormActivation(in_channels, num_filters * 4, 1, activation_layer=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv_down(x) if self.downsample else x
        return F.relu(self.conv3(self.conv2(self.conv1(x))) + residual)


class BasicBlock(nn.Module):
    """ResNet basic block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = Conv2dNormActivation(channels, channels, 3)
        self.conv2 = Conv2dNormActivation(channels, channels, 3, activation_layer=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv2(self.conv1(x)) + x)


class Layer(nn.Module):
    """HRNet stage 1: a chain of bottleneck blocks."""

    def __init__(self, in_channels: int, num_filters: int, num_blocks: int, name: str) -> None:
        super().__init__()
        self.blocks = []
        for i in range(num_blocks):
            block = BottleneckBlock(in_channels if i == 0 else num_filters * 4, num_filters, downsample=i == 0)
            self.add_module(f'bb_{name}_{i + 1}', block)
            self.blocks.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class TransitionLayer(nn.Module):
    """Adapts the branch widths between stages and opens one new branch."""

    def __init__(self, in_channels: list[int], out_channels: list[int], name: str) -> None:
        super().__init__()
        self.convs = []
        for i, out_ch in enumerate(out_channels):
            conv = None
            if i >= len(in_channels):
                conv = Conv2dNormActivation(in_channels[-1], out_ch, 3, stride=2, padding=0)
            elif in_channels[i] != out_ch:
                conv = Conv2dNormActivation(in_channels[i], out_ch, 3)
            if conv is not None:
                self.add_module(f'transition_{name}_layer_{i + 1}', conv)
            self.convs.append(conv)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        outs = []
        for i, conv in enumerate(self.convs):
            if conv is None:
                outs.append(x[i])
            else:
                outs.append(conv(x[i]) if i < len(x) else conv(pad_same_3x3_stride2(x[-1])))
        return outs


class Branches(nn.Module):
    """Runs every branch through its own chain of basic blocks."""

    def __init__(self, num_blocks: list[int], channels: list[int], name: str) -> None:
        super().__init__()
        self.branches = []
        for i, (n, ch) in enumerate(zip(num_blocks, channels, strict=True)):
            blocks = []
            for j in range(n):
                block = BasicBlock(ch)
                self.add_module(f'bb_{name}_branch_layer_{i + 1}_{j + 1}', block)
                blocks.append(block)
            self.branches.append(blocks)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        outs = []
        for feat, blocks in zip(x, self.branches, strict=True):
            for block in blocks:
                feat = block(feat)
            outs.append(feat)
        return outs


class FuseLayers(nn.Module):
    """Adds every other branch, resampled, to each branch."""

    def __init__(self, channels: list[int], name: str) -> None:
        super().__init__()
        self.num_branches = len(channels)
        self.convs = []
        for i in range(self.num_branches):
            for j in range(self.num_branches):
                if j > i:
                    conv = Conv2dNormActivation(channels[j], channels[i], 1, activation_layer=None)
                    self.add_module(f'residual_{name}_layer_{i + 1}_{j + 1}', conv)
                    self.convs.append(conv)
                elif j < i:
                    for k in range(i - j):
                        last = k == i - j - 1
                        out_ch = channels[i] if last else channels[j]
                        conv = Conv2dNormActivation(
                            channels[j], out_ch, 3, stride=2, padding=0, activation_layer=None if last else nn.ReLU
                        )
                        self.add_module(f'residual_{name}_layer_{i + 1}_{j + 1}_{k + 1}', conv)
                        self.convs.append(conv)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        convs = iter(self.convs)
        outs = []
        for i in range(self.num_branches):
            residual = x[i]
            for j in range(self.num_branches):
                if j > i:
                    y = next(convs)(x[j])
                    y = F.interpolate(y, residual.shape[2:], mode='bilinear', align_corners=False)
                    residual = residual + y
                elif j < i:
                    y = x[j]
                    for _ in range(i - j):
                        y = next(convs)(pad_same_3x3_stride2(y))
                    residual = residual + y
            outs.append(F.relu(residual))
        return outs


class HighResolutionModule(nn.Module):
    """Parallel branches followed by the fuse layer."""

    def __init__(self, num_blocks: list[int], channels: list[int], name: str) -> None:
        super().__init__()
        self.branches_func = Branches(num_blocks, channels, name)
        self.fuse_func = FuseLayers(channels, name)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        return self.fuse_func(self.branches_func(x))


class Stage(nn.Module):
    """A chain of high-resolution modules."""

    def __init__(self, num_modules: int, num_blocks: list[int], channels: list[int], name: str) -> None:
        super().__init__()
        self.modules_list = []
        for i in range(num_modules):
            module = HighResolutionModule(num_blocks, channels, f'{name}_{i + 1}')
            self.add_module(f'stage_{name}_{i + 1}', module)
            self.modules_list.append(module)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        for module in self.modules_list:
            x = module(x)
        return x


class HRNetW18(nn.Module):
    """HRNet-W18 backbone, returning features at strides 2, 4, 8, 16 and 32."""

    def __init__(self) -> None:
        super().__init__()
        stage2, stage3, stage4 = [18, 36], [18, 36, 72], [18, 36, 72, 144]
        self.feat_channels = [64, *stage4]

        self.conv_layer1_1 = Conv2dNormActivation(3, 64, 3, stride=2, padding=0)
        self.conv_layer1_2 = Conv2dNormActivation(64, 64, 3, stride=2, padding=0)
        self.la1 = Layer(64, num_filters=64, num_blocks=4, name='layer2')
        self.tr1 = TransitionLayer([64 * 4], stage2, name='tr1')
        self.st2 = Stage(1, [4, 4], stage2, name='st2')
        self.tr2 = TransitionLayer(stage2, stage3, name='tr2')
        self.st3 = Stage(4, [4, 4, 4], stage3, name='st3')
        self.tr3 = TransitionLayer(stage3, stage4, name='tr3')
        self.st4 = Stage(3, [4, 4, 4, 4], stage4, name='st4')

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        stem = self.conv_layer1_1(pad_same_3x3_stride2(x))
        x = self.la1(self.conv_layer1_2(pad_same_3x3_stride2(stem)))
        x = self.st2(self.tr1([x]))
        x = self.st3(self.tr2(x))
        x = self.st4(self.tr3(x))
        return [stem, *x]
