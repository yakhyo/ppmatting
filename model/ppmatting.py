# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo
#
# Architectures ported from PaddleSeg Matting/ppmatting/models (Apache-2.0):
#   https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.10/Matting/ppmatting/models

import torch
import torch.nn.functional as F
from torch import nn

from model.common import AdaptiveAvgPool2d, Conv2dNormActivation
from model.networks import STDC1, HRNetW18

__all__ = ['PPMatting', 'PPMattingV2']


# PP-Matting (HRNet-W18)
class PPModule(nn.Module):
    """Pyramid pooling module of PSPNet."""

    def __init__(self, in_channels: int, out_channels: int, bin_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(AdaptiveAvgPool2d(size), Conv2dNormActivation(in_channels, in_channels, 1, bias=True))
                for size in bin_sizes
            ]
        )
        self.conv_bn_relu2 = Conv2dNormActivation(in_channels * (len(bin_sizes) + 1), out_channels, 3, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        pooled = [F.interpolate(stage(x), size, mode='bilinear', align_corners=False) for stage in self.stages]
        # Paddle concatenates the pooled maps in reverse bin order.
        return self.conv_bn_relu2(torch.cat([x, *pooled[::-1]], dim=1))


def upsample(scale_factor: int) -> nn.Upsample:
    """Bilinear upsampling by an integer factor."""
    return nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)


class SCB(nn.Module):
    """Semantic context branch: foreground, transition and background logits from the stride-32 feature."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        stage_in = [512 + in_channels, 512, 256, 128, 128, 64]
        stage_mid = [512, 256, 128, 128, 64, 64]
        self.out_channels = [256, 128, 64, 64, 64, 3]

        self.psp_module = PPModule(in_channels, 512, bin_sizes=(1, 3, 5))
        self.psps = nn.ModuleList(
            [
                nn.Sequential(Conv2dNormActivation(512, self.out_channels[i], 3, bias=True), upsample(factor))
                for i, factor in enumerate([2, 4, 8, 16])
            ]
        )

        stages = [
            nn.Sequential(
                Conv2dNormActivation(stage_in[i], stage_mid[i], 3, bias=True),
                Conv2dNormActivation(stage_mid[i], stage_mid[i], 3, dilation=2 if i == 0 else 1, bias=True),
                Conv2dNormActivation(stage_mid[i], self.out_channels[i], 3, dilation=2 if i == 0 else 1, bias=True),
                upsample(2),
            )
            for i in range(len(stage_in) - 1)
        ]
        stages.append(
            nn.Sequential(
                Conv2dNormActivation(stage_in[-1], stage_mid[-1], 3, bias=True),
                Conv2dNormActivation(stage_mid[-1], stage_mid[-1], 3, bias=True),
                nn.Conv2d(stage_mid[-1], self.out_channels[-1], 3, padding=1),
            )
        )
        self.scb_stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        psp_x = self.psp_module(x)
        psps = [psp(psp_x) for psp in self.psps]

        logits = []
        for i, stage in enumerate(self.scb_stages):
            if i == 0:
                x = stage(torch.cat([psp_x, x], dim=1))
            elif i <= len(psps):
                x = stage(torch.cat([psps[i - 1], x], dim=1))
            else:
                x = stage(x)
            logits.append(x)
        return logits


class BasicBlock(nn.Module):
    """ResNet-vd basic block at stride 1."""

    def __init__(self, in_channels: int, out_channels: int, shortcut: bool = True) -> None:
        super().__init__()
        self.conv0 = Conv2dNormActivation(in_channels, out_channels, 3)
        self.conv1 = Conv2dNormActivation(out_channels, out_channels, 3, activation_layer=None)
        self.shortcut = shortcut
        if not shortcut:
            self.short = Conv2dNormActivation(in_channels, out_channels, 1, activation_layer=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        short = x if self.shortcut else self.short(x)
        return F.relu(short + self.conv1(self.conv0(x)))


class GatedSpatialConv2d(nn.Module):
    """Gated 1x1 conv, gated by the features and a guidance map."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        channels = in_channels + 1
        self._gate_conv = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor, gating: torch.Tensor) -> torch.Tensor:
        alphas = self._gate_conv(torch.cat([x, gating], dim=1))
        return self.conv(x * (alphas + 1))


class HRDB(nn.Module):
    """High-resolution detail branch: detail logit at the input size, gated by the SCB outputs."""

    def __init__(self, in_channels: int, scb_channels: list[int], gf_index: tuple[int, ...] = (0, 2, 4)) -> None:
        super().__init__()
        channels = [64, 32, 16, 8]
        self.gf_index = gf_index
        self.gf_list = nn.ModuleList([nn.Conv2d(scb_channels[i], 1, 1) for i in gf_index])
        self.res_list = nn.ModuleList(
            [BasicBlock(in_channels, channels[0], shortcut=False), *(BasicBlock(c, c) for c in channels[1:-1])]
        )
        self.convs = nn.ModuleList([nn.Conv2d(channels[i], channels[i + 1], 1) for i in range(len(channels) - 1)])
        self.gates = nn.ModuleList([GatedSpatialConv2d(c, c) for c in channels[1:]])
        self.detail_conv = nn.Conv2d(channels[-1], 1, 1, bias=False)

    def forward(self, x: torch.Tensor, scb_logits: list[torch.Tensor]) -> torch.Tensor:
        steps = zip(self.res_list, self.convs, self.gf_list, self.gf_index, self.gates, strict=True)
        for res, conv, gf_conv, index, gate in steps:
            x = conv(res(x))
            gf = F.interpolate(gf_conv(scb_logits[index]), x.shape[2:], mode='bilinear', align_corners=False)
            x = gate(x, gf)
        return self.detail_conv(x)


class PPMatting(nn.Module):
    """PP-Matting with the HRNet-W18 backbone (PaddleSeg `ppmatting-hrnet_w18-human_512`)."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = HRNetW18()
        feat_channels = self.backbone.feat_channels  # [64, 18, 36, 72, 144] at strides 2, 4, 8, 16, 32
        self.scb = SCB(feat_channels[-1])
        self.hrdb = HRDB(feat_channels[0] + feat_channels[1], self.scb.out_channels, gf_index=(0, 2, 4))

    @staticmethod
    def fusion(semantic_map: torch.Tensor, detail_map: torch.Tensor) -> torch.Tensor:
        """Combine the (B, 3, H, W) class probabilities with the (B, 1, H, W) detail alpha."""
        index = torch.argmax(semantic_map, dim=1, keepdim=True)
        return detail_map * (index == 1).to(detail_map.dtype) + (index == 0).to(detail_map.dtype)

    def forward_debug(self, img: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the SCB softmax, the HRDB sigmoid and the fused alpha."""
        feats = self.backbone(img)
        scb_logits = self.scb(feats[-1])
        semantic_map = torch.softmax(scb_logits[-1], dim=1)

        size = img.shape[2:]
        hrdb_input = torch.cat(
            [F.interpolate(feat, size, mode='bilinear', align_corners=False) for feat in feats[:2]], dim=1
        )
        detail_map = torch.sigmoid(self.hrdb(hrdb_input, scb_logits))
        alpha = self.fusion(semantic_map, detail_map)
        return {'scb_softmax': semantic_map, 'hrdb_sigmoid': detail_map, 'alpha': alpha}

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.forward_debug(img)['alpha']


# PP-MattingV2 (STDC1)
class MLP(nn.Module):
    """Feed-forward part of a transformer block."""

    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = Conv2dNormActivation(in_features, hidden_features, 1, activation_layer=None)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features)
        self.act = nn.ReLU6()
        self.fc2 = Conv2dNormActivation(hidden_features, in_features, 1, activation_layer=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.dwconv(self.fc1(x))))


class Attention(nn.Module):
    """Multi-head self-attention over the pixels of a feature map."""

    def __init__(self, dim: int, key_dim: int, num_heads: int, attn_ratio: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.d = attn_ratio * key_dim
        self.dh = self.d * num_heads

        self.to_q = Conv2dNormActivation(dim, key_dim * num_heads, 1, activation_layer=None)
        self.to_k = Conv2dNormActivation(dim, key_dim * num_heads, 1, activation_layer=None)
        self.to_v = Conv2dNormActivation(dim, self.dh, 1, activation_layer=None)
        self.proj = nn.Sequential(nn.ReLU6(), Conv2dNormActivation(self.dh, dim, 1, activation_layer=None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        q = self.to_q(x).reshape(b, self.num_heads, self.key_dim, -1).transpose(2, 3)
        k = self.to_k(x).reshape(b, self.num_heads, self.key_dim, -1)
        v = self.to_v(x).reshape(b, self.num_heads, self.d, -1).transpose(2, 3)

        # Paddle defines a scale but never applies it, and the weights were trained that way.
        attn = torch.softmax(torch.matmul(q, k), dim=-1)
        out = torch.matmul(attn, v).transpose(2, 3).reshape(b, self.dh, h, w)
        return self.proj(out)


class Block(nn.Module):
    """Transformer block: residual attention, then a residual MLP."""

    def __init__(self, dim: int, key_dim: int, num_heads: int, mlp_ratio: int, attn_ratio: int) -> None:
        super().__init__()
        self.attn = Attention(dim, key_dim, num_heads, attn_ratio)
        self.mlp = MLP(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return self.mlp(x) + x


class BasicLayer(nn.Module):
    """A stack of transformer blocks."""

    def __init__(
        self, num_blocks: int, dim: int, key_dim: int, num_heads: int, mlp_ratio: int, attn_ratio: int
    ) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [Block(dim, key_dim, num_heads, mlp_ratio, attn_ratio) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.transformer_blocks:
            x = block(x)
        return x


class PyramidPoolAgg(nn.Module):
    """Average-pools every input to the size of the last one, pooled once more by `stride`, and concatenates."""

    def __init__(self, stride: int) -> None:
        super().__init__()
        self.stride = stride

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        kernel_size = self.stride ** len(inputs)
        out = []
        for x in inputs:
            out.append(F.avg_pool2d(x, kernel_size, kernel_size))
            kernel_size //= 2
        return torch.cat(out, dim=1)


class DoublePyramidPoolModule(nn.Module):
    """Global context from two pyramid poolings, the second followed by a transformer block."""

    def __init__(
        self, stride: int, in_channels: int, mid_channels: int, out_channels: int, bin_sizes: tuple[int, ...]
    ) -> None:
        super().__init__()
        self.pp1 = PyramidPoolAgg(stride)
        self.conv_mid = Conv2dNormActivation(in_channels, mid_channels, 1, activation_layer=None, bias=True)
        self.pp2 = nn.ModuleList(
            [
                nn.Sequential(
                    AdaptiveAvgPool2d(size),
                    BasicLayer(1, mid_channels, key_dim=16, num_heads=8, mlp_ratio=2, attn_ratio=2),
                )
                for size in bin_sizes
            ]
        )
        self.conv_out = Conv2dNormActivation(mid_channels, out_channels, 1, bias=True)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        x = self.conv_mid(self.pp1(inputs))
        size = x.shape[2:]
        stages = [F.interpolate(stage(x), size, mode='bilinear', align_corners=False) for stage in self.pp2]
        # Paddle merges with `sum([x, *reversed(stages)])`; keep its order of additions.
        return self.conv_out(sum([x, *stages[::-1]]))


class MLFF(nn.Module):
    """Multi-level feature fusion with a spatial attention map per input."""

    def __init__(self, in_channels: list[int], mid_channels: list[int], out_channels: int) -> None:
        super().__init__()
        self.pwconvs = nn.ModuleList(
            [
                Conv2dNormActivation(i, m, 1, activation_layer=None)
                for i, m in zip(in_channels, mid_channels, strict=True)
            ]
        )
        self.dwconvs = nn.ModuleList([Conv2dNormActivation(m, m, 3, groups=m) for m in mid_channels])
        n = len(in_channels)
        self.conv_atten = nn.Sequential(
            Conv2dNormActivation(2 * n, n, 3), Conv2dNormActivation(n, n, 3, activation_layer=None)
        )
        self.conv_out = Conv2dNormActivation(sum(mid_channels), out_channels, 3)

    def forward(self, inputs: list[torch.Tensor], size: torch.Size) -> torch.Tensor:
        feats = []
        for x, pwconv, dwconv in zip(inputs, self.pwconvs, self.dwconvs, strict=True):
            x = F.interpolate(pwconv(x), size, mode='bilinear', align_corners=False)
            feats.append(dwconv(x))

        # Channel order is (mean_0, max_0, mean_1, max_1, ...).
        stats = [s for f in feats for s in (f.mean(dim=1, keepdim=True), f.amax(dim=1, keepdim=True))]
        atten = torch.sigmoid(self.conv_atten(torch.cat(stats, dim=1)))
        out = torch.cat([f * atten[:, i : i + 1] for i, f in enumerate(feats)], dim=1)
        return self.conv_out(out)


class MattingHead(nn.Module):
    """3x3 conv blocks, then a 1x1 conv to one channel and a sigmoid."""

    def __init__(self, in_channels: int, mid_channels: int, num_convs: int) -> None:
        super().__init__()
        self.conv = Conv2dNormActivation(in_channels, mid_channels, 3)
        self.mid_conv = nn.ModuleList(
            [Conv2dNormActivation(mid_channels, mid_channels, 3) for _ in range(num_convs - 1)]
        )
        self.conv_out = nn.Conv2d(mid_channels, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        for conv in self.mid_conv:
            x = conv(x)
        return torch.sigmoid(self.conv_out(x))


class PPMattingV2(nn.Module):
    """PP-MattingV2 with the STDC1 backbone (PaddleSeg `ppmattingv2-stdc1-human_512`)."""

    def __init__(self) -> None:
        super().__init__()
        decoder_channels = [128, 96, 64, 32, 16]
        dpp_channels = 256

        self.backbone = STDC1()
        feat_channels = self.backbone.feat_channels  # [32, 64, 256, 512, 1024] at strides 2, 4, 8, 16, 32
        self.dpp_index = [1, 2, 3, 4]
        self.dpp = DoublePyramidPoolModule(
            stride=2,
            in_channels=sum(feat_channels[i] for i in self.dpp_index),
            mid_channels=dpp_channels,
            out_channels=dpp_channels,
            bin_sizes=(2, 4, 6),
        )

        c0, c1, c2, c3, c4 = decoder_channels
        self.mlff32x = MLFF([feat_channels[-1], dpp_channels], [dpp_channels, dpp_channels], c0)
        self.mlff16x = MLFF([feat_channels[-2], c0, dpp_channels], [c0, c0, c0], c1)
        self.mlff8x = MLFF([feat_channels[-3], c1, dpp_channels], [c1, c1, c1], c2)
        self.mlff4x = MLFF([feat_channels[-4], c2, 3], [c2, c2, 3], c3)
        self.mlff2x = MLFF([feat_channels[-5], c3, 3], [c3, c3, 3], c4)
        self.matting_head_mlff2x = MattingHead(c4 + 3, mid_channels=8, num_convs=2)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(img)
        dpp_out = self.dpp([feats[i] for i in self.dpp_index])

        x = self.mlff32x([feats[-1], dpp_out], feats[-1].shape[-2:])
        x = self.mlff16x([feats[-2], x, dpp_out], feats[-2].shape[-2:])
        x = self.mlff8x([feats[-3], x, dpp_out], feats[-3].shape[-2:])
        # Paddle's `mode='area'` resize of the image is plain average pooling at these sizes.
        x = self.mlff4x([feats[-4], x, F.avg_pool2d(img, 4)], feats[-4].shape[-2:])
        x = self.mlff2x([feats[-5], x, F.avg_pool2d(img, 2)], feats[-5].shape[-2:])

        x = F.interpolate(x, img.shape[-2:], mode='bilinear', align_corners=False)
        return self.matting_head_mlff2x(torch.cat([x, img], dim=1))
