# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Export a PP-Matting `.pt` checkpoint to ONNX with dynamic batch, height and width.

Examples:
    python onnx_export.py --model ppmatting_v2 --simplify
    python onnx_export.py --model ppmatting_v1 --simplify
"""

from __future__ import annotations

import argparse
import os

import torch

from model import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export PP-Matting PyTorch checkpoint to ONNX')
    parser.add_argument('--model', required=True, choices=['ppmatting_v1', 'ppmatting_v2'], help='model to export')
    parser.add_argument('--weights', default=None, help='path to .pt checkpoint (default: weights/<model>.pt)')
    parser.add_argument('--output', default=None, help='output .onnx file (default: weights/<model>.onnx)')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', help='run onnxsim on the exported file')
    return parser.parse_args()


def export(model: torch.nn.Module, output: str, opset: int) -> None:
    """Write `model` to `output` as a graph taking `img` (B, 3, H, W) in [-1, 1] and returning `alpha` (B, 1, H, W)."""
    dynamic = {0: 'batch', 2: 'height', 3: 'width'}
    # The pooling in model/common.py keeps dynamic sizes only with the TorchScript exporter.
    torch.onnx.export(
        model,
        (torch.randn(1, 3, 480, 672),),
        output,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['img'],
        output_names=['alpha'],
        dynamic_axes={'img': dynamic, 'alpha': dynamic},
        dynamo=False,
    )


def simplify(path: str) -> None:
    """Simplify the ONNX file at `path` in place with onnxsim."""
    import onnx
    import onnxsim

    model, ok = onnxsim.simplify(onnx.load(path))
    if not ok:
        raise RuntimeError(f'onnxsim failed for {path}')
    onnx.save(model, path)


def main() -> None:
    args = parse_args()
    args.weights = args.weights or f'weights/{args.model}.pt'
    args.output = args.output or f'weights/{args.model}.onnx'
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    print(f'Loading PyTorch model: {args.weights}')
    model = load_model(args.model, args.weights)

    print(f'Exporting to ONNX (opset {args.opset}): {args.output}')
    export(model, args.output, args.opset)

    if args.simplify:
        print('Simplifying with onnxsim')
        simplify(args.output)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f'Done. {args.output} ({size_mb:.1f} MiB)')


if __name__ == '__main__':
    main()
