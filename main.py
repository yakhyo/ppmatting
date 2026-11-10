# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""PP-Matting PyTorch inference (image, video, or webcam).

Examples:
    python main.py --source assets/inputs/src_img1.jpg --output results/
    python main.py --source video.mp4 --model ppmatting_v1 --background bg.jpg --output out.mp4
    python main.py --source 0
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch

from model import load_model, postprocess, preprocess

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
GREEN = (0, 177, 64)  # BGR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='PP-Matting portrait matting (image, video, or webcam).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--source', required=True, help='image path, video path, or camera index (e.g. 0)')
    parser.add_argument('--model', default='ppmatting_v2', choices=['ppmatting_v1', 'ppmatting_v2'])
    parser.add_argument('--weights', default=None, help='path to .pt checkpoint (default: weights/<model>.pt)')
    parser.add_argument('--background', default=None, help='background image for compositing (default: green)')
    parser.add_argument('--output', default=None, help='output directory (image) or video path (video/webcam)')
    return parser.parse_args()


def compose(image: np.ndarray, alpha: np.ndarray, background: np.ndarray | None) -> np.ndarray:
    """Composite `image` over `background` (resized to it) or plain green, weighted by the matte."""
    h, w = image.shape[:2]
    bg = np.full_like(image, GREEN) if background is None else cv2.resize(background, (w, h))
    a = alpha[:, :, None]
    return (image * a + bg * (1 - a)).round().astype(np.uint8)


def run_image(args: argparse.Namespace, predict: Callable, background: np.ndarray | None) -> None:
    src = Path(args.source)
    image = cv2.imread(str(src))
    if image is None:
        raise SystemExit(f'Could not read image: {src}')

    alpha = predict(image)
    matte = (alpha * 255).round().astype(np.uint8)
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = matte
    outputs = {'matte': matte, 'green': compose(image, alpha, None), 'rgba': rgba}
    if background is not None:
        outputs['bg'] = compose(image, alpha, background)

    out_dir = Path(args.output) if args.output else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, result in outputs.items():
        out = out_dir / f'{src.stem}_{name}.png'
        cv2.imwrite(str(out), result)
        print(f'Saved: {out}')


def run_stream(args: argparse.Namespace, predict: Callable, background: np.ndarray | None, *, live: bool) -> None:
    cap = cv2.VideoCapture(int(args.source) if live else args.source)
    if not cap.isOpened():
        raise SystemExit(f'Could not open source: {args.source}')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Video files always produce an output; webcams only if --output is set.
    writer = None
    if args.output or not live:
        src = Path(args.source)
        out_path = Path(args.output) if args.output else src.with_name(f'{src.stem}_out.mp4')
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    window = 'PP-Matting (q to quit)'
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = compose(frame, predict(frame), background)
            if live:
                cv2.imshow(window, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if writer is not None:
                writer.write(frame)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f'Saved: {out_path}')
        if live:
            cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    weights = args.weights or f'weights/{args.model}.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.model, weights).to(device)
    background = cv2.imread(args.background) if args.background else None

    def predict(image: np.ndarray) -> np.ndarray:
        x, size = preprocess(image)
        with torch.inference_mode():
            alpha = model(torch.from_numpy(x).to(device))
        return postprocess(alpha.cpu().numpy(), size)

    suffix = Path(args.source).suffix.lower()
    if args.source.isdigit():
        run_stream(args, predict, background, live=True)
    elif suffix in VIDEO_EXTS:
        run_stream(args, predict, background, live=False)
    elif suffix in IMAGE_EXTS:
        run_image(args, predict, background)
    else:
        raise SystemExit(f'Unrecognized source: {args.source}. Expected camera index, image, or video.')


if __name__ == '__main__':
    main()
