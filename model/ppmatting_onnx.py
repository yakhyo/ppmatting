# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import cv2
import numpy as np
import onnxruntime as ort

__all__ = ['PPMattingONNX', 'postprocess', 'preprocess']


def preprocess(image_bgr: np.ndarray, max_short: int = 512, mult: int = 32) -> tuple[np.ndarray, tuple[int, int]]:
    """Resize and normalize a BGR image as PaddleSeg's val transforms do.

    Args:
        image_bgr: HxWx3 uint8 image in BGR order.
        max_short: The short side is shrunk to this length when it is longer. Images are never upscaled.
        mult: Both sides are floored to a multiple of this value.

    Returns:
        The (1, 3, H, W) float32 tensor in [-1, 1] and the original (height, width).

    Raises:
        ValueError: If either side of the image is shorter than `mult`.
    """
    h, w = image_bgr.shape[:2]
    if h < mult or w < mult:
        raise ValueError(f'image must be at least {mult}x{mult}, got {h}x{w}')

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # LimitShort: shrink the short side to max_short, rounding the other side like PaddleSeg's resize_short.
    short = min(h, w)
    if short > max_short:
        scale = float(max_short) / float(short)
        image = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR)

    # ResizeToIntMult: floor both sides to a multiple of mult.
    rh, rw = image.shape[:2]
    rh, rw = rh - rh % mult, rw - rw % mult
    if (rh, rw) != image.shape[:2]:
        image = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)

    # Normalize with mean = std = 0.5, in float32.
    x = image.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return np.ascontiguousarray(x.transpose(2, 0, 1)[np.newaxis]), (h, w)


def postprocess(alpha: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
    """Resize a predicted alpha matte back to the original image size.

    Args:
        alpha: The matte, shaped (1, 1, h, w) or (h, w).
        original_size: The (height, width) returned by `preprocess`.

    Returns:
        The (height, width) float32 matte.
    """
    if alpha.ndim == 4:
        alpha = alpha[0, 0]
    h, w = original_size
    return cv2.resize(alpha.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


class PPMattingONNX:
    """PP-Matting portrait matting (ONNX Runtime)."""

    def __init__(self, model_path: str, providers: list[str] | None = None) -> None:
        """Load a PP-Matting ONNX model.

        Args:
            model_path: Path to the ONNX file.
            providers: ONNX Runtime execution providers. Defaults to the CPU provider.
        """
        self.session = ort.InferenceSession(model_path, providers=providers or ['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """Predict the alpha matte of a BGR image.

        Args:
            image_bgr: HxWx3 uint8 image in BGR order, at least 32x32.

        Returns:
            The (H, W) float32 matte in [0, 1], at the size of the input image.
        """
        x, original_size = preprocess(image_bgr)
        (alpha,) = self.session.run([self.output_name], {self.input_name: x})
        return postprocess(alpha, original_size)
