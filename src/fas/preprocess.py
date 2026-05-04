from __future__ import annotations

from base64 import b64decode
from dataclasses import dataclass

import numpy as np


@dataclass
class DecodeResult:
    image_bgr: np.ndarray | None
    error: str | None = None


def decode_base64_image_to_bgr(image_base64: str) -> DecodeResult:
    try:
        image_bytes = b64decode(image_base64, validate=True)
    except Exception:
        return DecodeResult(image_bgr=None, error='Image payload is not valid base64.')

    try:
        import cv2  # type: ignore
    except ImportError:
        return DecodeResult(
            image_bgr=None,
            error='OpenCV is not installed. Install optional ML dependencies first.',
        )

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return DecodeResult(image_bgr=None, error='Could not decode image bytes into an RGB frame.')

    return DecodeResult(image_bgr=image_bgr, error=None)


def clamp_bbox_to_image(
    bbox_xyxy: tuple[int, int, int, int], image_height: int, image_width: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(x1 + 1, min(x2, image_width))
    y2 = max(y1 + 1, min(y2, image_height))
    return x1, y1, x2, y2


def expand_bbox(
    bbox_xyxy: tuple[int, int, int, int], image_height: int, image_width: int, margin_ratio: float = 0.15
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    width = x2 - x1
    height = y2 - y1
    margin_x = int(width * margin_ratio)
    margin_y = int(height * margin_ratio)
    expanded = (x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y)
    return clamp_bbox_to_image(expanded, image_height=image_height, image_width=image_width)


def resize_bgr_image(image_bgr: np.ndarray, image_size: int) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError('OpenCV is required for resize operations.') from exc

    return cv2.resize(image_bgr, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
