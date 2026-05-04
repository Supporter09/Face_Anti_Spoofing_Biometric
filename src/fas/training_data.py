from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from random import Random

import numpy as np

from fas.preprocess import clamp_bbox_to_image, resize_bgr_image


@dataclass
class LivenessSample:
    image_path: str
    label: int
    bbox_xyxy: tuple[int, int, int, int] | None = None


def load_manifest(manifest_path: str) -> list[LivenessSample]:
    payload = json.loads(Path(manifest_path).read_text())
    samples: list[LivenessSample] = []

    for row in payload:
        bbox = row.get('bbox_xyxy')
        bbox_tuple = tuple(bbox) if bbox else None
        samples.append(
            LivenessSample(
                image_path=row['image_path'],
                label=int(row['label']),
                bbox_xyxy=bbox_tuple,
            )
        )

    return samples


def split_samples(
    samples: list[LivenessSample], train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42
) -> tuple[list[LivenessSample], list[LivenessSample], list[LivenessSample]]:
    shuffled = list(samples)
    Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def crop_face_for_training(image_bgr: np.ndarray, bbox_xyxy: tuple[int, int, int, int] | None, image_size: int = 80) -> np.ndarray:
    if bbox_xyxy is None:
        cropped = image_bgr
    else:
        x1, y1, x2, y2 = clamp_bbox_to_image(
            bbox_xyxy,
            image_height=image_bgr.shape[0],
            image_width=image_bgr.shape[1],
        )
        cropped = image_bgr[y1:y2, x1:x2]

    return resize_bgr_image(cropped, image_size=image_size)


def to_chw_float(image_bgr: np.ndarray) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError('OpenCV is required for training preprocessing.') from exc

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    return np.transpose(normalized, (2, 0, 1))
