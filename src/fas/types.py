from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class FaceDetection:
    bbox_xyxy: tuple[int, int, int, int]
    landmarks: list[tuple[float, float]]
    aligned_crop_bgr: np.ndarray


class FaceDetector(Protocol):
    def detect(self, image_bgr: np.ndarray) -> FaceDetection | None:
        ...


class LivenessModel(Protocol):
    @property
    def is_ready(self) -> bool:
        ...

    @property
    def unavailable_reason(self) -> str | None:
        ...

    def predict_live_score(self, face_crop_bgr: np.ndarray) -> float:
        ...
