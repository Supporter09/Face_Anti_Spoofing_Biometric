from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fas.preprocess import expand_bbox, resize_bgr_image, pad_to_square_then_resize
from fas.types import FaceDetection


@dataclass
class InsightFaceDetector:
    det_size: tuple[int, int] = (640, 640)
    max_num_faces: int = 1
    margin_ratio: float = 0.2
    context_margin_ratio: float = 0.8   # bbox * 1.8 around face for context crop

    def __post_init__(self) -> None:
        self._app = None
        self._unavailable_reason: str | None = None

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    def _ensure_initialized(self) -> bool:
        if self._app is not None:
            return True

        try:
            from insightface.app import FaceAnalysis  # type: ignore
        except ImportError:
            self._unavailable_reason = (
                'insightface is not installed. Install optional ML dependencies first.'
            )
            return False

        try:
            app = FaceAnalysis(allowed_modules=['detection'])
            app.prepare(ctx_id=0, det_size=self.det_size)
        except Exception:
            app = FaceAnalysis(allowed_modules=['detection'])
            app.prepare(ctx_id=-1, det_size=self.det_size)

        self._app = app
        self._unavailable_reason = None
        return True

    def detect(self, image_bgr: np.ndarray) -> FaceDetection | None:
        if not self._ensure_initialized():
            return None

        assert self._app is not None
        faces = self._app.get(image_bgr, max_num=self.max_num_faces)
        if not faces:
            return None

        face = max(faces, key=lambda current: float((current.bbox[2] - current.bbox[0]) * (current.bbox[3] - current.bbox[1])))

        x1_raw, y1_raw, x2_raw, y2_raw = [int(value) for value in face.bbox]

        # Face-tight crop (legacy, 80x80)
        fx1, fy1, fx2, fy2 = expand_bbox(
            (x1_raw, y1_raw, x2_raw, y2_raw),
            image_height=image_bgr.shape[0],
            image_width=image_bgr.shape[1],
            margin_ratio=self.margin_ratio,
        )
        face_tight = image_bgr[fy1:fy2, fx1:fx2]
        aligned_crop = resize_bgr_image(face_tight, image_size=80)

        # Context crop (bbox * 1.8, padded square, NOT yet resized to model input)
        cx1, cy1, cx2, cy2 = expand_bbox(
            (x1_raw, y1_raw, x2_raw, y2_raw),
            image_height=image_bgr.shape[0],
            image_width=image_bgr.shape[1],
            margin_ratio=self.context_margin_ratio,
        )
        context_raw = image_bgr[cy1:cy2, cx1:cx2]
        # Pad to square at the raw crop's max-side size, no downsample yet.
        side = max(context_raw.shape[0], context_raw.shape[1])
        context_padded = pad_to_square_then_resize(context_raw, size=side)

        landmarks: list[tuple[float, float]] = []
        if hasattr(face, 'kps') and face.kps is not None:
            landmarks = [(float(point[0]), float(point[1])) for point in face.kps.tolist()]

        return FaceDetection(
            bbox_xyxy=(fx1, fy1, fx2, fy2),
            landmarks=landmarks,
            aligned_crop_bgr=aligned_crop,
            context_crop_bgr=context_padded,
        )
