from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FaceRecognitionModel:
    det_size: tuple[int, int] = (640, 640)
    max_num_faces: int = 1

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
                "insightface is not installed. Please install insightface first."
            )
            return False

        try:
            # buffalo_l includes detection + recognition models.
            app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
            app.prepare(ctx_id=0, det_size=self.det_size)
        except Exception:
            # fallback CPU
            app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
            app.prepare(ctx_id=-1, det_size=self.det_size)

        self._app = app
        self._unavailable_reason = None
        return True

    def get_embedding(self, image_bgr: np.ndarray) -> np.ndarray | None:
        if not self._ensure_initialized():
            return None

        assert self._app is not None

        faces = self._app.get(image_bgr, max_num=self.max_num_faces)
        if not faces:
            return None

        face = max(
            faces,
            key=lambda current: float(
                (current.bbox[2] - current.bbox[0])
                * (current.bbox[3] - current.bbox[1])
            ),
        )

        embedding = getattr(face, "embedding", None)
        if embedding is None:
            return None

        embedding = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding