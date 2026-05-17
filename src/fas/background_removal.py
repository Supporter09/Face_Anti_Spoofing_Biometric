"""Background removal with model caching - class-based."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BackgroundRemover:
    model_type: str = "mediapipe"
    model_path: str = "selfie_segmentation.tflite"

    def __post_init__(self) -> None:
        self._segmenter = None
        self._rembg_session = None
        self._unavailable_reason: str | None = None
        self._load_model()

    @property
    def is_ready(self) -> bool:
        return self._segmenter is not None or self._rembg_session is not None

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    def _load_model(self) -> None:
        if self.model_type == "mediapipe":
            self._load_mediapipe()
        elif self.model_type == "rembg":
            self._load_rembg()

    def _load_mediapipe(self) -> None:
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            self._unavailable_reason = f"mediapipe not installed: {exc}"
            return

        model_file = Path(self.model_path)
        if not model_file.exists():
            self._unavailable_reason = f"Model file not found: {self.model_path}"
            return

        try:
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.ImageSegmenterOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                output_category_mask=True
            )
            self._segmenter = vision.ImageSegmenter.create_from_options(options)
            self._unavailable_reason = None
            print("[BackgroundRemover] MediaPipe loaded on init")
        except Exception as exc:
            self._unavailable_reason = f"Failed to load MediaPipe: {exc}"

    def _load_rembg(self) -> None:
        try:
            from rembg import new_session
        except ImportError as exc:
            self._unavailable_reason = f"rembg not installed: {exc}"
            return

        try:
            self._rembg_session = new_session(model_name="u2net")
            self._unavailable_reason = None
            print("[BackgroundRemover] rembg loaded on init")
        except Exception as exc:
            self._unavailable_reason = f"Failed to load rembg: {exc}"

    def remove_background(self, image_bgr: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            return image_bgr

        if self.model_type == "mediapipe":
            return self._remove_background_mediapipe(image_bgr)
        elif self.model_type == "rembg":
            return self._remove_background_rembg(image_bgr)

        return image_bgr

    def _remove_background_mediapipe(self, image_bgr: np.ndarray) -> np.ndarray:
        import cv2
        import mediapipe as mp

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        segmentation_result = self._segmenter.segment(mp_image)
        mask = segmentation_result.category_mask

        if mask is None:
            return image_bgr

        mask_np = mask.numpy_view()
        if mask_np.max() <= 1:
            mask_np = mask_np * 255.0

        mask_uint8 = mask_np.astype(np.uint8)

        unique = np.unique(mask_uint8)
        if len(unique) == 2:
            mask_uint8 = 255 - mask_uint8

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=1)

        mask_3ch = cv2.merge([mask_uint8, mask_uint8, mask_uint8])
        # [0, 0, 0]   -> black
        # [0, 255, 0] -> green
        # [255, 0, 0] -> red
        # [0, 0, 255] -> blue
        # [255, 255, 255] -> white
        background_color = np.array([0, 0, 0], dtype=np.uint8)
        result = np.where(mask_3ch > 0, image_bgr, background_color)

        return result

    def _remove_background_rembg(self, image_bgr: np.ndarray) -> np.ndarray:
        import cv2
        from PIL import Image

        h, w = image_bgr.shape[:2]
        image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        output = self._rembg_session.predict(image_pil)

        mask = np.array(output)

        if mask.ndim == 2:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = (mask[:, :, 3] * 255).astype(np.uint8)

        mask_uint8 = cv2.resize(mask_uint8, (w, h))
        mask_uint8 = 255 - mask_uint8

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=2)

        mask_3ch = cv2.merge([mask_uint8, mask_uint8, mask_uint8])
        # [0, 0, 0]   -> black
        # [0, 255, 0] -> green
        # [255, 0, 0] -> red
        # [0, 0, 255] -> blue
        # [255, 255, 255] -> white
        background_color = np.array([0, 0, 0], dtype=np.uint8)
        result = np.where(mask_3ch > 0, image_bgr, background_color)

        return result


def remove_background_mediapipe(image_bgr: np.ndarray, model_path: str = "selfie_segmentation.tflite") -> np.ndarray:
    """Legacy function - use BackgroundRemover class instead."""
    remover = BackgroundRemover(model_type="mediapipe", model_path=model_path)
    return remover.remove_background(image_bgr)


def remove_background_rembg(image_bgr: np.ndarray) -> np.ndarray:
    """Legacy function - use BackgroundRemover class instead."""
    remover = BackgroundRemover(model_type="rembg")
    return remover.remove_background(image_bgr)


remove_background_combined = remove_background_mediapipe
remove_background_skin_color = remove_background_rembg