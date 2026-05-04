from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TorchLivenessModel:
    model_path: str | None = None
    input_size: int = 80

    def __post_init__(self) -> None:
        self._model = None
        self._torch = None
        self._unavailable_reason: str | None = None
        self._load_model()

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    def _load_model(self) -> None:
        try:
            import torch
        except ImportError:
            self._unavailable_reason = 'PyTorch is not installed. Install optional ML dependencies first.'
            return

        self._torch = torch

        if not self.model_path:
            self._unavailable_reason = (
                'No liveness model path configured. Set LIVENESS_MODEL_PATH to a torchscript file.'
            )
            return

        checkpoint = Path(self.model_path)
        if not checkpoint.exists():
            self._unavailable_reason = f'Model checkpoint not found: {checkpoint}'
            return

        try:
            model = torch.jit.load(str(checkpoint), map_location='cpu')
            model.eval()
        except Exception as exc:
            self._unavailable_reason = f'Could not load TorchScript liveness model: {exc}'
            return

        self._model = model
        self._unavailable_reason = None

    def predict_live_score(self, face_crop_bgr: np.ndarray) -> float:
        if self._model is None or self._torch is None:
            return 0.5

        tensor = self._preprocess(face_crop_bgr)
        with self._torch.no_grad():
            logits = self._model(tensor)

        score = self._extract_live_score(logits)
        return max(0.0, min(1.0, score))

    def _preprocess(self, face_crop_bgr: np.ndarray):
        torch = self._torch
        assert torch is not None

        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError('OpenCV is required for liveness preprocessing.') from exc

        resized = cv2.resize(face_crop_bgr, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        tensor = torch.from_numpy(chw).unsqueeze(0)
        return tensor

    def _extract_live_score(self, logits) -> float:
        torch = self._torch
        assert torch is not None

        if logits.ndim == 2 and logits.shape[1] >= 2:
            probabilities = torch.softmax(logits, dim=1)
            return float(probabilities[0, 1].item())

        if logits.ndim == 2 and logits.shape[1] == 1:
            probabilities = torch.sigmoid(logits)
            return float(probabilities[0, 0].item())

        if logits.ndim == 1 and logits.shape[0] >= 2:
            probabilities = torch.softmax(logits, dim=0)
            return float(probabilities[1].item())

        if logits.ndim == 1 and logits.shape[0] == 1:
            probabilities = torch.sigmoid(logits)
            return float(probabilities[0].item())

        return 0.5
