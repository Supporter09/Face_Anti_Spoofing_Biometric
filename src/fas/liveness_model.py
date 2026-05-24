from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class TorchLivenessModel:
    model_path: str | None = None
    input_size: int = 112

    def __post_init__(self) -> None:
        self._model = None
        self._torch = None
        self._unavailable_reason: str | None = None
        self._imagenet_norm: bool = self._read_imagenet_norm_flag()
        self._load_model()

    def _read_imagenet_norm_flag(self) -> bool:
        if not self.model_path:
            return True
        summary = Path(self.model_path).parent / 'run_summary.json'
        if not summary.exists():
            return False
        try:
            data = json.loads(summary.read_text())
            return data.get('preprocessing', 'div255_only') == 'imagenet_norm'
        except Exception:
            return False

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

    def predict_live_score_debug(self, face_crop_bgr: np.ndarray) -> tuple[float, dict]:
        """Same as predict_live_score but also returns intermediate debug info."""
        if self._model is None or self._torch is None:
            return 0.5, {'error': 'model_not_loaded'}

        try:
            import cv2  # type: ignore
        except ImportError:
            return 0.5, {'error': 'opencv_not_installed'}

        # Step 1: resize to model input size (before norm)
        resized = cv2.resize(face_crop_bgr, (self.input_size, self.input_size))
        rgb_112 = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        arr = rgb_112.astype(np.float32) / 255.0

        pixel_stats_before_norm = {
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'min': float(arr.min()),
            'max': float(arr.max()),
            'per_channel_mean': arr.reshape(-1, 3).mean(axis=0).tolist(),
        }

        # Step 2: apply norm (or not)
        if self._imagenet_norm:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr_normed = (arr - mean) / std
        else:
            arr_normed = arr

        pixel_stats_after_norm = {
            'mean': float(arr_normed.mean()),
            'std': float(arr_normed.std()),
            'min': float(arr_normed.min()),
            'max': float(arr_normed.max()),
        }

        # Step 3: inference
        chw = np.transpose(arr_normed, (2, 0, 1))
        tensor = self._torch.from_numpy(chw).unsqueeze(0)
        with self._torch.no_grad():
            logits = self._model(tensor)

        logits_list = logits[0].tolist() if logits.ndim == 2 else logits.tolist()
        probs = self._torch.softmax(logits if logits.ndim == 2 else logits.unsqueeze(0), dim=1)[0].tolist()
        score = max(0.0, min(1.0, self._extract_live_score(logits)))

        debug = {
            'imagenet_norm_applied': self._imagenet_norm,
            'input_size': self.input_size,
            'logits': logits_list,
            'probs': probs,
            'p_spoof': probs[0] if len(probs) >= 2 else None,
            'p_live': probs[1] if len(probs) >= 2 else probs[0],
            'live_score_final': score,
            'pixel_stats_before_norm': pixel_stats_before_norm,
            'pixel_stats_after_norm': pixel_stats_after_norm,
            # Carry arrays for image saving (not serializable, stripped before JSON dump)
            '_rgb_112': rgb_112,
            '_arr_normed': arr_normed,
        }
        return score, debug

    def _preprocess(self, face_crop_bgr: np.ndarray):
        torch = self._torch
        assert torch is not None

        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError('OpenCV is required for liveness preprocessing.') from exc

        resized = cv2.resize(face_crop_bgr, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        arr = rgb.astype(np.float32) / 255.0
        if self._imagenet_norm:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr = (arr - mean) / std
        chw = np.transpose(arr, (2, 0, 1))
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
