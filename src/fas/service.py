from __future__ import annotations

import os
from time import perf_counter

from fas.detection import InsightFaceDetector
from fas.liveness_model import TorchLivenessModel
from fas.preprocess import decode_base64_image_to_bgr
from fas.schemas import LivenessInferRequest, LivenessInferResponse
from fas.types import FaceDetector, LivenessModel


class LivenessService:
    def __init__(
        self,
        *,
        detector: FaceDetector | None = None,
        liveness_model: LivenessModel | None = None,
        threshold_live: float = 0.9,
        threshold_spoof: float = 0.3,
    ) -> None:
        model_path = os.environ.get('LIVENESS_MODEL_PATH')
        self.detector = detector or InsightFaceDetector()
        self.liveness_model = liveness_model or TorchLivenessModel(model_path=model_path)
        self.threshold_live = threshold_live
        self.threshold_spoof = threshold_spoof

    def infer(self, request: LivenessInferRequest) -> LivenessInferResponse:
        started_at = perf_counter()

        if not request.image_base64:
            return self._build_response(
                face_detected=False,
                score=0.0,
                label='no_face',
                started_at=started_at,
                message='No image payload was provided.',
            )

        decoded = decode_base64_image_to_bgr(request.image_base64)
        if decoded.image_bgr is None:
            return self._build_response(
                face_detected=False,
                score=0.0,
                label='no_face',
                started_at=started_at,
                message=decoded.error or 'Could not decode image payload.',
            )

        detection = self.detector.detect(decoded.image_bgr)
        if detection is None:
            detector_reason = getattr(self.detector, 'unavailable_reason', None)
            message = detector_reason or 'No face was detected in the frame.'
            return self._build_response(
                face_detected=False,
                score=0.0,
                label='no_face',
                started_at=started_at,
                message=message,
            )

        live_score = self.liveness_model.predict_live_score(detection.aligned_crop_bgr)
        label = self._label_from_score(live_score)

        if self.liveness_model.is_ready:
            message = 'Face detected and liveness score computed by loaded model.'
        else:
            message = self.liveness_model.unavailable_reason or 'Liveness model is running in fallback mode.'

        return self._build_response(
            face_detected=True,
            score=live_score,
            label=label,
            started_at=started_at,
            message=message,
            face_bbox_xyxy=list(detection.bbox_xyxy),
            face_landmarks=[[x, y] for x, y in detection.landmarks],
        )

    def _label_from_score(self, live_score: float) -> str:
        if live_score >= self.threshold_live:
            return 'live'
        if live_score <= self.threshold_spoof:
            return 'spoof'
        return 'uncertain'

    def _build_response(
        self,
        *,
        face_detected: bool,
        score: float,
        label: str,
        started_at: float,
        message: str,
        face_bbox_xyxy: list[int] | None = None,
        face_landmarks: list[list[float]] | None = None,
    ) -> LivenessInferResponse:
        return LivenessInferResponse(
            face_detected=face_detected,
            liveness_score=score,
            liveness_label=label,
            latency_ms=(perf_counter() - started_at) * 1000.0,
            message=message,
            face_bbox_xyxy=face_bbox_xyxy,
            face_landmarks=face_landmarks,
        )
