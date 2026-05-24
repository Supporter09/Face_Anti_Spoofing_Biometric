from __future__ import annotations

import os
from time import perf_counter

from fas.detection import InsightFaceDetector
from fas.liveness_model import TorchLivenessModel
from fas.pose import estimate_head_pose
from fas.preprocess import decode_base64_image_to_bgr
from fas.schemas import LivenessInferRequest, LivenessInferResponse
from fas.types import FaceDetector, LivenessModel


class LivenessService:
    def __init__(
        self,
        *,
        detector: FaceDetector | None = None,
        liveness_model: LivenessModel | None = None,
        threshold_live: float | None = None,
        threshold_spoof: float | None = None,
    ) -> None:
        model_path = os.environ.get('LIVENESS_MODEL_PATH')
        self.detector = detector or InsightFaceDetector()
        self.liveness_model = liveness_model or TorchLivenessModel(model_path=model_path)

        resolved_live = (
            threshold_live
            if threshold_live is not None
            else self._read_threshold_from_env('LIVENESS_LIVE_THRESHOLD', default=0.9)
        )
        resolved_spoof = (
            threshold_spoof
            if threshold_spoof is not None
            else self._read_threshold_from_env('LIVENESS_SPOOF_THRESHOLD', default=0.3)
        )

        self.threshold_live, self.threshold_spoof = self._normalize_threshold_pair(
            threshold_live=resolved_live,
            threshold_spoof=resolved_spoof,
        )

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

        crop_for_model = (
            detection.context_crop_bgr
            if detection.context_crop_bgr is not None
            else detection.aligned_crop_bgr
        )

        live_score = self.liveness_model.predict_live_score(crop_for_model)
        label = self._label_from_score(live_score)

        pose = None
        if len(detection.landmarks) == 5:
            try:
                pose = estimate_head_pose(detection.landmarks, decoded.image_bgr.shape)
            except Exception:
                pose = None

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
            pose=pose,
        )

    def _label_from_score(self, live_score: float) -> str:
        if live_score >= self.threshold_live:
            return 'live'
        if live_score <= self.threshold_spoof:
            return 'spoof'
        return 'uncertain'

    @staticmethod
    def _read_threshold_from_env(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            value = float(raw)
        except ValueError:
            return default
        return max(0.0, min(1.0, value))

    @staticmethod
    def _normalize_threshold_pair(*, threshold_live: float, threshold_spoof: float) -> tuple[float, float]:
        # Ensure a valid uncertain band exists between spoof and live decisions.
        if threshold_spoof >= threshold_live:
            return 0.9, 0.3
        return threshold_live, threshold_spoof

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
        pose: dict[str, float] | None = None,
    ) -> LivenessInferResponse:
        return LivenessInferResponse(
            face_detected=face_detected,
            liveness_score=score,
            liveness_label=label,
            latency_ms=(perf_counter() - started_at) * 1000.0,
            message=message,
            face_bbox_xyxy=face_bbox_xyxy,
            face_landmarks=face_landmarks,
            yaw_deg=pose['yaw_deg'] if pose and pose.get('ok') else None,
            pitch_deg=pose['pitch_deg'] if pose and pose.get('ok') else None,
            pose_ok=bool(pose and pose.get('ok')),
        )
