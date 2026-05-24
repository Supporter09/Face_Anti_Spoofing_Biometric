from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter, time

import numpy as np

from fas.detection import InsightFaceDetector
from fas.liveness_model import TorchLivenessModel
from fas.pose import estimate_head_pose
from fas.preprocess import decode_base64_image_to_bgr
from fas.schemas import LivenessInferRequest, LivenessInferResponse
from fas.types import FaceDetector, LivenessModel

_DEBUG_MAX_FRAMES = 300


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

        debug_dir_env = os.environ.get('LIVENESS_DEBUG_DIR')
        if debug_dir_env:
            self._debug_dir: Path | None = Path(debug_dir_env)
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            print(f'[DEBUG] Frame capture enabled → {self._debug_dir}')
        else:
            self._debug_dir = None
        self._debug_frame_count = 0

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

        if self._debug_dir is not None and self._debug_frame_count < _DEBUG_MAX_FRAMES:
            live_score, debug_info = self.liveness_model.predict_live_score_debug(crop_for_model)
            self._save_debug_frame(decoded.image_bgr, detection, live_score, debug_info)
        else:
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

    def _save_debug_frame(self, raw_bgr: np.ndarray, detection, live_score: float, debug_info: dict) -> None:
        try:
            import cv2  # type: ignore
        except ImportError:
            return

        assert self._debug_dir is not None
        self._debug_frame_count += 1
        n = self._debug_frame_count
        ts = int(time() * 1000)
        prefix = self._debug_dir / f'{ts}_{n:04d}'
        label = self._label_from_score(live_score)

        # 1. Raw frame với bbox và score vẽ lên
        vis = raw_bgr.copy()
        x1, y1, x2, y2 = [int(v) for v in detection.bbox_xyxy]
        color = (0, 200, 0) if label == 'live' else (0, 0, 220) if label == 'spoof' else (0, 165, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f'{label} {live_score:.3f}', (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(str(prefix) + '_1_raw.jpg', vis)

        # 2. Face crop từ detector (80x80 hoặc size thực tế)
        cv2.imwrite(str(prefix) + '_2_crop_detector.jpg', detection.aligned_crop_bgr)

        # 3. Model input 112x112 trước norm (uint8 RGB→BGR để lưu)
        rgb_112 = debug_info.pop('_rgb_112', None)
        if rgb_112 is not None:
            cv2.imwrite(str(prefix) + '_3_model_input_112.jpg', cv2.cvtColor(rgb_112, cv2.COLOR_RGB2BGR))

        # 4. Normalized array visualized (clip về [0,1], scale lên 255)
        arr_normed = debug_info.pop('_arr_normed', None)
        if arr_normed is not None:
            vis_norm = np.clip(arr_normed, 0, 1) if not debug_info.get('imagenet_norm_applied') \
                else np.clip((arr_normed - arr_normed.min()) / max(arr_normed.max() - arr_normed.min(), 1e-5), 0, 1)
            vis_norm_bgr = cv2.cvtColor((vis_norm * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(prefix) + '_4_model_input_vis.jpg', vis_norm_bgr)

        # 5. JSON metadata
        meta = {
            'frame': n,
            'timestamp_ms': ts,
            'live_score': live_score,
            'label': label,
            'threshold_live': self.threshold_live,
            'threshold_spoof': self.threshold_spoof,
            'face_bbox_xyxy': list(detection.bbox_xyxy),
            'frame_shape': list(raw_bgr.shape),
            **debug_info,
        }
        (self._debug_dir / f'{ts}_{n:04d}_meta.json').write_text(json.dumps(meta, indent=2))

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
