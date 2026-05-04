import numpy as np

from fas.schemas import LivenessInferRequest
from fas.service import LivenessService
from fas.types import FaceDetection


class FakeDetector:
    unavailable_reason: str | None = None

    def detect(self, image_bgr: np.ndarray) -> FaceDetection | None:
        assert image_bgr.shape[0] == 8
        return FaceDetection(
            bbox_xyxy=(1, 1, 7, 7),
            landmarks=[(2.0, 2.0), (6.0, 2.0), (4.0, 4.0), (2.0, 6.0), (6.0, 6.0)],
            aligned_crop_bgr=np.zeros((80, 80, 3), dtype=np.uint8),
        )


class FakeLivenessModel:
    is_ready = True
    unavailable_reason = None

    def __init__(self, score: float) -> None:
        self.score = score

    def predict_live_score(self, face_crop_bgr: np.ndarray) -> float:
        assert face_crop_bgr.shape[:2] == (80, 80)
        return self.score


class DecodeStub:
    def __init__(self, image_bgr: np.ndarray, error: str | None = None) -> None:
        self.image_bgr = image_bgr
        self.error = error


def test_service_reports_live_with_bbox_and_landmarks(monkeypatch) -> None:
    from fas import service as service_module

    monkeypatch.setattr(
        service_module,
        'decode_base64_image_to_bgr',
        lambda _: DecodeStub(image_bgr=np.zeros((8, 8, 3), dtype=np.uint8), error=None),
    )

    service = LivenessService(
        detector=FakeDetector(),
        liveness_model=FakeLivenessModel(score=0.95),
        threshold_live=0.9,
        threshold_spoof=0.2,
    )

    response = service.infer(LivenessInferRequest(image_base64='valid'))

    assert response.face_detected is True
    assert response.liveness_label == 'live'
    assert response.face_bbox_xyxy == [1, 1, 7, 7]
    assert len(response.face_landmarks or []) == 5


def test_service_reports_spoof_when_score_is_low(monkeypatch) -> None:
    from fas import service as service_module

    monkeypatch.setattr(
        service_module,
        'decode_base64_image_to_bgr',
        lambda _: DecodeStub(image_bgr=np.zeros((8, 8, 3), dtype=np.uint8), error=None),
    )

    service = LivenessService(
        detector=FakeDetector(),
        liveness_model=FakeLivenessModel(score=0.1),
        threshold_live=0.9,
        threshold_spoof=0.2,
    )

    response = service.infer(LivenessInferRequest(image_base64='valid'))

    assert response.face_detected is True
    assert response.liveness_label == 'spoof'
