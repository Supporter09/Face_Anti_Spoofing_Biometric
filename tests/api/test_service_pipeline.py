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
            context_crop_bgr=np.zeros((80, 80, 3), dtype=np.uint8),
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


def test_service_uses_env_thresholds_when_constructor_thresholds_omitted(monkeypatch) -> None:
    from fas import service as service_module

    monkeypatch.setattr(
        service_module,
        'decode_base64_image_to_bgr',
        lambda _: DecodeStub(image_bgr=np.zeros((8, 8, 3), dtype=np.uint8), error=None),
    )
    monkeypatch.setenv('LIVENESS_LIVE_THRESHOLD', '0.8')
    monkeypatch.setenv('LIVENESS_SPOOF_THRESHOLD', '0.1')

    service = LivenessService(
        detector=FakeDetector(),
        liveness_model=FakeLivenessModel(score=0.85),
    )

    response = service.infer(LivenessInferRequest(image_base64='valid'))

    assert response.face_detected is True
    assert response.liveness_label == 'live'


def test_service_falls_back_to_defaults_when_env_threshold_pair_is_invalid(monkeypatch) -> None:
    from fas import service as service_module

    monkeypatch.setattr(
        service_module,
        'decode_base64_image_to_bgr',
        lambda _: DecodeStub(image_bgr=np.zeros((8, 8, 3), dtype=np.uint8), error=None),
    )
    monkeypatch.setenv('LIVENESS_LIVE_THRESHOLD', '0.2')
    monkeypatch.setenv('LIVENESS_SPOOF_THRESHOLD', '0.7')

    service = LivenessService(
        detector=FakeDetector(),
        liveness_model=FakeLivenessModel(score=0.95),
    )

    assert service.threshold_live == 0.9
    assert service.threshold_spoof == 0.3
    response = service.infer(LivenessInferRequest(image_base64='valid'))
    assert response.liveness_label == 'live'


def test_service_explicit_thresholds_override_env(monkeypatch) -> None:
    from fas import service as service_module

    monkeypatch.setattr(
        service_module,
        'decode_base64_image_to_bgr',
        lambda _: DecodeStub(image_bgr=np.zeros((8, 8, 3), dtype=np.uint8), error=None),
    )
    monkeypatch.setenv('LIVENESS_LIVE_THRESHOLD', '0.99')
    monkeypatch.setenv('LIVENESS_SPOOF_THRESHOLD', '0.01')

    service = LivenessService(
        detector=FakeDetector(),
        liveness_model=FakeLivenessModel(score=0.85),
        threshold_live=0.8,
        threshold_spoof=0.1,
    )

    response = service.infer(LivenessInferRequest(image_base64='valid'))

    assert response.face_detected is True
    assert response.liveness_label == 'live'


def test_service_populates_pose_and_uses_context_crop(monkeypatch) -> None:
    from fas import service as service_module

    monkeypatch.setattr(
        service_module,
        'decode_base64_image_to_bgr',
        lambda _: DecodeStub(image_bgr=np.zeros((480, 640, 3), dtype=np.uint8), error=None),
    )

    # Fake detector returning a context_crop_bgr and 5 frontal landmarks
    class FakeDetectorWithContext:
        unavailable_reason = None
        def detect(self, image_bgr):
            return FaceDetection(
                bbox_xyxy=(100, 100, 200, 200),
                landmarks=[(285.0, 220.0), (355.0, 220.0), (320.0, 260.0),
                           (290.0, 300.0), (350.0, 300.0)],   # near-frontal
                aligned_crop_bgr=np.zeros((80, 80, 3), dtype=np.uint8),
                context_crop_bgr=np.zeros((200, 200, 3), dtype=np.uint8),
            )

    # Liveness model expects to receive the context crop (200x200), not the aligned 80x80
    class FakeContextLivenessModel:
        is_ready = True
        unavailable_reason = None
        def predict_live_score(self, face_crop_bgr):
            assert face_crop_bgr.shape[:2] == (200, 200), (
                f'expected context crop 200x200, got {face_crop_bgr.shape}'
            )
            return 0.91

    service = LivenessService(
        detector=FakeDetectorWithContext(),
        liveness_model=FakeContextLivenessModel(),
        threshold_live=0.9,
        threshold_spoof=0.3,
    )
    response = service.infer(LivenessInferRequest(image_base64='x'))

    assert response.liveness_score == 0.91
    assert response.liveness_label == 'live'
    assert response.pose_ok is True
    assert response.yaw_deg is not None
    assert abs(response.yaw_deg) < 30   # near-frontal landmarks → small yaw
