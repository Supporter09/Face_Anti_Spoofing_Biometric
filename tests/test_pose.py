import math

import numpy as np
import pytest

from fas.pose import estimate_head_pose


# Helper: project a 3D point given pose rotation matrix R and translation t, camera K
def _project(point_3d: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray) -> tuple[float, float]:
    cam = R @ point_3d + t
    img = K @ cam
    return float(img[0] / img[2]), float(img[1] / img[2])


def _build_synthetic_landmarks(yaw_deg: float, pitch_deg: float, image_w: int = 640, image_h: int = 480):
    """Generate 5 landmarks for a face at given yaw/pitch in front of camera at z=400mm."""
    from fas.pose import MODEL_POINTS_3D
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    Ry = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)],
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch), math.cos(pitch)],
    ])
    R = Ry @ Rx
    t = np.array([0.0, 0.0, 400.0])
    K = np.array([[image_w, 0, image_w / 2], [0, image_w, image_h / 2], [0, 0, 1]])
    return [_project(p, R, t, K) for p in MODEL_POINTS_3D]


def test_frontal_face_has_small_yaw():
    landmarks = _build_synthetic_landmarks(yaw_deg=0.0, pitch_deg=0.0)
    result = estimate_head_pose(landmarks, image_shape=(480, 640))
    assert result['ok'] is True
    assert abs(result['yaw_deg']) < 5.0
    assert abs(result['pitch_deg']) < 5.0


def test_right_turn_gives_positive_yaw():
    landmarks = _build_synthetic_landmarks(yaw_deg=25.0, pitch_deg=0.0)
    result = estimate_head_pose(landmarks, image_shape=(480, 640))
    assert result['ok'] is True
    assert result['yaw_deg'] > 15.0


def test_left_turn_gives_negative_yaw():
    landmarks = _build_synthetic_landmarks(yaw_deg=-25.0, pitch_deg=0.0)
    result = estimate_head_pose(landmarks, image_shape=(480, 640))
    assert result['ok'] is True
    assert result['yaw_deg'] < -15.0


def test_returns_not_ok_on_degenerate_landmarks():
    same_point = [(100.0, 100.0)] * 5
    result = estimate_head_pose(same_point, image_shape=(480, 640))
    assert result is None or result.get('ok') is False


def test_requires_exactly_five_landmarks():
    with pytest.raises(ValueError):
        estimate_head_pose([(0.0, 0.0)] * 4, image_shape=(480, 640))
