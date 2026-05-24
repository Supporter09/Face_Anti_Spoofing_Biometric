"""Head pose estimation from 5 InsightFace landmarks via solvePnP.

Sign convention:
- yaw_deg > 0  <=> user turns head to their RIGHT  (face tilts to LEFT in frame)
- yaw_deg < 0  <=> user turns head to their LEFT
- pitch_deg > 0 <=> user looks UP
- pitch_deg < 0 <=> user looks DOWN
"""
from __future__ import annotations

import math

import numpy as np

# Generic adult face template in millimetres. Nose tip at origin.
# Order matches InsightFace kps: left_eye, right_eye, nose, mouth_left, mouth_right.
MODEL_POINTS_3D = np.array([
    (-30.0,  30.0, -30.0),
    ( 30.0,  30.0, -30.0),
    (  0.0,   0.0,   0.0),
    (-25.0, -30.0, -30.0),
    ( 25.0, -30.0, -30.0),
], dtype=np.float64)


def _rotation_matrix_to_euler_zyx(R: np.ndarray) -> tuple[float, float, float]:
    """Return (yaw, pitch, roll) in degrees.

    Convention: yaw = Y rotation, pitch = X rotation, roll = Z rotation.
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        yaw = math.atan2(-R[2, 0], sy)
        pitch = math.atan2(R[2, 1], R[2, 2])
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        yaw = math.atan2(-R[2, 0], sy)
        pitch = math.atan2(-R[1, 2], R[1, 1])
        roll = 0.0

    return (
        math.degrees(yaw),
        math.degrees(pitch),
        math.degrees(roll),
    )


def estimate_head_pose(
    landmarks_xy: list[tuple[float, float]],
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> dict[str, float] | None:
    """Return {'yaw_deg', 'pitch_deg', 'roll_deg', 'ok'} or None on failure.

    landmarks_xy: 5 (x, y) points in pixel coords (left_eye, right_eye, nose, mouth_left, mouth_right).
    image_shape: (H, W) or (H, W, C).
    """
    if len(landmarks_xy) != 5:
        raise ValueError(f'estimate_head_pose requires exactly 5 landmarks, got {len(landmarks_xy)}')

    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError('OpenCV is required for estimate_head_pose.') from exc

    h, w = image_shape[0], image_shape[1]
    # Approx pinhole: focal length = image width (~60 deg FOV).
    K = np.array(
        [[float(w), 0.0, float(w) / 2.0],
         [0.0, float(w), float(h) / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.zeros((4, 1), dtype=np.float64)
    img_pts = np.asarray(landmarks_xy, dtype=np.float64).reshape(-1, 2)

    try:
        ok, rvec, _tvec = cv2.solvePnP(
            MODEL_POINTS_3D,
            img_pts,
            K,
            dist,
            flags=cv2.SOLVEPNP_SQPNP,
        )
    except cv2.error:
        return {'yaw_deg': 0.0, 'pitch_deg': 0.0, 'roll_deg': 0.0, 'ok': False}
    if not ok:
        return {'yaw_deg': 0.0, 'pitch_deg': 0.0, 'roll_deg': 0.0, 'ok': False}

    R, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = _rotation_matrix_to_euler_zyx(R)
    # solvePnP can flip yaw 180 deg; clamp to [-90, +90] domain for our use case.
    if yaw > 90:
        yaw -= 180
    if yaw < -90:
        yaw += 180
    return {
        'yaw_deg': float(yaw),
        'pitch_deg': float(pitch),
        'roll_deg': float(roll),
        'ok': True,
    }
