import base64

import numpy as np
from fastapi.testclient import TestClient

from services.api.app import app


def _png_base64() -> str:
    import cv2
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    ok, buf = cv2.imencode('.jpg', img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def test_frame_endpoint_returns_pose_fields():
    client = TestClient(app)
    response = client.post('/v1/liveness/frame', json={'image_base64': _png_base64()})
    assert response.status_code == 200
    body = response.json()
    # Schema fields exist (even if no face -> yaw_deg may be None, pose_ok False)
    assert 'liveness_score' in body
    assert 'liveness_label' in body
    assert 'yaw_deg' in body
    assert 'pitch_deg' in body
    assert 'pose_ok' in body
    assert body['pose_ok'] is False or isinstance(body['yaw_deg'], (float, int))
