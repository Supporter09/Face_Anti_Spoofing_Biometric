import base64
import json
from pathlib import Path

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


def test_frame_debug_endpoint_saves_request_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv('LIVENESS_DEBUG_CAPTURE_ROOT', str(tmp_path))
    client = TestClient(app)

    response = client.post('/v1/liveness/frame/debug?session_id=phone_replay', json={'image_base64': _png_base64()})
    assert response.status_code == 200

    session_dir = tmp_path / 'phone_replay'
    overlay_files = sorted(session_dir.glob('*_overlay.jpg'))
    meta_files = sorted(session_dir.glob('*_meta.json'))
    assert overlay_files, 'expected at least one saved overlay image'
    assert meta_files, 'expected at least one saved metadata file'

    meta = json.loads(meta_files[-1].read_text())
    assert meta['session_id'] == 'phone_replay'
    assert meta['overlay_path'] is not None
    if meta['context_model_input_raw_path'] is not None:
        assert (session_dir / Path(meta['context_model_input_raw_path']).name).exists()
    assert isinstance(meta.get('response', {}).get('liveness_score'), float)
