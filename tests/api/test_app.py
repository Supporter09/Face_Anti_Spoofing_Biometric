from base64 import b64encode

from fastapi.testclient import TestClient

from services.api.app import app


def test_health_endpoint_reports_ok() -> None:
    client = TestClient(app)

    response = client.get('/health')

    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_infer_endpoint_reports_no_face_when_image_missing() -> None:
    client = TestClient(app)

    response = client.post('/v1/liveness/infer', json={})

    assert response.status_code == 200
    assert response.json()['face_detected'] is False
    assert response.json()['liveness_label'] == 'no_face'
    assert response.json()['latency_ms'] >= 0.0


def test_infer_endpoint_returns_no_face_for_non_image_payload() -> None:
    client = TestClient(app)
    payload = {'image_base64': b64encode(b'x' * 64).decode('utf-8')}

    response = client.post('/v1/liveness/infer', json=payload)

    assert response.status_code == 200
    assert response.json()['face_detected'] is False
    assert response.json()['liveness_label'] == 'no_face'


def test_cors_preflight_allows_local_web_app() -> None:
    client = TestClient(app)

    response = client.options(
        '/v1/liveness/infer',
        headers={
            'Origin': 'http://127.0.0.1:5173',
            'Access-Control-Request-Method': 'POST',
        },
    )

    assert response.status_code == 200
    assert response.headers['access-control-allow-origin'] == 'http://127.0.0.1:5173'
