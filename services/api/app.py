import base64
from dotenv import load_dotenv
load_dotenv()
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from fas.schemas import LivenessInferRequest, LivenessInferResponse
from fas.service import LivenessService
from fas.auth_schemas import (
    FaceEnrollRequest,
    FaceEnrollResponse,
    FaceVerifyRequest,
    FaceVerifyResponse,
    FaceIdentifyRequest,
    FaceIdentifyResponse,
)
from fas.auth_service import FaceAuthService

app = FastAPI(title='Face Anti-Spoofing API', version='0.1.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:5174",
        "http://localhost:5174",
    ],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)
service = LivenessService()
auth_service = FaceAuthService()

@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/v1/liveness/infer', response_model=LivenessInferResponse)
def infer_liveness(payload: LivenessInferRequest) -> LivenessInferResponse:
    return service.infer(payload)


@app.post('/v1/liveness/frame', response_model=LivenessInferResponse)
def infer_frame(payload: LivenessInferRequest) -> LivenessInferResponse:
    return service.infer(payload)

@app.post("/v1/auth/enroll", response_model=FaceEnrollResponse)
def enroll_face(payload: FaceEnrollRequest) -> FaceEnrollResponse:
    return auth_service.enroll(payload)

@app.post("/v1/auth/identify", response_model=FaceIdentifyResponse)
def identify_face(payload: FaceIdentifyRequest) -> FaceIdentifyResponse:
    return auth_service.identify(payload)

@app.post("/v1/auth/verify", response_model=FaceVerifyResponse)
def verify_face(payload: FaceVerifyRequest) -> FaceVerifyResponse:
    return auth_service.verify(payload)

@app.websocket("/ws/liveness")
async def ws_liveness(websocket: WebSocket) -> None:
    await websocket.accept()
    service = LivenessService()
    try:
        while True:
            data = await websocket.receive_bytes()
            arr = np.frombuffer(data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                await websocket.send_json({"error": "could not decode frame"})
                continue
            _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64 = base64.b64encode(buf.tobytes()).decode()
            result = service.infer(LivenessInferRequest(image_base64=b64))
            await websocket.send_json(result.model_dump())
    except WebSocketDisconnect:
        pass
