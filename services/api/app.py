from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fas.schemas import LivenessInferRequest, LivenessInferResponse
from fas.service import LivenessService

app = FastAPI(title='Face Anti-Spoofing API', version='0.1.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://127.0.0.1:5173', 'http://localhost:5173'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)
service = LivenessService()


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/v1/liveness/infer', response_model=LivenessInferResponse)
def infer_liveness(payload: LivenessInferRequest) -> LivenessInferResponse:
    return service.infer(payload)
