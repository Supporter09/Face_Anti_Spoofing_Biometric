from typing import Literal

from pydantic import BaseModel, Field


LivenessLabel = Literal['live', 'spoof', 'no_face', 'uncertain']


class LivenessInferRequest(BaseModel):
    image_base64: str | None = Field(default=None, description='Base64 encoded image payload.')


class LivenessInferResponse(BaseModel):
    face_detected: bool
    liveness_score: float = Field(ge=0.0, le=1.0)
    liveness_label: LivenessLabel
    latency_ms: float = Field(ge=0.0)
    message: str | None = None
    face_bbox_xyxy: list[int] | None = None
    face_landmarks: list[list[float]] | None = None
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    pose_ok: bool = False
