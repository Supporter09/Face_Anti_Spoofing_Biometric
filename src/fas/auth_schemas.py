from __future__ import annotations

from pydantic import BaseModel, Field


class FaceEnrollRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    image_base64: str = Field(..., min_length=1)


class FaceEnrollResponse(BaseModel):
    success: bool
    user_id: str
    message: str


class FaceVerifyRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    image_base64: str = Field(..., min_length=1)


class FaceVerifyResponse(BaseModel):
    authenticated: bool
    user_id: str
    similarity: float
    threshold: float
    message: str