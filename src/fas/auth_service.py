from __future__ import annotations

import os

from fas.auth_schemas import (
    FaceEnrollRequest,
    FaceEnrollResponse,
    FaceVerifyRequest,
    FaceVerifyResponse,
)
from fas.auth_store import FaceTemplateStore
from fas.preprocess import decode_base64_image_to_bgr
from fas.recognition_model import FaceRecognitionModel
from fas.similarity import cosine_similarity


class FaceAuthService:
    def __init__(
        self,
        *,
        recognition_model: FaceRecognitionModel | None = None,
        store: FaceTemplateStore | None = None,
        threshold: float | None = None,
    ) -> None:
        self.recognition_model = recognition_model or FaceRecognitionModel()
        self.store = store or FaceTemplateStore()

        if threshold is not None:
            self.threshold = threshold
        else:
            raw_threshold = os.environ.get("FACE_AUTH_THRESHOLD", "0.50")
            try:
                self.threshold = float(raw_threshold)
            except ValueError:
                self.threshold = 0.50

    def enroll(self, request: FaceEnrollRequest) -> FaceEnrollResponse:
        decoded = decode_base64_image_to_bgr(request.image_base64)

        if decoded.image_bgr is None:
            return FaceEnrollResponse(
                success=False,
                user_id=request.user_id,
                message=decoded.error or "Could not decode image payload.",
            )

        embedding = self.recognition_model.get_embedding(decoded.image_bgr)

        if embedding is None:
            reason = self.recognition_model.unavailable_reason
            return FaceEnrollResponse(
                success=False,
                user_id=request.user_id,
                message=reason or "No face was detected for enrollment.",
            )

        self.store.save_template(request.user_id, embedding)

        return FaceEnrollResponse(
            success=True,
            user_id=request.user_id,
            message="Face enrolled successfully.",
        )

    def verify(self, request: FaceVerifyRequest) -> FaceVerifyResponse:
        stored_embedding = self.store.get_template(request.user_id)

        if stored_embedding is None:
            return FaceVerifyResponse(
                authenticated=False,
                user_id=request.user_id,
                similarity=0.0,
                threshold=self.threshold,
                message="No enrolled face template found for this user.",
            )

        decoded = decode_base64_image_to_bgr(request.image_base64)

        if decoded.image_bgr is None:
            return FaceVerifyResponse(
                authenticated=False,
                user_id=request.user_id,
                similarity=0.0,
                threshold=self.threshold,
                message=decoded.error or "Could not decode image payload.",
            )

        current_embedding = self.recognition_model.get_embedding(decoded.image_bgr)

        if current_embedding is None:
            reason = self.recognition_model.unavailable_reason
            return FaceVerifyResponse(
                authenticated=False,
                user_id=request.user_id,
                similarity=0.0,
                threshold=self.threshold,
                message=reason or "No face was detected for verification.",
            )

        similarity = cosine_similarity(current_embedding, stored_embedding)
        authenticated = similarity >= self.threshold

        return FaceVerifyResponse(
            authenticated=authenticated,
            user_id=request.user_id,
            similarity=similarity,
            threshold=self.threshold,
            message=(
                "Authentication successful."
                if authenticated
                else "Face does not match enrolled user."
            ),
        )