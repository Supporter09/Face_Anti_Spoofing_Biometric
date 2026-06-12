from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fas.auth_schemas import (
    FaceEnrollRequest,
    FaceEnrollResponse,
    FaceIdentifyRequest,
    FaceIdentifyResponse,
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

    def get_embedding(self, image_base64: str) -> np.ndarray | None:
        """Extract face embedding from a single image."""
        decoded = decode_base64_image_to_bgr(image_base64)
        if decoded.image_bgr is None:
            return None
        return self.recognition_model.get_embedding(decoded.image_bgr)

    def enroll(self, request: FaceEnrollRequest) -> FaceEnrollResponse:
        # Use pre-computed embedding if provided, otherwise extract from image
        if request.embedding is not None:
            embedding = np.array(request.embedding, dtype=np.float32)
        else:
            if request.image_base64 is None:
                return FaceEnrollResponse(
                    success=False,
                    user_id=request.user_id,
                    message="Either image_base64 or embedding must be provided.",
                )
            decoded = decode_base64_image_to_bgr(request.image_base64)
            if decoded.image_bgr is None:
                return FaceEnrollResponse(
                    success=False,
                    user_id=request.user_id,
                    message=decoded.error or "Could not decode image payload.",
                )
            embedding = self.recognition_model.get_embedding(decoded.image_bgr)

        if embedding is None:
            reason = self.recognition_model.unavailable_reason if hasattr(self.recognition_model, 'unavailable_reason') else "No face detected"
            return FaceEnrollResponse(
                success=False,
                user_id=request.user_id,
                message=reason or "Could not extract face embedding from image.",
            )

        self.store.save_template(request.user_id, embedding, request.image_base64)

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

    def identify(
        self,
        request: FaceIdentifyRequest,
        *,
        capture_debug: bool = False,
        debug_session_id: str = 'default',
    ) -> FaceIdentifyResponse:
        """Compare against all enrolled users, return the closest match."""
        all_templates = self.store.get_all_templates()

        if not all_templates:
            return FaceIdentifyResponse(
                authenticated=False,
                user_id=None,
                similarity=0.0,
                threshold=self.threshold,
                message="No enrolled users found.",
            )

        decoded = decode_base64_image_to_bgr(request.image_base64)
        if decoded.image_bgr is None:
            return FaceIdentifyResponse(
                authenticated=False,
                user_id=None,
                similarity=0.0,
                threshold=self.threshold,
                message=decoded.error or "Could not decode image payload.",
            )

        current_embedding = self.recognition_model.get_embedding(decoded.image_bgr)
        if current_embedding is None:
            reason = self.recognition_model.unavailable_reason
            return FaceIdentifyResponse(
                authenticated=False,
                user_id=None,
                similarity=0.0,
                threshold=self.threshold,
                message=reason or "No face detected in frame.",
            )

        best_user_id = max(
            all_templates,
            key=lambda uid: cosine_similarity(current_embedding, all_templates[uid]),
        )
        best_similarity = cosine_similarity(current_embedding, all_templates[best_user_id])
        authenticated = best_similarity >= self.threshold

        if capture_debug:
            self._save_enrollment_debug(
                user_id=best_user_id,
                session_id=debug_session_id,
                similarity=best_similarity,
                authenticated=authenticated,
            )

        return FaceIdentifyResponse(
            authenticated=authenticated,
            user_id=best_user_id,
            similarity=best_similarity,
            threshold=self.threshold,
            message=(
                f"Identified as {best_user_id}."
                if authenticated
                else f"No confident match (closest: {best_user_id}, score: {best_similarity:.3f})."
            ),
        )

    @staticmethod
    def _sanitize_debug_session_id(value: str) -> str:
        cleaned = re.sub(r'[^A-Za-z0-9._-]+', '_', value).strip('._-')
        return cleaned or 'default'

    def _save_enrollment_debug(
        self,
        *,
        user_id: str,
        session_id: str,
        similarity: float,
        authenticated: bool,
    ) -> None:
        try:
            import cv2  # type: ignore
        except ImportError:
            return

        _, image_base64 = self.store.get_template_with_image(user_id)
        if image_base64 is None:
            return

        root = Path(os.environ.get('AUTH_DEBUG_DIR', 'reports/auth_debug_frames'))
        session = self._sanitize_debug_session_id(session_id)
        session_dir = root / session
        session_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        base = session_dir / ts

        decoded = decode_base64_image_to_bgr(image_base64)
        if decoded.image_bgr is not None:
            cv2.imwrite(str(base) + '_enrolled_face.jpg', decoded.image_bgr)

        meta = {
            'session_id': session,
            'captured_at_utc': ts,
            'matched_user_id': user_id,
            'similarity': similarity,
            'authenticated': authenticated,
            'threshold': self.threshold,
            'enrolled_image_saved': decoded.image_bgr is not None,
        }
        (session_dir / f'{base.name}_meta.json').write_text(json.dumps(meta, indent=2))