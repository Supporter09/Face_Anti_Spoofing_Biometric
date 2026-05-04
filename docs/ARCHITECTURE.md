# Architecture

## MVP Pipeline

1. Browser captures a frame.
2. React submits the image to FastAPI.
3. The backend decodes the image, performs face localization/alignment, and runs liveness scoring.
4. The API returns `face_detected`, `liveness_score`, `liveness_label`, `latency_ms`, and an optional message.

## Planned Model Stack

- Detection/alignment: `insightface` (`SCRFD` based)
- Liveness model: `MiniFASNetV2`
- Recognition/authentication later: pre-trained `ArcFace` + `FAISS + SQLite`
