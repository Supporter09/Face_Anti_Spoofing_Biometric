# _meta.json Enrichment Specification

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enrich debug `_meta.json` with additional liveness detection fields for better debugging and analysis.

**Architecture:** Add fields to both `_save_debug_frame()` and `_save_capture_debug()` methods in `LivenessService`. Frontend sends phase info via request body; backend tracks latencies internally.

**Tech Stack:** Python FastAPI backend, React frontend

---

## Background

Currently `_meta.json` contains limited information. For effective debugging of liveness detection issues, we need more context about detection quality, timing, and the session phase.

## Requirements

### 1. Core Fields to Add

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `liveness_score` | float | Response | Liveness probability (0-1) |
| `liveness_label` | string | Response | "live", "spoof", "uncertain", "no_face" |
| `face_detected` | bool | Response | Whether face was detected |
| `yaw_deg` | float | Response | Yaw angle in degrees |
| `current_phase` | string | Request body | Current liveness phase |
| `detection_confidence` | float | Detection | Face detection confidence |
| `detection_latency_ms` | float | Internal | Time for face detection |
| `liveness_latency_ms` | float | Internal | Time for liveness inference |

### 2. Latency Breakdown

Track and report:
- `detection_latency_ms` — InsightFace detection time
- `liveness_latency_ms` — Liveness model inference time
- Already tracked: `latency_ms` (total)

### 3. Frontend Changes

Frontend must send `phase` in request body:

```typescript
// In useSession.ts captureAndSend()
body: JSON.stringify({
  image_base64: queued.imageBase64,
  phase: current.phase,  // ADD THIS
})
```

### 4. Schema Update

Update `LivenessInferRequest` to accept optional `phase`:

```python
class LivenessInferRequest(BaseModel):
    image_base64: str | None = None
    phase: str | None = None  # ADD THIS
```

---

## File Changes

### Modified Files
- `src/fas/schemas.py` — Add `phase` to request schema
- `src/fas/service.py` — Add fields to both `_meta.json` writers
- `src/fas/detection.py` — Expose detection confidence if available
- `apps/web/src/session/useSession.ts` — Send phase in request body
- `src/fas/liveness_model.py` — Expose inference latency if tracked

### No New Files

---

## Acceptance Criteria

1. `_meta.json` from both capture modes includes: `liveness_score`, `liveness_label`, `face_detected`, `yaw_deg`, `current_phase`, `detection_confidence`, `detection_latency_ms`, `liveness_latency_ms`
2. Frontend sends `phase` in request body
3. No breaking changes to existing API response format
4. Latency fields are accurate (within 5ms of actual timing)