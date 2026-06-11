# _meta.json Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

**Goal:** Enrich debug `_meta.json` with additional liveness detection fields for better debugging.

**Architecture:** Add fields to both `_save_debug_frame()` and `_save_capture_debug()` methods in `LivenessService`. Frontend sends phase info via request body; backend tracks latencies internally.

**Tech Stack:** Python FastAPI backend, React frontend

---

## Task List

### Task 1: Add phase field to request schema

**Files:**
- Modify: `src/fas/schemas.py`

- [ ] **Step 1: Add phase field to LivenessInferRequest**

```python
class LivenessInferRequest(BaseModel):
    image_base64: str | None = Field(default=None, description='Base64 encoded image payload.')
    phase: str | None = Field(default=None, description='Current liveness phase (forward, turn_A, center_1, turn_B, blink).')
```

- [ ] **Step 2: Run lint check**

Run: `cd src && python -c "from fas.schemas import LivenessInferRequest; print(LivenessInferRequest.model_fields)"`

- [ ] **Step 3: Commit**

---

### Task 2: Add latency tracking to liveness model

**Files:**
- Modify: `src/fas/liveness_model.py`

- [ ] **Step 1: Add inference latency tracking**

Add `predict_live_score_with_latency()` method or track timing in existing method.

- [ ] **Step 2: Test the latency tracking**

Run: `cd src && python -c "from fas.liveness_model import TorchLivenessModel; m = TorchLivenessModel(); print('ok')"`

- [ ] **Step 3: Commit**

---

### Task 3: Expose detection confidence from detector

**Files:**
- Modify: `src/fas/detection.py`

- [ ] **Step 1: Check if detection confidence available**

Read detection.py to find if confidence score is returned.

- [ ] **Step 2: Expose confidence in FaceDetection or detection output**

Add confidence field to return value.

- [ ] **Step 3: Commit**

---

### Task 4: Update service.py _meta.json for capture_debug mode

**Files:**
- Modify: `src/fas/service.py`

- [ ] **Step 1: Update _save_capture_debug() to include new fields**

Add to meta dict:
```python
meta = {
    ...
    'liveness_score': response.liveness_score,
    'liveness_label': response.liveness_label,
    'face_detected': response.face_detected,
    'yaw_deg': response.yaw_deg,
    'current_phase': request.phase,
    'detection_latency_ms': ...,  # From detection timing
    'liveness_latency_ms': ...,   # From liveness timing
    ...
}
```

- [ ] **Step 2: Commit**

---

### Task 5: Update service.py _meta.json for env var debug mode

**Files:**
- Modify: `src/fas/service.py`

- [ ] **Step 1: Update _save_debug_frame() meta with same fields**

- [ ] **Step 2: Commit**

---

### Task 6: Update frontend to send phase in request body

**Files:**
- Modify: `apps/web/src/session/useSession.ts`

- [ ] **Step 1: Add phase to request body in captureAndSend()**

```typescript
body: JSON.stringify({
  image_base64: queued.imageBase64,
  phase: current.phase,
})
```

- [ ] **Step 2: Test build**

Run: `cd apps/web && npm run build` (or check TypeScript)

- [ ] **Step 3: Commit**

---

### Task 7: Final verification

**Files:**
- All modified files

- [ ] **Step 1: Full integration test**

Start backend, trigger capture_debug mode, check _meta.json output.

- [ ] **Step 2: Commit**