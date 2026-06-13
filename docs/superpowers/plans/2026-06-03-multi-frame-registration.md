# Multi-Frame Registration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable multi-frame registration - capture 5 centered frames, extract embeddings, average them, save as single template.

**Architecture:** Frontend captures frames → sends to new embed endpoint → averages embeddings → enrolls with pre-computed embedding. Backend accepts optional embedding in enroll request.

**Tech Stack:** React (frontend), FastAPI (backend), InsightFace (embedding), PostgreSQL (storage)

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/fas/auth_schemas.py` | Add embedding field to request/response |
| `src/fas/auth_service.py` | Handle pre-computed embedding in enroll |
| `services/api/app.py` | Add `/v1/auth/embed` endpoint |
| `apps/web/src/session/useRegister.ts` | Multi-frame capture logic |
| `apps/web/src/session/fusion.ts` | Add mean embedding helper |

---

## Implementation Tasks

### Task 1: Add embedding field to FaceEnrollRequest schema

**Files:**
- Modify: `src/fas/auth_schemas.py:6-9`

- [ ] **Step 1: Add optional embedding field to FaceEnrollRequest**

```python
# Change from:
class FaceEnrollRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    image_base64: str = Field(..., min_length=1)

# To:
class FaceEnrollRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    image_base64: str | None = Field(None, min_length=1)
    embedding: list[float] | None = None
```

- [ ] **Step 2: Commit**

```bash
git add src/fas/auth_schemas.py
git commit -m "feat(auth_schemas): add optional embedding field to FaceEnrollRequest"
```

---

### Task 2: Create /v1/auth/embed endpoint

**Files:**
- Create: New endpoint in `services/api/app.py`
- Modify: `src/fas/auth_service.py` (get_embedding method)

- [ ] **Step 1: Add get_embedding() method to FaceAuthService**

In `src/fas/auth_service.py`, add method after the class starts:

```python
def get_embedding(self, image_base64: str) -> np.ndarray | None:
    """Extract face embedding from a single image."""
    decoded = decode_base64_image_to_bgr(image_base64)
    if decoded.image_bgr is None:
        return None
    return self.recognition_model.get_embedding(decoded.image_bgr)
```

- [ ] **Step 2: Add /v1/auth/embed endpoint in app.py**

```python
from pydantic import BaseModel

class EmbedRequest(BaseModel):
    image_base64: str

class EmbedResponse(BaseModel):
    embedding: list[float]

@app.post("/v1/auth/embed", response_model=EmbedResponse)
def embed_face(payload: EmbedRequest) -> EmbedResponse:
    embedding = auth_service.get_embedding(payload.image_base64)
    if embedding is None:
        raise HTTPException(status_code=400, detail="Could not extract face embedding")
    return EmbedResponse(embedding=embedding.tolist())
```

- [ ] **Step 3: Commit**

```bash
git add src/fas/auth_service.py services/api/app.py
git commit -m "feat(auth): add /v1/auth/embed endpoint for extracting face embeddings"
```

---

### Task 3: Modify enroll to accept pre-computed embedding

**Files:**
- Modify: `src/fas/auth_service.py:39-59`

- [ ] **Step 1: Update enroll method to accept pre-computed embedding**

Current code (around line 39-59):

```python
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
        return FaceEnrollResponse(
            success=False,
            user_id=request.user_id,
            message="Could not extract face embedding from image.",
        )

    self.store.save_template(request.user_id, embedding)
    return FaceEnrollResponse(success=True, user_id=request.user_id, message="Enrollment successful")
```

Replace with:

```python
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
        return FaceEnrollResponse(
            success=False,
            user_id=request.user_id,
            message="Could not extract face embedding from image.",
        )

    self.store.save_template(request.user_id, embedding)
    return FaceEnrollResponse(success=True, user_id=request.user_id, message="Enrollment successful")
```

- [ ] **Step 2: Commit**

```bash
git add src/fas/auth_service.py
git commit -m "feat(auth_service): accept pre-computed embedding in enroll"
```

---

### Task 4: Add mean embedding helper in frontend

**Files:**
- Modify: `apps/web/src/session/fusion.ts`

- [ ] **Step 1: Add meanEmbeddings helper function**

Add at the end of fusion.ts:

```typescript
/**
 * Average multiple embeddings into a single template.
 * @param embeddings Array of embedding arrays (each 512-dim)
 * @returns Mean embedding as flat array
 */
export function meanEmbeddings(embeddings: number[][]): number[] {
  if (embeddings.length === 0) return []
  if (embeddings.length === 1) return embeddings[0]

  const dim = embeddings[0].length
  const result = new Array(dim).fill(0)

  for (const emb of embeddings) {
    for (let i = 0; i < dim; i++) {
      result[i] += emb[i]
    }
  }

  for (let i = 0; i < dim; i++) {
    result[i] /= embeddings.length
  }

  return result
}
```

- [ ] **Step 2: Commit**

```bash
git add apps/web/src/session/fusion.ts
git commit -m "feat(fusion): add meanEmbeddings helper for multi-frame registration"
```

---

### Task 5: Update useRegister to capture multiple frames

**Files:**
- Modify: `apps/web/src/session/useRegister.ts`

- [ ] **Step 1: Add constants for multi-frame capture**

```typescript
const REGISTRATION_FRAME_COUNT = 5
const REGISTRATION_CAPTURE_TIMEOUT_MS = 4000
```

- [ ] **Step 2: Update state to track captured frames**

Add to RegisterState interface:
```typescript
captured_frames: string[]  // base64 images
```

Update initialState:
```typescript
captured_frames: [],
```

- [ ] **Step 3: Modify tick() to collect frames**

The tick() function runs every 20ms. Update to:
1. Capture frame
2. Send to /v1/liveness/frame for yaw check
3. If centered (abs(yaw) <= YAW_CENTER), add to captured_frames
4. Stop when captured_frames.length >= REGISTRATION_FRAME_COUNT OR timeout

Key changes in tick():
```typescript
// After getting yaw from liveness response:
const isCentered = faceDetected && yaw !== null && Math.abs(yaw) <= YAW_CENTER

if (isCentered && !state.captured_frames.includes(imageBase64)) {
  setState((s) => ({
    ...s,
    captured_frames: [...s.captured_frames, imageBase64],
  }))

  // If we have enough frames, stop capturing
  if (stateRef.current.captured_frames.length >= REGISTRATION_FRAME_COUNT) {
    // Trigger enrollment
  }
}
```

- [ ] **Step 4: Add enrollment function with multiple frames**

After capture completes:
1. Call /v1/auth/embed for each frame (parallel)
2. Get embeddings array
3. Call meanEmbeddings() to get average
4. Call /v1/auth/enroll with embedding field

```typescript
const enrollWithFrames = async (frames: string[]) => {
  // Extract embeddings in parallel
  const embedPromises = frames.map(frame =>
    fetch(`${API_BASE}/v1/auth/embed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_base64: frame }),
    }).then(res => res.json())
  )

  const results = await Promise.all(embedPromises)
  const embeddings = results.map(r => r.embedding).filter(Boolean)

  if (embeddings.length === 0) {
    throw new Error('Không thể trích xuất đặc trưng khuôn mặt')
  }

  // Average embeddings
  const avgEmbedding = meanEmbeddings(embeddings)

  // Enroll with pre-computed embedding
  await fetch(`${API_BASE}/v1/auth/enroll`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: currentUserId,
      embedding: avgEmbedding,
    }),
  })
}
```

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/session/useRegister.ts
git commit -m "feat(useRegister): capture 5 centered frames and average embeddings for registration"
```

---

### Task 6: Update UI to show progress

**Files:**
- Modify: `apps/web/src/session/RegisterView.tsx`

- [ ] **Step 1: Add progress indicator**

Show "Collecting frames: 3/5" during capture phase.

- [ ] **Step 2: Commit**

```bash
git add apps/web/src/session/RegisterView.tsx
git commit -m "feat(RegisterView): show multi-frame capture progress"
```

---

## Summary

| Task | Description |
|------|-------------|
| 1 | Add embedding field to FaceEnrollRequest schema |
| 2 | Create /v1/auth/embed endpoint |
| 3 | Modify enroll to accept pre-computed embedding |
| 4 | Add meanEmbeddings helper in frontend |
| 5 | Update useRegister to capture 5 frames + aggregate |
| 6 | Update UI to show progress |

---

## Plan Complete

**Plan saved to:** `docs/superpowers/plans/2026-06-03-multi-frame-registration.md`

**Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**