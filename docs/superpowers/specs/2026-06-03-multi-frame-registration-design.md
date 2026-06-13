# Multi-Frame Registration Design

## Problem Statement

Currently, face registration uses a single frame to create the embedding template. This makes the system sensitive to:
- Noise in the captured frame
- Face rotation/angle at capture moment
- Background variations
- Expression differences

## Solution

Capture multiple centered frames during registration and average their embeddings to create a more robust template. Similar to how authentication now uses multiple frames.

## Architecture

### Registration Flow

```
User clicks "Start Registration"
         │
         ▼
   3s Countdown (prepare)
         │
         ▼
Continuous capture: frames every 20ms
Filter: face_detected && abs(yaw) ≤ 10° (YAW_CENTER)
         │
         ▼
Collect 5 "good" frames (centered faces)
OR timeout after 4 seconds
         │
         ▼
Extract embedding from each frame (via /v1/auth/embed)
         │
         ▼
Mean() all 5 embeddings → single template
         │
         ▼
Save to database (same as before)
```

### Component Changes

#### Frontend (`useRegister.ts`)

| Component | Change |
|-----------|--------|
| Capture logic | Continuous capture after countdown (same as now) |
| Frame filtering | Use `selectBestFramesForAuth` criteria (face_detected, centered) |
| Stop condition | Collect 5 frames OR timeout at 4s |
| Embedding extraction | Call new `/v1/auth/embed` endpoint for each frame |
| Aggregation | Mean() all embeddings in frontend |
| Enrollment | Send aggregated embedding to `/v1/auth/enroll` |

#### Backend Changes

| Endpoint | Change |
|----------|--------|
| `/v1/auth/enroll` | Accept embedding directly (optional, for aggregated template) |
| `/v1/auth/embed` | **NEW** - Extract embedding from single image |

### Data Storage

No schema change required:
- Template stored as single 512-dim embedding (same as before)
- Computed from 5 frames instead of 1

### Timing

| Phase | Duration |
|-------|-----------|
| Countdown | 3s |
| Capture window | 4s max |
| Processing | ~500ms |
| **Total** | ~7.5s |

## API Design

### New Endpoint: `/v1/auth/embed`

**Request:**
```json
{
  "image_base64": "..."
}
```

**Response:**
```json
{
  "embedding": [0.1, -0.2, ...]  // 512-dim array
}
```

### Modified: `/v1/auth/enroll`

**Current behavior:** Accept image, extract embedding, save

**New behavior:** Accept optional `embedding` field for pre-computed embedding

```json
// Either:
{ "user_id": "john", "image_base64": "..." }

// Or (for multi-frame):
{ "user_id": "john", "embedding": [0.1, -0.2, ...] }
```

If both provided, prefer `embedding`. If only image provided, extract (backward compatible).

## Frontend Logic

### useRegister.ts Changes

```typescript
// Constants
const REGISTRATION_FRAME_COUNT = 5
const REGISTRATION_CAPTURE_MS = 4000  // 4s to collect frames

// New state
interface RegisterState {
  // ... existing fields
  captured_frames: string[]  // base64 images
  embedding_progress: number // 0-5 counter
}

// Flow:
// 1. After countdown (same as now)
// 2. tick() captures frames, filters by yaw
// 3. Store good frames in captured_frames array
// 4. Stop when captured_frames.length >= 5 OR timeout
// 5. Call /v1/auth/embed for each frame
// 6. Mean() all embeddings
// 7. Call /v1/auth/enroll with aggregated embedding
```

### Frame Selection Criteria

Same as authentication:
- `face_detected === true`
- `abs(yaw) ≤ 10°` (YAW_CENTER)
- Prefer higher `passive_score` (if available)

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Less than 5 frames collected | Use whatever frames available (≥1) |
| No frames collected | Show error "Không tìm thấy khuôn mặt nhìn thẳng" |
| Embedding extraction fails | Skip that frame, use others |
| All extractions fail | Show error "Không thể trích xuất đặc trưng khuôn mặt" |

## Comparison: Before vs After

| Aspect | Before | After |
|--------|--------|-------|
| Frames used | 1 | 5 |
| Template quality | Sensitive to single frame | Averaged, more robust |
| Registration time | ~3s | ~7s |
| Storage | Same | Same |
| Auth compatibility | Works | Works |

## Files to Modify

1. **`services/api/app.py`** - Add `/v1/auth/embed` endpoint
2. **`src/fas/auth_service.py`** - Add `get_embedding()` method (or reuse)
3. **`src/fas/auth_store.py`** - Accept pre-computed embedding in `enroll()`
4. **`apps/web/src/session/useRegister.ts`** - Multi-frame capture logic
5. **`apps/web/src/session/types.ts`** - Add `captured_frames` field

## Implementation Order

1. Add `/v1/auth/embed` endpoint in backend
2. Modify `/v1/auth/enroll` to accept pre-computed embedding
3. Update frontend capture logic to collect 5 frames
4. Add embedding extraction + aggregation in frontend
5. Test end-to-end

## Testing Checklist

- [ ] Registration collects exactly 5 centered frames
- [ ] Registration works with <5 frames (timeout case)
- [ ] No face centered → error shows correctly
- [ ] Authentication still works with new templates
- [ ] Similarity scores improve (less variance)