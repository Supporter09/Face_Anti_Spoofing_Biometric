# Enhanced Registration + Authentication Design

## Problem Statement

### Problem 1: Registration - No Face Centering Check

When user registers their face, there's no validation that they're looking straight at the camera. The current flow simply captures one frame after a 3-second countdown and sends it to the backend.

**Issue:** User might register while turning their head, resulting in a template that doesn't match well during verification.

### Problem 2: Authentication - Wrong Frame

The authentication process captures a NEW frame at the moment liveness detection ends. At that moment, the user's head may still be turned (from the turn_B phase).

**Issue:** The captured frame may not be centered, leading to lower similarity scores or false negatives.

## Solution

### Solution 1: Registration - Wait for Centered Face

Instead of capturing after a fixed countdown, continuously capture frames (every 20ms) and wait for a frame where the face is centered (`abs(yaw) <= YAW_CENTER`). If no good frame found within 4 seconds, show error.

### Solution 2: Authentication - Use Liveness Frames

Instead of capturing a new frame, reuse the frames already captured during liveness detection. Select the top 3 frames with highest passive scores that are also centered, then average their authentication scores.

## Architecture

### Frontend Changes

#### useRegister.ts

```
State: idle → countdown → capturing → success/error

Flow:
1. User enters user_id, clicks "Start"
2. Timer fires every 20ms, capturing frames
3. Each frame sent to /v1/liveness/frame (for pose info)
4. Check: if abs(yaw) <= YAW_CENTER AND face_detected
5. If good frame found → enroll with that frame
6. If 4s timeout → show error "Không tìm thấy khuôn mặt nhìn thẳng"
```

#### useSession.ts - authenticate()

```
Current flow:
  Liveness pass → capture new frame → POST /v1/auth/identify

New flow:
  Liveness pass →
    1. Get all frames from state.frames
    2. Filter: face_detected && abs(relative_yaw) <= YAW_CENTER && passive_score >= T_PASSIVE
    3. Sort by passive_score descending
    4. Take top 3
    5. For each: extract base64, call /v1/auth/identify
    6. Calculate: avg_similarity = sum(similarities) / 3
    7. If avg_similarity >= threshold → authenticated
```

#### fusion.ts

Add helper function to filter and select best frames:

```typescript
export function selectBestFramesForAuth(
  frames: FrameRecord[],
  yawBaseline: number,
  count: number = 3
): FrameRecord[]
```

### Backend Changes

No changes required. Existing endpoints handle:
- `/v1/liveness/frame` - returns yaw for pose check
- `/v1/auth/enroll` - accepts any frame
- `/v1/auth/identify` - returns similarity

## Data Flow

### Registration Flow

```
User clicks "Start Registration"
         │
         ▼
Timer fires every 20ms
         │
         ▼
┌─────────────────────────────────────┐
│ captureFrameBase64()                │
│ POST /v1/liveness/frame             │
│ Receive: face_detected, yaw_deg     │
└─────────────────────────────────────┘
         │
         ▼
Check: face_detected && abs(yaw_deg) <= YAW_CENTER
         │
    ┌────┴────┐
    │         │
  YES        NO
    │         │
    ▼         ▼
 Enroll    Timeout?
   │       /   \
   │      4s   continue
   │       \   /
   │        ▼
   │    Keep capturing
   ▼
Success/Error
```

### Authentication Flow

```
Liveness passes (phase = 'result')
         │
         ▼
Get all frames from state.frames (~150-200 frames)
         │
         ▼
Filter: face_detected && abs(yaw - baseline) <= YAW_CENTER && passive >= T_PASSIVE
         │
         ▼
Sort by passive_score descending
         │
         ▼
Take top 3 frames
         │
         ▼
For each frame (parallel or sequential):
  - Extract base64 from frame (need to store during capture)
  - POST /v1/auth/identify
  - Get similarity
         │
         ▼
avg_similarity = (s1 + s2 + s3) / 3
         │
         ▼
If avg_similarity >= threshold (0.50):
  → auth_status = 'authenticated'
Else:
  → auth_status = 'failed'
```

## API Changes

### /v1/auth/identify (multiple frames)

Option A: Call endpoint 3 times sequentially
Option B: Add new endpoint that accepts multiple frames and returns averaged result

**Recommendation:** Option A for simplicity (no backend changes needed)

## Component State Changes

### RegisterState (useRegister.ts)

```typescript
interface RegisterState {
  phase: 'idle' | 'countdown' | 'capturing' | 'success' | 'error'
  countdown: number
  userId: string
  error: string | null
  capturedFrame: string | null
  // NEW:
  latest_yaw: number | null
  face_detected: boolean
}
```

### SessionState (useSession.ts)

```typescript
interface SessionState {
  // ... existing fields ...
  // NEW: store base64 for each frame (optional, for auth reuse)
  frames: FrameRecord[]
}

interface FrameRecord {
  ts_ms: number
  phase: Phase
  face_detected: boolean
  passive_score: number
  yaw_deg: number | null
  pose_ok: boolean
  // NEW:
  image_base64?: string  // store for auth reuse
}
```

## Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `REGISTER_TIMEOUT_MS` | 4000 | Max time to wait for centered face |
| `CAPTURE_INTERVAL_MS` | 20 | Frame capture interval (unchanged) |
| `YAW_CENTER` | 10 | Max yaw for centered face (from fusion.ts) |
| `AUTH_FRAME_COUNT` | 3 | Number of frames to average for auth |

## Error Handling

### Registration Errors

| Error | Condition | Message |
|-------|-----------|---------|
| No face detected | After 4s with no face | "Không phát hiện khuôn mặt" |
| Face not centered | After 4s with no centered face | "Không tìm thấy khuôn mặt nhìn thẳng" |
| Enrollment failed | API error | Server error message |

### Authentication Errors

| Error | Condition | Message |
|-------|-----------|---------|
| No good frames | Less than 3 centered frames with high passive score | "Không đủ khung hình tốt để xác thực" |
| Auth failed | avg_similarity < threshold | "Xác thực thất bại" |

## Testing Checklist

- [ ] Registration: centered face → enrolls successfully
- [ ] Registration: face turned → waits for centered
- [ ] Registration: 4s timeout with no centered face → shows error
- [ ] Authentication: uses existing liveness frames
- [ ] Authentication: 3 frames averaged correctly
- [ ] Authentication: passes when avg_similarity >= 0.50

## Files to Modify

1. `apps/web/src/session/useRegister.ts` - Add yaw check, timer, timeout
2. `apps/web/src/session/useSession.ts` - Modify authenticate(), store base64 in frames
3. `apps/web/src/session/fusion.ts` - Add helper to select best frames
4. `apps/web/src/session/types.ts` - Add image_base64 to FrameRecord
5. `apps/web/src/session/RegisterView.tsx` - Update UI for new states
6. `apps/web/src/session/ResultView.tsx` - Show auth similarity

## Implementation Order

1. Add `image_base64` to FrameRecord type
2. Store base64 in each frame during liveness capture
3. Add `selectBestFramesForAuth` in fusion.ts
4. Modify authenticate() in useSession.ts
5. Modify useRegister.ts for centered-face detection
6. Update UI components