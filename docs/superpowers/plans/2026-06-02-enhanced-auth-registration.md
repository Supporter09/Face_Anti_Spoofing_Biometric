# Enhanced Registration + Authentication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement face centering check during registration, and use liveness frames for authentication instead of capturing a new frame.

**Architecture:** Registration uses continuous frame capture (20ms interval) with yaw check - identical to liveness flow. Authentication reuses ~150-200 frames captured during liveness, selects top 3 best frames, and averages similarity scores.

**Tech Stack:** React (hooks: useRef, useCallback, useEffect), TypeScript, FastAPI backend (existing endpoints)

---

## File Structure

| File | Responsibility |
|------|----------------|
| `apps/web/src/session/types.ts` | Add `image_base64` to FrameRecord |
| `apps/web/src/session/useSession.ts` | Store base64 in frames, modify authenticate() |
| `apps/web/src/session/fusion.ts` | Add `selectBestFramesForAuth` helper |
| `apps/web/src/session/useRegister.ts` | Add yaw check, timer, 4s timeout |
| `apps/web/src/session/RegisterView.tsx` | Update UI for yaw feedback |
| `apps/web/src/session/ResultView.tsx` | Show auth similarity |

---

## Implementation Tasks

### Task 1: Add image_base64 to FrameRecord type

**Files:**
- Modify: `apps/web/src/session/types.ts`

- [ ] **Step 1: Add image_base64 field to FrameRecord interface**

```typescript
// In types.ts, add image_base64 to FrameRecord
export interface FrameRecord {
  ts_ms: number
  phase: Phase
  face_detected: boolean
  passive_score: number
  yaw_deg: number | null
  pose_ok: boolean
  // NEW: store base64 for auth reuse
  image_base64?: string
}
```

- [ ] **Step 2: Commit**

```bash
git add apps/web/src/session/types.ts
git commit -m "feat(types): add image_base64 to FrameRecord for auth reuse"
```

---

### Task 2: Store base64 in each frame during liveness capture

**Files:**
- Modify: `apps/web/src/session/useSession.ts:185-250`

- [ ] **Step 1: Store image_base64 in the frame record**

In `processQueue`, where the FrameRecord is created, add the image_base64:

```typescript
const frame: FrameRecord = {
  ts_ms: Date.now() - sessionStartedAtRef.current,
  phase: currentPhase,
  face_detected: payload.face_detected,
  passive_score: payload.liveness_score,
  yaw_deg: smoothedYaw,
  pose_ok: payload.pose_ok,
  // NEW: store the base64 that was sent
  image_base64: queued.imageBase64,
}
```

- [ ] **Step 2: Commit**

```bash
git add apps/web/src/session/useSession.ts
git commit -m "feat(useSession): store image_base64 in each frame for auth reuse"
```

---

### Task 3: Add selectBestFramesForAuth helper in fusion.ts

**Files:**
- Modify: `apps/web/src/session/fusion.ts`

**Note:** Ensure `T_PASSIVE` is already exported from fusion.ts (it is - used in evaluateChallenge).

- [ ] **Step 1: Add helper function**

```typescript
import type { FrameRecord } from './types'

/**
 * Select the best N frames for authentication.
 * Filters for: face_detected, centered (abs(relative_yaw) <= YAW_CENTER), high passive score.
 * Returns top N frames sorted by passive_score descending.
 */
export function selectBestFramesForAuth(
  frames: FrameRecord[],
  yawBaseline: number,
  count: number = 3
): FrameRecord[] {
  const rel = (yaw: number | null) => yaw === null ? null : yaw - yawBaseline

  const goodFrames = frames.filter((f) => {
    if (!f.face_detected || !f.image_base64) return false
    if (f.passive_score < T_PASSIVE) return false
    const relativeYaw = rel(f.yaw_deg)
    if (relativeYaw === null) return false
    return Math.abs(relativeYaw) <= YAW_CENTER
  })

  // Sort by passive_score descending
  goodFrames.sort((a, b) => b.passive_score - a.passive_score)

  return goodFrames.slice(0, count)
}
```

- [ ] **Step 2: Commit**

```bash
git add apps/web/src/session/fusion.ts
git commit -m "feat(fusion): add selectBestFramesForAuth helper"
```

---

### Task 4: Modify authenticate() to use liveness frames

**Files:**
- Modify: `apps/web/src/session/useSession.ts:300-340`

- [ ] **Step 1: Rewrite authenticate function**

First, update the import to include the new helper:

```typescript
import { computeVerdict, selectBestFramesForAuth, YAW_CENTER, YAW_TARGET } from './fusion'
```

Then replace the current `authenticate` function:

```typescript
const authenticate = useCallback(async () => {
  const { frames, turn_A_dir } = stateRef.current
  if (!turn_A_dir || frames.length === 0) return

  setState((s) => ({
    ...s,
    auth_status: 'verifying',
    auth_message: 'Đang xác thực...',
    identified_user: null,
  }))

  try {
    // Calculate yaw baseline from forward phase
    const forwardFrames = frames.filter((f) => f.phase === 'forward' && f.yaw_deg !== null)
    const yawBaseline = forwardFrames.length > 0
      ? forwardFrames.reduce((a, b) => a + b.yaw_deg!, 0) / forwardFrames.length
      : 0

    // Select top 3 best frames for auth
    const bestFrames = selectBestFramesForAuth(frames, yawBaseline, 3)

    if (bestFrames.length === 0) {
      throw new Error('Không đủ khung hình tốt để xác thực')
    }

    // Run auth requests in parallel
    const authPromises = bestFrames.map((frame) =>
      fetch(`${API_BASE}/v1/auth/identify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: frame.image_base64 }),
      }).then((res) => res.json())
    )

    const results = await Promise.all(authPromises)

    // Average the similarities
    const similarities = results.map((r) => r.similarity ?? 0)
    const avgSimilarity = similarities.reduce((a, b) => a + b, 0) / similarities.length

    // Use the identify result from the best frame for user_id
    const bestResult = results[0]

    setState((s) => ({
      ...s,
      auth_status: bestResult.authenticated ? 'authenticated' : 'failed',
      auth_message: bestResult.message,
      similarity: avgSimilarity,
      identified_user: bestResult.authenticated ? bestResult.user_id : null,
    }))
  } catch (err) {
    setState((s) => ({
      ...s,
      auth_status: 'failed',
      auth_message: err instanceof Error ? err.message : 'Authentication failed',
      identified_user: null,
    }))
  }
}, [])
```

- [ ] **Step 2: Commit**

```bash
git add apps/web/src/session/useSession.ts
git commit -m "feat(useSession): use liveness frames for auth, average 3 similarities"
```

---

### Task 5: Add yaw check to registration

**Files:**
- Modify: `apps/web/src/session/useRegister.ts`

- [ ] **Step 1: Add imports and constants**

```typescript
import { useCallback, useEffect, useRef, useState } from 'react'

import { YAW_CENTER } from './fusion'
import type { FrameApiResponse } from './types'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'
const CAPTURE_INTERVAL_MS = 20
const REGISTER_TIMEOUT_MS = 4000

// Add to RegisterState interface
export interface RegisterState {
  phase: RegisterPhase
  countdown: number
  userId: string
  error: string | null
  capturedFrame: string | null
  // NEW:
  latest_yaw: number | null
  face_detected: boolean
}
```

- [ ] **Step 2: Update initialState**

```typescript
function initialState(): RegisterState {
  return {
    phase: 'idle',
    countdown: 3,
    userId: '',
    error: null,
    capturedFrame: null,
    latest_yaw: null,
    face_detected: false,
  }
}
```

- [ ] **Step 3: Add frame capture with yaw check**

Replace the countdown logic with continuous capture:

```typescript
// Replace the countdown useEffect with this:
useEffect(() => {
  if (state.phase !== 'countdown') return

  const startedAt = Date.now()
  const timerRef = { current: null as number | null }

  const tick = () => {
    const elapsed = Date.now() - startedAt
    if (elapsed >= REGISTER_TIMEOUT_MS) {
      // Timeout - no centered face found
      if (timerRef.current) clearInterval(timerRef.current)
      setState((s) => ({
        ...s,
        phase: 'error',
        error: 'Không tìm thấy khuôn mặt nhìn thẳng',
      }))
      return
    }

    const imageBase64 = captureFrameBase64()
    if (!imageBase64) return

    // Send to liveness endpoint to get yaw
    fetch(`${API_BASE}/v1/liveness/frame`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_base64: imageBase64 }),
    })
      .then((res) => res.json())
      .then((payload) => {
        const yaw = payload.yaw_deg
        const faceDetected = payload.face_detected

        setState((s) => ({
          ...s,
          latest_yaw: yaw,
          face_detected: faceDetected,
        }))

        // Check if face is centered
        if (faceDetected && yaw !== null && Math.abs(yaw) <= YAW_CENTER) {
          if (timerRef.current) clearInterval(timerRef.current)
          // Good frame - enroll
          setState((s) => ({
            ...s,
            capturedFrame: `data:image/jpeg;base64,${imageBase64}`,
          }))
          void enroll(imageBase64, stateRef.current.userId)
        }
      })
      .catch(() => {
        // Continue on error
      })
  }

  timerRef.current = window.setInterval(tick, CAPTURE_INTERVAL_MS)

  return () => {
    if (timerRef.current) clearInterval(timerRef.current)
  }
}, [captureFrameBase64, enroll])
```

- [ ] **Step 4: Add countdown timer (separate from capture)**

```typescript
// Add a separate effect for countdown display
useEffect(() => {
  if (state.phase !== 'countdown') return

  const startedAt = Date.now()
  const id = window.setInterval(() => {
    const elapsed = Date.now() - startedAt
    const next = Math.max(0, 3 - Math.floor(elapsed / 1000))
    setState((s) => ({ ...s, countdown: next }))
  }, 100)

  return () => window.clearInterval(id)
}, [state.phase])
```

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/session/useRegister.ts
git commit -m "feat(useRegister): add yaw check for face centering during registration"
```

---

### Task 6: Update RegisterView UI for yaw feedback

**Files:**
- Modify: `apps/web/src/session/RegisterView.tsx`

- [ ] **Step 1: Add yaw indicator**

Add after the video area:

```tsx
{state.phase === 'countdown' && state.latest_yaw !== null && (
  <div className="session-yaw-indicator">
    Góc quay: {state.latest_yaw.toFixed(1)}° (
    {Math.abs(state.latest_yaw) <= 10 ? '✓ Đã căn giữa' : '↔ Cần căn giữa'})
  </div>
)}
```

- [ ] **Step 2: Commit**

```bash
git add apps/web/src/session/RegisterView.tsx
git commit -m "feat(RegisterView): add yaw feedback indicator"
```

---

### Task 7: Update ResultView to show auth similarity

**Files:**
- Modify: `apps/web/src/session/ResultView.tsx`

- [ ] **Step 1: Add auth similarity display**

```tsx
// Add to ResultView props
interface Props {
  verdict: Verdict
  turn_A_dir: TurnDirection
  onRetry: () => void
  auth_status?: SessionState['auth_status']
  auth_message?: string | null
  similarity?: number | null
}
```

Add in the stats section (after existing fields):

```tsx
{auth_status !== 'idle' && (
  <>
    <div>
      <dt>Xác thực</dt>
      <dd>
        {auth_status === 'verifying' ? '🔐 Đang xác thực...' :
         auth_status === 'authenticated' ? '✅ Thành công' :
         auth_status === 'failed' ? '❌ Thất bại' : '—'}
      </dd>
    </div>
    {similarity !== null && (
      <div>
        <dt>Độ tương tự</dt>
        <dd>{(similarity * 100).toFixed(1)}%</dd>
      </div>
    )}
    {auth_message && (
      <p className="auth-message">{auth_message}</p>
    )}
  </>
)}
```

- [ ] **Step 2: Pass auth props from SessionView**

In SessionView, pass auth props to ResultView:

```tsx
<ResultView
  verdict={state.verdict}
  turn_A_dir={state.turn_A_dir}
  onRetry={reset}
  auth_status={state.auth_status}
  auth_message={state.auth_message}
  similarity={state.similarity}
/>
```

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/session/ResultView.tsx apps/web/src/session/SessionView.tsx
git commit -m "feat(ResultView): show auth similarity and status"
```

---

## Summary

| Task | Description |
|------|-------------|
| 1 | Add `image_base64` to FrameRecord type |
| 2 | Store base64 in each captured frame |
| 3 | Add `selectBestFramesForAuth` helper |
| 4 | Modify `authenticate()` to use top 3 frames, average similarities |
| 5 | Add yaw check to registration with 4s timeout |
| 6 | Update RegisterView UI with yaw feedback |
| 7 | Update ResultView to show auth status and similarity |

---

## Plan Complete

**Plan saved to:** `docs/superpowers/plans/2026-06-02-enhanced-auth-registration.md`

**Two execution options:**

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?