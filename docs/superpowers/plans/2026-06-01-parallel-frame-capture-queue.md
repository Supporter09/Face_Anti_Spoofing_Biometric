# Parallel Frame Capture Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace serial `inFlightRef` mechanism with queue-based parallel request system to achieve 17-20 fps (from current ~10 fps)

**Architecture:** Queue-based system with configurable NUM_WORKERS and MAX_QUEUE_SIZE. Timer fires every 20ms, adds frames to queue if space available. Workers pull from queue and process in parallel.

**Tech Stack:** React (hooks: useRef, useCallback, useEffect), TypeScript

---

## File Structure

- Modify: `apps/web/src/session/useSession.ts` — main implementation (replace inFlightRef logic)

---

## Implementation Tasks

### Task 1: Add Configuration Constants

**Files:**
- Modify: `apps/web/src/session/useSession.ts:1-22`

- [ ] **Step 1: Add configuration constants after existing constants**

```typescript
// Queue configuration
const NUM_WORKERS = 2
const MAX_QUEUE_SIZE = 5
```

- [ ] **Step 2: Add type definitions after imports**

```typescript
interface QueuedFrame {
  id: number
  imageBase64: string
  timestamp: number
  phase: FrameRecord['phase']
  turn_A_dir: TurnDirection
  resolve: (frame: FrameRecord) => void
  reject: (error: Error) => void
}
```

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/session/useSession.ts
git commit -m "feat: add queue configuration constants and QueuedFrame type"
```

---

### Task 2: Replace inFlightRef with Queue System

**Files:**
- Modify: `apps/web/src/session/useSession.ts:94-100`

- [ ] **Step 1: Replace inFlightRef with queue refs**

Change from:
```typescript
const inFlightRef = useRef(false)
```

To:
```typescript
const queueRef = useRef<QueuedFrame[]>([])
const activeWorkersRef = useRef(0)
const frameIdRef = useRef(0)
```

- [ ] **Step 2: Update reset function**

Change from:
```typescript
const reset = useCallback(() => {
  inFlightRef.current = false
```

To:
```typescript
const reset = useCallback(() => {
  queueRef.current = []
  activeWorkersRef.current = 0
```

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/session/useSession.ts
git commit -m "refactor: replace inFlightRef with queue-based refs"
```

---

### Task 3: Implement Worker Function

**Files:**
- Modify: `apps/web/src/session/useSession.ts` (add new function after captureFrameBase64)

- [ ] **Step 1: Add processQueue function before captureAndSend**

```typescript
const processQueue = useCallback(() => {
  const processNext = async () => {
    const queue = queueRef.current
    const current = stateRef.current
    
    // Check if we should stop
    if (!isActivePhase(current.phase) || queue.length === 0) {
      activeWorkersRef.current = Math.max(0, activeWorkersRef.current - 1)
      return
    }
    
    // Get next frame from queue
    const queued = queue.shift()
    if (!queued) {
      activeWorkersRef.current = Math.max(0, activeWorkersRef.current - 1)
      return
    }
    
    try {
      const frameEndpoint = captureDebug
        ? `/v1/liveness/frame/debug?session_id=${encodeURIComponent(captureSessionId)}`
        : '/v1/liveness/frame'
      const response = await fetch(`${API_BASE}${frameEndpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: queued.imageBase64 }),
      })
      if (!response.ok) throw new Error(`API request failed with status ${response.status}`)

      const payload = (await response.json()) as FrameApiResponse
      const rawYaw = payload.yaw_deg
      const smoothedYaw =
        rawYaw === null
          ? null
          : smoothedYawRef.current === null
            ? rawYaw
            : 0.5 * smoothedYawRef.current + 0.5 * rawYaw
      smoothedYawRef.current = smoothedYaw

      const frame: FrameRecord = {
        ts_ms: Date.now() - sessionStartedAtRef.current,
        phase: queued.phase,
        face_detected: payload.face_detected,
        passive_score: payload.liveness_score,
        yaw_deg: smoothedYaw,
        pose_ok: payload.pose_ok,
      }

      // Check phase advancement criteria
      const criterionMet =
        payload.face_detected && payload.pose_ok && phaseCriterionMet(queued.phase, smoothedYaw, queued.turn_A_dir)
      
      setState((latest) => ({
        ...latest,
        frames: [...latest.frames, frame],
        latest_yaw: smoothedYaw,
        latest_passive: payload.liveness_score,
        face_detected: payload.face_detected,
        latest_bbox: payload.face_bbox_xyxy ?? null,
        latest_landmarks: (payload.face_landmarks as [number, number][] | null) ?? null,
        error: null,
      }))

      // Update consecutive counter and check phase advancement
      const currentState = stateRef.current
      if (!isDiagnose && criterionMet && isActivePhase(currentState.phase)) {
        const newConsecutive = (consecutiveRef.current || 0) + 1
        consecutiveRef.current = newConsecutive
        if (newConsecutive >= REQUIRED_CONSECUTIVE_FRAMES) {
          advanceFrom(currentState.phase)
        }
      }

      queued.resolve(frame)
    } catch (caughtError) {
      const error = caughtError instanceof Error ? caughtError : new Error('Unexpected error')
      setState((latest) => ({
        ...latest,
        error: error.message,
      }))
      queued.reject(error)
    } finally {
      // Continue processing if queue has more items
      if (queueRef.current.length > 0) {
        processNext()
      } else {
        activeWorkersRef.current = Math.max(0, activeWorkersRef.current - 1)
      }
    }
  }
  
  // Start processing if workers available
  if (activeWorkersRef.current < NUM_WORKERS) {
    activeWorkersRef.current++
    processNext()
  }
}, [captureDebug, captureSessionId, isDiagnose, phaseCriterionMet, advanceFrom])
```

- [ ] **Step 2: Commit**

```bash
git add apps/web/src/session/useSession.ts
git commit -m "feat: implement processQueue worker function"
```

---

### Task 4: Rewrite captureAndSend for Queue

**Files:**
- Modify: `apps/web/src/session/useSession.ts:171-236`

- [ ] **Step 1: Replace captureAndSend implementation**

Replace the entire captureAndSend function with:

```typescript
const captureAndSend = useCallback(() => {
  const current = stateRef.current
  if (!isActivePhase(current.phase) || !current.turn_A_dir) return

  // Check queue capacity - drop if full
  if (queueRef.current.length >= MAX_QUEUE_SIZE) return

  const imageBase64 = captureFrameBase64()
  if (!imageBase64) return

  // Create promise-based frame capture
  return new Promise<void>((resolve, reject) => {
    const queued: QueuedFrame = {
      id: frameIdRef.current++,
      imageBase64,
      timestamp: Date.now(),
      phase: current.phase,
      turn_A_dir: current.turn_A_dir,
      resolve: () => resolve(),
      reject,
    }
    queueRef.current.push(queued)
    processQueue()
  })
}, [captureFrameBase64, processQueue])
```

- [ ] **Step 2: Update dependencies array**

The dependencies should be:
```typescript
}, [captureFrameBase64, processQueue])
```

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/session/useSession.ts
git commit -m "feat: rewrite captureAndSend to use queue system"
```

---

### Task 5: Update Timer to Not Await

**Files:**
- Modify: `apps/web/src/session/useSession.ts:256-271`

- [ ] **Step 1: Update timer effect to not await captureAndSend**

Current:
```typescript
void captureAndSend()
```

Should remain as-is (fire-and-forget, no await needed since we're using queue):

- [ ] **Step 2: Commit**

```bash
git add apps/web/src/session/useSession.ts
git commit -m "chore: update timer effect for queue-based capture"
```

---

### Task 6: Test and Verify

**Files:**
- Test: Run the app and verify frame count

- [ ] **Step 1: Start backend**

Run: `scripts/run_backend_demo.bat` (in terminal 1)

- [ ] **Step 2: Start frontend**

Run: `scripts/run_frontend_demo.bat` (in terminal 2)

- [ ] **Step 3: Run a session and check frame count**

Expected: Frame count should be ~150-200 (vs previous 60-80)

- [ ] **Step 4: Verify no errors in console**

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/session/useSession.ts
git commit -m "test: verify queue implementation works correctly"
```

---

## Summary

| Task | Description |
|------|-------------|
| 1 | Add NUM_WORKERS=2, MAX_QUEUE_SIZE=5 constants and QueuedFrame type |
| 2 | Replace inFlightRef with queueRef, activeWorkersRef, frameIdRef |
| 3 | Implement processQueue worker function |
| 4 | Rewrite captureAndSend to enqueue frames |
| 5 | Update timer effect (minimal change) |
| 6 | Test and verify |

---

## Plan Complete

**Plan saved to:** `docs/superpowers/plans/2026-06-01-parallel-frame-capture-queue.md`

Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?