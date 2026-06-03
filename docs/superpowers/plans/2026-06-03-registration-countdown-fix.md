# Registration Countdown Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix registration flow to match liveness detection: 3-second countdown for preparation, then 4-second window to find centered face.

**Architecture:** Change useRegister.ts logic to:
- 0-3s: Show countdown only (no enrollment)
- 3-7s: Check for centered face (yaw check)
- After 7s: Fail with error

**Tech Stack:** React (hooks: useRef, useCallback, useEffect), TypeScript

---

## File Structure

| File | Responsibility |
|------|----------------|
| `apps/web/src/session/useRegister.ts` | Fix registration countdown logic |

---

## Implementation Tasks

### Task 1: Update timeout constant and add countdown phase constant

**Files:**
- Modify: `apps/web/src/session/useRegister.ts:8`

- [ ] **Step 1: Update REGISTER_TIMEOUT_MS to account for countdown**

Current: `REGISTER_TIMEOUT_MS = 4000` (includes countdown time)
New: `REGISTER_TIMEOUT_MS = 7000` (3s countdown + 4s yaw check)

```typescript
// Change from:
const REGISTER_TIMEOUT_MS = 4000

// To:
const REGISTER_TIMEOUT_MS = 7000  // 3s countdown + 4s yaw check window
```

- [ ] **Step 2: Add constant for countdown duration**

```typescript
// Add after CAPTURE_INTERVAL_MS:
const COUNTDOWN_DURATION_MS = 3000  // 3 seconds for user to prepare
```

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/session/useRegister.ts
git commit -m "fix(useRegister): update timeout constants for proper countdown + yaw check"
```

---

### Task 2: Fix the tick() logic to not enroll at countdown=0

**Files:**
- Modify: `apps/web/src/session/useRegister.ts:135-165`

- [ ] **Step 1: Remove the early enroll at countdown=0**

Current code (lines 141-154) enrolls regardless of yaw when countdown reaches 0:

```typescript
// CURRENT (WRONG):
// If countdown completed (reached 0), enroll with current frame regardless of yaw
if (countdownFromElapsed === 0 && !hasEnrolled) {
  hasEnrolled = true
  if (timerRef.current) clearInterval(timerRef.current)
  const imageBase64 = captureFrameBase64()
  if (imageBase64) {
    setState((s) => ({
      ...s,
      capturedFrame: `data:image/jpeg;base64,${imageBase64}`,
    }))
    void enroll(imageBase64, currentUserId)
  }
  return
}
```

Replace with:

```typescript
// NEW (CORRECT):
// During countdown (0-3s), just capture and check yaw - don't enroll yet
// Only enroll if face is centered AND countdown has completed
const isCountdownPhase = elapsed < COUNTDOWN_DURATION_MS
const countdownFromElapsed = Math.max(0, 3 - Math.floor(elapsed / 1000))

// If countdown completed and face is centered, enroll
if (!hasEnrolled && countdownFromElapsed === 0 && faceDetected && yaw !== null && Math.abs(yaw) <= YAW_CENTER) {
  hasEnrolled = true
  if (timerRef.current) clearInterval(timerRef.current)
  setState((s) => ({
    ...s,
    capturedFrame: `data:image/jpeg;base64,${imageBase64}`,
  }))
  void enroll(imageBase64, currentUserId)
  return
}

// If countdown completed but no centered face yet, continue (timeout will catch at 7s)
if (!isCountdownPhase && countdownFromElapsed === 0) {
  // Countdown done, waiting for centered face...
  // Let the loop continue - will timeout at 7s if no centered face
}
```

- [ ] **Step 2: Commit**

```bash
git add apps/web/src/session/useRegister.ts
git commit -m "fix(useRegister): enroll only after countdown + centered face"
```

---

### Task 3: Update the yaw check logic to work after countdown

**Files:**
- Modify: `apps/web/src/session/useRegister.ts:186-200`

- [ ] **Step 1: Update yaw check condition**

Current code checks yaw during countdown but enrolls at countdown=0 regardless. Need to ensure:
- During countdown (0-3s): track yaw, show feedback, don't enroll
- After countdown (3-7s): if centered face found, enroll

```typescript
// Update the existing yaw check block:
.then((payload: FrameApiResponse) => {
  const yaw = payload.yaw_deg
  const faceDetected = payload.face_detected

  setState((s) => ({
    ...s,
    latest_yaw: yaw,
    face_detected: faceDetected,
  }))

  // Check if we should enroll:
  // 1. Not enrolled yet
  // 2. Countdown has completed (elapsed >= COUNTDOWN_DURATION_MS)
  // 3. Face detected and centered (abs(yaw) <= YAW_CENTER)
  const countdownComplete = elapsed >= COUNTDOWN_DURATION_MS
  if (!hasEnrolled && countdownComplete && faceDetected && yaw !== null && Math.abs(yaw) <= YAW_CENTER) {
    hasEnrolled = true
    if (timerRef.current) clearInterval(timerRef.current)
    setState((s) => ({
      ...s,
      capturedFrame: `data:image/jpeg;base64,${imageBase64}`,
    }))
    void enroll(imageBase64, currentUserId)
  }
})
```

- [ ] **Step 2: Commit**

```bash
git add apps/web/src/session/useRegister.ts
git commit -m "fix(useRegister): check yaw after countdown completes"
```

---

### Task 4: Test the flow

**Files:**
- Test: Manual testing in browser

- [ ] **Step 1: Start registration**
- [ ] **Step 2: Verify countdown shows 3, 2, 1**
- [ ] **Step 3: During countdown, yaw indicator shows but NO enrollment**
- [ ] **Step 4: After countdown, if face centered → enrollment happens**
- [ ] **Step 5: If not centered within 4 more seconds (total 7s) → error shows**

---

## Summary

| Task | Description |
|------|-------------|
| 1 | Update REGISTER_TIMEOUT_MS to 7000, add COUNTDOWN_DURATION_MS |
| 2 | Remove early enroll at countdown=0, only enroll after countdown + centered |
| 3 | Update yaw check to work after countdown completes |
| 4 | Manual test verification |

---

## Plan Complete

**Plan saved to:** `docs/superpowers/plans/2026-06-03-registration-countdown-fix.md`

**Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**