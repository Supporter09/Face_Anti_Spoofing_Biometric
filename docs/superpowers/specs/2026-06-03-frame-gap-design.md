# Multi-Frame Registration: Minimum Frame Gap Design

**Date:** 2026-06-03  
**Feature:** Multi-frame registration with temporal diversity  
**Status:** Draft

## Problem

The current multi-frame registration captures 5 consecutive frames at 20ms intervals. Since each frame is only 20ms apart, the captured images are nearly identical, making the multi-frame approach ineffective for noise reduction.

## Solution

Add a minimum time gap between captured frames while still scanning every 20ms. This ensures temporal diversity among the 5 captured frames.

## Design

### Constants

```typescript
// Existing constants
const CAPTURE_INTERVAL_MS = 20       // How often we check for frames (unchanged)
const REGISTRATION_FRAME_COUNT = 5   // Number of frames to capture
const REGISTRATION_CAPTURE_TIMEOUT_MS = 4000

// New constant
const MIN_FRAME_GAP_MS = 100          // Minimum gap between captured frames (ms)
```

### Implementation

1. **Add `useRef` to track last capture timestamp**
   ```typescript
   const lastCapturedRef = useRef<number>(0)
   ```

2. **Modify frame capture logic in `tick()`**
   ```typescript
   // Check both: centered face AND minimum gap passed
   const now = Date.now()
   const gapSufficient = now - lastCapturedRef.current >= MIN_FRAME_GAP_MS
   
   if (!hasEnrolled && countdownComplete && isCentered && gapSufficient && !currentFrames.includes(imageBase64)) {
     lastCapturedRef.current = now  // Update last capture time
     // ... rest of capture logic
   }
   ```

### Behavior

| Scenario | Result |
|----------|--------|
| tick runs every 20ms | Unchanged (still scans frequently) |
| Centered face found, but <100ms since last | Skip capture |
| Centered face found, ≥100ms since last | Capture frame |
| 5 frames captured | Stop and enroll |
| Timeout (4s) with <5 frames | Fallback: enroll with available frames |

**Expected timing for 5 frames:**
- Frame 1: t=0ms
- Frame 2: t≥100ms
- Frame 3: t≥200ms
- Frame 4: t≥300ms
- Frame 5: t≥400ms

Total spread: ~400ms minimum (vs ~80ms before)

## Files to Modify

- `apps/web/src/session/useRegister.ts`
  - Add `MIN_FRAME_GAP_MS` constant
  - Add `lastCapturedRef` with `useRef`
  - Update capture condition in `tick()`

## Testing

1. Verify frames are captured with ~100ms gap (not 20ms)
2. Verify fallback still works with 1-4 frames
3. Verify timeout behavior unchanged