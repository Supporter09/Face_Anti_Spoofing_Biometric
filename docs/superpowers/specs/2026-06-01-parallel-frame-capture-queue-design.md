# Parallel Frame Capture Queue Design

## Problem

The current liveness detection system captures ~60-80 frames in 8 seconds (~10 fps) instead of the target 17-20 fps due to serial request handling that blocks new captures while waiting for API responses.

## Solution

Replace the serial `inFlightRef` mechanism with a queue-based parallel request system.

## Architecture

```
Timer fires (20ms)
       │
       ▼
┌──────────────────┐
│ Queue (max 5)    │
│ [frame1, frame2] │
└──────────────────┘
       │
  ┌────┴────┐
  ▼         ▼
Worker 1  Worker 2
(API call) (API call)
  │         │
  ▼         ▼
  └────┬────┘
       │
       ▼
  Add to frames[]
```

## Configuration

| Variable | Value | Description |
|----------|-------|-------------|
| `NUM_WORKERS` | 2 | Concurrent API requests |
| `MAX_QUEUE_SIZE` | 5 | Maximum frames in queue |
| `CAPTURE_INTERVAL_MS` | 20 | Timer interval (unchanged) |

## Data Structures

```typescript
interface QueuedFrame {
  id: number
  imageBase64: string
  timestamp: number
  phase: Phase
  resolve: (frame: FrameRecord) => void
  reject: (error: Error) => void
}

// Refs
const queueRef = useRef<QueuedFrame[]>([])
const activeWorkersRef = useRef(0)
```

## Algorithm

### Timer Tick (every 20ms)
```
1. Get current state from stateRef
2. Check if phase is active (forward/turn_A/center_1/turn_B)
3. If queue.length >= MAX_QUEUE_SIZE → DROP frame
4. Else → capture frame, add to queue
5. If activeWorkers < NUM_WORKERS → start worker
```

### Worker Loop
```
1. While queue not empty and activeWorkers < NUM_WORKERS:
   - Pop frame from queue
   - activeWorkers++
   - Call API
   - On response: create FrameRecord, resolve
   - activeWorkers--
2. If queue empty → exit (timer will restart when new frame added)
```

## Error Handling

- **API failure**: Log error to state, continue with next frame
- **Queue overflow**: Drop silently (expected during bursts)
- **Worker crash**: Decrement activeWorkers, let timer restart

## Frame Counting

- Frames are added to state as they complete (not when queued)
- `frame_count` reflects actual processed frames
- Duration calculation unchanged

## Expected Results

| Metric | Before | After |
|--------|--------|-------|
| FPS | ~10 | ~17-20 |
| Frames per session | 60-80 | ~150-200 |
| Latency per frame | ~70ms | ~30-50ms |

## Files Modified

- `apps/web/src/session/useSession.ts` — main implementation

## Testing

- Unit test queue logic (enqueue/dequeue/overflow)
- Integration test with actual API
- Verify frame count increases ~2x
- Verify no memory leaks from queue

## Dependencies

- None — pure React state/refs implementation