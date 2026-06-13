# Multi-Frame Registration: Minimum Frame Gap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add minimum gap (100ms) between captured frames in multi-frame registration for temporal diversity

**Architecture:** Add `useRef` to track last capture timestamp, check gap before each capture

**Tech Stack:** TypeScript, React hooks

---

### Task 1: Add MIN_FRAME_GAP_MS constant

**Files:**
- Modify: `apps/web/src/session/useRegister.ts`

- [ ] **Step 1: Add constant after existing constants**

Locate line 10 (after `REGISTRATION_CAPTURE_TIMEOUT_MS = 4000`) and add:

```typescript
const MIN_FRAME_GAP_MS = 100  // Minimum gap between captured frames (ms)
```

- [ ] **Step 2: Commit**

```bash
git add apps/web/src/session/useRegister.ts
git commit -m "feat(useRegister): add MIN_FRAME_GAP_MS constant"
```

---

### Task 2: Add lastCapturedRef with useRef

**Files:**
- Modify: `apps/web/src/session/useRegister.ts`

- [ ] **Step 1: Add useRef import if needed**

Check existing imports - `useRef` should already be imported from React.

- [ ] **Step 2: Add lastCapturedRef after stateRef**

Locate line 28 (after `const stateRef = useRef(initialState)`) and add:

```typescript
const lastCapturedRef = useRef<number>(0)  // Timestamp of last captured frame
```

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/session/useRegister.ts
git commit -m "feat(useRegister): add lastCapturedRef to track capture timestamp"
```

---

### Task 3: Update frame capture condition with gap check

**Files:**
- Modify: `apps/web/src/session/useRegister.ts:241-247`

- [ ] **Step 1: Update capture condition**

Find the capture condition at line 241:
```typescript
if (!hasEnrolled && countdownComplete && isCentered && !currentFrames.includes(imageBase64)) {
```

Replace with:
```typescript
// Check both: centered face AND minimum gap passed
const now = Date.now()
const gapSufficient = now - lastCapturedRef.current >= MIN_FRAME_GAP_MS

if (!hasEnrolled && countdownComplete && isCentered && gapSufficient && !currentFrames.includes(imageBase64)) {
  lastCapturedRef.current = now  // Update last capture time
```

- [ ] **Step 2: Run TypeScript check**

```bash
cd apps/web && npx tsc --noEmit
```

Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/session/useRegister.ts
git commit -m "feat(useRegister): enforce minimum 100ms gap between captured frames"
```

---

### Task 4: Verify and test

**Files:**
- Verify: `apps/web/src/session/useRegister.ts`

- [ ] **Step 1: Review the changes**

Check that all modifications are correct:
- `MIN_FRAME_GAP_MS = 100` constant added
- `lastCapturedRef` initialized to 0
- Gap check added before frame capture
- `lastCapturedRef.current = now` updates after each capture

- [ ] **Step 2: Final TypeScript check**

```bash
cd apps/web && npx tsc --noEmit
```

Expected: No errors

- [ ] **Step 3: Commit final**

```bash
git add apps/web/src/session/useRegister.ts
git commit -m "feat(useRegister): multi-frame registration with temporal diversity"
```

---

## Spec Coverage

| Requirement | Task |
|-------------|------|
| Add MIN_FRAME_GAP_MS constant (100ms) | Task 1 |
| Track last capture timestamp | Task 2 |
| Check gap before capture | Task 3 |
| Fallback behavior (1-4 frames) | Already implemented |
| Timeout (4s) unchanged | Already implemented |

## Placeholder Scan

- ✅ No "TBD" or "TODO"
- ✅ All code provided
- ✅ No placeholder references