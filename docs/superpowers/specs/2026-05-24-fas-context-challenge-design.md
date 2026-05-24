# Face Anti-Spoofing — Context-Aware Passive + Active Challenge Hybrid

**Status**: Approved design, ready for implementation plan
**Date**: 2026-05-24
**Scope**: Liveness detection only (authentication module owned by separate teammate)
**Timeline**: 2.5 weeks
**Branch target**: `feat/fas-context-challenge`

---

## 1. Problem Statement

### Current state

The existing MVP (commit `b2e37de`) uses:

- **Detector**: InsightFace face detection on backend, returns bbox + 5 landmarks.
- **Liveness model**: MobileNetV2 trained on CelebA-Spoof, input 112×112, eating only the **tight face crop**. Best val ACER 0.0016 on CelebA-Spoof.
- **Frontend**: webcam polling (300ms HTTP) or WebSocket streaming, score smoothing window of 5.

### Diagnosed root cause

The model only sees a tight face crop, so it loses all environmental cues (phone bezel, hands holding device, moire patterns from screen replay, paper edges, screen reflections). In 2D RGB, a real face vs a face-photo-cropped-tight are visually equivalent. CelebA-Spoof has these context cues in the original frames, but the current data prep pipeline crops them out before training.

### Observed failure modes (real webcam)

- Printed photo held in front of webcam → classified `LIVE`.
- Phone screen replay → classified `LIVE`.
- Paper mask cutout → classified `LIVE`.

### Goal

Produce a working session-based liveness check ("attendance-terminal style", not continuous monitoring) that defeats print and phone-screen replay attacks, with clear before/after metrics for the report. Authentication and identity verification are out of scope.

---

## 2. Architecture Overview

### High-level flow

```
[Webcam] ──~10fps for 3-5s──▶ [Frontend Session State Machine]
                                       │
                                       ▼  per-frame
                              [Backend /v1/liveness/frame]
                                       │
                                       ├─ InsightFace detect → bbox + 5 landmarks
                                       ├─ Crop CONTEXT (bbox×1.8, pad-to-square) → 224×224 → context MobileNetV2 → passive_score
                                       ├─ solvePnP from 5 landmarks → yaw_deg, pitch_deg
                                       └─ Return { passive_score, yaw_deg, pitch_deg, bbox, landmarks, face_detected, pose_ok }
                                       │
                                       ▼
                              [Frontend Aggregator + State Machine]
                                       │  Phases: countdown → forward → turn_A → center_1 → turn_B → evaluating
                                       ▼
                              [Frontend Fusion]
                                  verdict = LIVE iff
                                    mean(passive_score in forward-phase frames) ≥ T_passive
                                    AND challenge sequence passed (yaw thresholds + no jumps)
                                    AND face_detected ≥ 90% of frames
                                       │
                                       ▼
                              [ResultView]
```

### Components

| Layer | Component | New / Modified | Files |
|---|---|---|---|
| A — Data prep | Context-crop pipeline | Modified | `src/fas/celeba_spoof_prep.py` |
| A — Training | Notebook for context MobileNetV2 | New | `notebooks/kaggle_full_07_train_context_mobilenetv2.ipynb` |
| A — Backend | Context crop in detection, model input 224 | Modified | `src/fas/detection.py`, `src/fas/service.py`, `src/fas/liveness_model.py` |
| B — Pose | Head pose estimation from 5 landmarks | New | `src/fas/pose.py` |
| B — Backend | Add yaw/pitch to response | Modified | `src/fas/schemas.py`, `src/fas/service.py`, `services/api/app.py` |
| C — Frontend | Session state machine + UI | Major rewrite | `apps/web/src/App.tsx`, `apps/web/src/session/` |
| C — Frontend | Fusion logic | New | `apps/web/src/session/fusion.ts` |
| Eval | Ablation notebook + collection script | New | `notebooks/eval_session_ablation.ipynb`, `scripts/collect_webcam_eval.py`, `docs/WEBCAM_COLLECTION_PROTOCOL.md` |

### Backend endpoint strategy

- **Keep** existing `/v1/liveness/infer` (legacy, continuous polling clients).
- **Add** new `/v1/liveness/frame` — same payload as `/infer`, response extended with `yaw_deg`, `pitch_deg`, `pose_ok`.
- **Keep** `/ws/liveness` for debug only.
- The frontend session UI uses only `/v1/liveness/frame`.

### Stateless backend, stateful frontend

- Backend remains fully stateless — no session storage. Each frame is processed independently.
- All session state (phase, frame buffer, challenge sequence, aggregation, verdict) lives in the frontend `useSession` hook.

---

## 3. Component A — Context-Aware Passive Model

### Data preparation

- Re-crop CelebA-Spoof with `context_margin_ratio = 0.8` → bounding box expanded to ~1.8× of original.
- **Pad to square** using image mean as pad value (avoids face distortion when aspect ratio is non-square).
- Resize to **224×224** to enable ImageNet-pretrained MobileNetV2 weights.
- Output dir: `data_processed/celeba_spoof_context_224/`
- Manifests: `manifest_train.json`, `manifest_val.json`, `manifest_test.json` — **same splits as baseline** (same seed) for fair comparison.

### Code changes

**`src/fas/celeba_spoof_prep.py`** (additive):
- Add field `context_margin_ratio` to `PrepConfig` (default 0.8) — independent from `bbox_margin_ratio`.
- Add helper `pad_to_square_then_resize(image_bgr, size, pad_value)`.
- Default `image_size` for new pipeline is 224.

### Training

**Notebook**: `notebooks/kaggle_full_07_train_context_mobilenetv2.ipynb`

Cell structure:

1. Setup (pip installs, imports, paths)
2. Config (margin=0.8, image_size=224, batch=64, epochs=10, lr=1e-4)
3. Load original CelebA-Spoof annotations
4. **VIZ-1**: Sample raw images (3 live + 3 spoof, full frame with bbox overlay)
5. Re-crop with context_margin=0.8, save to context_224 dir
6. **VIZ-2**: Before/after crop comparison (3 rows × 3 cols: original | bbox×1.0 baseline | bbox×1.8 padded 224)
7. Build Dataset + DataLoader with augmentation chain
8. **VIZ-3**: Training-time augmentation samples (4 images × 6 augmented versions each)
9. Build model: MobileNetV2 ImageNet pretrained, head `Linear(1280, 2)`
10. Training loop: AMP mixed precision, CosineAnnealingLR, track val_ACER, save best
11. Save artifacts: `best_model.pt`, `mobilenetv2_context_scripted.pt`, `run_summary.json`, `history.json`
12. Threshold sweep on val: APCER/BPCER/ACER per threshold, save `threshold_metrics.csv` + ROC + ACER curve PNG
13. **VIZ-4**: Confusion matrix on val at best threshold + 6 hard examples (3 live-as-spoof + 3 spoof-as-live)

**Augmentation pipeline**:

```
RandomResizedCrop(224, scale=(0.85, 1.0))   # keep context, no aggressive zoom
RandomHorizontalFlip
ColorJitter(0.2, 0.2, 0.2, 0.05)
RandomGaussianBlur(p=0.2, sigma=0.1-1.5)
RandomJPEG(p=0.3, quality=40-90)
RandomGrayscale(p=0.05)
Normalize(ImageNet mean/std)
```

**Loss / optimizer**:
- `CrossEntropyLoss` with class weights computed from train manifest (handle CelebA-Spoof imbalance).
- `AdamW`, lr=1e-4, weight_decay=1e-4, CosineAnnealingLR over 10 epochs.
- Batch size 64, mixed precision (`torch.cuda.amp`).
- Early stop on val_ACER plateau.

**Artifacts** (Kaggle output dir `context_mobilenetv2_224/`):
- `best_model.pt`
- `mobilenetv2_context_scripted.pt`
- `run_summary.json` — must contain `input_size: 224`, `preprocessing: imagenet_norm`, `best_acer`, `best_threshold`
- `history.json`
- `threshold_metrics.csv`
- ROC + ACER plots (PNG)

### Backend integration

**`src/fas/detection.py`**:
- `FaceDetection` adds field `context_crop_bgr: np.ndarray` — bbox×1.8 padded square, **NOT yet resized** (liveness model resizes per its `input_size`).
- Keep `aligned_crop_bgr` for backward compatibility / debug.

**`src/fas/service.py`**:
- `LivenessService.infer()` passes `detection.context_crop_bgr` to `liveness_model.predict_live_score(...)` instead of `aligned_crop_bgr`.

**`src/fas/liveness_model.py`**:
- `input_size` reads from `run_summary.json` (already supports this pattern). Default fallback 112 for legacy models.

### Model switch

- Setting `LIVENESS_MODEL_PATH=/path/to/mobilenetv2_context_scripted.pt` and ensuring its sibling `run_summary.json` has `input_size: 224` causes backend to use the context model.
- No code change needed to switch back to baseline — just change env var.

---

## 4. Component B — Active Challenge (Head Pose)

### Head pose estimation

**File**: `src/fas/pose.py`

**Method**: `cv2.solvePnP` with a generic 3D face template + approximate camera intrinsics.

```python
MODEL_POINTS_3D = np.array([
    (-30.0,  30.0, -30.0),   # left eye center
    ( 30.0,  30.0, -30.0),   # right eye center
    (  0.0,   0.0,   0.0),   # nose tip
    (-25.0, -30.0, -30.0),   # mouth left corner
    ( 25.0, -30.0, -30.0),   # mouth right corner
], dtype=np.float64)

def estimate_head_pose(landmarks_xy, image_shape):
    # Approx pinhole: f = w (assumes ~60° FOV)
    # Returns {'yaw_deg', 'pitch_deg', 'roll_deg', 'ok'}
```

**Sign convention** (documented in code and report):
- `yaw_deg > 0` ⟺ user turns head to **their right** (camera view: face tilts to left of frame).
- `yaw_deg < 0` ⟺ user turns head to **their left**.

**Robustness**: frontend applies a 3-frame moving average to `yaw_deg` before feeding into state machine (mitigates landmark noise).

### Backend response extension

**`src/fas/schemas.py`** — `LivenessInferResponse` adds:

```python
yaw_deg: float | None = None
pitch_deg: float | None = None
pose_ok: bool = False
```

### Challenge sequence

Session ≈ 5–8 seconds, challenge order randomized per session (defense against generic replay video):

| Phase | Duration (max) | Pass criterion | UI instruction |
|---|---|---|---|
| countdown | 2s | — | "Chuẩn bị… 3, 2, 1" |
| forward | 2s | `|yaw| ≤ 10°` for K=5 consec frames | "Nhìn thẳng vào camera" |
| turn_A (random: left or right) | 3s | yaw reaches ±20° target | "Quay đầu sang TRÁI/PHẢI" + arrow |
| center_1 | 2s | `|yaw| ≤ 10°` for K=5 consec | "Quay về giữa" |
| turn_B (opposite of A) | 3s | yaw reaches opposite ±20° target | "Quay đầu sang PHẢI/TRÁI" + arrow |
| evaluating | — | fusion logic runs | "Đang xử lý…" |

- Phase advances **early** when criterion met for K=5 consecutive frames (UX speed).
- Any phase that times out without meeting criterion → fail with reason → ResultView.
- On fail, user clicks "Retry" to return to `idle` (no auto-retry).

### Challenge validation rules

Computed once at end of session in `fusion.ts`:

```
MIN_DETECT_RATE = 0.9       # face must be detected in ≥ 90% of frames
MAX_JUMP = 15               # |yaw[i] - yaw[i-1]| > 15° → frame swap suspect
MIN_PHASE_RATE = 0.6        # ≥ 60% frames in each phase must meet criterion
YAW_TARGET = 20             # turn phases must reach this
YAW_CENTER = 10             # forward/center phases must satisfy |yaw| ≤ this
```

### Defense scope

| Attack | Defense |
|---|---|
| Printed photo | ✓ defeats (paper cannot turn) |
| Static photo on phone screen | ✓ defeats (image is static) |
| Pre-recorded video of someone turning head (wrong order) | ✓ ~50% defeats via random order |
| Pre-recorded video matching the random challenge order | ⚠️ ~50% bypass risk; documented limitation |
| 3D real-time mask | ✗ out of scope |
| Real-time deepfake | ✗ out of scope |

---

## 5. Component C — Session UI & Fusion

### Frame capture

- 10 fps (100 ms interval) during recording phases.
- **In-flight token**: skip new frame if previous request still in flight (prevents queue pile-up).
- Each response is timestamped client-side at send time → enables ordering even with out-of-order arrival.
- Backend target latency: < 200 ms per frame.

### Frontend file layout

```
apps/web/src/
├── App.tsx                          # Shell: switch idle ↔ session ↔ result
├── session/
│   ├── useSession.ts                # State machine + capture loop hook
│   ├── SessionView.tsx              # Recording UI (video + instruction overlay + realtime debug)
│   ├── ResultView.tsx               # Verdict display
│   ├── fusion.ts                    # Pure functions: evaluateChallenge, computeVerdict
│   └── types.ts                     # FrameRecord, Phase, Verdict
└── styles.css
```

### State machine

States: `idle → countdown → forward → turn_A → center_1 → turn_B → evaluating → result`

Failure transitions: any phase that times out without meeting criterion → `evaluating` with `challenge_eval.pass = false` → `result` with verdict SPOOF and reason.

### Fusion (`fusion.ts`)

```typescript
type FrameRecord = {
  ts_ms: number
  phase: 'forward' | 'turn_A' | 'center_1' | 'turn_B'
  face_detected: boolean
  passive_score: number
  yaw_deg: number | null
  pose_ok: boolean
}

const T_PASSIVE = 0.70   // initial; calibrated from threshold_metrics.csv

function computeVerdict(frames: FrameRecord[], challenge_eval: ChallengeEval): Verdict {
  const forwardFrames = frames.filter(f => f.phase === 'forward' && f.face_detected)
  const passiveAvg = forwardFrames.length
    ? forwardFrames.reduce((s, f) => s + f.passive_score, 0) / forwardFrames.length
    : 0
  const passivePass = passiveAvg >= T_PASSIVE

  if (!challenge_eval.pass) return { verdict: 'SPOOF', reason: 'challenge_failed', detail: challenge_eval.reason, passive_avg: passiveAvg }
  if (!passivePass)         return { verdict: 'SPOOF', reason: 'passive_low', passive_avg: passiveAvg }
  return { verdict: 'LIVE', passive_avg: passiveAvg }
}
```

**Why only forward-phase frames for passive_avg**:
- During turn phases, motion blur + extreme yaw angle degrade model accuracy.
- Training data (CelebA-Spoof) is predominantly frontal → forward-phase frames best match training conditions.
- Turn-phase scores are still logged for debug but excluded from decision.

### Result display

Both LIVE and SPOOF results show:
- `passive_avg` with target threshold and pass/fail icon
- Challenge breakdown (max yaw left, max yaw right, face detect rate, with target thresholds)
- Total frames, duration
- Failure reason (if SPOOF)
- "Verify Again" button

### Calibration knob

- `T_PASSIVE` hardcoded at 0.70 initially; updated based on `threshold_metrics.csv` after training completes.
- URL query override for live tuning during demo: `?t_passive=0.65`.

### Backward compatibility

- Existing `App.tsx` UI (continuous polling) is preserved behind URL query `?legacy=1` for side-by-side A/B demo in the report.

---

## 6. Training & Data Plan

### Critical path (non-blocking on data collection)

1. Train context model on Kaggle → produce `mobilenetv2_context_scripted.pt`.
2. Deploy to backend by setting `LIVENESS_MODEL_PATH`.
3. Smoke test live with team's actual webcams (qualitative pass/fail).
4. Backend + frontend integration → working demo.

### Parallel track (for report metrics)

While critical path runs, team collects structured webcam evaluation data:

```
data_collected/webcam_eval/
├── live/                      # 15 sessions (target)
├── print_photo/               # 15 sessions
└── phone_screen_replay/       # 15 sessions
```

**Categories** (3, after dropping `paper_mask`):
- `live`: real human, webcam, normal use
- `print_photo`: A4 laser or inkjet print held in front of webcam
- `phone_screen_replay`: photo displayed on Android or iPhone screen

**Variations per category**:
- Lighting: normal / dim / strong backlight
- Distance: ~30cm, ~50cm, ~80cm
- Spoof devices: 2 print methods (laser + inkjet) + 2 phone models (Android + iPhone)

**Minimum acceptable for report**: 30 sessions (10 per category). Target: 45 sessions.

**Collection tool**: `scripts/collect_webcam_eval.py` — standalone Python, no backend dependency. Records 5 seconds @ 10 fps, writes frames + `meta.json` to the appropriate category folder.

**Protocol doc**: `docs/WEBCAM_COLLECTION_PROTOCOL.md` — step-by-step for the team (lighting setup, distance markers, device list, naming conventions, common mistakes to avoid).

### Compute budget

- CelebA-Spoof full training on Kaggle T4/P100: ~30–60 min/epoch × 10 epochs ≈ 5–10 hours. Fits in 12h Kaggle quota.
- If smoke test shows >60 min/epoch, fall back to stratified subset (100K live + 100K spoof, 12 epochs).
- Webcam eval inference: < 5 minutes total on CPU (small dataset).

### Timeline (≈ 2.5 weeks)

| Week | Day | Task | Depends on |
|---|---|---|---|
| 1 | 1–2 | Modify `celeba_spoof_prep.py` + notebook 07 cells 1–7 (data + viz) | — |
| 1 | 3–4 | Notebook 07 cells 8–13 + smoke training (1–2 epochs) | data ready |
| 1 | 5–7 | Kaggle full training run + visualization cells | smoke passes |
| 1 | 5–7 | (parallel) Write collection script + protocol doc + start collecting | script ready |
| 2 | 8–9 | Backend: modify `detection.py`, `service.py`, `schemas.py`, `liveness_model.py` for context model | model trained |
| 2 | 10 | Add `src/fas/pose.py` + integrate pose into backend response | — |
| 2 | 11–13 | Frontend: session machine + SessionView + ResultView + fusion + tests | backend ready |
| 2 | 14 | Threshold calibration on webcam eval data | both models evaluated |
| 3 | 15–16 | End-to-end local demo testing + bug fixes | full integration |
| 3 | 17–18 | Eval ablation notebook + reports/ | webcam data evaluated |
| 3 | 19–21 | Buffer + final report write-up | — |

---

## 7. Evaluation Plan

### Metrics

```
APCER = (# spoof classified live) / (# spoof total)        ~ False Accept Rate
BPCER = (# live classified spoof) / (# live total)         ~ False Reject Rate
ACER  = (APCER + BPCER) / 2                                ~ averaged
TPR @ FPR=0.01  = True Positive Rate at FPR = 1%
EER   = error rate where APCER = BPCER
```

Primary metric: **ACER** (standard for CelebA-Spoof papers). APCER and BPCER reported separately for trade-off visibility.

### Two evaluation tiers

| Tier | Data | Granularity | Purpose |
|---|---|---|---|
| 1 | CelebA-Spoof val set | Frame-level | Benchmark accuracy comparison (academic reference) |
| 2 | Real webcam (45 sessions) | Frame-level + Session-level | Real-world improvement evidence (main report claim) |

### Comparison tables for report

**Tier 1 (CelebA-Spoof val)**:

| Model | Input | ACER ↓ | APCER | BPCER |
|---|---|---|---|---|
| Baseline MobileNetV2 (face-tight) | 112×112 | 0.0016 | … | … |
| Context MobileNetV2 (bbox×1.8) | 224×224 | TBD after training | … | … |

**Tier 2 frame-level (real webcam)**:

| Model | Live BPCER | Print APCER | Phone replay APCER | Overall ACER |
|---|---|---|---|---|
| Baseline (face-tight) | … | High (root cause) | High (root cause) | … |
| Context (bbox×1.8) | … | Expected to drop substantially | Expected to drop substantially | … |

**Tier 2 session-level (after challenge integration)**:

| Pipeline | Live accept rate | Print reject rate | Phone reject rate |
|---|---|---|---|
| A only (passive context) | … | … | … |
| A + B (passive + challenge) | … | Expected ~100% (static → fail challenge) | Expected ~100% |

### Hard example analysis

For both baseline and context models on Tier 2:
- Top 10 frames live classified as spoof (highest-confidence false rejects).
- Top 10 frames spoof classified as live (highest-confidence false accepts).
- Sessions failing challenge with detailed reason breakdown.

Saved as image grids in `reports/hard_examples/`.

### Report artifacts

```
reports/
├── tier1_celeba_spoof_comparison.md
├── tier1_threshold_metrics_context.csv
├── tier2_webcam_frame_level.md
├── tier2_webcam_session_level.md
├── hard_examples/
│   ├── live_misclassified_spoof.png
│   └── spoof_misclassified_live.png
├── confusion_matrices/
│   ├── baseline_celeba.png
│   ├── context_celeba.png
│   ├── baseline_webcam.png
│   └── context_webcam.png
└── pipeline_diagrams/
    ├── architecture_overview.png
    └── session_state_machine.png
```

Pipeline diagrams authored in mermaid, exported to PNG for the report.

### Final report structure

1. Mở đầu (mục tiêu, phạm vi)
2. Phương pháp baseline (đã có)
3. Vấn đề phát hiện (root cause + Tier 2 baseline numbers)
4. Cải tiến đề xuất (Lớp A, B, C)
5. Implementation (data pipeline, backend, frontend, API)
6. Kết quả (Tier 1, Tier 2 frame, Tier 2 session, hard examples)
7. Hạn chế & out-of-scope
8. Kết luận & hướng phát triển (tích hợp authentication, mở rộng challenge space)

---

## 8. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Context model training exceeds 12h on Kaggle | Med | High | Smoke 1–2 epochs first; fall back to stratified 200K subset |
| R2 | Context model ACER worse than baseline on CelebA-Spoof | Low | Med | Disable aggressive augment (Blur+JPEG); freeze backbone first 2 epochs |
| R3 | Context model still fails on real webcam | Med | High | Challenge layer (B) is an independent second line of defense |
| R4 | Pose estimation noise from 5 landmarks too large | Med | Med | 3-frame moving average; relax yaw_target to 15° if needed |
| R5 | Backend per-frame latency > 200 ms causes UI lag | Low | Med | In-flight token to drop frames; downscale image to 640×480 before sending |
| R6 | Frontend state machine race conditions | Med | Low | Pure-function fusion → unit testable; mock-data Storybook page for state transitions |
| R7 | Webcam collection slips deadline | Med | Med | Demo video fallback for qualitative evidence; minimum 30 sessions acceptable |
| R8 | Kaggle GPU quota exhausted | Low | High | Colab T4 fallback; training script not Kaggle-specific |

---

## 9. Out of Scope

| Item | Reason | Possible future work |
|---|---|---|
| Replay video matching random challenge order | Random 2-direction only gives 50% defense | Extend challenge space (3+ directions, random magnitude) |
| 3D real-time mask | Needs depth sensor or specialized dataset | Add rPPG or IR sensor input |
| Real-time deepfake | Out of timeline scope | Temporal CNN or audio-visual fusion |
| Continuous monitoring after pass | Explicitly de-scoped | — |
| Multi-face frames | Single-user session model only | Trivial extension |
| Authentication / identity matching | Owned by teammate | — |
| Monocular RGB depth | Too risky for timeline | MiDaS-FAS hybrid in future work |

---

## 10. Definition of Done

### Component A — Context passive
- [ ] `celeba_spoof_prep.py` produces `data_processed/celeba_spoof_context_224/` with pad-to-square crops.
- [ ] Notebook 07 runs end-to-end (data → train → eval → artifacts).
- [ ] All 4 visualization cells render correctly (raw, before/after, augmented, hard examples).
- [ ] `mobilenetv2_context_scripted.pt` loads in backend `TorchLivenessModel`.
- [ ] `run_summary.json` contains `input_size=224`, `preprocessing=imagenet_norm`, `best_acer`, `best_threshold`.
- [ ] Backend smoke: 1 live image → score > 0.7; 1 phone replay image → score < 0.5.

### Component B — Active challenge
- [ ] `src/fas/pose.py` provides `estimate_head_pose()` with unit tests on synthetic 5-landmark fixtures across 3 angles.
- [ ] `LivenessInferResponse` includes `yaw_deg`, `pitch_deg`, `pose_ok`.
- [ ] Backend smoke: frontal image → `|yaw| < 5°`; turned image → `|yaw| > 15°`.

### Component C — Session UI
- [ ] `useSession.ts` state machine handles full happy path + 3 failure modes (face_lost, yaw_jump, phase_timeout).
- [ ] `fusion.ts` has unit tests covering LIVE pass, challenge_failed, passive_low, all three reasons.
- [ ] `SessionView.tsx` renders countdown + instruction overlay + realtime debug.
- [ ] `ResultView.tsx` displays verdict + per-criterion breakdown.
- [ ] End-to-end manual test: live session passes; print photo session fails at challenge; phone screen session fails at passive.

### Backend infrastructure
- [ ] New endpoint `/v1/liveness/frame` functional; legacy `/v1/liveness/infer` preserved.
- [ ] CORS + existing tests pass (`python -m pytest tests/api -q`).
- [ ] `python -m compileall src services/api` clean.
- [ ] `cd apps/web && npm run lint && npm run typecheck` clean.

### Evaluation & reports
- [ ] Tier 1 comparison written to `reports/tier1_celeba_spoof_comparison.md`.
- [ ] Tier 2 frame-level written to `reports/tier2_webcam_frame_level.md` with bar charts and confusion matrices.
- [ ] Tier 2 session-level written to `reports/tier2_webcam_session_level.md` with A vs A+B ablation.
- [ ] `reports/hard_examples/` has at least 2 image grids.
- [ ] Pipeline diagrams (mermaid → PNG) embedded in final report.

### Verification commands (run at project end)

```bash
# Backend
python -m pytest tests/api tests/fas -q
python -m compileall src services/api
LIVENESS_MODEL_PATH=Kaggle_Outputs/context_mobilenetv2_224/mobilenetv2_context_scripted.pt \
    python services/api/benchmark_smoke.py

# Frontend
cd apps/web && npm run lint && npm run typecheck && npm run build

# Eval reproduction
jupyter nbconvert --execute notebooks/eval_session_ablation.ipynb \
    --to notebook --output eval_session_ablation.out.ipynb
```

---

## 11. Final Deliverables

1. **Code**: backend (context model + pose), frontend (session UI), notebook 07 (training), notebook ablation eval, collection script + protocol doc.
2. **Models**: `mobilenetv2_context_scripted.pt` + `run_summary.json`.
3. **Reports** (under `reports/`): Tier 1, Tier 2 frame, Tier 2 session, hard examples, confusion matrices.
4. **Pipeline diagrams**: PNG exported from mermaid.
5. **Demo video**: 3 clips (live pass, print fail, phone replay fail).
6. **Final BTL report**: structured per Section 7, embedding metrics and findings.
