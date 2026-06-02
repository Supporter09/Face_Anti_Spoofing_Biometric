import type { ChallengeEval, FrameRecord, TurnDirection, Verdict } from './types'

// Calibrated from Kaggle training threshold_metrics.csv
// At threshold=0.4: APCER=0.072%, BPCER=0.080%, ACER=0.076%
export const T_PASSIVE = 0.4

export const MIN_DETECT_RATE = 0.9
export const MAX_YAW_JUMP = 15 // degrees; consecutive jump > this = suspect frame swap

// Relative-yaw thresholds — measured as delta from the forward-phase baseline.
// solvePnP with approximate focal length gives a constant offset (typically 10-20°),
// so absolute thresholds are unreliable. Using relative yaw makes the challenge
// robust to camera position, FOV, and solvePnP calibration offset.
export const YAW_TARGET = 20 // minimum relative yaw for turn phases (increased for better liveness challenge)
export const YAW_CENTER = 10 // maximum |relative yaw| for forward/center phases (loosened from 8)

export function evaluateChallenge(
  frames: FrameRecord[],
  turn_A_dir: TurnDirection,
): ChallengeEval {
  const total = frames.length
  if (total === 0) {
    return {
      pass: false,
      detect_rate: 0,
      max_yaw_left: null,
      max_yaw_right: null,
      reason: 'no_frames',
    }
  }

  const detected = frames.filter((f) => f.face_detected).length
  const detect_rate = detected / total

  if (detect_rate < MIN_DETECT_RATE) {
    return {
      pass: false,
      detect_rate,
      max_yaw_left: null,
      max_yaw_right: null,
      reason: 'low_detect_rate',
    }
  }

  // Check for suspicious yaw jumps (frame swap attack indicator).
  // Operates on raw yaw (not relative) — consecutive frame diffs should be small.
  const yawFrames = frames.filter((f) => f.yaw_deg !== null)
  for (let i = 1; i < yawFrames.length; i++) {
    if (Math.abs(yawFrames[i].yaw_deg! - yawFrames[i - 1].yaw_deg!) > MAX_YAW_JUMP) {
      return {
        pass: false,
        detect_rate,
        max_yaw_left: null,
        max_yaw_right: null,
        reason: 'yaw_jump_detected',
      }
    }
  }

  // Compute yaw baseline from forward-phase frames.
  // solvePnP with approximate intrinsics gives a consistent offset; subtracting the
  // mean forward-phase yaw normalises it so thresholds work regardless of camera setup.
  const forwardYawValues = frames
    .filter((f) => f.phase === 'forward' && f.yaw_deg !== null)
    .map((f) => f.yaw_deg!)
  const yawBaseline =
    forwardYawValues.length > 0
      ? forwardYawValues.reduce((a, b) => a + b, 0) / forwardYawValues.length
      : 0

  const rel = (yaw: number) => yaw - yawBaseline

  // Forward phase: at least 60% of frames must be centred (within ±YAW_CENTER).
  // Using every() was too strict — a single outlier at the start/end of the phase
  // (captured while transitioning from countdown) would incorrectly fail the check.
  // The phase-advancement logic in useSession already guarantees 5 consecutive
  // frontal frames were seen; this 60% check is a light sanity guard only.
  const forwardFrames = frames.filter((f) => f.phase === 'forward' && f.yaw_deg !== null)
  const frontalCount = forwardFrames.filter((f) => Math.abs(rel(f.yaw_deg!)) <= YAW_CENTER).length
  const forwardPass =
    forwardFrames.length === 0 || frontalCount / forwardFrames.length >= 0.60

  // Turn-A phase: relative yaw must reach ±YAW_TARGET in the required direction.
  const turnAFrames = frames.filter((f) => f.phase === 'turn_A' && f.yaw_deg !== null)
  const turnAPass = turnAFrames.some((f) =>
    turn_A_dir === 'right' ? rel(f.yaw_deg!) >= YAW_TARGET : rel(f.yaw_deg!) <= -YAW_TARGET,
  )

  // Center-1 phase: relative yaw back within ±YAW_CENTER.
  const center1Frames = frames.filter((f) => f.phase === 'center_1' && f.yaw_deg !== null)
  const center1Pass =
    center1Frames.length === 0 || center1Frames.some((f) => Math.abs(rel(f.yaw_deg!)) <= YAW_CENTER)

  // Turn-B phase: relative yaw to the opposite direction.
  const turnBDir: TurnDirection = turn_A_dir === 'right' ? 'left' : 'right'
  const turnBFrames = frames.filter((f) => f.phase === 'turn_B' && f.yaw_deg !== null)
  const turnBPass = turnBFrames.some((f) =>
    turnBDir === 'right' ? rel(f.yaw_deg!) >= YAW_TARGET : rel(f.yaw_deg!) <= -YAW_TARGET,
  )

  // Report max relative yaw in each direction for the result display.
  const allRelYaws = frames
    .filter((f) => f.yaw_deg !== null)
    .map((f) => rel(f.yaw_deg!))
  const max_yaw_left = allRelYaws.length ? Math.min(...allRelYaws) : null
  const max_yaw_right = allRelYaws.length ? Math.max(...allRelYaws) : null

  if (!forwardPass)
    return { pass: false, detect_rate, max_yaw_left, max_yaw_right, reason: 'forward_not_frontal' }
  if (!turnAPass)
    return { pass: false, detect_rate, max_yaw_left, max_yaw_right, reason: 'turn_A_insufficient' }
  if (!center1Pass)
    return { pass: false, detect_rate, max_yaw_left, max_yaw_right, reason: 'center_1_not_frontal' }
  if (!turnBPass)
    return { pass: false, detect_rate, max_yaw_left, max_yaw_right, reason: 'turn_B_insufficient' }

  return { pass: true, detect_rate, max_yaw_left, max_yaw_right }
}

export function computeVerdict(
  frames: FrameRecord[],
  turn_A_dir: TurnDirection,
  duration_ms: number,
): Verdict {
  const challenge_eval = evaluateChallenge(frames, turn_A_dir)

  const forwardFrames = frames.filter((f) => f.phase === 'forward' && f.face_detected)
  const passive_avg = forwardFrames.length
    ? forwardFrames.reduce((s, f) => s + f.passive_score, 0) / forwardFrames.length
    : 0
  const passivePass = passive_avg >= T_PASSIVE

  const base = { passive_avg, challenge_eval, frame_count: frames.length, duration_ms }

  if (!challenge_eval.pass) return { ...base, verdict: 'SPOOF', reason: 'challenge_failed' }
  if (!passivePass) return { ...base, verdict: 'SPOOF', reason: 'passive_low' }
  return { ...base, verdict: 'LIVE' }
}

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

