import type { ChallengeEval, FrameRecord, TurnDirection, Verdict } from './types'

// Calibrated from Kaggle training threshold_metrics.csv
// At threshold=0.4: APCER=0.072%, BPCER=0.080%, ACER=0.076%
export const T_PASSIVE = 0.4

export const MIN_DETECT_RATE = 0.9
export const MAX_YAW_JUMP = 15
export const YAW_TARGET = 20
export const YAW_CENTER = 10

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

  const forwardFrames = frames.filter((f) => f.phase === 'forward' && f.yaw_deg !== null)
  const forwardPass = forwardFrames.length === 0 || forwardFrames.every((f) => Math.abs(f.yaw_deg!) <= YAW_CENTER)

  const turnAFrames = frames.filter((f) => f.phase === 'turn_A' && f.yaw_deg !== null)
  const turnATargetYaw = turn_A_dir === 'right' ? YAW_TARGET : -YAW_TARGET
  const turnAPass = turnAFrames.some((f) =>
    turn_A_dir === 'right' ? f.yaw_deg! >= turnATargetYaw : f.yaw_deg! <= turnATargetYaw,
  )

  const center1Frames = frames.filter((f) => f.phase === 'center_1' && f.yaw_deg !== null)
  const center1Pass = center1Frames.length === 0 || center1Frames.some((f) => Math.abs(f.yaw_deg!) <= YAW_CENTER)

  const turnBDir: TurnDirection = turn_A_dir === 'right' ? 'left' : 'right'
  const turnBFrames = frames.filter((f) => f.phase === 'turn_B' && f.yaw_deg !== null)
  const turnBTargetYaw = turnBDir === 'right' ? YAW_TARGET : -YAW_TARGET
  const turnBPass = turnBFrames.some((f) =>
    turnBDir === 'right' ? f.yaw_deg! >= turnBTargetYaw : f.yaw_deg! <= turnBTargetYaw,
  )

  const allYaws = frames.filter((f) => f.yaw_deg !== null).map((f) => f.yaw_deg!)
  const max_yaw_left = allYaws.length ? Math.min(...allYaws) : null
  const max_yaw_right = allYaws.length ? Math.max(...allYaws) : null

  if (!forwardPass) return { pass: false, detect_rate, max_yaw_left, max_yaw_right, reason: 'forward_not_frontal' }
  if (!turnAPass) return { pass: false, detect_rate, max_yaw_left, max_yaw_right, reason: 'turn_A_insufficient' }
  if (!center1Pass) return { pass: false, detect_rate, max_yaw_left, max_yaw_right, reason: 'center_1_not_frontal' }
  if (!turnBPass) return { pass: false, detect_rate, max_yaw_left, max_yaw_right, reason: 'turn_B_insufficient' }

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
