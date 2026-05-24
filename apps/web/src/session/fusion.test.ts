import { describe, expect, it } from 'vitest'

import { computeVerdict, evaluateChallenge } from './fusion'
import type { FrameRecord } from './types'

function frame(overrides: Partial<FrameRecord>): FrameRecord {
  return {
    ts_ms: 0,
    phase: 'forward',
    face_detected: true,
    passive_score: 0.8,
    yaw_deg: 0,
    pose_ok: true,
    ...overrides,
  }
}

function passingFrames(passive_score = 0.8): FrameRecord[] {
  return [
    frame({ phase: 'forward', passive_score, yaw_deg: 0 }),
    frame({ phase: 'forward', passive_score, yaw_deg: 2 }),
    frame({ phase: 'turn_A', yaw_deg: 8 }),
    frame({ phase: 'turn_A', yaw_deg: 18 }),
    frame({ phase: 'turn_A', yaw_deg: 25 }),
    frame({ phase: 'center_1', yaw_deg: 10 }),
    frame({ phase: 'center_1', yaw_deg: 2 }),
    frame({ phase: 'turn_B', yaw_deg: -8 }),
    frame({ phase: 'turn_B', yaw_deg: -18 }),
    frame({ phase: 'turn_B', yaw_deg: -25 }),
  ]
}

describe('computeVerdict', () => {
  it('returns LIVE when passive score and challenge pass', () => {
    const verdict = computeVerdict(passingFrames(), 'right', 5000)

    expect(verdict.verdict).toBe('LIVE')
    expect(verdict.passive_avg).toBeCloseTo(0.8)
    expect(verdict.challenge_eval.pass).toBe(true)
    expect(verdict.challenge_eval.detect_rate).toBe(1)
  })

  it('returns SPOOF for low detect rate', () => {
    const frames = passingFrames().map((f, index) => ({ ...f, face_detected: index % 2 === 0 }))
    const verdict = computeVerdict(frames, 'right', 5000)

    expect(verdict.verdict).toBe('SPOOF')
    expect(verdict.reason).toBe('challenge_failed')
    expect(verdict.challenge_eval.reason).toBe('low_detect_rate')
    expect(verdict.challenge_eval.detect_rate).toBe(0.5)
  })

  it('returns SPOOF when turn_A never reaches target yaw', () => {
    const frames = passingFrames().map((f) => (f.phase === 'turn_A' ? { ...f, yaw_deg: 12 } : f))
    const verdict = computeVerdict(frames, 'right', 5000)

    expect(verdict.verdict).toBe('SPOOF')
    expect(verdict.reason).toBe('challenge_failed')
    expect(verdict.challenge_eval.reason).toBe('turn_A_insufficient')
  })

  it('returns SPOOF when challenge passes but passive score is low', () => {
    const verdict = computeVerdict(passingFrames(0.2), 'right', 5000)

    expect(verdict.verdict).toBe('SPOOF')
    expect(verdict.reason).toBe('passive_low')
    expect(verdict.passive_avg).toBeCloseTo(0.2)
  })

  it('rejects suspicious yaw jumps', () => {
    const frames = [
      frame({ phase: 'forward', yaw_deg: 0 }),
      frame({ phase: 'turn_A', yaw_deg: 16 }),
      frame({ phase: 'turn_A', yaw_deg: 33 }),
      frame({ phase: 'center_1', yaw_deg: 8 }),
      frame({ phase: 'turn_B', yaw_deg: -24 }),
    ]
    const verdict = computeVerdict(frames, 'right', 5000)

    expect(verdict.verdict).toBe('SPOOF')
    expect(verdict.reason).toBe('challenge_failed')
    expect(verdict.challenge_eval.reason).toBe('yaw_jump_detected')
  })

  it('handles empty frames as challenge failure', () => {
    const verdict = computeVerdict([], 'right', 0)

    expect(verdict.verdict).toBe('SPOOF')
    expect(verdict.reason).toBe('challenge_failed')
    expect(verdict.challenge_eval.reason).toBe('no_frames')
  })
})

describe('evaluateChallenge', () => {
  it('tracks max left and right relative yaw for passing sessions', () => {
    // passingFrames(): forward yaws are [0, 2] → baseline = 1
    // turn_B min raw = -25 → relative = -25 - 1 = -26
    // turn_A max raw = 25  → relative = 25  - 1 = 24
    const result = evaluateChallenge(passingFrames(), 'right')

    expect(result.max_yaw_left).toBe(-26)
    expect(result.max_yaw_right).toBe(24)
  })
})
