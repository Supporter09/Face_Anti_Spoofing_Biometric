import { useCallback, useEffect, useRef, useState } from 'react'

import { computeVerdict, YAW_CENTER, YAW_TARGET } from './fusion'
import type { FrameApiResponse, FrameRecord, Phase, TurnDirection, Verdict } from './types'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'
const BYPASS_LIVENESS = import.meta.env.VITE_BYPASS_LIVENESS === 'true'

const CAPTURE_INTERVAL_MS = 100
const COUNTDOWN_MS = 3000 // 3s so backend JIT warmup finishes before first frame
const REQUIRED_CONSECUTIVE_FRAMES = 5
const CAPTURE_WIDTH = 640
const CAPTURE_HEIGHT = 480
const JPEG_QUALITY = 0.85

const PHASE_TIMEOUT_MS: Record<FrameRecord['phase'], number> = {
  forward: 3000,
  turn_A: 4000,
  center_1: 2500,
  turn_B: 4000,
}

const ACTIVE_PHASES: Phase[] = ['forward', 'turn_A', 'center_1', 'turn_B']

export interface SessionState {
  phase: Phase
  countdown: number
  instruction: string
  turn_A_dir: TurnDirection | null
  frames: FrameRecord[]
  verdict: Verdict | null
  latest_yaw: number | null
  latest_passive: number | null
  face_detected: boolean
  /** Tight face bbox in capture coordinates (640×480) — for overlay drawing */
  latest_bbox: [number, number, number, number] | null
  /** 5 landmarks [left_eye, right_eye, nose, mouth_left, mouth_right] in capture coords */
  latest_landmarks: [number, number][] | null
  error: string | null
  auth_status: 'idle' | 'verifying' | 'authenticated' | 'failed'
  auth_message: string | null
  similarity: number | null
  identified_user: string | null
}

function getTurnBDir(turn_A_dir: TurnDirection): TurnDirection {
  return turn_A_dir === 'right' ? 'left' : 'right'
}

function getInstruction(phase: Phase, countdown: number, turn_A_dir: TurnDirection | null): string {
  if (phase === 'countdown') return `Chuẩn bị… ${countdown}`
  if (phase === 'forward') return 'Nhìn thẳng vào camera'
  if (phase === 'turn_A') return turn_A_dir === 'left' ? 'Quay đầu sang TRÁI ←' : 'Quay đầu sang PHẢI →'
  if (phase === 'center_1') return 'Quay về giữa'
  if (phase === 'turn_B') {
    if (!turn_A_dir) return 'Quay đầu sang PHẢI →'
    return getTurnBDir(turn_A_dir) === 'left' ? 'Quay đầu sang TRÁI ←' : 'Quay đầu sang PHẢI →'
  }
  if (phase === 'evaluating') return 'Đang xử lý…'
  return ''
}

function initialState(): SessionState {
  return {
    phase: 'idle',
    countdown: 3,
    instruction: '',
    turn_A_dir: null,
    frames: [],
    verdict: null,
    latest_yaw: null,
    latest_passive: null,
    face_detected: false,
    latest_bbox: null,
    latest_landmarks: null,
    error: null,
    auth_status: 'idle',
    auth_message: null,
    similarity: null,
    identified_user: null,
  }
}

function isActivePhase(phase: Phase): phase is FrameRecord['phase'] {
  return ACTIVE_PHASES.includes(phase)
}

function bypassVerdict(durationMs: number): Verdict {
  return {
    verdict: 'LIVE',
    passive_avg: 1,
    challenge_eval: {
      pass: true,
      detect_rate: 1,
      max_yaw_left: null,
      max_yaw_right: null,
      reason: 'bypass_liveness_dev_mode',
    },
    frame_count: 0,
    duration_ms: durationMs,
  }
}

export function useSession(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
): {
  state: SessionState
  start: () => void
  reset: () => void
  isDiagnose: boolean
} {
  // Diagnose mode: all phases run to timeout; no early advancement.
  // Activate via ?diagnose=1 in URL.
  const isDiagnose = new URLSearchParams(window.location.search).has('diagnose')

  const [state, setState] = useState<SessionState>(initialState)
  const stateRef = useRef(state)
  const inFlightRef = useRef(false)
  const consecutiveRef = useRef(0)
  const phaseStartedAtRef = useRef(0)
  const sessionStartedAtRef = useRef(0)
  const smoothedYawRef = useRef<number | null>(null)

  useEffect(() => {
    stateRef.current = state
  }, [state])

  const setPhase = useCallback((phase: Phase) => {
    phaseStartedAtRef.current = Date.now()
    consecutiveRef.current = 0
    smoothedYawRef.current = null

    setState((current) => ({
      ...current,
      phase,
      instruction: getInstruction(phase, current.countdown, current.turn_A_dir),
    }))
  }, [])

  const reset = useCallback(() => {
    inFlightRef.current = false
    consecutiveRef.current = 0
    smoothedYawRef.current = null
    phaseStartedAtRef.current = 0
    sessionStartedAtRef.current = 0
    setState(initialState())
  }, [])

  const captureFrameBase64 = useCallback((): string | null => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return null

    canvas.width = CAPTURE_WIDTH
    canvas.height = CAPTURE_HEIGHT

    const ctx = canvas.getContext('2d')
    if (!ctx) return null

    ctx.drawImage(video, 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT)

    return canvas.toDataURL('image/jpeg', JPEG_QUALITY).split(',')[1] ?? null
  }, [canvasRef, videoRef])

  const authenticate = useCallback(async () => {
    const imageBase64 = captureFrameBase64()

    if (!imageBase64) {
      setState((s) => ({
        ...s,
        phase: 'result',
        auth_status: 'failed',
        auth_message: 'Không chụp được ảnh từ camera.',
        identified_user: null,
        similarity: null,
      }))
      return
    }

    setState((s) => ({
      ...s,
      phase: 'result',
      instruction: '',
      auth_status: 'verifying',
      auth_message: 'Đang xác thực...',
      identified_user: null,
      similarity: null,
    }))

    try {
      const response = await fetch(`${API_BASE}/v1/auth/identify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: imageBase64 }),
      })

      if (!response.ok) {
        throw new Error(`Auth API failed: ${response.status}`)
      }

      const payload = await response.json()
      // Expected payload: { authenticated, user_id, similarity, threshold, message }

      setState((s) => ({
        ...s,
        phase: 'result',
        auth_status: payload.authenticated ? 'authenticated' : 'failed',
        auth_message: payload.message,
        similarity: payload.similarity ?? null,
        identified_user: payload.authenticated ? payload.user_id : null,
      }))
    } catch (err) {
      setState((s) => ({
        ...s,
        phase: 'result',
        auth_status: 'failed',
        auth_message: err instanceof Error ? err.message : 'Authentication failed',
        identified_user: null,
        similarity: null,
      }))
    }
  }, [captureFrameBase64])

  const start = useCallback(() => {
    const now = Date.now()
    sessionStartedAtRef.current = now
    phaseStartedAtRef.current = now
    consecutiveRef.current = 0
    smoothedYawRef.current = null
    inFlightRef.current = false

    if (BYPASS_LIVENESS) {
      setState({
        ...initialState(),
        phase: 'result',
        instruction: '',
        verdict: bypassVerdict(0),
        auth_status: 'verifying',
        auth_message: 'Đang xác thực...',
      })

      void authenticate()
      return
    }

    const turn_A_dir: TurnDirection = Math.random() < 0.5 ? 'right' : 'left'

    setState({
      ...initialState(),
      phase: 'countdown',
      countdown: 3,
      instruction: getInstruction('countdown', 3, turn_A_dir),
      turn_A_dir,
    })
  }, [authenticate])

  const phaseCriterionMet = useCallback((phase: FrameRecord['phase'], yaw: number | null, turn_A_dir: TurnDirection) => {
    if (yaw === null) return false
    if (phase === 'forward' || phase === 'center_1') return Math.abs(yaw) <= YAW_CENTER
    if (phase === 'turn_A') return turn_A_dir === 'right' ? yaw >= YAW_TARGET : yaw <= -YAW_TARGET

    const turnBDir = getTurnBDir(turn_A_dir)
    return turnBDir === 'right' ? yaw >= YAW_TARGET : yaw <= -YAW_TARGET
  }, [])

  const advanceFrom = useCallback((phase: FrameRecord['phase']) => {
    if (phase === 'forward') setPhase('turn_A')
    else if (phase === 'turn_A') setPhase('center_1')
    else if (phase === 'center_1') setPhase('turn_B')
    else setPhase('evaluating')
  }, [setPhase])

  const captureAndSend = useCallback(async () => {
    const current = stateRef.current

    if (BYPASS_LIVENESS) return
    if (!isActivePhase(current.phase) || !current.turn_A_dir || inFlightRef.current) return

    const imageBase64 = captureFrameBase64()
    if (!imageBase64) return

    inFlightRef.current = true

    try {
      const response = await fetch(`${API_BASE}/v1/liveness/frame`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: imageBase64 }),
      })

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`)
      }

      const payload = (await response.json()) as FrameApiResponse

      const frame: FrameRecord = {
        ts_ms: Date.now() - sessionStartedAtRef.current,
        phase: current.phase,
        face_detected: payload.face_detected,
        passive_score: payload.liveness_score,
        yaw_deg: payload.yaw_deg,
        pose_ok: payload.pose_ok,
      }

      const rawYaw = payload.yaw_deg
      const smoothedYaw =
        rawYaw === null
          ? null
          : smoothedYawRef.current === null
            ? rawYaw
            : 0.5 * smoothedYawRef.current + 0.5 * rawYaw

      smoothedYawRef.current = smoothedYaw

      const criterionMet =
        payload.face_detected && payload.pose_ok && phaseCriterionMet(current.phase, smoothedYaw, current.turn_A_dir)

      consecutiveRef.current = criterionMet ? consecutiveRef.current + 1 : 0

      setState((latest) => ({
        ...latest,
        frames: [...latest.frames, frame],
        latest_yaw: payload.yaw_deg,
        latest_passive: payload.liveness_score,
        face_detected: payload.face_detected,
        latest_bbox: payload.face_bbox_xyxy ?? null,
        latest_landmarks: (payload.face_landmarks as [number, number][] | null) ?? null,
        error: null,
      }))

      // In diagnose mode every phase runs to timeout so all phases are always captured.
      if (!isDiagnose && consecutiveRef.current >= REQUIRED_CONSECUTIVE_FRAMES) {
        advanceFrom(current.phase)
      }
    } catch (caughtError) {
      setState((latest) => ({
        ...latest,
        error: caughtError instanceof Error ? caughtError.message : 'Unexpected error.',
      }))
    } finally {
      inFlightRef.current = false
    }
  }, [advanceFrom, captureFrameBase64, isDiagnose, phaseCriterionMet])

  useEffect(() => {
    if (BYPASS_LIVENESS) return
    if (state.phase !== 'countdown') return

    const startedAt = Date.now()

    const id = window.setInterval(() => {
      const elapsed = Date.now() - startedAt
      const nextCountdown = Math.max(1, 3 - Math.floor(elapsed / 666))

      setState((current) => ({
        ...current,
        countdown: nextCountdown,
        instruction: getInstruction('countdown', nextCountdown, current.turn_A_dir),
      }))

      if (elapsed >= COUNTDOWN_MS) setPhase('forward')
    }, 100)

    return () => window.clearInterval(id)
  }, [setPhase, state.phase])

  useEffect(() => {
    if (BYPASS_LIVENESS) return
    if (!isActivePhase(state.phase)) return

    phaseStartedAtRef.current = Date.now()

    const id = window.setInterval(() => {
      const current = stateRef.current

      if (!isActivePhase(current.phase)) return

      if (Date.now() - phaseStartedAtRef.current >= PHASE_TIMEOUT_MS[current.phase]) {
        setPhase('evaluating')
        return
      }

      void captureAndSend()
    }, CAPTURE_INTERVAL_MS)

    return () => window.clearInterval(id)
  }, [captureAndSend, setPhase, state.phase])

  useEffect(() => {
    if (BYPASS_LIVENESS) return
    if (state.phase !== 'evaluating' || !state.turn_A_dir) return

    const verdict = computeVerdict(
      state.frames,
      state.turn_A_dir,
      Date.now() - sessionStartedAtRef.current,
    )

    setState((current) => ({
      ...current,
      phase: 'result',
      verdict,
      instruction: '',
    }))

    if (verdict.verdict === 'LIVE') {
      void authenticate()
    }
  }, [state.frames, state.phase, state.turn_A_dir, authenticate])

  return { state, start, reset, isDiagnose }
}