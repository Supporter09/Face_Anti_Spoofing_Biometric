import { useCallback, useEffect, useRef, useState } from 'react'

import { computeVerdict, selectBestFramesForAuth, YAW_CENTER, YAW_TARGET } from './fusion'
import type { FrameApiResponse, FrameRecord, Phase, TurnDirection, Verdict } from './types'

interface QueuedFrame {
  id: number
  imageBase64: string
  timestamp: number
  phase: FrameRecord['phase']
  turn_A_dir: TurnDirection
  resolve: (frame: FrameRecord) => void
  reject: (error: Error) => void
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'
const CAPTURE_INTERVAL_MS = 20
const COUNTDOWN_MS = 3000  // 3s so backend JIT warmup (TorchScript first-inference) finishes before first frame
const REQUIRED_CONSECUTIVE_FRAMES = 20
const CAPTURE_WIDTH = 640
const CAPTURE_HEIGHT = 480
const JPEG_QUALITY = 0.85

// Queue configuration
const NUM_WORKERS = 3
const MAX_QUEUE_SIZE = 3
const AUTH_FRAME_COUNT = 3  // Number of frames to average for authentication

const PHASE_TIMEOUT_MS: Record<FrameRecord['phase'], number> = {
  forward: 3000,  // 2s was tight if camera startup adds latency at phase start
  turn_A: 4000,   // extended from 3s — solvePnP underestimates range, need more time
  center_1: 2500,
  turn_B: 4000,   // extended from 3s
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

export function useSession(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
): {
  state: SessionState
  start: () => void
  reset: () => void
  isDiagnose: boolean
} {
  const searchParams = new URLSearchParams(window.location.search)
  // Diagnose mode: all phases run to timeout; no early advancement.
  // Activate via ?diagnose=1 in URL.
  const isDiagnose = searchParams.has('diagnose')
  const captureDebug = searchParams.get('capture_debug') === '1'
  const captureSessionId = searchParams.get('capture_session') ?? 'web_session'

  const [state, setState] = useState<SessionState>(initialState)
  const stateRef = useRef(state)
  const queueRef = useRef<QueuedFrame[]>([])
  const activeWorkersRef = useRef(0)
  const frameIdRef = useRef(0)
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
    queueRef.current = []
    activeWorkersRef.current = 0
    frameIdRef.current = 0
    consecutiveRef.current = 0
    smoothedYawRef.current = null
    phaseStartedAtRef.current = 0
    sessionStartedAtRef.current = 0
    setState(initialState())
  }, [])

  const start = useCallback(() => {
    const turn_A_dir: TurnDirection = Math.random() < 0.5 ? 'right' : 'left'
    const now = Date.now()
    sessionStartedAtRef.current = now
    phaseStartedAtRef.current = now
    consecutiveRef.current = 0
    smoothedYawRef.current = null
    setState({
      ...initialState(),
      phase: 'countdown',
      countdown: 3,
      instruction: getInstruction('countdown', 3, turn_A_dir),
      turn_A_dir,
    })
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

  const processQueue = useCallback(() => {
    const processNext = async () => {
      const queue = queueRef.current
      const current = stateRef.current

      // Check if we should stop - let finally block handle worker decrement
      if (!isActivePhase(current.phase) || queue.length === 0) {
        return
      }

      // Get next frame from queue (FIFO)
      const queued = queue.shift()
      if (!queued) {
        return
      }

      // Worker already incremented at the call site, don't increment again

      try {
        const frameEndpoint = captureDebug
          ? `/v1/liveness/frame/debug?session_id=${encodeURIComponent(captureSessionId)}`
          : '/v1/liveness/frame'
        const response = await fetch(`${API_BASE}${frameEndpoint}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_base64: queued.imageBase64, phase: current.phase }),
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

        // Use current phase from stateRef (not queued.phase) because frames may be processed after phase changed
        const currentPhase = stateRef.current.phase

        const frame: FrameRecord = {
          ts_ms: Date.now() - sessionStartedAtRef.current,
          phase: currentPhase,
          face_detected: payload.face_detected,
          passive_score: payload.liveness_score,
          yaw_deg: smoothedYaw,
          pose_ok: payload.pose_ok,
          image_base64: queued.imageBase64,
        }

        // Check phase advancement criteria using current phase
        const criterionMet =
          payload.face_detected && payload.pose_ok && phaseCriterionMet(currentPhase, smoothedYaw, queued.turn_A_dir)

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
        // First decrement the current worker (it's done)
        activeWorkersRef.current = Math.max(0, activeWorkersRef.current - 1)

        // Then check if we should start a new worker
        if (queueRef.current.length > 0 && activeWorkersRef.current < NUM_WORKERS) {
          activeWorkersRef.current++
          processNext()
        }
      }
    }

    // Start processing if workers available and queue not empty
    if (activeWorkersRef.current < NUM_WORKERS && queueRef.current.length > 0) {
      activeWorkersRef.current++
      processNext()
    }
  }, [captureDebug, captureSessionId, isDiagnose, phaseCriterionMet, advanceFrom])

  const authenticate = useCallback(async () => {
    const { frames, turn_A_dir } = stateRef.current
    if (!turn_A_dir || frames.length === 0) return

    setState((s) => ({
      ...s,
      auth_status: 'verifying',
      auth_message: 'Đang xác thực...',
      identified_user: null,
    }))

    try {
      // Calculate yaw baseline from forward phase frames
      const forwardFrames = frames.filter((f) => f.phase === 'forward' && f.yaw_deg !== null)
      const yawBaseline = forwardFrames.length > 0
        ? forwardFrames.reduce((a, b) => a + b.yaw_deg!, 0) / forwardFrames.length
        : 0

      // Select top best frames for auth
      const bestFrames = selectBestFramesForAuth(frames, yawBaseline, AUTH_FRAME_COUNT)

      if (bestFrames.length === 0) {
        throw new Error('Không đủ khung hình tốt để xác thực')
      }

      // Run auth requests in parallel
      const authUrl = captureDebug
        ? `${API_BASE}/v1/auth/identify?capture_debug=1&debug_session=${encodeURIComponent(captureSessionId)}`
        : `${API_BASE}/v1/auth/identify`
      const authPromises = bestFrames.map((frame) =>
        fetch(authUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_base64: frame.image_base64 }),
        }).then((res) => res.json())
      )

      const results = await Promise.all(authPromises)

      // Validate results
      if (!results.length || !results[0]) {
        throw new Error('Không nhận được phản hồi từ server')
      }

      // Average the similarities and determine auth status
      const similarities = results.map((r) => r.similarity ?? 0)
      const avgSimilarity = similarities.reduce((a, b) => a + b, 0) / similarities.length
      // Use threshold from server response, fallback to 0.5
      const authThreshold = results[0]?.threshold ?? 0.5
      const isAuthenticated = avgSimilarity >= authThreshold

      // Determine user_id: use most common result, or best frame's if all same
      const userIds = results.map((r) => r.user_id).filter(Boolean)
      const identifiedUser = userIds.length > 0
        ? userIds.sort((a, b) => userIds.filter(v => v === a).length - userIds.filter(v => v === b).length).pop()
        : null

      setState((s) => ({
        ...s,
        auth_status: isAuthenticated ? 'authenticated' : 'failed',
        auth_message: isAuthenticated
          ? `Xác thực thành công (${(avgSimilarity * 100).toFixed(1)}%)`
          : `Xác thực thất bại (${(avgSimilarity * 100).toFixed(1)}%)`,
        similarity: avgSimilarity,
        identified_user: isAuthenticated ? identifiedUser : null,
      }))
    } catch (err) {
      setState((s) => ({
        ...s,
        auth_status: 'failed',
        auth_message: err instanceof Error ? err.message : 'Authentication failed',
        identified_user: null,
      }))
    }
  }, [])

  const captureAndSend = useCallback(() => {
    const current = stateRef.current
    if (!isActivePhase(current.phase) || !current.turn_A_dir) return

    // Check queue capacity - drop frame if queue is full
    if (queueRef.current.length >= MAX_QUEUE_SIZE) return

    const imageBase64 = captureFrameBase64()
    if (!imageBase64) return

    // Create promise-based frame capture and add to queue
    return new Promise<void>((resolve, reject) => {
      const queued: QueuedFrame = {
        id: frameIdRef.current++,
        imageBase64,
        timestamp: Date.now(),
        phase: current.phase,
        turn_A_dir: current.turn_A_dir,
        resolve: (frame) => {
          // The original QueuedFrame.resolve expected FrameRecord
          // We just resolve the promise as void for captureAndSend's return type
          resolve()
        },
        reject,
      }
      queueRef.current.push(queued)
      processQueue()
    })
  }, [captureFrameBase64, processQueue])

  useEffect(() => {
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
  }, [state.frames, state.phase, state.turn_A_dir])

  return { state, start, reset, isDiagnose }
}
