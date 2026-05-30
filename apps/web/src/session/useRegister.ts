import { useCallback, useEffect, useRef, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

const CAPTURE_WIDTH = 640
const CAPTURE_HEIGHT = 480
const JPEG_QUALITY = 0.85

// How long the countdown lasts before the face is captured
const COUNTDOWN_SEC = 3

export type RegisterPhase =
  | 'idle'         // initial state, form visible
  | 'countdown'    // 3-2-1 before capture
  | 'capturing'    // single frame grabbed, sent to API
  | 'success'      // enrolled OK
  | 'error'        // something went wrong

export interface RegisterState {
  phase: RegisterPhase
  countdown: number
  userId: string
  error: string | null
  /** base64 preview of the captured frame — shown as a still after capture */
  capturedFrame: string | null
}

function initialState(): RegisterState {
  return {
    phase: 'idle',
    countdown: COUNTDOWN_SEC,
    userId: '',
    error: null,
    capturedFrame: null,
  }
}

export function useRegister(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
) {
  const [state, setState] = useState<RegisterState>(initialState)
  const stateRef = useRef(state)
  useEffect(() => { stateRef.current = state }, [state])

  // ── Form field handler ────────────────────────────────────────────────────
  const setUserId = useCallback((userId: string) => {
    setState((s) => ({ ...s, userId }))
  }, [])

  // ── Reset ─────────────────────────────────────────────────────────────────
  const reset = useCallback(() => {
    setState(initialState())
  }, [])

  // ── Capture one frame to base64 ───────────────────────────────────────────
  const captureFrameBase64 = useCallback((): string | null => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return null

    canvas.width = CAPTURE_WIDTH
    canvas.height = CAPTURE_HEIGHT
    const ctx = canvas.getContext('2d')
    if (!ctx) return null

    ctx.drawImage(video, 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT)
    const dataUrl = canvas.toDataURL('image/jpeg', JPEG_QUALITY)
    return dataUrl.split(',')[1] ?? null
  }, [videoRef, canvasRef])

  // ── Enroll call ───────────────────────────────────────────────────────────
  const enroll = useCallback(async (imageBase64: string, userId: string) => {
    setState((s) => ({ ...s, phase: 'capturing' }))
    try {
      const response = await fetch(`${API_BASE}/v1/auth/enroll`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, image_base64: imageBase64 }),
      })
      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        throw new Error(body?.detail ?? `Server error ${response.status}`)
      }
      setState((s) => ({ ...s, phase: 'success' }))
    } catch (err) {
      setState((s) => ({
        ...s,
        phase: 'error',
        error: err instanceof Error ? err.message : 'Registration failed',
      }))
    }
  }, [])

  // ── Start: validate → begin countdown ────────────────────────────────────
  const start = useCallback(() => {
    const { userId } = stateRef.current
    if (!userId.trim()) {
      setState((s) => ({ ...s, error: 'Vui lòng nhập tên người dùng' }))
      return
    }
    setState((s) => ({
      ...s,
      phase: 'countdown',
      countdown: COUNTDOWN_SEC,
      error: null,
      capturedFrame: null,
    }))
  }, [])

  // ── Countdown tick → capture ──────────────────────────────────────────────
  useEffect(() => {
    if (state.phase !== 'countdown') return

    const id = window.setInterval(() => {
      setState((s) => {
        if (s.phase !== 'countdown') return s
        const next = s.countdown - 1
        if (next <= 0) {
          // Capture happens outside setState; signal via a sentinel
          return { ...s, countdown: 0 }
        }
        return { ...s, countdown: next }
      })
    }, 1000)

    return () => window.clearInterval(id)
  }, [state.phase])

  // ── When countdown hits 0 → grab frame + enroll ──────────────────────────
  useEffect(() => {
    if (state.phase !== 'countdown' || state.countdown !== 0) return

    const imageBase64 = captureFrameBase64()
    if (!imageBase64) {
      setState((s) => ({
        ...s,
        phase: 'error',
        error: 'Không thể chụp ảnh từ webcam',
      }))
      return
    }

    // Store preview frame (add back the data-url prefix for <img> src)
    setState((s) => ({
      ...s,
      capturedFrame: `data:image/jpeg;base64,${imageBase64}`,
    }))

    void enroll(imageBase64, stateRef.current.userId)
  }, [state.phase, state.countdown, captureFrameBase64, enroll])

  return { state, setUserId, start, reset }
}