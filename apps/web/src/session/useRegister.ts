import { useCallback, useEffect, useRef, useState } from 'react'

import { YAW_CENTER } from './fusion'
import type { FrameApiResponse } from './types'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'
const CAPTURE_INTERVAL_MS = 20
const REGISTER_TIMEOUT_MS = 4000

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
  // NEW:
  latest_yaw: number | null
  face_detected: boolean
}

function initialState(): RegisterState {
  return {
    phase: 'idle',
    countdown: COUNTDOWN_SEC,
    userId: '',
    error: null,
    capturedFrame: null,
    latest_yaw: null,
    face_detected: false,
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
    console.log('[Register] Start button clicked!')
    const { userId } = stateRef.current
    if (!userId.trim()) {
      setState((s) => ({ ...s, error: 'Vui lòng nhập tên người dùng' }))
      return
    }
    console.log('[Register] Setting phase to countdown, userId:', userId)
    setState((s) => ({
      ...s,
      phase: 'countdown',
      countdown: COUNTDOWN_SEC,
      error: null,
      capturedFrame: null,
    }))
  }, [])

  // ── Continuous frame capture with yaw check ──────────────────────────────────────
  useEffect(() => {
    console.log('[Register] useEffect triggered, phase:', state.phase)
    if (state.phase !== 'countdown') {
      console.log('[Register] Skipping - phase is not countdown')
      return
    }

    const startedAt = Date.now()
    const timerRef = { current: null as number | null }
    let hasEnrolled = false

    const tick = () => {
      const elapsed = Date.now() - startedAt
      const currentUserId = stateRef.current.userId
      // Calculate countdown from elapsed time (same as countdown effect)
      const countdownFromElapsed = Math.max(0, 3 - Math.floor(elapsed / 1000))

      // If countdown completed (reached 0), enroll with current frame regardless of yaw
      if (countdownFromElapsed === 0 && !hasEnrolled) {
        hasEnrolled = true
        if (timerRef.current) clearInterval(timerRef.current)
        const imageBase64 = captureFrameBase64()
        if (imageBase64) {
          setState((s) => ({
            ...s,
            capturedFrame: `data:image/jpeg;base64,${imageBase64}`,
          }))
          void enroll(imageBase64, currentUserId)
        }
        return
      }

      // Timeout check
      if (elapsed >= REGISTER_TIMEOUT_MS) {
        if (timerRef.current) clearInterval(timerRef.current)
        setState((s) => ({
          ...s,
          phase: 'error',
          error: 'Không tìm thấy khuôn mặt nhìn thẳng',
        }))
        return
      }

      const imageBase64 = captureFrameBase64()
      if (!imageBase64) return

      fetch(`${API_BASE}/v1/liveness/frame`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: imageBase64 }),
      })
        .then((res) => res.json())
        .then((payload: FrameApiResponse) => {
          const yaw = payload.yaw_deg
          const faceDetected = payload.face_detected

          setState((s) => ({
            ...s,
            latest_yaw: yaw,
            face_detected: faceDetected,
          }))

          // If centered face found, enroll immediately
          if (!hasEnrolled && faceDetected && yaw !== null && Math.abs(yaw) <= YAW_CENTER) {
            hasEnrolled = true
            if (timerRef.current) clearInterval(timerRef.current)
            setState((s) => ({
              ...s,
              capturedFrame: `data:image/jpeg;base64,${imageBase64}`,
            }))
            void enroll(imageBase64, currentUserId)
          }
        })
        .catch((err) => {
          console.warn('Frame capture error:', err)
        })
    }

    timerRef.current = window.setInterval(tick, CAPTURE_INTERVAL_MS)

    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [captureFrameBase64, enroll])

  // ── Countdown timer display (separate from capture logic) ──────────────────────────
  useEffect(() => {
    if (state.phase !== 'countdown') return

    const startedAt = Date.now()
    const id = window.setInterval(() => {
      const elapsed = Date.now() - startedAt
      const next = Math.max(0, 3 - Math.floor(elapsed / 1000))
      setState((s) => ({ ...s, countdown: next }))
    }, 100)

    return () => window.clearInterval(id)
  }, [state.phase])

  return { state, setUserId, start, reset }
}