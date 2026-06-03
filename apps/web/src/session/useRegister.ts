import { useCallback, useEffect, useRef, useState } from 'react'

import { YAW_CENTER } from './fusion'
import type { FrameApiResponse } from './types'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'
const CAPTURE_INTERVAL_MS = 20
const REGISTER_TIMEOUT_MS = 7000  // 3s countdown + 4s yaw check window
const REGISTRATION_FRAME_COUNT = 5
const REGISTRATION_CAPTURE_TIMEOUT_MS = 4000

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
  /** base64 images collected during multi-frame registration */
  captured_frames: string[]
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
    captured_frames: [],
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

  // ── Enroll with multiple frames ───────────────────────────────────────────
  const enrollWithFrames = useCallback(async (frames: string[]) => {
    setState((s) => ({ ...s, phase: 'capturing' }))

    try {
      // Extract embeddings in parallel
      const embedPromises = frames.map(frame =>
        fetch(`${API_BASE}/v1/auth/embed`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_base64: frame }),
        }).then(res => res.json())
      )

      const results = await Promise.all(embedPromises)
      const embeddings = results.map(r => r.embedding).filter(Boolean)

      if (embeddings.length === 0) {
        throw new Error('Không thể trích xuất đặc trưng khuôn mặt')
      }

      // Average embeddings
      const { meanEmbeddings } = await import('./fusion')
      const avgEmbedding = meanEmbeddings(embeddings)

      // Enroll with pre-computed embedding
      const response = await fetch(`${API_BASE}/v1/auth/enroll`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: stateRef.current.userId,
          embedding: avgEmbedding,
        }),
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
      captured_frames: [],
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

      // Timeout check
      if (elapsed >= REGISTRATION_CAPTURE_TIMEOUT_MS) {
        // If we have enough frames, enroll; otherwise error
        const frames = stateRef.current.captured_frames
        if (frames.length >= REGISTRATION_FRAME_COUNT) {
          hasEnrolled = true
          if (timerRef.current) clearInterval(timerRef.current)
          void enrollWithFrames(frames)
          return
        }
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

          // Countdown must complete before capturing frames
          const countdownComplete = elapsed >= COUNTDOWN_SEC * 1000

          // Check if centered face and collect frames
          const isCentered = faceDetected && yaw !== null && Math.abs(yaw) <= YAW_CENTER
          const currentFrames = stateRef.current.captured_frames

          if (!hasEnrolled && countdownComplete && isCentered && !currentFrames.includes(imageBase64)) {
            const newFrames = [...currentFrames, imageBase64]
            setState((s) => ({
              ...s,
              captured_frames: newFrames,
            }))

            // If we have enough frames, trigger enrollment
            if (newFrames.length >= REGISTRATION_FRAME_COUNT) {
              hasEnrolled = true
              if (timerRef.current) clearInterval(timerRef.current)
              setState((s) => ({
                ...s,
                capturedFrame: `data:image/jpeg;base64,${imageBase64}`,
              }))
              void enrollWithFrames(newFrames)
            }
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
  }, [captureFrameBase64, enroll, enrollWithFrames, state.phase])

  // ── Countdown timer display (separate from capture logic) ──────────────────────────
  useEffect(() => {
    if (state.phase !== 'countdown') return

    const startedAt = Date.now()
    const id = window.setInterval(() => {
      const elapsed = Date.now() - startedAt
      const next = Math.max(0, COUNTDOWN_SEC - Math.floor(elapsed / 1000))
      setState((s) => ({ ...s, countdown: next }))
    }, 100)

    return () => window.clearInterval(id)
  }, [state.phase])

  return { state, setUserId, start, reset }
}