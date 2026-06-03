import { useEffect, useRef } from 'react'
import type React from 'react'

import { useRegister } from './useRegister'

const CAP_W = 640
const CAP_H = 480

interface Props {
  videoRef: React.RefObject<HTMLVideoElement | null>
  canvasRef: React.RefObject<HTMLCanvasElement | null>
  onBack: () => void
}

export function RegisterView({ videoRef, canvasRef, onBack }: Props) {
  const { state, setUserId, start, reset } = useRegister(videoRef, canvasRef)
  const displayVideoRef = useRef<HTMLVideoElement | null>(null)

  // Mirror the existing stream into the visible <video> in this component.
  // The App-level <video> holds the MediaStream; we clone its tracks here
  // so the camera is never restarted.
  useEffect(() => {
    const src = videoRef.current
    const dst = displayVideoRef.current
    if (!src || !dst) return

    const attach = () => {
      if (src.srcObject instanceof MediaStream) {
        dst.srcObject = src.srcObject
      }
    }

    if (src.readyState >= 2) {
      // Already has data — attach immediately
      attach()
    } else {
      src.addEventListener('loadeddata', attach, { once: true })
    }

    return () => {
      src.removeEventListener('loadeddata', attach)
      dst.srcObject = null
    }
  }, [videoRef])

  const isActive = state.phase !== 'idle'
  const isLoading = state.phase === 'capturing'

  return (
    <main className="session-page">
      <section className="session-shell">

        {/* ── Header ──────────────────────────────────────────────────────── */}
        <div className="register-header">
          <button className="register-back-btn" onClick={onBack}>
            ← Quay lại
          </button>
          <h1 className="register-title">Đăng ký khuôn mặt</h1>
          <p className="register-subtitle">
            Nhìn thẳng vào camera và giữ yên khi đếm ngược kết thúc
          </p>
        </div>

        {/* ── Video area ───────────────────────────────────────────────────── */}
        <div className="session-video-wrap">
          {(state.phase === 'success' || state.phase === 'error') && state.capturedFrame ? (
            // Show still after capture
            <img
              src={state.capturedFrame}
              alt="Ảnh đã chụp"
              className="session-video register-still"
              style={{ transform: 'scaleX(-1)' }}
            />
          ) : (
            // Visible display video — streams from the shared App-level MediaStream
            <video
              ref={displayVideoRef}
              autoPlay
              playsInline
              muted
              className="session-video"
            />
          )}

          {state.phase === 'countdown' && state.countdown > 0 && (
            <div className="session-countdown">{state.countdown}</div>
          )}
          {state.phase === 'countdown' && (
            <div className="session-instruction">Nhìn thẳng vào camera…</div>
          )}
          {state.phase === 'countdown' && state.latest_yaw !== null && (
            <div className="session-yaw-indicator">
              Góc quay: {state.latest_yaw.toFixed(1)}° (
            {Math.abs(state.latest_yaw) <= 10 ? '✓ Đã căn giữa' : '↔ Cần căn giữa'})
            </div>
          )}
          {(state.phase === 'countdown' || state.phase === 'capturing') && state.captured_frames.length > 0 && (
            <div className="session-yaw-indicator">
              Đang thu thập: {state.captured_frames.length}/5 khung hình
            </div>
          )}
          {isLoading && (
            <div className="register-processing-overlay">
              <div className="register-spinner" />
              <span>Đang xử lý…</span>
            </div>
          )}
        </div>

        {/* ── Status banners ───────────────────────────────────────────────── */}
        {state.phase === 'success' && (
          <div className="session-auth-status">
            ✅ Đăng ký thành công! Khuôn mặt của bạn đã được lưu.
          </div>
        )}
        {state.phase === 'error' && state.error && (
          <p className="session-error">{state.error}</p>
        )}

        {/* ── Form ─────────────────────────────────────────────────────────── */}
        <div className="register-form">
          <label className="register-label" htmlFor="reg-username">
            Tên người dùng (User ID)
          </label>
          <input
            id="reg-username"
            className="register-input"
            type="text"
            placeholder="vd: student001"
            value={state.userId}
            disabled={isActive}
            onChange={(e) => setUserId(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !isActive) start() }}
          />
        </div>

        {/* ── Actions ──────────────────────────────────────────────────────── */}
        <div className="session-actions">
          {state.phase === 'idle' && (
            <button onClick={start}>📷 Bắt đầu đăng ký</button>
          )}
          {(state.phase === 'error' || state.phase === 'success') && (
            <button className="secondary-button" onClick={reset}>Thử lại</button>
          )}
          {state.phase === 'success' && (
            <button onClick={onBack}>Về trang chính</button>
          )}
        </div>

      </section>
    </main>
  )
}