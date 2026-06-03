import { useEffect, useRef } from 'react'
import type React from 'react'

import type { SessionState } from './useSession'

interface Props {
  videoRef: React.RefObject<HTMLVideoElement | null>
  state: SessionState
  onStart: () => void
  onReset: () => void
  onRegister: () => void
}

const CAP_W = 640
const CAP_H = 480
const CONTEXT_MARGIN_RATIO = 1.2
const LANDMARK_LABELS = ['L_eye', 'R_eye', 'nose', 'L_mouth', 'R_mouth'] as const
const LANDMARK_COLORS = ['#ff4757', '#3742fa', '#ffa502', '#2ed573', '#1e90ff'] as const

function drawOverlay(
  canvas: HTMLCanvasElement,
  bbox: [number, number, number, number] | null,
  landmarks: [number, number][] | null,
  passive: number | null,
  yaw: number | null,
) {
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.clearRect(0, 0, CAP_W, CAP_H)
  if (!bbox) return

  const [x1, y1, x2, y2] = bbox
  const bw = x2 - x1
  const bh = y2 - y1

  const mx = bw * CONTEXT_MARGIN_RATIO
  const my = bh * CONTEXT_MARGIN_RATIO
  const cx1 = Math.max(0, x1 - mx)
  const cy1 = Math.max(0, y1 - my)
  const cx2 = Math.min(CAP_W, x2 + mx)
  const cy2 = Math.min(CAP_H, y2 + my)
  const side = Math.max(cx2 - cx1, cy2 - cy1)
  const ccx = (cx1 + cx2) / 2
  const ccy = (cy1 + cy2) / 2
  const sqx1 = Math.max(0, ccx - side / 2)
  const sqy1 = Math.max(0, ccy - side / 2)
  const sqx2 = Math.min(CAP_W, ccx + side / 2)
  const sqy2 = Math.min(CAP_H, ccy + side / 2)

  ctx.save()
  ctx.strokeStyle = '#ff9f43'
  ctx.lineWidth = 2
  ctx.setLineDash([8, 5])
  ctx.strokeRect(sqx1, sqy1, sqx2 - sqx1, sqy2 - sqy1)
  ctx.setLineDash([])
  ctx.restore()

  ctx.save()
  ctx.strokeStyle = '#2ed573'
  ctx.lineWidth = 2.5
  ctx.strokeRect(x1, y1, bw, bh)
  ctx.restore()

  if (passive !== null) {
    const label = `live: ${passive.toFixed(2)}`
    ctx.save()
    ctx.font = 'bold 13px monospace'
    ctx.fillStyle = passive >= 0.4 ? '#2ed573' : '#ff6b81'
    const tw = ctx.measureText(label).width
    ctx.fillStyle = 'rgba(0,0,0,0.55)'
    ctx.fillRect(x1, y1 - 20, tw + 8, 18)
    ctx.fillStyle = passive >= 0.4 ? '#2ed573' : '#ff6b81'
    ctx.fillText(label, x1 + 4, y1 - 5)
    ctx.restore()
  }

  if (yaw !== null) {
    const yawLabel = `yaw: ${yaw.toFixed(1)}°`
    ctx.save()
    ctx.font = 'bold 12px monospace'
    const tw = ctx.measureText(yawLabel).width
    ctx.fillStyle = 'rgba(0,0,0,0.55)'
    ctx.fillRect(x1, y2 + 2, tw + 8, 17)
    ctx.fillStyle = '#ecf0f1'
    ctx.fillText(yawLabel, x1 + 4, y2 + 15)
    ctx.restore()
  }

  if (landmarks && landmarks.length === 5) {
    landmarks.forEach(([lx, ly], i) => {
      ctx.save()
      ctx.beginPath()
      ctx.arc(lx, ly, 4, 0, Math.PI * 2)
      ctx.fillStyle = LANDMARK_COLORS[i]
      ctx.fill()
      ctx.strokeStyle = '#000'
      ctx.lineWidth = 0.8
      ctx.stroke()
      ctx.restore()

      ctx.save()
      ctx.font = '9px monospace'
      ctx.fillStyle = LANDMARK_COLORS[i]
      ctx.fillText(LANDMARK_LABELS[i], lx + 6, ly - 3)
      ctx.restore()
    })
  }
}

export function SessionView({ videoRef, state, onStart, onReset, onRegister }: Props) {
  const showDebug = new URLSearchParams(window.location.search).get('debug') === '1'
  const overlayRef = useRef<HTMLCanvasElement | null>(null)
  const displayVideoRef = useRef<HTMLVideoElement | null>(null)

  // Mirror the App-level stream into the visible <video>
  useEffect(() => {
    const src = videoRef.current
    const dst = displayVideoRef.current
    if (!src || !dst) return
    const attach = () => {
      if (src.srcObject instanceof MediaStream) dst.srcObject = src.srcObject
    }
    if (src.readyState >= 2) attach()
    else src.addEventListener('loadeddata', attach, { once: true })
    return () => {
      src.removeEventListener('loadeddata', attach)
      dst.srcObject = null
    }
  }, [videoRef])

  // Redraw overlay whenever new frame data arrives
  useEffect(() => {
    if (overlayRef.current) {
      drawOverlay(
        overlayRef.current,
        state.latest_bbox,
        state.latest_landmarks,
        state.latest_passive,
        state.latest_yaw,
      )
    }
  }, [state.latest_bbox, state.latest_landmarks, state.latest_passive, state.latest_yaw])

  // Clear overlay on phase reset
  useEffect(() => {
    if ((state.phase === 'idle' || state.phase === 'countdown') && overlayRef.current) {
      const ctx = overlayRef.current.getContext('2d')
      ctx?.clearRect(0, 0, CAP_W, CAP_H)
    }
  }, [state.phase])

  return (
    <main className="session-page">
      <section className="session-shell">
        <div className="session-video-wrap">
          <video ref={displayVideoRef} autoPlay playsInline muted className="session-video" />

          <canvas
            ref={overlayRef}
            width={CAP_W}
            height={CAP_H}
            className="session-overlay"
          />

          {state.instruction ? <div className="session-instruction">{state.instruction}</div> : null}
          {state.phase === 'countdown' ? <div className="session-countdown">{state.countdown}</div> : null}
        </div>

        {showDebug ? (
          <div className="session-debug">
            yaw: {state.latest_yaw?.toFixed(1) ?? '—'}° | score: {state.latest_passive?.toFixed(2) ?? '—'} |
            phase: {state.phase} | face: {String(state.face_detected)}
          </div>
        ) : null}

        {state.error ? <p className="session-error">{state.error}</p> : null}

        {state.auth_status !== 'idle' ? (
          <div className="session-auth-status">
            {state.auth_status === 'verifying' && '🔐 Đang xác thực...'}

            {state.auth_status === 'authenticated' && (
              <>
                ✅ Xác thực thành công
                {state.identified_user ? (
                  <span className="session-auth-user"> — {state.identified_user}</span>
                ) : null}
                {state.similarity !== null ? (
                  <span className="session-auth-score"> ({(state.similarity * 100).toFixed(1)}%)</span>
                ) : null}
              </>
            )}

            {state.auth_status === 'failed' && (
              <>❌ {state.auth_message}</>
            )}
          </div>
        ) : null}

        {/* ── Actions ── */}
        <div className="session-actions">
          {state.phase === 'idle' ? (
            <>
              <button onClick={onStart}>Bắt đầu kiểm tra</button>
              <button className="secondary-button" onClick={onRegister}>
                📋 Đăng ký người dùng
              </button>
            </>
          ) : null}
          {state.phase !== 'idle' && state.phase !== 'result' ? (
            <button className="secondary-button" onClick={onReset}>
              Xác minh lại
            </button>
          ) : null}
        </div>

      </section>
    </main>
  )
}