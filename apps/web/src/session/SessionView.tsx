import type React from 'react'

import type { SessionState } from './useSession'

interface Props {
  videoRef: React.RefObject<HTMLVideoElement | null>
  state: SessionState
  onStart: () => void
  onReset: () => void
}

export function SessionView({ videoRef, state, onStart, onReset }: Props) {
  const showDebug = new URLSearchParams(window.location.search).get('debug') === '1'

  return (
    <main className="session-page">
      <section className="session-shell">
        <div className="session-video-wrap">
          <video ref={videoRef as React.Ref<HTMLVideoElement>} playsInline muted className="session-video" />
          {state.instruction ? <div className="session-instruction">{state.instruction}</div> : null}
          {state.phase === 'countdown' ? <div className="session-countdown">{state.countdown}</div> : null}
        </div>

        {showDebug ? (
          <div className="session-debug">
            yaw: {state.latest_yaw?.toFixed(1) ?? '—'}° | score: {state.latest_passive?.toFixed(2) ?? '—'} |
            phase: {state.phase}
          </div>
        ) : null}

        {state.error ? <p className="session-error">{state.error}</p> : null}

        <div className="session-actions">
          {state.phase === 'idle' ? <button onClick={onStart}>Bắt đầu kiểm tra</button> : null}
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
