import { T_PASSIVE, YAW_TARGET } from './fusion'
import type { SessionState } from './useSession'
import type { TurnDirection, Verdict } from './types'

interface Props {
  verdict: Verdict
  turn_A_dir: TurnDirection
  onRetry: () => void
  authStatus: SessionState['auth_status']
  identifiedUser: string | null
  similarity: number | null
  authMessage: string | null
}

function formatPct(value: number): string {
  return `${Math.round(value * 100)}%`
}

function failureReason(verdict: Verdict, turn_A_dir: TurnDirection): string {
  if (verdict.reason === 'passive_low') {
    return `Điểm liveness thấp (${verdict.passive_avg.toFixed(2)} < ${T_PASSIVE})`
  }
  if (verdict.reason === 'no_face') return 'Không phát hiện khuôn mặt'

  const challengeReason = verdict.challenge_eval.reason
  if (challengeReason === 'no_face' || challengeReason === 'no_frames') return 'Không phát hiện khuôn mặt'
  if (challengeReason === 'low_detect_rate') {
    return `Mất quá nhiều frames (${formatPct(verdict.challenge_eval.detect_rate)} < 90%)`
  }
  if (challengeReason === 'yaw_jump_detected') return 'Phát hiện chuyển động bất thường'
  if (challengeReason === 'forward_not_frontal') return 'Không nhìn thẳng trong giai đoạn chuẩn bị'
  if (challengeReason === 'turn_A_insufficient') {
    return `Không quay đầu đủ sang ${turn_A_dir === 'right' ? 'PHẢI' : 'TRÁI'}`
  }
  if (challengeReason === 'center_1_not_frontal') return 'Không quay về giữa'
  if (challengeReason === 'turn_B_insufficient') {
    return `Không quay đầu đủ sang ${turn_A_dir === 'right' ? 'TRÁI' : 'PHẢI'}`
  }
  return 'Không vượt qua thử thách chuyển động'
}

export function ResultView({
  verdict,
  turn_A_dir,
  onRetry,
  authStatus,
  identifiedUser,
  similarity,
  authMessage,
}: Props) {
  const isLive = verdict.verdict === 'LIVE'

  return (
    <main className="result-page">
      <section className={`session-result-card ${isLive ? 'live' : 'spoof'}`}>
        <div className="result-mark">{isLive ? '✓' : '✗'}</div>
        <h1>{isLive ? 'THẬT' : 'GIẢ MẠO'}</h1>

        {!isLive ? <p className="failure-reason">{failureReason(verdict, turn_A_dir)}</p> : null}

        {/* ── Auth result block (only shown for LIVE) ── */}
        {isLive && (
          <div className="result-auth-block">
            {authStatus === 'verifying' && (
              <div className="result-auth-verifying">
                <span className="result-auth-spinner" />
                Đang xác thực danh tính…
              </div>
            )}
            {authStatus === 'authenticated' && identifiedUser && (
              <div className="result-auth-success">
                <span className="result-auth-icon">👤</span>
                <span className="result-auth-name">{identifiedUser}</span>
                {similarity !== null && (
                  <span className="result-auth-score">({similarity.toFixed(3)})</span>
                )}
              </div>
            )}
            {authStatus === 'failed' && (
              <div className="result-auth-failed">
                ❌ {authMessage ?? 'Không nhận ra người dùng'}
              </div>
            )}
          </div>
        )}

        <dl className="result-stats">
          <div>
            <dt>Passive score</dt>
            <dd>
              {verdict.passive_avg.toFixed(2)} {verdict.passive_avg >= T_PASSIVE ? '≥' : '<'} {T_PASSIVE}
            </dd>
          </div>
          <div>
            <dt>Challenge</dt>
            <dd>{verdict.challenge_eval.pass ? '✓' : '✗'}</dd>
          </div>
          <div>
            <dt>Detect rate</dt>
            <dd>{formatPct(verdict.challenge_eval.detect_rate)} / 90%</dd>
          </div>
          <div>
            <dt>Frames</dt>
            <dd>{verdict.frame_count}</dd>
          </div>
          <div>
            <dt>Duration</dt>
            <dd>{(verdict.duration_ms / 1000).toFixed(1)}s</dd>
          </div>
          <div>
            <dt>Quay trái (rel.)</dt>
            <dd>
              {verdict.challenge_eval.max_yaw_left?.toFixed(1) ?? '—'}° / -{YAW_TARGET}°
            </dd>
          </div>
          <div>
            <dt>Quay phải (rel.)</dt>
            <dd>
              {verdict.challenge_eval.max_yaw_right?.toFixed(1) ?? '—'}° / +{YAW_TARGET}°
            </dd>
          </div>
        </dl>

        <button className={isLive ? 'secondary-button' : 'danger-button'} onClick={onRetry}>
          {isLive ? 'Xác minh lại' : 'Thử lại'}
        </button>
      </section>
    </main>
  )
}