import { T_PASSIVE, YAW_CENTER, YAW_TARGET } from './fusion'
import type { FrameRecord, TurnDirection } from './types'

interface Props {
  frames: FrameRecord[]
  turn_A_dir: TurnDirection
  onRetry: () => void
}

interface PhaseReport {
  phase: FrameRecord['phase']
  labelVi: string
  frameCount: number
  detectRate: number
  /** Relative yaw: most-negative value (most leftward) */
  relYawMin: number | null
  /** Relative yaw: most-positive value (most rightward) */
  relYawMax: number | null
  /** Fraction of frames with |rel_yaw| ≤ YAW_CENTER */
  frontalPct: number
  passiveAvg: number
  passed: boolean
  noteVi: string
}

function computeBaseline(frames: FrameRecord[]): number {
  const fwd = frames.filter((f) => f.phase === 'forward' && f.yaw_deg !== null)
  return fwd.length > 0 ? fwd.reduce((s, f) => s + f.yaw_deg!, 0) / fwd.length : 0
}

function buildReport(
  frames: FrameRecord[],
  phase: FrameRecord['phase'],
  baseline: number,
  turn_A_dir: TurnDirection,
): PhaseReport {
  const pf = frames.filter((f) => f.phase === phase)
  const detected = pf.filter((f) => f.face_detected)
  const withYaw = pf.filter((f) => f.yaw_deg !== null)
  const relYaws = withYaw.map((f) => f.yaw_deg! - baseline)

  const detectRate = pf.length > 0 ? detected.length / pf.length : 0
  const relYawMin = relYaws.length > 0 ? Math.min(...relYaws) : null
  const relYawMax = relYaws.length > 0 ? Math.max(...relYaws) : null
  const frontalPct =
    withYaw.length > 0
      ? withYaw.filter((f) => Math.abs(f.yaw_deg! - baseline) <= YAW_CENTER).length / withYaw.length
      : 0
  const passiveAvg =
    detected.length > 0 ? detected.reduce((s, f) => s + f.passive_score, 0) / detected.length : 0

  const turnBDir: TurnDirection = turn_A_dir === 'right' ? 'left' : 'right'

  let labelVi = ''
  let passed = false
  let noteVi = ''

  switch (phase) {
    case 'forward': {
      labelVi = 'Nhìn thẳng'
      const frontalOk = pf.length === 0 || frontalPct >= 0.6
      const passiveOk = passiveAvg >= T_PASSIVE
      passed = frontalOk && passiveOk
      if (!frontalOk)
        noteVi = `Chỉ ${Math.round(frontalPct * 100)}% frames nhìn thẳng (cần ≥60%) — thử giữ đầu thẳng hơn khi đếm ngược xong`
      else if (!passiveOk)
        noteVi = `Điểm liveness trung bình ${passiveAvg.toFixed(2)} < ${T_PASSIVE} — model đánh giá là spoof`
      else noteVi = 'Nhìn thẳng đạt yêu cầu, điểm liveness tốt'
      break
    }
    case 'turn_A': {
      labelVi = `Quay ${turn_A_dir === 'right' ? 'PHẢI →' : 'TRÁI ←'}`
      const reached =
        turn_A_dir === 'right' ? (relYawMax ?? 0) >= YAW_TARGET : (relYawMin ?? 0) <= -YAW_TARGET
      passed = reached
      const got = turn_A_dir === 'right' ? (relYawMax ?? 0) : -(relYawMin ?? 0)
      noteVi = reached
        ? `Đạt ${got.toFixed(1)}° (cần ≥${YAW_TARGET}°) ✓`
        : `Chỉ đạt ${got.toFixed(1)}° tương đối (cần ≥${YAW_TARGET}°) — quay mạnh hơn và giữ pose`
      break
    }
    case 'center_1': {
      labelVi = 'Quay về giữa'
      const backToCenter =
        pf.length === 0 || withYaw.some((f) => Math.abs(f.yaw_deg! - baseline) <= YAW_CENTER)
      passed = backToCenter
      noteVi = passed
        ? 'Trở về nhìn thẳng đúng cách'
        : `Không có frame nào có |yaw| ≤ ${YAW_CENTER}° (tương đối) — hãy quay đầu về thẳng nhanh hơn`
      break
    }
    case 'turn_B': {
      labelVi = `Quay ${turnBDir === 'right' ? 'PHẢI →' : 'TRÁI ←'}`
      const reached =
        turnBDir === 'right' ? (relYawMax ?? 0) >= YAW_TARGET : (relYawMin ?? 0) <= -YAW_TARGET
      passed = reached
      const got = turnBDir === 'right' ? (relYawMax ?? 0) : -(relYawMin ?? 0)
      noteVi = reached
        ? `Đạt ${got.toFixed(1)}° (cần ≥${YAW_TARGET}°) ✓`
        : `Chỉ đạt ${got.toFixed(1)}° tương đối (cần ≥${YAW_TARGET}°) — quay mạnh hơn và giữ pose`
      break
    }
  }

  return {
    phase,
    labelVi,
    frameCount: pf.length,
    detectRate,
    relYawMin,
    relYawMax,
    frontalPct,
    passiveAvg,
    passed,
    noteVi,
  }
}

function Bar({ value, max, color }: { value: number; max: number; color: string }) {
  const pct = Math.min(100, Math.max(0, (Math.abs(value) / max) * 100))
  return (
    <div className="diag-bar-track">
      <div className="diag-bar-fill" style={{ width: `${pct}%`, background: color }} />
      <span className="diag-bar-label">
        {value >= 0 ? '+' : ''}
        {value.toFixed(1)}° / {value < 0 ? '-' : '+'}
        {max}°
      </span>
    </div>
  )
}

function PhaseCard({ report }: { report: PhaseReport }) {
  const isTurnPhase = report.phase === 'turn_A' || report.phase === 'turn_B'
  const isFrontalPhase = report.phase === 'forward' || report.phase === 'center_1'

  return (
    <div className={`diag-phase-card ${report.passed ? 'pass' : 'fail'}`}>
      <div className="diag-phase-header">
        <span className="diag-phase-name">{report.labelVi}</span>
        <span className={`diag-phase-badge ${report.passed ? 'pass' : 'fail'}`}>
          {report.passed ? '✓ PASS' : '✗ FAIL'}
        </span>
      </div>

      <div className="diag-phase-meta">
        <span>{report.frameCount} frames</span>
        <span>·</span>
        <span>Detect {Math.round(report.detectRate * 100)}%</span>
        {isFrontalPhase && report.phase === 'forward' && (
          <>
            <span>·</span>
            <span>Score {report.passiveAvg.toFixed(2)}</span>
          </>
        )}
      </div>

      {isTurnPhase && report.relYawMax !== null && report.relYawMin !== null && (
        <div className="diag-bars">
          <div className="diag-bar-row">
            <span className="diag-bar-key">Max →</span>
            <Bar value={report.relYawMax} max={YAW_TARGET * 2} color="#3742fa" />
          </div>
          <div className="diag-bar-row">
            <span className="diag-bar-key">Max ←</span>
            <Bar value={-report.relYawMin} max={YAW_TARGET * 2} color="#ff4757" />
          </div>
          <div className="diag-bar-target-line" style={{ left: `${(YAW_TARGET / (YAW_TARGET * 2)) * 100}%` }} />
        </div>
      )}

      {isFrontalPhase && (
        <div className="diag-bars">
          <div className="diag-bar-row">
            <span className="diag-bar-key">Thẳng</span>
            <div className="diag-bar-track">
              <div
                className="diag-bar-fill"
                style={{
                  width: `${Math.round(report.frontalPct * 100)}%`,
                  background: report.frontalPct >= 0.6 ? '#2ed573' : '#ff6b81',
                }}
              />
              <span className="diag-bar-label">
                {Math.round(report.frontalPct * 100)}% / 60%
              </span>
            </div>
          </div>
        </div>
      )}

      <p className="diag-phase-note">{report.noteVi}</p>
    </div>
  )
}

export function DiagnoseView({ frames, turn_A_dir, onRetry }: Props) {
  const baseline = computeBaseline(frames)
  const phases: FrameRecord['phase'][] = ['forward', 'turn_A', 'center_1', 'turn_B']
  const reports = phases.map((p) => buildReport(frames, p, baseline, turn_A_dir))

  const allPass = reports.every((r) => r.passed)
  const firstFail = reports.find((r) => !r.passed)

  return (
    <main className="result-page">
      <section className="diag-shell">
        <div className="diag-header">
          <h1>🔬 Kết quả chẩn đoán</h1>
          <p className="diag-subtitle">
            Chế độ debug — tất cả các bước đã chạy đến hết thời gian
          </p>
          <div className="diag-overall-badge">
            {allPass ? (
              <span className="diag-overall pass">✓ Sẽ PASS (THẬT)</span>
            ) : (
              <span className="diag-overall fail">
                ✗ Sẽ FAIL tại: <strong>{firstFail?.labelVi}</strong>
              </span>
            )}
          </div>
          <p className="diag-baseline">
            Yaw baseline (tham chiếu): <strong>{baseline.toFixed(1)}°</strong> — tất cả góc hiển thị là tương đối so với baseline này
          </p>
        </div>

        <div className="diag-phases">
          {reports.map((r) => (
            <PhaseCard key={r.phase} report={r} />
          ))}
        </div>

        <div className="diag-footer">
          <button onClick={onRetry}>Thử lại (normal mode)</button>
          <a href="?diagnose=1" className="secondary-button" style={{ padding: '0.5rem 1.2rem' }}>
            Chẩn đoán lại
          </a>
        </div>
      </section>
    </main>
  )
}
