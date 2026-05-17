import { useEffect, useMemo, useRef, useState } from 'react'

type LivenessLabel = 'live' | 'spoof' | 'no_face' | 'uncertain'

type InferResponse = {
  face_detected: boolean
  liveness_score: number
  liveness_label: LivenessLabel
  latency_ms: number
  message?: string | null
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'
const POLL_INTERVAL_MS = 300
const SCORE_WINDOW_SIZE = 5
const THRESHOLD_LIVE = 0.85
const THRESHOLD_SPOOF = 0.35

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const scoreWindowRef = useRef<number[]>([])
  const [cameraReady, setCameraReady] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<InferResponse | null>(null)
  const [smoothedScore, setSmoothedScore] = useState<number | null>(null)
  const [smoothedLabel, setSmoothedLabel] = useState<LivenessLabel>('no_face')

  const statusTone = useMemo(() => {
    switch (smoothedLabel) {
      case 'live':      return 'good'
      case 'spoof':     return 'bad'
      case 'uncertain': return 'warn'
      default:          return 'neutral'
    }
  }, [smoothedLabel])

  function onScoreReceived(score: number) {
    const w = scoreWindowRef.current
    w.push(score)
    if (w.length > SCORE_WINDOW_SIZE) w.shift()
    const avg = w.reduce((a, b) => a + b, 0) / w.length
    setSmoothedScore(avg)
    setSmoothedLabel(
      avg >= THRESHOLD_LIVE ? 'live' : avg <= THRESHOLD_SPOOF ? 'spoof' : 'uncertain'
    )
  }

  function captureFrameBase64(): string | null {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return null
    canvas.width = video.videoWidth || 640
    canvas.height = video.videoHeight || 480
    const ctx = canvas.getContext('2d')
    if (!ctx) return null
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    const dataUrl = canvas.toDataURL('image/jpeg', 0.85)
    return dataUrl.split(',')[1]
  }

  function captureAndSendAsync() {
    const imageBase64 = captureFrameBase64()
    if (!imageBase64) return
    fetch(`${API_BASE}/v1/liveness/infer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_base64: imageBase64 }),
    })
      .then(r => r.ok ? r.json() : Promise.reject(new Error(`API ${r.status}`)))
      .then((payload: InferResponse) => {
        setResult(payload)
        if (payload.face_detected) onScoreReceived(payload.liveness_score)
      })
      .catch(() => { /* silent in continuous mode */ })
  }

  async function captureAndScan() {
    if (!videoRef.current || !canvasRef.current) return
    setBusy(true)
    setError(null)
    try {
      const imageBase64 = captureFrameBase64()
      if (!imageBase64) throw new Error('Could not capture frame.')
      const response = await fetch(`${API_BASE}/v1/liveness/infer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: imageBase64 }),
      })
      if (!response.ok) throw new Error(`API request failed with status ${response.status}`)
      const payload = (await response.json()) as InferResponse
      setResult(payload)
      if (payload.face_detected) onScoreReceived(payload.liveness_score)
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : 'Unexpected error.')
    } finally {
      setBusy(false)
    }
  }

  async function startCamera() {
    setError(null)
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } })
    if (!videoRef.current) return
    videoRef.current.srcObject = stream
    await videoRef.current.play()
    setCameraReady(true)
  }

  useEffect(() => {
    if (!cameraReady) return
    const id = setInterval(captureAndSendAsync, POLL_INTERVAL_MS)
    return () => clearInterval(id)
  }, [cameraReady])

  return (
    <main className="page-shell">
      <section className="hero-card">
        <p className="eyebrow">Face Liveness MVP</p>
        <h1>Scan first. Authenticate later.</h1>
        <p className="lede">
          Browser webcam capture with a FastAPI liveness backend powered by MobileNetV2.
        </p>
        <div className="actions">
          <button onClick={() => void startCamera()} disabled={cameraReady || busy}>
            {cameraReady ? 'Camera Ready' : 'Start Camera'}
          </button>
          <button onClick={() => void captureAndScan()} disabled={!cameraReady || busy}>
            {busy ? 'Scanning…' : 'Capture and Scan'}
          </button>
        </div>
      </section>

      <section className="demo-grid">
        <div className="video-card">
          <video ref={videoRef} playsInline muted className="video-feed" />
          <canvas ref={canvasRef} hidden />
        </div>
        <div className={`result-card ${statusTone}`}>
          <h2>Result</h2>
          {smoothedScore !== null && (
            <p>
              <strong>Live score (smoothed):</strong>{' '}
              {smoothedScore.toFixed(2)} — <strong>{smoothedLabel.toUpperCase()}</strong>
            </p>
          )}
          {!result && !error ? <p>No scan yet.</p> : null}
          {error ? <p className="error">{error}</p> : null}
          {result ? (
            <>
              <p><strong>Raw score:</strong> {result.liveness_score.toFixed(2)}</p>
              <p><strong>Face detected:</strong> {String(result.face_detected)}</p>
              <p><strong>Latency:</strong> {result.latency_ms.toFixed(2)} ms</p>
              {result.message ? <p className="message">{result.message}</p> : null}
            </>
          ) : null}
        </div>
      </section>
    </main>
  )
}

export default App
