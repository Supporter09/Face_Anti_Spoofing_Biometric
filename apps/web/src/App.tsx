import { useMemo, useRef, useState } from 'react'

type LivenessLabel = 'live' | 'spoof' | 'no_face' | 'uncertain'

type InferResponse = {
  face_detected: boolean
  liveness_score: number
  liveness_label: LivenessLabel
  latency_ms: number
  message?: string | null
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [cameraReady, setCameraReady] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<InferResponse | null>(null)

  const statusTone = useMemo(() => {
    switch (result?.liveness_label) {
      case 'live':
        return 'good'
      case 'spoof':
        return 'bad'
      case 'uncertain':
        return 'warn'
      default:
        return 'neutral'
    }
  }, [result])

  async function startCamera() {
    setError(null)
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } })
    if (!videoRef.current) return
    videoRef.current.srcObject = stream
    await videoRef.current.play()
    setCameraReady(true)
  }

  async function captureAndScan() {
    if (!videoRef.current || !canvasRef.current) return

    setBusy(true)
    setError(null)

    try {
      const video = videoRef.current
      const canvas = canvasRef.current
      canvas.width = video.videoWidth || 640
      canvas.height = video.videoHeight || 480
      const context = canvas.getContext('2d')
      if (!context) throw new Error('Canvas context unavailable.')
      context.drawImage(video, 0, 0, canvas.width, canvas.height)
      const dataUrl = canvas.toDataURL('image/jpeg', 0.85)
      const imageBase64 = dataUrl.split(',')[1]

      const response = await fetch(`${API_BASE}/v1/liveness/infer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: imageBase64 }),
      })

      if (!response.ok) throw new Error(`API request failed with status ${response.status}`)
      const payload = (await response.json()) as InferResponse
      setResult(payload)
    } catch (caughtError) {
      const message = caughtError instanceof Error ? caughtError.message : 'Unexpected error.'
      setError(message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="page-shell">
      <section className="hero-card">
        <p className="eyebrow">Face Liveness MVP</p>
        <h1>Scan first. Authenticate later.</h1>
        <p className="lede">
          Browser webcam capture with a FastAPI liveness backend, designed to swap in
          `SCRFD + MiniFASNet` when the model pipeline is ready.
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
          {!result && !error ? <p>No scan yet.</p> : null}
          {error ? <p className="error">{error}</p> : null}
          {result ? (
            <>
              <p><strong>Label:</strong> {result.liveness_label}</p>
              <p><strong>Score:</strong> {result.liveness_score.toFixed(2)}</p>
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
