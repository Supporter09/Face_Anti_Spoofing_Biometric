import { useMemo, useRef, useState } from 'react'

type LivenessLabel = 'live' | 'spoof' | 'no_face' | 'uncertain'

type InferResponse = {
  face_detected: boolean
  liveness_score: number
  liveness_label: LivenessLabel
  latency_ms: number
  message?: string | null
  face_bbox_xyxy?: number[] | null
  face_landmarks?: number[][] | null
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const inferenceCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const [cameraReady, setCameraReady] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<InferResponse | null>(null)
  const [captured, setCaptured] = useState(false)
  

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

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user' },
    })

    if (!videoRef.current) return

    videoRef.current.srcObject = stream
    await videoRef.current.play()

    setCameraReady(true)
  }

  function drawBoundingBox(payload: InferResponse) {
    const canvas = canvasRef.current
    if (!canvas || !payload.face_bbox_xyxy || payload.face_bbox_xyxy.length !== 4) {
      return
    }

    const context = canvas.getContext('2d')
    if (!context) return

    const [x1, y1, x2, y2] = payload.face_bbox_xyxy
    const width = x2 - x1
    const height = y2 - y1

    context.lineWidth = 4

    if (payload.liveness_label === 'live') {
      context.strokeStyle = '#22c55e'
      context.fillStyle = '#22c55e'
    } else if (payload.liveness_label === 'spoof') {
      context.strokeStyle = '#ef4444'
      context.fillStyle = '#ef4444'
    } else {
      context.strokeStyle = '#f59e0b'
      context.fillStyle = '#f59e0b'
    }

    context.strokeRect(x1, y1, width, height)

    const labelText = `${payload.liveness_label} ${payload.liveness_score.toFixed(2)}`
    context.font = '20px Arial'

    const textWidth = context.measureText(labelText).width
    const labelX = x1
    const labelY = Math.max(0, y1 - 28)

    context.fillRect(labelX, labelY, textWidth + 16, 28)

    context.fillStyle = '#ffffff'
    context.fillText(labelText, labelX + 8, labelY + 20)

    if (payload.face_landmarks) {
      for (const point of payload.face_landmarks) {
        if (point.length < 2) continue
        const [px, py] = point
        context.beginPath()
        context.arc(px, py, 3, 0, Math.PI * 2)
        context.fill()
      }
    }
  }

  function sleep(ms: number) {
    return new Promise((resolve) => setTimeout(resolve, ms))
  }

  function median(values: number[]) {
    if (values.length === 0) return 0

    const sorted = [...values].sort((a, b) => a - b)
    const middle = Math.floor(sorted.length / 2)

    if (sorted.length % 2 === 1) {
      return sorted[middle]
    }

    return (sorted[middle - 1] + sorted[middle]) / 2
  }

  async function captureAndScan() {
    if (!videoRef.current || !canvasRef.current || !inferenceCanvasRef.current) {
      return
    }

    setBusy(true)
    setError(null)
    setResult(null)

    try {
      const video = videoRef.current
      const displayCanvas = canvasRef.current
      const inferenceCanvas = inferenceCanvasRef.current

      const displayContext = displayCanvas.getContext('2d')
      const inferenceContext = inferenceCanvas.getContext('2d')

      if (!displayContext || !inferenceContext) {
        throw new Error('Canvas context unavailable.')
      }

      const width = video.videoWidth || 640
      const height = video.videoHeight || 480

      displayCanvas.width = width
      displayCanvas.height = height
      inferenceCanvas.width = width
      inferenceCanvas.height = height

      const attempts: Array<{
        payload: InferResponse
        image: ImageData
      }> = []

      const frameCount = 5
      const frameDelayMs = 180

      for (let i = 0; i < frameCount; i += 1) {
        inferenceContext.drawImage(video, 0, 0, width, height)

        const imageData = inferenceContext.getImageData(0, 0, width, height)

        // Keep same encoding style as your old version.
        const dataUrl = inferenceCanvas.toDataURL('image/jpeg', 0.85)
        const imageBase64 = dataUrl.split(',')[1]

        const response = await fetch(`${API_BASE}/v1/liveness/infer`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_base64: imageBase64 }),
        })

        if (!response.ok) {
          throw new Error(`API request failed with status ${response.status}`)
        }

        const payload = (await response.json()) as InferResponse

        if (payload.face_detected) {
          attempts.push({
            payload,
            image: imageData,
          })
        }

        if (i < frameCount - 1) {
          await sleep(frameDelayMs)
        }
      }

      if (attempts.length === 0) {
        const noFaceResult: InferResponse = {
          face_detected: false,
          liveness_score: 0,
          liveness_label: 'no_face',
          latency_ms: 0,
          message: 'No face detected in the captured frames.',
        }

        setResult(noFaceResult)
        return
      }

      const scores = attempts.map((item) => item.payload.liveness_score)
      const medianScore = median(scores)

      const selected = attempts.reduce((best, current) => {
        const bestDistance = Math.abs(best.payload.liveness_score - medianScore)
        const currentDistance = Math.abs(current.payload.liveness_score - medianScore)

        return currentDistance < bestDistance ? current : best
      })

      let finalLabel: LivenessLabel = 'uncertain'

      if (medianScore < 0.1) {
        finalLabel = 'spoof'
      } else if (medianScore >= 0.8) {
        finalLabel = 'live'
      } else {
        finalLabel = 'uncertain'
      }

      const totalLatency = attempts.reduce(
        (sum, item) => sum + item.payload.latency_ms,
        0
      )

      const finalPayload: InferResponse = {
        ...selected.payload,
        liveness_score: medianScore,
        liveness_label: finalLabel,
        latency_ms: totalLatency,
        message: `Median score from ${attempts.length}/${frameCount} valid frames.`,
      }

      displayContext.putImageData(selected.image, 0, 0)
      setCaptured(true)
      setResult(finalPayload)

      requestAnimationFrame(() => {
        drawBoundingBox(finalPayload)
      })
    } catch (caughtError) {
      const message =
        caughtError instanceof Error ? caughtError.message : 'Unexpected error.'
      setError(message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="page-shell">
      <section className="hero-card">
        <div>
          <p className="eyebrow">Face Anti-Spoofing</p>
          <h1>Face Liveness MVP</h1>
          <p className="subtitle">
            Capture one webcam frame, send it to the backend, then display the
            captured image with the detected face bounding box.
          </p>
        </div>

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
          <h2>Camera</h2>

          <video
            ref={videoRef}
            className={captured ? 'video-feed small-preview' : 'video-feed'}
            playsInline
            muted
          />

          <h2>Captured frame</h2>

          <canvas
            ref={canvasRef}
            className="capture-canvas"
            style={{ display: captured ? 'block' : 'none' }}
          />

          <canvas
            ref={inferenceCanvasRef}
            style={{ display: "none" }}
          />

          {!captured ? <p className="empty">No captured image yet.</p> : null}
        </div>

        <div className={`result-card ${statusTone}`}>
          <h2>Result</h2>

          {!result && !error ? <p className="empty">No scan yet.</p> : null}

          {error ? <p className="error">{error}</p> : null}

          {result ? (
            <>
              <p>
                <strong>Label:</strong> {result.liveness_label}
              </p>
              <p>
                <strong>Score:</strong> {result.liveness_score.toFixed(2)}
              </p>
              <p>
                <strong>Face detected:</strong> {String(result.face_detected)}
              </p>
              <p>
                <strong>Latency:</strong> {result.latency_ms.toFixed(2)} ms
              </p>

              {result.face_bbox_xyxy ? (
                <p>
                  <strong>Bounding box:</strong>{' '}
                  [{result.face_bbox_xyxy.join(', ')}]
                </p>
              ) : null}

              {result.message ? <p className="message">{result.message}</p> : null}
            </>
          ) : null}
        </div>
      </section>
    </main>
  )
}

export default App