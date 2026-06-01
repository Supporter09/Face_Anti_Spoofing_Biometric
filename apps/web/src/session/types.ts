export type Phase =
  | 'idle'
  | 'countdown'
  | 'forward'
  | 'turn_A'
  | 'center_1'
  | 'turn_B'
  | 'evaluating'
  | 'result'

export type TurnDirection = 'left' | 'right'

export interface FrameRecord {
  ts_ms: number
  phase: 'forward' | 'turn_A' | 'center_1' | 'turn_B'
  face_detected: boolean
  passive_score: number
  yaw_deg: number | null
  pose_ok: boolean
}

export interface ChallengeEval {
  pass: boolean
  detect_rate: number
  max_yaw_left: number | null
  max_yaw_right: number | null
  reason?: string
}

export type VerdictLabel = 'LIVE' | 'SPOOF'

export interface Verdict {
  verdict: VerdictLabel
  passive_avg: number
  challenge_eval: ChallengeEval
  reason?: 'challenge_failed' | 'passive_low' | 'no_face'
  frame_count: number
  duration_ms: number
}

/** Raw response from /v1/liveness/frame */
export interface FrameApiResponse {
  face_detected: boolean
  liveness_score: number
  liveness_label: string
  latency_ms: number
  yaw_deg: number | null
  pitch_deg: number | null
  pose_ok: boolean
  /** [x1, y1, x2, y2] in capture-image pixel coordinates (640×480) */
  face_bbox_xyxy: [number, number, number, number] | null
  /** 5 landmarks: [left_eye, right_eye, nose, mouth_left, mouth_right] each as [x, y] */
  face_landmarks: [number, number][] | null
  message?: string | null
}
export interface AuthResponse {
  authenticated?: boolean
  success?: boolean
  user_id: string
  similarity?: number
  threshold?: number
  message: string
}