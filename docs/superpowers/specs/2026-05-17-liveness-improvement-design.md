# Liveness Detection Improvement Design

**Date:** 2026-05-17
**Status:** Approved
**Scope:** MVP demo (university class)

---

## Problem Statement

The current liveness detection MVP works accurately under controlled conditions (dark room, phone screen spoof, removed background) but is unstable under:
- Varying lighting / bright ambient light
- Complex backgrounds (chairs, objects behind the subject)

Three root causes were identified from code analysis:

1. **Train-inference distribution shift (most critical):** Training uses CelebA-Spoof bbox + 15% margin crops (unaligned, 80×80). Inference uses InsightFace `aligned_crop_bgr` (landmark-aligned). The model sees a fundamentally different crop type at runtime than it was trained on.

2. **No augmentation + no ImageNet normalization:** `ManifestDataset.__getitem__` only does `/ 255.0`. The model has never seen lighting variation, so any change in illumination directly shifts the score.

3. **SmallFASNet is too shallow:** 3 conv layers, 128 channels max — insufficient capacity to learn robust texture-level features (real skin vs. screen texture). Prone to fitting background patterns from the training set instead of face texture.

---

## Chosen Approach: MobileNetV2 + Full Augmentation + Consistent Cropping

Replace `SmallFASNet` with a pretrained `MobileNetV2` backbone, fix the crop consistency between training and inference, add a full augmentation suite, and add ImageNet normalization everywhere.

The existing `LivenessService` and API contract remain unchanged. The model is a drop-in TorchScript replacement.

---

## Section 1: Architecture Overview

```
TRAINING PIPELINE (new Kaggle notebook)
  CelebA-Spoof raw images
      ↓
  InsightFaceDetector.detect() → aligned_crop (112×112)   [FIX #1: crop consistency]
      ↓  (drop images where no face detected, ~<5%)
  Augmentation (ColorJitter, Flip, Rotate, Blur)          [FIX #2: augmentation]
      ↓
  ImageNet Normalize (mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225])  [FIX #3]
      ↓
  MobileNetV2 + Linear(1280→2)                            [FIX #4: stronger backbone]
      ↓
  Export → TorchScript mobilenetv2_fas_scripted.pt

INFERENCE PIPELINE (minimal changes to existing code)
  Webcam frame (base64)
      ↓
  InsightFaceDetector.detect() → aligned_crop_bgr   [same crop as training now]
      ↓
  TorchLivenessModel._preprocess()   [update: 112×112 + ImageNet normalize]
      ↓
  MobileNetV2 TorchScript → liveness_score

FRONTEND
  Webcam → capture frame every 300ms → POST /v1/liveness/infer (async)
  Sliding window avg over last 5 scores → smoothed label display
  [GUIDE] WebSocket path for future phase
```

---

## Section 2: Data Pipeline & Augmentation

### 2.1 Crop Consistency Fix

The new training notebook will detect faces from CelebA-Spoof raw images using `InsightFaceDetector` (same detector used at inference) and save the `aligned_crop_bgr` as the training crop. Images where InsightFace finds no face are dropped (expected < 5% of dataset).

This eliminates the train-inference distribution shift entirely.

### 2.2 Augmentation

Applied to **train set only**. Val and test use the clean transform.

```python
# Train transform
transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Val/Test transform
transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

| Augmentation | Failure mode addressed |
|---|---|
| `ColorJitter brightness/contrast 0.4` | Ambient light variation |
| `RandomRotation(10°)` | Slight head tilt |
| `GaussianBlur` | Out-of-focus webcam, blurry spoof image |
| `RandomHorizontalFlip` | Left/right face symmetry, diversity |
| `ImageNet Normalize` | Compensates absolute pixel shift from lighting |

`RandomErasing` / `Cutout` intentionally excluded — masking parts of a 112×112 face crop removes discriminative features (eyes, nose region).

### 2.3 Class Balance

CelebA-Spoof has significantly more spoof than live samples. Use `WeightedRandomSampler` so each training batch is balanced:

```python
n_live  = (labels == 1).sum()
n_spoof = (labels == 0).sum()
sample_weights = [1/n_live if l == 1 else 1/n_spoof for l in labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights))
# Use sampler= in DataLoader, shuffle=False
```

---

## Section 3: Model Architecture & Training Config

### 3.1 Model

```python
def build_mobilenetv2_fas(pretrained: bool = True) -> nn.Module:
    backbone = models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, 2),
    )
    return backbone
```

| | SmallFASNet (current) | MobileNetV2 (new) |
|---|---|---|
| Parameters | ~180K | ~3.4M |
| Input size | 80×80 | 112×112 |
| Depth | 3 conv layers | 18 inverted residual blocks |
| Pretrained | No | ImageNet |
| CPU inference | ~5ms | ~20ms |

### 3.2 Training Config

```python
@dataclass
class TrainConfig:
    image_size: int = 112
    batch_size: int = 64
    epochs: int = 15
    lr_backbone: float = 1e-4   # pretrained weights: fine-tune gently
    lr_head: float = 1e-3       # new head: train faster
    weight_decay: float = 1e-4
    seed: int = 42
```

Two-phase learning rates: backbone uses `lr=1e-4` to preserve pretrained features; head uses `lr=1e-3` since it is randomly initialized.

```python
optimizer = torch.optim.AdamW([
    {'params': model.features.parameters(), 'lr': cfg.lr_backbone},
    {'params': model.classifier.parameters(), 'lr': cfg.lr_head},
], weight_decay=cfg.weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.epochs, eta_min=1e-6
)
```

### 3.3 Early Stopping & Metrics

- **Checkpoint metric:** `val_ACER` (Average Classification Error Rate = `(FPR + FNR) / 2`)
- **Early stopping patience:** 5 epochs
- ACER is the standard FAS metric — unlike accuracy, it catches models that classify all samples as one class.

### 3.4 Export

```python
model.eval()
scripted = torch.jit.script(model.cpu())
scripted.save('mobilenetv2_fas_scripted.pt')
```

Deployment: set `LIVENESS_MODEL_PATH=path/to/mobilenetv2_fas_scripted.pt`. No other service changes required.

### 3.5 Inference Preprocessing Update

Only `src/fas/liveness_model.py` needs two small changes:

```python
# TorchLivenessModel:
input_size: int = 112   # was 80

def _preprocess(self, face_crop_bgr):
    resized = cv2.resize(face_crop_bgr, (self.input_size, self.input_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = (arr - mean) / std
    chw  = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)
```

---

## Section 4: Real-time Frontend

### 4.1 HTTP Polling — MVP (Approach A)

Replace synchronous send-wait-send with an async fire-and-forget loop + sliding window smoothing.

**Capture loop (every 300ms, non-blocking):**
```typescript
useEffect(() => {
    const id = setInterval(() => {
        captureAndSendFrame();   // async, does not await
    }, 300);
    return () => clearInterval(id);
}, []);
```

**Score smoothing (deque of 5):**
```typescript
const scoreWindow = useRef<number[]>([]);

function onScoreReceived(score: number) {
    scoreWindow.current.push(score);
    if (scoreWindow.current.length > 5) scoreWindow.current.shift();
    const avg = scoreWindow.current.reduce((a, b) => a + b) / scoreWindow.current.length;
    setSmoothedScore(avg);
    setLabel(avg >= 0.85 ? 'LIVE' : avg <= 0.35 ? 'SPOOF' : 'UNCERTAIN');
}
```

Updated thresholds: `LIVE >= 0.85`, `SPOOF <= 0.35` (relaxed from 0.9/0.3 — the stronger model is more calibrated so the strict band is no longer needed).

### 4.2 WebSocket Streaming — Future Phase (Approach B)

**Backend — add one endpoint to `services/api/app.py`:**
```python
@app.websocket("/ws/liveness")
async def ws_liveness(websocket: WebSocket):
    await websocket.accept()
    service = LivenessService()
    try:
        while True:
            data = await websocket.receive_bytes()
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64 = base64.b64encode(buf).decode()
            result = service.infer(LivenessInferRequest(image_base64=b64))
            await websocket.send_json(result.model_dump())
    except WebSocketDisconnect:
        pass
```

**Frontend:**
```typescript
const ws = new WebSocket('ws://localhost:8000/ws/liveness');
ws.binaryType = 'arraybuffer';
setInterval(() => {
    canvas.toBlob(blob => blob?.arrayBuffer().then(buf => ws.send(buf)), 'image/jpeg', 0.7);
}, 100);   // ~10fps
ws.onmessage = e => onScoreReceived(JSON.parse(e.data).liveness_score);
```

**Implementation note:** Add `websockets` to `requirements.txt`. Each connection holds one InsightFace instance (~200MB RAM) — acceptable for single-user demo, needs pooling for multi-user.

---

## File Change Summary

| File | Change | Phase |
|---|---|---|
| `notebooks/kaggle_full_05_train_mobilenetv2.ipynb` | New training notebook (InsightFace crop + augmentation + MobileNetV2) | MVP |
| `src/fas/liveness_model.py` | `input_size=112` + ImageNet normalize in `_preprocess` | MVP |
| `apps/web/src/App.tsx` | Async polling loop + 5-frame score smoothing | MVP |
| `configs/training.yaml` | Update `image_size: 112`, add augmentation config | MVP |
| `services/api/app.py` | WebSocket endpoint `/ws/liveness` | Phase 2 |
| `requirements.txt` | Add `websockets` | Phase 2 |
