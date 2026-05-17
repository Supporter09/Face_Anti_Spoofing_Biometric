# Liveness Detection Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unstable SmallFASNet liveness model with MobileNetV2 + full augmentation, fix the train/inference crop distribution shift, and upgrade the frontend to continuous real-time polling with score smoothing.

**Architecture:** MobileNetV2 pretrained on ImageNet is fine-tuned on CelebA-Spoof using aligned face crops, full augmentation, and ImageNet normalization — matching exactly the preprocessing used at inference time. The TorchScript export format stays the same so `LivenessService` needs no changes beyond updating `_preprocess` in `liveness_model.py`. The frontend switches from manual one-shot scan to an async interval loop with a 5-frame sliding window to smooth out per-frame noise.

**Tech Stack:** PyTorch, torchvision (MobileNetV2), OpenCV, insightface, React + TypeScript (frontend), FastAPI (backend, Phase 2 WebSocket)

---

## File Map

| File | Action | Task |
|---|---|---|
| `tests/test_liveness_preprocess.py` | Create — unit tests for new preprocessing | Task 1 |
| `src/fas/liveness_model.py` | Modify — `input_size=112`, ImageNet normalize | Task 1 |
| `notebooks/kaggle_full_05_train_mobilenetv2.ipynb` | Create — full training notebook | Task 2 |
| `configs/training.yaml` | Modify — `image_size: 112` | Task 3 |
| `apps/web/src/App.tsx` | Modify — continuous polling + score smoothing | Task 4 |
| `services/api/app.py` | Modify — add `/ws/liveness` WebSocket endpoint | Task 5 (Phase 2) |
| `requirements.txt` | Modify — add `websockets` | Task 5 (Phase 2) |

---

## Task 1: Update Inference Preprocessing (`liveness_model.py`)

**Files:**
- Create: `tests/test_liveness_preprocess.py`
- Modify: `src/fas/liveness_model.py:10,69-83`

This is the only production code change needed for inference — everything else is training/frontend. Change `input_size` from 80 to 112 and add ImageNet mean/std normalization so the model sees the same pixel distribution it was trained on.

- [ ] **Step 1: Write failing tests**

Create `tests/test_liveness_preprocess.py`:

```python
import numpy as np
import pytest


def test_preprocess_output_shape():
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from src.fas.liveness_model import TorchLivenessModel

    model = TorchLivenessModel(model_path=None, input_size=112)
    face = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    tensor = model._preprocess(face)

    assert tensor.shape == (1, 3, 112, 112), f"expected (1,3,112,112) got {tuple(tensor.shape)}"


def test_preprocess_imagenet_normalization_white():
    """White BGR image → all channels 1.0 after /255 → normalized with ImageNet stats."""
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from src.fas.liveness_model import TorchLivenessModel

    model = TorchLivenessModel(model_path=None, input_size=112)
    face = np.full((100, 100, 3), 255, dtype=np.uint8)  # BGR white
    tensor = model._preprocess(face)

    # BGR→RGB: all channels still 255. After /255 = 1.0.
    # Channel 0 (R): (1.0 - 0.485) / 0.229 ≈ 2.249
    # Channel 1 (G): (1.0 - 0.456) / 0.224 ≈ 2.429
    # Channel 2 (B): (1.0 - 0.406) / 0.225 ≈ 2.640
    expected = [(1.0 - 0.485) / 0.229, (1.0 - 0.456) / 0.224, (1.0 - 0.406) / 0.225]
    for ch, exp in enumerate(expected):
        actual = float(tensor[0, ch, 0, 0].item())
        assert abs(actual - exp) < 0.01, f"channel {ch}: expected {exp:.3f} got {actual:.3f}"


def test_preprocess_imagenet_normalization_black():
    """Black image → after /255 = 0.0 → normalized: (0 - mean) / std (negative values)."""
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from src.fas.liveness_model import TorchLivenessModel

    model = TorchLivenessModel(model_path=None, input_size=112)
    face = np.zeros((100, 100, 3), dtype=np.uint8)
    tensor = model._preprocess(face)

    # Channel 0 (R): (0.0 - 0.485) / 0.229 ≈ -2.118
    expected_r = (0.0 - 0.485) / 0.229
    actual_r = float(tensor[0, 0, 0, 0].item())
    assert abs(actual_r - expected_r) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/charlie/code/Face_Anti_Spoofing_Biometric
pytest tests/test_liveness_preprocess.py -v
```

Expected: 3 FAILED (shape will be (1,3,80,80), values won't be ImageNet-normalized).

- [ ] **Step 3: Update `src/fas/liveness_model.py`**

Change line 11 (`input_size`):
```python
# Before
input_size: int = 80
# After
input_size: int = 112
```

Replace the `_preprocess` method (lines 69–83) with:
```python
def _preprocess(self, face_crop_bgr: np.ndarray):
    torch = self._torch
    assert torch is not None

    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError('OpenCV is required for liveness preprocessing.') from exc

    resized = cv2.resize(face_crop_bgr, (self.input_size, self.input_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    chw = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0)
    return tensor
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_liveness_preprocess.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Run existing tests to verify no regression**

```bash
pytest tests/ -v
```

Expected: all previously passing tests still PASS (FakeLivenessModel in test_service_pipeline is unaffected).

- [ ] **Step 6: Commit**

```bash
git add src/fas/liveness_model.py tests/test_liveness_preprocess.py
git commit -m "feat: update liveness preprocessing to 112x112 and ImageNet normalization"
```

---

## Task 2: Create MobileNetV2 Training Notebook

**Files:**
- Create: `notebooks/kaggle_full_05_train_mobilenetv2.ipynb`

This notebook runs on Kaggle with GPU. It reads the existing prepared manifests (`train.csv`, `val.csv`) from the `doraemongwa/celeba-spoof-prepared-full` dataset input, trains MobileNetV2, and exports a TorchScript checkpoint. The executor should create this file programmatically using the script below.

- [ ] **Step 1: Create notebook generator script and run it**

```bash
pip install nbformat -q 2>/dev/null || true
python3 - <<'PYEOF'
import json, pathlib
pathlib.Path('notebooks').mkdir(exist_ok=True)

def cell(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}

def md(src):
    return {"cell_type":"markdown","metadata":{},"source":src}

cells = [
md("# MobileNetV2 Liveness — Training Notebook\n\nTrains MobileNetV2 (pretrained ImageNet) on CelebA-Spoof crops with full augmentation.\nExports TorchScript checkpoint compatible with the existing inference service."),

cell("# !pip install -q torch torchvision opencv-python pandas numpy tqdm"),

cell("""\
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as tv_models
import torchvision.transforms as T
from tqdm import tqdm"""),

cell("""\
@dataclass
class TrainConfig:
    train_manifest: str
    val_manifest: str
    output_dir: str
    image_size: int = 112
    batch_size: int = 64
    epochs: int = 15
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    seed: int = 42"""),

cell("""\
# ----- Dataset ----------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def make_train_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def make_val_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class ManifestDataset(Dataset):
    LEGACY_ROOT = Path('/kaggle/working/celeba_spoof_prepared_full')

    def __init__(self, manifest_path: str, transform: T.Compose) -> None:
        self.df = pd.read_csv(manifest_path)
        self.transform = transform
        self.prepared_root = Path(manifest_path).parent.parent

    def __len__(self) -> int:
        return len(self.df)

    def _resolve(self, raw: str) -> Path:
        p = Path(raw)
        if p.exists():
            return p
        legacy = str(self.LEGACY_ROOT) + '/'
        if raw.startswith(legacy):
            candidate = self.prepared_root / raw[len(legacy):]
            if candidate.exists():
                return candidate
        marker = 'crops_80x80/'
        if marker in raw:
            candidate = self.prepared_root / 'crops_80x80' / raw.split(marker, 1)[1]
            if candidate.exists():
                return candidate
        return p

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_bgr = cv2.imread(str(self._resolve(row.image_path)))
        if img_bgr is None:
            raise RuntimeError(f'Cannot load: {row.image_path}')
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb)
        return tensor, int(row.label)"""),

cell("""\
# ----- Model ------------------------------------------------------------

def build_mobilenetv2_fas(pretrained: bool = True) -> nn.Module:
    weights = 'IMAGENET1K_V1' if pretrained else None
    backbone = tv_models.mobilenet_v2(weights=weights)
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, 2),
    )
    return backbone"""),

cell("""\
# ----- Metrics ----------------------------------------------------------

def compute_acer(scores: list[float], labels: list[int], threshold: float = 0.5) -> float:
    \"\"\"Average Classification Error Rate = (APCER + BPCER) / 2.\"\"\"
    tp = fp = tn = fn = 0
    for s, l in zip(scores, labels):
        pred = 1 if s >= threshold else 0
        if l == 1 and pred == 1: tp += 1
        elif l == 0 and pred == 1: fp += 1
        elif l == 0 and pred == 0: tn += 1
        else: fn += 1
    apcer = fp / max(fp + tn, 1)  # spoof classified as live
    bpcer = fn / max(fn + tp, 1)  # live classified as spoof
    return (apcer + bpcer) / 2


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = total_correct = total_count = 0
    all_scores: list[float] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)[:, 1]
            total_loss += float(loss.item()) * labels.size(0)
            total_correct += int((logits.argmax(1) == labels).sum().item())
            total_count += int(labels.size(0))
            all_scores.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return {
        'loss': total_loss / max(total_count, 1),
        'acc': total_correct / max(total_count, 1),
        'acer': compute_acer(all_scores, all_labels, threshold=0.5),
    }"""),

cell("""\
# ----- Paths (Kaggle) ---------------------------------------------------

MANIFEST_ROOT = Path(
    '/kaggle/input/datasets/doraemongwa/celeba-spoof-prepared-full'
    '/celeba_spoof_prepared_full/manifests'
)
OUT_ROOT = Path('/kaggle/working/mobilenetv2_fas_training')
OUT_ROOT.mkdir(parents=True, exist_ok=True)

cfg = TrainConfig(
    train_manifest=str(MANIFEST_ROOT / 'train.csv'),
    val_manifest=str(MANIFEST_ROOT / 'val.csv'),
    output_dir=str(OUT_ROOT),
)
print(cfg)
print('train manifest exists:', Path(cfg.train_manifest).exists())
print('val manifest   exists:', Path(cfg.val_manifest).exists())"""),

cell("""\
# ----- Data loaders -----------------------------------------------------

torch.manual_seed(cfg.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

train_ds = ManifestDataset(cfg.train_manifest, make_train_transform(cfg.image_size))
val_ds   = ManifestDataset(cfg.val_manifest,   make_val_transform(cfg.image_size))

# Weighted sampler for class balance
labels_list = train_ds.df['label'].tolist()
n_live  = sum(1 for l in labels_list if l == 1)
n_spoof = sum(1 for l in labels_list if l == 0)
print(f'train: {n_live} live, {n_spoof} spoof')
sample_weights = [1.0 / n_live if l == 1 else 1.0 / n_spoof for l in labels_list]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights))

train_loader = DataLoader(
    train_ds,
    batch_size=cfg.batch_size,
    sampler=sampler,
    num_workers=cfg.num_workers,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_ds,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=torch.cuda.is_available(),
)"""),

cell("""\
# ----- Model + optimizer ------------------------------------------------

model = build_mobilenetv2_fas(pretrained=True).to(device)

optimizer = torch.optim.AdamW([
    {'params': model.features.parameters(),    'lr': cfg.lr_backbone},
    {'params': model.classifier.parameters(),  'lr': cfg.lr_head},
], weight_decay=cfg.weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.epochs, eta_min=1e-6
)
criterion = nn.CrossEntropyLoss()
best_ckpt = Path(cfg.output_dir) / 'best_mobilenetv2.pt'"""),

cell("""\
# ----- Training loop ----------------------------------------------------

best_acer = 1.0
patience_left = 5
history: list[dict] = []

for epoch in range(1, cfg.epochs + 1):
    model.train()
    train_loss = train_correct = train_count = 0

    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch}/{cfg.epochs}'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += float(loss.item()) * labels.size(0)
        train_correct += int((logits.argmax(1) == labels).sum().item())
        train_count += int(labels.size(0))

    scheduler.step()

    val_metrics = evaluate(model, val_loader, device)
    row = {
        'epoch': epoch,
        'train_loss': train_loss / max(train_count, 1),
        'train_acc':  train_correct / max(train_count, 1),
        'val_loss':   val_metrics['loss'],
        'val_acc':    val_metrics['acc'],
        'val_acer':   val_metrics['acer'],
    }
    history.append(row)
    print(row)

    if val_metrics['acer'] < best_acer:
        best_acer = val_metrics['acer']
        patience_left = 5
        torch.save({'state_dict': model.state_dict(), 'image_size': cfg.image_size}, best_ckpt)
        print(f'  -> new best ACER {best_acer:.4f}, checkpoint saved')
    else:
        patience_left -= 1
        if patience_left == 0:
            print('Early stopping.')
            break

(Path(cfg.output_dir) / 'history.json').write_text(json.dumps(history, indent=2))
print('Best val ACER:', best_acer)"""),

cell("""\
# ----- Export TorchScript -----------------------------------------------

payload = torch.load(best_ckpt, map_location='cpu')
model_cpu = build_mobilenetv2_fas(pretrained=False)
model_cpu.load_state_dict(payload['state_dict'])
model_cpu.eval()

scripted_path = Path(cfg.output_dir) / 'mobilenetv2_fas_scripted.pt'
scripted = torch.jit.script(model_cpu)
scripted.save(str(scripted_path))

summary = {
    'best_acer': float(best_acer),
    'image_size': cfg.image_size,
    'best_checkpoint': str(best_ckpt),
    'scripted_checkpoint': str(scripted_path),
}
(Path(cfg.output_dir) / 'run_summary.json').write_text(json.dumps(summary, indent=2))
print(summary)"""),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

out = pathlib.Path('notebooks/kaggle_full_05_train_mobilenetv2.ipynb')
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print('Written:', out)
PYEOF
```

Expected output: `Written: notebooks/kaggle_full_05_train_mobilenetv2.ipynb`

- [ ] **Step 2: Verify notebook was created correctly**

```bash
python3 -c "
import json
nb = json.load(open('notebooks/kaggle_full_05_train_mobilenetv2.ipynb'))
code_cells = [c for c in nb['cells'] if c['cell_type']=='code']
print('code cells:', len(code_cells))
print('first 80 chars of last cell:', ''.join(code_cells[-1]['source'])[:80])
"
```

Expected: `code cells: 11` and last cell starts with `# ----- Export TorchScript`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/kaggle_full_05_train_mobilenetv2.ipynb
git commit -m "feat: add MobileNetV2 training notebook with augmentation and ACER metric"
```

> **Kaggle execution note:** Upload the notebook to Kaggle, attach dataset `doraemongwa/celeba-spoof-prepared-full`, enable GPU accelerator, and run all cells. Download `mobilenetv2_fas_scripted.pt` from Kaggle output. Set `LIVENESS_MODEL_PATH` in the backend env to point to the downloaded file.

---

## Task 3: Update Training Config

**Files:**
- Modify: `configs/training.yaml`

- [ ] **Step 1: Update `configs/training.yaml`**

Replace:
```yaml
  image_size: 80
```
With:
```yaml
  image_size: 112
```

The full updated file should look like:
```yaml
experiment_name: mobilenetv2_celeba_spoof
seed: 42
dataset:
  name: celeba_spoof
  raw_root: data/raw/celeba_spoof
  processed_root: data/processed/celeba_spoof_112x112
  image_size: 112
training:
  batch_size: 64
  epochs: 15
  learning_rate: 0.001
  device: auto
outputs:
  checkpoint_dir: models/artifacts/checkpoints
  report_dir: reports/generated/training
```

- [ ] **Step 2: Commit**

```bash
git add configs/training.yaml
git commit -m "config: update image_size to 112 for MobileNetV2 training"
```

---

## Task 4: Update Frontend — Continuous Polling + Score Smoothing

**Files:**
- Modify: `apps/web/src/App.tsx`

Replace the existing `App.tsx` with the continuous polling version. The key changes are:
1. Add `useEffect` to the imports
2. Add `scoreWindowRef` (deque of last 5 scores) and `smoothedResult` state
3. Add `captureAndSendAsync` — fire-and-forget version that does NOT set `busy`
4. Add `useEffect` interval that starts when `cameraReady` becomes true
5. Add `onScoreReceived` for smoothing
6. Keep the manual "Capture and Scan" button for one-shot use

- [ ] **Step 1: Replace `apps/web/src/App.tsx`**

```tsx
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
      case 'live':    return 'good'
      case 'spoof':   return 'bad'
      case 'uncertain': return 'warn'
      default:        return 'neutral'
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

  // Fire-and-forget: used by continuous interval — does not block UI
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

  // Blocking: used by the manual "Capture and Scan" button
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

  // Start continuous polling when camera is ready
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
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd apps/web && npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/App.tsx
git commit -m "feat: add continuous polling loop and 5-frame score smoothing to frontend"
```

---

## Task 5 [Phase 2]: WebSocket Streaming Backend + Frontend Mode

Implement after MVP is validated with the MobileNetV2 model. This task adds a persistent WebSocket connection at `/ws/liveness` for lower-latency streaming, and a toggle in the frontend to switch between HTTP polling (default) and WebSocket mode.

**Files:**
- Modify: `services/api/app.py`
- Modify: `requirements.txt`
- Modify: `apps/web/src/App.tsx`

- [ ] **Step 1: Add `websockets` to `requirements.txt`**

Append to `requirements.txt`:
```
websockets>=12.0
```

- [ ] **Step 2: Add WebSocket endpoint to `services/api/app.py`**

Add these imports at the top of `services/api/app.py` (after existing imports):
```python
import base64
import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
```

Add this endpoint after the existing `/v1/liveness/infer` route:
```python
@app.websocket("/ws/liveness")
async def ws_liveness(websocket: WebSocket) -> None:
    await websocket.accept()
    service = LivenessService()
    try:
        while True:
            data = await websocket.receive_bytes()
            arr = np.frombuffer(data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                await websocket.send_json({"error": "could not decode frame"})
                continue
            _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64 = base64.b64encode(buf.tobytes()).decode()
            result = service.infer(LivenessInferRequest(image_base64=b64))
            await websocket.send_json(result.model_dump())
    except WebSocketDisconnect:
        pass
```

- [ ] **Step 3: Verify backend starts without errors**

```bash
cd /home/charlie/code/Face_Anti_Spoofing_Biometric
pip install websockets -q
uvicorn services.api.app:app --port 8001 &
sleep 2
curl -s http://localhost:8001/health | head -c 200
kill %1
```

Expected: health check responds (or 404 if no `/health` route — what matters is the server starts without import errors).

- [ ] **Step 4: Add WebSocket mode to `apps/web/src/App.tsx`**

Add `useRef` for the WebSocket instance and a mode toggle button. Add these hooks inside the `App` function (after existing refs):

```tsx
const wsRef = useRef<WebSocket | null>(null)
const [wsMode, setWsMode] = useState(false)
```

Add `startWebSocket` and `stopWebSocket` functions:

```tsx
function startWebSocket() {
  if (wsRef.current) return
  const ws = new WebSocket(`${API_BASE.replace(/^http/, 'ws')}/ws/liveness`)
  ws.binaryType = 'arraybuffer'
  ws.onmessage = (e) => {
    const payload = JSON.parse(e.data as string) as InferResponse
    setResult(payload)
    if (payload.face_detected) onScoreReceived(payload.liveness_score)
  }
  wsRef.current = ws
  setWsMode(true)
}

function stopWebSocket() {
  wsRef.current?.close()
  wsRef.current = null
  setWsMode(false)
}
```

Replace the `useEffect` for continuous polling with a mode-aware version:

```tsx
useEffect(() => {
  if (!cameraReady) return
  if (wsMode) {
    // WebSocket mode: draw video to canvas first, then send as binary JPEG
    const id = setInterval(() => {
      const video = videoRef.current
      const canvas = canvasRef.current
      if (!video || !canvas || wsRef.current?.readyState !== WebSocket.OPEN) return
      canvas.width = video.videoWidth || 640
      canvas.height = video.videoHeight || 480
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      canvas.toBlob(
        blob => blob?.arrayBuffer().then(buf => wsRef.current?.send(buf)),
        'image/jpeg',
        0.7,
      )
    }, 100)
    return () => clearInterval(id)
  } else {
    // HTTP polling mode: fire-and-forget POST every 300ms
    const id = setInterval(captureAndSendAsync, POLL_INTERVAL_MS)
    return () => clearInterval(id)
  }
}, [cameraReady, wsMode])
```

Add a toggle button in the JSX actions section (after existing buttons):

```tsx
<button onClick={wsMode ? stopWebSocket : startWebSocket} disabled={!cameraReady}>
  {wsMode ? 'Switch to HTTP' : 'Switch to WebSocket'}
</button>
```

- [ ] **Step 5: Verify TypeScript compiles**

```bash
cd apps/web && npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add services/api/app.py requirements.txt apps/web/src/App.tsx
git commit -m "feat: add WebSocket streaming endpoint and frontend mode toggle"
```

---

## Post-MVP Checklist

After running the Kaggle training notebook and downloading `mobilenetv2_fas_scripted.pt`:

- [ ] Copy `mobilenetv2_fas_scripted.pt` to `models/artifacts/` (or set `LIVENESS_MODEL_PATH` env var)
- [ ] Start backend: `bash scripts/run_backend_demo.sh`
- [ ] Start frontend: `bash scripts/run_frontend_demo.sh`
- [ ] Test: open browser, start camera, verify continuous score updates every ~300ms
- [ ] Test spoof: hold a phone screen with someone else's photo — score should drop to spoof range
- [ ] Verify ACER from Kaggle eval notebook is < 10% (SmallFASNet baseline was typically 15–25%)
