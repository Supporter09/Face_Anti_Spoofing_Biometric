from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class TrainConfig:
    train_manifest: str
    val_manifest: str
    output_dir: str
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-3
    image_size: int = 80
    num_workers: int = 2
    seed: int = 42


class SmallFASNet:
    def __init__(self, image_size: int = 80) -> None:
        import torch.nn as nn

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )


class ManifestDataset:
    def __init__(self, manifest_path: str, image_size: int = 80) -> None:
        self.df = _load_manifest(manifest_path)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        try:
            import cv2  # type: ignore
            import numpy as np
            import torch
        except ImportError as exc:
            raise RuntimeError('Training requires torch and opencv.') from exc

        row = self.df.iloc[index]
        image = cv2.imread(str(row.image_path))
        if image is None:
            raise RuntimeError(f'Could not load image: {row.image_path}')

        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        tensor = torch.from_numpy(image)
        label = torch.tensor(int(row.label), dtype=torch.long)
        return tensor, label


def _load_manifest(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == '.json':
        return pd.read_json(p)
    return pd.read_csv(p)


def create_data_loaders(config: TrainConfig):
    import torch
    from torch.utils.data import DataLoader

    train_dataset = ManifestDataset(config.train_manifest, image_size=config.image_size)
    val_dataset = ManifestDataset(config.val_manifest, image_size=config.image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def train_model(config: TrainConfig) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    torch.manual_seed(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmallFASNet(image_size=config.image_size).model.to(device)
    train_loader, val_loader = create_data_loaders(config)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = output_dir / 'best_model.pt'

    best_val_acc = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item()) * labels.size(0)
            predictions = logits.argmax(dim=1)
            train_correct += int((predictions == labels).sum().item())
            train_total += int(labels.size(0))

        val_metrics = evaluate_model(model, val_loader, device)

        epoch_metrics = {
            'epoch': float(epoch),
            'train_loss': train_loss / max(train_total, 1),
            'train_acc': train_correct / max(train_total, 1),
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['acc'],
        }
        history.append(epoch_metrics)

        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            torch.save({'state_dict': model.state_dict(), 'image_size': config.image_size}, best_ckpt)

    scripted_path = output_dir / 'best_model_scripted.pt'
    model.eval()
    scripted = torch.jit.script(model.cpu())
    scripted.save(str(scripted_path))

    return {
        'best_checkpoint': str(best_ckpt),
        'best_scripted_checkpoint': str(scripted_path),
        'best_val_acc': float(best_val_acc),
        'history': history,
    }


def evaluate_model(model, data_loader, device) -> dict[str, float]:
    import torch
    import torch.nn as nn

    criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += float(loss.item()) * labels.size(0)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_count += int(labels.size(0))

    return {
        'loss': total_loss / max(total_count, 1),
        'acc': total_correct / max(total_count, 1),
    }


def run_checkpoint_inference(checkpoint_path: str, manifest_path: str, image_size: int = 80) -> pd.DataFrame:
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmallFASNet(image_size=image_size).model
    payload = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(payload['state_dict'])
    model.to(device)
    model.eval()

    dataset = ManifestDataset(manifest_path, image_size=image_size)
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for index in range(len(dataset)):
            image, label = dataset[index]
            logits = model(image.unsqueeze(0).to(device))
            prob_live = torch.softmax(logits, dim=1)[0, 1].item()
            rows.append(
                {
                    'image_path': dataset.df.iloc[index].image_path,
                    'label': int(label.item()),
                    'live_score': float(prob_live),
                }
            )

    return pd.DataFrame(rows)
