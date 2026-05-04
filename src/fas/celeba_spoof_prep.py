from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from fas.preprocess import clamp_bbox_to_image, resize_bgr_image


@dataclass
class PrepConfig:
    dataset_root: str
    image_rel_col: str
    label_col: str
    split_col: str | None = None
    bbox_cols_xywh: tuple[str, str, str, str] | None = None
    live_values: tuple[Any, ...] = (1, '1', 'live', 'real', True)
    spoof_values: tuple[Any, ...] = (0, '0', 'spoof', 'fake', False)
    image_size: int = 80
    bbox_margin_ratio: float = 0.15


def load_annotation_table(annotation_path: str) -> pd.DataFrame:
    path = Path(annotation_path)
    suffix = path.suffix.lower()

    if suffix == '.csv':
        return pd.read_csv(path)
    if suffix in {'.parquet', '.pq'}:
        return pd.read_parquet(path)
    if suffix == '.json':
        loaded = pd.read_json(path)
        if isinstance(loaded, pd.DataFrame):
            return loaded
        return pd.DataFrame(loaded)
    if suffix in {'.jsonl', '.ndjson'}:
        return pd.read_json(path, lines=True)

    raise ValueError(f'Unsupported annotation format: {suffix}')


def normalize_label(value: Any, live_values: tuple[Any, ...], spoof_values: tuple[Any, ...]) -> int:
    if value in live_values:
        return 1
    if value in spoof_values:
        return 0

    text = str(value).strip().lower()
    if text in {str(item).strip().lower() for item in live_values}:
        return 1
    if text in {str(item).strip().lower() for item in spoof_values}:
        return 0

    raise ValueError(f'Unknown label value: {value}')


def build_manifest_dataframe(
    table: pd.DataFrame,
    config: PrepConfig,
    split_value: str | int | None = None,
) -> pd.DataFrame:
    required = {config.image_rel_col, config.label_col}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise KeyError(f'Missing required columns: {missing}')

    view = table.copy()

    if config.split_col and split_value is not None:
        view = view[view[config.split_col] == split_value]

    view['image_path'] = view[config.image_rel_col].astype(str)
    view['label'] = [
        normalize_label(value, config.live_values, config.spoof_values)
        for value in view[config.label_col]
    ]

    if config.bbox_cols_xywh:
        bx, by, bw, bh = config.bbox_cols_xywh
        for column in (bx, by, bw, bh):
            if column not in view.columns:
                raise KeyError(f'Bbox column missing: {column}')
        view['bbox_x'] = view[bx].astype(float)
        view['bbox_y'] = view[by].astype(float)
        view['bbox_w'] = view[bw].astype(float)
        view['bbox_h'] = view[bh].astype(float)

    columns = ['image_path', 'label']
    if config.bbox_cols_xywh:
        columns += ['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']

    return view[columns].reset_index(drop=True)


def _expand_bbox_xywh(
    x: float,
    y: float,
    w: float,
    h: float,
    image_height: int,
    image_width: int,
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))

    margin_x = int((x2 - x1) * margin_ratio)
    margin_y = int((y2 - y1) * margin_ratio)

    return clamp_bbox_to_image(
        (x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y),
        image_height=image_height,
        image_width=image_width,
    )


def crop_faces_from_manifest(
    manifest_df: pd.DataFrame,
    config: PrepConfig,
    output_dir: str,
) -> pd.DataFrame:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError('OpenCV is required for cropping faces.') from exc

    root = Path(config.dataset_root)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    has_bbox = {'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'}.issubset(manifest_df.columns)

    iterator = tqdm(manifest_df.itertuples(index=False), total=len(manifest_df), desc='Cropping faces')
    for index, row in enumerate(iterator):
        src_path = root / row.image_path
        image = cv2.imread(str(src_path))
        if image is None:
            continue

        if has_bbox:
            x1, y1, x2, y2 = _expand_bbox_xywh(
                x=float(row.bbox_x),
                y=float(row.bbox_y),
                w=float(row.bbox_w),
                h=float(row.bbox_h),
                image_height=image.shape[0],
                image_width=image.shape[1],
                margin_ratio=config.bbox_margin_ratio,
            )
            face = image[y1:y2, x1:x2]
        else:
            x1, y1, x2, y2 = 0, 0, image.shape[1], image.shape[0]
            face = image

        resized = resize_bgr_image(face, config.image_size)
        dst_name = f'{index:08d}_{Path(row.image_path).stem}.jpg'
        dst_path = out_root / dst_name
        cv2.imwrite(str(dst_path), resized)

        rows.append(
            {
                'image_path': str(dst_path),
                'label': int(row.label),
                'bbox_xyxy': [int(x1), int(y1), int(x2), int(y2)],
            }
        )

    return pd.DataFrame(rows)


def create_split_manifests(
    cropped_df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(cropped_df))
    rng.shuffle(indices)

    shuffled = cropped_df.iloc[indices].reset_index(drop=True)

    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)

    return {
        'train': shuffled.iloc[:train_end].reset_index(drop=True),
        'val': shuffled.iloc[train_end:val_end].reset_index(drop=True),
        'test': shuffled.iloc[val_end:].reset_index(drop=True),
    }


def save_manifest(df: pd.DataFrame, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == '.json':
        df.to_json(path, orient='records', indent=2)
        return
    df.to_csv(path, index=False)
