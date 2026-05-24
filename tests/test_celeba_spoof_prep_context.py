import numpy as np
import pandas as pd

from fas.celeba_spoof_prep import PrepConfig, crop_faces_from_manifest


def test_crop_uses_context_margin_and_pad_to_square(tmp_path, monkeypatch):
    # Synthetic 200x300 image (height x width), bbox in centre 50x50
    import cv2

    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img[75:125, 125:175] = 255   # white square = face region

    src = tmp_path / 'dataset'
    src.mkdir()
    img_path = src / 'sample.jpg'
    cv2.imwrite(str(img_path), img)

    manifest = pd.DataFrame([{
        'image_path': 'sample.jpg',
        'label': 1,
        'bbox_x': 125.0,
        'bbox_y': 75.0,
        'bbox_w': 50.0,
        'bbox_h': 50.0,
    }])

    config = PrepConfig(
        dataset_root=str(src),
        image_rel_col='image_path',
        label_col='label',
        bbox_cols_xywh=('bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'),
        image_size=224,
        context_margin_ratio=0.8,
    )

    out_dir = tmp_path / 'out'
    result_df = crop_faces_from_manifest(manifest, config, str(out_dir))

    assert len(result_df) == 1
    saved_path = result_df.iloc[0]['image_path']
    cropped = cv2.imread(saved_path)
    assert cropped.shape == (224, 224, 3)
