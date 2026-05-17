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
    """White BGR image -> all channels 1.0 after /255 -> normalized with ImageNet stats."""
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from src.fas.liveness_model import TorchLivenessModel

    model = TorchLivenessModel(model_path=None, input_size=112)
    face = np.full((100, 100, 3), 255, dtype=np.uint8)  # BGR white
    tensor = model._preprocess(face)

    # BGR->RGB: all channels still 255. After /255 = 1.0.
    # Channel 0 (R): (1.0 - 0.485) / 0.229 ~= 2.249
    # Channel 1 (G): (1.0 - 0.456) / 0.224 ~= 2.429
    # Channel 2 (B): (1.0 - 0.406) / 0.225 ~= 2.640
    expected = [(1.0 - 0.485) / 0.229, (1.0 - 0.456) / 0.224, (1.0 - 0.406) / 0.225]
    for ch, exp in enumerate(expected):
        actual = float(tensor[0, ch, 0, 0].item())
        assert abs(actual - exp) < 0.01, f"channel {ch}: expected {exp:.3f} got {actual:.3f}"


def test_preprocess_imagenet_normalization_black():
    """Black image -> after /255 = 0.0 -> normalized: (0 - mean) / std (negative values)."""
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from src.fas.liveness_model import TorchLivenessModel

    model = TorchLivenessModel(model_path=None, input_size=112)
    face = np.zeros((100, 100, 3), dtype=np.uint8)
    tensor = model._preprocess(face)

    # Channel 0 (R): (0.0 - 0.485) / 0.229 ~= -2.118
    expected_r = (0.0 - 0.485) / 0.229
    actual_r = float(tensor[0, 0, 0, 0].item())
    assert abs(actual_r - expected_r) < 0.01
