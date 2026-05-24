import numpy as np
import pytest

from fas.preprocess import pad_to_square_then_resize


def test_pad_square_already_square_resizes():
    img = np.full((40, 40, 3), 128, dtype=np.uint8)
    out = pad_to_square_then_resize(img, size=20)
    assert out.shape == (20, 20, 3)
    assert out.dtype == np.uint8
    # Pure resize, no padding, all values stay ~128
    assert int(out.mean()) == 128


def test_pad_square_wide_image_pads_top_bottom():
    # 100 wide x 40 tall -> pad top/bottom to make 100x100, then resize to 50
    img = np.full((40, 100, 3), 200, dtype=np.uint8)
    out = pad_to_square_then_resize(img, size=50, pad_value=0)
    assert out.shape == (50, 50, 3)
    # Top and bottom strips should be near 0 (pad), middle near 200
    assert out[0, 25, 0] < 50    # top padded
    assert out[49, 25, 0] < 50   # bottom padded
    assert out[25, 25, 0] > 150  # middle is image content


def test_pad_square_tall_image_pads_left_right():
    img = np.full((100, 40, 3), 200, dtype=np.uint8)
    out = pad_to_square_then_resize(img, size=50, pad_value=0)
    assert out.shape == (50, 50, 3)
    assert out[25, 0, 0] < 50
    assert out[25, 49, 0] < 50
    assert out[25, 25, 0] > 150


def test_pad_square_default_pad_value_is_mean():
    img = np.zeros((30, 60, 3), dtype=np.uint8)
    img[:, :, 0] = 100   # blue channel
    img[:, :, 1] = 150   # green
    img[:, :, 2] = 200   # red
    out = pad_to_square_then_resize(img, size=64)
    # Default pad_value=None → use per-channel mean
    # Top/bottom padding pixels should be close to (100,150,200)
    assert abs(int(out[0, 32, 0]) - 100) < 20
    assert abs(int(out[0, 32, 1]) - 150) < 20
    assert abs(int(out[0, 32, 2]) - 200) < 20


def test_pad_square_empty_image_raises():
    img = np.zeros((0, 0, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        pad_to_square_then_resize(img, size=20)
