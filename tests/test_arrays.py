import numpy as np
import pytest
import torch

from backend.arrays import as_2d_array, clip_box


class TestAs2DArray:
    def test_plain_2d_array_passes_through(self):
        array = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = as_2d_array(array)
        np.testing.assert_array_equal(result, array)

    def test_squeezes_leading_singleton_dims(self):
        array = np.zeros((1, 1, 5, 7))
        assert as_2d_array(array).shape == (5, 7)

    def test_accepts_a_torch_tensor(self):
        tensor = torch.zeros((1, 1, 4, 6))
        result = as_2d_array(tensor)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 6)

    def test_rejects_a_shape_that_cant_squeeze_to_2d(self):
        with pytest.raises(ValueError):
            as_2d_array(np.zeros((2, 3, 4)))


class TestClipBox:
    def test_box_already_inside_bounds_is_unchanged(self):
        assert clip_box((10, 20, 30, 40), height=100, width=100) == (10, 20, 30, 40)

    def test_reversed_coordinates_are_sorted(self):
        assert clip_box((30, 40, 10, 20), height=100, width=100) == (10, 20, 30, 40)

    def test_box_extending_past_bounds_is_clamped(self):
        assert clip_box((-10, -10, 50, 50), height=40, width=40) == (0, 0, 40, 40)

    def test_empty_box_after_clipping_raises(self):
        with pytest.raises(ValueError):
            clip_box((-10, -10, -1, -1), height=40, width=40)

    def test_zero_area_box_raises(self):
        with pytest.raises(ValueError):
            clip_box((5, 5, 5, 20), height=40, width=40)

    def test_rescales_from_a_different_image_size(self):
        # a box drawn on a 200x200 display of a 100x100 density map halves
        box = clip_box((20, 40, 60, 80), height=100, width=100, image_shape=(200, 200))
        assert box == (10, 20, 30, 40)

    def test_rescale_then_clip_to_density_bounds(self):
        box = clip_box((0, 0, 400, 400), height=100, width=100, image_shape=(200, 200))
        assert box == (0, 0, 100, 100)
