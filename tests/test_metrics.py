import numpy as np
import pytest

from backend.metrics import compute_attention_multiplier, compute_attention_share

# log-density of a uniform density map: log(1) everywhere, so density == 1
UNIFORM = np.zeros((10, 10))


class TestAttentionMultiplier:
    def test_uniform_density_scores_exactly_one_everywhere(self):
        assert compute_attention_multiplier(UNIFORM, (0, 0, 10, 10)) == pytest.approx(1.0)
        assert compute_attention_multiplier(UNIFORM, (2, 2, 5, 5)) == pytest.approx(1.0)

    def test_hot_region_scores_above_one(self):
        density = np.zeros((10, 10))
        density[0:5, 0:5] = np.log(3.0)  # 3x the background density
        assert compute_attention_multiplier(density, (0, 0, 5, 5)) > 1.0

    def test_cold_region_scores_below_one(self):
        density = np.zeros((10, 10))
        density[0:5, 0:5] = np.log(3.0)
        # a box outside the hot region should score below the image average
        assert compute_attention_multiplier(density, (5, 5, 10, 10)) < 1.0

    def test_matches_hand_computed_value(self):
        # a 4x4 grid, top-left quadrant at density 4, rest at density 1
        density = np.zeros((4, 4))
        density[0:2, 0:2] = np.log(4.0)
        # image mean = (4*4 + 1*12) / 16 = 28/16 = 1.75
        # box mean = 4.0
        # multiplier = 4.0 / 1.75
        expected = 4.0 / 1.75
        assert compute_attention_multiplier(density, (0, 0, 2, 2)) == pytest.approx(expected)

    def test_whole_image_box_is_always_one(self):
        rng = np.random.default_rng(0)
        density = rng.normal(size=(20, 20))
        assert compute_attention_multiplier(density, (0, 0, 20, 20)) == pytest.approx(1.0)

    def test_empty_box_raises(self):
        with pytest.raises(ValueError):
            compute_attention_multiplier(UNIFORM, (5, 5, 5, 8))


class TestAttentionShare:
    def test_uniform_density_share_equals_box_area_fraction(self):
        # box covers 1/4 of a uniform image, so it should hold 1/4 of the total
        assert compute_attention_share(UNIFORM, (0, 0, 5, 10)) == pytest.approx(0.5)
        assert compute_attention_share(UNIFORM, (0, 0, 5, 5)) == pytest.approx(0.25)

    def test_whole_image_box_captures_everything(self):
        rng = np.random.default_rng(1)
        density = rng.normal(size=(15, 15))
        assert compute_attention_share(density, (0, 0, 15, 15)) == pytest.approx(1.0)

    def test_share_and_multiplier_agree_on_area_fraction(self):
        # share / multiplier == box area / image area, since the pixel counts
        # cancel out of the two ratios -- true for any density map
        rng = np.random.default_rng(2)
        density = rng.normal(size=(30, 40))
        box = (5, 5, 15, 20)
        x0, y0, x1, y1 = box
        area_fraction = ((x1 - x0) * (y1 - y0)) / (30 * 40)

        share = compute_attention_share(density, box)
        multiplier = compute_attention_multiplier(density, box)
        assert share / multiplier == pytest.approx(area_fraction)

    def test_result_is_between_zero_and_one(self):
        rng = np.random.default_rng(3)
        density = rng.normal(size=(25, 25))
        share = compute_attention_share(density, (3, 3, 10, 12))
        assert 0.0 <= share <= 1.0
