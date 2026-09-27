import numpy as np

from backend.heatmap import make_glow_overlay, make_overlay

IMAGE = np.full((20, 30, 3), 150, dtype=np.uint8)


def hot_spot_density(shape=(20, 30), spot=(2, 2, 6, 6), background=1.0, peak=50.0):
    """A log-density map that's flat except for one bright rectangle."""
    density = np.full(shape, np.log(background))
    x0, y0, x1, y1 = spot
    density[y0:y1, x0:x1] = np.log(peak)
    return density


class TestMakeOverlay:
    def test_output_shape_and_dtype_match_the_image(self):
        overlay = make_overlay(hot_spot_density(), IMAGE)
        assert overlay.shape == IMAGE.shape
        assert overlay.dtype == np.uint8

    def test_output_values_are_valid_pixel_range(self):
        overlay = make_overlay(hot_spot_density(), IMAGE)
        assert overlay.min() >= 0
        assert overlay.max() <= 255

    def test_resizes_a_density_map_of_a_different_resolution(self):
        small_density = hot_spot_density(shape=(5, 6), spot=(1, 1, 3, 3))
        overlay = make_overlay(small_density, IMAGE)
        assert overlay.shape == IMAGE.shape

    def test_flat_density_does_not_crash_on_zero_division(self):
        flat = np.zeros((20, 30))
        overlay = make_overlay(flat, IMAGE)
        assert overlay.shape == IMAGE.shape

    def test_alpha_zero_returns_the_original_image(self):
        overlay = make_overlay(hot_spot_density(), IMAGE, alpha=0.0)
        np.testing.assert_array_equal(overlay, IMAGE)


class TestMakeGlowOverlay:
    def test_output_shape_and_dtype_match_the_image(self):
        overlay = make_glow_overlay(hot_spot_density(), IMAGE)
        assert overlay.shape == IMAGE.shape
        assert overlay.dtype == np.uint8

    def test_hot_spot_is_glowing_red_orange(self):
        overlay = make_glow_overlay(hot_spot_density(), IMAGE, percentile=80, blur_sigma=0.5)
        r, g, b = (int(v) for v in overlay[4, 4])
        assert r > g and r > b, f"expected the hot spot to glow red-orange, got RGB=({r},{g},{b})"

    def test_background_is_darkened_and_not_recolored(self):
        overlay = make_glow_overlay(hot_spot_density(), IMAGE, percentile=80, blur_sigma=0.5)
        r, g, b = (int(v) for v in overlay[15, 15])
        # far from the hot spot: darker than the source, and still roughly gray
        assert r < IMAGE[15, 15, 0]
        assert abs(r - g) < 5 and abs(g - b) < 5

    def test_darken_factor_of_one_leaves_cold_pixels_unchanged(self):
        overlay = make_glow_overlay(hot_spot_density(), IMAGE, darken_factor=1.0, blur_sigma=0.5, percentile=95)
        np.testing.assert_allclose(overlay[15, 15], IMAGE[15, 15], atol=1)

    def test_no_blue_or_green_colormap_leakage(self):
        # the whole point of the glow style is no rainbow gradient
        overlay = make_glow_overlay(hot_spot_density(), IMAGE, percentile=50, blur_sigma=2)
        # every pixel should have red as the dominant or equal channel
        assert np.all(overlay[..., 0] >= overlay[..., 1])
        assert np.all(overlay[..., 0] >= overlay[..., 2])
