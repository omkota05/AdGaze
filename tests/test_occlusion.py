import numpy as np
from PIL import Image

from validate_occlusion import hottest_and_coldest_box, occlude


def test_hottest_box_centers_on_the_peak():
    density = np.ones((100, 200), dtype=np.float64)
    density[20:40, 140:180] = 10.0
    hot, cold = hottest_and_coldest_box(density, frac_w=0.2, frac_h=0.2)
    x0, y0, x1, y1 = hot
    assert x1 - x0 == 40
    assert y1 - y0 == 20
    # the bright patch should sit inside the chosen window
    assert x0 <= 140 and x1 >= 180
    assert y0 <= 20 and y1 >= 40
    cx0, cy0, cx1, cy1 = cold
    assert (cx0, cy0, cx1, cy1) != hot
    assert density[cy0:cy1, cx0:cx1].mean() < density[y0:y1, x0:x1].mean()


def test_occlude_fills_only_the_box():
    image = Image.fromarray(np.full((40, 60, 3), 80, dtype=np.uint8))
    pixels = np.array(image)
    pixels[5:15, 10:30] = (200, 10, 10)
    image = Image.fromarray(pixels)
    box = (10, 5, 30, 15)
    out = np.array(occlude(image, box, ring=4))
    assert not np.array_equal(out[5:15, 10:30], pixels[5:15, 10:30])
    assert np.array_equal(out[:5], pixels[:5])
    assert np.array_equal(out[15:], pixels[15:])
