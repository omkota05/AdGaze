"""End-to-end regression tests against the real DeepGaze model.

Slow (~60s to load the model): run explicitly with `pytest -m slow`.
These pin the exact values reported throughout the project's development, so
a refactor that silently changes the math will fail loudly here.
"""

import numpy as np
import pytest
from PIL import Image

from backend.metrics import compute_attention_multiplier, compute_attention_share
from backend.model import SaliencyModel

pytestmark = pytest.mark.slow

TEST_IMAGE = "test_images/test_image.jpg"


@pytest.fixture(scope="module")
def log_density():
    image = np.array(Image.open(TEST_IMAGE).convert("RGB"))
    model = SaliencyModel()
    return model.predict(image)


class TestKnownRegions:
    def test_athletes_face(self, log_density):
        box = (320, 340, 540, 560)
        assert compute_attention_multiplier(log_density, box) == pytest.approx(5.393543574940448)
        assert compute_attention_share(log_density, box) == pytest.approx(0.16145457645190997)

    def test_subway_storefront_sign(self, log_density):
        box = (160, 1040, 470, 1100)
        assert compute_attention_multiplier(log_density, box) == pytest.approx(5.955565457792614)
        assert compute_attention_share(log_density, box) == pytest.approx(0.06851201690878958)

    def test_empty_sky(self, log_density):
        box = (60, 60, 280, 280)
        assert compute_attention_multiplier(log_density, box) < 0.1
        assert compute_attention_share(log_density, box) < 0.01
