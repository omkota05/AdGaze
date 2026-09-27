"""Generate before/after pairs that differ in exactly one way.

Writes lossless PNGs at the source image's own dimensions, so comparing them in
the app isolates the edit: a resize would change what share of the image a box
covers, and a JPEG re-encode would perturb the saliency map everywhere.

    python make_pairs.py
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

SOURCE = Path("test_images/test_image.jpg")
OUTPUT_DIR = Path("test_images_compare")

# the Nike swoosh and "Just do it." call to action, in SOURCE's pixels
CTA_BOX = (400, 895, 570, 960)

VARIANTS = {
    "control": None,
    "cta_boosted": {"color": 2.0, "contrast": 1.6},
    "cta_muted": {"color": 0.25, "contrast": 0.7},
}


def edit_region(image, box, color, contrast):
    """Return a copy of `image` with only `box` recolored."""
    region = image.crop(box)
    region = ImageEnhance.Color(region).enhance(color)
    region = ImageEnhance.Contrast(region).enhance(contrast)

    edited = image.copy()
    edited.paste(region, box)
    return edited


def count_changes(original, variant, box):
    """Return (pixels changed, pixels changed outside the box)."""
    changed = np.any(np.asarray(original) != np.asarray(variant), axis=2)
    x0, y0, x1, y1 = box
    return int(changed.sum()), int(changed.sum() - changed[y0:y1, x0:x1].sum())


def main():
    image = Image.open(SOURCE).convert("RGB")
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"source {SOURCE} {image.width}x{image.height}, editing {CTA_BOX}")
    for name, settings in VARIANTS.items():
        variant = image if settings is None else edit_region(image, CTA_BOX, **settings)
        path = OUTPUT_DIR / f"{name}.png"
        variant.save(path)

        changed, outside = count_changes(image, variant, CTA_BOX)
        if outside:
            raise SystemExit(f"{name}: {outside} pixels changed outside the box")
        print(f"  {path}  {variant.width}x{variant.height}  {changed} pixels changed")


if __name__ == "__main__":
    main()
