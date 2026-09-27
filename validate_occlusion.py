"""Occlude the hottest region in every test image and measure the drop.

The Nike billboard is one edited pair. This script asks the same question
across every photo in test_images/: if you cover the thing the model looks
at most, does that region's Attention Multiplier and Attention Share fall?

For each image:
  1. Run DeepGaze and find the fixed-size window with the highest mean density.
  2. Score that box (the hot element).
  3. Fill the box with the mean color of a ring around it, so the element
     disappears without introducing a high-contrast sticker.
  4. Score the same box on the occluded copy.
  5. As a control, occlude the coldest window of the same size and score the
     original hot box again. That drop should stay near zero.

    venv/bin/python validate_occlusion.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

IMAGE_DIR = Path("test_images")
WINDOW_FRAC = (0.16, 0.08)
RING = 18


def hottest_and_coldest_box(density, frac_w=WINDOW_FRAC[0], frac_h=WINDOW_FRAC[1]):
    """Return the max-mean and min-mean windows of a fixed size, as (x0,y0,x1,y1)."""
    height, width = density.shape
    box_w = max(12, int(round(width * frac_w)))
    box_h = max(12, int(round(height * frac_h)))
    box_w = min(box_w, width)
    box_h = min(box_h, height)

    window_mean = uniform_filter(density.astype(np.float64), size=(box_h, box_w), mode="nearest")

    def box_at(index):
        cy, cx = np.unravel_index(index, window_mean.shape)
        x0 = int(np.clip(cx - box_w // 2, 0, width - box_w))
        y0 = int(np.clip(cy - box_h // 2, 0, height - box_h))
        return x0, y0, x0 + box_w, y0 + box_h

    return box_at(int(np.argmax(window_mean))), box_at(int(np.argmin(window_mean)))


def occlude(image, box, ring=RING):
    """Replace `box` with the mean color of a ring just outside it."""
    array = np.asarray(image)
    height, width = array.shape[:2]
    x0, y0, x1, y1 = box
    rx0, ry0 = max(0, x0 - ring), max(0, y0 - ring)
    rx1, ry1 = min(width, x1 + ring), min(height, y1 + ring)

    ring_mask = np.zeros((height, width), dtype=bool)
    ring_mask[ry0:ry1, rx0:rx1] = True
    ring_mask[y0:y1, x0:x1] = False
    if not ring_mask.any():
        fill = array.mean(axis=(0, 1))
    else:
        fill = array[ring_mask].mean(axis=0)

    edited = array.copy()
    edited[y0:y1, x0:x1] = fill
    return Image.fromarray(edited.astype(np.uint8))


def list_images():
    files = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"} and not path.name.startswith(".")
    )
    if not files:
        raise SystemExit(f"no images in {IMAGE_DIR}")
    return files


def main():
    from backend.metrics import compute_attention_multiplier, compute_attention_share
    from backend.model import SaliencyModel

    print("loading DeepGaze IIE...", flush=True)
    model = SaliencyModel()

    rows = []
    images = list_images()
    print(f"scoring {len(images)} images\n", flush=True)
    print(
        f"{'image':<22} {'hot box':<24} {'mult':>7} {'->':>2} {'after':>7} {'dM':>8} "
        f"{'share':>7} {'->':>2} {'after':>7} {'dS':>8} {'ctrl dM':>8}"
    )
    print("-" * 112)

    for path in images:
        image = Image.open(path).convert("RGB")
        frame = np.array(image)
        log_density = model.predict(frame)
        density = np.exp(log_density)
        hot, cold = hottest_and_coldest_box(density)

        before_m = compute_attention_multiplier(log_density, hot)
        before_s = compute_attention_share(log_density, hot)

        after_log = model.predict(np.array(occlude(image, hot)))
        after_m = compute_attention_multiplier(after_log, hot)
        after_s = compute_attention_share(after_log, hot)

        control_log = model.predict(np.array(occlude(image, cold)))
        control_m = compute_attention_multiplier(control_log, hot)

        row = {
            "image": path.name,
            "box": list(hot),
            "multiplier_before": before_m,
            "multiplier_after": after_m,
            "multiplier_delta": after_m - before_m,
            "share_before": before_s,
            "share_after": after_s,
            "share_delta": after_s - before_s,
            "control_multiplier": control_m,
            "control_delta": control_m - before_m,
        }
        rows.append(row)
        print(
            f"{path.name:<22} {str(hot):<24} {before_m:7.2f} -> {after_m:7.2f} "
            f"{row['multiplier_delta']:+8.2f} {before_s:7.1%} -> {after_s:7.1%} "
            f"{row['share_delta']*100:+7.1f}pt {row['control_delta']:+8.2f}",
            flush=True,
        )

    drops = [row["multiplier_delta"] for row in rows]
    share_drops = [row["share_delta"] for row in rows]
    controls = [row["control_delta"] for row in rows]
    n_down = sum(delta < 0 for delta in drops)

    print("\nsummary")
    print(f"  n = {len(rows)}")
    print(
        f"  hot-region occluded:  mean multiplier {np.mean(drops):+.2f}x  "
        f"median {np.median(drops):+.2f}x  ({n_down}/{len(rows)} images dropped)"
    )
    print(
        f"  hot-region occluded:  mean share {np.mean(share_drops)*100:+.2f} pts  "
        f"median {np.median(share_drops)*100:+.2f} pts"
    )
    print(
        f"  cold-region occluded (control, score the hot box):  "
        f"mean multiplier {np.mean(controls):+.2f}x  median {np.median(controls):+.2f}x"
    )

    out = Path("docs/occlusion_sweep.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
