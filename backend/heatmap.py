import cv2
import numpy as np

from backend.arrays import as_2d_array

GLOW_COLOR_RGB = (255, 60, 20)


def _density_and_image(log_density, image):
    """Return (density, image) as float density and RGB uint8 image of equal size."""
    density = np.exp(as_2d_array(log_density))

    image = np.ascontiguousarray(np.asarray(image)[..., :3]).astype(np.uint8)
    height, width = image.shape[:2]

    if density.shape != (height, width):
        density = cv2.resize(density, (width, height), interpolation=cv2.INTER_LINEAR)

    return density, image


def make_overlay(log_density, image, alpha=0.45):
    """Blend a jet-colormapped saliency heatmap over the original image.

    Returns an RGB uint8 numpy array the same size as `image`.
    """
    density, image = _density_and_image(log_density, image)

    low, high = float(density.min()), float(density.max())
    scaled = (density - low) / (high - low) if high > low else np.zeros_like(density)

    heatmap = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(heatmap, alpha, image, 1.0 - alpha, 0.0)


def make_glow_overlay(log_density, image, percentile=80, darken_factor=0.35, blur_sigma=15):
    """Dim the image and make only its highest-attention areas glow red-orange.

    Density above `percentile` is ramped from 0 to 1 and blurred, so the glow
    fades out smoothly instead of ending on a hard edge.

    Returns an RGB uint8 numpy array the same size as `image`.
    """
    density, image = _density_and_image(log_density, image)

    high = float(density.max())
    density = density / high if high > 0 else np.zeros_like(density)

    threshold = float(np.percentile(density, percentile))
    headroom = 1.0 - threshold
    if headroom > 0:
        mask = np.clip((density - threshold) / headroom, 0.0, 1.0)
    else:
        mask = (density >= threshold).astype(np.float64)

    mask = cv2.GaussianBlur(mask, (0, 0), blur_sigma)
    mask = mask[..., None]

    darkened = image.astype(np.float64) * darken_factor
    glow = np.asarray(GLOW_COLOR_RGB, dtype=np.float64)

    blended = darkened * (1.0 - mask) + glow * mask

    return np.clip(blended, 0, 255).astype(np.uint8)
