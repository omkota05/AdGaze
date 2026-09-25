import cv2
import numpy as np

from backend.arrays import as_2d_array


def make_overlay(log_density, image, alpha=0.45):
    """Blend a jet-colormapped saliency heatmap over the original image.

    Returns an RGB uint8 numpy array the same size as `image`.
    """
    density = np.exp(as_2d_array(log_density))

    image = np.ascontiguousarray(np.asarray(image)[..., :3]).astype(np.uint8)
    height, width = image.shape[:2]

    if density.shape != (height, width):
        density = cv2.resize(density, (width, height), interpolation=cv2.INTER_LINEAR)

    low, high = float(density.min()), float(density.max())
    scaled = (density - low) / (high - low) if high > low else np.zeros_like(density)

    heatmap = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(heatmap, alpha, image, 1.0 - alpha, 0.0)
