import numpy as np


def as_2d_array(log_density):
    """Accept a torch tensor or numpy array of shape (H, W), (1, H, W) or
    (1, 1, H, W) and return a plain 2D float numpy array."""
    if hasattr(log_density, "detach"):
        log_density = log_density.detach().cpu().numpy()

    array = np.asarray(log_density, dtype=np.float64)
    array = np.squeeze(array)

    if array.ndim != 2:
        raise ValueError(f"expected a 2D density map, got shape {np.shape(log_density)}")

    return array


def clip_box(box, height, width, image_shape=None):
    """Clip a (x0, y0, x1, y1) pixel box to a density map of the given size.

    If image_shape is given, the box is assumed to be in that image's pixel
    coordinates and is rescaled to the density map's resolution first.
    """
    x0, y0, x1, y1 = box

    if image_shape is not None:
        image_height, image_width = image_shape[:2]
        x_scale = width / image_width
        y_scale = height / image_height
        x0, x1 = x0 * x_scale, x1 * x_scale
        y0, y1 = y0 * y_scale, y1 * y_scale

    x0, x1 = sorted((int(round(x0)), int(round(x1))))
    y0, y1 = sorted((int(round(y0)), int(round(y1))))

    x0, x1 = max(x0, 0), min(x1, width)
    y0, y1 = max(y0, 0), min(y1, height)

    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"box {tuple(box)} is empty after clipping to {width}x{height}")

    return x0, y0, x1, y1
