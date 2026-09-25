import numpy as np

from backend.arrays import as_2d_array, clip_box


def compute_prominence(log_density, box, image_shape=None):
    """Mean predicted attention inside the box relative to the image average.

    Above 1.0 means the region draws more attention than an average region of
    the image, below 1.0 means less.
    """
    density = np.exp(as_2d_array(log_density))
    x0, y0, x1, y1 = clip_box(box, *density.shape, image_shape=image_shape)

    return float(density[y0:y1, x0:x1].mean() / density.mean())


def compute_on_target_salience(log_density, box, image_shape=None):
    """Fraction of the total predicted attention that lands inside the box.

    Result is between 0 and 1.
    """
    density = np.exp(as_2d_array(log_density))
    x0, y0, x1, y1 = clip_box(box, *density.shape, image_shape=image_shape)

    return float(density[y0:y1, x0:x1].sum() / density.sum())
