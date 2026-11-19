"""Saliency-benchmark metrics used on CAT2000 (Bylinskii / MIT protocol)."""

import numpy as np


def _as_map(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"expected a 2D map, got shape {array.shape}")
    return array


def _resize_to(source, height, width):
    if source.shape == (height, width):
        return source
    import cv2

    return cv2.resize(source, (width, height), interpolation=cv2.INTER_LINEAR)


def nss(saliency, fixations):
    """Normalized Scanpath Saliency: z-scored map averaged at fixation pixels.

    Higher is better. Chance is 0. DeepGaze IIE reports about 2.11 on CAT2000 test.
    """
    saliency = _as_map(saliency)
    fixations = _as_map(fixations)
    if fixations.shape != saliency.shape:
        fixations = _resize_to(fixations, *saliency.shape)
    locations = fixations > 0
    if not locations.any():
        raise ValueError("fixation map has no positive pixels")
    std = saliency.std()
    if std == 0:
        return 0.0
    zscored = (saliency - saliency.mean()) / std
    return float(zscored[locations].mean())


def auc_judd(saliency, fixations):
    """AUC-Judd: saliency as a classifier of fixation vs non-fixation pixels.

    Computed as the Mann-Whitney U statistic (equivalent to the ROC area).
    Higher is better. Chance is 0.5. DeepGaze IIE reports about 0.87 on CAT2000 test.
    """
    saliency = _as_map(saliency)
    fixations = _as_map(fixations)
    if fixations.shape != saliency.shape:
        fixations = _resize_to(fixations, *saliency.shape)
    positives = (fixations > 0).ravel()
    if not positives.any():
        raise ValueError("fixation map has no positive pixels")

    from scipy.stats import rankdata

    scores = saliency.ravel()
    n_pos = int(positives.sum())
    n_neg = scores.size - n_pos
    # average ranks for ties, so a flat map scores exactly 0.5
    ranks = rankdata(scores, method="average")
    u = ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def cc(saliency, fixation_map):
    """Pearson correlation between the predicted map and the human density map.

    Higher is better. DeepGaze IIE reports about 0.82 on CAT2000 test.
    """
    saliency = _as_map(saliency)
    fixation_map = _as_map(fixation_map)
    if fixation_map.shape != saliency.shape:
        fixation_map = _resize_to(fixation_map, *saliency.shape)
    if saliency.std() == 0 or fixation_map.std() == 0:
        return 0.0
    return float(np.corrcoef(saliency.ravel(), fixation_map.ravel())[0, 1])


def box_share(weights, box):
    """Fraction of a map's total mass inside box (x0, y0, x1, y1)."""
    weights = _as_map(weights)
    total = weights.sum()
    if total <= 0:
        return 0.0
    x0, y0, x1, y1 = box
    return float(weights[y0:y1, x0:x1].sum() / total)


def _clear_gap(scores, first, second, min_gap):
    larger = np.maximum(scores[first], scores[second])
    return (larger > 0) & (np.abs(scores[first] - scores[second]) >= min_gap * larger)


def pairwise_accuracy(predicted, human, min_gap=0.1, tied_under=None):
    """How often `predicted` ranks a pair of regions the same way humans did.

    `predicted` and `human` are sequences of scores for regions that are
    compared against each other (same image, same box size). Pairs whose
    human scores differ by less than `min_gap` of the larger one are near
    ties and are skipped. If `tied_under` is given, only pairs that those
    scores cannot separate are counted, which isolates what `predicted`
    knows beyond them. Returns (correct, counted).
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    human = np.asarray(human, dtype=np.float64)
    first, second = np.triu_indices(len(human), k=1)
    decisive = _clear_gap(human, first, second, min_gap)
    if tied_under is not None:
        reference = np.asarray(tied_under, dtype=np.float64)
        decisive &= ~_clear_gap(reference, first, second, min_gap)
    agree = np.sign(predicted[first] - predicted[second]) == np.sign(human[first] - human[second])
    return int((agree & decisive).sum()), int(decisive.sum())
