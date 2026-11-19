import numpy as np
import pytest

from backend.benchmark import auc_judd, box_share, cc, nss, pairwise_accuracy


def test_nss_is_zero_on_a_flat_map():
    saliency = np.ones((20, 30))
    fixations = np.zeros((20, 30))
    fixations[5, 8] = 1
    assert nss(saliency, fixations) == 0.0


def test_nss_is_high_when_fixations_sit_on_the_peak():
    saliency = np.zeros((20, 30))
    saliency[4:8, 10:14] = 10.0
    fixations = np.zeros((20, 30))
    fixations[5, 12] = 1
    off = np.zeros((20, 30))
    off[18, 2] = 1
    assert nss(saliency, fixations) > 2.0
    assert nss(saliency, off) < 0.0


def test_auc_is_one_for_a_perfect_ranking():
    saliency = np.zeros((10, 10))
    saliency[0, 0] = 1.0
    fixations = np.zeros((10, 10))
    fixations[0, 0] = 1
    assert auc_judd(saliency, fixations) == pytest.approx(1.0, abs=0.02)


def test_auc_is_chance_when_the_map_is_flat():
    saliency = np.ones((12, 12))
    fixations = np.zeros((12, 12))
    fixations[3, 4] = 1
    assert auc_judd(saliency, fixations) == pytest.approx(0.5, abs=0.05)


def test_cc_is_one_for_identical_maps():
    density = np.linspace(0, 1, 100).reshape(10, 10)
    assert cc(density, density) == pytest.approx(1.0)


def test_cc_is_minus_one_for_inverted_maps():
    density = np.linspace(0, 1, 100).reshape(10, 10)
    assert cc(density, 1.0 - density) == pytest.approx(-1.0)


def test_empty_fixations_raise():
    with pytest.raises(ValueError):
        nss(np.ones((4, 4)), np.zeros((4, 4)))


def test_auc_is_below_chance_for_an_inverted_map():
    saliency = np.ones((10, 10))
    saliency[0, 0] = 0.0
    fixations = np.zeros((10, 10))
    fixations[0, 0] = 1
    assert auc_judd(saliency, fixations) == pytest.approx(0.0)


def test_box_share_matches_area_fraction_on_a_flat_map():
    assert box_share(np.ones((10, 20)), (0, 0, 10, 5)) == pytest.approx(0.25)


def test_box_share_of_an_empty_map_is_zero():
    assert box_share(np.zeros((4, 4)), (0, 0, 2, 2)) == 0.0


def test_pairwise_accuracy_is_perfect_for_matching_rankings():
    human = [0.5, 0.3, 0.1, 0.05]
    correct, counted = pairwise_accuracy([4, 3, 2, 1], human)
    assert counted == 6
    assert correct == 6


def test_pairwise_accuracy_is_zero_for_reversed_rankings():
    correct, counted = pairwise_accuracy([1, 2, 3], [0.5, 0.3, 0.1])
    assert (correct, counted) == (0, 3)


def test_pairwise_accuracy_can_keep_only_pairs_a_reference_cannot_separate():
    human = [0.6, 0.3, 0.1]
    predicted = [3, 2, 1]
    # the reference separates every pair except (0, 1)
    reference = [0.50, 0.49, 0.10]
    assert pairwise_accuracy(predicted, human, tied_under=reference) == (1, 1)


def test_pairwise_accuracy_skips_near_ties_and_empty_pairs():
    # 0.50 vs 0.48 is a near tie; 0 vs 0 has no human signal
    correct, counted = pairwise_accuracy([2, 1, 3, 4], [0.50, 0.48, 0.0, 0.0])
    assert counted == 4
    assert correct == 0
