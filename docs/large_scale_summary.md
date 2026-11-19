# Large-scale validation on CAT2000

1,000 images from the public CAT2000 training split, 50 from each of 20 categories, each with fixations from 18 real viewers. Every result is compared to a center-only baseline (the model's own center-bias prior with no image content) so the numbers show what the deep model adds.

## Headline numbers

- **Matches human eye tracking at AUC 0.858** (chance 0.500, center-only 0.833); beats the center-only baseline on **73%** of images.
- **Attention Share picks the region humans looked at more 86.0% of the time** across 39,220 head-to-head comparisons (center-only 81.3%, chance 50%), **25% fewer wrong calls** than the center-only baseline.
- **When position alone cannot decide** (two regions the center-only baseline rates equally), **AdGaze still picks the region humans looked at more 76.2% of the time** across 2,658 comparisons, versus 55.9% for the baseline.
- Finds the single most-viewed region out of 10 candidates 61% of the time (chance 10%), on par with center-only (60%) because CAT2000's most-viewed region is usually the center.
- **Attention Share tracks the human fixation share at Spearman 0.90** across 10,000 regions (center-only 0.84).
- **Covering the hottest region lowered its attention in 160/200 images** (median -32%); covering a cold region of the same size moved it +0.0%.
- Runs at 2.4s per 1920x1080 image (~1,495 images per hour) on mps.

## Test A: heatmap vs human fixations

| | AUC | NSS | CC |
|---|---|---|---|
| AdGaze (DeepGaze IIE) | 0.858 | 1.90 | 0.726 |
| Center-only baseline | 0.833 | 1.74 | 0.685 |
| Published, CAT2000 test set | 0.869 | 2.11 | 0.819 |

The published row is the MIT/Tuebingen leaderboard on the held-out test set, scored with the benchmark's own protocol, so it is a sanity reference rather than an exact target. CC blurs each prediction by 38px (1 degree of visual angle), the same kernel CAT2000 used to build its human fixation maps.

### By category

| Category | AUC | Center AUC | NSS | CC |
|---|---|---|---|---|
| Action | 0.878 | 0.831 | 2.33 | 0.800 |
| Affective | 0.877 | 0.821 | 2.43 | 0.814 |
| Art | 0.851 | 0.826 | 1.83 | 0.721 |
| BlackWhite | 0.878 | 0.851 | 2.19 | 0.740 |
| Cartoon | 0.841 | 0.809 | 1.84 | 0.743 |
| Fractal | 0.870 | 0.856 | 2.06 | 0.756 |
| Indoor | 0.841 | 0.806 | 1.64 | 0.752 |
| Inverted | 0.837 | 0.814 | 1.62 | 0.701 |
| Jumbled | 0.826 | 0.789 | 1.35 | 0.693 |
| LineDrawing | 0.840 | 0.832 | 1.56 | 0.674 |
| LowResolution | 0.865 | 0.879 | 1.83 | 0.639 |
| Noisy | 0.859 | 0.852 | 1.88 | 0.689 |
| Object | 0.877 | 0.834 | 2.14 | 0.754 |
| OutdoorManMade | 0.855 | 0.831 | 1.92 | 0.774 |
| OutdoorNatural | 0.845 | 0.831 | 1.76 | 0.714 |
| Pattern | 0.852 | 0.858 | 1.66 | 0.635 |
| Random | 0.876 | 0.825 | 1.93 | 0.687 |
| Satelite | 0.803 | 0.809 | 1.38 | 0.637 |
| Sketch | 0.913 | 0.896 | 2.62 | 0.797 |
| Social | 0.868 | 0.816 | 2.07 | 0.803 |

## Test B: Attention Share vs human fixation share

10 random boxes per image, one size per image (10%, 20%, or 30% of each side), 10,000 regions total. Human share is the fraction of the image's fixated pixels inside the box. Pairs whose human shares differ by less than 10% are near ties and are skipped.

| | AdGaze | Center only | Chance |
|---|---|---|---|
| Head-to-head accuracy | 86.0% | 81.3% | 50% |
| Head-to-head, position-tied pairs (2,658) | 76.2% | 55.9% | 50% |
| Top region out of 10 | 60.9% | 59.9% | 10% |
| Spearman, all regions | 0.903 | 0.837 | 0 |
| Pearson, all regions | 0.895 | 0.852 | 0 |

| Box size | Regions | AdGaze Spearman | Center Spearman |
|---|---|---|---|
| 10% box | 3,340 | 0.746 | 0.646 |
| 20% box | 3,330 | 0.854 | 0.718 |
| 30% box | 3,330 | 0.877 | 0.795 |

## Test C: occlusion

200 images (10 per category). The hottest window is filled with the mean color of its surroundings and re-scored; the control fills the coldest window of the same size and re-scores the hot box.

| | Hot region covered | Cold region covered (control) |
|---|---|---|
| Images where attention dropped | 160/200 | n/a |
| Median change in multiplier | -32.2% | +0.0% |
| Mean change in multiplier | -3.76x | +0.01x |
| Mean change in share | -4.92 pts | n/a |

Images where covering the hottest region lowered its attention, by category. Misses cluster in texture-heavy categories (noise, patterns, natural scenes) where a flat patch is itself an eye-catching anomaly and there is no distinct element to hide.

| Category | Dropped |
|---|---|
| Action | 10/10 |
| Inverted | 10/10 |
| Jumbled | 10/10 |
| Sketch | 10/10 |
| Social | 10/10 |
| Affective | 9/10 |
| Art | 9/10 |
| Cartoon | 9/10 |
| LineDrawing | 9/10 |
| Object | 9/10 |
| Fractal | 8/10 |
| Indoor | 8/10 |
| OutdoorManMade | 8/10 |
| LowResolution | 7/10 |
| Random | 7/10 |
| Satelite | 7/10 |
| BlackWhite | 6/10 |
| Pattern | 6/10 |
| Noisy | 4/10 |
| OutdoorNatural | 4/10 |

## Caveat

CAT2000 is general scenes, not ads. These results show the predictions agree with real human eye tracking; they do not show that a higher score sells more product.
