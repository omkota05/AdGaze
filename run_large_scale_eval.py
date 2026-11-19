"""Large-scale validation of AdGaze against real human eye tracking (CAT2000).

Three subcommands, run in order:

    venv/bin/python run_large_scale_eval.py predict     # model pass, cached, resumable
    venv/bin/python run_large_scale_eval.py analyze     # tests A and B from the cache
    venv/bin/python run_large_scale_eval.py occlusion   # test C on a 200-image subset

Test A: does the heatmap match where 18 real viewers looked (AUC, NSS, CC)?
Test B: does Attention Share rank regions the way human fixations do?
Test C: does covering the hottest region reliably lower its attention?

Each is scored against a center-only baseline (the model's own center-bias
prior with no image content), so the numbers show what the deep model adds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from backend.benchmark import auc_judd, box_share, cc, nss, pairwise_accuracy
from evaluate_cat2000 import (
    PUBLISHED,
    ensure_dataset,
    find_roots,
    list_pairs,
    load_binary_map,
    load_density_map,
    subsample,
)

CACHE_DIR = Path("data/cache/cat2000")
PREDICTIONS = CACHE_DIR / "predictions.jsonl"
OCCLUSIONS = CACHE_DIR / "occlusion.jsonl"
DOCS = Path("docs")
DOWNSAMPLE = 4
PER_CATEGORY = 50
OCCLUSION_PER_CATEGORY = 10
SEED = 0
BOX_FRACTIONS = [(0.10, 0.10), (0.20, 0.20), (0.30, 0.30)]
BOXES_PER_IMAGE = 10
NEAR_TIE = 0.10
# CAT2000's released fixation maps are fixations blurred by 1 degree of visual
# angle (38px); CC compares like with like by blurring predictions the same way.
FIXATION_MAP_SIGMA = 38


def cache_path(name):
    return CACHE_DIR / (name.replace("/", "__").rsplit(".", 1)[0] + ".npz")


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def downsample(array, factor=DOWNSAMPLE):
    import cv2

    height, width = array.shape
    size = (max(1, width // factor), max(1, height // factor))
    return cv2.resize(array.astype(np.float32), size, interpolation=cv2.INTER_AREA)


def blur_like_fixation_maps(density):
    import cv2

    return cv2.GaussianBlur(density.astype(np.float32), (0, 0), FIXATION_MAP_SIGMA)


def to_cache_dtype(density):
    """Scale to a max of 1 so float16 keeps precision; shares are scale-free."""
    peak = float(density.max())
    return (density / peak if peak > 0 else density).astype(np.float16)


def sample_images(per_category):
    stimuli, locations, maps = find_roots(ensure_dataset())
    return subsample(list_pairs(stimuli, locations, maps), per_category, SEED)


def load_model(device):
    from backend.model import SaliencyModel

    return SaliencyModel(device=device)


def format_duration(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{seconds:02d}s"


# ---------------------------------------------------------------- predict


def predict(args):
    pairs = sample_images(args.per_category)
    if args.limit:
        pairs = pairs[: args.limit]
    done = {row["image"] for row in read_jsonl(PREDICTIONS)}
    todo = [pair for pair in pairs if pair["name"] not in done]
    print(f"{len(pairs)} images selected, {len(done & {p['name'] for p in pairs})} cached, {len(todo)} to run", flush=True)
    if not todo:
        return 0

    print(f"loading DeepGaze IIE on {args.device}...", flush=True)
    model = load_model(args.device)
    started = time.time()

    for index, pair in enumerate(todo, start=1):
        frame = np.array(Image.open(pair["image"]).convert("RGB"))
        tick = time.time()
        log_density = model.predict(frame)
        latency = time.time() - tick

        density = np.exp(log_density.astype(np.float64))
        height, width = density.shape
        center = np.exp(model._centerbias_for(height, width))
        fixations = load_binary_map(pair["locations"], (height, width))
        human_map = (
            load_density_map(pair["fixation_map"], (height, width))
            if pair["fixation_map"] is not None
            else None
        )
        if not fixations.any():
            print(f"skip {pair['name']}: no fixations", flush=True)
            continue

        row = {
            "image": pair["name"],
            "category": pair["category"],
            "shape": [height, width],
            "latency_s": latency,
            "fixation_pixels": int(fixations.sum()),
            "auc": auc_judd(density, fixations),
            "nss": nss(density, fixations),
            "cc": cc(blur_like_fixation_maps(density), human_map) if human_map is not None else None,
            "center_auc": auc_judd(center, fixations),
            "center_nss": nss(center, fixations),
            "center_cc": cc(blur_like_fixation_maps(center), human_map) if human_map is not None else None,
        }

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path(pair["name"]),
            model=to_cache_dtype(downsample(density)),
            center=to_cache_dtype(downsample(center)),
            human=downsample(fixations) * DOWNSAMPLE * DOWNSAMPLE,
        )
        append_jsonl(PREDICTIONS, row)

        elapsed = time.time() - started
        remaining = elapsed / index * (len(todo) - index)
        print(
            f"[{index}/{len(todo)}] {pair['name']:<28} AUC {row['auc']:.3f} "
            f"(center {row['center_auc']:.3f})  {latency:.1f}s  eta {format_duration(remaining)}",
            flush=True,
        )
    return 0


# ---------------------------------------------------------------- analyze


def mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def sample_boxes(shape, fraction, count, rng):
    height, width = shape
    box_w = max(2, int(round(width * fraction[0])))
    box_h = max(2, int(round(height * fraction[1])))
    xs = rng.integers(0, width - box_w + 1, size=count)
    ys = rng.integers(0, height - box_h + 1, size=count)
    return [(int(x), int(y), int(x) + box_w, int(y) + box_h) for x, y in zip(xs, ys)]


def attention_share_test(rows):
    """Test B: Attention Share against the share of real human fixations."""
    from scipy.stats import pearsonr, spearmanr

    rng = np.random.default_rng(SEED)
    regions = []
    tallies = {"model": [0, 0], "center": [0, 0]}
    position_ties = [0, 0, 0]
    top_pick = {"model": 0, "center": 0, "images": 0}

    for index, row in enumerate(sorted(rows, key=lambda item: item["image"])):
        cached = np.load(cache_path(row["image"]))
        model_map = cached["model"].astype(np.float64)
        center_map = cached["center"].astype(np.float64)
        human_map = cached["human"].astype(np.float64)

        size_class = index % len(BOX_FRACTIONS)
        boxes = sample_boxes(model_map.shape, BOX_FRACTIONS[size_class], BOXES_PER_IMAGE, rng)
        model_share = [box_share(model_map, box) for box in boxes]
        center_share = [box_share(center_map, box) for box in boxes]
        human_share = [box_share(human_map, box) for box in boxes]

        for name, predicted in (("model", model_share), ("center", center_share)):
            correct, counted = pairwise_accuracy(predicted, human_share, NEAR_TIE)
            tallies[name][0] += correct
            tallies[name][1] += counted
        correct, counted = pairwise_accuracy(model_share, human_share, NEAR_TIE, tied_under=center_share)
        position_ties[0] += correct
        position_ties[1] += counted
        position_ties[2] += pairwise_accuracy(center_share, human_share, NEAR_TIE, tied_under=center_share)[0]

        if max(human_share) > 0:
            winner = int(np.argmax(human_share))
            top_pick["images"] += 1
            top_pick["model"] += int(np.argmax(model_share) == winner)
            top_pick["center"] += int(np.argmax(center_share) == winner)

        for box, ms, cs, hs in zip(boxes, model_share, center_share, human_share):
            regions.append(
                {"size_class": size_class, "model": ms, "center": cs, "human": hs, "category": row["category"]}
            )

    def correlations(key, subset):
        predicted = [region[key] for region in subset]
        human = [region["human"] for region in subset]
        return {
            "pearson": float(pearsonr(predicted, human)[0]),
            "spearman": float(spearmanr(predicted, human)[0]),
        }

    per_size = {}
    for size_class, fraction in enumerate(BOX_FRACTIONS):
        subset = [region for region in regions if region["size_class"] == size_class]
        per_size[f"{int(fraction[0] * 100)}% box"] = {
            "regions": len(subset),
            "model": correlations("model", subset),
            "center": correlations("center", subset),
        }

    return {
        "regions": len(regions),
        "boxes_per_image": BOXES_PER_IMAGE,
        "near_tie_threshold": NEAR_TIE,
        "pairwise": {
            name: {
                "correct": correct,
                "compared": counted,
                "accuracy": correct / counted if counted else None,
            }
            for name, (correct, counted) in tallies.items()
        },
        "position_tied_pairs": {
            "correct": position_ties[0],
            "compared": position_ties[1],
            "model_accuracy": position_ties[0] / position_ties[1] if position_ties[1] else None,
            "center_accuracy": position_ties[2] / position_ties[1] if position_ties[1] else None,
        },
        "error_reduction_vs_center": 1 - (
            (tallies["model"][1] - tallies["model"][0]) / tallies["model"][1]
        ) / ((tallies["center"][1] - tallies["center"][0]) / tallies["center"][1]),
        "top_pick": {
            "images": top_pick["images"],
            "model_accuracy": top_pick["model"] / top_pick["images"],
            "center_accuracy": top_pick["center"] / top_pick["images"],
            "chance": 1.0 / BOXES_PER_IMAGE,
        },
        "correlation_all": {
            "model": correlations("model", regions),
            "center": correlations("center", regions),
        },
        "correlation_by_size": per_size,
    }


def benchmark_test(rows):
    """Test A: heatmap agreement with human fixations, overall and per category."""
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    def summarize(subset):
        return {
            "n": len(subset),
            "auc": mean(row["auc"] for row in subset),
            "nss": mean(row["nss"] for row in subset),
            "cc": mean(row["cc"] for row in subset),
            "center_auc": mean(row["center_auc"] for row in subset),
            "center_nss": mean(row["center_nss"] for row in subset),
            "center_cc": mean(row["center_cc"] for row in subset),
            "beats_center": float(np.mean([row["auc"] > row["center_auc"] for row in subset])),
        }

    latencies = [row["latency_s"] for row in rows]
    return {
        "overall": summarize(rows),
        "per_category": {category: summarize(subset) for category, subset in sorted(by_category.items())},
        "published_cat2000_test": PUBLISHED,
        "latency": {
            "mean_s": float(np.mean(latencies)),
            "median_s": float(np.median(latencies)),
            "images_per_hour": 3600.0 / float(np.mean(latencies)),
        },
    }


def occlusion_summary(rows):
    if not rows:
        return None
    drops = np.array([row["multiplier_after"] - row["multiplier_before"] for row in rows])
    relative = np.array([row["multiplier_after"] / row["multiplier_before"] - 1 for row in rows])
    shares = np.array([row["share_after"] - row["share_before"] for row in rows])
    controls = np.array([row["control_multiplier"] - row["multiplier_before"] for row in rows])
    control_relative = np.array([row["control_multiplier"] / row["multiplier_before"] - 1 for row in rows])
    per_category = defaultdict(lambda: [0, 0])
    for row, drop in zip(rows, drops):
        per_category[row["category"]][0] += int(drop < 0)
        per_category[row["category"]][1] += 1
    return {
        "per_category": {category: {"dropped": d, "n": n} for category, (d, n) in sorted(per_category.items())},
        "n": len(rows),
        "dropped": int((drops < 0).sum()),
        "dropped_fraction": float((drops < 0).mean()),
        "mean_multiplier_change": float(drops.mean()),
        "median_relative_change": float(np.median(relative)),
        "mean_share_change_pts": float(shares.mean() * 100),
        "control_mean_multiplier_change": float(controls.mean()),
        "control_median_relative_change": float(np.median(control_relative)),
    }


def write_chart(results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairwise = results["attention_share"]["pairwise"]
    overall = results["benchmark"]["overall"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    panels = [
        ("Heatmap vs human fixations\n(AUC, 0.5 = chance)", [0.5, overall["center_auc"], overall["auc"]], 0.4, 1.0, "{:.3f}"),
        ("Picks the region humans looked at more\n(head-to-head accuracy)",
         [0.5, pairwise["center"]["accuracy"], pairwise["model"]["accuracy"]], 0.0, 1.0, "{:.0%}"),
        ("Same comparison when position alone\ncannot decide (equally central regions)",
         [0.5, results["attention_share"]["position_tied_pairs"]["center_accuracy"],
          results["attention_share"]["position_tied_pairs"]["model_accuracy"]], 0.0, 1.0, "{:.0%}"),
    ]
    labels = ["Chance", "Center only", "AdGaze"]
    colors = ["#c8bfb0", "#8a8273", "#c45c26"]
    for axis, (title, values, low, high, fmt) in zip(axes, panels):
        bars = axis.bar(labels, values, color=colors)
        axis.set_ylim(low, high)
        axis.set_title(title, fontsize=10)
        axis.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + (high - low) * 0.015,
                      fmt.format(value), ha="center", fontsize=10)
    n = results["benchmark"]["overall"]["n"]
    fig.suptitle(f"AdGaze vs real human eye tracking on {n:,} CAT2000 images (18 viewers each)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(results, path):
    bench = results["benchmark"]
    overall = bench["overall"]
    share = results["attention_share"]
    pairwise = share["pairwise"]
    top = share["top_pick"]
    ties = share["position_tied_pairs"]
    corr = share["correlation_all"]
    occlusion = results.get("occlusion")
    published = bench["published_cat2000_test"]

    lines = [
        "# Large-scale validation on CAT2000",
        "",
        f"{overall['n']:,} images from the public CAT2000 training split, {overall['n'] // 20} from each of 20 "
        "categories, each with fixations from 18 real viewers. Every result is compared to a center-only "
        "baseline (the model's own center-bias prior with no image content) so the numbers show what the "
        "deep model adds.",
        "",
        "## Headline numbers",
        "",
        f"- **Matches human eye tracking at AUC {overall['auc']:.3f}** (chance 0.500, center-only "
        f"{overall['center_auc']:.3f}); beats the center-only baseline on "
        f"**{overall['beats_center']:.0%}** of images.",
        f"- **Attention Share picks the region humans looked at more {pairwise['model']['accuracy']:.1%} of the "
        f"time** across {pairwise['model']['compared']:,} head-to-head comparisons "
        f"(center-only {pairwise['center']['accuracy']:.1%}, chance 50%), "
        f"**{share['error_reduction_vs_center']:.0%} fewer wrong calls** than the center-only baseline.",
        f"- **When position alone cannot decide** (two regions the center-only baseline rates equally), "
        f"**AdGaze still picks the region humans looked at more {ties['model_accuracy']:.1%} of the time** "
        f"across {ties['compared']:,} comparisons, versus {ties['center_accuracy']:.1%} for the baseline.",
        f"- Finds the single most-viewed region out of 10 candidates {top['model_accuracy']:.0%} of the time "
        f"(chance {top['chance']:.0%}), on par with center-only ({top['center_accuracy']:.0%}) because CAT2000's "
        "most-viewed region is usually the center.",
        f"- **Attention Share tracks the human fixation share at Spearman {corr['model']['spearman']:.2f}** across "
        f"{share['regions']:,} regions (center-only {corr['center']['spearman']:.2f}).",
    ]
    if occlusion:
        lines.append(
            f"- **Covering the hottest region lowered its attention in {occlusion['dropped']}/{occlusion['n']} "
            f"images** (median {occlusion['median_relative_change']:+.0%}); covering a cold region of the same "
            f"size moved it {occlusion['control_median_relative_change']:+.1%}."
        )
    lines += [
        f"- Runs at {bench['latency']['mean_s']:.1f}s per 1920x1080 image "
        f"(~{bench['latency']['images_per_hour']:,.0f} images per hour) on {results['device']}.",
        "",
        "## Test A: heatmap vs human fixations",
        "",
        "| | AUC | NSS | CC |",
        "|---|---|---|---|",
        f"| AdGaze (DeepGaze IIE) | {overall['auc']:.3f} | {overall['nss']:.2f} | {overall['cc']:.3f} |",
        f"| Center-only baseline | {overall['center_auc']:.3f} | {overall['center_nss']:.2f} | {overall['center_cc']:.3f} |",
        f"| Published, CAT2000 test set | {published['auc']:.3f} | {published['nss']:.2f} | {published['cc']:.3f} |",
        "",
        "The published row is the MIT/Tuebingen leaderboard on the held-out test set, scored with the "
        "benchmark's own protocol, so it is a sanity reference rather than an exact target. CC blurs each "
        f"prediction by {FIXATION_MAP_SIGMA}px (1 degree of visual angle), the same kernel CAT2000 used to "
        "build its human fixation maps.",
        "",
        "### By category",
        "",
        "| Category | AUC | Center AUC | NSS | CC |",
        "|---|---|---|---|---|",
    ]
    for category, stats in bench["per_category"].items():
        lines.append(
            f"| {category} | {stats['auc']:.3f} | {stats['center_auc']:.3f} | {stats['nss']:.2f} | {stats['cc']:.3f} |"
        )
    lines += [
        "",
        "## Test B: Attention Share vs human fixation share",
        "",
        f"{share['boxes_per_image']} random boxes per image, one size per image (10%, 20%, or 30% of each "
        f"side), {share['regions']:,} regions total. Human share is the fraction of the image's fixated "
        f"pixels inside the box. Pairs whose human shares differ by less than {share['near_tie_threshold']:.0%} "
        "are near ties and are skipped.",
        "",
        "| | AdGaze | Center only | Chance |",
        "|---|---|---|---|",
        f"| Head-to-head accuracy | {pairwise['model']['accuracy']:.1%} | {pairwise['center']['accuracy']:.1%} | 50% |",
        f"| Head-to-head, position-tied pairs ({ties['compared']:,}) | {ties['model_accuracy']:.1%} | "
        f"{ties['center_accuracy']:.1%} | 50% |",
        f"| Top region out of 10 | {top['model_accuracy']:.1%} | {top['center_accuracy']:.1%} | {top['chance']:.0%} |",
        f"| Spearman, all regions | {corr['model']['spearman']:.3f} | {corr['center']['spearman']:.3f} | 0 |",
        f"| Pearson, all regions | {corr['model']['pearson']:.3f} | {corr['center']['pearson']:.3f} | 0 |",
        "",
        "| Box size | Regions | AdGaze Spearman | Center Spearman |",
        "|---|---|---|---|",
    ]
    for label, stats in share["correlation_by_size"].items():
        lines.append(
            f"| {label} | {stats['regions']:,} | {stats['model']['spearman']:.3f} | {stats['center']['spearman']:.3f} |"
        )
    if occlusion:
        lines += [
            "",
            "## Test C: occlusion",
            "",
            f"{occlusion['n']} images ({OCCLUSION_PER_CATEGORY} per category). The hottest window is filled "
            "with the mean color of its surroundings and re-scored; the control fills the coldest window "
            "of the same size and re-scores the hot box.",
            "",
            "| | Hot region covered | Cold region covered (control) |",
            "|---|---|---|",
            f"| Images where attention dropped | {occlusion['dropped']}/{occlusion['n']} | n/a |",
            f"| Median change in multiplier | {occlusion['median_relative_change']:+.1%} | "
            f"{occlusion['control_median_relative_change']:+.1%} |",
            f"| Mean change in multiplier | {occlusion['mean_multiplier_change']:+.2f}x | "
            f"{occlusion['control_mean_multiplier_change']:+.2f}x |",
            f"| Mean change in share | {occlusion['mean_share_change_pts']:+.2f} pts | n/a |",
            "",
            "Images where covering the hottest region lowered its attention, by category. Misses cluster in "
            "texture-heavy categories (noise, patterns, natural scenes) where a flat patch is itself an "
            "eye-catching anomaly and there is no distinct element to hide.",
            "",
            "| Category | Dropped |",
            "|---|---|",
        ]
        ordered = sorted(occlusion["per_category"].items(), key=lambda item: (-item[1]["dropped"], item[0]))
        for category, stats in ordered:
            lines.append(f"| {category} | {stats['dropped']}/{stats['n']} |")
    lines += [
        "",
        "## Caveat",
        "",
        "CAT2000 is general scenes, not ads. These results show the predictions agree with real human eye "
        "tracking; they do not show that a higher score sells more product.",
        "",
    ]
    path.write_text("\n".join(lines))


def analyze(args):
    rows = read_jsonl(PREDICTIONS)
    if not rows:
        raise SystemExit("no cached predictions; run `predict` first")
    results = {
        "dataset": "CAT2000 train (public, 18 observers)",
        "device": args.device,
        "benchmark": benchmark_test(rows),
        "attention_share": attention_share_test(rows),
        "occlusion": occlusion_summary(read_jsonl(OCCLUSIONS)),
    }
    DOCS.mkdir(exist_ok=True)
    (DOCS / "large_scale_eval.json").write_text(json.dumps(results, indent=2) + "\n")
    write_summary(results, DOCS / "large_scale_summary.md")
    write_chart(results, DOCS / "large_scale_chart.png")
    print((DOCS / "large_scale_summary.md").read_text())
    print("wrote docs/large_scale_eval.json, docs/large_scale_summary.md, docs/large_scale_chart.png")
    return 0


# ---------------------------------------------------------------- occlusion


def occlusion(args):
    from backend.metrics import compute_attention_multiplier, compute_attention_share
    from validate_occlusion import hottest_and_coldest_box, occlude

    cached = {row["image"] for row in read_jsonl(PREDICTIONS)}
    pairs = [pair for pair in sample_images(OCCLUSION_PER_CATEGORY) if pair["name"] in cached]
    done = {row["image"] for row in read_jsonl(OCCLUSIONS)}
    todo = [pair for pair in pairs if pair["name"] not in done]
    print(f"{len(pairs)} images in the occlusion subset, {len(todo)} to run", flush=True)
    if not todo:
        return analyze(args)

    model = load_model(args.device)
    started = time.time()

    def small_log(frame):
        return np.log(np.maximum(downsample(np.exp(model.predict(frame).astype(np.float64))), 1e-30))

    for index, pair in enumerate(todo, start=1):
        image = Image.open(pair["image"]).convert("RGB")
        before_log = np.log(np.maximum(np.load(cache_path(pair["name"]))["model"].astype(np.float64), 1e-30))
        hot, cold = hottest_and_coldest_box(np.exp(before_log))
        full_hot = tuple(value * DOWNSAMPLE for value in hot)
        full_cold = tuple(value * DOWNSAMPLE for value in cold)

        after_log = small_log(np.array(occlude(image, full_hot)))
        control_log = small_log(np.array(occlude(image, full_cold)))

        row = {
            "image": pair["name"],
            "category": pair["category"],
            "hot_box": list(full_hot),
            "multiplier_before": compute_attention_multiplier(before_log, hot),
            "multiplier_after": compute_attention_multiplier(after_log, hot),
            "share_before": compute_attention_share(before_log, hot),
            "share_after": compute_attention_share(after_log, hot),
            "control_multiplier": compute_attention_multiplier(control_log, hot),
        }
        append_jsonl(OCCLUSIONS, row)
        elapsed = time.time() - started
        print(
            f"[{index}/{len(todo)}] {pair['name']:<28} {row['multiplier_before']:6.2f}x -> "
            f"{row['multiplier_after']:6.2f}x  control {row['control_multiplier']:6.2f}x  "
            f"eta {format_duration(elapsed / index * (len(todo) - index))}",
            flush=True,
        )
    return analyze(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["predict", "analyze", "occlusion"])
    parser.add_argument("--device", default="cpu", help="cpu or mps")
    parser.add_argument("--per-category", type=int, default=PER_CATEGORY)
    parser.add_argument("--limit", type=int, default=0, help="only the first N selected images (smoke test)")
    args = parser.parse_args()
    return {"predict": predict, "analyze": analyze, "occlusion": occlusion}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
