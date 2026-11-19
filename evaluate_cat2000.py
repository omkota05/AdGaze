"""Score DeepGaze IIE on the public CAT2000 train split.

The official CAT2000 *test* fixations are held out by the MIT/Tuebingen
benchmark, so nobody can reproduce that leaderboard locally. The public
train set (2000 images, 18 observers) is the standard stand-in: same
categories, same image size, published DeepGaze IIE test numbers to
compare against.

    venv/bin/python evaluate_cat2000.py              # 10 images x 20 categories
    venv/bin/python evaluate_cat2000.py --per-category 5
    venv/bin/python evaluate_cat2000.py --full

Needs data/CAT2000/trainSet extracted (see the download in this script).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from backend.benchmark import auc_judd, cc, nss

DATA_DIR = Path("data/CAT2000")
ZIP_PATH = DATA_DIR / "trainSet.zip"
EXTRACT_DIR = DATA_DIR / "trainSet"
ZIP_URL = "http://saliency.mit.edu/trainSet.zip"
ZIP_MD5 = "56ad5c77e6c8f72ed9ef2901628d6e48"

# DeepGaze IIE on the official CAT2000 *test* set (MIT/Tuebingen leaderboard).
# Our run is the public train split, so these are a reference, not a target to hit exactly.
PUBLISHED = {"auc": 0.8692, "nss": 2.1122, "cc": 0.8189}


def md5(path):
    import hashlib

    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dataset():
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.rglob("*.jpg")):
        return EXTRACT_DIR

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print(f"downloading {ZIP_URL} (~660 MB)...", flush=True)
        import urllib.request

        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)

    digest = md5(ZIP_PATH)
    if digest != ZIP_MD5:
        raise SystemExit(f"trainSet.zip md5 {digest} != {ZIP_MD5}")

    print("extracting trainSet.zip...", flush=True)
    import zipfile

    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(DATA_DIR)
    return EXTRACT_DIR


def find_roots(extracted):
    """Return (stimuli_dir, loc_dir, map_dir) under the extracted tree."""
    stimuli = next(extracted.rglob("Stimuli"), None)
    if stimuli is None or not stimuli.is_dir():
        raise SystemExit(f"no Stimuli/ folder under {extracted}")
    locations = next(
        (path for name in ("FIXATIONLOCS", "FIXATIONLOC", "FixationLocs")
         if (path := stimuli.parent / name).is_dir()),
        None,
    )
    maps = next(
        (path for name in ("FIXATIONMAPS", "FIXATIONMAP", "FixationMaps")
         if (path := stimuli.parent / name).is_dir()),
        None,
    )
    if locations is None:
        raise SystemExit(f"no FIXATIONLOCS/ folder next to {stimuli}")
    return stimuli, locations, maps


def list_pairs(stimuli, locations, maps):
    pairs = []
    for image_path in sorted(stimuli.rglob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if "output" in image_path.parts:
            continue
        relative = image_path.relative_to(stimuli)
        loc = _sibling(locations, relative)
        fmap = _sibling(maps, relative) if maps else None
        if loc is None:
            continue
        pairs.append(
            {
                "category": relative.parts[0] if len(relative.parts) > 1 else "uncategorized",
                "name": relative.as_posix(),
                "image": image_path,
                "locations": loc,
                "fixation_map": fmap,
            }
        )
    if not pairs:
        raise SystemExit("found images but no matching FIXATIONLOCS files")
    return pairs


def _sibling(root, relative):
    stem = relative.with_suffix("")
    for suffix in (".png", ".jpg", ".jpeg", ".bmp", ".mat"):
        candidate = root / stem.with_suffix(suffix)
        if candidate.exists():
            return candidate
    matches = list(root.glob(str(stem) + ".*"))
    return matches[0] if matches else None


def load_binary_map(path, shape):
    """Load fixation locations as a 2D 0/1 array matching `shape` (H, W)."""
    if path.suffix.lower() == ".mat":
        from scipy.io import loadmat

        payload = loadmat(path)
        array = next(
            np.asarray(value)
            for key, value in payload.items()
            if not key.startswith("_") and np.asarray(value).ndim == 2
        )
    else:
        array = np.array(Image.open(path).convert("L"))
    if array.shape != shape:
        import cv2

        array = cv2.resize(array.astype(np.float64), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (array > 0).astype(np.float64)


def load_density_map(path, shape):
    array = np.array(Image.open(path).convert("L"), dtype=np.float64)
    if array.shape != shape:
        import cv2

        array = cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return array


def subsample(pairs, per_category, seed):
    by_category = defaultdict(list)
    for pair in pairs:
        by_category[pair["category"]].append(pair)
    rng = random.Random(seed)
    chosen = []
    for category in sorted(by_category):
        bucket = by_category[category]
        rng.shuffle(bucket)
        chosen.extend(bucket[:per_category])
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-category", type=int, default=10, help="images kept from each of the 20 categories")
    parser.add_argument("--full", action="store_true", help="score all 2000 train images")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    extracted = ensure_dataset()
    stimuli, locations, maps = find_roots(extracted)
    pairs = list_pairs(stimuli, locations, maps)
    if not args.full:
        pairs = subsample(pairs, args.per_category, args.seed)
    print(f"scoring {len(pairs)} images across {len({p['category'] for p in pairs})} categories\n", flush=True)

    from backend.model import SaliencyModel

    print("loading DeepGaze IIE...", flush=True)
    model = SaliencyModel()

    rows = []
    print(f"{'category':<18} {'image':<28} {'AUC':>6} {'NSS':>6} {'CC':>6}")
    print("-" * 68)
    for pair in pairs:
        frame = np.array(Image.open(pair["image"]).convert("RGB"))
        log_density = model.predict(frame)
        density = np.exp(log_density)
        height, width = density.shape
        fixations = load_binary_map(pair["locations"], (height, width))
        try:
            image_nss = nss(density, fixations)
            image_auc = auc_judd(density, fixations)
        except ValueError:
            print(f"skip {pair['name']}: empty fixation map", flush=True)
            continue
        image_cc = None
        if pair["fixation_map"] is not None:
            image_cc = cc(density, load_density_map(pair["fixation_map"], (height, width)))
        row = {
            "category": pair["category"],
            "image": pair["name"],
            "auc": image_auc,
            "nss": image_nss,
            "cc": image_cc,
        }
        rows.append(row)
        cc_text = f"{image_cc:6.3f}" if image_cc is not None else "   n/a"
        print(
            f"{pair['category']:<18} {Path(pair['name']).name:<28} "
            f"{image_auc:6.3f} {image_nss:6.2f} {cc_text}",
            flush=True,
        )

    summary = {
        "n": len(rows),
        "auc": float(np.mean([row["auc"] for row in rows])),
        "nss": float(np.mean([row["nss"] for row in rows])),
        "cc": float(np.mean([row["cc"] for row in rows if row["cc"] is not None])),
        "published_cat2000_test": PUBLISHED,
        "split": "CAT2000 train (18 observers, public)",
        "per_category": None if args.full else args.per_category,
    }
    print("\nsummary")
    print(f"  n   = {summary['n']}")
    print(f"  AUC = {summary['auc']:.4f}   published test {PUBLISHED['auc']}")
    print(f"  NSS = {summary['nss']:.4f}   published test {PUBLISHED['nss']}")
    print(f"  CC  = {summary['cc']:.4f}   published test {PUBLISHED['cc']}")

    out = Path("docs/cat2000_eval.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
