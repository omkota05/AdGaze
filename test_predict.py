"""Smoke test for the running /predict endpoint.

Start the server first:
    uvicorn backend.main:app --reload

Then:
    python test_predict.py [image_path] [x0 y0 x1 y1]
"""

import argparse
import base64
import subprocess
import sys

import requests

URL = "http://127.0.0.1:8000/predict"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", default="test_image.jpg")
    parser.add_argument("box", nargs="*", type=int, default=[100, 100, 300, 300])
    parser.add_argument("--style", choices=["jet", "glow"], default="jet", help="overlay style")
    parser.add_argument("--no-open", action="store_true", help="skip opening the saved overlay")
    args = parser.parse_args()
    if len(args.box) != 4:
        parser.error(f"box needs 4 integers (x0 y0 x1 y1), got {len(args.box)}")
    return args


def main():
    args = parse_args()
    x0, y0, x1, y1 = args.box

    try:
        with open(args.image, "rb") as handle:
            response = requests.post(
                URL,
                params={"style": args.style},
                files={"image": (args.image, handle, "image/jpeg")},
                data={"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                timeout=300,
            )
    except FileNotFoundError:
        sys.exit(f"image not found: {args.image}")
    except requests.ConnectionError:
        sys.exit(
            f"could not reach {URL} -- start the server first:\n"
            "    uvicorn backend.main:app --reload"
        )
    except requests.Timeout:
        sys.exit("request timed out; the model may still be loading, try again in a moment")

    if not response.ok:
        sys.exit(f"request failed with {response.status_code}: {response.text}")

    payload = response.json()
    missing = {"overlay_png_base64", "prominence", "on_target_salience"} - payload.keys()
    if missing:
        sys.exit(f"response is missing fields: {sorted(missing)}")

    print(f"style:               {args.style}")
    print(f"box:                 ({x0}, {y0}, {x1}, {y1})")
    print(f"prominence:          {payload['prominence']:.4f}")
    print(f"on_target_salience:  {payload['on_target_salience']:.4f}")
    print(f"background salience: {1.0 - payload['on_target_salience']:.4f}")

    overlay = base64.b64decode(payload["overlay_png_base64"])
    if not overlay.startswith(b"\x89PNG"):
        sys.exit("decoded overlay is not a PNG")

    output_path = f"overlay_output_{args.style}.png"
    with open(output_path, "wb") as handle:
        handle.write(overlay)
    print(f"saved {output_path} ({len(overlay) / 1024:.0f} KB)")

    if not args.no_open and sys.platform == "darwin":
        subprocess.run(["open", output_path], check=False)


if __name__ == "__main__":
    main()
