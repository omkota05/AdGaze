import base64
import hashlib
import io
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from backend.heatmap import make_glow_overlay, make_overlay
from backend.metrics import compute_attention_multiplier, compute_attention_share
from backend.model import SaliencyModel

OVERLAY_STYLES = {"jet": make_overlay, "glow": make_glow_overlay}
Style = Literal["jet", "glow"]

DENSITY_CACHE_SIZE = 8
_density_cache = OrderedDict()

saliency = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global saliency
    saliency = SaliencyModel()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


async def read_frame(image: UploadFile):
    """Return the upload as (RGB array, raw bytes); the bytes key the density cache."""
    raw = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="uploaded file is not a readable image")
    return np.array(pil_image), raw


def density_for(raw, frame):
    """Reuse the model output for an image already scored, so re-boxing is cheap."""
    key = hashlib.sha256(raw).hexdigest()

    if key in _density_cache:
        _density_cache.move_to_end(key)
        return _density_cache[key]

    log_density = saliency.predict(frame)
    _density_cache[key] = log_density
    if len(_density_cache) > DENSITY_CACHE_SIZE:
        _density_cache.popitem(last=False)
    return log_density


def encode_overlay(log_density, frame, style="jet"):
    overlay = OVERLAY_STYLES[style](log_density, frame)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode overlay as PNG")
    return encoded.tobytes()


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    x0: int | None = Form(None),
    y0: int | None = Form(None),
    x1: int | None = Form(None),
    y1: int | None = Form(None),
    style: Style = "jet",
):
    frame, raw = await read_frame(image)
    log_density = density_for(raw, frame)

    box = (x0, y0, x1, y1)
    attention_multiplier = attention_share = None
    if all(coordinate is not None for coordinate in box):
        try:
            attention_multiplier = compute_attention_multiplier(log_density, box)
            attention_share = compute_attention_share(log_density, box)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))
    elif any(coordinate is not None for coordinate in box):
        raise HTTPException(status_code=400, detail="give all four of x0, y0, x1, y1 or none of them")

    return {
        "overlay_png_base64": base64.b64encode(encode_overlay(log_density, frame, style)).decode("ascii"),
        "attention_multiplier": attention_multiplier,
        "attention_share": attention_share,
    }


@app.post("/overlay", response_class=Response, responses={200: {"content": {"image/png": {}}}})
async def overlay(image: UploadFile = File(...), style: Style = "jet"):
    frame, raw = await read_frame(image)
    log_density = density_for(raw, frame)
    return Response(content=encode_overlay(log_density, frame, style), media_type="image/png")
