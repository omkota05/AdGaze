import base64
import io
from contextlib import asynccontextmanager
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from backend.heatmap import make_glow_overlay, make_overlay
from backend.metrics import compute_on_target_salience, compute_prominence
from backend.model import SaliencyModel

OVERLAY_STYLES = {"jet": make_overlay, "glow": make_glow_overlay}
Style = Literal["jet", "glow"]

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
    try:
        pil_image = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="uploaded file is not a readable image")
    return np.array(pil_image)


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
    frame = await read_frame(image)
    log_density = saliency.predict(frame)

    box = (x0, y0, x1, y1)
    prominence = on_target_salience = None
    if all(coordinate is not None for coordinate in box):
        try:
            prominence = compute_prominence(log_density, box)
            on_target_salience = compute_on_target_salience(log_density, box)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))
    elif any(coordinate is not None for coordinate in box):
        raise HTTPException(status_code=400, detail="give all four of x0, y0, x1, y1 or none of them")

    return {
        "overlay_png_base64": base64.b64encode(encode_overlay(log_density, frame, style)).decode("ascii"),
        "prominence": prominence,
        "on_target_salience": on_target_salience,
    }


@app.post("/overlay", response_class=Response, responses={200: {"content": {"image/png": {}}}})
async def overlay(image: UploadFile = File(...), style: Style = "jet"):
    frame = await read_frame(image)
    log_density = saliency.predict(frame)
    return Response(content=encode_overlay(log_density, frame, style), media_type="image/png")
