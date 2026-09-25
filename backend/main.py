import base64
import io
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from backend.heatmap import make_overlay
from backend.metrics import compute_on_target_salience, compute_prominence
from backend.model import SaliencyModel

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


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    x0: int = Form(...),
    y0: int = Form(...),
    x1: int = Form(...),
    y1: int = Form(...),
):
    try:
        pil_image = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="uploaded file is not a readable image")

    frame = np.array(pil_image)
    log_density = saliency.predict(frame)
    box = (x0, y0, x1, y1)

    try:
        prominence = compute_prominence(log_density, box)
        on_target_salience = compute_on_target_salience(log_density, box)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    overlay = make_overlay(log_density, frame)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode overlay as PNG")

    return {
        "overlay_png_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
        "prominence": prominence,
        "on_target_salience": on_target_salience,
    }
