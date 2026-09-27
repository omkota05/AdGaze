import base64
import io

import requests
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

PREDICT_URL = "http://127.0.0.1:8000/predict"
DISPLAY_WIDTH = 700

st.set_page_config(page_title="AdGaze", layout="centered")
st.title("AdGaze")
st.caption("Predict where viewers look on an ad, and score a specific element.")

uploaded = st.file_uploader("Ad image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload a jpg or png to get started.")
    st.stop()

image_bytes = uploaded.getvalue()
image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

scale = min(1.0, DISPLAY_WIDTH / image.width)
display_size = (round(image.width * scale), round(image.height * scale))

st.subheader("Draw a box around the element you want to score")
st.caption("Optional -- skip it to get the heatmap alone. Only the last box you draw is used.")

canvas = st_canvas(
    background_image=image,
    background_image_fit="contain",
    drawing_mode="labeled_rect",
    stroke_width=2,
    stroke_color="#00FF00",
    fill_color="rgba(0, 255, 0, 0.15)",
    width=display_size[0],
    height=display_size[1],
    key="box",
)


def box_from_canvas(canvas_result):
    """Return the last drawn rectangle in original-image pixels, or None."""
    boxes = canvas_result.boxes_in_image_space if canvas_result else None
    if not boxes:
        return None

    last = boxes[-1]
    x0, y0 = round(last["left"]), round(last["top"])
    x1, y1 = round(last["left"] + last["width"]), round(last["top"] + last["height"])

    x0, x1 = sorted((max(x0, 0), min(x1, image.width)))
    y0, y1 = sorted((max(y0, 0), min(y1, image.height)))

    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return x0, y0, x1, y1


box = box_from_canvas(canvas)
if box:
    st.write(f"Selected box in original pixels: `{box}`")

style = st.radio("Overlay style", ["glow", "jet"], horizontal=True)

if st.button("Analyze", type="primary"):
    data = dict(zip(("x0", "y0", "x1", "y1"), box)) if box else {}

    with st.spinner("Running the saliency model..."):
        try:
            response = requests.post(
                PREDICT_URL,
                params={"style": style},
                files={"image": (uploaded.name, image_bytes, uploaded.type or "image/jpeg")},
                data=data,
                timeout=300,
            )
        except requests.ConnectionError:
            st.error(
                f"Could not reach the backend at {PREDICT_URL}. Make sure the FastAPI server "
                "is running:\n\n`venv/bin/uvicorn backend.main:app --reload`"
            )
            st.stop()
        except requests.Timeout:
            st.error("The backend took too long to respond. It may still be loading the model.")
            st.stop()

    if not response.ok:
        st.error(f"Backend returned {response.status_code}: {response.text}")
        st.stop()

    payload = response.json()
    st.image(base64.b64decode(payload["overlay_png_base64"]), caption=f"{style} overlay")

    if box:
        prominence, on_target = payload["prominence"], payload["on_target_salience"]
        left, right = st.columns(2)
        left.metric("Prominence", f"{prominence:.2f}x", help="Attention per pixel vs. the image average")
        right.metric("On-target salience", f"{on_target:.1%}", help="Share of the image's total attention")
        st.caption(f"{1 - on_target:.1%} of attention went to the rest of the image.")
    else:
        st.caption("No element selected, so no scores -- draw a box to get prominence and on-target salience.")
