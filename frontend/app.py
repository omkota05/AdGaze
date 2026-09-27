import base64
import io

import requests
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import boxes_to_drawing, st_canvas

PREDICT_URL = "http://127.0.0.1:8000/predict"
DISPLAY_WIDTH = 520
IMAGE_TYPES = ["jpg", "jpeg", "png"]
BOX_STROKE = "#00FF00"
BOX_FILL = "rgba(0, 255, 0, 0.15)"

st.set_page_config(page_title="AdGaze", layout="wide")
st.title("AdGaze")
st.caption("Predict where viewers look on an ad, score a specific element, and compare two versions.")


@st.cache_data(show_spinner=False)
def analyze(image_bytes, style, box):
    """POST one image to the backend. Cached so repeat runs skip the round trip."""
    response = requests.post(
        PREDICT_URL,
        params={"style": style},
        files={"image": ("upload", image_bytes, "image/jpeg")},
        data=dict(zip(("x0", "y0", "x1", "y1"), box)) if box else {},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def clamp_box(box, image):
    """Round a box to ints, keep it inside the image, and reject empty ones."""
    x0, y0, x1, y1 = (round(value) for value in box)
    x0, x1 = sorted((max(x0, 0), min(x1, image.width)))
    y0, y1 = sorted((max(y0, 0), min(y1, image.height)))

    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return x0, y0, x1, y1


def display_size_for(image):
    """Return the canvas size for an image, plus the scale that produced it."""
    scale = min(1.0, DISPLAY_WIDTH / image.width)
    return (round(image.width * scale), round(image.height * scale)), scale


def draw_box(image, key):
    """Show a canvas over the image and return the last box in original pixels."""
    size, _ = display_size_for(image)

    canvas = st_canvas(
        background_image=image,
        background_image_fit="contain",
        drawing_mode="labeled_rect",
        stroke_width=2,
        stroke_color=BOX_STROKE,
        fill_color=BOX_FILL,
        width=size[0],
        height=size[1],
        key=key,
    )

    boxes = canvas.boxes_in_image_space if canvas else None
    if not boxes:
        return None

    last = boxes[-1]
    return clamp_box(
        (last["left"], last["top"], last["left"] + last["width"], last["top"] + last["height"]),
        image,
    )


def show_box_preview(image, box, key):
    """Show the image with a read-only copy of another image's box drawn on it."""
    size, scale = display_size_for(image)

    drawing = None
    if box:
        x0, y0, x1, y1 = box
        drawing = boxes_to_drawing(
            [
                {
                    "left": x0 * scale,
                    "top": y0 * scale,
                    "width": (x1 - x0) * scale,
                    "height": (y1 - y0) * scale,
                }
            ],
            stroke_color=BOX_STROKE,
            stroke_width=2,
            fill_color=BOX_FILL,
        )

    st_canvas(
        background_image=image,
        background_image_fit="contain",
        drawing_mode="labeled_rect",
        initial_drawing=drawing,
        disabled=True,
        width=size[0],
        height=size[1],
        key=key,
    )


def show_result(title, payload, box, baseline=None):
    """Render one overlay plus its metrics, with deltas when a baseline is given."""
    st.markdown(f"**{title}**")
    st.image(base64.b64decode(payload["overlay_png_base64"]), width="stretch")

    if box is None:
        st.caption("No element selected, so no scores -- draw a box to get the attention multiplier and attention share.")
        return

    multiplier, share = payload["attention_multiplier"], payload["attention_share"]
    multiplier_delta = share_delta = None
    if baseline is not None:
        multiplier_delta = f"{multiplier - baseline['attention_multiplier']:+.2f}x"
        share_delta = f"{(share - baseline['attention_share']) * 100:+.1f} pts"

    left, right = st.columns(2)
    left.metric(
        "Attention Multiplier",
        f"{multiplier:.2f}x",
        delta=multiplier_delta,
        help="Attention per pixel vs. this image's average. Above 1.0x beats average.",
    )
    right.metric(
        "Attention Share",
        f"{share:.1%}",
        delta=share_delta,
        help="Share of this image's total attention landing in the box.",
    )


upload_columns = st.columns(2)
with upload_columns[0]:
    upload_a = st.file_uploader("Image A", type=IMAGE_TYPES, key="upload_a")
with upload_columns[1]:
    upload_b = st.file_uploader("Image B (optional, to compare)", type=IMAGE_TYPES, key="upload_b")

if upload_a is None:
    st.info("Upload an image to get started. Add a second one to compare two versions side by side.")
    st.stop()

bytes_a = upload_a.getvalue()
image_a = Image.open(io.BytesIO(bytes_a)).convert("RGB")

bytes_b = image_b = None
if upload_b is not None:
    bytes_b = upload_b.getvalue()
    image_b = Image.open(io.BytesIO(bytes_b)).convert("RGB")

if image_b is not None and image_a.size != image_b.size:
    st.warning(
        f"Image A is {image_a.width}x{image_a.height} but image B is {image_b.width}x{image_b.height}. "
        "The scores will not be comparable: the same box covers a different share of each image, and the "
        "model's output shifts with resolution. Re-export the edited copy at image A's exact size."
    )

style = st.radio("Overlay style", ["glow", "jet"], horizontal=True)

share_box = False
if image_b is not None:
    share_box = st.checkbox(
        "Score the same box on both images",
        value=True,
        help="Keeps the comparison honest -- a different box size changes the scores on its own.",
    )

st.subheader("Draw a box around the element you want to score")
st.caption("Optional -- skip it for the heatmap alone. Only the last box on a canvas is used.")

if image_b is None:
    box_a = draw_box(image_a, "box_a")
    box_b = None
else:
    canvas_columns = st.columns(2)
    with canvas_columns[0]:
        st.caption("Image A")
        box_a = draw_box(image_a, "box_a")
    with canvas_columns[1]:
        if share_box:
            st.caption("Image B -- scored with image A's box")
            box_b = clamp_box(box_a, image_b) if box_a else None
            show_box_preview(image_b, box_b, "box_b_preview")
            if box_a and box_b is None:
                st.warning("Image A's box falls outside image B. Uncheck the box option to draw one here.")
        else:
            st.caption("Image B")
            box_b = draw_box(image_b, "box_b")

if box_a:
    st.write(f"Box on image A: `{box_a}`" + (f" -- on image B: `{box_b}`" if box_b else ""))

if st.button("Analyze", type="primary"):
    try:
        with st.spinner("Scoring image A..."):
            result_a = analyze(bytes_a, style, box_a)
        result_b = None
        if image_b is not None:
            with st.spinner("Scoring image B..."):
                result_b = analyze(bytes_b, style, box_b)
    except requests.ConnectionError:
        st.error(
            f"Could not reach the backend at {PREDICT_URL}. Make sure the FastAPI server "
            "is running:\n\n`venv/bin/uvicorn backend.main:app --reload`"
        )
        st.stop()
    except requests.Timeout:
        st.error("The backend took too long to respond. It may still be loading the model.")
        st.stop()
    except requests.HTTPError as error:
        st.error(f"Backend returned {error.response.status_code}: {error.response.text}")
        st.stop()

    if result_b is None:
        show_result("Result", result_a, box_a)
    else:
        result_columns = st.columns(2)
        with result_columns[0]:
            show_result("Image A (before)", result_a, box_a)
        with result_columns[1]:
            comparable = box_a is not None and box_b is not None
            show_result("Image B (after)", result_b, box_b, baseline=result_a if comparable else None)

        if box_a and box_b and box_a != box_b:
            st.caption(
                "The two boxes differ, so part of any delta comes from the box itself, not the creative."
            )
