import base64
import io

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_drawable_canvas import boxes_to_drawing, st_canvas

PREDICT_URL = "http://127.0.0.1:8000/predict"
DISPLAY_WIDTH = 520
IMAGE_TYPES = ["jpg", "jpeg", "png"]
BOX_STROKE = "#C45C26"
BOX_FILL = "rgba(196, 92, 38, 0.14)"
EXPORT_WIDTH = 700
EXPORT_HEADER = 86
FONT_PATHS = ["/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"]

APP_CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap");

#MainMenu, header[data-testid="stHeader"], footer, .stAppDeployButton,
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }

.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: #efeae2;
}
[data-testid="stAppViewContainer"] > .main { background: #efeae2; }

.block-container {
    padding: 1.35rem 2rem 2.4rem !important;
    max-width: 1360px;
}

html, body, [class*="css"], .stMarkdown, .stCaption, p, label, span,
.stApp, [data-testid="stAppViewContainer"], [data-testid="stWidgetLabel"],
[data-testid="stFileUploader"], [data-testid="stFileUploader"] *,
[data-testid="stRadio"], [data-testid="stCheckbox"],
[data-testid="stMetricLabel"], [data-testid="stMetricValue"],
[data-testid="stMetricDelta"], [data-testid="stCaptionContainer"],
.stAlert, .stAlert * {
    color: #111111 !important;
}

html, body, [class*="css"], .stMarkdown, .stCaption, p, label, span {
    font-family: "IBM Plex Sans", "Avenir Next", sans-serif;
}

.ag-masthead {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 1.5rem;
    padding: 0 0 1.05rem;
    margin-bottom: 1.1rem;
    border-bottom: 1px solid #d4cdc0;
}
.ag-wordmark {
    font-family: "Source Serif 4", "Iowan Old Style", Georgia, serif;
    font-size: 2.05rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #1c1915;
    line-height: 1;
}
.ag-tag {
    margin-top: 0.35rem;
    color: #111111;
    font-size: 0.92rem;
    letter-spacing: 0.01em;
}
.ag-kicker {
    color: #111111;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}
.ag-panel-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #111111;
    margin: 0 0 0.65rem;
}
.ag-empty {
    border: 1px dashed #c8bfb0;
    background: #f7f3ec;
    padding: 1.4rem 1.5rem;
    color: #111111;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #f7f3ec;
    border: 1px solid #d4cdc0 !important;
    border-radius: 10px !important;
}

div[data-testid="stMetric"] {
    background: transparent;
    padding: 0.15rem 0.15rem 0.35rem;
}
div[data-testid="stMetric"] label { color: #111111 !important; }
div[data-testid="stMetricValue"] {
    font-family: "Source Serif 4", Georgia, serif;
    font-weight: 600;
    color: #111111 !important;
}

.stButton > button[kind="primary"] {
    background: #c45c26;
    border: 1px solid #a64b1d;
    color: #fffaf4;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.stButton > button[kind="primary"]:hover {
    background: #a64b1d;
    border-color: #8a3e18;
    color: #fffaf4;
}

[data-testid="stFileUploader"] { padding-top: 0.15rem; }
.stRadio > label, .stCheckbox > label { color: #111111 !important; }
.stButton > button { color: #111111 !important; }
.stButton > button[kind="primary"] { color: #111111 !important; }
.stDownloadButton > button { color: #111111 !important; }
</style>
"""

st.set_page_config(page_title="AdGaze", layout="wide", initial_sidebar_state="collapsed")
st.markdown(APP_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="ag-masthead">
      <div>
        <div class="ag-wordmark">AdGaze</div>
        <div class="ag-tag">Predict where viewers look, score one element, compare two versions.</div>
      </div>
      <div class="ag-kicker">DeepGaze IIE &nbsp;·&nbsp; attention analytics</div>
    </div>
    """,
    unsafe_allow_html=True,
)


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


def load_font(size):
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def export_panel(payload, box, title):
    """Render one labeled panel: the overlay with its box outlined and its metrics above."""
    image = Image.open(io.BytesIO(base64.b64decode(payload["overlay_png_base64"]))).convert("RGB")
    if box:
        draw_on_image = ImageDraw.Draw(image)
        draw_on_image.rectangle(box, outline=BOX_STROKE, width=max(2, round(image.width / 250)))

    image = image.resize(
        (EXPORT_WIDTH, round(image.height * EXPORT_WIDTH / image.width)), Image.LANCZOS
    )
    panel = Image.new("RGB", (EXPORT_WIDTH, image.height + EXPORT_HEADER), (247, 243, 236))
    panel.paste(image, (0, EXPORT_HEADER))

    if box:
        summary = (
            f"{payload['attention_multiplier']:.2f}x multiplier"
            f"    {payload['attention_share']:.1%} share"
        )
    else:
        summary = "no element scored"

    draw = ImageDraw.Draw(panel)
    draw.text((14, 12), title, fill="#1c1915", font=load_font(30))
    draw.text((14, 50), summary, fill="#5c564c", font=load_font(24))
    return panel


def build_comparison_png(result_a, box_a, result_b, box_b):
    """Compose the two panels into one shareable image."""
    panels = [
        export_panel(result_a, box_a, "BEFORE"),
        export_panel(result_b, box_b, "AFTER"),
    ]
    gap = 16
    sheet = Image.new(
        "RGB",
        (sum(panel.width for panel in panels) + gap, max(panel.height for panel in panels)),
        (239, 234, 226),
    )
    offset = 0
    for panel in panels:
        sheet.paste(panel, (offset, 0))
        offset += panel.width + gap

    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG")
    return buffer.getvalue()


def box_comparison_note(box_a, box_b):
    """Flag a box-area change, which distorts the delta; a pure move is valid methodology."""
    width_a, height_a = box_a[2] - box_a[0], box_a[3] - box_a[1]
    width_b, height_b = box_b[2] - box_b[0], box_b[3] - box_b[1]

    if (width_a, height_a) != (width_b, height_b):
        return "warning", (
            f"Image B's box is {width_b}x{height_b} but image A's is {width_a}x{height_a}, "
            f"a {(width_b * height_b) / (width_a * height_a):.2f}x change in area. Attention share scales "
            "with box area, so part of that delta is the box rather than the creative. Attention "
            "multiplier is area-normalized and stays meaningful."
        )
    if box_a != box_b:
        return "caption", (
            "The boxes are the same size in different places, which is the right setup for testing a "
            "moved element. Both scores stay comparable."
        )
    return None


def show_result(title, payload, box, baseline=None):
    """Render one overlay plus its metrics, with deltas when a baseline is given."""
    with st.container(border=True):
        st.markdown(f'<p class="ag-panel-label">{title}</p>', unsafe_allow_html=True)
        st.image(base64.b64decode(payload["overlay_png_base64"]), width="stretch")

        if box is None:
            st.caption(
                "No element selected, so no scores. Draw a box to get the attention "
                "multiplier and attention share."
            )
            return

        multiplier, share = payload["attention_multiplier"], payload["attention_share"]
        multiplier_delta = share_delta = None
        if baseline is not None:
            multiplier_delta = f"{multiplier - baseline['attention_multiplier']:+.2f}x"
            share_delta = f"{(share - baseline['attention_share']) * 100:+.1f} pts"

        with st.container(border=True):
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
    with st.container(border=True):
        st.markdown('<p class="ag-panel-label">Version A</p>', unsafe_allow_html=True)
        upload_a = st.file_uploader("Upload the original creative", type=IMAGE_TYPES, key="upload_a")
with upload_columns[1]:
    with st.container(border=True):
        st.markdown('<p class="ag-panel-label">Version B</p>', unsafe_allow_html=True)
        upload_b = st.file_uploader(
            "Optional second version for comparison", type=IMAGE_TYPES, key="upload_b"
        )

if upload_a is None:
    st.markdown(
        '<div class="ag-empty">Upload an image to start. Add a second one when you want '
        "a before/after comparison of the same ad.</div>",
        unsafe_allow_html=True,
    )
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

with st.container(border=True):
    control_columns = st.columns((1.1, 1.7, 1.0))
    with control_columns[0]:
        style = st.radio("Overlay style", ["glow", "jet"], horizontal=True)
    with control_columns[1]:
        share_box = False
        if image_b is not None:
            share_box = st.checkbox(
                "Score the same box on both images",
                value=True,
                help="Keeps the comparison honest. A different box size changes the scores on its own.",
            )
        else:
            st.caption("Draw a box on the image below, or skip it and Analyze for the heatmap alone.")
    with control_columns[2]:
        analyze_clicked = st.button("Analyze", type="primary", width="stretch")

st.markdown('<p class="ag-panel-label">Region to score</p>', unsafe_allow_html=True)
st.caption("Optional. Draw a rectangle around a logo, headline, or CTA. Only the last box is used.")

if image_b is None:
    with st.container(border=True):
        box_a = draw_box(image_a, "box_a")
        box_b = None
else:
    canvas_columns = st.columns(2)
    with canvas_columns[0]:
        with st.container(border=True):
            st.markdown('<p class="ag-panel-label">Version A</p>', unsafe_allow_html=True)
            box_a = draw_box(image_a, "box_a")
    with canvas_columns[1]:
        with st.container(border=True):
            if share_box:
                st.markdown(
                    '<p class="ag-panel-label">Version B · same box</p>', unsafe_allow_html=True
                )
                box_b = clamp_box(box_a, image_b) if box_a else None
                show_box_preview(image_b, box_b, "box_b_preview")
                if box_a and box_b is None:
                    st.warning(
                        "Image A's box falls outside image B. Uncheck the box option to draw one here."
                    )
            else:
                st.markdown('<p class="ag-panel-label">Version B</p>', unsafe_allow_html=True)
                box_b = draw_box(image_b, "box_b")

if box_a:
    st.caption(
        f"Box on A: `{box_a}`" + (f"  ·  on B: `{box_b}`" if box_b else "")
    )

if analyze_clicked:
    try:
        with st.spinner("Analyzing visual attention with DeepGaze IIE..."):
            result_a = analyze(bytes_a, style, box_a)
            result_b = None
            if image_b is not None:
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

    st.session_state.results = {
        "a": result_a,
        "b": result_b,
        "box_a": box_a,
        "box_b": box_b,
    }

results = st.session_state.get("results")
if results:
    result_a, result_b = results["a"], results["b"]
    scored_a, scored_b = results["box_a"], results["box_b"]

    st.markdown('<p class="ag-panel-label">Results</p>', unsafe_allow_html=True)

    if result_b is None:
        show_result("Overlay", result_a, scored_a)
        st.download_button(
            "Download overlay",
            data=base64.b64decode(result_a["overlay_png_base64"]),
            file_name="adgaze_overlay.png",
            mime="image/png",
        )
    else:
        result_columns = st.columns(2)
        with result_columns[0]:
            show_result("Version A · before", result_a, scored_a)
        with result_columns[1]:
            comparable = scored_a is not None and scored_b is not None
            show_result(
                "Version B · after", result_b, scored_b, baseline=result_a if comparable else None
            )

        if scored_a and scored_b:
            note = box_comparison_note(scored_a, scored_b)
            if note:
                level, text = note
                (st.warning if level == "warning" else st.caption)(text)

        st.download_button(
            "Download comparison",
            data=build_comparison_png(result_a, scored_a, result_b, scored_b),
            file_name="adgaze_comparison.png",
            mime="image/png",
            help="A side by side image with both overlays, boxes, and metrics.",
        )
