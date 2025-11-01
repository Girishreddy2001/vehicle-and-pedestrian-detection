import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io


# compatibility helper for measuring text across Pillow versions
def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont = None):
    """Return (width, height) of text using available APIs across Pillow versions."""
    # Pillow 11.3.0: prefer ImageDraw.textbbox which gives precise bbox
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])
    except Exception:
        pass

    # fallback to ImageFont.getbbox (also available in newer Pillow)
    try:
        if font is not None:
            bbox = font.getbbox(text)
            return (bbox[2] - bbox[0], bbox[3] - bbox[1])
    except Exception:
        pass

    # last resorts
    try:
        return draw.textsize(text, font=font)
    except Exception:
        try:
            if font is not None:
                return font.getsize(text)
        except Exception:
            return (0, 0)


# Config
UI_BACKEND_URL = "http://ui_backend_container:8500/upload"

st.set_page_config(page_title="YOLOv8 Detection UI", layout="wide")
st.title("YOLOv8 — Upload image and draw detections")

st.markdown("""
Upload a picture, and the app will analyze it to find what is in it — like people or vehicles. You will see boxes drawn around the detected objects and a short summary of what was found.
""")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # load full-resolution image for annotation but display centered at half-window (middle column)
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded image", use_column_width=True)
        run = st.button("Run detection")

    if run:
        with st.spinner("Sending image to backend and waiting for detections..."):
            try:
                # Prepare file payload
                uploaded_file.seek(0)
                files = {
                    "file": (uploaded_file.name, uploaded_file.read(), "image/jpeg")
                }
                resp = requests.post(UI_BACKEND_URL, files=files, timeout=30)

                if resp.status_code != 200:
                    st.error(f"Backend error ({resp.status_code}): {resp.text}")
                else:
                    data = resp.json()
                    detections = data.get("detections", [])
                    summary = data.get("summary", {})

                    # draw boxes on a copy of the uploaded image
                    annotated = image.copy()
                    draw = ImageDraw.Draw(annotated)
                    try:
                        font = ImageFont.load_default()
                    except Exception:
                        font = None

                    for d in detections:
                        bbox = d.get("bbox")
                        label = d.get("label", "")
                        conf = d.get("confidence", 0.0)
                        if bbox and len(bbox) == 4:
                            x1, y1, x2, y2 = map(int, bbox)
                            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                            text = f"{label} {conf:.2f}"
                            text_w, text_h = _text_size(draw, text, font=font)
                            # choose whether to place label above or below box depending on space
                            text_y = y1 - text_h - 4
                            if text_y < 0:
                                # not enough room above, put label below the box
                                text_y = y1 + 4

                            text_bg = [x1, text_y, x1 + text_w + 4, text_y + text_h + 4]
                            draw.rectangle(text_bg, fill="red")
                            draw.text(
                                (x1 + 2, text_y + 2), text, fill="white", font=font
                            )

                    with col2:
                        st.subheader("Detections")
                        st.image(annotated, use_column_width=True)

                        st.subheader("Summary")
                        people = summary.get("people", 0)
                        vehicles = summary.get("vehicles", 0)
                        st.write(f"People: {people}")
                        st.write(f"Vehicles: {vehicles}")

            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

else:
    st.info("Upload an image to begin.")
