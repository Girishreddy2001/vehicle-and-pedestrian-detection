from fastapi import FastAPI, File, UploadFile, Request, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from ultralytics import YOLO
from contextlib import asynccontextmanager
import torch
import os
import io
from PIL import Image
import uvicorn
import time
import logging
import json
import uuid
from pathlib import Path
from PIL import ImageDraw, ImageFont
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

# basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ai_backend")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL = None

UI_BACKEND_ORIGIN = os.getenv("UI_BACKEND_ORIGIN", "http://localhost:8500")
API_KEY = os.getenv("AI_BACKEND_API_KEY", "")
# Split multiple origins if comma-separated
ALLOWED_ORIGINS = [origin.strip() for origin in UI_BACKEND_ORIGIN.split(",")]

# limits
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", 10 * 1024 * 1024))  # 10 MB default


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Code to run on STARTUP ---
    global MODEL
    logger.info("Lifespan event: startup")
    try:
        logger.info("Loading model ")

        # --- model loading logic ---
        try:
            MODEL = YOLO("yolov8n.pt")
        except Exception:
            pass
        # --- End of model loading ---

        logger.info("Model loaded successfully")
    except Exception as e:
        logger.exception("Failed to load model on startup: %s", e)

    yield

    # --- Code to run on SHUTDOWN ---
    logger.info("Lifespan event: shutdown")
    MODEL = None
    logger.info("Model resources released")


app = FastAPI(
    title="AI Backend - Vehicle & Pedestrian Detection API", lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "YOLOv5 Vehicle & Pedestrian Detection API is running"}


# simple API key verifier for server-to-server requests
async def verify_api_key(request: Request):
    if not API_KEY:
        raise HTTPException(status_code=401, detail="API key not configured")
    incoming = request.headers.get("x-api-key")
    if incoming != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True


@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...), _ok: bool = Depends(verify_api_key)
):
    try:
        # Read and convert image
        image_bytes = await file.read()

        if len(image_bytes) > MAX_UPLOAD_SIZE:
            return JSONResponse({"error": "file too large"}, status_code=413)

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if MODEL is None:
            return JSONResponse({"error": "model not loaded"}, status_code=503)

        # Running inference in a threadpool to avoid blocking the event loop
        def _run(img):
            with torch.no_grad():
                return MODEL(img)

        start = time.time()
        results = await run_in_threadpool(_run, img)
        elapsed = time.time() - start

        # Normalize to a single Results object for consistent processing.
        if isinstance(results, list) and len(results) > 0:
            res = results[0]
        else:
            res = results

        # Extract detections as a pandas DataFrame when available; fall back to boxes attributes otherwise.
        output = []
        try:
            detections = res.pandas().xyxy[0]
            for _, row in detections.iterrows():
                label = row["name"]
                if label in ["car", "bus", "truck", "motorbike", "bicycle", "person"]:
                    output.append(
                        {
                            "label": label,
                            "confidence": float(row["confidence"]),
                            "bbox": [
                                float(row["xmin"]),
                                float(row["ymin"]),
                                float(row["xmax"]),
                                float(row["ymax"]),
                            ],
                        }
                    )
        except Exception:
            # Fallback: iterate over res.boxes if pandas() isn't available or fails
            try:
                for box in getattr(res, "boxes", []):
                    # convert to Python types
                    xyxy = box.xyxy.tolist()[0] if hasattr(box, "xyxy") else None
                    conf = float(box.conf.tolist()[0]) if hasattr(box, "conf") else None
                    cls = int(box.cls.tolist()[0]) if hasattr(box, "cls") else None

                    # map class id to name if names available
                    name = None
                    if hasattr(res, "names") and cls is not None:
                        name = res.names.get(cls, str(cls))

                    if name in [
                        "car",
                        "bus",
                        "truck",
                        "motorbike",
                        "bicycle",
                        "person",
                    ]:
                        output.append(
                            {
                                "label": name,
                                "confidence": conf,
                                "bbox": [float(xy) for xy in xyxy]
                                if xyxy is not None
                                else [],
                            }
                        )
            except Exception:
                # If even fallback fails, leave output empty and report nothing detected
                output = []

        # Save annotated image + json file
        out_dir = Path(__file__).resolve().parent / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        annotated = img.copy()
        draw = ImageDraw.Draw(annotated)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        for det in output:
            bbox = det.get("bbox", [])
            if len(bbox) >= 4:
                xmin, ymin, xmax, ymax = [int(round(x)) for x in bbox[:4]]
                draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
                label_text = f"{det.get('label', '')} {det.get('confidence', 0):.2f}"
                text_origin = (xmin + 3, max(0, ymin - 12))
                if font:
                    draw.text(text_origin, label_text, fill="red", font=font)
                else:
                    draw.text(text_origin, label_text, fill="red")

        uid = uuid.uuid4().hex
        ts = int(time.time() * 1000)
        base_name = f"detection_{ts}_{uid}"
        image_filename = out_dir / f"{base_name}.jpg"
        json_filename = out_dir / f"{base_name}.json"

        try:
            annotated.save(image_filename, format="JPEG")
        except Exception as e:
            logger.exception("Failed saving annotated image: %s", e)

        payload = {
            "request_filename": getattr(file, "filename", None),
            "saved_image": str(image_filename),
            "detections": output,
            "inference_time_s": elapsed,
            "timestamp_ms": ts,
        }

        try:
            with open(json_filename, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.exception("Failed saving json: %s", e)

        return JSONResponse(
            {
                "detections": output,
                "inference_time_s": elapsed,
                "saved_image": str(image_filename),
                "saved_json": str(json_filename),
            },
            status_code=200,
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
