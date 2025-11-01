from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBasic
from fastapi.responses import Response
from fastapi.concurrency import run_in_threadpool
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import requests
import uuid
from pathlib import Path
import os
import uvicorn
import logging
from logging.handlers import RotatingFileHandler
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import aiofiles
from dotenv import load_dotenv
from typing import Optional


# Load environment variables
load_dotenv(override=True)

# Configuration
AI_BACKEND_URL = os.getenv("AI_BACKEND_URL", "http://localhost:8000/detect")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB default
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")
STREAMLIT_HOST = os.getenv("STREAMLIT_HOST", "localhost:8501")
TRUSTED_HOSTS = [
    "localhost:8501",
    "localhost:8500",
    "localhost",
    "127.0.0.1",
    "ui_frontend_container:8501",
    "ui_backend_container:8500",
    "ui_backend_container",
    "ui_frontend_container",
    "ai_backend_container:8000",
    "ai_backend_container",
]
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
CLEANUP_INTERVAL = int(os.getenv("CLEANUP_INTERVAL", "3600"))  # 1 hour
API_KEY = os.getenv("AI_BACKEND_API_KEY", "")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        RotatingFileHandler(
            "ui_backend.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ui_backend")

# Setup rate limiter
limiter = Limiter(key_func=get_remote_address)


class UploadException(Exception):
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code


async def cleanup_old_files():
    """Cleanup temporary files older than 1 hour"""
    while True:
        try:
            current_time = datetime.now().timestamp()
            for file_path in TEMP_DIR.glob("*"):
                if current_time - file_path.stat().st_mtime > 3600:  # 1 hour
                    try:
                        file_path.unlink()
                        logger.info(f"Cleaned up old file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error cleaning up {file_path}: {e}")
            await asyncio.sleep(CLEANUP_INTERVAL)
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(60)  # Wait a minute before retrying


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    cleanup_task = asyncio.create_task(cleanup_old_files())
    logger.info("Started temporary file cleanup task")

    yield

    # Shutdown
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    logger.info("Cleanup task stopped")


app = FastAPI(
    title="Vehicle & Pedestrian Detection UI Backend",
    version="1.0",
    lifespan=lifespan,
)

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)


# Add rate limiter error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=TRUSTED_HOSTS,
)


async def validate_file_type(file: UploadFile) -> bool:
    """Validate file type is an image"""
    allowed_types = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed_types:
        raise UploadException("Only JPEG and PNG images are allowed")
    return True


async def save_upload_file(file: UploadFile, temp_filepath: Path) -> None:
    """Save uploaded file with proper error handling"""
    try:
        async with aiofiles.open(temp_filepath, "wb") as buffer:
            # Read and write in chunks to handle large files
            chunk_size = 1024 * 8  # 8KB chunks
            while chunk := await file.read(chunk_size):
                await buffer.write(chunk)
    except Exception as e:
        logger.error(f"Error saving file {temp_filepath}: {e}")
        raise UploadException("Failed to save uploaded file", 500)


@app.get("/")
def main_page():
    return {"message": "UI Backend upload API is running", "version": "1.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint that also verifies AI backend connectivity"""
    try:
        response = await run_in_threadpool(
            lambda: requests.get(
                AI_BACKEND_URL.replace("/detect", "/"),
                timeout=5,  # 5 second timeout for health checks
            )
        )
        ai_backend_healthy = response.status_code == 200
    except (requests.RequestException, requests.Timeout):
        ai_backend_healthy = False

    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "ai_backend": "healthy" if ai_backend_healthy else "unhealthy",
    }

    status_code = 200 if ai_backend_healthy else 503
    return JSONResponse(content=health_status, status_code=status_code)


@app.post("/upload")
@limiter.limit(f"{RATE_LIMIT_REQUESTS}/{RATE_LIMIT_WINDOW}s")
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    x_request_id: Optional[str] = None,
):
    request_id = x_request_id or str(uuid.uuid4())
    logger.info(f"Processing upload request {request_id} for file {file.filename}")

    try:
        # Validate file size
        file.file.seek(0, 2)  # Seek to end
        size = file.file.tell()
        file.file.seek(0)  # Reset file position

        if size > MAX_FILE_SIZE:
            raise UploadException(
                f"File size exceeds maximum allowed size of {MAX_FILE_SIZE / 1024 / 1024}MB"
            )

        # Validate file type
        await validate_file_type(file)

        # Save uploaded file to temp with proper cleanup
        temp_filename = f"{uuid.uuid4()}_{file.filename}"
        temp_filepath = TEMP_DIR / temp_filename

        try:
            await save_upload_file(file, temp_filepath)

            # Forward to AI Backend with timeout and proper async handling
            async with aiofiles.open(temp_filepath, "rb") as f:
                file_content = await f.read()

            # Use run_in_threadpool for the blocking requests call
            response = await run_in_threadpool(
                lambda: requests.post(
                    AI_BACKEND_URL,
                    files={"file": (file.filename, file_content, file.content_type)},
                    headers={"x-api-key": API_KEY},
                    timeout=30,  # 30 second timeout
                )
            )

            if response.status_code != 200:
                raise UploadException(
                    f"AI Backend error: {response.text}",
                    status_code=response.status_code,
                )

            data = response.json()
            detections = data.get("detections", [])

            # Summary counts for streamlit UI
            vehicle_labels = {"car", "bus", "truck", "motorbike", "bicycle"}
            people_count = sum(1 for d in detections if d.get("label") == "person")
            vehicle_count = sum(
                1 for d in detections if d.get("label") in vehicle_labels
            )

            response_data = {
                "request_id": request_id,
                "detections": detections,
                "summary": {"people": people_count, "vehicles": vehicle_count},
            }

            logger.info(
                f"Request {request_id} completed successfully: "
                f"{people_count} people, {vehicle_count} vehicles detected"
            )

            return JSONResponse(response_data)

        finally:
            # Cleanup temp file
            try:
                if temp_filepath.exists():
                    temp_filepath.unlink()
            except Exception as e:
                logger.error(f"Error cleaning up temp file {temp_filepath}: {e}")

    except UploadException as e:
        logger.warning(f"Request {request_id} failed with {e.status_code}: {e.message}")
        return JSONResponse(
            {"request_id": request_id, "error": e.message}, status_code=e.status_code
        )
    except Exception as e:
        logger.error(f"Request {request_id} failed with unexpected error: {e}")
        return JSONResponse(
            {"request_id": request_id, "error": "Internal server error"},
            status_code=500,
        )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8500,
        log_config=None,  # Disable uvicorn's default logging
    )
