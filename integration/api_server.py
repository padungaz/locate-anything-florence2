"""
api_server.py — FastAPI REST Service for Florence-2 Vision Grounding (Commercial Edition)

Production-ready API service that:
  1. Loads Microsoft Florence-2-large on startup (Singleton Engine, MIT License).
  2. Exposes POST /v1/locate for object detection via image upload + prompt.
  3. Provides /healthz and /ready endpoints for container orchestration.
  4. Returns annotated images with bounding boxes via /v1/locate/annotated.
  5. All inference calls are thread-safe via florence_wrapper mutex.

License: MIT (fully commercial)

Run:
    uvicorn api_server:app --host 0.0.0.0 --port 8080
"""

import io
import os
import sys
import json
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Add integration dir to path
sys.path.insert(0, str(Path(__file__).parent))

from florence_wrapper import FlorenceEngine, FlorenceError, Mode

# ---------------------------------------------------------------------------
# Configuration (via environment variables with sensible defaults)
# ---------------------------------------------------------------------------
MODEL_ID     = os.environ.get("FLORENCE_MODEL_ID", "microsoft/Florence-2-large")
DEVICE       = os.environ.get("FLORENCE_DEVICE", None)  # auto-detect cuda/cpu
DEFAULT_MODE = os.environ.get("FLORENCE_MODE", "grounding").lower()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("florence-api")

# ---------------------------------------------------------------------------
# Global engine reference
# ---------------------------------------------------------------------------
engine: Optional[FlorenceEngine] = None
engine_ready = False
startup_time: Optional[float] = None


def _parse_mode(mode_str: str):
    """Convert string mode to Mode enum. Accepts legacy LocateAnything modes too."""
    mode_map = {
        "od": Mode.OD,
        "grounding": Mode.GROUNDING,
        "dense": Mode.DENSE_CAPTION,
        # Legacy LocateAnything compatibility
        "fast": Mode.GROUNDING,
        "hybrid": Mode.GROUNDING,
        "slow": Mode.GROUNDING,
    }
    m = mode_map.get(mode_str.lower())
    if m is None:
        valid = ", ".join(sorted(mode_map.keys()))
        raise ValueError(f"Invalid mode '{mode_str}'. Must be one of: {valid}")
    return m


# ---------------------------------------------------------------------------
# FastAPI Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load engine on startup, release on shutdown."""
    global engine, engine_ready, startup_time

    logger.info("Starting Florence-2 engine...")
    logger.info(f"  Model:  {MODEL_ID}")
    logger.info(f"  Device: {DEVICE or 'auto-detect'}")
    logger.info(f"  Default mode: {DEFAULT_MODE}")

    t0 = time.monotonic()
    try:
        engine = FlorenceEngine(model_id=MODEL_ID, device=DEVICE)
        startup_time = time.monotonic() - t0
        engine_ready = True
        logger.info(f"Engine loaded in {startup_time:.1f}s — ready to serve requests.")
    except Exception as e:
        logger.error(f"Failed to load engine: {e}")
        engine_ready = False

    yield  # Server is running

    # Shutdown
    logger.info("Shutting down engine...")
    if engine:
        engine.close()
        engine = None
        engine_ready = False
    logger.info("Engine released. Goodbye.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Florence-2 Vision Grounding API (Commercial Edition)",
    description="Open-vocabulary object detection powered by Microsoft Florence-2 (MIT License)",
    version="2.0.0",
    lifespan=lifespan
)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------
class Detection(BaseModel):
    label: str = Field(..., description="Detected object label")
    box: list = Field(..., description="Bounding box [x1, y1, x2, y2] in pixels")


class DetectionResponse(BaseModel):
    detections: list[Detection] = Field(..., description="List of detected objects")
    count: int = Field(..., description="Number of detections")
    latency_ms: float = Field(..., description="Inference time in milliseconds")
    mode: str = Field(..., description="Decode mode used")


class HealthResponse(BaseModel):
    status: str
    engine_loaded: bool
    model: str
    startup_time_s: Optional[float] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/healthz", response_model=HealthResponse, tags=["Health"])
async def healthz():
    """Liveness probe — returns 200 if the server process is alive."""
    return HealthResponse(
        status="alive",
        engine_loaded=engine_ready,
        model=MODEL_ID,
        startup_time_s=round(startup_time, 2) if startup_time else None
    )


@app.get("/ready", tags=["Health"])
async def ready():
    """Readiness probe — returns 200 only when the model is loaded and ready."""
    if not engine_ready or not engine or not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Engine not ready")
    return {"status": "ready"}


@app.post("/v1/locate", response_model=DetectionResponse, tags=["Detection"])
async def locate(
    image: UploadFile = File(..., description="Input image (PNG/JPG)"),
    prompt: str = Form(..., description="Detection prompt. Separate categories with </c>"),
    mode: str = Form(default=DEFAULT_MODE, description="Decode mode: grounding, od, dense, fast, hybrid, slow"),
):
    """
    Detect objects in an uploaded image using open-vocabulary grounding.

    **Prompt format:**
    - Single category: `"person"`
    - Multiple categories: `"person</c>car</c>traffic light"`
    - Full template: `"Locate all the instances that matches the following description: person</c>car."`

    **Returns:** JSON with bounding boxes in pixel coordinates relative to the original image.
    """
    if not engine_ready or not engine:
        raise HTTPException(status_code=503, detail="Engine not ready")

    # Validate mode
    try:
        decode_mode = _parse_mode(mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Read image bytes
    try:
        image_bytes = await image.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image file")

    # Validate image format (basic magic byte check)
    if not (image_bytes[:8].startswith(b'\x89PNG') or
            image_bytes[:2] in (b'\xff\xd8',) or  # JPEG
            image_bytes[:4] == b'RIFF'):            # WebP
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format. Please upload PNG, JPEG, or WebP."
        )

    # Run inference
    t0 = time.monotonic()
    try:
        detections = engine.detect_buffer(image_bytes, prompt, mode=decode_mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FlorenceError as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    except Exception as e:
        logger.exception("Unexpected inference error")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    latency_ms = (time.monotonic() - t0) * 1000

    logger.info(
        f"POST /v1/locate — prompt=\"{prompt[:60]}\" mode={mode} "
        f"detections={len(detections)} latency={latency_ms:.0f}ms"
    )

    return DetectionResponse(
        detections=[Detection(**d) for d in detections],
        count=len(detections),
        latency_ms=round(latency_ms, 1),
        mode=mode
    )


@app.post("/v1/locate/annotated", tags=["Detection"])
async def locate_annotated(
    image: UploadFile = File(..., description="Input image (PNG/JPG)"),
    prompt: str = Form(..., description="Detection prompt"),
    mode: str = Form(default=DEFAULT_MODE, description="Decode mode"),
):
    """
    Detect objects AND return an annotated image with bounding boxes drawn.

    Returns a PNG image with detection boxes overlaid. Also includes
    detection metadata in the X-Detections response header as JSON.
    """
    if not engine_ready or not engine:
        raise HTTPException(status_code=503, detail="Engine not ready")

    try:
        decode_mode = _parse_mode(mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image file")

    # Run detection
    t0 = time.monotonic()
    try:
        detections = engine.detect_buffer(image_bytes, prompt, mode=decode_mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FlorenceError as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    latency_ms = (time.monotonic() - t0) * 1000

    # Draw bounding boxes on image using PIL
    from PIL import Image as PILImage, ImageDraw, ImageFont

    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Color palette for different labels
    COLORS = [
        (255, 85, 85), (85, 255, 85), (85, 85, 255),
        (255, 255, 85), (255, 85, 255), (85, 255, 255),
        (255, 170, 85), (170, 85, 255), (85, 255, 170),
    ]
    label_colors = {}
    color_idx = 0

    for det in detections:
        label = det["label"]
        box = det["box"]  # [x1, y1, x2, y2]

        if label not in label_colors:
            label_colors[label] = COLORS[color_idx % len(COLORS)]
            color_idx += 1
        color = label_colors[label]

        # Draw rectangle (3px border)
        for offset in range(3):
            draw.rectangle(
                [box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset],
                outline=color
            )

        # Draw label background + text
        text = f" {label} "
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except (IOError, OSError):
            font = ImageFont.load_default()

        text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        draw.rectangle(
            [box[0], box[1] - text_h - 4, box[0] + text_w + 4, box[1]],
            fill=color
        )
        draw.text((box[0] + 2, box[1] - text_h - 2), text, fill=(255, 255, 255), font=font)

    # Encode to PNG
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # Include detection metadata in response header
    det_header = json.dumps({"count": len(detections), "latency_ms": round(latency_ms, 1)})

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={
            "X-Detections": det_header,
            "X-Detection-Count": str(len(detections)),
            "X-Latency-Ms": str(round(latency_ms, 1)),
        }
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("FLORENCE_PORT", "8080"))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, workers=1)
