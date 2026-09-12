"""
florence_client.py — Python Client SDK for Florence-2 Vision Service

Usage:
    from florence_client import FlorenceClient

    client = FlorenceClient("http://localhost:8080", api_key="optional_key")
    
    # 1. Open-Vocabulary Visual Grounding
    dets = client.locate("street.jpg", prompt="bus</c>car")
    for d in dets.detections:
        print(d.label, d.box)

    # 2. Detect All Objects (<OD>)
    all_objs = client.detect_all("photo.jpg")

    # 3. OCR Text Extraction
    ocr_result = client.ocr("license_plate.jpg")
    print("Detected text:", ocr_result.full_text)

    # 4. Download Annotated Image
    client.download_annotated("street.jpg", "bus", "annotated_bus.png")
"""

import io
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import requests


class Detection:
    def __init__(self, label: str, box: List[float]):
        self.label = label
        self.box = box  # [x1, y1, x2, y2] in pixels

    def __repr__(self):
        return f"<Detection label='{self.label}' box={self.box}>"


class DetectionResult:
    def __init__(self, data: dict):
        self.raw = data
        self.count = data.get("count", 0)
        self.latency_ms = data.get("latency_ms", 0.0)
        self.mode = data.get("mode", "")
        self.detections = [
            Detection(d["label"], d["box"]) for d in data.get("detections", [])
        ]

    def __repr__(self):
        return f"<DetectionResult count={self.count} latency={self.latency_ms}ms detections={self.detections}>"


class OCRItem:
    def __init__(self, text: str, quad_box: List[float], box: List[float]):
        self.text = text
        self.quad_box = quad_box
        self.box = box

    def __repr__(self):
        return f"<OCRItem text='{self.text}' box={self.box}>"


class OCRResult:
    def __init__(self, data: dict):
        self.raw = data
        self.count = data.get("count", 0)
        self.full_text = data.get("full_text", "")
        self.latency_ms = data.get("latency_ms", 0.0)
        self.regions = [
            OCRItem(r["text"], r["quad_box"], r["box"]) for r in data.get("regions", [])
        ]

    def __repr__(self):
        return f"<OCRResult count={self.count} full_text='{self.full_text}'>"


class FlorenceClient:
    """
    Client for interacting with the Florence-2 Vision Grounding Service.
    """

    def __init__(self, base_url: str = "http://localhost:8080", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._session = requests.Session()
        if self.api_key:
            self._session.headers.update({"X-API-Key": self.api_key})

    def is_ready(self) -> bool:
        """Check if the backend cluster is ready to accept requests."""
        try:
            r = self._session.get(f"{self.base_url}/ready", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def cluster_status(self) -> dict:
        """Fetch cluster health, active worker nodes, and telemetry."""
        r = self._session.get(f"{self.base_url}/cluster/status", timeout=5)
        r.raise_for_status()
        return r.json()

    def locate(self, image: Union[str, Path, bytes, io.BytesIO],
               prompt: str, mode: str = "grounding") -> DetectionResult:
        """
        Locate objects described in the text prompt.
        """
        files, to_close = self._prepare_image_file(image)
        try:
            data = {"prompt": prompt, "mode": mode}
            r = self._session.post(f"{self.base_url}/v1/locate", files=files, data=data, timeout=60)
            r.raise_for_status()
            return DetectionResult(r.json())
        finally:
            if to_close:
                to_close.close()

    def detect_all(self, image: Union[str, Path, bytes, io.BytesIO]) -> DetectionResult:
        """
        Automatically detect all objects in the image without typing a prompt (<OD>).
        """
        return self.locate(image, prompt="<OD>", mode="od")

    def ocr(self, image: Union[str, Path, bytes, io.BytesIO]) -> OCRResult:
        """
        Extract text and detect region quad boxes from an image.
        """
        files, to_close = self._prepare_image_file(image)
        try:
            r = self._session.post(f"{self.base_url}/v1/ocr", files=files, timeout=60)
            r.raise_for_status()
            return OCRResult(r.json())
        finally:
            if to_close:
                to_close.close()

    def download_annotated(self, image: Union[str, Path, bytes, io.BytesIO],
                           prompt: str, output_path: Union[str, Path],
                           mode: str = "grounding") -> Path:
        """
        Detect objects and save the annotated image with bounding boxes overlaid.
        """
        files, to_close = self._prepare_image_file(image)
        try:
            data = {"prompt": prompt, "mode": mode}
            r = self._session.post(f"{self.base_url}/v1/locate/annotated", files=files, data=data, timeout=60)
            r.raise_for_status()
            out_path = Path(output_path)
            out_path.write_bytes(r.content)
            return out_path
        finally:
            if to_close:
                to_close.close()

    @staticmethod
    def _prepare_image_file(image):
        """Helper to convert various image input formats into requests files dict."""
        if isinstance(image, (str, Path)):
            f = open(image, "rb")
            return {"image": (Path(image).name, f, "image/png")}, f
        elif isinstance(image, bytes):
            return {"image": ("image.png", image, "image/png")}, None
        elif isinstance(image, io.BytesIO):
            image.seek(0)
            return {"image": ("image.png", image.read(), "image/png")}, None
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
