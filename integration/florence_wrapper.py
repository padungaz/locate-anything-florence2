#!/usr/bin/env python3
"""
florence_wrapper.py — Thread-Safe Singleton Wrapper for Microsoft Florence-2

Drop-in replacement for la_wrapper.py. Provides the same public API:
  - FlorenceEngine(model_id) — load model
  - .detect_buffer(image_bytes, prompt, mode) -> List[Dict]
  - .detect(image_path, prompt, mode) -> List[Dict]
  - .close()
  - .is_loaded property
  - Context manager support (with statement)

Output format is identical to LocateAnything:
  [{"label": "cat", "box": [x1, y1, x2, y2]}, ...]

License: MIT (model weights + this code = fully commercial)
"""

import io
import os
import re
import time
import threading
import logging
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Union

from PIL import Image

logger = logging.getLogger("florence-engine")


class Mode(IntEnum):
    """Decode modes — kept for API compatibility with la_wrapper."""
    OD = 0              # Pure object detection (auto-discover all objects)
    GROUNDING = 1       # Caption-to-phrase grounding (find specific objects)
    DENSE_CAPTION = 2   # Dense region captioning


class FlorenceError(Exception):
    """Raised when inference fails."""
    pass


class FlorenceEngine:
    """
    Thread-safe Florence-2 inference engine.

    Usage:
        engine = FlorenceEngine("microsoft/Florence-2-large")
        detections = engine.detect("photo.jpg", "cat</c>remote")
        engine.close()

    Or as context manager:
        with FlorenceEngine("microsoft/Florence-2-large") as eng:
            dets = eng.detect_buffer(img_bytes, "person</c>car")
    """

    def __init__(self, model_id: str = "microsoft/Florence-2-large",
                 device: Optional[str] = None, **kwargs):
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._device = device
        self._model_id = model_id
        self._loaded = False

        self._load(model_id)

    def _load(self, model_id: str):
        """Load Florence-2 model and processor from HuggingFace."""
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        if self._device is None:
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"

        dtype = torch.float16 if "cuda" in self._device else torch.float32

        logger.info(f"Loading Florence-2 model: {model_id}")
        logger.info(f"  Device: {self._device}, dtype: {dtype}")

        self._processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True
        ).to(self._device)

        self._dtype = dtype
        self._loaded = True
        logger.info("Florence-2 model loaded successfully.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def close(self):
        """Release model from memory."""
        with self._lock:
            if self._model is not None:
                del self._model
                del self._processor
                self._model = None
                self._processor = None
                self._loaded = False

                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                logger.info("Florence-2 engine released.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        status = "loaded" if self._loaded else "closed"
        return f"<FlorenceEngine model={self._model_id} device={self._device} {status}>"

    # -----------------------------------------------------------------------
    # Prompt Translation
    # -----------------------------------------------------------------------
    @staticmethod
    def _translate_prompt(prompt: str):
        """
        Convert LocateAnything-style prompt to Florence-2 task + text_input.

        Input formats supported:
          - "cat</c>remote"  -> CAPTION_TO_PHRASE_GROUNDING with generated caption
          - "cat"            -> CAPTION_TO_PHRASE_GROUNDING
          - "Locate all..."  -> strip prefix, extract labels
          - "" or None       -> fall back to <OD> (detect everything)

        Returns: (task_token, text_input_or_None)
        """
        if not prompt or not prompt.strip():
            return "<OD>", None

        p = prompt.strip()

        # Strip the LocateAnything canonical prefix if present
        prefix = "Locate all the instances that matches the following description:"
        if p.lower().startswith(prefix.lower()):
            p = p[len(prefix):].strip().rstrip(".")

        # Split on </c> separator (LocateAnything multi-label format)
        labels = [lbl.strip() for lbl in p.split("</c>") if lbl.strip()]

        if not labels:
            return "<OD>", None

        # Build a natural-language caption for phrase grounding
        caption = " and ".join(labels)
        # e.g. "cat and remote" -> Florence finds "cat" and "remote"

        return "<CAPTION_TO_PHRASE_GROUNDING>", caption

    # -----------------------------------------------------------------------
    # Core Inference
    # -----------------------------------------------------------------------
    def _run_inference(self, image: Image.Image, task: str,
                       text_input: Optional[str] = None) -> dict:
        """Run Florence-2 inference with the given task token and optional text."""
        import torch

        if not self._loaded:
            raise RuntimeError("Engine is closed. Cannot run inference.")

        prompt = task if text_input is None else task + text_input

        with self._lock:
            inputs = self._processor(
                text=prompt, images=image, return_tensors="pt"
            ).to(self._device, self._dtype)

            with torch.no_grad():
                generated_ids = self._model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False,
                )

            generated_text = self._processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]

            parsed = self._processor.post_process_generation(
                generated_text,
                task=task,
                image_size=(image.width, image.height),
            )

        return parsed

    def _parse_detections(self, parsed: dict, task: str,
                          requested_labels: List[str]) -> List[Dict]:
        """
        Convert Florence-2 parsed output to LocateAnything-compatible format.

        Florence output:
          {'<CAPTION_TO_PHRASE_GROUNDING>': {'bboxes': [[x1,y1,x2,y2], ...], 'labels': [...]}}
          {'<OD>': {'bboxes': [[x1,y1,x2,y2], ...], 'labels': [...]}}

        LocateAnything output:
          [{"label": "cat", "box": [x1, y1, x2, y2]}, ...]
        """
        result_key = task
        data = parsed.get(result_key, {})

        bboxes = data.get("bboxes", [])
        labels = data.get("labels", [])

        detections = []
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            # bbox from Florence is already [x1, y1, x2, y2] in pixel coords
            detections.append({
                "label": label.strip().lower() if label else f"object_{i}",
                "box": [round(c, 3) for c in bbox],
            })

        return detections

    # -----------------------------------------------------------------------
    # Public API (matches la_wrapper interface)
    # -----------------------------------------------------------------------
    def detect(self, image_path: str, prompt: str,
               mode: Union[Mode, int, str] = Mode.GROUNDING) -> List[Dict]:
        """
        Detect objects in an image file.

        Args:
            image_path: Path to image file (PNG/JPG/WebP).
            prompt:     Open-vocabulary query (supports </c> separator).
            mode:       Ignored for compatibility; Florence auto-selects task.

        Returns:
            List of {"label": str, "box": [x1, y1, x2, y2]}
        """
        if not self._loaded:
            raise RuntimeError("Engine is closed.")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image file: {e}")
        return self._detect_image(image, prompt)

    def detect_buffer(self, image_bytes: bytes, prompt: str,
                      mode: Union[Mode, int, str] = Mode.GROUNDING) -> List[Dict]:
        """
        Detect objects from in-memory image bytes.

        Args:
            image_bytes: Raw image bytes (PNG/JPG/WebP).
            prompt:      Open-vocabulary query (supports </c> separator).
            mode:        Ignored for compatibility.

        Returns:
            List of {"label": str, "box": [x1, y1, x2, y2]}
        """
        if not self._loaded:
            raise RuntimeError("Engine is closed.")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image: {e}")
        return self._detect_image(image, prompt)

    def _detect_image(self, image: Image.Image, prompt: str) -> List[Dict]:
        """Internal: run detection on a PIL Image."""
        task, text_input = self._translate_prompt(prompt)

        # Extract requested labels for filtering
        p = prompt.strip()
        prefix = "Locate all the instances that matches the following description:"
        if p.lower().startswith(prefix.lower()):
            p = p[len(prefix):].strip().rstrip(".")
        requested_labels = [lbl.strip().lower() for lbl in p.split("</c>") if lbl.strip()]

        parsed = self._run_inference(image, task, text_input)
        detections = self._parse_detections(parsed, task, requested_labels)

        # Heuristic: Florence-2 phrase grounding defaults to the whole frame (~90-100% area)
        # when an object is completely absent. Filter out such spurious full-frame boxes.
        if task == "<CAPTION_TO_PHRASE_GROUNDING>":
            img_area = image.width * image.height
            valid_detections = []
            for d in detections:
                b = d["box"]
                box_area = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
                if (box_area / img_area) < 0.90:
                    valid_detections.append(d)
            detections = valid_detections

        # If grounding returned nothing, fall back to <OD> (generic detection)
        if not detections and task != "<OD>":
            logger.info("Grounding returned 0 results, falling back to <OD>")
            parsed_od = self._run_inference(image, "<OD>", None)
            all_dets = self._parse_detections(parsed_od, "<OD>", requested_labels)
            # Filter to only requested labels if any
            if requested_labels:
                detections = [
                    d for d in all_dets
                    if any(rl in d["label"] for rl in requested_labels)
                ]
            else:
                detections = all_dets

        return detections
