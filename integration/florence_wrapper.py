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
        """
        if not prompt or not prompt.strip() or prompt.strip().upper() in ("<OD>", "OD"):
            return "<OD>", None

        p = prompt.strip()

        # Strip the LocateAnything canonical prefix if present
        prefix = "Locate all the instances that matches the following description:"
        if p.lower().startswith(prefix.lower()):
            p = p[len(prefix):].strip().rstrip(".")

        # Split on </c> separator (LocateAnything multi-label format)
        labels = [lbl.strip() for lbl in p.split("</c>") if lbl.strip() and lbl.strip().upper() not in ("<OD>", "OD")]

        if not labels:
            return "<OD>", None

        # Build a natural-language caption for phrase grounding
        caption = " and ".join(labels)
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

        prompt = task if not text_input else task + text_input

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
        """
        result_key = task
        data = parsed.get(result_key, {})

        bboxes = data.get("bboxes", [])
        labels = data.get("labels", [])

        detections = []
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
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
        if not self._loaded:
            raise RuntimeError("Engine is closed.")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image file: {e}")
        return self._detect_image(image, prompt)

    def detect_buffer(self, image_bytes: bytes, prompt: str,
                      mode: Union[Mode, int, str] = Mode.GROUNDING) -> List[Dict]:
        if not self._loaded:
            raise RuntimeError("Engine is closed.")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image: {e}")
        return self._detect_image(image, prompt)

    def _detect_image(self, image: Image.Image, prompt: str) -> List[Dict]:
        """Internal: run detection on a PIL Image with high-recall hybrid detection."""
        p = prompt.strip()
        if not p or p.upper() in ("<OD>", "OD"):
            parsed_od = self._run_inference(image, "<OD>", None)
            return self._parse_detections(parsed_od, "<OD>", [])

        # Parse requested labels
        prefix = "Locate all the instances that matches the following description:"
        if p.lower().startswith(prefix.lower()):
            p = p[len(prefix):].strip().rstrip(".")
        requested_labels = [lbl.strip().lower() for lbl in p.split("</c>") if lbl.strip() and lbl.strip().upper() not in ("<OD>", "OD")]

        if not requested_labels:
            parsed_od = self._run_inference(image, "<OD>", None)
            return self._parse_detections(parsed_od, "<OD>", [])

        # 1. First run <OD> to capture all exhaustive standard object instances in the scene
        parsed_od = self._run_inference(image, "<OD>", None)
        all_od_dets = self._parse_detections(parsed_od, "<OD>", requested_labels)

        od_matched = []
        found_labels = set()
        for d in all_od_dets:
            for req in requested_labels:
                if req == d["label"] or (len(req) > 3 and req in d["label"]) or (len(d["label"]) > 3 and d["label"] in req):
                    d["label"] = req  # normalize label
                    od_matched.append(d)
                    found_labels.add(req)
                    break

        # 2. For any label NOT matched in OD (e.g. fine-grained phrases, parts), run phrase grounding
        missing_labels = [lbl for lbl in requested_labels if lbl not in found_labels]
        grounding_dets = []
        img_area = image.width * image.height

        for missing in missing_labels:
            task = "<CAPTION_TO_PHRASE_GROUNDING>"
            parsed_gr = self._run_inference(image, task, missing)
            gr_dets = self._parse_detections(parsed_gr, task, [missing])
            for d in gr_dets:
                b = d["box"]
                box_area = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
                if (box_area / img_area) < 0.90:
                    d["label"] = missing
                    grounding_dets.append(d)

        final_dets = od_matched + grounding_dets

        # 3. Fallback to individual phrase grounding if OD returned 0 matches
        if not final_dets:
            for req in requested_labels:
                task = "<CAPTION_TO_PHRASE_GROUNDING>"
                parsed_gr = self._run_inference(image, task, req)
                gr_dets = self._parse_detections(parsed_gr, task, [req])
                for d in gr_dets:
                    b = d["box"]
                    box_area = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
                    if (box_area / img_area) < 0.90:
                        d["label"] = req
                        final_dets.append(d)

        return final_dets

    def ocr_buffer(self, image_bytes: bytes) -> List[Dict]:
        """
        Extract text and quad bounding boxes from an image (<OCR_WITH_REGION>).
        Returns list of {"text": str, "quad_box": [x1, y1, x2, y2, x3, y3, x4, y4], "box": [x1, y1, x2, y2]}
        """
        if not self._loaded:
            raise RuntimeError("Engine is closed.")
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image: {e}")

        task = "<OCR_WITH_REGION>"
        parsed = self._run_inference(image, task, None)
        data = parsed.get(task, {})
        quad_boxes = data.get("quad_boxes", [])
        labels = data.get("labels", [])

        results = []
        for qbox, text in zip(quad_boxes, labels):
            xs = qbox[0::2]
            ys = qbox[1::2]
            bbox = [round(min(xs), 3), round(min(ys), 3), round(max(xs), 3), round(max(ys), 3)]
            results.append({
                "text": text.strip(),
                "quad_box": [round(c, 3) for c in qbox],
                "box": bbox,
            })
        return results

    def ocr(self, image_path: str) -> List[Dict]:
        with open(image_path, "rb") as f:
            return self.ocr_buffer(f.read())

    def segment_buffer(self, image_bytes: bytes, prompt: str) -> List[Dict]:
        """
        Segment objects matching prompt using <REFERRING_EXPRESSION_SEGMENTATION>.
        Returns list of {"label": str, "polygons": [...], "box": [x1, y1, x2, y2]}
        """
        if not self._loaded:
            raise RuntimeError("Engine is closed.")
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image: {e}")

        task = "<REFERRING_EXPRESSION_SEGMENTATION>"
        clean_prompt = prompt.replace("</c>", " and ").strip()
        parsed = self._run_inference(image, task, clean_prompt)
        data = parsed.get(task, {})
        polygons = data.get("polygons", [])
        labels = data.get("labels", [])

        results = []
        for poly_list, label in zip(polygons, labels):
            all_x = []
            all_y = []
            for poly in poly_list:
                all_x.extend(poly[0::2])
                all_y.extend(poly[1::2])
            bbox = [round(min(all_x), 3), round(min(all_y), 3), round(max(all_x), 3), round(max(all_y), 3)] if all_x else [0, 0, 0, 0]
            results.append({
                "label": label.strip().lower(),
                "polygons": poly_list,
                "box": bbox,
            })
        return results

    def segment(self, image_path: str, prompt: str) -> List[Dict]:
        with open(image_path, "rb") as f:
            return self.segment_buffer(f.read(), prompt)
