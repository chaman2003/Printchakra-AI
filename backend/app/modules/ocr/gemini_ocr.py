"""
OCR through Gemini vision.

Replaces PaddleOCR / Tesseract / the Ollama post-processing step on serverless hosting (no GPU,
no system binaries). One structured call returns every text line with its type and bounding box,
in the same shape the frontend already reads (OCRResult in frontend/src/types.ts).

Environment:
    GEMINI_API_KEY   required
    OCR_MODEL        optional, default gemini-3.8-flash
"""

import io
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-flash-lite"
# Tried in order when the primary model is overloaded (503), slow (504) or rate limited (429).
# OCR needs no reasoning, so the fast lite models come first; the larger ones are the safety net.
FALLBACK_MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3.6-flash"]
# Total time budget across all models. Vercel's function limit is 120 s and the pipeline needs the rest.
TOTAL_BUDGET_S = 70
# Larger images add upload time and tokens without improving text recognition.
MAX_SIDE = 2000
REQUEST_TIMEOUT_MS = 25_000
# Gemini does not report per-line confidence. A fixed value keeps the UI fields populated.
LINE_CONFIDENCE = 0.95

UNIT_TYPES = ["title", "heading", "paragraph", "list_item", "table_cell", "footer", "other"]

PROMPT = (
    "Extract every piece of readable text from this document image. "
    "Return one entry per visual text line, in reading order. "
    "For each line give its exact text, its role (title, heading, paragraph, list_item, table_cell, footer or other) "
    "and its bounding box as [ymin, xmin, ymax, xmax] with values from 0 to 1000 relative to the image. "
    "Also give a short document title. Do not invent text. If there is no text, return an empty lines array."
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "type": {"type": "string", "enum": UNIT_TYPES},
                    "box_2d": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["text", "type", "box_2d"],
            },
        },
    },
    "required": ["title", "lines"],
}


def is_available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def _prepare_image(data: bytes):
    """Return (jpeg_bytes, original_width, original_height). Shrinks very large images."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("RGB")
        width, height = image.size
        longest = max(width, height)
        if longest > MAX_SIDE:
            scale = MAX_SIDE / longest
            image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))))
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=88)
    return out.getvalue(), width, height


def _bbox(box: List[int], width: int, height: int) -> Dict[str, Any]:
    """Convert a Gemini [ymin, xmin, ymax, xmax] box (0-1000) to the frontend bbox shape in pixels."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return {"x": 0, "y": 0, "width": 0, "height": 0, "points": []}
    ymin, xmin, ymax, xmax = (max(0, min(1000, float(v))) for v in box)
    x0, x1 = sorted((xmin / 1000 * width, xmax / 1000 * width))
    y0, y1 = sorted((ymin / 1000 * height, ymax / 1000 * height))
    return {
        "x": x0,
        "y": y0,
        "width": x1 - x0,
        "height": y1 - y0,
        "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
    }


def _call_gemini(jpeg: bytes, model: str) -> Dict[str, Any]:
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )
    response = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0,
        ),
    )
    return json.loads(response.text or "{}")


def _is_transient(error: Exception) -> bool:
    """Overload, rate limit and timeout errors are worth retrying or trying another model for."""
    text = str(error).lower()
    return any(marker in text for marker in ("503", "429", "unavailable", "overloaded", "high demand", "deadline", "timed out", "timeout"))


def _models() -> List[str]:
    primary = os.environ.get("OCR_MODEL", DEFAULT_MODEL)
    return [primary] + [m for m in FALLBACK_MODELS if m != primary]


def ocr_image_bytes(data: bytes) -> Dict[str, Any]:
    """Run OCR on an encoded image (JPEG, PNG, WebP...). Returns an OCRResult-shaped dict."""
    if not is_available():
        raise RuntimeError("GEMINI_API_KEY is not set")

    started = time.time()
    jpeg, width, height = _prepare_image(data)

    parsed: Optional[Dict[str, Any]] = None
    last_error: Optional[Exception] = None
    # One attempt per model: an overloaded model rarely recovers within seconds, the next one often works.
    for model in _models():
        if time.time() - started > TOTAL_BUDGET_S:
            break
        try:
            parsed = _call_gemini(jpeg, model)
            break
        except Exception as exc:  # network errors, malformed JSON, quota
            last_error = exc
            logger.warning("Gemini OCR (%s) failed: %s", model, exc)
            if not _is_transient(exc):
                raise RuntimeError(f"Gemini OCR failed: {exc}") from exc
    if parsed is None:
        raise RuntimeError(f"Gemini OCR failed: {last_error}")

    raw_results: List[Dict[str, Any]] = []
    structured_units: List[Dict[str, Any]] = []
    texts: List[str] = []
    word_index = 0
    for line in parsed.get("lines") or []:
        text = str(line.get("text") or "").strip()
        if not text:
            continue
        bbox = _bbox(line.get("box_2d"), width, height)
        unit_type = line.get("type") if line.get("type") in UNIT_TYPES else "other"
        words = text.split()
        texts.append(text)
        raw_results.append({"text": text, "confidence": LINE_CONFIDENCE, "bbox": bbox})
        structured_units.append(
            {
                "text": text,
                "type": unit_type,
                "bbox": bbox,
                "confidence": LINE_CONFIDENCE,
                "word_indices": list(range(word_index, word_index + len(words))),
            }
        )
        word_index += len(words)

    full_text = "\n".join(texts)
    title = str(parsed.get("title") or "").strip() or (texts[0] if texts else "")
    return {
        "raw_results": raw_results,
        "structured_units": structured_units,
        "full_text": full_text,
        "derived_title": title[:120],
        "confidence_avg": LINE_CONFIDENCE if raw_results else 0.0,
        "word_count": word_index,
        "timestamp": datetime.now().isoformat(),
        "processing_time_ms": (time.time() - started) * 1000,
        "image_dimensions": [width, height],
        "engine": "gemini",
    }


def ocr_image_file(path: str) -> Dict[str, Any]:
    with open(path, "rb") as handle:
        return ocr_image_bytes(handle.read())
