"""
Standalone phone -> laptop processing flow.

This module centralizes the entire runtime path:
1) receive file from phone
2) store upload artifact
3) emit realtime "new_file"
4) run enhanced pipeline asynchronously for images
5) persist OCR artifacts
6) emit realtime completion/error for dashboard refresh
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from app.core import socketio
from app.core.config import get_data_dirs
from app.core.paths import IS_CLOUD

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "pdf", "doc", "docx", "txt"}
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
DEBUG_LOG_PATH = "C:/Users/chama/OneDrive/Desktop/printchakra-new/debug-68db5a.log"


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_image_extension(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""
    return ext in IMAGE_EXTENSIONS


def parse_enhancement_params(form: Any) -> dict[str, Any]:
    return {
        "brightness_boost": int(form.get("brightness_boost", 25)),
        "equalization_strength": float(form.get("equalization_strength", 0.4)),
        "clahe_clip_limit": float(form.get("clahe_clip_limit", 2.0)),
        "clahe_tile_size": int(form.get("clahe_tile_size", 8)),
    }


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    try:
        payload = {
            "sessionId": "68db5a",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


def _emit_processing_progress(filename: str, progress_data: dict[str, Any]) -> None:
    socketio.emit("processing_progress", {"filename": filename, **progress_data})


@dataclass
class StoredUpload:
    filename: str
    original_name: str
    upload_path: str
    processed_path: str
    text_path: str
    size: int
    is_image: bool


class PhoneToDashboardFlow:
    """Standalone orchestration for phone capture processing."""

    def __init__(self, source: str = "phone") -> None:
        self.source = source
        self.dirs = get_data_dirs()
        self.upload_dir = self.dirs["UPLOAD_DIR"]
        self.processed_dir = self.dirs["PROCESSED_DIR"]
        self.text_dir = self.dirs["TEXT_DIR"]
        self.ocr_data_dir = self.dirs["OCR_DATA_DIR"]
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.text_dir, exist_ok=True)
        os.makedirs(self.ocr_data_dir, exist_ok=True)

    def store_upload(self, file: FileStorage) -> StoredUpload:
        original_name = secure_filename(file.filename or "")
        unique_name = f"{uuid.uuid4().hex}_{original_name}"
        upload_path = os.path.join(self.upload_dir, unique_name)
        processed_path = os.path.join(self.processed_dir, unique_name)
        text_path = os.path.join(self.text_dir, f"{os.path.splitext(unique_name)[0]}.txt")
        file.save(upload_path)
        size = os.path.getsize(upload_path)
        return StoredUpload(
            filename=unique_name,
            original_name=original_name,
            upload_path=upload_path,
            processed_path=processed_path,
            text_path=text_path,
            size=size,
            is_image=is_image_extension(unique_name),
        )

    def emit_new_file(self, artifact: StoredUpload, processing: bool = True) -> None:
        socketio.emit(
            "new_file",
            {
                "filename": artifact.filename,
                "timestamp": datetime.now().isoformat(),
                "processing": processing,
                "source": self.source,
            },
        )

    def start_processing(self, artifact: StoredUpload, enhancement_params: dict[str, Any]) -> str:
        if artifact.is_image:
            _emit_processing_progress(
                artifact.filename,
                {
                    "step": 0,
                    "total_steps": 12,
                    "stage_name": "Queued",
                    "message": "Starting 12-step processing",
                },
            )
            if IS_CLOUD:
                # Serverless functions are frozen as soon as the response is sent, so a background
                # thread would never finish. Process inside the request instead.
                self._run_image_pipeline_async(artifact, enhancement_params)
                return "enhanced_12_step"

            worker = threading.Thread(
                target=self._run_image_pipeline_async,
                args=(artifact, enhancement_params),
                daemon=True,
            )
            worker.start()
            return "enhanced_12_step"

        # Non-image files are copied into processed store immediately.
        shutil.copy2(artifact.upload_path, artifact.processed_path)
        socketio.emit(
            "processing_complete",
            {
                "filename": artifact.filename,
                "has_text": False,
                "source": self.source,
                "pipeline": "direct_copy",
            },
        )
        return "direct_copy"

    def _run_image_pipeline_async(self, artifact: StoredUpload, enhancement_params: dict[str, Any]) -> None:
        try:
            _debug_log(
                "run1",
                "H3",
                "backend/app/features/phone/upload/phone_to_dashboard_flow.py:_run_image_pipeline_async:start",
                "pipeline_start",
                {
                    "filename": artifact.filename,
                    "upload_path": artifact.upload_path,
                    "processed_path": artifact.processed_path,
                },
            )

            from app.modules.pipeline.enhanced import EnhancedDocumentPipeline

            pipeline = EnhancedDocumentPipeline(
                storage_dir=os.path.dirname(artifact.processed_path),
                emit_callback=lambda data: _emit_processing_progress(artifact.filename, data),
            )
            success, extracted_text, stats = pipeline.process_complete_pipeline(
                input_path=artifact.upload_path,
                output_path=artifact.processed_path,
                enhancement_params=enhancement_params,
            )

            if not success:
                error_msg = extracted_text or "Pipeline failed"
                _debug_log(
                    "run1",
                    "H3",
                    "backend/app/features/phone/upload/phone_to_dashboard_flow.py:_run_image_pipeline_async:failure",
                    "pipeline_failed",
                    {"filename": artifact.filename, "error": error_msg},
                )
                socketio.emit("processing_error", {"filename": artifact.filename, "error": error_msg})
                logger.error("Pipeline failed for %s: %s", artifact.filename, error_msg)
                return

            has_text = bool(extracted_text and extracted_text.strip())
            if has_text:
                with open(artifact.text_path, "w", encoding="utf-8") as f:
                    f.write(extracted_text)

            ocr_result = (stats or {}).get("ocr_result")
            # An empty result usually means OCR failed (e.g. the model was overloaded). Do not save it,
            # otherwise it would be served as the final answer and block a later retry.
            if isinstance(ocr_result, dict) and ocr_result.get("word_count"):
                ocr_json_path = os.path.join(
                    self.ocr_data_dir,
                    f"{os.path.splitext(artifact.filename)[0]}_ocr.json",
                )
                with open(ocr_json_path, "w", encoding="utf-8") as f:
                    json.dump(ocr_result, f, ensure_ascii=True, indent=2)

            socketio.emit(
                "processing_complete",
                {
                    "filename": artifact.filename,
                    "has_text": has_text,
                    "source": self.source,
                    "pipeline": "enhanced_12_step",
                    "stats": {
                        "total_steps": (stats or {}).get("total_steps", 12),
                        "success": (stats or {}).get("success", True),
                        "ocr_engine": (ocr_result or {}).get("engine"),
                    },
                },
            )

            processed_exists = os.path.isfile(artifact.processed_path)
            processed_size = os.path.getsize(artifact.processed_path) if processed_exists else 0
            step7 = (stats or {}).get("steps", {}).get("step_7", {})
            _debug_log(
                "run1",
                "H3",
                "backend/app/features/phone/upload/phone_to_dashboard_flow.py:_run_image_pipeline_async:complete",
                "pipeline_complete_emitted",
                {
                    "filename": artifact.filename,
                    "processed_exists": processed_exists,
                    "processed_size": processed_size,
                    "step7": step7,
                },
            )
            logger.info("[OK] 12-step processing complete for %s (has_text=%s)", artifact.filename, has_text)
        except Exception as exc:
            logger.error("Background pipeline error for %s: %s", artifact.filename, exc)
            socketio.emit("processing_error", {"filename": artifact.filename, "error": str(exc)})
