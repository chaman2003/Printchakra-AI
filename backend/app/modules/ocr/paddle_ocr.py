"""
PaddleOCR Module
Advanced OCR with bounding boxes, confidence scores, and Ollama post-processing
"""

# CRITICAL: Set environment variables BEFORE any imports
import os
# Disable all connectivity checks for faster startup
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'  # Skip slow connectivity check
os.environ['PADDLEX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'  # PaddleX specific
os.environ['HUB_HOME'] = os.path.expanduser('~/.paddlex')  # Use local cache
os.environ['FLAGS_check_nan_inf'] = '0'  # Disable NaN checking for speed
os.environ['GLOG_v'] = '0'  # Reduce logging verbosity
os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'  # Memory optimization

import json
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import requests
import re

import cv2
import numpy as np
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)

# Lazy load PaddleOCR to avoid import overhead
_paddle_ocr_instance = None


def _detect_paddle_device() -> str:
    """
    Detect the best available device for PaddleOCR.
    PaddleOCR 3.3+ uses: 'gpu', 'cpu', 'xpu', 'npu', 'mlu', etc.
    NOT 'cuda' - that's PyTorch syntax.
    """
    try:
        import paddle
        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            logger.info("[OK] PaddleOCR will use GPU acceleration")
            return 'gpu'
        else:
            logger.info("[INFO] PaddleOCR will use CPU (GPU not available in PaddlePaddle)")
            return 'cpu'
    except Exception as e:
        logger.debug(f"[DEBUG] Paddle device detection failed: {e}, defaulting to CPU")
        return 'cpu'


def get_paddle_ocr():
    """Lazy load PaddleOCR instance with proper device configuration for v3.3+"""
    global _paddle_ocr_instance
    if _paddle_ocr_instance is None:
        try:
            import sys
            from io import StringIO
            
            # Suppress PaddleOCR startup messages and connectivity checks
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            
            try:
                from paddleocr import PaddleOCR
                
                # Detect best device (gpu/cpu)
                device = _detect_paddle_device()
                
                # Initialize PaddleOCR v3.3+ API
                # NOTE: use_angle_cls is now part of the pipeline, not a separate parameter
                _paddle_ocr_instance = PaddleOCR(
                    lang='en',  # Primary language
                    det_db_thresh=0.3,  # Text detection threshold
                    det_db_box_thresh=0.5,  # Box threshold
                    rec_batch_num=6,  # Recognition batch size
                    device=device,  # 'gpu' or 'cpu' (NOT 'cuda')
                )
            finally:
                # Restore stdout/stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            
            logger.info(f"[OK] PaddleOCR v3.3+ initialized successfully on {device.upper()}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize PaddleOCR: {e}")
            # Try CPU fallback with minimal options
            try:
                logger.info("[INFO] Attempting CPU-only PaddleOCR initialization...")
                from paddleocr import PaddleOCR
                _paddle_ocr_instance = PaddleOCR(
                    lang='en',
                    device='cpu',
                )
                logger.info("[OK] PaddleOCR initialized on CPU (fallback)")
            except Exception as fallback_error:
                logger.error(f"[ERROR] PaddleOCR CPU fallback also failed: {fallback_error}")
                raise
    return _paddle_ocr_instance


class OCRResult:
    """Structured OCR result with bounding boxes and confidence"""
    
    def __init__(self):
        self.raw_results: List[Dict] = []
        self.structured_units: List[Dict] = []
        self.full_text: str = ""
        self.derived_title: str = ""
        self.confidence_avg: float = 0.0
        self.word_count: int = 0
        self.timestamp: str = ""
        self.processing_time_ms: float = 0.0
        self.image_dimensions: Tuple[int, int] = (0, 0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_results": self.raw_results,
            "structured_units": self.structured_units,
            "full_text": self.full_text,
            "derived_title": self.derived_title,
            "confidence_avg": self.confidence_avg,
            "word_count": self.word_count,
            "timestamp": self.timestamp,
            "processing_time_ms": self.processing_time_ms,
            "image_dimensions": list(self.image_dimensions),
        }


class PaddleOCRProcessor:
    """
    PaddleOCR processor with Ollama post-processing for text structuring
    """
    
    OLLAMA_URL = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = "phi3:mini"  # Using phi3:mini for text structuring
    
    def __init__(self, ocr_data_dir: str):
        """
        Initialize OCR processor
        
        Args:
            ocr_data_dir: Directory to store OCR results
        """
        self.ocr_data_dir = ocr_data_dir
        os.makedirs(ocr_data_dir, exist_ok=True)
        logger.info(f"[OK] PaddleOCRProcessor initialized, data dir: {ocr_data_dir}")
    
    def process_image(self, image_path: str) -> OCRResult:
        """
        Run PaddleOCR on an image and return structured results
        
        Args:
            image_path: Path to the image file
            
        Returns:
            OCRResult with all extracted data
        """
        import time
        start_time = time.time()
        
        result = OCRResult()
        result.timestamp = datetime.now().isoformat()
        
        try:
            # Validate image exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Load image and get dimensions
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            result.image_dimensions = (img.shape[1], img.shape[0])  # width, height
            
            logger.info(f"[OCR] Processing image: {os.path.basename(image_path)}")
            logger.info(f"[OCR] Image dimensions: {result.image_dimensions}")
            
            # Run PaddleOCR v3.3+ (cls parameter removed - angle classification is automatic)
            # Suppress any connectivity check messages during OCR processing
            import sys
            from io import StringIO
            
            ocr = get_paddle_ocr()
            
            # Suppress output from ocr.ocr() call to prevent connectivity check messages
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            
            try:
                ocr_output = ocr.ocr(image_path)
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            
            print(f"[DEBUG] OCR raw output type: {type(ocr_output)}, has data: {bool(ocr_output)}")
            
            # Extract results from OCRResult object (dict-like structure in PaddleX 3.x)
            ocr_results = []
            if ocr_output and len(ocr_output) > 0:
                page_result = ocr_output[0]
                print(f"[DEBUG] ocr_output[0] type: {type(page_result)}")
                
                # OCRResult is dict-like - use bracket access, not getattr
                # Keys: rec_texts, rec_scores, dt_polys, rec_polys, etc.
                try:
                    # Try dict-style access (PaddleX 3.x OCRResult is dict-like)
                    if hasattr(page_result, 'keys'):
                        keys = list(page_result.keys())
                        print(f"[DEBUG] OCRResult keys: {keys}")
                        
                        # Get the actual data using dict access
                        rec_texts = page_result.get('rec_texts', [])
                        rec_scores = page_result.get('rec_scores', [])
                        dt_polys = page_result.get('dt_polys', [])
                        
                        print(f"[DEBUG] rec_texts ({len(rec_texts) if rec_texts else 0}): {rec_texts[:2] if rec_texts else 'empty'}...")
                        print(f"[DEBUG] rec_scores ({len(rec_scores) if rec_scores else 0}): {rec_scores[:2] if rec_scores else 'empty'}...")
                        print(f"[DEBUG] dt_polys ({len(dt_polys) if dt_polys else 0})")
                        
                        if rec_texts and dt_polys:
                            # Build results from parallel arrays
                            for i in range(len(rec_texts)):
                                text = rec_texts[i] if i < len(rec_texts) else ""
                                bbox = dt_polys[i] if i < len(dt_polys) else None
                                score = rec_scores[i] if rec_scores and i < len(rec_scores) else 0.0
                                if text and bbox is not None:
                                    ocr_results.append({
                                        'text': str(text).strip(),
                                        'bbox': bbox,
                                        'confidence': float(score) if score else 0.0
                                    })
                            print(f"[DEBUG] Built {len(ocr_results)} results from parallel arrays")
                    else:
                        print(f"[DEBUG] OCRResult has no 'keys' method, trying direct attribute access")
                        # Fallback to attribute access
                        rec_texts = getattr(page_result, 'rec_texts', None) or getattr(page_result, 'rec_text', [])
                        rec_scores = getattr(page_result, 'rec_scores', None) or getattr(page_result, 'rec_score', [])
                        dt_polys = getattr(page_result, 'dt_polys', [])
                        
                        if rec_texts and dt_polys:
                            for i in range(len(rec_texts)):
                                text = rec_texts[i] if i < len(rec_texts) else ""
                                bbox = dt_polys[i] if i < len(dt_polys) else None
                                score = rec_scores[i] if rec_scores and i < len(rec_scores) else 0.0
                                if text and bbox is not None:
                                    ocr_results.append({
                                        'text': str(text).strip(),
                                        'bbox': bbox,
                                        'confidence': float(score) if score else 0.0
                                    })
                            print(f"[DEBUG] Built {len(ocr_results)} from attribute access")
                except Exception as extract_err:
                    print(f"[DEBUG] Extraction error: {extract_err}")
            
            if not ocr_results:
                logger.warning(f"[OCR] No text detected in image (empty or no extractable results)")
                result.full_text = ""
                result.derived_title = "Untitled Document"
                result.processing_time_ms = (time.time() - start_time) * 1000
                return result
            
            # Process OCR results
            raw_results = []
            all_text_parts = []
            total_confidence = 0.0
            skipped_entries = 0
            
            for idx, line in enumerate(ocr_results):
                parsed = self._parse_ocr_line(line)
                if not parsed:
                    print(f"[DEBUG] Entry {idx} skipped - raw type: {type(line)}, content: {str(line)[:100]}")
                    skipped_entries += 1
                    continue

                raw_results.append(parsed)
                all_text_parts.append(parsed["text"])
                total_confidence += parsed["confidence"]

            if skipped_entries:
                logger.info(f"[OCR] Skipped {skipped_entries} entries, processed {len(raw_results)} successfully")
            logger.info(f"[OCR] Processed {len(raw_results)} text regions")
            
            result.raw_results = raw_results
            result.full_text = " ".join(all_text_parts)
            result.word_count = len(result.full_text.split())
            result.confidence_avg = round(total_confidence / len(raw_results), 4) if raw_results else 0.0
            
            logger.info(f"[OCR] Detected {len(raw_results)} text regions")
            logger.info(f"[OCR] Average confidence: {result.confidence_avg:.2%}")
            
            # Sort results by reading order (top-to-bottom, left-to-right)
            result.raw_results = self._sort_by_reading_order(result.raw_results)
            
            # Post-process with Ollama to structure text
            result.structured_units = self._structure_with_ollama(result.raw_results)
            
            # Derive document title
            result.derived_title = self._derive_title_with_ollama(result.full_text, result.raw_results)
            
            result.processing_time_ms = (time.time() - start_time) * 1000
            logger.info(f"[OCR] Processing complete in {result.processing_time_ms:.0f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"[OCR] Error processing image: {e}")
            traceback.print_exc()
            result.processing_time_ms = (time.time() - start_time) * 1000
            result.derived_title = "Error Processing Document"
            return result
    
    def _parse_ocr_line(self, line: Any) -> Optional[Dict]:
        """Safely parse a PaddleOCR line into text, confidence, and bbox.
        
        Handles multiple formats:
        - Dict with 'text', 'bbox', 'confidence' keys (our internal format)
        - [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], [text, confidence, angle]
        - [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], [text, confidence]
        - paddleocr.results.TextResult objects with box, text, score attributes
        """
        try:
            if not line:
                return None

            text = ""
            confidence = 0.0
            bbox = None

            # Handle our internal dict format (from parallel arrays extraction)
            if isinstance(line, dict):
                text = str(line.get('text', '')).strip()
                confidence = float(line.get('confidence', 0.0))
                bbox = line.get('bbox')
                if text and bbox is not None:
                    # bbox is already in the right format (list of 4 points)
                    try:
                        bbox_list = list(bbox) if not isinstance(bbox, list) else bbox
                        if len(bbox_list) == 4:
                            x_coords = [float(p[0]) for p in bbox_list]
                            y_coords = [float(p[1]) for p in bbox_list]
                            bbox_rect = {
                                "x": int(min(x_coords)),
                                "y": int(min(y_coords)),
                                "width": int(max(x_coords) - min(x_coords)) or 1,
                                "height": int(max(y_coords) - min(y_coords)) or 1,
                                "points": [[int(p[0]), int(p[1])] for p in bbox_list],
                            }
                            return {
                                "text": text,
                                "confidence": round(max(0.0, min(1.0, confidence)), 4),
                                "bbox": bbox_rect,
                            }
                    except Exception as e:
                        print(f"[DEBUG] Dict bbox parsing failed: {e}")
                return None

            # Handle paddleocr TextResult object
            if hasattr(line, 'box') and hasattr(line, 'text'):
                bbox = line.box
                text = str(line.text).strip() if line.text else ""
                if hasattr(line, 'score'):
                    try:
                        confidence = float(line.score) if line.score is not None else 0.0
                    except (TypeError, ValueError):
                        confidence = 0.0
            
            # Handle list/tuple format: [bbox, [text, conf, ...]]
            elif isinstance(line, (list, tuple)) and len(line) >= 2:
                bbox = line[0]
                text_data = line[1]
                
                # Extract text and confidence
                if isinstance(text_data, (list, tuple)):
                    if len(text_data) > 0:
                        text = str(text_data[0]).strip() if text_data[0] is not None else ""
                    if len(text_data) > 1:
                        try:
                            confidence = float(text_data[1]) if text_data[1] is not None else 0.0
                        except (TypeError, ValueError):
                            confidence = 0.0
                elif isinstance(text_data, str):
                    text = text_data.strip()
                else:
                    text = str(text_data).strip() if text_data else ""
            
            else:
                print(f"[DEBUG] Unknown line format: {type(line)}")
                return None

            # Validate text
            if not text:
                return None

            # Extract/validate bbox coordinates
            if bbox is None:
                return None
            
            try:
                bbox_list = list(bbox) if not isinstance(bbox, list) else bbox
                
                if len(bbox_list) != 4:
                    return None

                x_coords = []
                y_coords = []
                for point in bbox_list:
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        x_coords.append(float(point[0]))
                        y_coords.append(float(point[1]))
                    else:
                        return None
                
                if len(x_coords) != 4 or len(y_coords) != 4:
                    return None

                bbox_rect = {
                    "x": int(min(x_coords)),
                    "y": int(min(y_coords)),
                    "width": int(max(x_coords) - min(x_coords)) or 1,
                    "height": int(max(y_coords) - min(y_coords)) or 1,
                    "points": [[int(point[0]), int(point[1])] for point in bbox_list],
                }

                return {
                    "text": text,
                    "confidence": round(max(0.0, min(1.0, confidence)), 4),
                    "bbox": bbox_rect,
                }
            except Exception as coord_error:
                print(f"[DEBUG] Coordinate extraction failed: {coord_error}")
                return None
                
        except Exception as parse_err:
            print(f"[DEBUG] Line parsing failed: {parse_err}, type: {type(line)}")
            return None
    
    def _sort_by_reading_order(self, results: List[Dict]) -> List[Dict]:
        """
        Sort OCR results by reading order (top-to-bottom, left-to-right)
        Groups text into lines based on vertical proximity
        """
        if not results:
            return results
        
        # Sort by Y first (top to bottom), then X (left to right)
        # Use a tolerance for Y to group items on the same line
        line_tolerance = 20  # pixels
        
        # Group by approximate Y position
        lines = []
        current_line = []
        current_y = None
        
        # Sort by Y position first
        sorted_by_y = sorted(results, key=lambda r: r["bbox"]["y"])
        
        for item in sorted_by_y:
            item_y = item["bbox"]["y"]
            
            if current_y is None:
                current_y = item_y
                current_line = [item]
            elif abs(item_y - current_y) <= line_tolerance:
                current_line.append(item)
            else:
                # Sort current line by X and add to lines
                current_line.sort(key=lambda r: r["bbox"]["x"])
                lines.extend(current_line)
                current_line = [item]
                current_y = item_y
        
        # Don't forget the last line
        if current_line:
            current_line.sort(key=lambda r: r["bbox"]["x"])
            lines.extend(current_line)
        
        return lines
    
    def _structure_with_ollama(self, raw_results: List[Dict]) -> List[Dict]:
        """
        Use Ollama phi4-mini to group words into sentences/lines
        Returns structured text units with spatial information
        """
        if not raw_results:
            return []
        
        # Build text with position info for Ollama
        text_with_positions = []
        for i, item in enumerate(raw_results):
            text_with_positions.append(f"{i}: \"{item['text']}\" (y={item['bbox']['y']})")
        
        prompt = f"""Analyze these OCR text fragments and group them into logical lines/sentences.
Each fragment has an index, text, and Y-position.

Text fragments:
{chr(10).join(text_with_positions)}

Group these into logical units (lines, sentences, or paragraphs).
For each unit, provide:
- The combined text
- Which fragment indices it contains
- A unit type (title, heading, paragraph, list_item, table_cell, footer, other)

Respond in JSON format only:
{{"units": [
  {{"text": "combined text", "indices": [0, 1, 2], "type": "paragraph"}},
  ...
]}}"""

        try:
            response = requests.post(
                self.OLLAMA_URL,
                json={
                    "model": self.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 2000,
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                
                # Extract JSON from response
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    parsed = json.loads(json_match.group())
                    units = parsed.get("units", [])
                    
                    # Enrich units with bbox info
                    structured = []
                    for unit in units:
                        indices = unit.get("indices", [])
                        if indices:
                            # Calculate combined bounding box
                            boxes = [raw_results[i]["bbox"] for i in indices if i < len(raw_results)]
                            if boxes:
                                combined_bbox = {
                                    "x": min(b["x"] for b in boxes),
                                    "y": min(b["y"] for b in boxes),
                                    "width": max(b["x"] + b["width"] for b in boxes) - min(b["x"] for b in boxes),
                                    "height": max(b["y"] + b["height"] for b in boxes) - min(b["y"] for b in boxes),
                                }
                                avg_confidence = sum(raw_results[i]["confidence"] for i in indices if i < len(raw_results)) / len(indices)
                                
                                structured.append({
                                    "text": unit.get("text", ""),
                                    "type": unit.get("type", "paragraph"),
                                    "bbox": combined_bbox,
                                    "confidence": round(avg_confidence, 4),
                                    "word_indices": indices,
                                })
                    
                    return structured if structured else self._fallback_structure(raw_results)
            
            logger.warning("[OCR] Ollama structuring failed, using fallback")
            return self._fallback_structure(raw_results)
            
        except Exception as e:
            logger.debug(f"[DEBUG] Ollama structuring unavailable (Ollama not running?): {type(e).__name__}, using fallback")
            return self._fallback_structure(raw_results)
    
    def _fallback_structure(self, raw_results: List[Dict]) -> List[Dict]:
        """
        Fallback structuring when Ollama is unavailable
        Groups text by vertical proximity
        """
        if not raw_results:
            return []
        
        line_tolerance = 25
        lines = []
        current_line = []
        current_y = None
        
        for item in raw_results:
            item_y = item["bbox"]["y"]
            
            if current_y is None or abs(item_y - current_y) <= line_tolerance:
                current_line.append(item)
                if current_y is None:
                    current_y = item_y
            else:
                if current_line:
                    lines.append(current_line)
                current_line = [item]
                current_y = item_y
        
        if current_line:
            lines.append(current_line)
        
        # Convert to structured units
        structured = []
        for i, line in enumerate(lines):
            text = " ".join(item["text"] for item in line)
            boxes = [item["bbox"] for item in line]
            avg_confidence = sum(item["confidence"] for item in line) / len(line)
            
            combined_bbox = {
                "x": min(b["x"] for b in boxes),
                "y": min(b["y"] for b in boxes),
                "width": max(b["x"] + b["width"] for b in boxes) - min(b["x"] for b in boxes),
                "height": max(b["y"] + b["height"] for b in boxes) - min(b["y"] for b in boxes),
            }
            
            # Guess type based on position and text
            unit_type = "paragraph"
            if i == 0 and len(text.split()) <= 10:
                unit_type = "title"
            elif text.startswith(("•", "-", "*", "1.", "2.", "3.")):
                unit_type = "list_item"
            
            structured.append({
                "text": text,
                "type": unit_type,
                "bbox": combined_bbox,
                "confidence": round(avg_confidence, 4),
                "word_indices": list(range(len(line))),
            })
        
        return structured
    
    def _derive_title_with_ollama(self, full_text: str, raw_results: List[Dict]) -> str:
        """
        Use Ollama to derive a document title from OCR content
        """
        if not full_text.strip():
            return "Untitled Document"
        
        # Get first few lines for context
        first_lines = full_text[:500]
        
        prompt = f"""Based on this document text, derive a concise, descriptive title (3-8 words).
The title should capture the main subject/purpose of the document.

Document text:
{first_lines}

Respond with ONLY the title, no explanation or quotes."""

        try:
            response = requests.post(
                self.OLLAMA_URL,
                json={
                    "model": self.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 50,
                    }
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                title = result.get("response", "").strip()
                # Clean up title
                title = title.strip('"\'')
                title = re.sub(r'^(title:|document title:)\s*', '', title, flags=re.IGNORECASE)
                if title and len(title) <= 100:
                    return title
            
            # Fallback: use first text as title
            return self._fallback_title(raw_results)
            
        except Exception as e:
            logger.debug(f"[DEBUG] Ollama title derivation unavailable: {type(e).__name__}, using fallback")
            return self._fallback_title(raw_results)
    
    def _fallback_title(self, raw_results: List[Dict]) -> str:
        """Fallback title derivation"""
        if not raw_results:
            return "Untitled Document"
        
        # Use first text block as title
        first_text = raw_results[0].get("text", "")
        if first_text:
            # Truncate if too long
            words = first_text.split()[:8]
            return " ".join(words)
        
        return "Scanned Document"
    
    def generate_filename_from_ocr(self, result: OCRResult, timestamp: str = None) -> str:
        """
        Generate a filename based on OCR content using Ollama phi3:mini
        
        Args:
            result: OCR result object
            timestamp: Optional timestamp to include
            
        Returns:
            Generated filename (without extension)
        """
        if not result.full_text.strip():
            return f"scan_{timestamp}" if timestamp else "scan_untitled"
        
        # Use derived_title if available, or derive one
        title = result.derived_title if result.derived_title and result.derived_title != "Untitled Document" else None
        
        if not title:
            title = self._derive_filename_with_ollama(result.full_text)
        
        # Clean up title for filename
        filename = self._sanitize_filename(title)
        
        # Add timestamp if provided
        if timestamp:
            filename = f"{filename}_{timestamp}"
        
        return filename
    
    def _derive_filename_with_ollama(self, full_text: str) -> str:
        """
        Use Ollama phi3:mini to derive a short filename-friendly string from OCR content
        """
        # Get first portion of text
        first_text = full_text[:500]
        
        prompt = f"""Based on this document text, generate a SHORT filename (2-5 words, no spaces use underscores).
The filename should describe the document's content/purpose.
Examples: invoice_march_2024, student_id_card, electricity_bill, medical_report

Document text:
{first_text}

Respond with ONLY the filename (lowercase, underscores, no extension)."""

        try:
            response = requests.post(
                self.OLLAMA_URL,
                json={
                    "model": self.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 30,
                    }
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                filename = result.get("response", "").strip().lower()
                # Clean up
                filename = re.sub(r'[^a-z0-9_]', '_', filename)
                filename = re.sub(r'_+', '_', filename)
                filename = filename.strip('_')
                if filename and 3 <= len(filename) <= 50:
                    logger.info(f"[OCR] Ollama generated filename: {filename}")
                    return filename
            
            return self._fallback_filename_from_text(first_text)
            
        except Exception as e:
            logger.warning(f"[OCR] Ollama filename generation error: {e}")
            return self._fallback_filename_from_text(first_text)
    
    def _fallback_filename_from_text(self, text: str) -> str:
        """Fallback filename generation from first words"""
        words = re.findall(r'[a-zA-Z]+', text)[:3]
        if words:
            return "_".join(w.lower() for w in words)
        return "scanned_document"
    
    def _sanitize_filename(self, title: str) -> str:
        """Convert a title to a safe filename"""
        if not title:
            return "untitled"
        # Convert to lowercase, replace spaces with underscores
        filename = title.lower().strip()
        filename = re.sub(r'[^a-z0-9]+', '_', filename)
        filename = re.sub(r'_+', '_', filename)
        filename = filename.strip('_')
        # Limit length
        if len(filename) > 50:
            filename = filename[:50].rsplit('_', 1)[0]
        return filename or "untitled"
    
    def save_result(self, filename: str, result: OCRResult) -> Optional[str]:
        """
        Save OCR result to JSON file
        
        Args:
            filename: Original image filename
            result: OCR result to save
            
        Returns:
            Path to saved JSON file, or None if result is invalid/error
        """
        # Don't save error results with no text
        if result.word_count == 0 and result.derived_title in ["Error Processing Document", "Untitled Document"]:
            logger.warning(f"[OCR] Skipping save for error/empty result: {filename}")
            return None
            
        # Create filename for OCR result
        base_name = os.path.splitext(filename)[0]
        json_filename = f"{base_name}_ocr.json"
        json_path = os.path.join(self.ocr_data_dir, json_filename)
        
        # Save result
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"[OCR] Result saved to: {json_path}")
        return json_path
    
    def load_result(self, filename: str) -> Optional[Dict]:
        """
        Load existing OCR result for a file
        
        Args:
            filename: Original image filename
            
        Returns:
            OCR result dict or None if not found or invalid
        """
        base_name = os.path.splitext(filename)[0]
        json_filename = f"{base_name}_ocr.json"
        json_path = os.path.join(self.ocr_data_dir, json_filename)
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                
                # Validate result - skip error/empty results
                if result.get('word_count', 0) == 0 and result.get('derived_title') in ["Error Processing Document", "Untitled Document"]:
                    logger.warning(f"[OCR] Skipping invalid/error result for: {filename}")
                    # Delete the bad result file
                    try:
                        os.remove(json_path)
                        logger.info(f"[OCR] Deleted invalid result file: {json_path}")
                    except:
                        pass
                    return None
                    
                return result
            except json.JSONDecodeError as e:
                logger.error(f"[OCR] Failed to parse JSON for {filename}: {e}")
                return None
        
        return None
    
    def has_ocr_result(self, filename: str) -> bool:
        """Check if valid OCR result exists for a file"""
        # Use load_result which validates the result
        result = self.load_result(filename)
        return result is not None and result.get('word_count', 0) > 0


# Global processor instance
_ocr_processor: Optional[PaddleOCRProcessor] = None


def get_ocr_processor(ocr_data_dir: str = None) -> PaddleOCRProcessor:
    """Get or create global OCR processor"""
    global _ocr_processor
    if _ocr_processor is None:
        if ocr_data_dir is None:
            raise ValueError("ocr_data_dir must be provided on first call")
        _ocr_processor = PaddleOCRProcessor(ocr_data_dir)
    return _ocr_processor
