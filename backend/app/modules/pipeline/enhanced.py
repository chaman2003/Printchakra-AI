"""
Enhanced Document Processing Pipeline - Notebook Integration
Implements all 12 steps from processing.ipynb with adjustable parameters
"""

import logging
import os
import sys
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Optional: not installed on serverless hosting (OCR runs through Gemini there).
try:
    import pytesseract
except ImportError:
    pytesseract = None

_tesseract_cmd = os.environ.get("TESSERACT_CMD")
if _tesseract_cmd and pytesseract is not None:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd

from ..image import ImageEnhancer

logger = logging.getLogger(__name__)


class EnhancedDocumentPipeline:
    """
    Complete document processing pipeline matching notebook implementation

    12-Step Pipeline:
    1. Load Original Image
    2. Downscale for Detection
    3. Convert to Grayscale (Detection)
    4. Gaussian Blur
    5. Canny Edge Detection
    6. Find Document Contour
    7. Perspective Transform (Crop)
    8. Grayscale & Downscale for Processing
    9. Morphological Background Estimation
    10. Smooth and Upscale Background
    11. Shadow Removal (Enhancement)
    12. OCR Text Extraction
    """

    _ocr_engine = None
    _ocr_device = "cpu"
    _ocr_lock = threading.Lock()

    def __init__(self, storage_dir: str = None, emit_callback=None):
        """
        Initialize the pipeline

        Args:
            storage_dir: Directory to store processed images
            emit_callback: Callback function for progress updates (for Socket.IO)
        """
        self.storage_dir = storage_dir or os.getcwd()
        self.emit_callback = emit_callback
        self.enhancer = ImageEnhancer()

        logger.info(f"EnhancedDocumentPipeline initialized with storage_dir: {self.storage_dir}")

    def emit_progress(self, step: int, total_steps: int, stage_name: str, message: str) -> None:
        """
        Emit progress update

        Args:
            step: Current step number
            total_steps: Total steps
            stage_name: Name of current stage
            message: Progress message
        """
        if self.emit_callback:
            try:
                self.emit_callback(
                    {
                        "step": step,
                        "total_steps": total_steps,
                        "stage_name": stage_name,
                        "message": message,
                    }
                )
            except Exception as e:
                logger.warning(f"Emit callback error: {e}")

        logger.info(f"[STEP {step}/{total_steps}] {stage_name} - {message}")

    def _remove_shadows_keep_color(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Notebook-style shadow removal while preserving color output."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        max_dim = 800
        bg_scale = min(max_dim / max(h, w), 1.0)
        if bg_scale < 1.0:
            small_gray = cv2.resize(gray, (int(w * bg_scale), int(h * bg_scale)))
        else:
            small_gray = gray

        sh, sw = small_gray.shape
        k = max(sh, sw) // 8
        k = k if k % 2 == 1 else k + 1
        k = max(31, min(k, 127))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bg_raw = cv2.morphologyEx(small_gray, cv2.MORPH_CLOSE, kernel)
        bg_smoothed = cv2.GaussianBlur(bg_raw, (k, k), 0)

        if bg_scale < 1.0:
            bg = cv2.resize(bg_smoothed, (w, h), interpolation=cv2.INTER_LINEAR)
            bg = cv2.GaussianBlur(bg, (31, 31), 0)
        else:
            bg = bg_smoothed

        bg_safe = bg.astype(np.float32)
        mask = bg_safe > 10

        result_color = np.zeros_like(image_bgr, dtype=np.float32)
        for c in range(3):
            channel = image_bgr[:, :, c].astype(np.float32)
            result_color[:, :, c][mask] = (channel[mask] / bg_safe[mask]) * 255.0

        enhanced_color = np.clip(result_color, 0, 255).astype(np.uint8)
        enhanced_gray = cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2GRAY)
        return enhanced_color, enhanced_gray

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """Order contour points as top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Apply the notebook's perspective transform."""
        rect = self._order_points(pts.astype("float32"))
        (tl, tr, br, bl) = rect

        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        dst = np.array(
            [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, matrix, (max_width, max_height))

    @staticmethod
    def _is_reasonable_document_quad(
        contour: np.ndarray,
        image_shape: Tuple[int, int],
        min_area_ratio: float = 0.22,
        max_area_ratio: float = 0.97,
    ) -> bool:
        """Validate that a 4-point contour looks like a full-page document."""
        img_h, img_w = image_shape[:2]
        if img_h <= 0 or img_w <= 0:
            return False

        contour_area = float(cv2.contourArea(contour))
        image_area = float(img_h * img_w)
        if image_area <= 0:
            return False
        area_ratio = contour_area / image_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            return False

        # Reject highly elongated or tiny projected rectangles.
        x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
        if w <= 0 or h <= 0:
            return False
        aspect = max(w / max(h, 1e-6), h / max(w, 1e-6))
        if aspect > 3.0:
            return False

        return True

    @staticmethod
    def _quad_confidence(contour: np.ndarray, image_shape: Tuple[int, int]) -> float:
        """Score how likely a quad is to be the actual document boundary."""
        img_h, img_w = image_shape[:2]
        if img_h <= 0 or img_w <= 0:
            return 0.0

        area = float(cv2.contourArea(contour))
        image_area = float(img_h * img_w)
        if image_area <= 0:
            return 0.0
        area_ratio = area / image_area

        # Prefer medium-large quads (ID cards/sheets), avoid full-frame borders.
        area_target = 0.48
        area_score = max(0.0, 1.0 - abs(area_ratio - area_target) / area_target)

        rect = EnhancedDocumentPipeline._order_points(contour.astype("float32"))
        tl, tr, br, bl = rect
        edges = [
            np.linalg.norm(tr - tl),
            np.linalg.norm(br - tr),
            np.linalg.norm(bl - br),
            np.linalg.norm(tl - bl),
        ]
        if min(edges) <= 1e-6:
            return 0.0

        # Penalize highly skewed/degenerate quads.
        edge_ratio = min(edges) / max(edges)
        edge_score = max(0.0, min(1.0, edge_ratio * 2.0))

        # Right-angle consistency using cosine similarity (0 is ideal).
        def _abs_cos(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
            ab = a - b
            cb = c - b
            denom = float(np.linalg.norm(ab) * np.linalg.norm(cb))
            if denom <= 1e-6:
                return 1.0
            return abs(float(np.dot(ab, cb)) / denom)

        angle_err = np.mean(
            [
                _abs_cos(tl, tr, br),
                _abs_cos(tr, br, bl),
                _abs_cos(br, bl, tl),
                _abs_cos(bl, tl, tr),
            ]
        )
        angle_score = max(0.0, 1.0 - min(1.0, angle_err * 2.0))

        # Prefer candidates near frame center (handheld captures usually center target).
        centroid = np.mean(contour.astype(np.float32), axis=0)
        cx, cy = float(centroid[0]), float(centroid[1])
        img_cx, img_cy = img_w / 2.0, img_h / 2.0
        dist = np.sqrt((cx - img_cx) ** 2 + (cy - img_cy) ** 2)
        max_dist = np.sqrt((img_w / 2.0) ** 2 + (img_h / 2.0) ** 2)
        center_score = max(0.0, 1.0 - (dist / max(max_dist, 1e-6)))

        return float(0.45 * area_score + 0.18 * edge_score + 0.22 * angle_score + 0.15 * center_score)

    @staticmethod
    def _quad_border_touch_count(
        quad: np.ndarray,
        image_shape: Tuple[int, int],
        margin_ratio: float = 0.02,
    ) -> int:
        """Count how many image borders a quad touches."""
        h, w = image_shape[:2]
        if h <= 0 or w <= 0:
            return 4
        margin_x = w * margin_ratio
        margin_y = h * margin_ratio
        xs = quad[:, 0]
        ys = quad[:, 1]
        touches = 0
        if np.min(xs) <= margin_x:
            touches += 1
        if np.max(xs) >= (w - margin_x):
            touches += 1
        if np.min(ys) <= margin_y:
            touches += 1
        if np.max(ys) >= (h - margin_y):
            touches += 1
        return touches

    @staticmethod
    def _expand_quad(contour: np.ndarray, image_shape: Tuple[int, int], expand_ratio: float = 0.03) -> np.ndarray:
        """Expand quad slightly so card/page borders are not clipped."""
        img_h, img_w = image_shape[:2]
        pts = contour.astype(np.float32)
        center = np.mean(pts, axis=0)
        expanded = center + (pts - center) * (1.0 + expand_ratio)
        expanded[:, 0] = np.clip(expanded[:, 0], 0, max(img_w - 1, 0))
        expanded[:, 1] = np.clip(expanded[:, 1], 0, max(img_h - 1, 0))
        return expanded

    def _find_brightness_quad(self, image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Fallback detector: find paper/card-like bright region in HSV space.
        Returns (quad, score) in image coordinates.
        """
        if image_bgr is None or image_bgr.size == 0:
            return None, 0.0

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        # White/light paper mask: low saturation + high value.
        mask = cv2.inRange(hsv, (0, 0, 120), (180, 95, 255))
        mask = cv2.medianBlur(mask, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0

        img_h, img_w = image_bgr.shape[:2]
        img_area = float(img_h * img_w)
        best_quad = None
        best_score = 0.0

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < img_area * 0.08:
                continue

            rect = cv2.minAreaRect(contour)
            (rw, rh) = rect[1]
            if rw <= 1 or rh <= 1:
                continue

            quad = cv2.boxPoints(rect).astype(np.float32)
            if not self._is_reasonable_document_quad(
                quad,
                (img_h, img_w),
                min_area_ratio=0.10,
                max_area_ratio=0.95,
            ):
                continue

            area_ratio = area / img_area
            aspect = max(rw / max(rh, 1e-6), rh / max(rw, 1e-6))
            # Prefer larger, less-elongated candidates.
            score = (area_ratio * 1.2) - ((aspect - 1.0) * 0.10)
            if score > best_score:
                best_score = score
                best_quad = quad

        return best_quad, float(max(best_score, 0.0))

    def _tight_crop_bright_region(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Final cleanup crop: trims excess background around a bright paper/card region.
        Returns (image, applied).
        """
        quad, score = self._find_brightness_quad(image_bgr)
        if quad is None or score <= 0:
            return image_bgr, False

        h, w = image_bgr.shape[:2]
        img_area = float(h * w)
        quad_area = float(cv2.contourArea(quad))
        if img_area <= 0:
            return image_bgr, False
        area_ratio = quad_area / img_area

        # Tight crop should only trim margins, never re-crop aggressively.
        if area_ratio < 0.72 or area_ratio > 0.98:
            return image_bgr, False

        rect = self._order_points(quad.astype(np.float32))
        # Small outward padding keeps the edge of the card visible.
        rect = self._expand_quad(rect, (h, w), expand_ratio=0.01)
        warped = self._four_point_transform(image_bgr, rect)
        if warped is None or warped.size == 0:
            return image_bgr, False

        wh, ww = warped.shape[:2]
        if wh < 120 or ww < 120:
            return image_bgr, False

        return warped, True

    @staticmethod
    def _quad_touches_border(quad: np.ndarray, image_shape: Tuple[int, int], margin_ratio: float = 0.02) -> bool:
        """Detect likely incomplete detections that hug image borders."""
        touches = EnhancedDocumentPipeline._quad_border_touch_count(
            quad, image_shape, margin_ratio=margin_ratio
        )
        # One/two-side proximity can still be valid for handheld captures.
        return touches >= 3

    def _tight_crop_edge_region(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Final edge-driven crop for clear card/document borders on textured backgrounds.
        Returns (image, applied).
        """
        if image_bgr is None or image_bgr.size == 0:
            return image_bgr, False

        h, w = image_bgr.shape[:2]
        img_area = float(h * w)
        if img_area <= 0:
            return image_bgr, False

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 160)
        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=2,
        )

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image_bgr, False

        best_quad = None
        best_score = 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < img_area * 0.12:
                continue

            rect = cv2.minAreaRect(contour)
            rw, rh = rect[1]
            if rw <= 1 or rh <= 1:
                continue

            quad = cv2.boxPoints(rect).astype(np.float32)
            if not self._is_reasonable_document_quad(
                quad,
                (h, w),
                min_area_ratio=0.12,
                max_area_ratio=0.98,
            ):
                continue

            # Reject contours hugging multiple image borders (scene boundary artifacts).
            if self._quad_border_touch_count(quad, (h, w), margin_ratio=0.015) >= 2:
                continue

            area_ratio = area / img_area
            if area_ratio > 0.90:
                # Avoid near full-frame crops from global scene boundaries.
                continue
            aspect = max(rw / max(rh, 1e-6), rh / max(rw, 1e-6))
            if aspect > 2.6:
                continue

            # Reuse the same quad scoring used in main detector to favor document-like shapes.
            conf = self._quad_confidence(quad, (h, w))
            score = (conf * 0.75) + (area_ratio * 0.25)
            if score > best_score:
                best_score = score
                best_quad = quad

        if best_quad is None:
            return image_bgr, False

        rect = self._order_points(best_quad.astype(np.float32))
        rect = self._expand_quad(rect, (h, w), expand_ratio=0.006)
        warped = self._four_point_transform(image_bgr, rect)
        if warped is None or warped.size == 0:
            return image_bgr, False

        wh, ww = warped.shape[:2]
        if wh < 120 or ww < 120:
            return image_bgr, False

        return warped, True

    def _find_foreground_quad_from_border_bg(self, image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Fallback: estimate background color from image border and segment foreground object.
        Useful for desk/table captures where document differs from border region.
        """
        if image_bgr is None or image_bgr.size == 0:
            return None, 0.0

        h, w = image_bgr.shape[:2]
        if h < 40 or w < 40:
            return None, 0.0

        border = max(8, int(min(h, w) * 0.06))
        top = image_bgr[:border, :, :]
        bottom = image_bgr[h - border :, :, :]
        left = image_bgr[:, :border, :]
        right = image_bgr[:, w - border :, :]
        border_pixels = np.concatenate(
            [
                top.reshape(-1, 3),
                bottom.reshape(-1, 3),
                left.reshape(-1, 3),
                right.reshape(-1, 3),
            ],
            axis=0,
        )
        bg_color = np.median(border_pixels, axis=0).astype(np.float32)

        diff = np.linalg.norm(image_bgr.astype(np.float32) - bg_color, axis=2)
        diff_u8 = np.clip(diff, 0, 255).astype(np.uint8)
        _, mask = cv2.threshold(diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Make mask robust and fill small gaps.
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0

        img_area = float(h * w)
        best_quad = None
        best_score = 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < img_area * 0.07:
                continue

            rect = cv2.minAreaRect(contour)
            rw, rh = rect[1]
            if rw <= 1 or rh <= 1:
                continue

            quad = cv2.boxPoints(rect).astype(np.float32)
            if self._quad_touches_border(quad, (h, w), margin_ratio=0.015):
                continue
            if not self._is_reasonable_document_quad(
                quad,
                (h, w),
                min_area_ratio=0.08,
                max_area_ratio=0.95,
            ):
                continue

            area_ratio = area / img_area
            aspect = max(rw / max(rh, 1e-6), rh / max(rw, 1e-6))
            score = (area_ratio * 1.15) - ((aspect - 1.0) * 0.10)
            if score > best_score:
                best_score = score
                best_quad = quad

        return best_quad, float(max(best_score, 0.0))

    def _simple_document_crop(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Simple from-scratch document cropper for handheld captures.
        Prioritizes reliability over aggressive perspective correction.
        """
        h, w = image_bgr.shape[:2]
        info: Dict[str, Any] = {
            "document_found": False,
            "method": "none",
            "selected_area_ratio": 0.0,
            "selected_confidence": 0.0,
            "applied": False,
            "warp_area_ratio": 0.0,
            "warp_aspect": 0.0,
        }

        if h <= 0 or w <= 0:
            return image_bgr, info

        # Work in a normalized detection resolution.
        detect_scale = min(1200.0 / float(max(h, w)), 1.0)
        detect = (
            cv2.resize(image_bgr, (int(w * detect_scale), int(h * detect_scale)))
            if detect_scale < 1.0
            else image_bgr.copy()
        )
        dh, dw = detect.shape[:2]
        gray = cv2.cvtColor(detect, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Multi-cue edge/mask combination.
        edges = cv2.Canny(gray, 40, 140)
        edges2 = cv2.Canny(gray, 80, 220)
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 6
        )
        combined = cv2.bitwise_or(cv2.bitwise_or(edges, edges2), cv2.Canny(th, 20, 80))
        combined = cv2.morphologyEx(
            combined,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=2,
        )

        contours, _ = cv2.findContours(combined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        info["contours_found"] = len(contours)

        best_quad: Optional[np.ndarray] = None
        best_score = 0.0
        detect_area = float(dh * dw)

        for contour in contours[:120]:
            contour_area = float(cv2.contourArea(contour))
            if contour_area < detect_area * 0.02:
                continue

            # Try poly quad first.
            perimeter = cv2.arcLength(contour, True)
            for eps_ratio in (0.015, 0.02, 0.03, 0.04):
                approx = cv2.approxPolyDP(contour, eps_ratio * perimeter, True)
                if len(approx) != 4:
                    continue
                if not cv2.isContourConvex(approx):
                    continue
                quad = approx.reshape(4, 2).astype(np.float32)
                if not self._is_reasonable_document_quad(
                    quad, (dh, dw), min_area_ratio=0.12, max_area_ratio=0.98
                ):
                    continue
                # Reject candidates stuck to multiple borders (typical false positives).
                if self._quad_border_touch_count(quad, (dh, dw), margin_ratio=0.018) >= 2:
                    continue
                score = self._quad_confidence(quad, (dh, dw))
                if score > best_score:
                    best_score = score
                    best_quad = quad
                    info["method"] = "quad_poly"

            # Fallback candidate from min-area rectangle.
            rect = cv2.minAreaRect(contour)
            rw, rh = rect[1]
            if rw > 1 and rh > 1:
                quad = cv2.boxPoints(rect).astype(np.float32)
                if self._is_reasonable_document_quad(
                    quad, (dh, dw), min_area_ratio=0.12, max_area_ratio=0.98
                ):
                    if self._quad_border_touch_count(quad, (dh, dw), margin_ratio=0.018) >= 2:
                        continue
                    score = self._quad_confidence(quad, (dh, dw)) * 0.95
                    if score > best_score:
                        best_score = score
                        best_quad = quad
                        info["method"] = "min_area_rect"

        # Color fallback for low-contrast backgrounds.
        if best_quad is None:
            bright_quad, bright_score = self._find_brightness_quad(image_bgr)
            if bright_quad is not None:
                best_quad = bright_quad * detect_scale
                best_score = min(1.0, max(0.0, bright_score))
                info["method"] = "brightness_segmentation"

        # Background-subtraction fallback for desk/table style scenes.
        if best_quad is None:
            bg_quad, bg_score = self._find_foreground_quad_from_border_bg(image_bgr)
            if bg_quad is not None:
                best_quad = bg_quad * detect_scale
                best_score = min(1.0, max(0.0, bg_score))
                info["method"] = "border_bg_subtraction"

        if best_quad is None or best_score < 0.42:
            return image_bgr, info

        # Map detection quad to original image.
        quad_original = best_quad / max(detect_scale, 1e-6)
        if self._quad_touches_border(quad_original, (h, w), margin_ratio=0.015):
            return image_bgr, info
        quad_original = self._expand_quad(quad_original, (h, w), expand_ratio=0.015)

        warped = self._four_point_transform(image_bgr, quad_original)
        if warped is None or warped.size == 0:
            return image_bgr, info

        wh, ww = warped.shape[:2]
        warp_area_ratio = float((wh * ww) / float(h * w))
        warp_aspect = float(ww / max(wh, 1))
        selected_area_ratio = float(cv2.contourArea(best_quad)) / detect_area

        info["document_found"] = True
        info["selected_area_ratio"] = selected_area_ratio
        info["selected_confidence"] = best_score
        info["warp_area_ratio"] = warp_area_ratio
        info["warp_aspect"] = warp_aspect

        # Conservative acceptance guard.
        if warp_area_ratio < 0.12 or not (0.40 <= warp_aspect <= 2.50):
            return image_bgr, info

        info["applied"] = True
        return warped, info

    def _init_paddle_ocr(self) -> Tuple[Any, str]:
        """Initialize a shared PaddleOCR engine, falling back when unavailable."""
        with self._ocr_lock:
            if self._ocr_engine is not None:
                return self._ocr_engine, self._ocr_device

            if "torch" in sys.modules:
                raise RuntimeError("Torch already loaded; skipping PaddleOCR to avoid shm.dll conflicts")

            device = "cpu"
            try:
                if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    device = "gpu"
            except Exception:
                device = "cpu"

            from paddleocr import PaddleOCR

            try:
                engine = PaddleOCR(
                    lang="en",
                    det_db_thresh=0.3,
                    det_db_box_thresh=0.5,
                    rec_batch_num=6,
                    device=device,
                )
            except TypeError:
                engine = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    device=device,
                )

            self.__class__._ocr_engine = engine
            self.__class__._ocr_device = device
            return engine, device

    def _run_tesseract_ocr(self, image: np.ndarray) -> Tuple[str, List[Dict[str, Any]], str]:
        """Run the notebook's Tesseract OCR fallback."""
        pil_image = Image.fromarray(image)
        data = pytesseract.image_to_data(
            pil_image,
            lang="eng",
            config="--oem 3 --psm 3",
            output_type=pytesseract.Output.DICT,
        )

        boxes: List[Dict[str, Any]] = []
        texts: List[str] = []
        total = len(data.get("text", []))
        for idx in range(total):
            text = (data["text"][idx] or "").strip()
            try:
                conf = float(data["conf"][idx])
            except Exception:
                conf = -1.0
            if not text or conf <= 0:
                continue

            x = int(data["left"][idx])
            y = int(data["top"][idx])
            w = int(data["width"][idx])
            h = int(data["height"][idx])
            texts.append(text)
            boxes.append(
                {
                    "text": text,
                    "bbox": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                    "confidence": conf / 100.0,
                }
            )

        return " ".join(texts).strip(), boxes, "tesseract"

    def _run_paddle_ocr(self, image: np.ndarray) -> Tuple[str, List[Dict[str, Any]], str]:
        """Run PaddleOCR using the notebook-style configuration."""
        engine, device = self._init_paddle_ocr()
        result = engine.ocr(image)

        boxes: List[Dict[str, Any]] = []
        texts: List[str] = []
        for page in result or []:
            if not page:
                continue
            for line in page:
                points = line[0] if isinstance(line[0], list) else []
                text = line[1][0] if len(line) > 1 else ""
                score = float(line[1][1]) if len(line) > 1 else 0.0
                cleaned = (text or "").strip()
                if not cleaned:
                    continue
                texts.append(cleaned)
                boxes.append(
                    {
                        "text": cleaned,
                        "bbox": points,
                        "confidence": score,
                    }
                )

        return "\n".join(texts).strip(), boxes, f"paddleocr_{device}"

    def process_complete_pipeline(
        self,
        input_path: str,
        output_path: str,
        enhancement_params: Dict[str, Any] = None,
        options: Dict[str, Any] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Execute complete 12-step processing pipeline

        Args:
            input_path: Path to input image
            output_path: Path to save processed image
            enhancement_params: Custom enhancement parameters
            options: Processing options

        Returns:
            Tuple of (success, text_or_error, pipeline_stats)
        """
        try:
            if enhancement_params:
                self.enhancer.update_parameters(**enhancement_params)

            options = options or {}
            total_steps = 12
            pipeline_stats = {
                "steps": {},
                "start_time": datetime.now().isoformat(),
                "total_steps": total_steps,
                "parameters": self.enhancer.get_parameters(),
            }

            # ========== STEP 1: Load Original Image ==========
            self.emit_progress(
                1, total_steps, "Load Original Image", "Loading captured image..."
            )
            img = cv2.imread(input_path)
            if img is None:
                raise ValueError("Could not read input image")

            original_shape = img.shape
            pipeline_stats["steps"]["step_1"] = {
                "stage": "Load Original Image",
                "input_shape": original_shape,
                "timestamp": datetime.now().isoformat(),
            }

            # ========== STEP 2: Downscale for Detection ==========
            self.emit_progress(
                2, total_steps, "Downscale for Detection", "Creating smaller image for contour detection..."
            )
            detection_scale = 0.25
            small = cv2.resize(img, (0, 0), fx=detection_scale, fy=detection_scale)
            pipeline_stats["steps"]["step_2"] = {
                "stage": "Downscale for Detection",
                "scale": detection_scale,
                "output_shape": small.shape,
            }

            # ========== STEP 3: Convert to Grayscale (Detection) ==========
            self.emit_progress(
                3, total_steps, "Convert to Grayscale", "Preparing detection grayscale image..."
            )
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            pipeline_stats["steps"]["step_3"] = {
                "stage": "Convert to Grayscale (Detection)",
                "output_shape": gray.shape,
                "channels": 1,
            }

            # ========== STEP 4: Gaussian Blur ==========
            self.emit_progress(4, total_steps, "Gaussian Blur", "Smoothing detection image...")
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            pipeline_stats["steps"]["step_4"] = {
                "stage": "Gaussian Blur",
                "method": "Gaussian 5x5",
                "output_shape": blurred.shape,
            }

            # Working copy for final output path (kept in color).
            working_color = img.copy()

            # ========== STEP 5: Canny Edge Detection ==========
            self.emit_progress(5, total_steps, "Edge Detection", "Finding document edges with Canny...")
            try:
                # Use multi-scale Canny matching notebook: (30, 100), (50, 150), (75, 200)
                edges = self.enhancer.apply_canny_edge_detection(
                    blurred, use_multi_scale=True, debug=True
                )
                # Dilate edges to close small gaps (matching notebook)
                edges = cv2.dilate(edges, np.ones((2, 2)), iterations=2)
                
                pipeline_stats["steps"]["step_5"] = {
                    "stage": "Edge Detection",
                    "method": "Canny Multi-Scale (30-100, 50-150, 75-200)",
                    "output_shape": edges.shape,
                    "edge_pixels": int(np.sum(edges > 0)),
                }
            except Exception as edge_err:
                logger.warning(f"Edge detection failed: {edge_err}, using fallback")
                # Fallback to simple Canny
                edges = cv2.Canny(blurred, 50, 150)
                pipeline_stats["steps"]["step_5"] = {
                    "stage": "Edge Detection",
                    "method": "Canny Fallback (50, 150)",
                    "error": str(edge_err),
                    "output_shape": edges.shape,
                }

            # ========== STEP 6: Find Document Contour ==========
            self.emit_progress(
                6, total_steps, "Find Document Contour", "Locating document boundary..."
            )
            cropped_color, crop_info = self._simple_document_crop(working_color)
            pipeline_stats["steps"]["step_6"] = {
                "stage": "Find Document Contour",
                "contours_found": crop_info.get("contours_found", 0),
                "document_found": bool(crop_info.get("document_found")),
                "selected_area_ratio": float(crop_info.get("selected_area_ratio", 0.0)),
                "selected_confidence": float(crop_info.get("selected_confidence", 0.0)),
                "method": crop_info.get("method", "none"),
            }

            # ========== STEP 7: Perspective Transform (Crop) ==========
            self.emit_progress(
                7, total_steps, "Perspective Transform", "Applying document crop..."
            )
            working_color = cropped_color
            pipeline_stats["steps"]["step_7"] = {
                "stage": "Perspective Transform (Crop)",
                "applied": bool(crop_info.get("applied", False)),
                "new_shape": working_color.shape,
                "warp_area_ratio": float(crop_info.get("warp_area_ratio", 0.0)),
                "warp_aspect": float(crop_info.get("warp_aspect", 0.0)),
            }

            # Final trim to remove residual margins if still present.
            tightened, tight_applied = self._tight_crop_bright_region(working_color)
            edge_tightened, edge_tight_applied = self._tight_crop_edge_region(
                tightened if tight_applied else working_color
            )

            if edge_tight_applied:
                working_color = edge_tightened
            elif tight_applied:
                working_color = tightened

            pipeline_stats["steps"]["step_7"]["tight_crop_applied"] = bool(
                tight_applied or edge_tight_applied
            )
            pipeline_stats["steps"]["step_7"]["tight_crop_method"] = (
                "edge_region" if edge_tight_applied else ("bright_region" if tight_applied else "none")
            )
            pipeline_stats["steps"]["step_7"]["tight_crop_shape"] = (
                list(working_color.shape) if (tight_applied or edge_tight_applied) else None
            )

            # ========== STEP 8: Grayscale & Downscale for Processing ==========
            self.emit_progress(8, total_steps, "Processing Prep", "Preparing image for enhancement...")
            processing_gray = cv2.cvtColor(working_color, cv2.COLOR_BGR2GRAY)
            h, w = processing_gray.shape
            max_dim = 800
            processing_scale = min(max_dim / max(h, w), 1.0)
            if processing_scale < 1.0:
                small_gray = cv2.resize(
                    processing_gray,
                    (int(w * processing_scale), int(h * processing_scale)),
                )
            else:
                small_gray = processing_gray
            pipeline_stats["steps"]["step_8"] = {
                "stage": "Grayscale & Downscale for Processing",
                "gray_shape": processing_gray.shape,
                "processing_scale": processing_scale,
                "downscaled_shape": small_gray.shape,
            }

            # ========== STEP 9: Morphological Background Estimation ==========
            self.emit_progress(9, total_steps, "Background Estimation", "Estimating page background...")
            sh, sw = small_gray.shape
            k = max(sh, sw) // 8
            k = k if k % 2 == 1 else k + 1
            k = max(31, min(k, 127))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            bg_raw = cv2.morphologyEx(small_gray, cv2.MORPH_CLOSE, kernel)
            pipeline_stats["steps"]["step_9"] = {
                "stage": "Morphological Background Estimation",
                "kernel_size": k,
                "output_shape": bg_raw.shape,
            }

            # ========== STEP 10: Smooth and Upscale Background ==========
            self.emit_progress(10, total_steps, "Background Smoothing", "Smoothing estimated background...")
            bg_smoothed = cv2.GaussianBlur(bg_raw, (k, k), 0)
            if processing_scale < 1.0:
                bg = cv2.resize(bg_smoothed, (w, h), interpolation=cv2.INTER_LINEAR)
                bg = cv2.GaussianBlur(bg, (31, 31), 0)
            else:
                bg = bg_smoothed
            pipeline_stats["steps"]["step_10"] = {
                "stage": "Smooth and Upscale Background",
                "output_shape": bg.shape,
            }

            # ========== STEP 11: Shadow Removal (Enhancement) ==========
            self.emit_progress(11, total_steps, "Shadow Removal", "Removing shadows and enhancing document...")
            bg_safe = bg.astype(np.float32)
            mask = bg_safe > 10
            result_color = np.zeros_like(working_color, dtype=np.float32)
            for channel_index in range(3):
                channel = working_color[:, :, channel_index].astype(np.float32)
                result_color[:, :, channel_index][mask] = (channel[mask] / bg_safe[mask]) * 255.0
            enhanced_color = np.clip(result_color, 0, 255).astype(np.uint8)
            enhanced_gray = cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2GRAY)
            gray_std = float(enhanced_gray.std())
            gray_range = int(enhanced_gray.max()) - int(enhanced_gray.min())
            output_image = enhanced_color

            # Guard against collapsed outputs that become almost solid-color blocks.
            if gray_std < 8.0 or gray_range < 24:
                logger.warning(
                    "Enhanced output collapsed to near-uniform image; falling back to cropped/original image"
                )
                output_image = working_color
                enhanced_gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
            pipeline_stats["steps"]["step_11"] = {
                "stage": "Shadow Removal (Enhancement)",
                "output_shape": enhanced_gray.shape,
                "method": "Notebook-style color-preserving shadow removal",
                "stddev": gray_std,
                "range": gray_range,
                "fallback_used": output_image is not enhanced_color,
            }

            # Save the enhanced document for dashboard preview and downloads.
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, output_image, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # ========== STEP 12: OCR Text Extraction ==========
            self.emit_progress(12, total_steps, "OCR Text Extraction", "Extracting text from processed document...")
            text = ""
            raw_results: List[Dict[str, Any]] = []
            ocr_engine = "tesseract"
            gemini_result: Optional[Dict[str, Any]] = None
            try:
                from ..ocr.gemini_ocr import is_available as gemini_available, ocr_image_bytes

                if gemini_available():
                    ok, encoded = cv2.imencode(".jpg", output_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if not ok:
                        raise RuntimeError("Could not encode the processed image for OCR")
                    gemini_result = ocr_image_bytes(encoded.tobytes())
                    text = gemini_result["full_text"]
                    raw_results = gemini_result["raw_results"]
                    ocr_engine = "gemini"
                else:
                    try:
                        text, raw_results, ocr_engine = self._run_paddle_ocr(enhanced_gray)
                    except Exception as paddle_error:
                        logger.warning(f"PaddleOCR unavailable, falling back to Tesseract: {paddle_error}")
                        text, raw_results, ocr_engine = self._run_tesseract_ocr(enhanced_gray)
            except Exception as ocr_error:
                logger.warning(f"OCR failed: {ocr_error}")

            confidence_avg = (
                sum(float(item.get("confidence", 0.0)) for item in raw_results) / len(raw_results)
                if raw_results
                else 0.0
            )
            text_lines = [line.strip() for line in text.splitlines() if line.strip()]
            structured_units = [{"text": line, "type": "paragraph"} for line in text_lines]
            pipeline_stats["steps"]["step_12"] = {
                "stage": "OCR Text Extraction",
                "engine": ocr_engine,
                "text_length": len(text),
                "words_extracted": len(text.split()),
                "regions_detected": len(raw_results),
                "confidence": float(confidence_avg * 100),
            }
            pipeline_stats["ocr_result"] = gemini_result or {
                "raw_results": raw_results,
                "structured_units": structured_units,
                "full_text": text,
                "derived_title": text_lines[0][:120] if text_lines else "",
                "confidence_avg": float(confidence_avg),
                "word_count": len([word for word in text.replace("\n", " ").split(" ") if word.strip()]),
                "timestamp": datetime.now().isoformat(),
                "processing_time_ms": 0,
                "image_dimensions": [int(output_image.shape[1]), int(output_image.shape[0])],
                "engine": ocr_engine,
            }
            pipeline_stats["output"] = {
                "output_path": output_path,
                "file_size": os.path.getsize(output_path),
                "format": "JPEG",
                "compression_quality": 90,
            }

            pipeline_stats["end_time"] = datetime.now().isoformat()
            pipeline_stats["success"] = True

            logger.info(
                f"✅ Pipeline complete. Text extracted: {len(text)} characters using {ocr_engine}"
            )

            return True, text, pipeline_stats

        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            pipeline_stats["error"] = str(e)
            pipeline_stats["success"] = False
            return False, str(e), pipeline_stats

    def process_with_custom_params(
        self,
        input_path: str,
        output_path: str,
        enhancement_params: Dict[str, Any] = None,
        edge_thresholds: Tuple[int, int] = None,
        morpho_params: Dict[str, int] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Process image with fully custom parameters

        Useful for A/B testing and parameter optimization

        Args:
            input_path: Input image path
            output_path: Output image path
            enhancement_params: Dict with brightness_boost, equalization_strength, etc.
            edge_thresholds: Tuple of (threshold1, threshold2) for Canny
            morpho_params: Dict with kernel_size and iterations for morphological ops

        Returns:
            Tuple of (success, text_or_error, pipeline_stats)
        """
        # Apply custom enhancement params
        if enhancement_params:
            self.enhancer.update_parameters(**enhancement_params)

        # Could extend with custom thresholds, etc.
        return self.process_complete_pipeline(input_path, output_path)

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get current pipeline configuration and stats"""
        return {
            "total_steps": 12,
            "current_parameters": self.enhancer.get_parameters(),
            "available_enhancements": {
                "brightness_boost_range": (0, 50),
                "equalization_strength_range": (0.0, 1.0),
                "clahe_clip_limit_range": (1.0, 4.0),
                "clahe_tile_size_options": [4, 8, 16],
            },
            "supported_formats": ["jpg", "jpeg", "png", "bmp"],
            "max_image_size": "50MB",
            "ocr_languages": ["eng"],
        }
