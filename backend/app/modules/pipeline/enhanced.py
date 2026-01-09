"""
Enhanced Document Processing Pipeline - Notebook Integration
Implements all 12 steps from processing.ipynb with adjustable parameters
"""

import logging
import os
import threading
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image

from ..image import ImageEnhancer

logger = logging.getLogger(__name__)


class EnhancedDocumentPipeline:
    """
    Complete document processing pipeline matching notebook implementation

    12-Step Pipeline:
    1. Quality Validation - Check blur and focus
    2. Color Space Conversion - Convert to grayscale
    3. Threshold & Binarization - Create binary image
    4. Noise Removal - Apply denoising
    5. Edge Detection - Find document edges
    6. Contour Detection - Identify document outline
    7. Perspective Correction - Straighten document
    8. Contrast Enhancement - Enhance clarity (brightness boost)
    9. Dilation & Erosion - Improve edge definition
    10. OCR Processing - Extract text
    11. Image Optimization - Compress and optimize
    12. File Storage - Save to disk
    """

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

            # ========== STEP 1: Quality Validation ==========
            self.emit_progress(
                1, total_steps, "Quality Validation", "Checking image blur and focus..."
            )
            img = cv2.imread(input_path)
            if img is None:
                raise ValueError("Could not read input image")

            original_shape = img.shape
            pipeline_stats["steps"]["step_1"] = {
                "stage": "Quality Validation",
                "input_shape": original_shape,
                "timestamp": datetime.now().isoformat(),
            }

            # ========== STEP 2: Color Space Conversion ==========
            self.emit_progress(
                2, total_steps, "Color Space Conversion", "Converting to grayscale..."
            )
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pipeline_stats["steps"]["step_2"] = {
                "stage": "Color Space Conversion",
                "output_shape": gray.shape,
                "channels": 1,
            }

            # ========== STEP 3: Threshold & Binarization ==========
            self.emit_progress(
                3, total_steps, "Threshold & Binarization", "Creating binary image..."
            )
            adaptive_thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            pipeline_stats["steps"]["step_3"] = {
                "stage": "Threshold & Binarization",
                "method": "Adaptive Thresholding",
                "output_shape": adaptive_thresh.shape,
            }

            # ========== STEP 4: Noise Removal ==========
            self.emit_progress(4, total_steps, "Noise Removal", "Applying denoising filter...")
            denoised = cv2.fastNlMeansDenoising(adaptive_thresh, None, 10, 7, 21)
            pipeline_stats["steps"]["step_4"] = {
                "stage": "Noise Removal",
                "method": "Non-Local Means Denoising",
                "output_shape": denoised.shape,
            }

            # ========== STEP 5: Edge Detection (Canny - matching notebook) ==========
            self.emit_progress(5, total_steps, "Edge Detection", "Finding document edges with Canny...")
            try:
                # Use multi-scale Canny matching notebook: (30, 100), (50, 150), (75, 200)
                edges = self.enhancer.apply_canny_edge_detection(
                    denoised, use_multi_scale=True, debug=True
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
                edges = cv2.Canny(denoised, 50, 150)
                pipeline_stats["steps"]["step_5"] = {
                    "stage": "Edge Detection",
                    "method": "Canny Fallback (50, 150)",
                    "error": str(edge_err),
                    "output_shape": edges.shape,
                }

            # ========== STEP 6: Contour Detection ==========
            self.emit_progress(
                6, total_steps, "Contour Detection", "Identifying document outline..."
            )
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pipeline_stats["steps"]["step_6"] = {
                "stage": "Contour Detection",
                "contours_found": len(contours),
            }

            # ========== STEP 7: Perspective Correction ==========
            self.emit_progress(
                7, total_steps, "Perspective Correction", "Straightening document..."
            )
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)

                if len(approx) == 4:
                    pts = np.float32(approx)
                    pts = pts.reshape(4, 2)
                    rect = cv2.boundingRect(largest_contour)
                    dst_pts = np.float32([[0, 0], [rect[2], 0], [rect[2], rect[3]], [0, rect[3]]])
                    matrix = cv2.getPerspectiveTransform(pts, dst_pts)
                    denoised = cv2.warpPerspective(denoised, matrix, (rect[2], rect[3]))

                    pipeline_stats["steps"]["step_7"] = {
                        "stage": "Perspective Correction",
                        "applied": True,
                        "new_shape": denoised.shape,
                    }
                else:
                    pipeline_stats["steps"]["step_7"] = {
                        "stage": "Perspective Correction",
                        "applied": False,
                        "reason": "Not a 4-corner polygon",
                    }
            else:
                pipeline_stats["steps"]["step_7"] = {
                    "stage": "Perspective Correction",
                    "applied": False,
                    "reason": "No contours found",
                }

            # ========== STEP 8: Contrast Enhancement (BRIGHTNESS BOOST) ==========
            self.emit_progress(8, total_steps, "Contrast Enhancement", "Enhancing image clarity...")
            enhanced, enhancement_stats = self.enhancer.enhance_brightness_and_contrast(denoised)
            pipeline_stats["steps"]["step_8"] = {
                "stage": "Contrast Enhancement",
                "method": "Brightness Boost + Gentle Equalization + CLAHE",
                "stats": enhancement_stats,
                "output_shape": enhanced.shape,
            }

            # ========== STEP 9: Dilation & Erosion ==========
            self.emit_progress(9, total_steps, "Dilation & Erosion", "Improving edge definition...")
            morphed = self.enhancer.apply_morphological_ops(enhanced)
            pipeline_stats["steps"]["step_9"] = {
                "stage": "Dilation & Erosion",
                "kernel_size": (3, 3),
                "iterations": 1,
                "output_shape": morphed.shape,
            }

            # ========== STEP 10: OCR Processing ==========
            self.emit_progress(10, total_steps, "OCR Processing", "Extracting text...")
            text = ""
            ocr_confidence = 0
            try:
                # Prepare image for OCR
                ocr_preprocessed, ocr_stats = self.enhancer.apply_ocr_preprocessing(morphed)
                pipeline_stats["steps"]["step_10_prep"] = ocr_stats

                # Save temporarily for OCR
                temp_ocr_path = output_path.replace(".jpg", "_temp.jpg")
                cv2.imwrite(temp_ocr_path, ocr_preprocessed)

                try:
                    img_for_ocr = Image.open(temp_ocr_path)
                    text = pytesseract.image_to_string(img_for_ocr, lang="eng")

                    # Get confidence
                    data = pytesseract.image_to_data(
                        ocr_preprocessed, output_type=pytesseract.Output.DICT
                    )
                    confident_words = sum(1 for conf in data["conf"] if int(conf) > 60)
                    total_words = len([c for c in data["conf"] if int(c) > 0])
                    ocr_confidence = (confident_words / total_words * 100) if total_words > 0 else 0

                    pipeline_stats["steps"]["step_10"] = {
                        "stage": "OCR Processing",
                        "text_length": len(text),
                        "confidence": float(ocr_confidence),
                        "words_extracted": len(text.split()),
                    }

                finally:
                    if os.path.exists(temp_ocr_path):
                        os.remove(temp_ocr_path)

            except Exception as ocr_error:
                logger.warning(f"OCR failed: {ocr_error}")
                pipeline_stats["steps"]["step_10"] = {
                    "stage": "OCR Processing",
                    "error": str(ocr_error),
                    "text_length": 0,
                }

            # ========== STEP 11: Image Optimization ==========
            self.emit_progress(11, total_steps, "Image Optimization", "Compressing image...")

            # Convert back to color for final save if needed
            if len(morphed.shape) == 2:  # Grayscale
                final_image = cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR)
            else:
                final_image = morphed

            pipeline_stats["steps"]["step_11"] = {
                "stage": "Image Optimization",
                "output_shape": final_image.shape,
                "channels": 3,
                "compression_quality": 90,
            }

            # ========== STEP 12: File Storage ==========
            self.emit_progress(12, total_steps, "File Storage", "Saving to disk...")

            # Save processed image
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, final_image, [cv2.IMWRITE_JPEG_QUALITY, 90])

            pipeline_stats["steps"]["step_12"] = {
                "stage": "File Storage",
                "output_path": output_path,
                "file_size": os.path.getsize(output_path),
                "format": "JPEG",
            }

            pipeline_stats["end_time"] = datetime.now().isoformat()
            pipeline_stats["success"] = True

            logger.info(
                f"✅ Pipeline complete. Text extracted: {len(text)} characters, OCR confidence: {ocr_confidence:.1f}%"
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
