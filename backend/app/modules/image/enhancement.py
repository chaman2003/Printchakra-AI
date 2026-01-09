"""
Image Enhancement Module - Improved multi-stage contrast enhancement
Based on printchakra_clean.ipynb Section 4 implementation
"""

import logging
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageEnhancer:
    """Advanced image enhancement with multi-stage processing"""

    # Default parameters from improved notebook
    DEFAULT_PARAMS = {
        "brightness_boost": 25,  # 0-50 units
        "equalization_strength": 0.4,  # 0.0-1.0 blend ratio
        "clahe_clip_limit": 2.0,  # 1.0-4.0 contrast limit
        "clahe_tile_size": 8,  # 4-16 (power of 2)
    }

    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize enhancer with optional custom parameters

        Args:
            params: Dict with custom parameter values
        """
        self.params = self.DEFAULT_PARAMS.copy()
        if params:
            self.params.update(params)

        logger.info(f"ImageEnhancer initialized with params: {self.params}")

    def enhance_contrast(
        self, image: np.ndarray, brightness: int = 25, eq_strength: float = 0.4
    ) -> np.ndarray:
        """
        Multi-stage contrast enhancement from notebook Section 4

        Args:
            image: Input image (BGR or grayscale)
            brightness: Brightness boost (0-50)
            eq_strength: Histogram equalization strength (0.0-1.0)

        Returns:
            Enhanced grayscale image
        """
        try:
            # Convert to grayscale if needed
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

            # Step 1: Brightness boost
            brightened = cv2.convertScaleAbs(gray, alpha=1.0, beta=brightness)

            # Step 2: Gentle blended histogram equalization
            equalized_full = cv2.equalizeHist(brightened)
            equalized = cv2.addWeighted(
                brightened, 1.0 - eq_strength, equalized_full, eq_strength, 0
            )

            # Step 3: CLAHE enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_enhanced = clahe.apply(equalized)

            # Step 4: Final blend (50/50)
            enhanced = cv2.addWeighted(equalized, 0.5, clahe_enhanced, 0.5, 0)

            return enhanced

        except Exception as e:
            logger.error(f"Contrast enhancement error: {str(e)}")
            return image

    def preprocess_for_ocr(self, image: np.ndarray) -> List[Tuple[np.ndarray, str]]:
        """
        Generate OCR-ready preprocessing variants for handwritten & printed text
        From notebook Section 4

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            List of (preprocessed_image, variant_name) tuples
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            variants = []

            # Variant 1: Bilateral filter for noise reduction while preserving edges
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            variants.append((bilateral, "bilateral"))

            # Variant 2: Adaptive thresholding with different block sizes
            adaptive1 = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            variants.append((adaptive1, "adaptive_11"))

            adaptive2 = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3
            )
            variants.append((adaptive2, "adaptive_15"))

            # Variant 3: CLAHE + Sharpening for handwriting
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(bilateral)
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            sharpened_thresh = cv2.adaptiveThreshold(
                sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2
            )
            variants.append((sharpened_thresh, "clahe_sharpened"))

            # Variant 4: High contrast for handwriting
            clahe_high = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            contrast = clahe_high.apply(bilateral)
            contrast_thresh = cv2.adaptiveThreshold(
                contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3
            )
            variants.append((contrast_thresh, "high_contrast"))

            return variants

        except Exception as e:
            logger.error(f"OCR preprocessing error: {str(e)}")
            return [(image, "original")]

    def enhance_brightness_and_contrast(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply balanced brightness and contrast enhancement with statistics
        Legacy wrapper for compatibility

        Args:
            image: Grayscale image (height, width) or BGR image

        Returns:
            Tuple of (enhanced_image, enhancement_stats)
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Record original stats
            original_mean = gray.mean()
            original_std = gray.std()
            original_min = gray.min()
            original_max = gray.max()

            # Use improved enhance_contrast
            enhanced = self.enhance_contrast(
                gray,
                brightness=self.params["brightness_boost"],
                eq_strength=self.params["equalization_strength"],
            )

            # Calculate enhancement statistics
            enhanced_mean = enhanced.mean()
            enhanced_std = enhanced.std()
            enhanced_min = enhanced.min()
            enhanced_max = enhanced.max()

            contrast_improvement = enhanced_std / original_std if original_std > 0 else 1.0

            stats = {
                "original": {
                    "mean": float(original_mean),
                    "std": float(original_std),
                    "min": float(original_min),
                    "max": float(original_max),
                    "range": float(original_max - original_min),
                },
                "enhanced": {
                    "mean": float(enhanced_mean),
                    "std": float(enhanced_std),
                    "min": float(enhanced_min),
                    "max": float(enhanced_max),
                    "range": float(enhanced_max - enhanced_min),
                },
                "improvement": {
                    "brightness_boost": self.params["brightness_boost"],
                    "contrast_factor": float(contrast_improvement),
                    "mean_change": float(enhanced_mean - original_mean),
                    "std_change": float(enhanced_std - original_std),
                },
                "parameters": self.params.copy(),
            }

            logger.info(
                f"Brightness & contrast enhancement complete. Contrast improvement: {contrast_improvement:.2f}x"
            )

            return enhanced, stats

        except Exception as e:
            logger.error(f"Brightness enhancement error: {str(e)}")
            return image, {"error": str(e)}

    def apply_gaussian_blur(
        self, image: np.ndarray, kernel_size: Tuple[int, int] = None, sigma: float = None
    ) -> np.ndarray:
        """
        Apply Gaussian blur (Step 5 from notebook)

        Args:
            image: Input image
            kernel_size: Blur kernel size (must be odd)
            sigma: Gaussian standard deviation

        Returns:
            Blurred image
        """
        if kernel_size is None:
            kernel_size = (5, 5)  # Default from notebook
        if sigma is None:
            sigma = 0  # Auto-calculate

        return cv2.GaussianBlur(image, kernel_size, sigma)

    def apply_canny_edge_detection(
        self, image: np.ndarray, threshold1: int = None, threshold2: int = None,
        use_multi_scale: bool = False, debug: bool = False
    ) -> np.ndarray:
        """
        Apply Canny edge detection (matching notebook implementation)

        The notebook uses thresholds: (30, 100), (50, 150), (75, 200)
        Default is (50, 150) which works well for most documents.

        Args:
            image: Input grayscale image
            threshold1: Lower threshold (default: 50)
            threshold2: Upper threshold (default: 150)
            use_multi_scale: If True, try multiple thresholds and combine
            debug: Print debug information

        Returns:
            Edge map
        """
        try:
            # Validate input
            if image is None or image.size == 0:
                if debug:
                    logger.warning("Canny: Invalid input image")
                return np.zeros_like(image) if image is not None else np.zeros((100, 100), dtype=np.uint8)

            # Ensure grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            if threshold1 is None:
                threshold1 = 50  # Default from notebook
            if threshold2 is None:
                threshold2 = 150  # Default from notebook

            if use_multi_scale:
                # Multi-scale approach matching notebook
                combined_edges = np.zeros_like(gray, dtype=np.uint8)
                
                for low, high in [(30, 100), (50, 150), (75, 200)]:
                    try:
                        edges = cv2.Canny(gray, low, high)
                        combined_edges = cv2.bitwise_or(combined_edges, edges)
                        if debug:
                            logger.info(f"Canny({low}, {high}) applied successfully")
                    except Exception as e:
                        if debug:
                            logger.warning(f"Canny({low}, {high}) failed: {e}")
                        continue
                
                return combined_edges
            else:
                # Single threshold approach
                return cv2.Canny(gray, threshold1, threshold2)

        except Exception as e:
            logger.error(f"Canny edge detection error: {str(e)}")
            # Return empty edge map on error
            return np.zeros_like(image) if image is not None else np.zeros((100, 100), dtype=np.uint8)

    def apply_binary_thresholding(
        self, image: np.ndarray, method: str = "otsu"
    ) -> Tuple[np.ndarray, float]:
        """
        Apply binary thresholding (Step 7 from notebook)

        Args:
            image: Input grayscale image
            method: 'otsu' for automatic or 'manual' with specific value

        Returns:
            Tuple of (binary_image, threshold_value)
        """
        if method == "otsu":
            threshold_value, binary = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return binary, threshold_value
        else:
            # Manual threshold
            threshold_value, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            return binary, 127.0

    def apply_morphological_ops(
        self, image: np.ndarray, kernel_size: Tuple[int, int] = None, iterations: int = None
    ) -> np.ndarray:
        """
        Apply morphological operations (Step 8 from notebook)
        Erosion followed by dilation

        Args:
            image: Input binary image
            kernel_size: Morphology kernel size
            iterations: Number of iterations

        Returns:
            Processed image
        """
        if kernel_size is None:
            kernel_size = (3, 3)  # Default from notebook
        if iterations is None:
            iterations = 1  # Default from notebook

        kernel = np.ones(kernel_size, np.uint8)

        # Erosion (removes small white noise)
        eroded = cv2.erode(image, kernel, iterations=iterations)

        # Dilation (fills small black gaps)
        dilated = cv2.dilate(eroded, kernel, iterations=iterations)

        return dilated

    def apply_adaptive_threshold(
        self, image: np.ndarray, block_size: int = None, c_value: float = None
    ) -> np.ndarray:
        """
        Apply adaptive thresholding (Step 12 OCR preprocessing)

        Args:
            image: Input grayscale image
            block_size: Neighborhood size (must be odd)
            c_value: Constant subtracted from mean

        Returns:
            Binary image
        """
        if block_size is None:
            block_size = 11  # Default from notebook
        if c_value is None:
            c_value = 2  # Default from notebook

        # Ensure block size is odd
        if block_size % 2 == 0:
            block_size += 1

        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, int(c_value)
        )

    def apply_ocr_preprocessing(
        self,
        image: np.ndarray,
        adaptive_block_size: int = None,
        adaptive_const: int = None,
        clahe_limit: float = None,
        clahe_grid: int = None,
        sharpen_strength: float = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Complete OCR preprocessing pipeline (Step 12 from notebook)

        Args:
            image: Input grayscale image
            adaptive_block_size: Adaptive threshold block size
            adaptive_const: Adaptive threshold constant
            clahe_limit: CLAHE clip limit
            clahe_grid: CLAHE tile grid size
            sharpen_strength: Sharpening kernel multiplier

        Returns:
            Tuple of (preprocessed_image, preprocessing_stats)
        """
        try:
            if adaptive_block_size is None:
                adaptive_block_size = 11
            if adaptive_const is None:
                adaptive_const = 2
            if clahe_limit is None:
                clahe_limit = 1.5
            if clahe_grid is None:
                clahe_grid = 8
            if sharpen_strength is None:
                sharpen_strength = 1.0

            # Step 1: Adaptive thresholding
            adaptive_thresh = self.apply_adaptive_threshold(
                image, adaptive_block_size, adaptive_const
            )

            # Step 2: Denoise with morphological operations
            kernel_denoise = np.ones((1, 1), np.uint8)
            denoised = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel_denoise)

            # Step 3: CLAHE enhancement
            clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(clahe_grid, clahe_grid))
            enhanced_for_ocr = clahe.apply(image)

            # Step 4: Image sharpening
            kernel_sharpen = (
                np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
                * sharpen_strength
            )

            sharpened = cv2.filter2D(enhanced_for_ocr, -1, kernel_sharpen)

            stats = {
                "preprocessing_steps": 4,
                "parameters": {
                    "adaptive_block_size": adaptive_block_size,
                    "adaptive_const": adaptive_const,
                    "clahe_limit": clahe_limit,
                    "clahe_grid": clahe_grid,
                    "sharpen_strength": sharpen_strength,
                },
            }

            logger.info("OCR preprocessing complete")
            return sharpened, stats

        except Exception as e:
            logger.error(f"OCR preprocessing error: {str(e)}")
            return image, {"error": str(e)}

    def update_parameters(self, **kwargs) -> None:
        """
        Update enhancement parameters at runtime

        Args:
            **kwargs: Parameter name-value pairs
        """
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
                logger.info(f"Updated {key} to {value}")
            else:
                logger.warning(f"Unknown parameter: {key}")

    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters"""
        return self.params.copy()

    def reset_parameters(self) -> None:
        """Reset to default parameters"""
        self.params = self.DEFAULT_PARAMS.copy()
        logger.info("Parameters reset to defaults")


print("[OK] Enhancement module loaded (improved)")
