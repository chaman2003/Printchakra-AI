"""
Section 2: Image Processing Module
Advanced image enhancement, edge detection, and perspective correction
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


class ImageProcessingModule:
    """
    Handles advanced document image processing
    """

    def __init__(self):
        """Initialize image processing module"""
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def find_document_contours(self, image: np.ndarray, debug: bool = False) -> Tuple[Optional[np.ndarray], List]:
        """
        Find document boundaries using multi-scale Canny edge detection (matching notebook).

        Uses the notebook's approach:
        - Scale down to 25% for faster processing
        - Gaussian blur (5,5)
        - Multiple Canny thresholds: [(30, 100), (50, 150), (75, 200)]
        - Dilation with (2,2) kernel, 2 iterations
        - Area threshold: 10% of image area

        Args:
            image: Input image
            debug: Print debug information

        Returns:
            Tuple of (document_contour, all_contours)
        """
        try:
            # Validate input
            if image is None or image.size == 0:
                if debug:
                    print("      [Contour Detection] Invalid input image")
                return None, []

            # Scale down for faster processing (matching notebook scale=0.25)
            scale = 0.25
            small = cv2.resize(image, (0, 0), fx=scale, fy=scale)

            # Convert to grayscale safely
            if len(small.shape) == 3:
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            else:
                gray = small.copy()

            # Apply Gaussian blur (5,5) - matching notebook exactly
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            all_contours = []
            document_contour = None

            # Try multiple Canny thresholds - matching notebook: [(30, 100), (50, 150), (75, 200)]
            for low, high in [(30, 100), (50, 150), (75, 200)]:
                try:
                    # Canny edge detection
                    edges = cv2.Canny(blurred, low, high)
                    
                    # Dilation with (2,2) kernel, 2 iterations - matching notebook
                    edges = cv2.dilate(edges, np.ones((2, 2)), iterations=2)

                    # Find contours
                    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                    if not contours:
                        continue

                    # Sort by area (largest first) and take top 10 - matching notebook
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
                    all_contours.extend(contours)

                    # Find rectangular contour (document)
                    for contour in contours:
                        peri = cv2.arcLength(contour, True)
                        if peri == 0:
                            continue

                        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                        # Document should have 4 corners
                        if len(approx) == 4:
                            area = cv2.contourArea(approx)
                            # Check area > 10% of image area (matching notebook)
                            if area > gray.shape[0] * gray.shape[1] * 0.1:
                                # Scale contour back to original size
                                scaled_approx = (approx.reshape(4, 2) / scale).astype(np.int32)
                                document_contour = scaled_approx.reshape(4, 1, 2)
                                
                                if debug:
                                    print(f"      [Contour Detection] Found document with Canny({low}, {high})")
                                
                                # Scale all contours back to original size
                                scaled_contours = []
                                for c in all_contours:
                                    scaled_c = (c.astype(np.float32) / scale).astype(np.int32)
                                    scaled_contours.append(scaled_c)
                                
                                return document_contour, scaled_contours

                except Exception as canny_err:
                    if debug:
                        print(f"      [Contour Detection] Canny({low}, {high}) failed: {str(canny_err)}")
                    continue

            # Scale remaining contours if no document found
            scaled_contours = []
            for c in all_contours:
                scaled_c = (c.astype(np.float32) / scale).astype(np.int32)
                scaled_contours.append(scaled_c)

            if debug:
                print(f"      [Contour Detection] No 4-corner document found, returning {len(scaled_contours)} contours")

            return document_contour, scaled_contours

        except Exception as e:
            print(f"      [Contour Detection Error] {str(e)}")
            return None, []

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in clockwise order: top-left, top-right, bottom-right, bottom-left

        Args:
            pts: Array of 4 points

        Returns:
            Ordered points
        """
        rect = np.zeros((4, 2), dtype="float32")

        # Sum and difference to find corners
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect[0] = pts[np.argmin(s)]  # Top-left (smallest sum)
        rect[2] = pts[np.argmax(s)]  # Bottom-right (largest sum)
        rect[1] = pts[np.argmin(diff)]  # Top-right (smallest difference)
        rect[3] = pts[np.argmax(diff)]  # Bottom-left (largest difference)

        return rect

    def perspective_transform(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        output_width: int = 850,
        output_height: int = 1100,
    ) -> np.ndarray:
        """
        Apply perspective transformation to get bird's-eye view with error handling

        Args:
            image: Input image
            corners: 4 corner points of document
            output_width: Desired output width
            output_height: Desired output height

        Returns:
            Transformed image or original if transformation fails
        """
        try:
            # Validate corners
            if corners is None or len(corners) == 0:
                return image

            # Order the corners
            rect = self.order_points(corners.reshape(4, 2))

            # Destination points for perspective transform
            dst = np.array(
                [
                    [0, 0],
                    [output_width - 1, 0],
                    [output_width - 1, output_height - 1],
                    [0, output_height - 1],
                ],
                dtype="float32",
            )

            # Calculate perspective transform matrix
            M = cv2.getPerspectiveTransform(rect, dst)

            # Validate transform matrix
            if M is None:
                return image

            # Apply transformation
            warped = cv2.warpPerspective(image, M, (output_width, output_height))

            return warped

        except Exception as e:
            print(f"      [Perspective Transform Error] {str(e)}, returning original image")
            return image

    def correct_skew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect and correct skew using Hough Transform with error handling

        Args:
            image: Input image

        Returns:
            Tuple of (corrected_image, rotation_angle)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

            # Binarize
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Detect lines using Hough Transform
            lines = cv2.HoughLinesP(
                binary, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
            )

            if lines is None or len(lines) == 0:
                return image, 0.0

            # Calculate angles
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Normalize angles to -45 to 45 range
                if angle > 45:
                    angle = angle - 90
                elif angle < -45:
                    angle = angle + 90
                angles.append(angle)

            if not angles:
                return image, 0.0

            # Get median angle
            median_angle = np.median(angles)

            # Clip angle to reasonable range
            median_angle = np.clip(median_angle, -30, 30)

            if abs(median_angle) < 0.5:  # Skip very small rotations
                return image, 0.0

            # Rotate image
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )

            return rotated, median_angle

        except Exception as e:
            print(f"      [Skew Correction Error] {str(e)}, returning original image")
            return image, 0.0

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE with error handling

        Args:
            image: Input image

        Returns:
            Enhanced image
        """
        try:
            # Convert to LAB color space
            if len(image.shape) == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)

                # Apply CLAHE to L channel
                l = self.clahe.apply(l)

                # Merge channels
                enhanced = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            else:
                enhanced = self.clahe.apply(image)

            return enhanced

        except Exception as e:
            print(f"      [Contrast Enhancement Error] {str(e)}, returning original image")
            return image

    def denoise_image(self, image: np.ndarray, method: str = "bilateral") -> np.ndarray:
        """
        Remove noise from image with error handling and fallback

        Args:
            image: Input image
            method: 'bilateral', 'gaussian', or 'nlmeans'

        Returns:
            Denoised image
        """
        try:
            if method == "bilateral":
                # Bilateral filter preserves edges
                denoised = cv2.bilateralFilter(image, 9, 75, 75)

            elif method == "gaussian":
                # Gaussian blur
                denoised = cv2.GaussianBlur(image, (5, 5), 0)

            elif method == "nlmeans":
                # Non-local means denoising (slower but better quality)
                try:
                    if len(image.shape) == 3:
                        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
                    else:
                        denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
                except Exception as nlm_error:
                    print(
                        f"      [NLM Denoising] Failed: {str(nlm_error)}, falling back to bilateral"
                    )
                    denoised = cv2.bilateralFilter(image, 9, 75, 75)

            else:
                denoised = image.copy()

            return denoised

        except Exception as e:
            print(f"      [Denoising Error] {str(e)}, returning original image")
            return image

    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive thresholding for better text clarity

        Args:
            image: Input image (grayscale)

        Returns:
            Thresholded image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        return thresh

    def process_document(self, image: np.ndarray, auto_crop: bool = True) -> np.ndarray:
        """
        Complete document processing pipeline with sequential execution

        Args:
            image: Input image
            auto_crop: Whether to auto-detect and crop document

        Returns:
            Processed image
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image: empty or None")

        processed = image.copy()
        original_shape = processed.shape

        try:
            # Step 1: Auto-crop if enabled
            if auto_crop:
                try:
                    contour, _ = self.find_document_contours(processed)
                    if contour is not None:
                        processed = self.perspective_transform(processed, contour)
                        print(f"    [Auto-crop] Shape: {original_shape} → {processed.shape}")
                    else:
                        print("    [Auto-crop] No contour found, proceeding with original image")
                except Exception as e:
                    print(f"    [Auto-crop] Warning: {str(e)}, continuing with original")

            # Step 2: Correct skew - ensure execution
            try:
                processed, angle = self.correct_skew(processed)
                print(f"    [Skew Correction] Angle: {angle:.2f}°")
            except Exception as e:
                print(f"    [Skew Correction] Warning: {str(e)}, continuing")

            # Step 3: Denoise - ensure execution
            try:
                processed = self.denoise_image(processed, method="bilateral")
                print(f"    [Denoising] Completed")
            except Exception as e:
                print(f"    [Denoising] Warning: {str(e)}, continuing")

            # Step 4: Enhance contrast - ensure execution
            try:
                processed = self.enhance_contrast(processed)
                print(f"    [Contrast Enhancement] Completed")
            except Exception as e:
                print(f"    [Contrast Enhancement] Warning: {str(e)}, continuing")

            return processed

        except Exception as e:
            print(f"    [Error] Fatal error in processing: {str(e)}")
            # Return at least the last successful state
            return processed

    def create_document_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Create binary mask of document region

        Args:
            image: Input image

        Returns:
            Binary mask
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu's thresholding
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask
