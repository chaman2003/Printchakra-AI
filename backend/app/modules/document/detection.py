"""
Document Detection Module - Improved Detection with Geometric Scoring
Multi-method document boundary detection with corner refinement from notebook
Enhanced for live camera feed accuracy with WHITE PAPER detection
"""

import json
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class DocumentDetector:
    """
    Advanced document detection with geometric scoring and corner refinement
    Optimized for live camera feed with improved accuracy
    Enhanced with WHITE PAPER detection for better accuracy
    """

    def __init__(self):
        """Initialize document detector with improved parameters"""
        self.min_area = 5000  # Reduced minimum contour area for better small document detection
        self.min_perimeter = 100
        # Temporal smoothing buffer for live camera feeds
        self.recent_detections: List[np.ndarray] = []
        self.max_recent = 5

    def detect_white_paper(self, image: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
        """
        Detect white paper using color-based segmentation.
        This method is specifically designed for detecting white documents
        on darker backgrounds.
        
        Args:
            image: Input image (BGR)
            debug: Print debug information
            
        Returns:
            Detected corners as numpy array (4, 2) or None
        """
        try:
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            height, width = image.shape[:2]
            
            # Method 1: White color detection in HSV
            # White has low saturation and high value
            lower_white_hsv = np.array([0, 0, 180])  # Low saturation, high brightness
            upper_white_hsv = np.array([180, 50, 255])
            white_mask_hsv = cv2.inRange(hsv, lower_white_hsv, upper_white_hsv)
            
            # Method 2: Use LAB color space - white paper has high L value
            l_channel = lab[:, :, 0]
            # Threshold to get bright regions (paper is bright)
            _, white_mask_lab = cv2.threshold(l_channel, 170, 255, cv2.THRESH_BINARY)
            
            # Method 3: High grayscale threshold for white paper
            _, white_mask_gray = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            
            # Combine masks (OR operation - any method detecting white)
            combined_mask = cv2.bitwise_or(white_mask_hsv, white_mask_lab)
            combined_mask = cv2.bitwise_or(combined_mask, white_mask_gray)
            
            # Clean up the mask
            kernel = np.ones((7, 7), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Fill holes in the mask
            combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
            combined_mask = cv2.erode(combined_mask, kernel, iterations=2)
            
            # Find contours in the white mask
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                if debug:
                    print("  ⚠️ No white regions found")
                return None
            
            # Sort by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            best_candidate = None
            best_score = -1000
            
            for c in contours[:5]:  # Check top 5 largest contours
                area = cv2.contourArea(c)
                
                # Skip too small or too large
                if area < height * width * 0.05 or area > height * width * 0.90:
                    continue
                
                # Approximate to polygon
                peri = cv2.arcLength(c, True)
                if peri == 0:
                    continue
                    
                # Try multiple epsilon values
                for epsilon in [0.02, 0.025, 0.03, 0.035, 0.04]:
                    approx = cv2.approxPolyDP(c, epsilon * peri, True)
                    
                    if len(approx) == 4:
                        # Score this candidate
                        quad = approx.reshape(4, 2)
                        scores = self.score_contour(quad.reshape(4, 1, 2), (height, width))
                        
                        # Bonus for being detected via white paper method
                        scores["score"] += 50
                        
                        if scores["score"] > best_score:
                            best_score = scores["score"]
                            best_candidate = quad
            
            if best_candidate is not None and best_score > 0:
                if debug:
                    print(f"  ✅ White paper detected (score: {best_score:.1f})")
                return best_candidate
            
            return None
            
        except Exception as e:
            if debug:
                print(f"  ⚠️ White paper detection error: {e}")
            return None

    def score_contour(self, contour: np.ndarray, image_shape: Tuple[int, int]) -> Dict:
        """
        Score contour based on geometric criteria - improved to avoid background edges
        Enhanced with confidence scoring for live camera feeds

        Args:
            contour: Contour to score
            image_shape: Image dimensions (height, width)

        Returns:
            Dictionary with scoring details including confidence
        """
        area = cv2.contourArea(contour)
        image_area = image_shape[0] * image_shape[1]
        area_ratio = area / image_area
        height, width = image_shape
        pts = contour.reshape(-1, 2)

        # Margin analysis - STRICT: penalize edges touching the image boundary
        min_x, min_y = np.min(pts[:, 0]), np.min(pts[:, 1])
        max_x, max_y = np.max(pts[:, 0]), np.max(pts[:, 1])

        # Calculate margins as percentage of image size
        left_margin = min_x / width
        right_margin = (width - max_x) / width
        top_margin = min_y / height
        bottom_margin = (height - max_y) / height
        min_margin = min(left_margin, right_margin, top_margin, bottom_margin)

        # CRITICAL: Heavy penalty if edges are too close to image boundary
        if min_margin < 0.03:
            margin_score = -700  # Increased penalty - very strong
        elif min_margin < 0.05:
            margin_score = -400  # Strong penalty
        elif min_margin < 0.10:
            margin_score = -100  # Moderate penalty
        elif min_margin < 0.15:
            margin_score = 0    # Neutral
        else:
            margin_score = 120  # Good margin bonus

        # Rectangularity check
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box_area = cv2.contourArea(box)

        if box_area > 0:
            rectangularity = area / box_area
        else:
            rectangularity = 0

        # Angle analysis - measure how close corners are to 90 degrees
        angles = []
        for i in range(len(pts)):
            p0, p1, p2 = pts[i], pts[(i + 1) % len(pts)], pts[(i + 2) % len(pts)]
            v1, v2 = p0 - p1, p2 - p1
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 > 0 and norm2 > 0:
                angle = np.arccos(np.clip(np.dot(v1, v2) / (norm1 * norm2), -1, 1))
                angles.append(np.degrees(angle))

        if angles:
            angle_error = np.mean([abs(a - 90) for a in angles])
        else:
            angle_error = 180

        # Scoring for rectangularity with tighter thresholds
        if angle_error < 6:
            rect_score = 120  # Excellent rectangle
        elif angle_error < 10:
            rect_score = 80
        elif angle_error < 15:
            rect_score = 40
        elif angle_error < 20:
            rect_score = 10
        else:
            rect_score = -120  # Poor rectangle

        # Aspect ratio check (prefer close to 1:1.5 or similar)
        if rect[1][0] > 0 and rect[1][1] > 0:
            aspect = max(rect[1]) / min(rect[1])
            
            # A4 aspect ratio is approx 1.414
            # Letter aspect ratio is approx 1.29
            if 1.35 <= aspect <= 1.48:  # A4 sweet spot
                aspect_score = 100
            elif 1.25 <= aspect <= 1.55:  # General document range (A4/Letter)
                aspect_score = 70
            elif 1.15 <= aspect <= 1.65:  # Broader range
                aspect_score = 40
            elif 1.1 <= aspect <= 2.5:  # Very broad document-like aspect ratio
                aspect_score = 10
            elif aspect > 2.5 or aspect < 1.1:
                aspect_score = -100
            else:
                aspect_score = 5
        else:
            aspect_score = 0

        # Area ratio scoring with better thresholds for live camera
        if 0.12 <= area_ratio <= 0.75:
            area_score = 120  # Ideal range
        elif 0.08 <= area_ratio < 0.12:
            area_score = 60   # Acceptable small document
        elif 0.75 < area_ratio <= 0.85:
            area_score = 20   # Large but acceptable
        elif area_ratio > 0.85:
            area_score = -500  # Too large, definitely capturing background
        else:
            area_score = -80  # Too small

        # Convexity check
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if hull_area > 0:
            solidity = area / hull_area
            if solidity > 0.97:
                solidity_score = 60
            elif solidity > 0.93:
                solidity_score = 30
            elif solidity > 0.88:
                solidity_score = 10
            else:
                solidity_score = -80
        else:
            solidity_score = 0

        # Perimeter-to-area ratio (documents should have moderate complexity)
        perimeter = cv2.arcLength(contour, True)
        if area > 0:
            complexity = (perimeter * perimeter) / area
            # Lower complexity = simpler shape (better for documents)
            if complexity < 60:
                complexity_score = 40
            elif complexity < 100:
                complexity_score = 20
            elif complexity < 150:
                complexity_score = 0
            else:
                complexity_score = -40
        else:
            complexity_score = 0

        # Total score
        score = (margin_score + rect_score + aspect_score + 
                area_score + solidity_score + complexity_score)

        # Calculate confidence (0-100)
        # High confidence means all metrics are good
        confidence_factors = []
        confidence_factors.append(100 if min_margin > 0.10 else max(0, min_margin * 1000))
        confidence_factors.append(100 if angle_error < 10 else max(0, 100 - angle_error * 2))
        confidence_factors.append(100 if 0.15 <= area_ratio <= 0.70 else max(0, 100 - abs(area_ratio - 0.4) * 200))
        confidence_factors.append(solidity * 100)
        
        confidence = np.mean(confidence_factors)

        return {
            "score": score,
            "confidence": confidence,
            "area_ratio": area_ratio,
            "angle_error": angle_error,
            "min_margin": min_margin,
            "rectangularity": rectangularity,
            "margin_score": margin_score,
            "rect_score": rect_score,
            "area_score": area_score,
            "aspect_ratio": aspect if rect[1][0] > 0 and rect[1][1] > 0 else 0,
        }

    def detect_document(self, image: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
        """
        Document detection with strict margin filtering and multi-method approach
        Using Canny edge detection matching notebook implementation.
        PRIORITY: White paper detection first, then edge-based fallback.

        Args:
            image: Input image (BGR)
            debug: Print debug information

        Returns:
            Detected corners as numpy array (4, 1, 2) or None
        """
        candidates = []
        orig_shape = image.shape[:2]

        try:
            # Validate input image
            if image is None or image.size == 0:
                if debug:
                    print("⚠️ Invalid input image")
                return None
            
            # PRIORITY 1: Try white paper detection first (best for white documents)
            if debug:
                print("  → Trying white paper detection...")
            white_paper_result = self.detect_white_paper(image, debug=debug)
            if white_paper_result is not None:
                # White paper detected - add with high priority
                scores = self.score_contour(white_paper_result.reshape(4, 1, 2), orig_shape)
                # Add bonus for white paper detection
                candidates.append((white_paper_result, "WhitePaper", scores["score"] + 100))

            # Downscale for faster processing (matching notebook's scale=0.25)
            scale = 0.25
            small = cv2.resize(image, (0, 0), fx=scale, fy=scale)
            
            # Convert to grayscale safely
            if len(small.shape) == 3:
                gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            else:
                gray_small = small
            
            # Also prepare full-size grayscale for fallback methods
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Gaussian blur (5,5) - matching notebook exactly
            blurred_small = cv2.GaussianBlur(gray_small, (5, 5), 0)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)

            # Method 1: Canny edge detection with notebook thresholds
            # Enhanced with additional threshold pairs for better live camera detection
            for low, high in [(25, 90), (30, 100), (40, 120), (50, 150), (75, 200)]:
                try:
                    edges = cv2.Canny(blurred_small, low, high)
                    # Dilate with (2,2) kernel, 2 iterations - matching notebook
                    edges = cv2.dilate(edges, np.ones((2, 2)), iterations=2)

                    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    # Sort by area (largest first) - matching notebook
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

                    for c in contours:
                        peri = cv2.arcLength(c, True)
                        if peri == 0:
                            continue
                        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                        if len(approx) == 4:
                            area = cv2.contourArea(approx)
                            # Check area > 8% of small image for better small document detection
                            if area > gray_small.shape[0] * gray_small.shape[1] * 0.08:
                                # Scale back to original coordinates
                                scaled_pts = (approx.reshape(4, 2) / scale).astype(np.float32)
                                # Also add to candidates with original-scale area
                                orig_area = area / (scale * scale)
                                candidates.append((scaled_pts, f"Canny_{low}_{high}", orig_area))
                except Exception as canny_err:
                    if debug:
                        print(f"  ⚠️ Canny {low}-{high} failed: {str(canny_err)}")
                    continue

            # Method 1b: Canny on full-size image for larger documents
            # Enhanced with better threshold ranges
            for low, high in [(40, 115), (45, 125), (55, 160), (65, 180), (70, 200)]:
                try:
                    edges = cv2.Canny(blurred, low, high)
                    edges = cv2.dilate(edges, np.ones((5, 5)), iterations=2)
                    edges = cv2.erode(edges, np.ones((2, 2)), iterations=1)

                    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    for c in contours:
                        area = cv2.contourArea(c)
                        if self.min_area < area < image.shape[0] * image.shape[1] * 0.80:
                            peri = cv2.arcLength(c, True)
                            # Try multiple epsilon values for better approximation
                            for epsilon in [0.012, 0.015, 0.020, 0.025, 0.028]:
                                approx = cv2.approxPolyDP(c, epsilon * peri, True)
                                if len(approx) == 4:
                                    candidates.append((approx.reshape(4, 2), f"CannyFull_{low}_{high}", area))
                except Exception as canny_full_err:
                    if debug:
                        print(f"  ⚠️ Canny full {low}-{high} failed: {str(canny_full_err)}")
                    continue

            # Method 2: Adaptive thresholding (fallback)
            try:
                adaptive = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 5
                )
                adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, np.ones((7, 7)), iterations=2)

                contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for c in contours:
                    area = cv2.contourArea(c)
                    if self.min_area < area < image.shape[0] * image.shape[1] * 0.75:
                        peri = cv2.arcLength(c, True)
                        if peri == 0:
                            continue
                        for epsilon in [0.020, 0.028, 0.038]:
                            approx = cv2.approxPolyDP(c, epsilon * peri, True)
                            if len(approx) == 4:
                                candidates.append((approx.reshape(4, 2), "Adaptive", area))
            except Exception as adaptive_err:
                if debug:
                    print(f"  ⚠️ Adaptive threshold failed: {str(adaptive_err)}")

            if not candidates:
                if debug:
                    print("⚠️ No valid contours found")
                return None

        except Exception as e:
            if debug:
                print(f"⚠️ Document detection error: {str(e)}")
            return None

        # Score all candidates - handle both (quad, method, score) and (quad, method, area) formats
        scored = []
        for item in candidates:
            quad, method = item[0], item[1]
            if method == "WhitePaper":
                # Already scored with bonus - use the score directly
                pre_score = item[2]
                scores = self.score_contour(quad.reshape(4, 1, 2), orig_shape)
                scores["score"] = pre_score  # Override with white paper bonus
                scored.append((quad, method, pre_score, scores))
            else:
                # Score edge-based candidates
                scores = self.score_contour(quad.reshape(4, 1, 2), orig_shape)
                scored.append((quad, method, scores["score"], scores))

        # Sort by score (higher is better)
        scored.sort(key=lambda x: x[2], reverse=True)

        if debug and len(scored) > 0:
            best = scored[0]
            print(f"✅ Document detected (score: {best[2]:.1f}, confidence: {best[3]['confidence']:.1f}%)")
            print(f"   Method: {best[1]}")
            print(f"   Margin: {best[3]['min_margin']:.3f}")
            print(f"   Area ratio: {best[3]['area_ratio']:.3f}")
            print(f"   Angle error: {best[3]['angle_error']:.1f}°")

        # Return best candidate if acceptable score and confidence
        # Lower threshold for white paper detection which is more reliable
        if len(scored) > 0:
            best = scored[0]
            # White paper detections are more reliable, use lower thresholds
            if best[1] == "WhitePaper" and best[2] > 20:
                return best[0].astype("int32").reshape(4, 1, 2)
            # Edge-based detections need higher confidence
            elif best[2] > 40 and best[3]["confidence"] > 30:
                return best[0].astype("int32").reshape(4, 1, 2)

        return None

    def refine_document_corners(
        self, image: np.ndarray, corners: np.ndarray, inset_pixels: int = 35
    ) -> np.ndarray:
        """
        Refine detected corners by moving them inward to avoid shadow boundaries.
        This accounts for lighting gradients at document edges.
        INCREASED inset to 35px to handle typical phone camera shadows.

        Args:
            image: Input image
            corners: Detected corner points
            inset_pixels: Number of pixels to move corners inward (default 35 for shadow avoidance)

        Returns:
            Refined corner points
        """
        if corners is None or len(corners) != 4:
            return corners

        corners = corners.reshape(4, 2).astype(np.float32)

        # Calculate center of document
        center = corners.mean(axis=0)
        
        # Calculate document size to scale inset appropriately
        # For larger detected documents, use proportionally larger inset
        doc_width = np.max(corners[:, 0]) - np.min(corners[:, 0])
        doc_height = np.max(corners[:, 1]) - np.min(corners[:, 1])
        doc_diagonal = np.sqrt(doc_width**2 + doc_height**2)
        
        # Scale inset: ~2.5% of diagonal, minimum 35px, maximum 80px
        scaled_inset = max(inset_pixels, min(int(doc_diagonal * 0.025), 80))

        # Move each corner toward center to account for shadows
        refined = corners.copy()
        for i in range(4):
            direction = center - corners[i]
            norm = np.linalg.norm(direction)
            if norm > 0:
                # Move inward by scaled inset pixels
                direction = direction / norm
                refined[i] = corners[i] + direction * scaled_inset

        return refined.astype(np.int32)

    def detect_document_refined(
        self, image: np.ndarray, debug: bool = False, inset: int = 35
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Document detection with corner refinement

        Args:
            image: Input image (BGR)
            debug: Print debug information
            inset: Inset pixels for corner refinement (default 35 for shadow removal)

        Returns:
            Tuple of (refined corners as numpy array (4, 1, 2), confidence score) or None
        """
        detected = self.detect_document(image, debug=debug)

        if detected is not None:
            detected_clean = detected.reshape(4, 2)
            refined = self.refine_document_corners(image, detected_clean, inset_pixels=inset)

            if debug:
                print(f"   Corners refined (inset: {inset}px)")

            # Calculate confidence based on detection quality
            # Re-score the refined contour for confidence
            scores = self.score_contour(refined.reshape(4, 1, 2), image.shape[:2])
            confidence = scores.get("confidence", 50.0)

            return refined.reshape(4, 1, 2).astype("int32"), confidence

        return None

    def detect_document_borders(self, image: np.ndarray) -> Dict:
        """
        Detect document borders and corner points using improved detection

        Args:
            image: Input image (BGR)

        Returns:
            Dict with border data, visualization info, and confidence score
        """
        try:
            # Use improved detection with corner refinement
            result = self.detect_document_refined(image, debug=False, inset=12)

            if result is None:
                return {"success": False, "message": "No document detected", "corners": [], "confidence": 0.0}

            document_contour, confidence = result

            # Extract corners
            corners = document_contour.reshape(4, 2)

            # Order corners
            corners = self._order_corners(corners)

            # Normalize corners to percentage coordinates (0-100)
            height, width = image.shape[:2]
            normalized_corners = self._normalize_corners(corners, width, height)

            # Calculate area
            contour_area = cv2.contourArea(document_contour)

            return {
                "success": True,
                "message": "Document detected",
                "corners": normalized_corners,  # Normalized [0-100]
                "pixel_corners": corners.tolist(),  # Pixel coordinates
                "contour_area": float(contour_area),
                "image_area": float(width * height),
                "coverage": float(contour_area / (width * height) * 100),
                "confidence": float(confidence),  # Detection confidence 0-100
            }

        except Exception as e:
            return {"success": False, "message": f"Detection error: {str(e)}", "corners": [], "confidence": 0.0}

    def _order_corners(self, contour: np.ndarray) -> np.ndarray:
        """
        Order corners in consistent order: top-left, top-right, bottom-right, bottom-left

        Args:
            contour: Contour with 4 points

        Returns:
            Ordered corner points
        """
        pts = contour.reshape(4, 2).astype(float)

        # Calculate the center
        center = pts.mean(axis=0)

        # Sort points by angle from center
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        sorted_indices = np.argsort(angles)

        return pts[sorted_indices].astype(int)

    def _normalize_corners(self, corners: np.ndarray, width: int, height: int) -> List[Dict]:
        """
        Convert pixel coordinates to normalized [0-100] coordinates

        Args:
            corners: Pixel coordinates
            width: Image width
            height: Image height

        Returns:
            List of normalized corner coordinates
        """
        normalized = []
        for i, corner in enumerate(corners):
            x_norm = (float(corner[0]) / width) * 100
            y_norm = (float(corner[1]) / height) * 100

            # Label corners
            corner_names = ["top-left", "top-right", "bottom-right", "bottom-left"]

            normalized.append(
                {
                    "id": i,
                    "name": corner_names[i],
                    "x": x_norm,
                    "y": y_norm,
                    "pixel_x": int(corner[0]),
                    "pixel_y": int(corner[1]),
                }
            )

        return normalized

    def draw_detection_overlay(self, image: np.ndarray, detection_result: Dict) -> np.ndarray:
        """
        Draw detection overlay on image

        Args:
            image: Input image
            detection_result: Detection result from detect_document_borders

        Returns:
            Image with drawn overlay
        """
        overlay = image.copy()
        height, width = image.shape[:2]

        if not detection_result["success"] or not detection_result["corners"]:
            return overlay

        # Draw border lines
        corners = detection_result["corners"]
        pixel_corners = []

        for corner in corners:
            x = int((corner["x"] / 100) * width)
            y = int((corner["y"] / 100) * height)
            pixel_corners.append([x, y])

        pixel_corners = np.array(pixel_corners, dtype=np.int32)

        # Draw polygon border
        cv2.polylines(overlay, [pixel_corners], True, (0, 255, 0), 3)

        # Draw corner points with labels
        for i, corner in enumerate(corners):
            x = int((corner["x"] / 100) * width)
            y = int((corner["y"] / 100) * height)

            # Draw circle at corner
            cv2.circle(overlay, (x, y), 8, (0, 255, 0), -1)
            cv2.circle(overlay, (x, y), 8, (255, 255, 255), 2)

            # Draw corner label
            label_text = corner["name"]
            cv2.putText(
                overlay, label_text, (x + 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        # Draw coverage info
        coverage = detection_result.get("coverage", 0)
        coverage_text = f"Coverage: {coverage:.1f}%"
        cv2.putText(overlay, coverage_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return overlay


def detect_and_serialize(image_bytes: bytes) -> Dict:
    """
    Detect document in image and return serialized result

    Args:
        image_bytes: Image data as bytes

    Returns:
        Serializable detection result
    """
    import io

    from PIL import Image as PILImage

    try:
        # Decode image
        pil_image = PILImage.open(io.BytesIO(image_bytes))
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Detect
        detector = DocumentDetector()
        result = detector.detect_document_borders(image)

        return result
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}", "corners": []}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read {image_path}")
            sys.exit(1)

        # Detect
        detector = DocumentDetector()
        result = detector.detect_document_borders(image)

        # Print result
        print(json.dumps(result, indent=2))

        # Display with overlay
        overlay = detector.draw_detection_overlay(image, result)
        cv2.imshow("Document Detection", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


print("[OK] Document detection module loaded (improved v4 - enhanced for live camera)")
