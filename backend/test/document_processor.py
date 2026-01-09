"""
CamScanner Document Processing Module
Consolidated processing functions used by all test scripts
"""
import cv2
import numpy as np
import os
from typing import Tuple, Optional


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order points: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply perspective transform to get bird's eye view of document"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def find_document(image: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
    """
    Detect document boundaries using multi-scale Canny edge detection.
    Returns 4 corner points or None if not detected.
    
    Matching notebook implementation:
    - Scale down to 25% for faster processing
    - Gaussian blur (5,5)
    - Multiple Canny thresholds: [(30, 100), (50, 150), (75, 200)]
    - Dilation with (2,2) kernel, 2 iterations
    - Area threshold: 10% of image area
    """
    try:
        # Validate input
        if image is None or image.size == 0:
            if debug:
                print("find_document: Invalid input image")
            return None
        
        # Scale down for faster processing (matching notebook)
        scale = 0.25
        small = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        
        # Convert to grayscale safely
        if len(small.shape) == 3:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        else:
            gray = small.copy()
        
        # Gaussian blur (5,5) - matching notebook exactly
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try multiple Canny thresholds - matching notebook
        for low, high in [(30, 100), (50, 150), (75, 200)]:
            try:
                edges = cv2.Canny(blurred, low, high)
                # Dilation with (2,2) kernel, 2 iterations - matching notebook
                edges = cv2.dilate(edges, np.ones((2, 2)), iterations=2)
                
                contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                # Sort by area and take top 10 - matching notebook
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
                
                for c in contours:
                    peri = cv2.arcLength(c, True)
                    if peri == 0:
                        continue
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                    if len(approx) == 4:
                        area = cv2.contourArea(approx)
                        # Check area > 10% of image (matching notebook)
                        if area > gray.shape[0] * gray.shape[1] * 0.1:
                            if debug:
                                print(f"find_document: Found with Canny({low}, {high})")
                            return (approx.reshape(4, 2) / scale).astype(np.float32)
                            
            except Exception as canny_err:
                if debug:
                    print(f"find_document: Canny({low}, {high}) failed: {str(canny_err)}")
                continue
        
        if debug:
            print("find_document: No 4-corner document found")
        return None
        
    except Exception as e:
        if debug:
            print(f"find_document: Error - {str(e)}")
        return None


def estimate_background(gray: np.ndarray, kernel_ratio: int = 8) -> np.ndarray:
    """
    Estimate background illumination using downscaling for speed.
    Uses morphological closing to capture local background.
    """
    h, w = gray.shape
    
    # Downscale for faster processing
    max_dim = 800
    scale = min(max_dim / max(h, w), 1.0)
    
    if scale < 1.0:
        small = cv2.resize(gray, (int(w * scale), int(h * scale)))
    else:
        small = gray
    
    sh, sw = small.shape
    k_size = max(sh, sw) // kernel_ratio
    k_size = k_size if k_size % 2 == 1 else k_size + 1
    k_size = max(k_size, 31)
    k_size = min(k_size, 151)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    bg = cv2.GaussianBlur(bg, (k_size, k_size), 0)
    
    if scale < 1.0:
        bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
        bg = cv2.GaussianBlur(bg, (51, 51), 0)
    
    return bg


def remove_shadows(gray: np.ndarray) -> np.ndarray:
    """
    Remove shadows from document using illumination normalization.
    Formula: normalized = (pixel / background) * 255
    """
    background = estimate_background(gray, kernel_ratio=6)
    
    result = np.zeros_like(gray, dtype=np.float32)
    mask = background > 10  # Avoid division by zero
    result[mask] = (gray[mask].astype(np.float32) / background[mask].astype(np.float32)) * 255
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def whiten_background(gray: np.ndarray) -> np.ndarray:
    """
    Scale image so 95th percentile becomes white (255).
    This ensures white paper background appears truly white.
    """
    p95 = np.percentile(gray, 95)
    if 0 < p95 < 250:
        result = np.clip(gray.astype(np.float32) * (255.0 / p95), 0, 255).astype(np.uint8)
        return result
    return gray


def apply_gamma(gray: np.ndarray, gamma: float = 0.9) -> np.ndarray:
    """
    Apply gamma correction.
    gamma < 1.0 darkens text, gamma > 1.0 lightens.
    """
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(gray, table)


def enhance_document(gray: np.ndarray) -> np.ndarray:
    """
    Complete CamScanner-style enhancement pipeline.
    Steps: Shadow removal (illumination normalization)
    """
    # Remove shadows via illumination normalization
    result = remove_shadows(gray)
    
    return result


def process_document(image: np.ndarray, detect_bounds: bool = True) -> Tuple[np.ndarray, dict]:
    """
    Full document processing pipeline.
    
    Returns:
        processed: Final processed grayscale image
        info: Dictionary with processing metadata
    """
    info = {
        'original_size': (image.shape[1], image.shape[0]),
        'detected_document': False,
        'cropped_size': None
    }
    
    # Step 1: Detect and crop document
    if detect_bounds:
        pts = find_document(image)
        if pts is not None:
            warped = four_point_transform(image, pts)
            info['detected_document'] = True
            info['cropped_size'] = (warped.shape[1], warped.shape[0])
        else:
            warped = image
            info['cropped_size'] = info['original_size']
    else:
        warped = image
        info['cropped_size'] = info['original_size']
    
    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Step 3: Enhance document
    processed = enhance_document(gray)
    
    return processed, info


def generate_pipeline_images(image: np.ndarray, output_dir: str) -> dict:
    """
    Generate all intermediate processing step images.
    Used for visualization and debugging.
    
    Returns dictionary mapping step names to file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    outputs = {}
    
    # Step 1: Original
    path = os.path.join(output_dir, "step_01_original.jpg")
    cv2.imwrite(path, image)
    outputs['original'] = path
    
    # Step 2: Downscale
    scale = 0.25
    small = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    path = os.path.join(output_dir, "step_02_downscaled.jpg")
    cv2.imwrite(path, small)
    outputs['downscaled'] = path

    # Step 3: Grayscale (Detect)
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    path = os.path.join(output_dir, "step_03_grayscale_detect.jpg")
    cv2.imwrite(path, gray_small)
    outputs['grayscale_detect'] = path

    # Step 4: Blur
    blurred = cv2.GaussianBlur(gray_small, (5, 5), 0)
    path = os.path.join(output_dir, "step_04_blurred.jpg")
    cv2.imwrite(path, blurred)
    outputs['blurred'] = path

    # Step 5: Edges
    edges = cv2.Canny(blurred, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((2, 2)), iterations=2)
    path = os.path.join(output_dir, "step_05_edges.jpg")
    cv2.imwrite(path, edges_dilated)
    outputs['edges'] = path

    # Step 6: Document detection
    pts = find_document(image)
    vis = image.copy()
    if pts is not None:
        pts_int = pts.astype(int)
        cv2.polylines(vis, [pts_int], True, (0, 255, 0), 3)
    path = os.path.join(output_dir, "step_06_contour.jpg")
    cv2.imwrite(path, vis)
    outputs['contour'] = path
    
    # Step 7: Warp
    if pts is not None:
        warped = four_point_transform(image, pts)
    else:
        warped = image
    path = os.path.join(output_dir, "step_07_warped.jpg")
    cv2.imwrite(path, warped)
    outputs['warped'] = path
    
    # Step 8: Grayscale (Process)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    path = os.path.join(output_dir, "step_08_grayscale_process.jpg")
    cv2.imwrite(path, gray)
    outputs['grayscale_process'] = path
    
    # Step 9: Downscale (Background)
    h, w = gray.shape
    max_dim = 800
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        small_bg = cv2.resize(gray, (int(w * scale), int(h * scale)))
    else:
        small_bg = gray
    path = os.path.join(output_dir, "step_09_downscaled_bg.jpg")
    cv2.imwrite(path, small_bg)
    outputs['downscaled_bg'] = path
    
    # Step 10: Morph Background
    sh, sw = small_bg.shape
    k = max(sh, sw) // 8
    k = k if k % 2 == 1 else k + 1
    k = max(31, min(k, 127))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg_raw = cv2.morphologyEx(small_bg, cv2.MORPH_CLOSE, kernel)
    path = os.path.join(output_dir, "step_10_morph_bg.jpg")
    cv2.imwrite(path, bg_raw)
    outputs['morph_bg'] = path

    # Step 11: Smooth/Upscale Background
    bg_smoothed = cv2.GaussianBlur(bg_raw, (k, k), 0)
    if scale < 1.0:
        bg = cv2.resize(bg_smoothed, (w, h), interpolation=cv2.INTER_LINEAR)
        bg = cv2.GaussianBlur(bg, (31, 31), 0)
    else:
        bg = bg_smoothed
    path = os.path.join(output_dir, "step_11_final_bg.jpg")
    cv2.imwrite(path, bg)
    outputs['final_bg'] = path
    
    # Step 12: Shadow Removal (Final)
    result = np.zeros_like(gray, dtype=np.float32)
    mask = bg > 10
    result[mask] = (gray[mask].astype(np.float32) / bg[mask].astype(np.float32)) * 255
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Final output - shadow removed is the final result
    path = os.path.join(output_dir, "step_12_final_output.jpg")
    cv2.imwrite(path, result)
    outputs['final'] = path
    
    return outputs
