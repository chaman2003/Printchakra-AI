"""
Image processing module for document image enhancement and manipulation.
"""

from .enhancement import *
from .processing import *

import cv2
import numpy as np


def process_scan(image_bytes: bytes) -> bytes:
    """
    Process a scanned/captured image for document scanning.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Processed image bytes
    """
    try:
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return image_bytes
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply slight sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Convert back to BGR for JPEG encoding
        result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        
        # Encode to JPEG bytes
        _, encoded = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        return encoded.tobytes()
    
    except Exception:
        # Return original on any error
        return image_bytes


__all__ = ["enhancement", "processing", "process_scan"]
