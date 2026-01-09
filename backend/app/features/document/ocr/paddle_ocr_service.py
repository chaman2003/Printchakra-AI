"""
PrintChakra Backend - PaddleOCR Service

Wrapper service for PaddleOCR functionality.
This module provides a clean interface to the existing OCR processor.
"""

import os
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class OCRResult:
    """Structured OCR result with bounding boxes and confidence."""
    
    def __init__(self):
        self.raw_results: list = []
        self.structured_units: list = []
        self.full_text: str = ""
        self.derived_title: str = ""
        self.confidence_avg: float = 0.0
        self.word_count: int = 0
        self.timestamp: str = ""
        self.processing_time_ms: float = 0.0
        self.image_dimensions: tuple = (0, 0)
    
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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OCRResult':
        """Create OCRResult from dictionary."""
        result = cls()
        result.raw_results = data.get("raw_results", [])
        result.structured_units = data.get("structured_units", [])
        result.full_text = data.get("full_text", "")
        result.derived_title = data.get("derived_title", "")
        result.confidence_avg = data.get("confidence_avg", 0.0)
        result.word_count = data.get("word_count", 0)
        result.timestamp = data.get("timestamp", "")
        result.processing_time_ms = data.get("processing_time_ms", 0.0)
        dims = data.get("image_dimensions", [0, 0])
        result.image_dimensions = tuple(dims) if dims else (0, 0)
        return result


class PaddleOCRService:
    """
    Service wrapper for PaddleOCR operations.
    
    Provides a clean interface for OCR processing, result storage,
    and retrieval. Delegates to the existing paddle_ocr module.
    """
    
    def __init__(self, ocr_data_dir: str):
        """
        Initialize OCR service.
        
        Args:
            ocr_data_dir: Directory to store OCR results
        """
        self.ocr_data_dir = ocr_data_dir
        os.makedirs(ocr_data_dir, exist_ok=True)
        self._processor = None
    
    def _get_processor(self):
        """Lazy load the OCR processor."""
        if self._processor is None:
            try:
                from app.modules.ocr.paddle_ocr import PaddleOCRProcessor
                self._processor = PaddleOCRProcessor(self.ocr_data_dir)
            except ImportError as e:
                logger.error(f"Failed to import PaddleOCRProcessor: {e}")
                raise
        return self._processor
    
    def process_image(self, image_path: str) -> OCRResult:
        """
        Run OCR on an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            OCRResult with extracted text and metadata
        """
        try:
            processor = self._get_processor()
            result = processor.process_image(image_path)
            
            # Convert to our OCRResult if needed
            if hasattr(result, 'to_dict'):
                # Already an OCRResult-like object
                our_result = OCRResult()
                data = result.to_dict()
                our_result.raw_results = data.get("raw_results", [])
                our_result.structured_units = data.get("structured_units", [])
                our_result.full_text = data.get("full_text", "")
                our_result.derived_title = data.get("derived_title", "")
                our_result.confidence_avg = data.get("confidence_avg", 0.0)
                our_result.word_count = data.get("word_count", 0)
                our_result.timestamp = data.get("timestamp", "")
                our_result.processing_time_ms = data.get("processing_time_ms", 0.0)
                dims = data.get("image_dimensions", [0, 0])
                our_result.image_dimensions = tuple(dims) if dims else (0, 0)
                return our_result
            
            return result
        
        except Exception as e:
            logger.error(f"OCR processing error: {e}")
            # Return empty result on error
            result = OCRResult()
            result.timestamp = datetime.now().isoformat()
            result.derived_title = "Error Processing Document"
            return result
    
    def save_result(self, filename: str, result: OCRResult) -> str:
        """
        Save OCR result to JSON file.
        
        Args:
            filename: Original filename (used to generate result filename)
            result: OCR result to save
            
        Returns:
            Path to saved JSON file
        """
        try:
            json_path = self._get_result_path(filename)
            
            # Convert result to dict
            if hasattr(result, 'to_dict'):
                data = result.to_dict()
            else:
                data = result
            
            # Add metadata
            data['source_filename'] = filename
            data['saved_at'] = datetime.now().isoformat()
            
            # Save to file
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"OCR result saved: {json_path}")
            return json_path
        
        except Exception as e:
            logger.error(f"Failed to save OCR result: {e}")
            raise
    
    def _get_result_path(self, filename: str) -> str:
        """
        Get the path for OCR result JSON file.
        
        Args:
            filename: Original filename
            
        Returns:
            Path to OCR result JSON file
        """
        base_name = os.path.splitext(filename)[0]
        json_filename = f"{base_name}_ocr.json"
        return os.path.join(self.ocr_data_dir, json_filename)
    
    def load_result(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Load existing OCR result.
        
        Args:
            filename: Original filename
            
        Returns:
            OCR result dict or None if not found
        """
        try:
            base_name = os.path.splitext(filename)[0]
            json_filename = f"{base_name}_ocr.json"
            json_path = os.path.join(self.ocr_data_dir, json_filename)
            
            if not os.path.exists(json_path):
                return None
            
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        except Exception as e:
            logger.warning(f"Failed to load OCR result: {e}")
            return None
    
    def has_ocr_result(self, filename: str) -> bool:
        """
        Check if OCR result exists for a file.
        
        Args:
            filename: Original filename
            
        Returns:
            True if OCR result exists
        """
        base_name = os.path.splitext(filename)[0]
        json_filename = f"{base_name}_ocr.json"
        json_path = os.path.join(self.ocr_data_dir, json_filename)
        return os.path.exists(json_path)
    
    def delete_result(self, filename: str) -> bool:
        """
        Delete OCR result for a file.
        
        Args:
            filename: Original filename
            
        Returns:
            True if deleted, False if not found
        """
        try:
            base_name = os.path.splitext(filename)[0]
            json_filename = f"{base_name}_ocr.json"
            json_path = os.path.join(self.ocr_data_dir, json_filename)
            
            if os.path.exists(json_path):
                os.remove(json_path)
                return True
            return False
        
        except Exception as e:
            logger.error(f"Failed to delete OCR result: {e}")
            return False
    
    def get_text(self, filename: str) -> str:
        """
        Get plain text from OCR result.
        
        Args:
            filename: Original filename
            
        Returns:
            Extracted text or empty string
        """
        result = self.load_result(filename)
        if result:
            return result.get("full_text", "")
        return ""
    
    def get_title(self, filename: str) -> str:
        """
        Get derived title from OCR result.
        
        Args:
            filename: Original filename
            
        Returns:
            Derived title or empty string
        """
        result = self.load_result(filename)
        if result:
            return result.get("derived_title", "")
        return ""


# Factory function for compatibility
def get_ocr_service(ocr_data_dir: str) -> PaddleOCRService:
    """
    Get OCR service instance.
    
    Args:
        ocr_data_dir: Directory for OCR results
        
    Returns:
        PaddleOCRService instance
    """
    return PaddleOCRService(ocr_data_dir)
