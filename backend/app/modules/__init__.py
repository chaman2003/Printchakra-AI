"""
PrintChakra Document Processing Modules
Complete document scanning, enhancement, OCR, and export system

Module Organization:
- ocr/           : OCR and AI-powered document recognition
- voice/         : Voice control and speech-to-text
- image/         : Image processing and enhancement
- document/      : Document detection, conversion, and management
- pipeline/      : Processing pipeline orchestration
"""

__version__ = "2.1.0"
__author__ = "Chaman S"

# Import from submodules
# Heavy or machine-specific modules are optional: serverless hosting has no Tesseract, scikit-learn,
# Whisper or GPU, and OCR/voice run through Gemini there instead.
from app.core.paths import IS_CLOUD

try:
    from .api_endpoints import create_enhanced_endpoints
except ImportError:
    create_enhanced_endpoints = None

try:
    from .ocr import OCRModule, AIEnhancer, DocumentClassifier
except ImportError:
    OCRModule = AIEnhancer = DocumentClassifier = None

if IS_CLOUD:
    VoiceAIOrchestrator = WhisperTranscriptionService = VoiceChatService = None
else:
    from .voice import VoiceAIOrchestrator, WhisperTranscriptionService, VoiceChatService
from .image import ImageProcessingModule, ImageEnhancer
from .document import (
    DocumentDetector,
    detect_and_serialize,
    ScanningModule,
    validate_image_file,
    StorageModule,
    ExportModule,
)

# Import pipeline with error handling (optional dependency)
try:
    from .pipeline import EnhancedDocumentPipeline
except ImportError as e:
    import logging
    logging.warning(f"Pipeline module not fully available: {e}")
    EnhancedDocumentPipeline = None

from .utility import *

__all__ = [
    # OCR & AI
    "OCRModule",
    "AIEnhancer",
    "DocumentClassifier",
    # Voice
    "VoiceAIOrchestrator",
    "WhisperTranscriptionService",
    "VoiceChatService",
    # Image Processing
    "ImageProcessingModule",
    "ImageEnhancer",
    # Document Management
    "DocumentDetector",
    "detect_and_serialize",
    "ScanningModule",
    "validate_image_file",
    "StorageModule",
    "ExportModule",
    # Pipeline
    "EnhancedDocumentPipeline",
    "create_enhanced_endpoints",
]
