"""
PrintChakra Backend - Dashboard Routes

Routes for dashboard functionality including health checks,
system information, and file listing.
"""

import os
import platform
import subprocess
from datetime import datetime
from flask import Blueprint, jsonify, current_app
from app.core.config import get_data_dirs

dashboard_bp = Blueprint('dashboard', __name__)


# =============================================================================
# HEALTH CHECK ENDPOINTS
# =============================================================================

@dashboard_bp.route("/")
def index():
    """Health check endpoint with feature information."""
    return jsonify({
        "status": "ok",
        "service": "PrintChakra Backend",
        "version": "2.0.0",
        "features": {
            "basic_processing": True,
            "advanced_pipeline": True,
            "ocr": True,
            "socket_io": True,
            "pdf_export": True,
            "document_classification": True,
            "quality_validation": True,
            "batch_processing": True,
        },
        "endpoints": {
            "health": "/health",
            "upload": "/phone/upload",
            "files": "/files",
            "processed": "/document/processed/<filename>",
            "delete": "/document/delete/<filename>",
            "print": "/print",
            "ocr": "/ocr/<filename>",
        },
    })


@dashboard_bp.route("/health")
def health_check():
    """Detailed health check endpoint."""
    dirs = get_data_dirs()
    
    # Check directory accessibility
    dir_status = {}
    for name, path in dirs.items():
        if name.endswith('_DIR'):
            dir_status[name] = {
                "path": path,
                "exists": os.path.exists(path),
                "writable": os.access(path, os.W_OK) if os.path.exists(path) else False,
            }
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "directories": dir_status,
        "python_version": platform.python_version(),
        "platform": platform.system(),
    })


@dashboard_bp.route("/favicon.ico")
def favicon():
    """Serve favicon to prevent 404 errors."""
    return "", 204


# =============================================================================
# FILE LISTING ENDPOINTS
# =============================================================================

@dashboard_bp.route("/files")
def list_files():
    """List all processed files with processing status."""
    try:
        dirs = get_data_dirs()
        PROCESSED_DIR = dirs['PROCESSED_DIR']
        TEXT_DIR = dirs['TEXT_DIR']
        OCR_DATA_DIR = dirs['OCR_DATA_DIR']
        
        # Import OCR processor for checking OCR results
        try:
            from app.features.document.ocr.paddle_ocr_service import PaddleOCRService
            ocr_processor = PaddleOCRService()
        except ImportError:
            ocr_processor = None
        
        files = []
        
        # List files in processed directory
        if os.path.exists(PROCESSED_DIR):
            for filename in os.listdir(PROCESSED_DIR):
                file_path = os.path.join(PROCESSED_DIR, filename)
                if os.path.isfile(file_path):
                    file_stat = os.stat(file_path)
                    
                    # Check if text file exists
                    text_filename = f"{os.path.splitext(filename)[0]}.txt"
                    text_path = os.path.join(TEXT_DIR, text_filename)
                    has_text = os.path.exists(text_path)
                    
                    # Check if OCR result exists
                    has_ocr = False
                    if ocr_processor:
                        has_ocr = ocr_processor.has_ocr_result(filename)
                    
                    file_info = {
                        "filename": filename,
                        "size": file_stat.st_size,
                        "created": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                        "has_text": has_text,
                        "has_ocr": has_ocr,
                        "processing": False,
                    }
                    files.append(file_info)
        
        # Sort by creation time (newest first)
        files.sort(key=lambda x: x["created"], reverse=True)
        
        return jsonify({"files": files, "count": len(files)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# SYSTEM INFORMATION ENDPOINTS
# =============================================================================

@dashboard_bp.route("/system/info", methods=["GET"])
def system_info():
    """Get system information including printers."""
    from app.features.dashboard.services.system_service import SystemService
    
    service = SystemService()
    info = service.get_system_info()
    
    return jsonify(info)


@dashboard_bp.route("/system/set-default-printer", methods=["POST"])
def set_default_printer():
    """Set the default printer."""
    from flask import request
    from app.features.dashboard.services.system_service import SystemService
    
    data = request.get_json() or {}
    printer_name = data.get("printer_name")
    
    if not printer_name:
        return jsonify({"success": False, "error": "No printer name provided"}), 400
    
    service = SystemService()
    result = service.set_default_printer(printer_name)
    
    if result["success"]:
        return jsonify(result)
    else:
        return jsonify(result), 500


# =============================================================================
# OPTIONS HANDLERS
# =============================================================================

@dashboard_bp.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@dashboard_bp.route("/<path:path>", methods=["OPTIONS"])
def handle_options(path):
    """Handle CORS preflight requests."""
    from app.core.middleware.cors import create_options_response
    return create_options_response()
