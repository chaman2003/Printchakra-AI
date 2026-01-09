"""
PrintChakra Backend - Phone Capture Routes

Routes for phone camera capture and upload functionality.
"""

import os
import uuid
import traceback
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from app.core.config import get_data_dirs
from app.core.middleware.cors import create_options_response

phone_bp = Blueprint('phone', __name__)


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# =============================================================================
# UPLOAD ENDPOINTS
# =============================================================================

@phone_bp.route("/upload", methods=["POST", "OPTIONS"])
def upload_file():
    """
    Upload and process an image from phone camera.
    
    Accepts multipart form data with:
    - file: The image file
    - autoCrop: Whether to auto-crop document (default: true)
    - aiEnhance: Whether to apply AI enhancement (default: true)
    - strictQuality: Whether to apply strict quality checks (default: true)
    """
    if request.method == "OPTIONS":
        return create_options_response()
    
    dirs = get_data_dirs()
    UPLOAD_DIR = dirs['UPLOAD_DIR']
    PROCESSED_DIR = dirs['PROCESSED_DIR']
    
    try:
        # Check for file in request
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files["file"]
        
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Get processing options
        auto_crop = request.form.get("autoCrop", "true").lower() == "true"
        ai_enhance = request.form.get("aiEnhance", "true").lower() == "true"
        strict_quality = request.form.get("strictQuality", "true").lower() == "true"
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_ext = os.path.splitext(secure_filename(file.filename))[1].lower()
        unique_id = str(uuid.uuid4())[:8]
        
        upload_filename = f"upload_{timestamp}_{unique_id}{original_ext}"
        processed_filename = f"processed_{timestamp}_{unique_id}.jpg"
        
        upload_path = os.path.join(UPLOAD_DIR, upload_filename)
        processed_path = os.path.join(PROCESSED_DIR, processed_filename)
        
        # Ensure directories exist
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        
        # Save uploaded file
        file.save(upload_path)
        current_app.logger.info(f"[UPLOAD] Saved: {upload_path}")
        
        # Process the image
        from app.features.phone.upload.processor import ImageProcessor
        
        processor = ImageProcessor()
        result = processor.process_upload(
            upload_path,
            processed_path,
            auto_crop=auto_crop,
            ai_enhance=ai_enhance,
            strict_quality=strict_quality
        )
        
        if result["success"]:
            # Emit socket event for real-time update
            try:
                # Import socketio from the main app module (where it's initialized)
                # This ensures we use the same SocketIO instance connected to the Flask app
                from __main__ import socketio
                current_app.logger.info(f"[Socket.IO] Emitting new_file event for {processed_filename}")
                socketio.emit("new_file", {
                    "filename": processed_filename,
                    "original": upload_filename,
                    "timestamp": timestamp,
                })
                current_app.logger.info(f"[Socket.IO] Emitted successfully")
            except ImportError:
                # Fallback: try importing from app.core (for testing)
                try:
                    from app.core import socketio
                    current_app.logger.info(f"[Socket.IO] Using fallback import from app.core for {processed_filename}")
                    socketio.emit("new_file", {
                        "filename": processed_filename,
                        "original": upload_filename,
                        "timestamp": timestamp,
                    })
                except Exception as e2:
                    current_app.logger.warning(f"Socket emit failed (fallback): {e2}")
            except Exception as e:
                current_app.logger.warning(f"Socket emit failed: {e}")
            
            return jsonify({
                "success": True,
                "message": "File uploaded and processed successfully",
                "filename": processed_filename,
                "original_filename": upload_filename,
                "processing": result.get("processing_info", {}),
            })
        else:
            return jsonify({
                "success": False,
                "error": result.get("error", "Processing failed"),
            }), 500
    
    except Exception as e:
        current_app.logger.error(f"Upload error: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


# =============================================================================
# QUALITY VALIDATION ENDPOINTS
# =============================================================================

@phone_bp.route("/validate-quality", methods=["POST", "OPTIONS"])
def validate_quality():
    """
    Validate image quality before full processing.
    
    Accepts base64 encoded image data or file upload.
    Returns quality metrics and recommendations.
    """
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        from app.features.phone.quality.validator import QualityValidator
        
        validator = QualityValidator()
        
        # Check if file upload or base64 data
        if "file" in request.files:
            file = request.files["file"]
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                file.save(tmp.name)
                result = validator.validate_image(tmp.name)
                os.unlink(tmp.name)
        
        elif request.is_json and "image_data" in request.json:
            import base64
            import tempfile
            
            image_data = request.json["image_data"]
            # Remove data URL prefix if present
            if "," in image_data:
                image_data = image_data.split(",")[1]
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(base64.b64decode(image_data))
                tmp.flush()
                result = validator.validate_image(tmp.name)
                os.unlink(tmp.name)
        else:
            return jsonify({"error": "No image provided"}), 400
        
        return jsonify(result)
    
    except Exception as e:
        current_app.logger.error(f"Quality validation error: {str(e)}")
        return jsonify({"error": str(e)}), 500


# =============================================================================
# CAMERA VALIDATION ENDPOINTS
# =============================================================================

@phone_bp.route("/validate-camera", methods=["POST", "OPTIONS"])
def validate_camera():
    """
    Validate that phone camera is capturing frames.
    
    Accepts a test frame to validate camera connectivity.
    """
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        # Check if we received a frame
        if "frame" in request.files:
            file = request.files["frame"]
            if file.filename:
                return jsonify({
                    "success": True,
                    "message": "Camera frame received",
                    "timestamp": datetime.now().isoformat(),
                })
        
        elif request.is_json and "frame_data" in request.json:
            return jsonify({
                "success": True,
                "message": "Camera frame received",
                "timestamp": datetime.now().isoformat(),
            })
        
        return jsonify({
            "success": False,
            "error": "No camera frame provided",
        }), 400
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500
