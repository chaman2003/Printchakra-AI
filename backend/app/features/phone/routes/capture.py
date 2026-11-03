"""
PrintChakra Backend - Phone Capture Routes

Camera capture endpoints from phone.
"""

import os
from app.core.paths import app_data_dir
import logging
import base64
from flask import jsonify, request
from werkzeug.utils import secure_filename
from app.features.phone.routes import phone_bp
from app.core import socketio
from app.core.middleware.cors import create_options_response

logger = logging.getLogger(__name__)

def _capture_folder():
    return os.path.join(app_data_dir(), "uploads", "captures")


@phone_bp.route("/capture", methods=["POST", "OPTIONS"])
def capture_image():
    """Capture image from phone camera."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        # Get base64 image data
        image_data = data.get("image")
        if not image_data:
            return jsonify({"success": False, "error": "No image data provided"}), 400
        
        # Remove data URL prefix if present
        if "," in image_data:
            image_data = image_data.split(",")[1]
        
        # Decode base64
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({"success": False, "error": "Invalid base64 image data"}), 400
        
        # Generate filename
        import uuid
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
        
        # Ensure folder exists
        os.makedirs(_capture_folder(), exist_ok=True)
        
        filepath = os.path.join(_capture_folder(), filename)
        
        # Save image
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        
        logger.info(f"[OK] Image captured: {filename}")
        
        return jsonify({
            "success": True,
            "filename": filename,
            "path": filepath,
            "size": len(image_bytes)
        })
    
    except Exception as e:
        logger.error(f"Capture error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@phone_bp.route("/capture/scan", methods=["POST", "OPTIONS"])
def scan_document():
    """Scan document from phone camera with edge detection."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        data = request.get_json()
        
        if not data or "image" not in data:
            return jsonify({"success": False, "error": "No image data provided"}), 400
        
        image_data = data.get("image")
        
        # Remove data URL prefix if present
        if "," in image_data:
            image_data = image_data.split(",")[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Process with edge detection (if available)
        try:
            from app.modules.image import process_scan
            processed_bytes = process_scan(image_bytes)
        except ImportError:
            # Fallback to raw image
            processed_bytes = image_bytes
        
        # Generate filename
        import uuid
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scan_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
        
        os.makedirs(_capture_folder(), exist_ok=True)
        filepath = os.path.join(_capture_folder(), filename)
        
        with open(filepath, "wb") as f:
            f.write(processed_bytes)
        
        logger.info(f"[OK] Document scanned: {filename}")
        
        return jsonify({
            "success": True,
            "filename": filename,
            "path": filepath,
            "size": len(processed_bytes)
        })
    
    except Exception as e:
        logger.error(f"Scan error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@phone_bp.route("/captures", methods=["GET", "OPTIONS"])
def list_captures():
    """List all captured images."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        if not os.path.exists(_capture_folder()):
            return jsonify({"success": True, "captures": []})
        
        captures = []
        for filename in os.listdir(_capture_folder()):
            filepath = os.path.join(_capture_folder(), filename)
            if os.path.isfile(filepath):
                captures.append({
                    "filename": filename,
                    "size": os.path.getsize(filepath),
                    "modified": os.path.getmtime(filepath)
                })
        
        # Sort by modified time, newest first
        captures.sort(key=lambda x: x["modified"], reverse=True)
        
        return jsonify({
            "success": True,
            "captures": captures,
            "total": len(captures)
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@phone_bp.route("/capture/<filename>", methods=["DELETE", "OPTIONS"])
def delete_capture(filename):
    """Delete a captured image."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        from werkzeug.utils import secure_filename
        
        filepath = os.path.join(_capture_folder(), secure_filename(filename))
        
        if not os.path.exists(filepath):
            return jsonify({"success": False, "error": "File not found"}), 404
        
        os.remove(filepath)
        logger.info(f"[OK] Capture deleted: {filename}")
        
        return jsonify({"success": True, "message": "File deleted"})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@phone_bp.route("/test-capture", methods=["POST", "OPTIONS"])
def upload_test_capture():
    """Upload calibration test capture from phone stream."""
    if request.method == "OPTIONS":
        return create_options_response()

    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"success": False, "error": "No file provided"}), 400

        safe_name = secure_filename(file.filename or "")
        if not safe_name:
            safe_name = "test_capture.jpg"

        os.makedirs(_capture_folder(), exist_ok=True)
        target_path = os.path.join(_capture_folder(), safe_name)
        file.save(target_path)
        payload = {
            "filename": safe_name,
            "size": os.path.getsize(target_path),
            "modified": os.path.getmtime(target_path),
        }
        socketio.emit("test_capture_received", payload)
        return jsonify({"success": True, "capture": payload})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@phone_bp.route("/test-captures", methods=["GET", "OPTIONS"])
def list_test_captures():
    """List uploaded calibration test captures."""
    if request.method == "OPTIONS":
        return create_options_response()

    try:
        captures = []
        if os.path.isdir(_capture_folder()):
            for filename in os.listdir(_capture_folder()):
                full_path = os.path.join(_capture_folder(), filename)
                if os.path.isfile(full_path):
                    captures.append(
                        {
                            "filename": filename,
                            "size": os.path.getsize(full_path),
                            "modified": os.path.getmtime(full_path),
                        }
                    )
        captures.sort(key=lambda x: x["modified"], reverse=True)
        return jsonify({"success": True, "captures": captures})
    except Exception as e:
        return jsonify({"success": False, "captures": [], "error": str(e)}), 500


@phone_bp.route("/test-captures/clear", methods=["POST", "OPTIONS"])
def clear_test_captures():
    """Clear calibration test captures."""
    if request.method == "OPTIONS":
        return create_options_response()
    try:
        removed = 0
        if os.path.isdir(_capture_folder()):
            for filename in os.listdir(_capture_folder()):
                full_path = os.path.join(_capture_folder(), filename)
                if os.path.isfile(full_path):
                    os.remove(full_path)
                    removed += 1
        socketio.emit("test_captures_cleared")
        return jsonify({"success": True, "removed": removed})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
