"""
PrintChakra Backend - OCR Routes

Routes for OCR (Optical Character Recognition) operations.
"""

import os
import traceback
from flask import Blueprint, request, jsonify, current_app
from app.core.config import get_data_dirs
from app.core.middleware.cors import create_options_response

ocr_bp = Blueprint('ocr', __name__)


# =============================================================================
# SINGLE FILE OCR ENDPOINTS
# =============================================================================

@ocr_bp.route("/<path:filename>", methods=["POST", "OPTIONS"])
def run_ocr(filename):
    """
    Run PaddleOCR on a processed image.
    
    Returns structured OCR results with bounding boxes and derived title.
    Notifies frontend via Socket.IO on completion.
    """
    if request.method == "OPTIONS":
        return create_options_response()
    
    dirs = get_data_dirs()
    PROCESSED_DIR = dirs['PROCESSED_DIR']
    UPLOAD_DIR = dirs['UPLOAD_DIR']
    OCR_DATA_DIR = dirs['OCR_DATA_DIR']
    
    try:
        from app.features.document.ocr.paddle_ocr_service import PaddleOCRService
        
        # Security: prevent directory traversal
        if ".." in filename:
            return jsonify({"error": "Invalid filename"}), 400
        
        # Find the image file
        image_path = os.path.join(PROCESSED_DIR, filename)
        if not os.path.exists(image_path):
            # Try uploads directory
            image_path = os.path.join(UPLOAD_DIR, filename)
        
        if not os.path.exists(image_path):
            return jsonify({"error": f"File not found: {filename}"}), 404
        
        current_app.logger.info(f"OCR PROCESSING: {filename}")
        
        # Get or create OCR service
        service = PaddleOCRService(OCR_DATA_DIR)
        
        # Run OCR
        result = service.process_image(image_path)
        
        # Save result
        service.save_result(filename, result)
        
        # Auto-rename file based on OCR content if title is available
        new_filename = filename
        if result.derived_title and result.derived_title not in ["Untitled Document", "Error Processing Document"]:
            try:
                # Generate sanitized filename from derived title
                import re
                sanitized_title = result.derived_title.lower().strip()
                sanitized_title = re.sub(r'[^a-z0-9]+', '_', sanitized_title)
                sanitized_title = re.sub(r'_+', '_', sanitized_title)
                sanitized_title = sanitized_title.strip('_')
                
                # Limit length
                if len(sanitized_title) > 50:
                    sanitized_title = sanitized_title[:50].rsplit('_', 1)[0]
                
                if sanitized_title and sanitized_title != "untitled":
                    # Extract timestamp from original filename if present
                    timestamp_match = re.search(r'_(\d{8}_\d{6})_', filename)
                    timestamp = timestamp_match.group(1) if timestamp_match else None
                    
                    # Get extension
                    ext = os.path.splitext(filename)[1]
                    
                    # Build new filename
                    if timestamp:
                        new_filename_base = f"{sanitized_title}_{timestamp}"
                    else:
                        new_filename_base = sanitized_title
                    
                    new_filename = f"{new_filename_base}{ext}"
                    new_image_path = os.path.join(os.path.dirname(image_path), new_filename)
                    
                    # Avoid overwriting existing files
                    counter = 1
                    while os.path.exists(new_image_path) and new_image_path != image_path:
                        new_filename = f"{new_filename_base}_{counter}{ext}"
                        new_image_path = os.path.join(os.path.dirname(image_path), new_filename)
                        counter += 1
                    
                    # Rename the file
                    if new_image_path != image_path and not os.path.exists(new_image_path):
                        os.rename(image_path, new_image_path)
                        current_app.logger.info(f"File renamed: {filename} -> {new_filename}")
                        
                        # Update OCR result with new filename
                        service.save_result(new_filename, result)
                        
                        # Remove old OCR result file
                        old_ocr_path = service._get_result_path(filename)
                        if os.path.exists(old_ocr_path) and old_ocr_path != service._get_result_path(new_filename):
                            try:
                                os.remove(old_ocr_path)
                            except:
                                pass
            except Exception as rename_error:
                current_app.logger.warning(f"File rename failed: {rename_error}")
                # Continue with original filename
        
        # Prepare response
        response_data = {
            "success": True,
            "filename": new_filename,  # Return new filename
            "original_filename": filename if new_filename != filename else None,
            "ocr_result": result.to_dict(),
            "ocr_ready": True,
        }
        
        current_app.logger.info(
            f"OCR complete: {result.word_count} words, "
            f"{len(result.raw_results)} regions, "
            f"{result.processing_time_ms:.0f}ms"
        )
        
        # Notify frontend via Socket.IO
        try:
            from app.core import socketio
            socketio.emit("ocr_complete", {
                "filename": new_filename,
                "original_filename": filename if new_filename != filename else None,
                "success": True,
                "result": result.to_dict(),
                "derived_title": result.derived_title,
                "word_count": result.word_count,
                "confidence": result.confidence_avg,
                "has_text": result.word_count > 0,
                "renamed": new_filename != filename,
            })
        except Exception as socket_error:
            current_app.logger.warning(f"Socket.IO emit failed: {socket_error}")
        
        return jsonify(response_data)
    
    except Exception as e:
        error_msg = f"OCR error: {str(e)}"
        current_app.logger.error(error_msg)
        traceback.print_exc()
        return jsonify({"success": False, "error": error_msg}), 500


@ocr_bp.route("/<path:filename>", methods=["GET"])
def get_ocr_result(filename):
    """
    Get existing OCR result for a file.
    Returns cached result if available.
    """
    dirs = get_data_dirs()
    OCR_DATA_DIR = dirs['OCR_DATA_DIR']
    
    try:
        from app.features.document.ocr.paddle_ocr_service import PaddleOCRService
        
        # Security: prevent directory traversal
        if ".." in filename:
            return jsonify({"error": "Invalid filename"}), 400
        
        # Get OCR service
        service = PaddleOCRService(OCR_DATA_DIR)
        
        # Check if OCR result exists
        result = service.load_result(filename)
        
        if result:
            return jsonify({
                "success": True,
                "filename": filename,
                "ocr_result": result,
                "ocr_ready": True,
            })
        else:
            return jsonify({
                "success": True,
                "filename": filename,
                "ocr_result": None,
                "ocr_ready": False,
            })
    
    except Exception as e:
        error_msg = f"Error fetching OCR result: {str(e)}"
        current_app.logger.error(error_msg)
        return jsonify({"success": False, "error": error_msg}), 500


# =============================================================================
# OCR STATUS ENDPOINTS
# =============================================================================

@ocr_bp.route("/status/<path:filename>", methods=["GET"])
def get_ocr_status(filename):
    """Quick check if OCR has been run on a file."""
    dirs = get_data_dirs()
    OCR_DATA_DIR = dirs['OCR_DATA_DIR']
    
    try:
        from app.features.document.ocr.paddle_ocr_service import PaddleOCRService
        
        if ".." in filename:
            return jsonify({"error": "Invalid filename"}), 400
        
        service = PaddleOCRService(OCR_DATA_DIR)
        has_ocr = service.has_ocr_result(filename)
        
        return jsonify({
            "filename": filename,
            "ocr_ready": has_ocr,
        })
    
    except Exception as e:
        return jsonify({"filename": filename, "ocr_ready": False})


@ocr_bp.route("/batch-status", methods=["POST", "OPTIONS"])
def get_batch_ocr_status():
    """Check OCR status for multiple files at once."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    dirs = get_data_dirs()
    OCR_DATA_DIR = dirs['OCR_DATA_DIR']
    
    try:
        from app.features.document.ocr.paddle_ocr_service import PaddleOCRService
        
        data = request.get_json() or {}
        filenames = data.get("filenames", [])
        
        service = PaddleOCRService(OCR_DATA_DIR)
        
        statuses = {}
        for filename in filenames:
            if ".." not in filename:
                has_ocr = service.has_ocr_result(filename)
                status_info = {"has_ocr": has_ocr}
                
                # If OCR exists, try to get the derived title
                if has_ocr:
                    try:
                        result = service.load_result(filename)
                        if result and isinstance(result, dict) and result.get("derived_title"):
                            status_info["derived_title"] = result["derived_title"]
                    except:
                        pass
                
                statuses[filename] = status_info
        
        return jsonify({"success": True, "statuses": statuses})
    
    except Exception as e:
        return jsonify({"success": False, "statuses": {}, "error": str(e)})


# =============================================================================
# BATCH OCR PROCESSING
# =============================================================================

@ocr_bp.route("/batch", methods=["POST", "OPTIONS"])
def run_batch_ocr():
    """
    Run OCR on multiple files.
    Processes files sequentially and emits progress updates.
    """
    if request.method == "OPTIONS":
        return create_options_response()
    
    dirs = get_data_dirs()
    PROCESSED_DIR = dirs['PROCESSED_DIR']
    UPLOAD_DIR = dirs['UPLOAD_DIR']
    OCR_DATA_DIR = dirs['OCR_DATA_DIR']
    
    try:
        from app.features.document.ocr.paddle_ocr_service import PaddleOCRService
        
        data = request.get_json() or {}
        filenames = data.get("filenames", [])
        force = data.get("force", False)  # Force re-run even if exists
        
        if not filenames:
            return jsonify({"error": "No filenames provided"}), 400
        
        service = PaddleOCRService(OCR_DATA_DIR)
        
        results = {
            "success": True,
            "processed": [],
            "skipped": [],
            "errors": [],
            "total": len(filenames)
        }
        
        for i, filename in enumerate(filenames):
            try:
                # Check for existing result
                if not force and service.has_ocr_result(filename):
                    results["skipped"].append(filename)
                    continue
                
                # Find file
                image_path = os.path.join(PROCESSED_DIR, filename)
                if not os.path.exists(image_path):
                    image_path = os.path.join(UPLOAD_DIR, filename)
                
                if not os.path.exists(image_path):
                    results["errors"].append({
                        "filename": filename,
                        "error": "File not found"
                    })
                    continue
                
                # Process OCR
                result = service.process_image(image_path)
                service.save_result(filename, result)
                
                results["processed"].append({
                    "filename": filename,
                    "word_count": result.word_count,
                    "derived_title": result.derived_title
                })
                
                # Emit progress
                try:
                    from app.core import socketio
                    socketio.emit("ocr_batch_progress", {
                        "current": i + 1,
                        "total": len(filenames),
                        "filename": filename,
                        "success": True
                    })
                except:
                    pass
            
            except Exception as e:
                results["errors"].append({
                    "filename": filename,
                    "error": str(e)
                })
        
        # Emit completion
        try:
            from app.core import socketio
            socketio.emit("ocr_batch_complete", {
                "processed": len(results["processed"]),
                "skipped": len(results["skipped"]),
                "errors": len(results["errors"]),
                "total": len(filenames)
            })
        except:
            pass
        
        return jsonify(results)
    
    except Exception as e:
        current_app.logger.error(f"Batch OCR error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# OCR TEXT ENDPOINTS
# =============================================================================

@ocr_bp.route("/text/<path:filename>", methods=["GET"])
def get_ocr_text(filename):
    """Get plain text extracted by OCR."""
    dirs = get_data_dirs()
    OCR_DATA_DIR = dirs['OCR_DATA_DIR']
    
    try:
        from app.features.document.ocr.paddle_ocr_service import PaddleOCRService
        
        if ".." in filename:
            return jsonify({"error": "Invalid filename"}), 400
        
        service = PaddleOCRService(OCR_DATA_DIR)
        result = service.load_result(filename)
        
        if result:
            # Extract text from result
            text = result.get("full_text", "") if isinstance(result, dict) else ""
            return jsonify({
                "success": True,
                "filename": filename,
                "text": text,
                "word_count": len(text.split()) if text else 0
            })
        else:
            return jsonify({
                "success": False,
                "filename": filename,
                "text": "",
                "error": "No OCR result found"
            })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
