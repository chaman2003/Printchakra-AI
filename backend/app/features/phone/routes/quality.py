"""
PrintChakra Backend - Phone Quality Routes

Image quality and enhancement endpoints.
"""

import os
from app.core.paths import app_data_dir
import logging
from flask import jsonify, request, send_file
from app.features.phone.routes import phone_bp
from app.core.middleware.cors import create_options_response

logger = logging.getLogger(__name__)

UPLOAD_FOLDER = os.path.join(app_data_dir(), "uploads")


@phone_bp.route("/quality/check", methods=["POST", "OPTIONS"])
def check_quality():
    """Check image quality metrics."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400
        
        file = request.files["file"]
        
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        # Read image data
        image_data = file.read()
        
        # Try to analyze quality
        try:
            from app.features.phone.quality.validator import ImageQualityValidator
            
            validator = ImageQualityValidator()
            quality_result = validator.analyze(image_data)
            
            return jsonify({
                "success": True,
                "quality": quality_result
            })
        
        except ImportError:
            # Basic quality check without advanced validator
            return jsonify({
                "success": True,
                "quality": {
                    "size_bytes": len(image_data),
                    "quality_score": 0.8,  # Default score
                    "message": "Basic quality check (advanced validator not available)"
                }
            })
    
    except Exception as e:
        logger.error(f"Quality check error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@phone_bp.route("/quality/enhance", methods=["POST", "OPTIONS"])
def enhance_image():
    """Enhance image quality."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400
        
        file = request.files["file"]
        
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        # Get enhancement options
        brightness = request.form.get("brightness", 1.0, type=float)
        contrast = request.form.get("contrast", 1.0, type=float)
        sharpness = request.form.get("sharpness", 1.0, type=float)
        denoise = request.form.get("denoise", "false").lower() == "true"
        
        image_data = file.read()
        
        # Apply enhancements
        try:
            from PIL import Image, ImageEnhance
            import io
            
            img = Image.open(io.BytesIO(image_data))
            
            # Apply brightness
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(brightness)
            
            # Apply contrast
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(contrast)
            
            # Apply sharpness
            if sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(sharpness)
            
            # Save enhanced image
            output = io.BytesIO()
            img.save(output, format="JPEG", quality=95)
            enhanced_data = output.getvalue()
            
            # Save to file
            import uuid
            from werkzeug.utils import secure_filename
            
            original_name = secure_filename(file.filename)
            enhanced_name = f"enhanced_{uuid.uuid4().hex[:8]}_{original_name}"
            
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            filepath = os.path.join(UPLOAD_FOLDER, enhanced_name)
            
            with open(filepath, "wb") as f:
                f.write(enhanced_data)
            
            logger.info(f"[OK] Image enhanced: {enhanced_name}")
            
            return jsonify({
                "success": True,
                "filename": enhanced_name,
                "path": filepath,
                "size": len(enhanced_data),
                "enhancements": {
                    "brightness": brightness,
                    "contrast": contrast,
                    "sharpness": sharpness,
                    "denoise": denoise
                }
            })
        
        except ImportError:
            return jsonify({
                "success": False,
                "error": "PIL/Pillow not available for image enhancement"
            }), 503
    
    except Exception as e:
        logger.error(f"Enhancement error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@phone_bp.route("/quality/resize", methods=["POST", "OPTIONS"])
def resize_image():
    """Resize image to specified dimensions."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400
        
        file = request.files["file"]
        
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        # Get resize options
        width = request.form.get("width", type=int)
        height = request.form.get("height", type=int)
        max_size = request.form.get("max_size", type=int)
        maintain_aspect = request.form.get("maintain_aspect", "true").lower() == "true"
        
        if not width and not height and not max_size:
            return jsonify({
                "success": False,
                "error": "Provide width, height, or max_size"
            }), 400
        
        image_data = file.read()
        
        try:
            from PIL import Image
            import io
            
            img = Image.open(io.BytesIO(image_data))
            original_size = img.size
            
            if max_size:
                # Resize to fit within max_size while maintaining aspect ratio
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            elif maintain_aspect:
                # Calculate new dimensions maintaining aspect ratio
                orig_w, orig_h = img.size
                if width and not height:
                    height = int(orig_h * width / orig_w)
                elif height and not width:
                    width = int(orig_w * height / orig_h)
                img = img.resize((width, height), Image.Resampling.LANCZOS)
            else:
                # Force exact dimensions
                if width and height:
                    img = img.resize((width, height), Image.Resampling.LANCZOS)
            
            # Save resized image
            output = io.BytesIO()
            img.save(output, format="JPEG", quality=95)
            resized_data = output.getvalue()
            
            # Save to file
            import uuid
            from werkzeug.utils import secure_filename
            
            original_name = secure_filename(file.filename)
            resized_name = f"resized_{uuid.uuid4().hex[:8]}_{original_name}"
            
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            filepath = os.path.join(UPLOAD_FOLDER, resized_name)
            
            with open(filepath, "wb") as f:
                f.write(resized_data)
            
            logger.info(f"[OK] Image resized: {resized_name}")
            
            return jsonify({
                "success": True,
                "filename": resized_name,
                "path": filepath,
                "size": len(resized_data),
                "original_dimensions": original_size,
                "new_dimensions": img.size
            })
        
        except ImportError:
            return jsonify({
                "success": False,
                "error": "PIL/Pillow not available for image resizing"
            }), 503
    
    except Exception as e:
        logger.error(f"Resize error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
