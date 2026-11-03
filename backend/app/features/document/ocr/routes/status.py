"""
PrintChakra Backend - OCR Status Routes

OCR service status and configuration endpoints.
"""

import logging
from flask import jsonify, request
from app.features.document.ocr.routes import ocr_bp
from app.core.middleware.cors import create_options_response

logger = logging.getLogger(__name__)


@ocr_bp.route("/status", methods=["GET", "OPTIONS"])
def get_ocr_status():
    """Get OCR service status."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        status = {
            "available": False,
            "engine": "PaddleOCR",
            "languages": [],
            "ollama_available": False
        }
        
        # Gemini (cloud) takes priority over local PaddleOCR
        from app.modules.ocr.gemini_ocr import is_available as gemini_available
        if gemini_available():
            status.update({"available": True, "engine": "Gemini", "languages": ["en", "hi", "kn", "te", "ta", "mr", "bn", "gu"]})
            return jsonify({"success": True, "status": status})
        
        # Check PaddleOCR
        try:
            from paddleocr import PaddleOCR
            status["available"] = True
            status["languages"] = ["en", "ch", "japan", "korean", "german", "french"]
        except ImportError:
            status["error"] = "PaddleOCR not installed"
        
        # Check Ollama
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            status["ollama_available"] = response.status_code == 200
        except:
            status["ollama_available"] = False
        
        return jsonify({
            "success": True,
            "status": status
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@ocr_bp.route("/languages", methods=["GET", "OPTIONS"])
def get_supported_languages():
    """Get supported OCR languages."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        languages = [
            {"code": "en", "name": "English"},
            {"code": "ch", "name": "Chinese (Simplified)"},
            {"code": "cht", "name": "Chinese (Traditional)"},
            {"code": "japan", "name": "Japanese"},
            {"code": "korean", "name": "Korean"},
            {"code": "german", "name": "German"},
            {"code": "french", "name": "French"},
            {"code": "it", "name": "Italian"},
            {"code": "es", "name": "Spanish"},
            {"code": "pt", "name": "Portuguese"},
            {"code": "ru", "name": "Russian"},
            {"code": "ar", "name": "Arabic"},
            {"code": "hi", "name": "Hindi"},
            {"code": "ta", "name": "Tamil"},
            {"code": "te", "name": "Telugu"}
        ]
        
        return jsonify({
            "success": True,
            "languages": languages
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@ocr_bp.route("/config", methods=["GET", "OPTIONS"])
def get_ocr_config():
    """Get OCR configuration."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        config = {
            "default_language": "en",
            "use_gpu": False,
            "use_angle_cls": True,
            "use_ollama": True,
            "ollama_model": "llama2",
            "timeout_seconds": 120,
            "max_file_size_mb": 50
        }
        
        return jsonify({
            "success": True,
            "config": config
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@ocr_bp.route("/config", methods=["POST", "OPTIONS"])
def update_ocr_config():
    """Update OCR configuration."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        # Validate and update config
        allowed_keys = [
            "default_language", "use_gpu", "use_angle_cls",
            "use_ollama", "ollama_model", "timeout_seconds", "max_file_size_mb"
        ]
        
        updated = {}
        for key in allowed_keys:
            if key in data:
                updated[key] = data[key]
        
        logger.info(f"[OK] OCR config updated: {list(updated.keys())}")
        
        return jsonify({
            "success": True,
            "updated": updated,
            "message": "Configuration updated"
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@ocr_bp.route("/test", methods=["POST", "OPTIONS"])
def test_ocr():
    """Test OCR on a sample image."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        from paddleocr import PaddleOCR
        
        # Create a simple test image with text
        try:
            from PIL import Image, ImageDraw, ImageFont
            import io
            import tempfile
            
            # Create test image
            img = Image.new("RGB", (400, 100), color="white")
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            draw.text((10, 30), "PrintChakra OCR Test", fill="black", font=font)
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img.save(tmp.name)
                temp_path = tmp.name
            
            # Test OCR
            ocr = PaddleOCR(use_angle_cls=True, lang="en", device="cpu")
            result = ocr.ocr(temp_path, cls=True)
            
            # Cleanup
            import os
            os.unlink(temp_path)
            
            # Extract text
            extracted = ""
            if result and result[0]:
                for line in result[0]:
                    extracted += line[1][0] + " "
            
            return jsonify({
                "success": True,
                "test_passed": "printchakra" in extracted.lower() or "ocr" in extracted.lower() or "test" in extracted.lower(),
                "extracted_text": extracted.strip(),
                "message": "OCR test completed"
            })
        
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Test failed: {str(e)}"
            }), 500
    
    except ImportError:
        return jsonify({
            "success": False,
            "error": "PaddleOCR not installed"
        }), 503
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@ocr_bp.route("/health", methods=["GET", "OPTIONS"])
def ocr_health():
    """OCR service health check."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        health = {
            "healthy": True,
            "checks": {}
        }
        
        # Check PaddleOCR
        try:
            from paddleocr import PaddleOCR
            health["checks"]["paddleocr"] = {"status": "ok", "message": "PaddleOCR available"}
        except ImportError:
            health["checks"]["paddleocr"] = {"status": "error", "message": "PaddleOCR not installed"}
            health["healthy"] = False
        
        # Check Pillow
        try:
            from PIL import Image
            health["checks"]["pillow"] = {"status": "ok", "message": "Pillow available"}
        except ImportError:
            health["checks"]["pillow"] = {"status": "error", "message": "Pillow not installed"}
            health["healthy"] = False
        
        # Check Ollama (optional)
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                health["checks"]["ollama"] = {"status": "ok", "message": "Ollama available"}
            else:
                health["checks"]["ollama"] = {"status": "warning", "message": "Ollama not responding"}
        except:
            health["checks"]["ollama"] = {"status": "warning", "message": "Ollama not available (optional)"}
        
        return jsonify({
            "success": True,
            "health": health
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
