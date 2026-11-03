"""
PrintChakra Backend - Document Conversion Routes

Document format conversion endpoints.
"""

import os
from app.core.paths import app_data_dir
import logging
from flask import jsonify, request
from werkzeug.utils import secure_filename
from app.core.config import get_data_dirs
from app.features.document.routes import document_bp
from app.core.middleware.cors import create_options_response

logger = logging.getLogger(__name__)

DATA_DIR = app_data_dir()
CONVERTED_DIR = os.path.join(DATA_DIR, "converted")


@document_bp.route("/convert", methods=["POST", "OPTIONS"])
def convert_document():
    """Convert document to different format."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        doc_id = data.get("document_id")
        target_format = data.get("format", "pdf").lower()
        
        if not doc_id:
            return jsonify({"success": False, "error": "No document ID provided"}), 400
        
        filename = secure_filename(doc_id)
        
        # Find source document
        source_path = None
        for folder in ["uploads", "pdfs", "processed"]:
            filepath = os.path.join(DATA_DIR, folder, filename)
            if os.path.exists(filepath):
                source_path = filepath
                break
        
        if not source_path:
            return jsonify({"success": False, "error": "Document not found"}), 404
        
        # Get source extension
        source_ext = os.path.splitext(filename)[1].lower()
        
        # Generate output filename
        import uuid
        output_name = f"{os.path.splitext(filename)[0]}_{uuid.uuid4().hex[:8]}.{target_format}"
        
        os.makedirs(CONVERTED_DIR, exist_ok=True)
        output_path = os.path.join(CONVERTED_DIR, output_name)
        
        # Perform conversion
        try:
            if source_ext in [".png", ".jpg", ".jpeg", ".gif"] and target_format == "pdf":
                convert_image_to_pdf(source_path, output_path)
            
            elif source_ext == ".pdf" and target_format in ["png", "jpg", "jpeg"]:
                output_paths = convert_pdf_to_images(source_path, output_path, target_format)
                return jsonify({
                    "success": True,
                    "converted": output_paths,
                    "format": target_format
                })
            
            elif source_ext == ".pdf" and target_format == "txt":
                convert_pdf_to_text(source_path, output_path)
            
            else:
                return jsonify({
                    "success": False,
                    "error": f"Conversion from {source_ext} to {target_format} not supported"
                }), 400
            
            logger.info(f"[OK] Document converted: {filename} -> {output_name}")
            
            return jsonify({
                "success": True,
                "converted": output_name,
                "path": output_path,
                "format": target_format
            })
        
        except Exception as e:
            logger.error(f"Conversion error: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@document_bp.route("/convert/pdf-to-images/<doc_id>", methods=["GET", "OPTIONS"])
def convert_pdf_to_images_route(doc_id):
    """Convert PDF to images."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        filename = secure_filename(doc_id)
        format_type = request.args.get("format", "png").lower()
        dpi = request.args.get("dpi", 150, type=int)
        
        # Find PDF
        source_path = None
        for folder in ["uploads", "pdfs", "processed"]:
            filepath = os.path.join(DATA_DIR, folder, filename)
            if os.path.exists(filepath):
                source_path = filepath
                break
        
        if not source_path:
            return jsonify({"success": False, "error": "Document not found"}), 404
        
        # Convert
        import uuid
        output_base = os.path.join(CONVERTED_DIR, f"{os.path.splitext(filename)[0]}_{uuid.uuid4().hex[:8]}")
        
        os.makedirs(CONVERTED_DIR, exist_ok=True)
        
        output_paths = convert_pdf_to_images(source_path, output_base, format_type, dpi)
        
        return jsonify({
            "success": True,
            "images": output_paths,
            "total": len(output_paths)
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@document_bp.route("/convert/images-to-pdf", methods=["POST", "OPTIONS"])
def convert_images_to_pdf_route():
    """Combine multiple images into a single PDF."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        data = request.get_json()
        
        if not data or "images" not in data:
            return jsonify({"success": False, "error": "No images provided"}), 400
        
        images = data.get("images", [])
        output_name = data.get("output_name", "combined.pdf")
        
        if not images:
            return jsonify({"success": False, "error": "Empty image list"}), 400
        
        # Find image files
        image_paths = []
        for img_id in images:
            filename = secure_filename(img_id)
            for folder in ["uploads", "processed", "converted"]:
                filepath = os.path.join(DATA_DIR, folder, filename)
                if os.path.exists(filepath):
                    image_paths.append(filepath)
                    break
        
        if not image_paths:
            return jsonify({"success": False, "error": "No valid images found"}), 404
        
        # Combine into PDF
        import uuid
        output_name = secure_filename(output_name)
        output_filename = f"{os.path.splitext(output_name)[0]}_{uuid.uuid4().hex[:8]}.pdf"
        
        os.makedirs(CONVERTED_DIR, exist_ok=True)
        output_path = os.path.join(CONVERTED_DIR, output_filename)
        
        combine_images_to_pdf(image_paths, output_path)
        
        logger.info(f"[OK] Combined {len(image_paths)} images to PDF: {output_filename}")
        
        return jsonify({
            "success": True,
            "pdf": output_filename,
            "path": output_path,
            "images_combined": len(image_paths)
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def convert_image_to_pdf(image_path, pdf_path):
    """Convert a single image to PDF."""
    try:
        from PIL import Image
        
        img = Image.open(image_path)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        
        img.save(pdf_path, "PDF", resolution=150)
    
    except ImportError:
        raise Exception("PIL/Pillow not available")


def convert_pdf_to_images(pdf_path, output_base, format_type="png", dpi=150):
    """Convert PDF pages to images."""
    try:
        import fitz  # PyMuPDF
        from PIL import Image
        import io
        
        doc = fitz.open(pdf_path)
        output_paths = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Set zoom based on DPI (72 is default PDF DPI)
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            
            pix = page.get_pixmap(matrix=mat)
            
            # Save as image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            output_path = f"{output_base}_page{page_num + 1}.{format_type}"
            
            if format_type.lower() in ["jpg", "jpeg"]:
                img.save(output_path, "JPEG", quality=95)
            else:
                img.save(output_path, "PNG")
            
            output_paths.append(os.path.basename(output_path))
        
        doc.close()
        return output_paths
    
    except ImportError:
        raise Exception("PyMuPDF or PIL not available")


def convert_pdf_to_text(pdf_path, text_path):
    """Extract text from PDF."""
    try:
        import fitz
        
        doc = fitz.open(pdf_path)
        text = ""
        
        for page in doc:
            text += page.get_text()
        
        doc.close()
        
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)
    
    except ImportError:
        raise Exception("PyMuPDF not available")


def combine_images_to_pdf(image_paths, pdf_path):
    """Combine multiple images into a single PDF."""
    try:
        from PIL import Image
        
        images = []
        for path in image_paths:
            img = Image.open(path)
            if img.mode == "RGBA":
                img = img.convert("RGB")
            images.append(img)
        
        if images:
            images[0].save(
                pdf_path,
                "PDF",
                resolution=150,
                save_all=True,
                append_images=images[1:] if len(images) > 1 else []
            )
    
    except ImportError:
        raise Exception("PIL/Pillow not available")


@document_bp.route("/converted", methods=["GET", "OPTIONS"])
def list_converted_files():
    """List converted output files."""
    if request.method == "OPTIONS":
        return create_options_response()

    try:
        dirs = get_data_dirs()
        converted_dir = dirs["CONVERTED_DIR"]
        files = []
        if os.path.isdir(converted_dir):
            for filename in os.listdir(converted_dir):
                full_path = os.path.join(converted_dir, filename)
                if os.path.isfile(full_path):
                    files.append(
                        {
                            "filename": filename,
                            "size": os.path.getsize(full_path),
                            "modified": os.path.getmtime(full_path),
                        }
                    )
        files.sort(key=lambda x: x["modified"], reverse=True)
        return jsonify({"success": True, "files": files})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "files": []}), 500


@document_bp.route("/converted/<path:filename>", methods=["DELETE", "OPTIONS"])
def delete_converted_file(filename):
    """Delete a converted output file."""
    if request.method == "OPTIONS":
        return create_options_response()

    try:
        safe_name = secure_filename(filename)
        if not safe_name:
            return jsonify({"success": False, "error": "Invalid filename"}), 400
        converted_path = os.path.join(get_data_dirs()["CONVERTED_DIR"], safe_name)
        if not os.path.isfile(converted_path):
            return jsonify({"success": False, "error": "File not found"}), 404
        os.remove(converted_path)
        return jsonify({"success": True, "message": "Converted file deleted", "filename": safe_name})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
