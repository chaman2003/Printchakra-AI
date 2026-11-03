"""
PrintChakra Backend - OCR Batch Routes

Batch OCR processing endpoints.
"""

import os
from app.core.paths import app_data_dir
import logging
from flask import jsonify, request
from werkzeug.utils import secure_filename
from app.features.document.ocr.routes import ocr_bp
from app.core.middleware.cors import create_options_response

logger = logging.getLogger(__name__)

DATA_DIR = app_data_dir()

# Store for batch job status
batch_jobs = {}


@ocr_bp.route("/batch", methods=["POST", "OPTIONS"])
def start_batch_ocr():
    """Start batch OCR processing for multiple documents."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        data = request.get_json()
        
        if not data or "documents" not in data:
            return jsonify({"success": False, "error": "No documents provided"}), 400
        
        documents = data.get("documents", [])
        language = data.get("language", "en")
        use_ollama = data.get("use_ollama", True)
        
        if not documents:
            return jsonify({"success": False, "error": "Empty document list"}), 400
        
        # Generate batch job ID
        import uuid
        from datetime import datetime
        
        batch_id = str(uuid.uuid4())
        
        # Create batch job
        batch_job = {
            "id": batch_id,
            "status": "queued",
            "total": len(documents),
            "completed": 0,
            "failed": 0,
            "results": [],
            "created_at": datetime.now().isoformat(),
            "language": language,
            "use_ollama": use_ollama
        }
        
        batch_jobs[batch_id] = batch_job
        
        # Start processing in background (in production, use Celery or similar)
        import threading
        
        def process_batch():
            process_batch_ocr(batch_id, documents, language, use_ollama)
        
        thread = threading.Thread(target=process_batch)
        thread.start()
        
        logger.info(f"[OK] Batch OCR started: {batch_id} ({len(documents)} documents)")
        
        return jsonify({
            "success": True,
            "batch_id": batch_id,
            "total_documents": len(documents),
            "status": "queued"
        })
    
    except Exception as e:
        logger.error(f"Batch OCR error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@ocr_bp.route("/batch/<batch_id>", methods=["GET", "OPTIONS"])
def get_batch_status(batch_id):
    """Get batch OCR job status."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        if batch_id not in batch_jobs:
            return jsonify({"success": False, "error": "Batch job not found"}), 404
        
        job = batch_jobs[batch_id]
        
        return jsonify({
            "success": True,
            "batch": job
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@ocr_bp.route("/batch/<batch_id>/cancel", methods=["POST", "OPTIONS"])
def cancel_batch_ocr(batch_id):
    """Cancel a batch OCR job."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        if batch_id not in batch_jobs:
            return jsonify({"success": False, "error": "Batch job not found"}), 404
        
        batch_jobs[batch_id]["status"] = "cancelled"
        
        logger.info(f"[OK] Batch OCR cancelled: {batch_id}")
        
        return jsonify({
            "success": True,
            "message": "Batch job cancelled"
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@ocr_bp.route("/batch/<batch_id>/results", methods=["GET", "OPTIONS"])
def get_batch_results(batch_id):
    """Get batch OCR results."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        if batch_id not in batch_jobs:
            return jsonify({"success": False, "error": "Batch job not found"}), 404
        
        job = batch_jobs[batch_id]
        
        return jsonify({
            "success": True,
            "batch_id": batch_id,
            "status": job["status"],
            "results": job["results"],
            "completed": job["completed"],
            "failed": job["failed"],
            "total": job["total"]
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@ocr_bp.route("/batch/list", methods=["GET", "OPTIONS"])
def list_batch_jobs():
    """List all batch OCR jobs."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        jobs = []
        for batch_id, job in batch_jobs.items():
            jobs.append({
                "id": batch_id,
                "status": job["status"],
                "total": job["total"],
                "completed": job["completed"],
                "failed": job["failed"],
                "created_at": job["created_at"]
            })
        
        # Sort by created_at, newest first
        jobs.sort(key=lambda x: x["created_at"], reverse=True)
        
        return jsonify({
            "success": True,
            "jobs": jobs,
            "total": len(jobs)
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def process_batch_ocr(batch_id: str, documents: list, language: str, use_ollama: bool):
    """Process batch OCR in background."""
    try:
        from app.features.document.ocr.routes.extraction import perform_ocr
        
        job = batch_jobs[batch_id]
        job["status"] = "processing"
        
        for doc_id in documents:
            # Check if cancelled
            if job["status"] == "cancelled":
                break
            
            filename = secure_filename(doc_id)
            
            # Find document
            document_path = None
            for folder in ["uploads", "pdfs", "processed"]:
                filepath = os.path.join(DATA_DIR, folder, filename)
                if os.path.exists(filepath):
                    document_path = filepath
                    break
            
            if not document_path:
                job["results"].append({
                    "document": doc_id,
                    "success": False,
                    "error": "Document not found"
                })
                job["failed"] += 1
                continue
            
            # Perform OCR
            result = perform_ocr(document_path, language=language, use_ollama=use_ollama)
            
            if result.get("success"):
                job["results"].append({
                    "document": doc_id,
                    "success": True,
                    "text": result.get("text", ""),
                    "character_count": result.get("character_count", 0)
                })
                job["completed"] += 1
            else:
                job["results"].append({
                    "document": doc_id,
                    "success": False,
                    "error": result.get("error", "Unknown error")
                })
                job["failed"] += 1
        
        # Update final status
        if job["status"] != "cancelled":
            if job["failed"] == job["total"]:
                job["status"] = "failed"
            elif job["failed"] > 0:
                job["status"] = "completed_with_errors"
            else:
                job["status"] = "completed"
        
        logger.info(f"[OK] Batch OCR completed: {batch_id} ({job['completed']}/{job['total']} successful)")
    
    except Exception as e:
        logger.error(f"Batch OCR processing error: {e}")
        if batch_id in batch_jobs:
            batch_jobs[batch_id]["status"] = "error"
            batch_jobs[batch_id]["error"] = str(e)
