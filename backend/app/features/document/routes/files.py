"""
PrintChakra Backend - Document File Routes

Document file management endpoints.
"""

import os
import logging
import json
import time
from flask import jsonify, request, send_file
from flask import send_from_directory
from werkzeug.utils import secure_filename
from app.features.document.routes import document_bp
from app.core.config import get_data_dirs
from app.core.middleware.cors import create_options_response

logger = logging.getLogger(__name__)
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "gif", "doc", "docx", "txt", "rtf"}
DEBUG_LOG_PATH = "C:/Users/chama/OneDrive/Desktop/printchakra-new/debug-68db5a.log"


def allowed_file(filename):
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def delete_ocr_artifacts(filename):
    """Delete OCR sidecar files that belong to a document."""
    dirs = get_data_dirs()
    stem = os.path.splitext(filename)[0]
    candidate_paths = [
        os.path.join(dirs["TEXT_DIR"], f"{stem}.txt"),
        os.path.join(dirs["TEXT_DIR"], f"processed_{stem}.txt"),
        os.path.join(dirs["OCR_DATA_DIR"], f"{stem}_ocr.json"),
    ]

    deleted = []
    for path in candidate_paths:
        if os.path.isfile(path):
            os.remove(path)
            deleted.append(path)
    return deleted


def _debug_log(run_id, hypothesis_id, location, message, data):
    # region agent log
    try:
        payload = {
            "sessionId": "68db5a",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass
    # endregion


def _find_document_file(filename: str, folder: str | None = None):
    """Find a document in root directories or processed subfolders."""
    dirs = get_data_dirs()
    processed_dir = dirs["PROCESSED_DIR"]
    candidate_paths = [
        os.path.join(processed_dir, filename),
        os.path.join(dirs["UPLOAD_DIR"], filename),
        os.path.join(dirs["PDF_DIR"], filename),
        os.path.join(dirs["CONVERTED_DIR"], filename),
    ]

    # Explicit folder hint (used by dashboard folder view).
    if folder:
        safe_folder = secure_filename(folder)
        if safe_folder:
            candidate_paths.insert(0, os.path.join(processed_dir, safe_folder, filename))

    for path in candidate_paths:
        if os.path.isfile(path):
            return path

    # Fallback: search one-level processed subfolders.
    if os.path.isdir(processed_dir):
        for entry in os.listdir(processed_dir):
            subdir = os.path.join(processed_dir, entry)
            if not os.path.isdir(subdir):
                continue
            nested = os.path.join(subdir, filename)
            if os.path.isfile(nested):
                return nested

    return None


def _find_all_document_files(filename: str, folder: str | None = None) -> list[str]:
    """Find all copies of a document across known data directories."""
    dirs = get_data_dirs()
    processed_dir = dirs["PROCESSED_DIR"]

    candidate_paths = [
        os.path.join(processed_dir, filename),
        os.path.join(dirs["UPLOAD_DIR"], filename),
        os.path.join(dirs["PDF_DIR"], filename),
        os.path.join(dirs["CONVERTED_DIR"], filename),
    ]

    if folder:
        safe_folder = secure_filename(folder)
        if safe_folder:
            candidate_paths.insert(0, os.path.join(processed_dir, safe_folder, filename))

    existing: list[str] = []
    seen: set[str] = set()
    for path in candidate_paths:
        if path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            existing.append(path)

    # Include one-level nested files in processed subfolders.
    if os.path.isdir(processed_dir):
        for entry in os.listdir(processed_dir):
            subdir = os.path.join(processed_dir, entry)
            if not os.path.isdir(subdir):
                continue
            nested = os.path.join(subdir, filename)
            if nested not in seen and os.path.isfile(nested):
                existing.append(nested)
                seen.add(nested)

    return existing


@document_bp.route("/processed/<path:filename>", methods=["GET", "OPTIONS"])
def serve_processed_file(filename):
    """Serve a processed file (image/PDF) from the processed directory."""
    if request.method == "OPTIONS":
        return create_options_response()

    # Security: reject path traversal
    if ".." in filename or filename.startswith("/"):
        return jsonify({"error": "Invalid filename"}), 400

    dirs = get_data_dirs()
    processed_dir = dirs["PROCESSED_DIR"]
    folder = request.args.get("folder", "").strip() or None

    # Prefer explicit folder when provided, then root, then one-level subfolders.
    candidate_paths = []
    if folder:
        safe_folder = secure_filename(folder)
        if safe_folder:
            candidate_paths.append(os.path.join(processed_dir, safe_folder, filename))
    candidate_paths.append(os.path.join(processed_dir, filename))

    selected_path = next((path for path in candidate_paths if os.path.isfile(path)), None)
    if not selected_path and os.path.isdir(processed_dir):
        for entry in os.listdir(processed_dir):
            subdir = os.path.join(processed_dir, entry)
            if not os.path.isdir(subdir):
                continue
            nested = os.path.join(subdir, filename)
            if os.path.isfile(nested):
                selected_path = nested
                break

    if not selected_path:
        _debug_log(
            "run1",
            "H1",
            "backend/app/features/document/routes/files.py:serve_processed_file:not_found",
            "processed_file_missing",
            {"filename": filename, "folder": folder, "searched_paths": candidate_paths},
        )
        logger.warning(f"Processed file not found: {filename} (folder={folder})")
        return jsonify({"error": "File not found"}), 404

    selected_dir = os.path.dirname(selected_path)
    selected_name = os.path.basename(selected_path)
    _debug_log(
        "run1",
        "H1",
        "backend/app/features/document/routes/files.py:serve_processed_file:served",
        "processed_file_served",
        {
            "filename": filename,
            "folder": folder,
            "file_path": selected_path,
            "file_size": os.path.getsize(selected_path),
        },
    )
    response = send_from_directory(selected_dir, selected_name)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Cache-Control"] = "private, max-age=3600"
    return response


@document_bp.route("/files", methods=["GET", "OPTIONS"])
def list_documents():
    """List all documents."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        documents = []
        dirs = get_data_dirs()
        
        # Scan uploads folder
        uploads_folder = dirs["UPLOAD_DIR"]
        if os.path.exists(uploads_folder):
            for filename in os.listdir(uploads_folder):
                filepath = os.path.join(uploads_folder, filename)
                if os.path.isfile(filepath):
                    ext = os.path.splitext(filename)[1].lower()
                    documents.append({
                        "id": filename,
                        "name": filename,
                        "path": filepath,
                        "size": os.path.getsize(filepath),
                        "type": ext[1:] if ext else "unknown",
                        "modified": os.path.getmtime(filepath),
                        "folder": "uploads"
                    })
        
        # Sort by modified time, newest first
        documents.sort(key=lambda x: x["modified"], reverse=True)
        
        return jsonify({
            "success": True,
            "documents": documents,
            "total": len(documents)
        })
    
    except Exception as e:
        logger.error(f"List documents error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@document_bp.route("/files/<doc_id>", methods=["GET", "OPTIONS"])
def get_document(doc_id):
    """Get document details."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        filename = secure_filename(doc_id)
        dirs = get_data_dirs()
        
        # Search for document
        folder_paths = {
            "uploads": dirs["UPLOAD_DIR"],
            "pdfs": dirs["PDF_DIR"],
            "processed": dirs["PROCESSED_DIR"],
        }
        for folder, base_path in folder_paths.items():
            filepath = os.path.join(base_path, filename)
            if os.path.exists(filepath):
                ext = os.path.splitext(filename)[1].lower()
                return jsonify({
                    "success": True,
                    "document": {
                        "id": filename,
                        "name": filename,
                        "path": filepath,
                        "size": os.path.getsize(filepath),
                        "type": ext[1:] if ext else "unknown",
                        "modified": os.path.getmtime(filepath),
                        "folder": folder
                    }
                })
        
        return jsonify({"success": False, "error": "Document not found"}), 404
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@document_bp.route("/info/<path:doc_id>", methods=["GET", "OPTIONS"])
def get_document_info(doc_id):
    """Get lightweight document info for preview/orchestration UIs."""
    if request.method == "OPTIONS":
        return create_options_response()

    try:
        filename = secure_filename(doc_id)
        if not filename:
            return jsonify({"success": False, "error": "Invalid filename"}), 400

        dirs = get_data_dirs()
        candidate_paths = [
            os.path.join(dirs["PROCESSED_DIR"], filename),
            os.path.join(dirs["UPLOAD_DIR"], filename),
            os.path.join(dirs["PDF_DIR"], filename),
            os.path.join(dirs["CONVERTED_DIR"], filename),
        ]
        file_path = next((path for path in candidate_paths if os.path.isfile(path)), None)
        if not file_path:
            return jsonify({"success": False, "error": "Document not found"}), 404

        ext = os.path.splitext(filename)[1].lower()
        return jsonify(
            {
                "success": True,
                "filename": filename,
                "file_type": ext[1:] if ext else "unknown",
                "size": os.path.getsize(file_path),
                "pages": [{"pageNumber": 1, "thumbnailUrl": f"/document/thumbnails/{filename}"}],
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@document_bp.route("/files/<doc_id>/download", methods=["GET", "OPTIONS"])
def download_document(doc_id):
    """Download a document."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        filename = secure_filename(doc_id)
        dirs = get_data_dirs()
        
        # Search for document
        folder_paths = {
            "uploads": dirs["UPLOAD_DIR"],
            "pdfs": dirs["PDF_DIR"],
            "processed": dirs["PROCESSED_DIR"],
            "converted": dirs["CONVERTED_DIR"],
        }
        for _, base_path in folder_paths.items():
            filepath = os.path.join(base_path, filename)
            if os.path.exists(filepath):
                return send_file(
                    filepath,
                    as_attachment=True,
                    download_name=filename
                )
        
        return jsonify({"success": False, "error": "Document not found"}), 404
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@document_bp.route("/files/<doc_id>", methods=["DELETE", "OPTIONS"])
def delete_document(doc_id):
    """Delete a document."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        filename = secure_filename(doc_id)

        folder = request.args.get("folder", "").strip() or None
        matching_paths = _find_all_document_files(filename, folder=folder)
        if matching_paths:
            for path in matching_paths:
                os.remove(path)
            deleted_ocr_artifacts = delete_ocr_artifacts(filename)
            logger.info(f"[OK] Document deleted: {filename} ({len(matching_paths)} file(s))")
            return jsonify(
                {
                    "success": True,
                    "message": "Document deleted",
                    "deleted_files": len(matching_paths),
                    "deleted_paths": matching_paths,
                    "deleted_ocr_artifacts": len(deleted_ocr_artifacts),
                }
            )
        
        return jsonify({"success": False, "error": "Document not found"}), 404
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@document_bp.route("/files/<doc_id>", methods=["PUT", "OPTIONS"])
def rename_document(doc_id):
    """Rename a document file, preserving extension when omitted."""
    if request.method == "OPTIONS":
        return create_options_response()

    try:
        filename = secure_filename(doc_id)
        if not filename:
            return jsonify({"success": False, "error": "Invalid filename"}), 400

        data = request.get_json(silent=True) or {}
        requested_name = (data.get("name") or "").strip()
        folder = (data.get("folder") or request.args.get("folder") or "").strip() or None

        if not requested_name:
            return jsonify({"success": False, "error": "New name is required"}), 400

        safe_requested = secure_filename(requested_name)
        if not safe_requested:
            return jsonify({"success": False, "error": "Invalid new filename"}), 400

        old_path = _find_document_file(filename, folder=folder)
        if not old_path:
            return jsonify({"success": False, "error": "Document not found"}), 404

        _, old_ext = os.path.splitext(filename)
        _, new_ext = os.path.splitext(safe_requested)
        if not new_ext:
            safe_requested = f"{safe_requested}{old_ext}"
        elif old_ext and new_ext.lower() != old_ext.lower():
            return jsonify({"success": False, "error": "File extension cannot be changed"}), 400

        if safe_requested == filename:
            return jsonify({"success": True, "filename": filename, "message": "No changes made"})

        new_path = os.path.join(os.path.dirname(old_path), safe_requested)
        if os.path.exists(new_path):
            return jsonify({"success": False, "error": "A file with that name already exists"}), 409

        os.rename(old_path, new_path)
        logger.info(f"[OK] Document renamed: {filename} -> {safe_requested}")

        return jsonify(
            {
                "success": True,
                "old_filename": filename,
                "filename": safe_requested,
                "folder": folder,
                "message": "Document renamed",
            }
        )
    except Exception as e:
        logger.error(f"Rename document error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@document_bp.route("/upload", methods=["POST", "OPTIONS"])
def upload_document():
    """Upload a document."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400
        
        file = request.files["file"]
        
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        filename = secure_filename(file.filename)
        
        # Generate unique filename
        import uuid
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        
        uploads_folder = get_data_dirs()["UPLOAD_DIR"]
        os.makedirs(uploads_folder, exist_ok=True)
        
        filepath = os.path.join(uploads_folder, unique_filename)
        file.save(filepath)
        
        file_size = os.path.getsize(filepath)
        
        logger.info(f"[OK] Document uploaded: {unique_filename}")
        
        return jsonify({
            "success": True,
            "document": {
                "id": unique_filename,
                "name": filename,
                "path": filepath,
                "size": file_size
            }
        })
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
