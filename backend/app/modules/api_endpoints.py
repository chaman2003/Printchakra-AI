"""
Enhanced Processing API Endpoints - for integration with notebook pipeline

"""

import logging
import os
import threading
import uuid
from datetime import datetime

from flask import jsonify, request

logger = logging.getLogger(__name__)


def create_enhanced_endpoints(
    app,
    socketio,
    BASE_DIR,
    UPLOAD_DIR,
    PROCESSED_DIR,
    TEXT_DIR,
    processing_status,
    update_processing_status,
    clear_processing_status,
):
    """
    Create enhanced API endpoints using the notebook pipeline

    Call this function in app.py to register the endpoints:
        from modules.api_endpoints import create_enhanced_endpoints
        create_enhanced_endpoints(app, socketio, BASE_DIR, UPLOAD_DIR, PROCESSED_DIR, TEXT_DIR,
                                 processing_status, update_processing_status, clear_processing_status)
    """

    from app.modules.pipeline.enhanced import EnhancedDocumentPipeline

    # Initialize pipeline with callback for Socket.IO progress
    def emit_progress(progress_data):
        try:
            socketio.emit("processing_progress", progress_data)
        except Exception as e:
            logger.warning(f"Socket emit error: {e}")

    pipeline = EnhancedDocumentPipeline(storage_dir=PROCESSED_DIR, emit_callback=emit_progress)

    # ========== ENDPOINT 1: Upload with Enhanced Processing ==========
    @app.route("/upload-enhanced", methods=["POST"])
    def upload_enhanced():
        """
        Enhanced upload endpoint with notebook pipeline processing

        Returns immediate response, processes in background with real-time progress
        """
        try:
            file = request.files.get("file") or request.files.get("photo")
            if not file:
                return jsonify({"error": "No file provided", "success": False}), 400

            if file.filename == "":
                return jsonify({"error": "Empty filename", "success": False}), 400

            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            original_ext = os.path.splitext(file.filename)[1] or ".jpg"
            filename = f"doc_{timestamp}_{unique_id}{original_ext}"
            processed_filename = f"processed_{filename}"

            logger.info(f"📤 Enhanced upload: {filename}")

            # Save uploaded file
            upload_path = os.path.join(UPLOAD_DIR, filename)
            file.save(upload_path)

            # Initialize processing status
            update_processing_status(processed_filename, 0, 12, "Initializing", is_complete=False)

            # Return immediate response
            response = {
                "status": "uploaded",
                "success": True,
                "message": "File uploaded, processing with enhanced pipeline...",
                "filename": processed_filename,
                "upload_filename": filename,
                "timestamp": timestamp,
                "processing": True,
                "pipeline": "enhanced",  # Indicate enhanced pipeline is being used
            }

            # Emit Socket.IO notification
            try:
                socketio.emit(
                    "new_file",
                    {
                        "filename": processed_filename,
                        "upload_filename": filename,
                        "timestamp": timestamp,
                        "processing": True,
                        "pipeline": "enhanced",
                    },
                )
            except Exception as e:
                logger.warning(f"Socket notification error: {e}")

            # Background processing
            def background_enhanced_process():
                try:
                    processed_path = os.path.join(PROCESSED_DIR, processed_filename)

                    # Get enhancement parameters from request (optional)
                    enhancement_params = {
                        "brightness_boost": int(request.form.get("brightness_boost", 25)),
                        "equalization_strength": float(
                            request.form.get("equalization_strength", 0.4)
                        ),
                        "clahe_clip_limit": float(request.form.get("clahe_clip_limit", 2.0)),
                        "clahe_tile_size": int(request.form.get("clahe_tile_size", 8)),
                    }

                    # Process with enhanced pipeline
                    success, text_or_error, pipeline_stats = pipeline.process_complete_pipeline(
                        upload_path, processed_path, enhancement_params=enhancement_params
                    )

                    if not success:
                        logger.error(f"Processing failed: {text_or_error}")
                        update_processing_status(
                            processed_filename,
                            12,
                            12,
                            "Error",
                            is_complete=True,
                            error=text_or_error,
                        )
                        socketio.emit(
                            "processing_error",
                            {
                                "filename": processed_filename,
                                "error": text_or_error,
                                "pipeline": "enhanced",
                            },
                        )
                        return

                    # Save extracted text
                    text_filename = f"{os.path.splitext(processed_filename)[0]}.txt"
                    text_path = os.path.join(TEXT_DIR, text_filename)

                    try:
                        with open(text_path, "w", encoding="utf-8") as f:
                            f.write(text_or_error)
                        logger.info(f"✓ Text saved: {len(text_or_error)} characters")
                    except Exception as e:
                        logger.warning(f"Failed to save text: {e}")

                    # Mark as complete
                    update_processing_status(
                        processed_filename, 12, 12, "Complete", is_complete=True
                    )

                    # Notify completion with stats
                    socketio.emit(
                        "processing_complete",
                        {
                            "filename": processed_filename,
                            "has_text": len(text_or_error) > 0,
                            "text_length": len(text_or_error),
                            "pipeline": "enhanced",
                            "stats": {
                                "total_steps": pipeline_stats.get("total_steps"),
                                "parameters": pipeline_stats.get("parameters"),
                            },
                        },
                    )

                    logger.info(f"✅ Enhanced processing complete for {processed_filename}")

                    # Clear status after delay
                    threading.Timer(
                        60.0, lambda: clear_processing_status(processed_filename)
                    ).start()

                except Exception as e:
                    error_msg = f"Background processing error: {str(e)}"
                    logger.error(error_msg)
                    update_processing_status(
                        processed_filename, 12, 12, "Error", is_complete=True, error=error_msg
                    )
                    socketio.emit(
                        "processing_error", {"filename": processed_filename, "error": error_msg}
                    )

            # Start background thread
            thread = threading.Thread(target=background_enhanced_process)
            thread.daemon = True
            thread.start()

            logger.info("✓ Upload response sent, enhanced processing started")
            return jsonify(response)

        except Exception as e:
            error_msg = f"Upload error: {str(e)}"
            logger.error(error_msg)
            return jsonify({"error": error_msg, "success": False}), 500

    # ========== ENDPOINT 2: Get Pipeline Configuration ==========
    @app.route("/pipeline/config", methods=["GET"])
    def get_pipeline_config():
        """Get current pipeline configuration and supported parameters"""
        try:
            config = pipeline.get_pipeline_info()
            return jsonify(
                {"success": True, "pipeline": "enhanced", "version": "1.0", "config": config}
            )
        except Exception as e:
            return jsonify({"error": str(e), "success": False}), 500

    # ========== ENDPOINT 3: Test Enhancement Parameters ==========
    @app.route("/pipeline/test-params", methods=["POST"])
    def test_enhancement_params():
        """
        Test enhancement parameters on uploaded file

        Expects: multipart with 'file' and optional JSON params
        Returns: Processing stats without saving
        """
        try:
            file = request.files.get("file")
            if not file:
                return jsonify({"error": "No file provided"}), 400

            # Save temporarily
            temp_path = os.path.join(UPLOAD_DIR, f"temp_{uuid.uuid4()}.jpg")
            file.save(temp_path)

            try:
                # Get parameters from request
                enhancement_params = {
                    "brightness_boost": int(request.form.get("brightness_boost", 25)),
                    "equalization_strength": float(request.form.get("equalization_strength", 0.4)),
                    "clahe_clip_limit": float(request.form.get("clahe_clip_limit", 2.0)),
                    "clahe_tile_size": int(request.form.get("clahe_tile_size", 8)),
                }

                # Process with test output path
                test_output_path = temp_path.replace(".jpg", "_processed.jpg")
                success, text, stats = pipeline.process_complete_pipeline(
                    temp_path, test_output_path, enhancement_params=enhancement_params
                )

                if success:
                    # Return stats (delete test output)
                    if os.path.exists(test_output_path):
                        os.remove(test_output_path)

                    return jsonify(
                        {
                            "success": True,
                            "text_extracted": len(text),
                            "parameters_used": enhancement_params,
                            "stats": stats.get("steps", {}).get("step_8", {}).get("stats", {}),
                        }
                    )
                else:
                    return jsonify({"success": False, "error": text}), 500

            finally:
                # Clean up temp files
                for path in [temp_path, test_output_path]:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except:
                            pass

        except Exception as e:
            logger.error(f"Test params error: {e}")
            return jsonify({"error": str(e)}), 500

    # ========== ENDPOINT 4: Batch Upload with Custom Parameters ==========
    @app.route("/upload-batch-enhanced", methods=["POST"])
    def upload_batch_enhanced():
        """
        Batch upload with enhanced processing and custom parameters

        Expects: multipart with 'files[]' and optional custom parameters
        """
        try:
            files = request.files.getlist("files[]")
            if not files:
                return jsonify({"error": "No files provided", "success": False}), 400

            # Get global enhancement params
            global_params = {
                "brightness_boost": int(request.form.get("brightness_boost", 25)),
                "equalization_strength": float(request.form.get("equalization_strength", 0.4)),
                "clahe_clip_limit": float(request.form.get("clahe_clip_limit", 2.0)),
                "clahe_tile_size": int(request.form.get("clahe_tile_size", 8)),
            }

            logger.info(f"📦 Batch upload: {len(files)} files with params {global_params}")

            upload_info = []

            for i, file in enumerate(files):
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_id = str(uuid.uuid4())[:8]
                    ext = os.path.splitext(file.filename)[1] or ".jpg"
                    filename = f"batch_{timestamp}_{unique_id}_{i}{ext}"
                    processed_filename = f"processed_{filename}"

                    # Save file
                    upload_path = os.path.join(UPLOAD_DIR, filename)
                    file.save(upload_path)

                    upload_info.append(
                        {"filename": processed_filename, "upload_filename": filename, "index": i}
                    )

                    # Initialize processing
                    update_processing_status(
                        processed_filename, 0, 12, "Initializing", is_complete=False
                    )

                except Exception as e:
                    logger.error(f"Batch file {i} error: {e}")
                    continue

            if not upload_info:
                return jsonify({"error": "Failed to process any files"}), 500

            # Start batch background processing
            def background_batch_process():
                for info in upload_info:
                    try:
                        processed_filename = info["filename"]
                        upload_filename = info["upload_filename"]
                        upload_path = os.path.join(UPLOAD_DIR, upload_filename)
                        processed_path = os.path.join(PROCESSED_DIR, processed_filename)

                        # Process with enhanced pipeline
                        success, text, stats = pipeline.process_complete_pipeline(
                            upload_path, processed_path, enhancement_params=global_params
                        )

                        if success:
                            # Save text
                            text_filename = f"{os.path.splitext(processed_filename)[0]}.txt"
                            text_path = os.path.join(TEXT_DIR, text_filename)
                            with open(text_path, "w", encoding="utf-8") as f:
                                f.write(text)

                            update_processing_status(
                                processed_filename, 12, 12, "Complete", is_complete=True
                            )
                        else:
                            update_processing_status(
                                processed_filename, 12, 12, "Error", is_complete=True, error=text
                            )

                        socketio.emit(
                            "processing_complete",
                            {"filename": processed_filename, "success": success, "batch": True},
                        )

                    except Exception as e:
                        logger.error(f"Batch processing error: {e}")

            thread = threading.Thread(target=background_batch_process)
            thread.daemon = True
            thread.start()

            return jsonify(
                {
                    "success": True,
                    "message": f"Batch processing started for {len(upload_info)} files",
                    "files_count": len(upload_info),
                    "files": upload_info,
                    "parameters": global_params,
                }
            )

        except Exception as e:
            logger.error(f"Batch upload error: {e}")
            return jsonify({"error": str(e), "success": False}), 500

    # ========== ENDPOINT 5: Get Pipeline Statistics ==========
    @app.route("/pipeline/stats", methods=["GET"])
    def get_pipeline_stats():
        """Get pipeline configuration and statistics"""
        try:
            stats = {
                "pipeline_type": "enhanced",
                "steps": 12,
                "configuration": pipeline.get_pipeline_info(),
                "current_parameters": pipeline.enhancer.get_parameters(),
            }
            return jsonify(stats)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    logger.info("✅ Enhanced pipeline endpoints registered")
    return pipeline  # Return pipeline for reference if needed
