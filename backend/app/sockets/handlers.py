"""
PrintChakra Backend - Socket.IO Event Handlers

Handles real-time communication events including:
- Client connections/disconnections
- Frame detection
- Progress updates
- File notifications
"""

import base64
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Reference to socketio instance (set during registration)
_socketio = None


def register_handlers(socketio):
    """
    Register Socket.IO event handlers.
    
    Args:
        socketio: Flask-SocketIO instance
    """
    global _socketio
    _socketio = socketio
    
    @socketio.on("connect")
    def handle_connect():
        """Handle client connection."""
        from flask import request
        logger.info(f"Socket connected: {request.sid}")  # type: ignore[attr-defined]
        return True
    
    @socketio.on("disconnect")
    def handle_disconnect():
        """Handle client disconnection."""
        from flask import request
        logger.info(f"Socket disconnected: {request.sid}")  # type: ignore[attr-defined]
    
    @socketio.on("error")
    def handle_error(e):
        """Handle socket errors."""
        logger.error(f"Socket error: {e}")
    
    @socketio.on("ping")
    def handle_ping():
        """Handle ping from client."""
        from flask_socketio import emit
        emit("pong", {"timestamp": datetime.now().isoformat()})
    
    @socketio.on("detect_frame")
    def handle_frame_detection(data):
        """
        Real-time frame detection via WebSocket.
        
        Expects: base64 encoded image data
        Emits: detection result with corners
        """
        from flask_socketio import emit
        
        try:
            # Check if detection service is available
            try:
                from app.modules.document import detect_and_serialize
            except ImportError:
                emit("detection_result", {
                    "success": False,
                    "message": "Detection service unavailable"
                })
                return
            
            image_data = data.get("image")
            if not image_data:
                emit("detection_result", {
                    "success": False,
                    "message": "No image data"
                })
                return
            
            # Remove data URL prefix if present
            if "," in image_data:
                image_data = image_data.split(",")[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            
            # Detect
            result = detect_and_serialize(image_bytes)
            
            # Emit result
            emit("detection_result", result)
        
        except Exception as e:
            logger.error(f"Frame detection error: {str(e)}")
            emit("detection_result", {
                "success": False,
                "message": f"Detection error: {str(e)}"
            })
    
    @socketio.on("camera_frame")
    def handle_camera_frame(data):
        """
        Handle camera frame from phone for processing.
        
        Expects: { "frame": base64_image, "quality": int }
        """
        from flask_socketio import emit
        
        try:
            frame_data = data.get("frame")
            if not frame_data:
                return
            
            # Broadcast to other clients (e.g., dashboard)
            emit("camera_preview", {
                "frame": frame_data,
                "timestamp": datetime.now().isoformat()
            }, broadcast=True, include_self=False)
        
        except Exception as e:
            logger.error(f"Camera frame error: {e}")
    
    @socketio.on("capture_complete")
    def handle_capture_complete(data):
        """
        Handle capture completion notification from phone.
        
        Expects: { "filename": str, "success": bool }
        """
        from flask_socketio import emit
        
        try:
            emit("capture_result", {
                "filename": data.get("filename"),
                "success": data.get("success", True),
                "timestamp": datetime.now().isoformat()
            }, broadcast=True)
        
        except Exception as e:
            logger.error(f"Capture complete error: {e}")
    
    @socketio.on("processing_progress")
    def handle_processing_progress(data):
        """
        Handle processing progress updates.
        
        Expects: { "step": str, "progress": int, "message": str }
        """
        from flask_socketio import emit
        
        try:
            emit("progress_update", {
                "step": data.get("step"),
                "progress": data.get("progress", 0),
                "message": data.get("message", ""),
                "timestamp": datetime.now().isoformat()
            }, broadcast=True)
        

        except Exception as e:
            logger.error(f"Progress broadcast error: {e}")

    @socketio.on("start_auto_capture")
    def handle_start_auto_capture(data):
        """
        Handle auto-capture start command from Dashboard.
        Relays the command to all clients (specifically the Phone).
        """
        from flask_socketio import emit
        try:
            logger.info(f"Received start_auto_capture: {data}")
            emit("start_auto_capture", data, broadcast=True)
        except Exception as e:
            logger.error(f"Error in start_auto_capture handler: {e}")

    @socketio.on("stop_auto_capture")
    def handle_stop_auto_capture():
        """
        Handle auto-capture stop command from Dashboard.
        """
        from flask_socketio import emit
        try:
            logger.info("Received stop_auto_capture")
            emit("stop_auto_capture", broadcast=True)
        except Exception as e:
            logger.error(f"Error in stop_auto_capture handler: {e}")
    
    @socketio.on("auto_capture_state_changed")
    def handle_auto_capture_state(data):
        """
        Handle state change updates (enabled/disabled) to sync Dashboard and Phone.
        """
        from flask_socketio import emit
        try:
            emit("auto_capture_state_changed", data, broadcast=True, include_self=False)
        except Exception as e:
            logger.error(f"Error in auto_capture_state_changed handler: {e}")

    @socketio.on("delay_settings_updated")
    def handle_delay_settings_updated(data):
        """
        Handle delay settings updates to sync across all devices.
        Broadcasts to all connected clients including Dashboard and Phone.
        """
        from flask_socketio import emit
        try:
            logger.info(f"Received delay_settings_updated: {data}")
            emit("delay_settings_updated", data, broadcast=True, include_self=False)
        except Exception as e:
            logger.error(f"Error in delay_settings_updated handler: {e}")


def emit_event(event: str, data: dict, broadcast: bool = True):
    """
    Emit a Socket.IO event.
    
    Args:
        event: Event name
        data: Event data
        broadcast: Whether to broadcast to all clients
    """
    if _socketio:
        _socketio.emit(event, data, broadcast=broadcast)


def emit_file_uploaded(filename: str, processed: bool = False):
    """Emit file uploaded notification."""
    emit_event("file_uploaded", {
        "filename": filename,
        "processed": processed,
        "timestamp": datetime.now().isoformat()
    })


def emit_file_processed(filename: str, output_filename: str):
    """Emit file processed notification."""
    emit_event("file_processed", {
        "filename": filename,
        "output_filename": output_filename,
        "timestamp": datetime.now().isoformat()
    })


def emit_file_deleted(filename: str):
    """Emit file deleted notification."""
    emit_event("file_deleted", {
        "filename": filename,
        "timestamp": datetime.now().isoformat()
    })


def emit_ocr_complete(filename: str, result: dict):
    """Emit OCR completion notification."""
    emit_event("ocr_complete", {
        "filename": filename,
        "success": True,
        "result": result,
        "timestamp": datetime.now().isoformat()
    })


def emit_print_job_started(filename: str, job_id: str = None):
    """Emit print job started notification."""
    emit_event("print_job_started", {
        "filename": filename,
        "job_id": job_id,
        "timestamp": datetime.now().isoformat()
    })


def emit_orchestration_update(update_type: str, data: dict):
    """Emit orchestration state update."""
    emit_event("orchestration_update", {
        "type": update_type,
        **data,
        "timestamp": datetime.now().isoformat()
    })
