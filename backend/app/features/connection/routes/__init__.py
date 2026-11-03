"""
PrintChakra Backend - Connection Routes

Route registration for connection feature.
"""

from flask import Blueprint, jsonify, request

from app.core.paths import IS_CLOUD

connection_bp = Blueprint("connection", __name__, url_prefix="/connection")

BROWSER_PRINTER = {"name": "Browser Print", "status": "Ready", "offline": False, "connected": True}


@connection_bp.before_request
def _cloud_hardware_stubs():
    """The cloud deployment has no server-side WiFi adapter, webcam or printer. Answer for them here."""
    if not IS_CLOUD or request.method == "OPTIONS":
        return None

    path = request.path
    unsupported = jsonify({"success": False, "error": "Not available in the cloud deployment"}), 400

    if path.startswith("/connection/wifi/status"):
        return jsonify({"success": True, "wifi": {"connected": True, "ssid": "Internet", "signal": None}})
    if path.startswith("/connection/wifi/networks"):
        return jsonify({"success": True, "networks": [], "total": 0})
    if path.startswith("/connection/wifi/"):
        return unsupported

    if path.startswith("/connection/camera/status"):
        return jsonify({"success": True, "camera": {"available": False, "cameras": [], "total": 0}})
    if path.startswith("/connection/camera/list"):
        return jsonify({"success": True, "cameras": [], "total": 0})
    if path.startswith("/connection/camera/test"):
        return jsonify({"success": False, "error": "Use the phone or browser camera in the cloud deployment"}), 400

    if path.startswith("/connection/printer/status") or path.startswith("/connection/printer/discover"):
        return jsonify({"success": True, "connected": True, "printers": [BROWSER_PRINTER], "total": 1})
    if path.startswith("/connection/printer/test/"):
        return jsonify({"success": True, "message": "Printing uses the browser print dialog"})
    if path.startswith("/connection/printer/"):
        return unsupported

    return None


# Import route modules to register endpoints
from app.features.connection.routes import wifi
from app.features.connection.routes import camera
from app.features.connection.routes import printer
from app.features.connection.routes import services

__all__ = ["connection_bp"]
