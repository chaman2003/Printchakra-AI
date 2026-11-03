"""
HTTP replacement for Socket.IO on serverless hosting.

GET  /events/poll?since=<id>   -> events newer than <id>. since=-1 returns only the latest id.
POST /events/emit              -> relay an event from one client to the others (e.g. dashboard -> phone).
"""

import logging

from flask import Blueprint, jsonify, request

from app.core import cloud
from app.core.paths import IS_CLOUD

logger = logging.getLogger(__name__)

events_bp = Blueprint("events", __name__, url_prefix="/events")

# Events a client may relay to other clients, with the name the receivers listen for.
# This mirrors the broadcast handlers in app/sockets/handlers.py.
RELAYED_EVENTS = {
    "start_auto_capture": "start_auto_capture",
    "stop_auto_capture": "stop_auto_capture",
    "auto_capture_state_changed": "auto_capture_state_changed",
    "delay_settings_updated": "delay_settings_updated",
    "set_active_folder": "active_folder_changed",
    "capture_complete": "capture_result",
    "capture_now": "capture_now",
    "calibration_test_mode": "calibration_test_mode",
    "start_test_capture": "start_test_capture",
    "cancel_test_capture": "cancel_test_capture",
}


class CloudSocketIO:
    """Stands in for flask_socketio.SocketIO: server-side `socketio.emit(...)` writes an event row."""

    def init_app(self, *args, **kwargs):
        return None

    def on(self, *args, **kwargs):
        # Used as a decorator by handlers that cannot run on serverless; keep the function, drop the wiring.
        return lambda func: func

    def emit(self, name, data=None, *args, **kwargs):
        cloud.emit(name, data if isinstance(data, dict) else {"value": data})


@events_bp.route("/poll", methods=["GET"])
def poll():
    if not (IS_CLOUD and cloud.enabled()):
        return jsonify({"success": True, "events": [], "last_id": 0, "mode": "disabled"})
    try:
        since = int(request.args.get("since", "-1"))
    except ValueError:
        since = -1
    try:
        events, last_id = cloud.poll_events(since)
    except Exception as exc:
        logger.error("poll failed: %s", exc)
        return jsonify({"success": False, "error": "event store unavailable"}), 503
    response = jsonify({"success": True, "events": events, "last_id": last_id})
    response.headers["Cache-Control"] = "no-store"
    return response


@events_bp.route("/emit", methods=["POST"])
def emit_event():
    body = request.get_json(silent=True) or {}
    name = str(body.get("name") or "")
    target = RELAYED_EVENTS.get(name)
    if not target:
        return jsonify({"success": False, "error": "event not relayable"}), 400
    data = body.get("data")
    sender = str(body.get("sender") or "")[:64] or None
    event_id = cloud.emit(target, data if isinstance(data, dict) else {}, sender=sender)
    return jsonify({"success": event_id is not None, "id": event_id})
