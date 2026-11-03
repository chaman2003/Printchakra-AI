"""
PrintChakra Backend - Core Package

This module provides the Flask application factory and core configuration.
"""

import os

from flask import Flask
from flask_cors import CORS

from app.core.paths import IS_CLOUD

# Global extensions (initialized later)
if IS_CLOUD:
    # Serverless hosting has no websockets: events go through the database and clients poll for them.
    from app.core.events import CloudSocketIO

    socketio = CloudSocketIO()
else:
    from flask_socketio import SocketIO

    socketio = SocketIO()
cors = CORS()


def create_app(config_name: str = "development") -> Flask:
    """
    Application factory for creating Flask app instances.
    
    Args:
        config_name: Configuration environment name
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Load configuration
    from app.core.config import get_config
    app.config.from_object(get_config(config_name))
    
    if IS_CLOUD or os.environ.get("REQUIRE_LOGIN") == "1":
        # Registered before the cloud sync hooks so a request without a valid token never touches the database.
        from app.features.auth import require_login

        app.before_request(require_login)

    if IS_CLOUD:
        from app.core import cloud
        from app.core.events import events_bp

        app.register_blueprint(events_bp)
        cloud.init_app(app)

    from app.features.auth import auth_bp

    app.register_blueprint(auth_bp)

    # Initialize extensions
    _init_extensions(app)
    
    # Register blueprints
    _register_blueprints(app)
    
    # Register socket handlers
    _register_socket_handlers()
    
    # Setup error handlers
    _setup_error_handlers(app)
    
    return app


def _init_extensions(app: Flask) -> None:
    """Initialize Flask extensions."""
    from app.core.extensions import init_cors

    init_cors(app)
    if not IS_CLOUD:
        from app.core.extensions import init_socketio

        init_socketio(app, socketio)


# Blueprints that need local hardware or local models and have no serverless equivalent.
# Voice runs in the browser through Gemini Live, so the Whisper/Ollama/Pipecat routes stay off.
_CLOUD_SKIPPED_BLUEPRINTS = {"voice", "pipecat"}


def _register_blueprints(app: Flask) -> None:
    """Register all feature blueprints."""
    blueprint_specs = [
        ("dashboard", "app.features.dashboard.routes", "dashboard_bp", None),
        ("phone", "app.features.phone.routes", "phone_bp", "/phone"),
        ("document", "app.features.document.routes", "document_bp", "/document"),
        ("OCR", "app.features.document.ocr.routes", "ocr_bp", "/ocr"),
        ("print", "app.features.print.routes", "print_bp", "/print"),
        ("voice", "app.features.voice.routes", "voice_bp", "/voice"),
        ("pipecat", "app.features.voice.pipecat_routes", "pipecat_bp", "/pipecat"),
        ("orchestration", "app.features.orchestration.routes", "orchestration_bp", "/orchestrate"),
        ("connection", "app.features.connection.routes", "connection_bp", "/connection"),
    ]

    for feature_name, module_path, blueprint_name, prefix in blueprint_specs:
        if IS_CLOUD and feature_name in _CLOUD_SKIPPED_BLUEPRINTS:
            continue
        try:
            module = __import__(module_path, fromlist=[blueprint_name])
            blueprint = getattr(module, blueprint_name)
            if prefix:
                app.register_blueprint(blueprint, url_prefix=prefix)
            else:
                app.register_blueprint(blueprint)
        except ImportError as e:
            app.logger.warning(f"Could not load {feature_name} routes: {e}")


def _register_socket_handlers() -> None:
    """Register Socket.IO event handlers."""
    if IS_CLOUD:
        return
    try:
        from app.sockets.handlers import register_handlers
        register_handlers(socketio)
    except ImportError as e:
        print(f"Could not load socket handlers: {e}")


def _setup_error_handlers(app: Flask) -> None:
    """Setup application error handlers."""
    try:
        from app.core.middleware.error_handler import register_error_handlers
        register_error_handlers(app)
    except ImportError:
        # Fallback error handlers
        @app.errorhandler(400)
        def handle_bad_request(e):
            from flask import jsonify
            return jsonify({"error": "Bad request"}), 400
        
        @app.errorhandler(404)
        def handle_not_found(e):
            from flask import jsonify
            return jsonify({"error": "Not found"}), 404
        
        @app.errorhandler(500)
        def handle_internal_error(e):
            from flask import jsonify
            return jsonify({"error": "Internal server error"}), 500
