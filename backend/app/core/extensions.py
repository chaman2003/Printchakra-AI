"""
PrintChakra Backend - Flask Extensions

Initialize and configure Flask extensions.
"""

import logging
from flask import Flask
from flask_cors import CORS

logger = logging.getLogger(__name__)


def init_cors(app: Flask) -> CORS:
    """
    Initialize CORS for the Flask application.
    
    Args:
        app: Flask application instance
        
    Returns:
        Configured CORS instance
    """
    cors = CORS(
        app,
        resources={
            r"/*": {
                "origins": r"https?://.*",  # Regex matches any http/https origin
                "methods": ["GET", "POST", "DELETE", "OPTIONS", "PUT", "PATCH"],
                "allow_headers": [
                    "Content-Type",
                    "Authorization",
                    "ngrok-skip-browser-warning",
                    "X-Requested-With",
                    "X-Forwarded-For",
                    "X-Forwarded-Proto",
                    "Accept",
                ],
                "expose_headers": ["Content-Type", "Content-Disposition"],
                "supports_credentials": True,
                "max_age": 3600,
            }
        },
    )
    
    logger.info("[CORS] Configured for multi-environment: local, ngrok, deployed")
    return cors


def init_socketio(app: Flask, socketio: "SocketIO") -> "SocketIO":
    """
    Initialize Socket.IO for the Flask application.
    
    Args:
        app: Flask application instance
        socketio: SocketIO instance to initialize
        
    Returns:
        Configured SocketIO instance
    """
    def allow_all_origins(origin):
        """Allow all origins for Socket.IO with credentials support."""
        return True
    
    socketio.init_app(
        app,
        cors_allowed_origins=allow_all_origins,
        async_mode=app.config.get('SOCKETIO_ASYNC_MODE', 'threading'),
        logger=False,
        engineio_logger=False,
        ping_timeout=app.config.get('SOCKETIO_PING_TIMEOUT', 60),
        ping_interval=app.config.get('SOCKETIO_PING_INTERVAL', 25),
        max_http_buffer_size=app.config.get('SOCKETIO_MAX_HTTP_BUFFER_SIZE', int(1e7)),
        always_connect=True,
        transports=["polling", "websocket"],
        cors_credentials=True,
    )
    
    logger.info("[Socket.IO] Initialized with threading mode")
    return socketio
