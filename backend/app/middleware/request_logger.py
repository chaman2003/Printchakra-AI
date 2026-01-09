"""
Request logger middleware
Logs incoming requests
"""

import time

from flask import request

from utils.logger import get_logger

logger = get_logger(__name__)


class RequestLogger:
    """Middleware to log HTTP requests"""

    def __init__(self, app=None):
        """
        Initialize request logger

        Args:
            app: Flask application instance (optional)
        """
        self.app = app
        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        """
        Initialize with Flask app

        Args:
            app: Flask application instance
        """
        app.before_request(self.before_request)
        app.after_request(self.after_request)

    def before_request(self):
        """Log before request processing"""
        request.start_time = time.time()  # type: ignore[attr-defined]

    def after_request(self, response):
        """Log after request processing"""
        if hasattr(request, "start_time"):
            duration = time.time() - request.start_time  # type: ignore[attr-defined]
            logger.info(
                f"{request.method} {request.path} - "
                f"Status: {response.status_code} - "
                f"Duration: {duration:.3f}s"
            )
        return response
