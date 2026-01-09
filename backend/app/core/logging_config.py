"""
PrintChakra Backend - Logging Configuration

Centralized logging setup for the application.
"""

import logging
import sys
from typing import Optional


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """
    Configure application logging.
    
    Args:
        level: Logging level
        log_file: Optional file path for log output
    """
    # Fix Windows console encoding issues
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
            sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=_get_handlers(level, log_file)
    )
    
    # Suppress verbose third-party logging
    _configure_third_party_loggers()


def _get_handlers(level: int, log_file: Optional[str]) -> list:
    """Get logging handlers."""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        handlers.append(file_handler)
    
    return handlers


def _configure_third_party_loggers() -> None:
    """Configure logging levels for third-party libraries."""
    # Suppress verbose werkzeug logging (Flask request logs)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    
    # Suppress Socket.IO logging
    logging.getLogger("socketio").setLevel(logging.WARNING)
    logging.getLogger("engineio").setLevel(logging.WARNING)
    
    # Suppress PIL/Pillow logging
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    # Suppress urllib3 logging
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
