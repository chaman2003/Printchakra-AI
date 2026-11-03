"""
PrintChakra Backend - Application Configuration

Centralized configuration management for the Flask application.
"""

import os
from typing import Dict, Any

from app.core.paths import data_root, user_root


class BaseConfig:
    """Base configuration shared across all environments."""
    
    # Flask settings
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    
    # CORS settings
    CORS_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:5000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5000",
    ]
    
    # File paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PUBLIC_DIR = os.path.join(BASE_DIR, "public")
    DATA_DIR = data_root(os.path.join(PUBLIC_DIR, "data"))
    
    # Data directories
    UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
    TEXT_DIR = os.path.join(DATA_DIR, "processed_text")
    PDF_DIR = os.path.join(DATA_DIR, "pdfs")
    CONVERTED_DIR = os.path.join(DATA_DIR, "converted")
    OCR_DATA_DIR = os.path.join(DATA_DIR, "ocr_results")
    MODELS_DIR = os.path.join(DATA_DIR, "models")
    
    # Socket.IO settings
    SOCKETIO_ASYNC_MODE = "threading"
    SOCKETIO_PING_TIMEOUT = 60
    SOCKETIO_PING_INTERVAL = 25
    SOCKETIO_MAX_HTTP_BUFFER_SIZE = int(1e7)
    
    # File upload settings
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf', 'webp'}
    
    # Processing settings
    DEFAULT_IMAGE_QUALITY = 95
    THUMBNAIL_SIZE = (300, 300)
    
    # OCR settings
    OCR_TIMEOUT = 120  # seconds
    
    # Voice settings
    VOICE_MODEL_PATH = os.path.join(MODELS_DIR, "vosk-model-small-en-us-0.15")
    
    @classmethod
    def init_app(cls, app) -> None:
        """Initialize application with this configuration."""
        # Ensure all directories exist
        for dir_attr in ['UPLOAD_DIR', 'PROCESSED_DIR', 'TEXT_DIR', 'PDF_DIR', 
                         'CONVERTED_DIR', 'OCR_DATA_DIR', 'MODELS_DIR']:
            dir_path = getattr(cls, dir_attr)
            os.makedirs(dir_path, exist_ok=True)


class DevelopmentConfig(BaseConfig):
    """Development environment configuration."""
    DEBUG = True
    TESTING = False
    
    # Add ngrok URL if set
    @property
    def CORS_ORIGINS(self):
        origins = super().CORS_ORIGINS.copy()
        if os.environ.get('NGROK_URL'):
            origins.append(os.environ.get('NGROK_URL'))
        if os.environ.get('FRONTEND_URL'):
            origins.append(os.environ.get('FRONTEND_URL'))
        return origins


class ProductionConfig(BaseConfig):
    """Production environment configuration."""
    DEBUG = False
    TESTING = False
    
    # Override with environment variables
    SECRET_KEY = os.environ.get("SECRET_KEY")
    
    @property
    def CORS_ORIGINS(self):
        origins = super().CORS_ORIGINS.copy()
        if os.environ.get('FRONTEND_URL'):
            origins.append(os.environ.get('FRONTEND_URL'))
        if os.environ.get('NGROK_URL'):
            origins.append(os.environ.get('NGROK_URL'))
        return origins


class TestingConfig(BaseConfig):
    """Testing environment configuration."""
    DEBUG = True
    TESTING = True
    
    # Use temporary directories for testing
    UPLOAD_DIR = "/tmp/printchakra/uploads"
    PROCESSED_DIR = "/tmp/printchakra/processed"


# Configuration mapping
config_map: Dict[str, BaseConfig] = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}


def get_config(config_name: str = None) -> BaseConfig:
    """
    Get configuration class by name.
    
    Args:
        config_name: Configuration environment name
        
    Returns:
        Configuration class
    """
    if config_name is None:
        config_name = os.environ.get("FLASK_ENV", "development")
    
    return config_map.get(config_name, config_map["default"])


def get_data_dirs() -> Dict[str, str]:
    """
    Get all data directory paths.
    
    Returns:
        Dictionary of directory names to paths
    """
    config = get_config()
    dirs = {
        'BASE_DIR': config.BASE_DIR,
        'PUBLIC_DIR': config.PUBLIC_DIR,
        'DATA_DIR': config.DATA_DIR,
        'UPLOAD_DIR': config.UPLOAD_DIR,
        'PROCESSED_DIR': config.PROCESSED_DIR,
        'TEXT_DIR': config.TEXT_DIR,
        'PDF_DIR': config.PDF_DIR,
        'CONVERTED_DIR': config.CONVERTED_DIR,
        'OCR_DATA_DIR': config.OCR_DATA_DIR,
        'MODELS_DIR': config.MODELS_DIR,
    }
    # A logged-in user works inside a private folder. MODELS_DIR stays shared (it holds no user data).
    root = user_root(config.DATA_DIR)
    if root != config.DATA_DIR:
        for key in ('UPLOAD_DIR', 'PROCESSED_DIR', 'TEXT_DIR', 'PDF_DIR', 'CONVERTED_DIR', 'OCR_DATA_DIR'):
            dirs[key] = os.path.join(root, os.path.relpath(dirs[key], config.DATA_DIR))
        dirs['DATA_DIR'] = root
    return dirs
