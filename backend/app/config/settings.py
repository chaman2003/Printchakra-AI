"""
PrintChakra Backend Configuration
Central configuration for all modules and settings
"""

import os
from pathlib import Path
from typing import List

from app.core.paths import data_root, logs_root

try:  # Optional dependency, documented in requirements
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback when python-dotenv missing
    load_dotenv = None

# Base directories
CONFIG_DIR = Path(__file__).parent
BASE_DIR = CONFIG_DIR.parent.absolute()
DATA_DIR = Path(data_root(BASE_DIR / "data"))

# Optional .env overrides (keeps configuration English-editable)
ENV_FILE = BASE_DIR / ".env"
if load_dotenv and ENV_FILE.exists():
    load_dotenv(ENV_FILE)


def env(key: str, default: str | None = None) -> str | None:
    """Read environment variables with sane defaults"""

    return os.environ.get(key, default)


def env_bool(key: str, default: bool = False) -> bool:
    value = os.environ.get(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_list(key: str, default: List[str] | None = None, separator: str = ",") -> List[str]:
    raw_value = os.environ.get(key)
    if raw_value is None:
        return list(default or [])
    return [item.strip() for item in raw_value.split(separator) if item.strip()]

# Directory Configuration
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
TEXT_DIR = DATA_DIR / "processed_text"
PDF_DIR = DATA_DIR / "pdfs"
CONVERTED_DIR = DATA_DIR / "converted"
MODELS_DIR = DATA_DIR / "models"
PRINT_DIR = BASE_DIR / "app" / "print_scripts"
LOGS_DIR = Path(logs_root(BASE_DIR / "logs"))

# Ensure all directories exist
for directory in [
    DATA_DIR,
    UPLOAD_DIR,
    PROCESSED_DIR,
    TEXT_DIR,
    PDF_DIR,
    CONVERTED_DIR,
    MODELS_DIR,
    PRINT_DIR,
    LOGS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Prompt assets (can be edited without touching Python)
prompts_dir_override = env("PROMPTS_DIR")
PROMPTS_DIR = (
    Path(prompts_dir_override).expanduser()
    if prompts_dir_override
    else CONFIG_DIR / "prompts"
)
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
SYSTEM_PROMPT_FILE = Path(env("VOICE_SYSTEM_PROMPT_FILE", str(PROMPTS_DIR / "system_prompt.txt")))
COMMAND_MAPPINGS_FILE = Path(
    env("VOICE_COMMAND_MAPPINGS_FILE", str(PROMPTS_DIR / "command_mappings.json"))
)

# Processing Configuration
PROCESSING_CONFIG = {
    "blur_threshold": 100.0,
    "focus_threshold": 50.0,
    "brightness_boost": 25,
    "equalization_strength": 0.4,
    "clahe_clip_limit": 2.0,
    "clahe_tile_size": 8,
    "corner_inset": 12,  # Pixels to move corners inward to avoid shadows
}

# OCR Configuration
OCR_CONFIG = {
    "language": "eng",
    "psm": 3,  # Page segmentation mode
    "oem": 3,  # OCR engine mode
    "configs": [
        ("PSM 3 (Auto)", "--oem 3 --psm 3"),
        ("PSM 6 (Block)", "--oem 3 --psm 6"),
        ("PSM 4 (Column)", "--oem 3 --psm 4"),
    ],
}

# Image Enhancement Configuration
ENHANCEMENT_CONFIG = {
    "bilateral_filter": {"d": 9, "sigma_color": 75, "sigma_space": 75},
    "adaptive_threshold": {"block_sizes": [11, 13, 15], "constants": [2, 3]},
    "clahe": {"clip_limits": [2.0, 3.0], "tile_grid_size": (8, 8)},
}

# Export Configuration
EXPORT_CONFIG = {"jpeg_quality": 95, "pdf_page_size": "A4", "compression_quality": 85}

# API Configuration
DEFAULT_CORS_ORIGINS = [
    "https://printchakra.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*",  # Allow all as fallback
]
cors_origins = env_list("API_CORS_ORIGINS", DEFAULT_CORS_ORIGINS)

API_CONFIG = {
    "cors_origins": cors_origins,
    "max_file_size": 10 * 1024 * 1024,  # 10 MB
    "allowed_extensions": {".jpg", ".jpeg", ".png", ".pdf"},
    "socket_io": {"ping_timeout": 120, "ping_interval": 30, "max_http_buffer_size": 1e7},
}

# Connection + AI runtime configuration
_ollama_base = (env("OLLAMA_BASE_URL", "http://localhost:11434") or "").rstrip("/")
if not _ollama_base:
    _ollama_base = "http://localhost:11434"

CONNECTION_CONFIG = {
    "frontend_origin": env("FRONTEND_URL", "http://localhost:3000"),
    "backend_public_url": env("BACKEND_PUBLIC_URL", ""),
    "ollama": {
        "base_url": _ollama_base,
        "chat_endpoint": env("OLLAMA_CHAT_ENDPOINT", "/api/chat"),
        "tags_endpoint": env("OLLAMA_TAGS_ENDPOINT", "/api/tags"),
        "timeout": int(env("OLLAMA_TIMEOUT", "15")),
        "verify_ssl": env_bool("OLLAMA_VERIFY_SSL", False),
    },
    "groq": {
        "base_url": env("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/"),
        "api_key": env("GROQ_API_KEY", ""),
        "llm_model": env("GROQ_LLM_MODEL", "llama-3.1-8b-instant"),
        "stt_model": env("GROQ_STT_MODEL", "whisper-large-v3-turbo"),
        "tts_model": env("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english"),
        "tts_endpoint": env("GROQ_TTS_ENDPOINT", "https://api.groq.com/openai/v1/audio/speech"),
        "timeout": int(env("GROQ_TIMEOUT", "30")),
        "verify_ssl": env_bool("GROQ_VERIFY_SSL", True),
    },
}

AI_PROMPT_CONFIG = {
    "default_model": env("VOICE_AI_MODEL", "smollm2:135m"),
    "system_prompt_file": str(SYSTEM_PROMPT_FILE),
    "command_mappings_file": str(COMMAND_MAPPINGS_FILE),
    "models_dir": str(MODELS_DIR),
}

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(LOGS_DIR / "printchakra.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
    },
    "loggers": {
        "": {"handlers": ["console", "file"], "level": "INFO"},  # Root logger
        "werkzeug": {"level": "WARNING"},
        "socketio": {"level": "WARNING"},
        "engineio": {"level": "WARNING"},
    },
}

# Feature Flags
FEATURES = {
    "advanced_pipeline": True,
    "document_detection": True,
    "corner_refinement": True,
    "multi_config_ocr": True,
    "ai_enhancement": False,  # Requires additional dependencies
    "cloud_storage": False,  # Not yet implemented
    "batch_processing": True,
    "pdf_export": True,
    "document_classification": False,  # Requires training
}

# Version
VERSION = "2.1.0"


# Flask Configuration Class
class Config:
    """Flask configuration"""

    SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-secret-key-change-in-production"

    # Directory paths
    UPLOAD_FOLDER = str(UPLOAD_DIR)
    PROCESSED_FOLDER = str(PROCESSED_DIR)
    CONVERTED_FOLDER = str(CONVERTED_DIR)
    TEXT_FOLDER = str(TEXT_DIR)
    PDF_FOLDER = str(PDF_DIR)

    # File upload settings
    MAX_CONTENT_LENGTH = API_CONFIG["max_file_size"]
    ALLOWED_EXTENSIONS = API_CONFIG["allowed_extensions"]

    # CORS settings
    CORS_ORIGINS = API_CONFIG["cors_origins"]


class DevelopmentConfig(Config):
    """Development configuration"""

    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""

    DEBUG = False
    TESTING = False


class TestingConfig(Config):
    """Testing configuration"""

    DEBUG = True
    TESTING = True


# Configuration dictionary
config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}
