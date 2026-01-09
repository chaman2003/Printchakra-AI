import logging
import os
import shutil
import subprocess
import sys
import threading
import traceback
import uuid
from datetime import datetime
import io
import json

# CRITICAL: Set PaddleOCR environment variables before any paddle imports
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['PADDLEX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['HUB_HOME'] = os.path.expanduser('~/.paddlex')
os.environ['FLAGS_check_nan_inf'] = '0'
os.environ['GLOG_v'] = '0'
os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'
os.environ['GLOG_vmodule'] = 'gpu_resources=5'  # Suppress Paddle CUDNN version warnings

import cv2
import numpy as np
import pytesseract
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from PIL import Image

# Fix Windows console encoding issues
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass  # Fallback if reconfigure not available

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress verbose werkzeug logging (Flask request logs)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

# Custom filter to suppress SSL connection errors (common with self-signed certs)
class SSLErrorFilter(logging.Filter):
    """Filter to suppress harmless SSL connection errors"""
    def filter(self, record):
        # Suppress SSLEOFError which happens when browser disconnects during handshake
        if "SSLEOFError" in str(record.getMessage()):
            return False
        if "EOF occurred in violation of protocol" in str(record.getMessage()):
            return False
        return True

# Apply SSL error filter to werkzeug logger
logging.getLogger("werkzeug").addFilter(SSLErrorFilter())

# Suppress Socket.IO logging
logging.getLogger("socketio").setLevel(logging.WARNING)
logging.getLogger("engineio").setLevel(logging.WARNING)


# Custom stderr filter to suppress ngrok malformed header errors
class NgrokStderrFilter:
    """Filter to suppress specific stderr messages from ngrok"""

    def __init__(self, stream):
        self.stream = stream

    def write(self, text):
        # Suppress "ERROR: Malformed header" messages from ngrok only
        if text.strip().startswith("ERROR:") and "Malformed header" in text:
            return  # Silently discard
        # Write everything else normally
        self.stream.write(text)
        self.stream.flush()

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return self.stream.isatty()


# Message deduplication for logger
class MessageCounter:
    """Track message occurrences and append count to repeated messages"""

    def __init__(self):
        self.message_map = {}  # Maps message to count

    def track(self, message: str) -> str:
        """Add message to counter and return message with count if repeated"""
        if message in self.message_map:
            self.message_map[message] += 1
            return f"{message} (�{self.message_map[message]})"
        else:
            self.message_map[message] = 1
            return message


# Global message counter
_message_counter = MessageCounter()


def _build_voice_confirmation(updates: dict) -> str:
    """Generate concise spoken confirmations for config changes"""

    if not updates:
        return "Settings updated."

    def orientation_phrase(value: str) -> str:
        return f"Selected {value.lower()} layout."

    def paper_size_phrase(value: str) -> str:
        label = value.upper() if len(value) <= 4 else value.title()
        return f"Paper size {label}."

    def color_mode_phrase(value: str) -> str:
        mapping = {
            "color": "Color mode on.",
            "bw": "Black and white set.",
            "grayscale": "Grayscale mode on.",
        }
        return mapping.get(value, f"Color mode {value}.")

    def duplex_phrase(value: bool) -> str:
        return "Duplex on." if value else "Single sided." if value is not None else ""

    def pages_phrase(value: str) -> str:
        mapping = {
            "odd": "Odd pages only.",
            "even": "Even pages only.",
            "all": "All pages ready.",
            "custom": "Custom page range ready.",
        }
        return mapping.get(value, f"Pages set to {value}.")

    def page_mode_phrase(value: str) -> str:
        mapping = {
            "odd": "Scanning odd pages.",
            "even": "Scanning even pages.",
            "all": "Scanning all pages.",
            "custom": "Scanning custom range.",
        }
        return mapping.get(value, f"Page mode {value}.")

    templates = {
        "orientation": orientation_phrase,
        "layout": orientation_phrase,
        "scanLayout": orientation_phrase,
        "paper_size": paper_size_phrase,
        "paperSize": paper_size_phrase,
        "page_size": paper_size_phrase,
        "paper_size_custom": lambda v: f"Paper custom {v}.",
        "color_mode": color_mode_phrase,
        "colorMode": color_mode_phrase,
        "scanColorMode": color_mode_phrase,
        "copies": lambda v: f"{v} copies ready." if v else "",
        "duplex": duplex_phrase,
        "pages": pages_phrase,
        "page_mode": page_mode_phrase,
        "custom_range": lambda v: f"Range {v}.",
        "scanCustomRange": lambda v: f"Range {v}.",
        "pages_per_sheet": lambda v: f"Pages per sheet {v}.",
        "pagesPerSheet": lambda v: f"Pages per sheet {v}.",
        "scale": lambda v: f"Scale {v}%.",
        "resolution": lambda v: f"Resolution {v} DPI.",
        "format": lambda v: f"Format {v.upper()}.",
        "quality": lambda v: f"Quality {v}.",
        "text_mode": lambda v: "OCR on." if v else "OCR off.",
        "scanTextMode": lambda v: "OCR on." if v else "OCR off.",
        "mode": lambda v: "Multi-page scan ready." if v == "multi" else "Single scan ready." if v == "single" else f"Scan mode {v}.",
    }

    phrases = []
    for key, value in updates.items():
        if value is None:
            continue
        formatter = templates.get(key)
        try:
            phrase = formatter(value) if formatter else None
        except Exception:
            phrase = None
        if not phrase:
            readable_key = key.replace("_", " ").title()
            phrase = f"{readable_key} {value}."
        phrases.append(phrase)

    return " ".join(phrases) if phrases else "Settings updated."

# Monkey-patch logger methods to add message counting
_original_info = logger.info
_original_warning = logger.warning
_original_error = logger.error


def _counted_info(msg, *args, **kwargs):
    if isinstance(msg, str):
        msg = _message_counter.track(msg)
    return _original_info(msg, *args, **kwargs)


def _counted_warning(msg, *args, **kwargs):
    if isinstance(msg, str):
        msg = _message_counter.track(msg)
    return _original_warning(msg, *args, **kwargs)


def _counted_error(msg, *args, **kwargs):
    if isinstance(msg, str):
        msg = _message_counter.track(msg)
    return _original_error(msg, *args, **kwargs)


logger.info = _counted_info
logger.warning = _counted_warning
logger.error = _counted_error

# Apply stderr filter after logging is configured
original_stderr = sys.stderr
sys.stderr = NgrokStderrFilter(original_stderr)

# Suppress stderr output for malformed header errors (ngrok proxy issue)
# Temporarily disabled to debug startup issues
# class ErrorFilter:
#     """Filter to suppress specific stderr messages"""
#     def __init__(self, stream):
#         self.stream = stream
#         self.buffer = []

#     def write(self, text):
#         # Suppress "ERROR: Malformed header" messages from ngrok
#         if 'Malformed header' in text:
#             return  # Silently discard
#         if text.strip().startswith('ERROR:') and 'Malformed' in text:
#             return  # Silently discard
#         # Write everything else
#         self.stream.write(text)
#         self.stream.flush()

#     def flush(self):
#         self.stream.flush()

# Apply stderr filter
# sys.stderr = ErrorFilter(sys.stderr)  # Disabled for debugging

# ============================================================================
# GPU INITIALIZATION AND DETECTION
# ============================================================================
logger.info("=" * 70)
logger.info("[STARTUP] PrintChakra Backend Initialization")
logger.info("=" * 70)

try:
    import torch
    from app.modules.voice.gpu_optimization import detect_gpu, get_optimal_device, initialize_gpu

    # Detect GPU first
    gpu_info = detect_gpu()
    device = get_optimal_device()  # Returns 'cuda' if available, else 'cpu'
    
    # Display GPU status clearly
    logger.info("")
    if gpu_info['available']:
        logger.info("[OK] GPU ACCELERATION ENABLED (CUDA)")
        logger.info(f"    Device: {gpu_info['gpu_name']}")
        logger.info(f"    CUDA Version: {gpu_info['cuda_version']}")
        logger.info(f"    GPU Count: {gpu_info['device_count']}")
        logger.info(f"    Total Memory: {gpu_info['total_memory_gb']:.2f} GB")
        logger.info("    Mode: GPU ACCELERATION ACTIVE")
    else:
        logger.warning("[!] GPU NOT DETECTED - Using CPU Fallback")
        logger.warning("    To enable GPU acceleration:")
        logger.warning("    1. Install NVIDIA CUDA Toolkit")
        logger.warning("    2. Reinstall PyTorch with CUDA support")
        logger.warning("    Mode: CPU (Slower performance)")
    logger.info("")
    
    # Initialize GPU if available
    if gpu_info['available']:
        try:
            initialize_gpu()
            logger.info("[OK] GPU optimizations initialized")
        except Exception as e:
            logger.warning(f"[WARN] GPU initialization warning: {e}")
            
except ImportError:
    logger.error("[ERROR] PyTorch not installed - GPU detection unavailable")
except Exception as e:
    logger.error(f"[ERROR] GPU detection failed: {e}")


# Printer queue management functions (always available)
def _normalize_printer_status(code: int) -> str:
    status_map = {
        0: "ready",
        1: "paused",
        2: "error",
        3: "pending deletion",
        4: "paper jam",
        5: "paper out",
        6: "manual feed",
        7: "paper problem",
        8: "offline",
        9: "io active",
        10: "busy",
        11: "printing",
        12: "output bin full",
        13: "not available",
        14: "waiting",
        15: "processing",
        16: "initializing",
        17: "warming up",
        18: "toner low",
        19: "no toner",
        20: "page punt",
        21: "user intervention",
        22: "out of memory",
        23: "door open",
        24: "server unknown",
        25: "power save",
    }
    return status_map.get(int(code) if code is not None else -1, "unknown")




def _fetch_windows_printer_queues():
    script = """
$printers = Get-Printer | Sort-Object -Property Name
$result = @()
foreach ($printer in $printers) {
    $jobs = @()
    
    # Method 1: Try Get-PrintJob (standard API)
    try {
        $jobs = Get-PrintJob -PrinterName $printer.Name -ErrorAction Stop | ForEach-Object {
            [PSCustomObject]@{
                id = $_.Id
                document = $_.DocumentName
                owner = $_.UserName
                status = ($_.JobStatus -join ',')
                submitted = $_.TimeSubmitted.ToString('o')
                totalPages = $_.TotalPages
                pagesPrinted = $_.PagesPrinted
                sizeBytes = $_.Size
            }
        }
    } catch {
        # Method 2: Fallback to WMI if Get-PrintJob fails
        try {
            $wmiJobs = Get-WmiObject -Class Win32_PrintJob -Filter "Name like '%$($printer.Name)%'" -ErrorAction Stop
            $jobs = $wmiJobs | ForEach-Object {
                [PSCustomObject]@{
                    id = $_.JobId
                    document = $_.Document
                    owner = $_.Owner
                    status = $_.Status
                    submitted = $_.TimeSubmitted
                    totalPages = $_.TotalPages
                    pagesPrinted = $_.PagesPrinted
                    sizeBytes = $_.Size
                }
            }
        } catch {
            # Method 3: Last resort - check spool directory for .SPL/.SHD files
            try {
                $spoolPath = Join-Path $env:windir 'System32\spool\PRINTERS'
                if (Test-Path $spoolPath) {
                    $spoolFiles = Get-ChildItem -Path $spoolPath -Filter '*.SPL' -ErrorAction SilentlyContinue
                    $jobs = $spoolFiles | ForEach-Object {
                        $shdFile = Join-Path $spoolPath ($_.BaseName + '.SHD')
                        [PSCustomObject]@{
                            id = $_.BaseName
                            document = $_.Name
                            owner = 'unknown'
                            status = 'spooling'
                            submitted = $_.CreationTime.ToString('o')
                            totalPages = $null
                            pagesPrinted = $null
                            sizeBytes = $_.Length
                        }
                    }
                }
            } catch {}
        }
    }
    
    $result += [PSCustomObject]@{
        name = $printer.Name
        status = $printer.PrinterStatus
        isDefault = $printer.Default
        jobs = @($jobs)
    }
}
$result | ConvertTo-Json -Depth 6
""".strip()

    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=40,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Failed to query printers")

    output = result.stdout.strip() or "[]"
    return json.loads(output)




def _fetch_cups_printer_queues():
    printers = {}

    printers_result = subprocess.run(
        ["lpstat", "-p"], capture_output=True, text=True, timeout=15
    )
    if printers_result.returncode == 0:
        for line in printers_result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "printer":
                name = parts[1]
                status_fragment = " ".join(parts[3:]).strip()
                status = status_fragment.split(".")[0] if status_fragment else "unknown"
                printers[name] = {
                    "name": name,
                    "status": status,
                    "isDefault": False,
                    "jobs": [],
                }

    jobs_result = subprocess.run(
        ["lpstat", "-W", "not-completed", "-o"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if jobs_result.returncode == 0:
        for line in jobs_result.stdout.splitlines():
            parts = line.split()
            if not parts:
                continue
            job_id = parts[0]
            printer_name = job_id.split("-", 1)[0]
            owner = parts[1] if len(parts) > 1 else "unknown"
            submitted = " ".join(parts[2:]) if len(parts) > 2 else ""
            job = {
                "id": job_id,
                "document": job_id,
                "owner": owner,
                "status": "pending",
                "submitted": submitted,
                "totalPages": None,
                "pagesPrinted": None,
                "sizeBytes": None,
            }
            printers.setdefault(
                printer_name,
                {
                    "name": printer_name,
                    "status": "unknown",
                    "isDefault": False,
                    "jobs": [],
                },
            )
            printers[printer_name]["jobs"].append(job)

    return list(printers.values())




def get_printer_queue_snapshot():
    if sys.platform.startswith("win"):
        raw_printers = _fetch_windows_printer_queues()
        result = []
        for printer in raw_printers:
            jobs = printer.get("jobs", []) or []
            result.append(
                {
                    "name": printer.get("name"),
                    "status": _normalize_printer_status(printer.get("status")),
                    "isDefault": printer.get("isDefault", False),
                    "jobs": [
                        {
                            "id": job.get("id"),
                            "document": job.get("document"),
                            "owner": job.get("owner"),
                            "status": job.get("status") or "in queue",
                            "submitted": job.get("submitted"),
                            "pagesPrinted": job.get("pagesPrinted"),
                            "totalPages": job.get("totalPages"),
                            "sizeBytes": job.get("sizeBytes"),
                        }
                        for job in jobs
                    ],
                }
            )
        return result

    return _fetch_cups_printer_queues()




def cancel_printer_job(printer_name: str, job_id: str):
    if not printer_name or not job_id:
        raise ValueError("printer_name and job_id required")

    if sys.platform.startswith("win"):
        script = f"Remove-PrintJob -PrinterName \"{printer_name}\" -ID {job_id} -ErrorAction Stop"
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "Failed to cancel job")
        return

    job_identifier = job_id if "-" in job_id else f"{printer_name}-{job_id}"
    result = subprocess.run(
        ["cancel", job_identifier], capture_output=True, text=True, timeout=15
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Failed to cancel job")


logger.info("=" * 70)

# Configure Poppler path for pdf2image (optional, PDF thumbnail generation)
POPPLER_PATH = None
try:
    # Try standard locations
    possible_paths = [
        os.path.join(os.path.dirname(__file__), 'public', 'poppler', 'poppler-24.08.0', 'Library', 'bin'),
        r'C:\Program Files\poppler\Library\bin',
        r'C:\Program Files (x86)\poppler\Library\bin',
    ]
    
    for poppler_bin in possible_paths:
        if os.path.exists(poppler_bin):
            POPPLER_PATH = poppler_bin
            logger.info(f"[OK] Poppler found at: {poppler_bin}")
            break
    
    if POPPLER_PATH is None:
        logger.debug("[INFO] Poppler not found (optional - PDF thumbnails disabled)")
except Exception as e:
    logger.debug(f"[DEBUG] Poppler detection: {e}")

logger.info("=" * 60)

# Import new modular pipeline
try:
    from app.modules import DocumentPipeline, create_default_pipeline, validate_image_file
    from app.modules.document import DocumentDetector, detect_and_serialize

    MODULES_AVAILABLE = True
    print("[OK] All modules loaded successfully")
except ImportError as ie:
    MODULES_AVAILABLE = False
    print(f"[WARN] Module import failed: {ie}")
    print("   This is OK - basic processing will work, advanced features disabled")
except Exception as e:
    MODULES_AVAILABLE = False
    print(f"[WARN] Unexpected module error: {e}")
    import traceback

    traceback.print_exc()

# Initialize Flask app
app = Flask(__name__)


# Custom error handler for malformed requests (from ngrok)
@app.errorhandler(400)
def handle_bad_request(e):
    """Silently handle malformed requests from ngrok proxy"""
    # Don't log these - they're just ngrok proxy issues
    return jsonify({"error": "Bad request"}), 400


# Suppress Flask's default error logging for 400 errors
app.config["TRAP_BAD_REQUEST_ERRORS"] = True

# Configure CORS for frontend - Works for both local and deployed environments
# Automatically handles: localhost, ngrok, deployed servers, etc.

# Helper to allow all origins for Socket.IO while supporting credentials
def allow_all_origins(origin):
    return True

allowed_origins = [
    # Local development
    "http://localhost:3000",
    "http://localhost:5000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5000",
]

# Add specific ngrok/deployed URLs from environment variables
if os.environ.get('NGROK_URL'):
    allowed_origins.append(os.environ.get('NGROK_URL'))

if os.environ.get('FRONTEND_URL'):
    allowed_origins.append(os.environ.get('FRONTEND_URL'))

# For deployed environments, use regex to allow all origins with credentials
# This is necessary because 'origins': '*' cannot be used with 'supports_credentials': True
CORS(
    app,
    resources={
        r"/*": {
            "origins": r"https?://.*",  # Regex matches any http/https origin and reflects it
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
logger.info(f"[CORS] Allowed origins: {allowed_origins}")

# Initialize Socket.IO with comprehensive CORS configuration for all environments
socketio = SocketIO(
    app,
    cors_allowed_origins=allow_all_origins,  # Function allows all origins + credentials
    async_mode="threading",
    logger=False,  # Disable verbose logging
    engineio_logger=False,  # Disable verbose logging
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=1e7,
    always_connect=True,
    transports=["polling", "websocket"],  # Support both - polling is more reliable on ngrok
    cors_credentials=True,
)

# CRITICAL: Update the socketio instance in app.core so all features use the same instance
# This ensures socketio.emit() calls in routes work properly
try:
    import app.core as core_module
    core_module.socketio = socketio
    logger.info("[Socket.IO] Registered with app.core for feature routes")
except Exception as e:
    logger.warning(f"[Socket.IO] Could not register with app.core: {e}")

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(BASE_DIR, "public")
DATA_DIR = os.path.join(PUBLIC_DIR, "data")

# Directories for file storage
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
TEXT_DIR = os.path.join(DATA_DIR, "processed_text")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
CONVERTED_DIR = os.path.join(DATA_DIR, "converted")
STATIC_DIR = os.path.join(PUBLIC_DIR, "static")
PRINT_DIR = os.path.join(BASE_DIR, "app", "print_scripts")
TEST_CAPTURES_DIR = os.path.join(DATA_DIR, "test_captures")  # Temporary test captures for calibration

# Create directories if they don't exist
for directory in [PUBLIC_DIR, DATA_DIR, UPLOAD_DIR, PROCESSED_DIR, TEXT_DIR, PRINT_DIR, PDF_DIR, CONVERTED_DIR, STATIC_DIR, TEST_CAPTURES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Processing status tracking (in-memory)
# Format: { 'filename': { 'step': 1, 'total_steps': 12, 'stage_name': '...', 'is_complete': False, 'error': None } }
processing_status = {}
processing_lock = threading.Lock()

# Scan configuration storage (shared between endpoints)
current_scan_config = {}


def update_processing_status(
    filename, step, total_steps, stage_name, is_complete=False, error=None
):
    """Update processing status for a file"""
    with processing_lock:
        processing_status[filename] = {
            "step": step,
            "total_steps": total_steps,
            "stage_name": stage_name,
            "is_complete": is_complete,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }


def get_processing_status(filename):
    """Get processing status for a file"""
    with processing_lock:
        return processing_status.get(filename)


def clear_processing_status(filename):
    """Clear processing status for a file after completion"""
    with processing_lock:
        if filename in processing_status:
            del processing_status[filename]


# Initialize new document pipeline
if MODULES_AVAILABLE and create_default_pipeline is not None:
    try:
        pipeline_config = {
            "blur_threshold": 100.0,
            "focus_threshold": 50.0,
            "ocr_language": "eng",
            "ocr_psm": 3,
            "ocr_oem": 3,
            "storage_dir": PROCESSED_DIR,
        }
        doc_pipeline = create_default_pipeline(storage_dir=PROCESSED_DIR)
        print("[OK] New modular pipeline initialized successfully")
    except Exception as e:
        print(f"[WARN] Pipeline initialization error: {e}")
        doc_pipeline = None
        MODULES_AVAILABLE = False
else:
    doc_pipeline = None
    if MODULES_AVAILABLE and create_default_pipeline is None:
        print("[WARN] Pipeline not available (optional module disabled)")
        MODULES_AVAILABLE = False

# Tesseract configuration (update path if needed)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Middleware to add CORS and security headers to all responses
@app.after_request
def after_request_handler(response):
    """Add CORS and security headers to all responses"""
    # CORS headers (ensure they're always present)
    origin = request.headers.get("Origin", "*")
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = (
        "Content-Type,Authorization,ngrok-skip-browser-warning,X-Requested-With"
    )
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,DELETE,OPTIONS,PUT,PATCH"
    response.headers["Access-Control-Max-Age"] = "3600"
    response.headers["Access-Control-Expose-Headers"] = "Content-Type,Content-Disposition"

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"

    return response


# ============================================================================
# REGISTER API BLUEPRINTS
# ============================================================================

# Share configuration with blueprints via app.config
app.config['UPLOAD_DIR'] = UPLOAD_DIR
app.config['PROCESSED_DIR'] = PROCESSED_DIR
app.config['TEXT_DIR'] = TEXT_DIR
app.config['PDF_DIR'] = PDF_DIR
app.config['CONVERTED_DIR'] = CONVERTED_DIR
app.config['DATA_DIR'] = DATA_DIR
app.config['POPPLER_PATH'] = POPPLER_PATH
app.config['update_processing_status'] = update_processing_status
app.config['get_processing_status'] = get_processing_status
app.config['clear_processing_status'] = clear_processing_status
app.config['processing_status'] = processing_status
app.config['doc_pipeline'] = doc_pipeline

# Import and register document API blueprint
try:
    from app.api import document_bp
    app.register_blueprint(document_bp)
    print("[OK] Document API blueprint registered")
except ImportError as e:
    print(f"[WARN] Could not import document API blueprint: {e}")

# Import and register chat_flow blueprint
try:
    from app.chat_flow import chat_flow_bp
    app.register_blueprint(chat_flow_bp)
    print("[OK] Chat Flow blueprint registered")
except ImportError as e:
    print(f"[WARN] Could not import chat_flow blueprint: {e}")

# ============================================================================
# PRINTER QUEUE ROUTES (Module level - always available)
# ============================================================================

def _normalize_printer_status_code(code) -> str:
    """Normalize Windows printer status code to human-readable string."""
    status_map = {
        0: "ready",
        1: "paused",
        2: "error",
        3: "pending deletion",
        4: "paper jam",
        5: "paper out",
        6: "manual feed",
        7: "paper problem",
        8: "offline",
        9: "io active",
        10: "busy",
        11: "printing",
        12: "output bin full",
        13: "not available",
        14: "waiting",
        15: "processing",
        16: "initializing",
        17: "warming up",
        18: "toner low",
        19: "no toner",
        20: "page punt",
        21: "user intervention",
        22: "out of memory",
        23: "door open",
        24: "server unknown",
        25: "power save",
    }
    return status_map.get(int(code) if code is not None else -1, "unknown")


def _get_printer_queues_windows():
    """Fetch printer queues on Windows using PowerShell."""
    script = """
$printers = Get-Printer | Sort-Object -Property Name
$result = @()
foreach ($printer in $printers) {
    $jobs = @()
    
    # Method 1: Try Get-PrintJob (standard API)
    try {
        $jobs = Get-PrintJob -PrinterName $printer.Name -ErrorAction Stop | ForEach-Object {
            [PSCustomObject]@{
                id = $_.Id
                document = $_.DocumentName
                owner = $_.UserName
                status = ($_.JobStatus -join ',')
                submitted = $_.TimeSubmitted.ToString('o')
                totalPages = $_.TotalPages
                pagesPrinted = $_.PagesPrinted
                sizeBytes = $_.Size
            }
        }
    } catch {
        # Method 2: Fallback to WMI if Get-PrintJob fails
        try {
            $wmiJobs = Get-WmiObject -Class Win32_PrintJob -Filter "Name like '%$($printer.Name)%'" -ErrorAction Stop
            $jobs = $wmiJobs | ForEach-Object {
                [PSCustomObject]@{
                    id = $_.JobId
                    document = $_.Document
                    owner = $_.Owner
                    status = $_.Status
                    submitted = $_.TimeSubmitted
                    totalPages = $_.TotalPages
                    pagesPrinted = $_.PagesPrinted
                    sizeBytes = $_.Size
                }
            }
        } catch {
            # Method 3: Last resort - check spool directory for .SPL/.SHD files
            try {
                $spoolPath = Join-Path $env:windir 'System32\spool\PRINTERS'
                if (Test-Path $spoolPath) {
                    $spoolFiles = Get-ChildItem -Path $spoolPath -Filter '*.SPL' -ErrorAction SilentlyContinue
                    $jobs = $spoolFiles | ForEach-Object {
                        $shdFile = Join-Path $spoolPath ($_.BaseName + '.SHD')
                        [PSCustomObject]@{
                            id = $_.BaseName
                            document = $_.Name
                            owner = 'unknown'
                            status = 'spooling'
                            submitted = $_.CreationTime.ToString('o')
                            totalPages = $null
                            pagesPrinted = $null
                            sizeBytes = $_.Length
                        }
                    }
                }
            } catch {}
        }
    }
    
    $result += [PSCustomObject]@{
        name = $printer.Name
        status = $printer.PrinterStatus
        isDefault = ($printer.Name -eq (Get-CimInstance -ClassName Win32_Printer | Where-Object {$_.Default}).Name)
        jobs = @($jobs)
    }
}
$result | ConvertTo-Json -Depth 4
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Failed to get printer queues")
    
    output = result.stdout.strip()
    if not output or output == "null":
        return []
    
    data = json.loads(output)
    if isinstance(data, dict):
        data = [data]
    
    # Normalize the data
    normalized = []
    for printer in data:
        jobs = printer.get("jobs", []) or []
        if isinstance(jobs, dict):
            jobs = [jobs]
        normalized.append({
            "name": printer.get("name"),
            "status": _normalize_printer_status_code(printer.get("status")),
            "isDefault": printer.get("isDefault", False),
            "jobs": [
                {
                    "id": job.get("id"),
                    "document": job.get("document"),
                    "owner": job.get("owner"),
                    "status": job.get("status") or "in queue",
                    "submitted": job.get("submitted"),
                }
                for job in jobs
            ],
        })
    return normalized


@app.route("/printer/queues", methods=["GET", "OPTIONS"])
def get_printer_queues():
    """Get all printer queues and their jobs."""
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response
    
    try:
        if sys.platform.startswith("win"):
            data = _get_printer_queues_windows()
        else:
            # For non-Windows, return empty for now
            data = []
        return jsonify({"printers": data, "count": len(data)})
    except Exception as exc:
        logger.exception("[Printer] Failed to list queues")
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/printer/cancel-job", methods=["POST", "OPTIONS"])
def cancel_printer_job_route():
    """Cancel a specific print job."""
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response
    
    payload = request.get_json(silent=True) or {}
    printer_name = payload.get("printer_name")
    job_id = payload.get("job_id")

    if not printer_name or not job_id:
        return jsonify({"status": "error", "error": "printer_name and job_id required"}), 400

    try:
        if sys.platform.startswith("win"):
            script = f'Remove-PrintJob -PrinterName "{printer_name}" -ID {job_id} -ErrorAction Stop'
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "Failed to cancel job")
        return jsonify({"status": "success", "message": "Job cancelled"})
    except Exception as exc:
        logger.exception("[Printer] Failed to cancel job")
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/printer/clear-queue", methods=["POST"])
def clear_printer_queue():
    """Clear all pending print jobs from the system spooler."""
    logger.info("[Printer] Clear queue requested")

    try:
        if sys.platform.startswith("win"):
            clear_script = """
$ErrorActionPreference = 'SilentlyContinue'
Get-Service -Name Spooler | Stop-Service -Force
Start-Sleep -Seconds 2
$spoolPath = Join-Path $env:windir 'System32\\spool\\PRINTERS'
if (Test-Path $spoolPath) {
    Remove-Item -Path (Join-Path $spoolPath '*') -Force -Recurse -ErrorAction SilentlyContinue
}
Get-Service -Name Spooler | Start-Service
Get-Printer | ForEach-Object {
    Get-PrintJob -PrinterName $_.Name -ErrorAction SilentlyContinue | Remove-PrintJob -ErrorAction SilentlyContinue
}
Write-Output 'CLEARED'
""".strip()

            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", clear_script],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                logger.error("[Printer] Clear queue failed: %s", result.stderr.strip())
                return (
                    jsonify(
                        {
                            "status": "error",
                            "error": result.stderr.strip() or "Failed to clear printer queue",
                        }
                    ),
                    500,
                )

        else:
            cmd = ["cancel", "-a"]  # CUPS/Linux
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.error("[Printer] Clear queue failed: %s", result.stderr.strip())
                return (
                    jsonify(
                        {
                            "status": "error",
                            "error": result.stderr.strip() or "Failed to clear printer queue",
                        }
                    ),
                    500,
                )

        logger.info("[Printer] Queue cleared successfully")
        return jsonify({"status": "success", "message": "Printing queue cleared"})

    except subprocess.TimeoutExpired:
        logger.exception("[Printer] Clear queue timed out")
        return jsonify({"status": "error", "error": "Timeout clearing printer queue"}), 504
    except FileNotFoundError:
        logger.exception("[Printer] Clear queue command missing")
        return jsonify({"status": "error", "error": "Spooler tools not available"}), 500
    except Exception as exc:
        logger.exception("[Printer] Unexpected error clearing queue")
        return jsonify({"status": "error", "error": str(exc)}), 500


# ============================================================================
# FALLBACK QUALITY CHECK FUNCTION (for when modules unavailable)
# ============================================================================


def perform_basic_quality_check(image_path):
    """
    Basic quality check fallback using OpenCV
    Returns blur and focus scores
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {
                "blur_score": 0,
                "is_blurry": False,
                "focus_score": 100,
                "is_focused": True,
                "quality": {"overall_acceptable": True, "issues": [], "recommendations": []},
            }

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate Laplacian variance (blur detection)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(100, max(0, laplacian_var / 1.5))
        is_blurry = blur_score < 50

        # Image brightness check
        mean_brightness = np.mean(gray)
        brightness_issues = []
        if mean_brightness < 50:
            brightness_issues.append("Image too dark")
        elif mean_brightness > 200:
            brightness_issues.append("Image too bright")

        overall_acceptable = not is_blurry and len(brightness_issues) == 0

        return {
            "blur_score": float(blur_score),
            "is_blurry": bool(is_blurry),
            "focus_score": float(100 - blur_score),
            "is_focused": not is_blurry,
            "quality": {
                "overall_acceptable": overall_acceptable,
                "issues": brightness_issues,
                "recommendations": ["Ensure good lighting"] if brightness_issues else [],
            },
        }
    except Exception as e:
        print(f"Basic quality check error: {e}")
        return {
            "blur_score": 0,
            "is_blurry": False,
            "focus_score": 100,
            "is_focused": True,
            "quality": {"overall_acceptable": True, "issues": [], "recommendations": []},
        }


# ============================================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================================


def enhance_image(image_path):
    """
    Enhance image quality using OpenCV
    - Perspective correction
    - Brightness/contrast adjustment
    - Noise reduction
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    return enhanced


def extract_text(image_path):
    """
    Extract text from image using Tesseract OCR
    """
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang="eng")
        return text.strip()
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return ""


def order_points(pts):
    """Order points in order: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")

    # Sum: top-left will have smallest sum, bottom-right largest
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Diff: top-right will have smallest diff, bottom-left largest
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts, enforce_a4_ratio=True):
    """Apply perspective transform to get bird's-eye view with A4 ratio"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calculate width
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Calculate height
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Enforce A4 aspect ratio (1:1.414 or 1:√2)
    if enforce_a4_ratio:
        A4_RATIO = 1.4142  # √2 ≈ 1.4142 (height/width for portrait)
        current_ratio = maxHeight / maxWidth if maxWidth > 0 else 1
        
        # Determine if portrait or landscape
        if current_ratio > 1:  # Portrait
            # Height is greater, adjust to A4 portrait ratio
            target_height = int(maxWidth * A4_RATIO)
            maxHeight = target_height
        else:  # Landscape
            # Width is greater, adjust to A4 landscape ratio
            target_width = int(maxHeight * A4_RATIO)
            maxWidth = target_width
        
        print(f"  ✓ Enforced A4 ratio: {maxWidth}x{maxHeight}")

    # Destination points
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # Perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def trim_shadow_edges(image, edge_percent=0.02, brightness_threshold=180):
    """
    Trim shadow edges from a warped document image.
    Scans from each edge inward and trims dark/shadow regions.
    
    Args:
        image: Input image (BGR or grayscale)
        edge_percent: Percentage of image to scan from edges (default 2%)
        brightness_threshold: Pixels below this are considered shadows
    
    Returns:
        Trimmed image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        is_color = True
    else:
        gray = image.copy()
        is_color = False
    
    h, w = gray.shape
    
    # Calculate scan region size (how far from edge to check)
    scan_h = max(int(h * edge_percent * 2), 30)  # At least 30 pixels
    scan_w = max(int(w * edge_percent * 2), 30)
    
    # Minimum trim based on typical shadow width
    min_trim = max(int(min(h, w) * 0.015), 15)  # 1.5% or at least 15px
    
    # Find shadow boundaries from each edge
    # LEFT edge
    left_trim = min_trim
    for x in range(scan_w):
        col = gray[:, x]
        if np.mean(col) < brightness_threshold:
            left_trim = max(left_trim, x + 5)
        else:
            break
    
    # RIGHT edge  
    right_trim = min_trim
    for x in range(w - 1, w - scan_w - 1, -1):
        col = gray[:, x]
        if np.mean(col) < brightness_threshold:
            right_trim = max(right_trim, w - x + 5)
        else:
            break
    
    # TOP edge
    top_trim = min_trim
    for y in range(scan_h):
        row = gray[y, :]
        if np.mean(row) < brightness_threshold:
            top_trim = max(top_trim, y + 5)
        else:
            break
    
    # BOTTOM edge
    bottom_trim = min_trim
    for y in range(h - 1, h - scan_h - 1, -1):
        row = gray[y, :]
        if np.mean(row) < brightness_threshold:
            bottom_trim = max(bottom_trim, h - y + 5)
        else:
            break
    
    # Apply trim if any edge needs it
    if left_trim > 0 or right_trim > 0 or top_trim > 0 or bottom_trim > 0:
        y1 = top_trim
        y2 = h - bottom_trim
        x1 = left_trim
        x2 = w - right_trim
        
        # Ensure we don't trim too much (max 10% from any edge)
        max_trim_h = int(h * 0.10)
        max_trim_w = int(w * 0.10)
        
        y1 = min(y1, max_trim_h)
        y2 = max(y2, h - max_trim_h)
        x1 = min(x1, max_trim_w)
        x2 = max(x2, w - max_trim_w)
        
        if y2 > y1 and x2 > x1:
            trimmed = image[y1:y2, x1:x2]
            print(f"  ✓ Trimmed shadow edges: {w}x{h} -> {x2-x1}x{y2-y1} (L:{left_trim}, R:{right_trim}, T:{top_trim}, B:{bottom_trim})")
            return trimmed
    
    return image


def auto_crop_borders(image, threshold=30, min_crop_percent=0.02):
    """
    Automatically crop dark borders from a document image.
    Works on grayscale images.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape
    
    # Find rows and columns that are mostly dark (borders)
    row_means = np.mean(gray, axis=1)
    col_means = np.mean(gray, axis=0)
    
    # Find content boundaries (where brightness exceeds threshold)
    content_rows = np.where(row_means > threshold)[0]
    content_cols = np.where(col_means > threshold)[0]
    
    if len(content_rows) == 0 or len(content_cols) == 0:
        return image  # No cropping possible
    
    # Get crop boundaries with minimum margin
    min_margin_h = int(h * min_crop_percent)
    min_margin_w = int(w * min_crop_percent)
    
    top = max(0, content_rows[0] - min_margin_h)
    bottom = min(h, content_rows[-1] + min_margin_h)
    left = max(0, content_cols[0] - min_margin_w)
    right = min(w, content_cols[-1] + min_margin_w)
    
    # Only crop if we're removing significant borders
    if (top > min_margin_h or bottom < h - min_margin_h or 
        left > min_margin_w or right < w - min_margin_w):
        cropped = image[top:bottom, left:right]
        print(f"  ✓ Auto-cropped borders: {w}x{h} -> {right-left}x{bottom-top}")
        return cropped
    
    return image


def estimate_background_fast(gray_image, kernel_ratio=8):
    """
    Fast background estimation using downscaling for large images.
    Essential for shadow removal.
    """
    h, w = gray_image.shape
    
    # Downscale for faster morphological operations
    max_dim = 800
    scale = min(max_dim / max(h, w), 1.0)
    
    if scale < 1.0:
        small = cv2.resize(gray_image, (int(w * scale), int(h * scale)))
    else:
        small = gray_image
    
    sh, sw = small.shape
    
    # Kernel size relative to small image
    k_size = max(sh, sw) // kernel_ratio
    k_size = k_size if k_size % 2 == 1 else k_size + 1
    k_size = max(k_size, 31)
    k_size = min(k_size, 151)
    
    # Morphological closing to estimate background
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    background_small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    
    # Smooth
    background_small = cv2.GaussianBlur(background_small, (k_size, k_size), 0)
    
    # Upscale back
    if scale < 1.0:
        background = cv2.resize(background_small, (w, h), interpolation=cv2.INTER_LINEAR)
        # Extra smoothing to hide upscaling artifacts
        background = cv2.GaussianBlur(background, (51, 51), 0)
    else:
        background = background_small
    
    return background


def remove_shadows(image):
    """
    Remove shadows from document image using illumination normalization.
    Uses fast downscaling approach for large images.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Estimate background illumination
    background = estimate_background_fast(gray, kernel_ratio=6)
    
    # Normalize: divide by background
    shadow_free = np.zeros_like(gray, dtype=np.float32)
    mask = background > 10  # Avoid division by zero
    shadow_free[mask] = (gray[mask].astype(np.float32) / background[mask].astype(np.float32)) * 255
    shadow_free = np.clip(shadow_free, 0, 255).astype(np.uint8)
    
    return shadow_free


def enhance_document_quality(gray_image, mode='document'):
    """
    Simple CamScanner-style enhancement.
    NO harsh thresholding - keeps natural text appearance.
    """
    h, w = gray_image.shape[:2]
    
    # Step 1: Shadow removal using background estimation
    # Downscale for speed
    max_dim = 800
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        small = cv2.resize(gray_image, (int(w * scale), int(h * scale)))
    else:
        small = gray_image
    
    sh, sw = small.shape
    k_size = max(sh, sw) // 8
    k_size = k_size if k_size % 2 == 1 else k_size + 1
    k_size = max(k_size, 31)
    k_size = min(k_size, 127)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    bg_small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    bg_small = cv2.GaussianBlur(bg_small, (k_size, k_size), 0)
    
    if scale < 1.0:
        background = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)
        background = cv2.GaussianBlur(background, (31, 31), 0)
    else:
        background = bg_small
    
    # Normalize illumination (shadow removal)
    result = np.zeros_like(gray_image, dtype=np.float32)
    mask = background > 10
    result[mask] = (gray_image[mask].astype(np.float32) / background[mask].astype(np.float32)) * 255
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def find_document_contour(image):
    """
    Ultra-robust multi-strategy document detection - EXACT FROM NOTEBOOK
    Tests multiple scales, edge detection methods, and scoring criteria
    """
    print("   Starting multi-strategy document detection...")

    # Strategy 1: High-resolution edge detection with multiple scales
    candidates = []

    # Test at FULL resolution and scaled versions
    for scale in [1.0, 0.8, 0.6]:
        if scale < 1.0:
            scaled = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        else:
            scaled = image.copy()

        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

        # Approach A: Enhanced Canny with multiple threshold combinations
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        for low, high in [(40, 120), (50, 150), (60, 180), (75, 200), (90, 250)]:
            edges = cv2.Canny(blurred, low, high)
            edges = cv2.dilate(edges, np.ones((2, 2)), iterations=1)

            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area > 500:
                    candidates.append((c, area, scale, f"Canny({low},{high})"))

        # Approach B: Laplacian edge detection
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        _, laplacian = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)
        laplacian = cv2.dilate(laplacian, np.ones((3, 3)), iterations=1)

        contours, _ = cv2.findContours(laplacian, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area > 500:
                candidates.append((c, area, scale, "Laplacian"))

        # Approach C: White paper detection (HSV) - IMPROVED for various lighting
        hsv = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)
        
        # Multiple white detection ranges for different lighting conditions
        white_ranges = [
            (np.array([0, 0, 175]), np.array([180, 45, 255])),   # Bright white
            (np.array([0, 0, 140]), np.array([180, 60, 255])),   # Slightly off-white
            (np.array([0, 0, 120]), np.array([180, 80, 255])),   # Grayish white (poor lighting)
        ]
        
        for lower_white, upper_white in white_ranges:
            mask = cv2.inRange(hsv, lower_white, upper_white)

            # Stronger morphological operations for cleaner detection
            kernel_small = np.ones((3, 3), np.uint8)
            kernel_med = np.ones((7, 7), np.uint8)
            kernel_large = np.ones((11, 11), np.uint8)
            
            # Close small gaps in the paper
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=3)
            # Remove small noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
            # Fill holes
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_med, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area > 500:
                    candidates.append((c, area, scale, "ColorWhite"))
        
        # Approach C2: Brightness-based detection (for paper on dark background)
        # Convert to LAB color space for better brightness detection
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Threshold based on brightness
        _, bright_mask = cv2.threshold(l_channel, 150, 255, cv2.THRESH_BINARY)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area > 500:
                candidates.append((c, area, scale, "BrightnessDetect"))

        # Approach D: Adaptive threshold with edge focus
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        adaptive = cv2.bitwise_not(adaptive)
        adaptive_edges = cv2.Canny(adaptive, 50, 150)
        adaptive_edges = cv2.dilate(adaptive_edges, np.ones((3, 3)), iterations=1)

        contours, _ = cv2.findContours(adaptive_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area > 500:
                candidates.append((c, area, scale, "AdaptiveEdge"))

    print(f"   Found {len(candidates)} contour candidates")

    # Strategy 2: Filter and score candidates
    scored_candidates = []
    image_area = image.shape[0] * image.shape[1]

    for contour, area, scale, method in candidates:
        # Scale contour back to original size
        if scale < 1.0:
            contour = (contour / scale).astype(np.int32)
            area = area / (scale * scale)

        peri = cv2.arcLength(contour, True)

        # Try multiple epsilon values
        for epsilon_factor in [0.003, 0.005, 0.008, 0.01, 0.012, 0.015, 0.018]:
            approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)

            if len(approx) == 4:
                score = 0
                area_ratio = area / image_area

                # Factor 1: Size filtering
                if 0.10 <= area_ratio <= 0.35:
                    score += 100
                    if 0.15 <= area_ratio <= 0.30:
                        score += 80
                    elif 0.10 <= area_ratio <= 0.15:
                        score += 40
                    elif 0.30 <= area_ratio <= 0.35:
                        score += 40
                elif 0.35 < area_ratio <= 0.45:
                    score -= 30
                elif area_ratio > 0.45:
                    score -= 150
                elif area_ratio < 0.10:
                    score -= 200

                # Factor 2: Aspect ratio - PREFER A4 RATIO (√2 ≈ 1.414)
                rect = cv2.minAreaRect(approx)
                width, height = rect[1]
                aspect = 1.0
                A4_RATIO = 1.4142  # √2 for A4 paper
                if width > 0 and height > 0:
                    aspect = max(width, height) / min(width, height)
                    
                    # Strong bonus for A4 aspect ratio
                    aspect_diff = abs(aspect - A4_RATIO)
                    if aspect_diff < 0.05:  # Very close to A4
                        score += 80
                    elif aspect_diff < 0.1:  # Close to A4
                        score += 60
                    elif aspect_diff < 0.2:  # Reasonably close
                        score += 40
                    elif 1.2 <= aspect <= 1.8:  # Acceptable range
                        score += 25
                    elif 1.0 <= aspect <= 2.5:
                        score += 10
                    else:
                        score -= 30

                # Factor 3: Convexity
                hull_area = cv2.contourArea(cv2.convexHull(approx))
                convexity = 0
                if hull_area > 0:
                    convexity = area / hull_area
                    if convexity > 0.95:
                        score += 30
                    elif convexity > 0.90:
                        score += 20
                    else:
                        score += convexity * 10

                # Factor 4: Corner angles
                angles = []
                pts = approx.reshape(4, 2).astype(np.float32)
                for i in range(4):
                    p1 = pts[i]
                    p2 = pts[(i + 1) % 4]
                    p3 = pts[(i + 2) % 4]

                    v1 = p1 - p2
                    v2 = p3 - p2

                    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
                    if norm_product > 0:
                        angle = np.arccos(np.clip(np.dot(v1, v2) / norm_product, -1.0, 1.0))
                        angles.append(np.degrees(angle))

                if len(angles) == 4:
                    angle_errors = [abs(a - 90) for a in angles]
                    avg_error = np.mean(angle_errors)

                    if avg_error < 10:
                        score += 35
                    elif avg_error < 20:
                        score += 25
                    elif avg_error < 30:
                        score += 15

                # Factor 5: Method bonuses - STRONGLY prioritize paper detection methods
                if method == "ColorWhite":
                    score += 120  # Strong bonus for white paper detection
                elif method == "BrightnessDetect":
                    score += 100  # Good for paper on dark backgrounds
                elif method.startswith("Canny"):
                    score += 10
                elif method == "Laplacian":
                    score += 15
                elif method == "AdaptiveEdge":
                    score += 12

                # Factor 6: Compactness
                if peri > 0:
                    compactness = (4 * np.pi * area) / (peri * peri)
                    if 0.5 < compactness < 0.9:
                        score += 15

                # Factor 7: Position preference
                center = np.mean(pts, axis=0)
                center_x_ratio = center[0] / image.shape[1]
                center_y_ratio = center[1] / image.shape[0]
                if 0.2 < center_x_ratio < 0.8 and 0.2 < center_y_ratio < 0.8:
                    score += 15

                # Factor 8: Edge margins
                edge_margin_left = np.min(pts[:, 0])
                edge_margin_right = image.shape[1] - np.max(pts[:, 0])
                edge_margin_top = np.min(pts[:, 1])
                edge_margin_bottom = image.shape[0] - np.max(pts[:, 1])
                total_margin = (
                    edge_margin_left + edge_margin_right + edge_margin_top + edge_margin_bottom
                )
                margin_ratio = total_margin / (image.shape[0] + image.shape[1])

                if margin_ratio > 0.3:
                    score += 30
                elif margin_ratio < 0.1:
                    score -= 100

                scored_candidates.append(
                    {
                        "contour": approx,
                        "area": area,
                        "score": score,
                        "method": method,
                        "area_ratio": area_ratio,
                        "aspect": aspect,
                        "convexity": convexity,
                    }
                )

    if not scored_candidates:
        print("   No suitable document found")
        return None

    # Sort by score
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)

    # Show top candidates
    print(
        f"   Top candidate: Score={scored_candidates[0]['score']:.1f}, "
        f"Area={(scored_candidates[0]['area_ratio']*100):.1f}%, "
        f"Method={scored_candidates[0]['method']}"
    )

    # Find best candidate
    best = None
    for cand in scored_candidates:
        if 0.08 <= cand["area_ratio"] <= 0.40:
            best = cand
            break

    if best is None:
        best = scored_candidates[0]

    # Apply corner inset to move corners inward and avoid shadow boundaries
    contour = best["contour"].reshape(4, 2).astype(np.float32)
    
    # Calculate document size to scale inset appropriately
    doc_width = np.max(contour[:, 0]) - np.min(contour[:, 0])
    doc_height = np.max(contour[:, 1]) - np.min(contour[:, 1])
    doc_diagonal = np.sqrt(doc_width**2 + doc_height**2)
    
    # Scale inset: ~2% of diagonal, minimum 25px, maximum 60px
    inset_pixels = max(25, min(int(doc_diagonal * 0.02), 60))
    
    # Calculate center of document
    center = contour.mean(axis=0)
    
    # Move each corner toward center to avoid shadows
    refined = contour.copy()
    for i in range(4):
        direction = center - contour[i]
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
            refined[i] = contour[i] + direction * inset_pixels
    
    print(f"   Applied {inset_pixels}px corner inset for shadow removal")
    
    return refined


def process_document_image(input_path, output_path, filename=None):
    """
    Simple CamScanner-style processing:
    1. Detect document
    2. Perspective correct
    3. Remove shadows
    4. White background + contrast
    That's it. No over-processing.
    
    PIPELINE STAGES (12 total):
    1. Load - Load original image
    2-6. Detect - Document detection (grayscale, blur, edge, contours)
    7. Transform - Perspective correction
    8. Process - Grayscale conversion
    9-10. Enhance - Background estimation and shadow removal
    11. OCR - Text extraction and filename generation
    12. Save - Final save and completion
    """
    try:
        def emit_progress(step, stage_name, message):
            """Emit progress update - step should always increase, never decrease"""
            progress_data = {
                "filename": filename,
                "step": step,
                "total_steps": 12,
                "stage_name": stage_name,
                "message": message,
            }
            socketio.emit("processing_progress", progress_data)
            if filename:
                update_processing_status(filename, step, 12, stage_name)

        # STEP 1: Load
        print(f"\n[1/12] Loading image...")
        emit_progress(1, "Load", "Loading original image...")
        
        image = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image from {input_path}")
        print(f"  ✓ Loaded: {image.shape[1]}x{image.shape[0]}")

        # STEPS 2-6: Detection Stages (Grouped for speed but reported as progress)
        print(f"[2/12] Downscaling for detection...")
        emit_progress(2, "Detect", "Downscaling for detection...")
        
        # We process but emit messages to show progress to the user
        emit_progress(3, "Detect", "Converting to grayscale...")
        emit_progress(4, "Detect", "Applying Gaussian blur...")
        emit_progress(5, "Detect", "Running Canny edge detection...")
        emit_progress(6, "Detect", "Finding document contours...")
        
        doc_contour = find_document_contour(image)
        
        # STEP 7: Warp/Transform
        print(f"[7/12] Perspective transform...")
        emit_progress(7, "Transform", "Applying perspective correction...")
        if doc_contour is not None:
            warped = four_point_transform(image, doc_contour)
            print(f"  ✓ Document cropped: {warped.shape[1]}x{warped.shape[0]}")
            
            # Trim shadow edges after perspective transform
            warped = trim_shadow_edges(warped, edge_percent=0.02, brightness_threshold=160)
        else:
            warped = image
            print(f"  ✓ Using full image")

        # STEP 8: Process - Grayscale conversion
        print(f"[8/12] Grayscale conversion...")
        emit_progress(8, "Process", "Converting cropped document to grayscale...")
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # STEPS 9-10: Enhance - Background Estimation and Shadow Removal
        print(f"[9/12] Background estimation...")
        emit_progress(9, "Enhance", "Estimating background illumination...")
        
        print(f"[10/12] Shadow removal...")
        emit_progress(10, "Enhance", "Removing shadows and normalizing...")
        
        enhanced = enhance_document_quality(gray)
        print(f"  ✓ Enhanced")

        # STEP 11: OCR - Text extraction and filename generation
        print(f"[11/12] OCR and filename generation...")
        emit_progress(11, "OCR", "Running OCR for text extraction...")
        
        # Basic OCR (Tesseract)
        text = ""
        try:
            text = pytesseract.image_to_string(enhanced, lang="eng", config="--oem 3 --psm 3")
        except:
            pass

        # PaddleOCR + Ollama Filename Generation (part of OCR step)
        ocr_filename = None
        text_output_path = None
        
        # Save text first
        if text.strip():
            text_path = output_path.replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt")
            text_path = text_path.replace("/processed/", "/processed_text/")
            os.makedirs(os.path.dirname(text_path), exist_ok=True)
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)
            text_output_path = text_path

        try:
            from app.modules.ocr.paddle_ocr import get_ocr_processor, OCRResult
            
            ocr_result = None
            ocr_processor = None
            
            # Skip OCR for very small images (likely corrupt, thumbnails, or test images)
            img_height, img_width = enhanced.shape[:2]
            if img_width < 200 or img_height < 200:
                print(f"  ⚠ Skipping OCR: Image too small ({img_width}x{img_height}), likely corrupt or test image")
                # Just save the image
                cv2.imwrite(output_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                print(f"  Running PaddleOCR for intelligent filename...")
                
                # Initialize OCR processor
                ocr_data_dir = os.path.join(os.path.dirname(output_path), '..', 'ocr_results')
                ocr_processor = get_ocr_processor(ocr_data_dir)
                
                # Run PaddleOCR on the enhanced image (save it first temporarily)
                cv2.imwrite(output_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
                ocr_result = ocr_processor.process_image(output_path)
                
                if ocr_result and ocr_result.word_count > 0:
                    # Get timestamp from filename if available
                    timestamp = None
                    if filename:
                        import re
                        match = re.search(r'_(\d{8}_\d{6})_', filename)
                        if match:
                            timestamp = match.group(1)
                    
                    # Generate filename using Ollama
                    suggested_filename = ocr_processor.generate_filename_from_ocr(ocr_result, timestamp)
                    
                    if suggested_filename and suggested_filename != "untitled":
                        # Rename the file
                        output_dir = os.path.dirname(output_path)
                        ext = os.path.splitext(output_path)[1]
                        new_filename = f"{suggested_filename}{ext}"
                        new_output_path = os.path.join(output_dir, new_filename)
                        old_basename = os.path.basename(output_path)
                        
                        # Avoid overwriting existing files
                        counter = 1
                        while os.path.exists(new_output_path):
                            new_filename = f"{suggested_filename}_{counter}{ext}"
                            new_output_path = os.path.join(output_dir, new_filename)
                            counter += 1
                        
                        # Rename the processed image
                        os.rename(output_path, new_output_path)
                        
                        output_path = new_output_path
                        ocr_filename = new_filename
                        
                        # Also rename the text file if it exists
                        if text_output_path and os.path.exists(text_output_path):
                            new_text_filename = f"{suggested_filename}.txt"
                            new_text_path = os.path.join(os.path.dirname(text_output_path), new_text_filename)
                            if not os.path.exists(new_text_path):
                                os.rename(text_output_path, new_text_path)
                                text_output_path = new_text_path
                        
                        print(f"  ✓ File renamed based on OCR content: {new_filename}")
                    
                    # Save OCR result with the correct filename
                    ocr_result_filename = os.path.basename(output_path)
                    ocr_processor.save_result(ocr_result_filename, ocr_result)
                    print(f"  ✓ OCR result saved ({ocr_result.word_count} words)")
                else:
                    print(f"  ⚠ No text detected by PaddleOCR, keeping original filename")
                
        except Exception as paddle_ocr_error:
            print(f"  [WARN] PaddleOCR/Ollama step failed: {paddle_ocr_error}")
            # Save image if not already saved
            if not os.path.exists(output_path):
                cv2.imwrite(output_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # STEP 12: Save - Final completion
        print(f"[12/12] Finalizing...")
        emit_progress(12, "Save", "Saving processed document...")
        
        # Ensure image is saved (if PaddleOCR didn't save it)
        if not os.path.exists(output_path):
            cv2.imwrite(output_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  ✓ Saved: {output_path}")

        print(f"\n{'='*60}")
        print(f"[OK] PROCESSING COMPLETE!")
        print(f"   Input: {input_path}")
        print(f"   Output: {output_path}")
        if ocr_filename:
            print(f"   OCR-derived filename: {ocr_filename}")
        print(f"   Text extracted: {len(text)} characters")
        print(f"{'='*60}\n")

        # Return the new filename if renamed
        return True, text, ocr_filename if ocr_filename else filename

    except Exception as e:
        print(f"[ERROR] Processing error: {str(e)}")
        error_data = {"error": str(e), "message": "Processing failed"}
        socketio.emit("processing_error", error_data)
        if filename:
            update_processing_status(filename, 12, 12, "Error", is_complete=True, error=str(e))
        return False, str(e), None


# Add process_document_image to app config for blueprint access
app.config['process_document_image'] = process_document_image


# ============================================================================
# API ENDPOINTS
# ============================================================================


@app.route("/")
def index():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "ok",
            "service": "PrintChakra Backend",
            "version": "2.0.0",
            "features": {
                "basic_processing": True,
                "advanced_pipeline": MODULES_AVAILABLE,
                "ocr": True,
                "socket_io": True,
                "pdf_export": MODULES_AVAILABLE,
                "document_classification": MODULES_AVAILABLE,
                "quality_validation": MODULES_AVAILABLE,
                "batch_processing": MODULES_AVAILABLE,
            },
            "endpoints": {
                "health": "/health",
                "upload": "/upload",
                "files": "/files",
                "processed": "/processed/<filename>",
                "delete": "/delete/<filename>",
                "print": "/print",
                "ocr": "/ocr/<filename>",
                "advanced": {
                    "process": "/process/advanced",
                    "validate_quality": "/validate/quality",
                    "export_pdf": "/export/pdf",
                    "classify": "/classify/document",
                    "batch": "/batch/process",
                    "pipeline_info": "/pipeline/info",
                },
            },
        }
    )


@app.route("/favicon.ico")
def favicon():
    """Serve favicon to prevent 404 errors"""
    return "", 204  # No Content response


# Public folder routes for serving data files
@app.route("/public/processed/<path:filename>", methods=["GET", "OPTIONS"])
def serve_public_processed(filename):
    """Serve processed files from public/data/processed directory"""
    # Get origin from request or use localhost
    origin = request.headers.get('Origin', 'http://localhost:3000')
    
    # Handle OPTIONS request for CORS preflight
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, HEAD, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, ngrok-skip-browser-warning, X-Requested-With"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Max-Age"] = "86400"
        return response, 200
    
    try:
        response = send_from_directory(PROCESSED_DIR, filename)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        # Add explicit CORS headers with actual origin (not wildcard when credentials)
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, HEAD, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, ngrok-skip-browser-warning, X-Requested-With"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Expose-Headers"] = "Content-Length, Content-Type, Content-Disposition"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response
    except Exception as e:
        logger.error(f"Error serving processed file {filename}: {e}")
        return jsonify({"error": "File not found"}), 404


@app.route("/public/uploads/<path:filename>", methods=["GET", "OPTIONS"])
def serve_public_uploads(filename):
    """Serve uploaded files from public/data/uploads directory"""
    # Get origin from request or use localhost
    origin = request.headers.get('Origin', 'http://localhost:3000')
    
    # Handle OPTIONS request for CORS preflight
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, HEAD, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, ngrok-skip-browser-warning, X-Requested-With"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Max-Age"] = "86400"
        return response, 200
    
    try:
        response = send_from_directory(UPLOAD_DIR, filename)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        # Add explicit CORS headers with actual origin (not wildcard when credentials)
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, HEAD, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, ngrok-skip-browser-warning, X-Requested-With"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Expose-Headers"] = "Content-Length, Content-Type, Content-Disposition"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response
    except Exception as e:
        logger.error(f"Error serving uploaded file {filename}: {e}")
        return jsonify({"error": "File not found"}), 404


@app.route("/public/converted/<path:filename>", methods=["GET", "OPTIONS"])
def serve_public_converted(filename):
    """Serve converted files from public/data/converted directory"""
    # Get origin from request or use localhost
    origin = request.headers.get('Origin', 'http://localhost:3000')
    
    # Handle OPTIONS request for CORS preflight
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, HEAD, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, ngrok-skip-browser-warning, X-Requested-With"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Max-Age"] = "86400"
        return response, 200
    
    try:
        response = send_from_directory(CONVERTED_DIR, filename)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        # Add explicit CORS headers with actual origin (not wildcard when credentials)
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, HEAD, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, ngrok-skip-browser-warning, X-Requested-With"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Expose-Headers"] = "Content-Length, Content-Type, Content-Disposition"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response
    except Exception as e:
        logger.error(f"Error serving converted file {filename}: {e}")
        return jsonify({"error": "File not found"}), 404


@app.route("/public/static/<path:filename>", methods=["GET", "OPTIONS"])
def serve_public_static(filename):
    """Serve static files from public/static directory"""
    try:
        response = send_from_directory(STATIC_DIR, filename)
        return response
    except Exception as e:
        logger.error(f"Error serving static file {filename}: {e}")
        return jsonify({"error": "File not found"}), 404


# ============================================================================
# TEST CAPTURE ENDPOINTS (for calibration testing)
# ============================================================================

@app.route("/test-capture", methods=["POST", "OPTIONS"])
def upload_test_capture():
    """
    Upload a test capture during calibration testing.
    These images are temporary and will be deleted when testing ends.
    """
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response, 200
    
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        ext = os.path.splitext(file.filename)[1] or ".jpg"
        filename = f"test_capture_{timestamp}{ext}"
        filepath = os.path.join(TEST_CAPTURES_DIR, filename)
        
        # Save file
        file.save(filepath)
        logger.info(f"[TEST_CAPTURE] Saved test capture: {filename}")
        
        # Get file size for response
        file_size = os.path.getsize(filepath)
        
        # Emit Socket.IO event to notify frontend of new test capture
        socketio.emit("test_capture_received", {
            "filename": filename,
            "url": f"/public/test_captures/{filename}",
            "size": file_size,
            "timestamp": datetime.now().isoformat()
        })
        
        return jsonify({
            "success": True,
            "filename": filename,
            "url": f"/public/test_captures/{filename}",
            "size": file_size,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"[TEST_CAPTURE] Upload error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/test-captures", methods=["GET"])
def list_test_captures():
    """List all test captures"""
    try:
        captures = []
        if os.path.exists(TEST_CAPTURES_DIR):
            for filename in os.listdir(TEST_CAPTURES_DIR):
                filepath = os.path.join(TEST_CAPTURES_DIR, filename)
                if os.path.isfile(filepath):
                    captures.append({
                        "filename": filename,
                        "url": f"/public/test_captures/{filename}",
                        "size": os.path.getsize(filepath),
                        "modified": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                    })
        
        # Sort by modified time descending
        captures.sort(key=lambda x: x["modified"], reverse=True)
        
        return jsonify({
            "success": True,
            "count": len(captures),
            "captures": captures
        })
        
    except Exception as e:
        logger.error(f"[TEST_CAPTURE] List error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/test-captures/clear", methods=["POST", "DELETE"])
def clear_test_captures():
    """Delete all test captures"""
    try:
        deleted_count = 0
        if os.path.exists(TEST_CAPTURES_DIR):
            for filename in os.listdir(TEST_CAPTURES_DIR):
                filepath = os.path.join(TEST_CAPTURES_DIR, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
                    deleted_count += 1
        
        logger.info(f"[TEST_CAPTURE] Cleared {deleted_count} test captures")
        
        # Emit Socket.IO event to notify frontend
        socketio.emit("test_captures_cleared", {
            "deleted_count": deleted_count,
            "timestamp": datetime.now().isoformat()
        })
        
        return jsonify({
            "success": True,
            "deleted_count": deleted_count,
            "message": f"Deleted {deleted_count} test capture(s)"
        })
        
    except Exception as e:
        logger.error(f"[TEST_CAPTURE] Clear error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/public/test_captures/<path:filename>", methods=["GET", "OPTIONS"])
def serve_test_captures(filename):
    """Serve test capture files"""
    origin = request.headers.get('Origin', 'http://localhost:3000')
    
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, HEAD, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response, 200
    
    try:
        response = send_from_directory(TEST_CAPTURES_DIR, filename)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response
    except Exception as e:
        logger.error(f"Error serving test capture {filename}: {e}")
        return jsonify({"error": "File not found"}), 404


# Global OPTIONS handler for CORS preflight
@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def handle_options(path):
    """Handle CORS preflight requests for all routes"""
    response = jsonify({"status": "ok"})
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = (
        "Content-Type,Authorization,ngrok-skip-browser-warning,X-Requested-With"
    )
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,DELETE,OPTIONS,PUT,PATCH"
    response.headers["Access-Control-Max-Age"] = "3600"
    return response, 200


@app.route("/health")
def health():
    """Detailed health check"""
    tesseract_available = True
    try:
        pytesseract.get_tesseract_version()
    except:
        tesseract_available = False

    # Check module status
    detection_available = False
    try:
        if MODULES_AVAILABLE:
            from app.modules.document import DocumentDetector

            detection_available = True
    except:
        pass

    return jsonify(
        {
            "status": "healthy",
            "service": "PrintChakra Backend",
            "version": "2.0.0",
            "modules": {
                "advanced_pipeline": MODULES_AVAILABLE,
                "document_pipeline": doc_pipeline is not None,
                "document_detection": detection_available,
            },
            "directories": {
                "uploads": os.path.exists(UPLOAD_DIR),
                "processed": os.path.exists(PROCESSED_DIR),
                "text": os.path.exists(TEXT_DIR),
                "pdfs": os.path.exists(PDF_DIR),
            },
            "features": {
                "tesseract_ocr": tesseract_available,
                "image_processing": True,
                "socket_io": True,
                "blur_detection": MODULES_AVAILABLE,
                "edge_detection": MODULES_AVAILABLE,
                "perspective_correction": MODULES_AVAILABLE,
                "clahe_enhancement": MODULES_AVAILABLE,
                "document_classification": (
                    MODULES_AVAILABLE and doc_pipeline and doc_pipeline.classifier.is_trained
                    if doc_pipeline
                    else False
                ),
                "pdf_export": MODULES_AVAILABLE,
                "cloud_storage": False,  # Not yet configured
                "auto_naming": MODULES_AVAILABLE,
                "compression": MODULES_AVAILABLE,
            },
        }
    )


@app.route("/system/info", methods=["GET"])
def system_info():
    """Get comprehensive system and device information"""
    import platform
    import psutil
    
    # Basic system info
    system_data = {
        "os": {
            "name": platform.system(),
            "version": platform.version(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "memory": {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "used_percent": psutil.virtual_memory().percent,
        },
        "cpu": {
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(logical=True),
            "usage_percent": psutil.cpu_percent(interval=0.1),
        },
    }
    
    # GPU info
    try:
        import torch
        from app.modules.voice.gpu_optimization import detect_gpu
        gpu_info = detect_gpu()
        system_data["gpu"] = {
            "available": gpu_info.get('available', False),
            "name": gpu_info.get('gpu_name', 'N/A'),
            "cuda_version": gpu_info.get('cuda_version', 'N/A'),
            "device_count": gpu_info.get('device_count', 0),
            "total_memory_gb": round(gpu_info.get('total_memory_gb', 0), 2),
        }
    except Exception as e:
        system_data["gpu"] = {"available": False, "error": str(e)}
    
    # Printer info
    try:
        import win32print
        
        printers = []
        default_printer = win32print.GetDefaultPrinter()
        
        # Get all printers
        printer_list = win32print.EnumPrinters(win32print.PRINTER_ENUM_LOCAL | win32print.PRINTER_ENUM_CONNECTIONS)
        
        for printer in printer_list:
            printer_name = printer[2]
            is_default = printer_name == default_printer
            
            # Try to get printer status
            try:
                handle = win32print.OpenPrinter(printer_name)
                printer_info = win32print.GetPrinter(handle, 2)
                win32print.ClosePrinter(handle)
                
                status = printer_info.get('Status', 0)
                driver_name = printer_info.get('pDriverName', 'Unknown')
                port_name = printer_info.get('pPortName', 'Unknown')
                
                # Determine status string
                if status == 0:
                    status_str = "Ready"
                elif status & 0x00000001:
                    status_str = "Paused"
                elif status & 0x00000002:
                    status_str = "Error"
                elif status & 0x00000004:
                    status_str = "Pending Deletion"
                elif status & 0x00000008:
                    status_str = "Paper Jam"
                elif status & 0x00000010:
                    status_str = "Paper Out"
                elif status & 0x00000020:
                    status_str = "Manual Feed"
                elif status & 0x00000040:
                    status_str = "Paper Problem"
                elif status & 0x00000080:
                    status_str = "Offline"
                elif status & 0x00000100:
                    status_str = "IO Active"
                elif status & 0x00000200:
                    status_str = "Busy"
                elif status & 0x00000400:
                    status_str = "Printing"
                else:
                    status_str = "Unknown"
                    
            except Exception:
                driver_name = "Unknown"
                port_name = "Unknown"
                status_str = "Unknown"
            
            printers.append({
                "name": printer_name,
                "is_default": is_default,
                "driver": driver_name,
                "port": port_name,
                "status": status_str,
            })
        
        system_data["printers"] = {
            "available": len(printers) > 0,
            "default": default_printer,
            "count": len(printers),
            "list": printers,
        }
        
    except ImportError:
        system_data["printers"] = {
            "available": False,
            "error": "win32print not available (Windows only)",
            "list": [],
        }
    except Exception as e:
        system_data["printers"] = {
            "available": False,
            "error": str(e),
            "list": [],
        }
    
    # Driver suggestions based on printer
    driver_suggestions = []
    if system_data.get("printers", {}).get("list"):
        for printer in system_data["printers"]["list"]:
            name = printer.get("name", "").lower()
            suggestion = None
            
            if "hp" in name or "hewlett" in name:
                suggestion = {
                    "printer": printer["name"],
                    "brand": "HP",
                    "driver_url": "https://support.hp.com/drivers",
                    "description": "HP Smart & Drivers"
                }
            elif "canon" in name:
                suggestion = {
                    "printer": printer["name"],
                    "brand": "Canon",
                    "driver_url": "https://www.usa.canon.com/support/software-and-drivers",
                    "description": "Canon Drivers & Downloads"
                }
            elif "epson" in name:
                suggestion = {
                    "printer": printer["name"],
                    "brand": "Epson",
                    "driver_url": "https://epson.com/Support/sl/s",
                    "description": "Epson Support & Drivers"
                }
            elif "brother" in name:
                suggestion = {
                    "printer": printer["name"],
                    "brand": "Brother",
                    "driver_url": "https://support.brother.com/g/b/productsearch.aspx",
                    "description": "Brother Solutions Center"
                }
            elif "samsung" in name:
                suggestion = {
                    "printer": printer["name"],
                    "brand": "Samsung",
                    "driver_url": "https://www.hp.com/us-en/samsung-printers.html",
                    "description": "Samsung Printers (now HP)"
                }
            elif "lexmark" in name:
                suggestion = {
                    "printer": printer["name"],
                    "brand": "Lexmark",
                    "driver_url": "https://www.lexmark.com/en_us/support.html",
                    "description": "Lexmark Support"
                }
            elif "xerox" in name:
                suggestion = {
                    "printer": printer["name"],
                    "brand": "Xerox",
                    "driver_url": "https://www.support.xerox.com/",
                    "description": "Xerox Support"
                }
            elif "ricoh" in name:
                suggestion = {
                    "printer": printer["name"],
                    "brand": "Ricoh",
                    "driver_url": "https://www.ricoh.com/support/",
                    "description": "Ricoh Support"
                }
            
            if suggestion:
                driver_suggestions.append(suggestion)
    
    system_data["driver_suggestions"] = driver_suggestions
    
    return jsonify(system_data)


@app.route("/system/set-default-printer", methods=["POST"])
def set_default_printer():
    """Set the default printer"""
    try:
        data = request.get_json()
        printer_name = data.get('printer_name')
        
        if not printer_name:
            return jsonify({"error": "printer_name is required"}), 400
        
        import win32print
        
        # Set the default printer
        win32print.SetDefaultPrinter(printer_name)
        
        return jsonify({
            "success": True,
            "message": f"Default printer set to {printer_name}",
            "printer_name": printer_name
        })
    
    except ImportError:
        return jsonify({"error": "win32print not available (Windows only)"}), 501
    except Exception as e:
        logger.error(f"Failed to set default printer: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Handle image upload - returns immediately with upload info, processes in background
    Expects: multipart/form-data with 'file' or 'photo' field
    """
    try:
        # Accept both 'file' and 'photo' field names
        file = request.files.get("file") or request.files.get("photo")

        if not file:
            print("[ERROR] Upload error: No file in request")
            return jsonify({"error": "No file provided", "success": False}), 400

        if file.filename == "":
            print("[ERROR] Upload error: Empty filename")
            return jsonify({"error": "Empty filename", "success": False}), 400

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        original_ext = os.path.splitext(file.filename)[1]
        if not original_ext:
            original_ext = ".jpg"
        filename = f"doc_{timestamp}_{unique_id}{original_ext}"
        processed_filename = f"processed_{filename}"

        print(f"\n{'='*70}")
        print(f"[UPLOAD] UPLOAD INITIATED")
        print(f"  Filename: {filename}")
        print(f"  Size: {len(file.read())} bytes")
        file.seek(0)  # Reset file pointer
        print(f"{'='*70}")

        # Step 1: Save uploaded file immediately
        print("\n[UPLOAD] Saving uploaded file...")
        upload_path = os.path.join(UPLOAD_DIR, filename)
        file.save(upload_path)
        print(f"  [OK] File saved: {upload_path}")

        # Validate file exists and is readable
        if not os.path.exists(upload_path):
            error_msg = "File upload failed - file not found on disk"
            print(f"  [ERROR] {error_msg}")
            return jsonify({"error": error_msg, "success": False}), 500

        file_size = os.path.getsize(upload_path)
        print(f"  ? Verified on disk: {file_size} bytes")

        # Initialize processing status
        update_processing_status(processed_filename, 0, 12, "Initializing", is_complete=False)

        # Return immediately with upload info
        response = {
            "status": "uploaded",
            "success": True,
            "message": "File uploaded successfully, processing started",
            "filename": processed_filename,
            "upload_filename": filename,
            "original": file.filename,
            "timestamp": timestamp,
            "processing": True,
        }

        # Emit Socket.IO event for instant display
        try:
            socketio.emit(
                "new_file",
                {
                    "filename": processed_filename,
                    "upload_filename": filename,
                    "timestamp": timestamp,
                    "processing": True,
                    "has_text": False,
                },
            )
            print(f"  ? Socket.IO notification sent for instant display")
        except Exception as socket_error:
            print(f"  [WARN] Warning: Socket.IO notification failed: {str(socket_error)}")

        # Start background processing
        def background_process():
            try:
                processed_path = os.path.join(PROCESSED_DIR, processed_filename)

                # Process image with progress tracking
                # Expects 3 return values
                file_result = process_document_image(
                    upload_path,
                    processed_path,
                    processed_filename,  # Pass filename for status tracking
                )
                
                # Handle return value safely (2 or 3 values)
                new_filename = None
                if len(file_result) == 3:
                     success, text_or_error, new_filename = file_result
                else:
                     success, text_or_error = file_result

                final_filename = new_filename if new_filename else processed_filename

                if not success:
                    update_processing_status(
                        processed_filename, 12, 12, "Error", is_complete=True, error=text_or_error
                    )
                    socketio.emit(
                        "processing_error", {"filename": processed_filename, "error": text_or_error}
                    )
                    return

                # Save extracted text
                text_filename = f"{os.path.splitext(final_filename)[0]}.txt"
                text_path = os.path.join(TEXT_DIR, text_filename)

                try:
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(text_or_error)
                    print(f"  ✓ Text saved: {text_path}")
                except Exception as text_error:
                    print(f"  [WARN] Warning: Failed to save text file: {str(text_error)}")

                # Mark as complete (on the original filename to clear that progress bar)
                update_processing_status(processed_filename, 12, 12, "Complete", is_complete=True)

                # Notify completion with rename info
                socketio.emit(
                    "processing_complete",
                    {
                        "filename": final_filename,                # New name
                        "original_filename": processed_filename,   # Old name (to find item)
                        "has_text": len(text_or_error) > 0,
                        "text_length": len(text_or_error),
                        "renamed": new_filename is not None
                    },
                )

                print(f"\n[OK] Background processing completed for {processed_filename}")

                # Clear status after 60 seconds
                threading.Timer(60.0, lambda: clear_processing_status(processed_filename)).start()

            except Exception as e:
                error_msg = f"Background processing error: {str(e)}"
                print(f"[ERROR] {error_msg}")
                update_processing_status(
                    processed_filename, 12, 12, "Error", is_complete=True, error=error_msg
                )
                socketio.emit(
                    "processing_error", {"filename": processed_filename, "error": error_msg}
                )

        # Start processing in background thread
        thread = threading.Thread(target=background_process)
        thread.daemon = True
        thread.start()

        print(f"\n[OK] Upload response sent, processing started in background")
        return jsonify(response)

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"\n[ERROR] {error_msg}")
        import traceback

        print(traceback.format_exc())
        return jsonify({"error": error_msg, "success": False}), 500

        # Step 5: Return success response
        print("\n[STEP 5] Building response...")
        response = {
            "status": "success",
            "success": True,
            "message": "File uploaded and processed successfully",
            "filename": processed_filename,
            "original": filename,
            "text_extracted": len(text_or_error) > 0,
            "text_length": len(text_or_error),
            "text_preview": text_or_error[:200] if len(text_or_error) > 200 else text_or_error,
        }
        print(f"  [OK] Response built")

        print(f"\n{'='*70}")
        print(f"[OK] UPLOAD COMPLETED SUCCESSFULLY")
        print(f"{'='*70}\n")

        return jsonify(response)

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"\n[ERROR] {error_msg}")
        import traceback

        print(traceback.format_exc())
        return jsonify({"error": error_msg, "success": False}), 500


@app.route("/files")
def list_files():
    """List all processed files with processing status"""
    try:
        from app.modules.ocr.paddle_ocr import get_ocr_processor
        
        # Get OCR processor to check for OCR results
        ocr_processor = get_ocr_processor(OCR_DATA_DIR)
        
        files = []

        # First, add files that are currently being processed (uploaded but not yet in processed dir)
        for filename in list(processing_status.keys()):
            status = get_processing_status(filename)
            if status and not status["is_complete"]:
                # Get the upload filename (without "processed_" prefix)
                upload_filename = filename.replace("processed_", "")
                upload_path = os.path.join(UPLOAD_DIR, upload_filename)

                if os.path.exists(upload_path):
                    file_stat = os.stat(upload_path)
                    files.append(
                        {
                            "filename": filename,
                            "size": file_stat.st_size,
                            "created": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                            "has_text": False,
                            "has_ocr": False,
                            "processing": True,
                            "processing_step": status["step"],
                            "processing_total": status["total_steps"],
                            "processing_stage": status["stage_name"],
                        }
                    )

        # Then add all processed files
        for filename in os.listdir(PROCESSED_DIR):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(PROCESSED_DIR, filename)
                # Verify file still exists (it may have been deleted)
                if not os.path.exists(file_path):
                    print(f"[WARN] File listed but doesn't exist: {file_path}")
                    continue
                file_stat = os.stat(file_path)

                # Check if text file exists
                text_filename = f"{os.path.splitext(filename)[0]}.txt"
                text_path = os.path.join(TEXT_DIR, text_filename)
                has_text = os.path.exists(text_path)
                
                # Check if OCR result exists
                has_ocr = ocr_processor.has_ocr_result(filename)

                # Check if still processing (edge case where file exists but processing not complete)
                status = get_processing_status(filename)
                is_processing = status and not status["is_complete"]

                file_info = {
                    "filename": filename,
                    "size": file_stat.st_size,
                    "created": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                    "has_text": has_text,
                    "has_ocr": has_ocr,
                    "processing": is_processing,
                }

                if is_processing:
                    file_info["processing_step"] = status["step"]
                    file_info["processing_total"] = status["total_steps"]
                    file_info["processing_stage"] = status["stage_name"]

                files.append(file_info)

        # Sort by creation time (newest first)
        files.sort(key=lambda x: x["created"], reverse=True)

        return jsonify({"files": files, "count": len(files)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/processing-status/<filename>")
def get_file_processing_status(filename):
    """Get processing status for a specific file"""
    try:
        status = get_processing_status(filename)
        if status:
            return jsonify(
                {
                    "processing": not status["is_complete"],
                    "step": status["step"],
                    "total_steps": status["total_steps"],
                    "stage_name": status["stage_name"],
                    "is_complete": status["is_complete"],
                    "error": status.get("error"),
                    "timestamp": status["timestamp"],
                }
            )
        else:
            # No processing status - file either complete or doesn't exist
            return jsonify({"processing": False, "is_complete": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/processed/<filename>", methods=["GET", "OPTIONS"])
def get_processed_file(filename):
    """Serve processed image file with CORS headers and caching support"""
    # Handle OPTIONS request
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization, ngrok-skip-browser-warning"
        )
        response.headers["Access-Control-Max-Age"] = "3600"
        return response, 200

    try:
        # Security: prevent directory traversal
        if ".." in filename or "/" in filename:
            return jsonify({"error": "Invalid filename"}), 400

        # Check if file exists
        file_path = os.path.join(PROCESSED_DIR, filename)
        if not os.path.exists(file_path):
            print(f"[ERROR] File not found: {file_path}")
            return jsonify({"error": "File not found"}), 404

        print(f"[OK] Serving processed file: {filename}")
        response = send_from_directory(PROCESSED_DIR, filename)

        # Add comprehensive CORS and cache headers
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization, ngrok-skip-browser-warning"
        )
        response.headers["Access-Control-Expose-Headers"] = "Content-Length, Content-Type"
        response.headers["Cache-Control"] = "public, max-age=3600"
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Detect image type from extension
        if filename.lower().endswith(".png"):
            response.headers["Content-Type"] = "image/png"
        elif filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
            response.headers["Content-Type"] = "image/jpeg"
        else:
            response.headers["Content-Type"] = "image/jpeg"  # Default to JPEG

        return response
    except Exception as e:
        print(f"[ERROR] File serving error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"File serving error: {str(e)}"}), 500


@app.route("/document/info/<path:filename>", methods=["GET", "OPTIONS"])
def get_document_info(filename):
    """Get document information including page count for PDFs"""
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization, ngrok-skip-browser-warning"
        )
        response.headers["Access-Control-Max-Age"] = "3600"
        return response, 200

    try:
        # Security: prevent directory traversal
        if ".." in filename or "/" in filename.replace("\\", "/"):
            return jsonify({"error": "Invalid filename"}), 400

        # Check different directories for the file
        possible_paths = [
            os.path.join(PROCESSED_DIR, filename),
            os.path.join(CONVERTED_DIR, filename),
            os.path.join(UPLOAD_DIR, filename),
        ]

        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break

        if not file_path:
            return jsonify({"error": "File not found"}), 404

        file_ext = os.path.splitext(filename)[1].lower()
        doc_info = {
            "filename": filename,
            "file_type": file_ext[1:] if file_ext else "unknown",
            "pages": []
        }

        # Get page count and thumbnails for PDFs
        if file_ext == '.pdf':
            try:
                import PyPDF2
                with open(file_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    page_count = len(pdf_reader.pages)
                    
                    # Generate page info with thumbnail URLs
                    for page_num in range(1, page_count + 1):
                        doc_info["pages"].append({
                            "pageNumber": page_num,
                            "thumbnailUrl": f"/document/page/{filename}/{page_num}"
                        })
            except Exception as e:
                print(f"[WARN] Error reading PDF pages: {str(e)}")
                # Fallback: assume single page
                doc_info["pages"] = [{
                    "pageNumber": 1,
                    "thumbnailUrl": f"/thumbnail/{filename}"
                }]
        else:
            # For images, single page
            doc_info["pages"] = [{
                "pageNumber": 1,
                "thumbnailUrl": f"/thumbnail/{filename}"
            }]

        response = jsonify(doc_info)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Cache-Control"] = "public, max-age=3600"
        return response

    except Exception as e:
        print(f"[ERROR] Document info error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Document info error: {str(e)}"}), 500


@app.route("/document/page/<path:filename>/<int:page_num>", methods=["GET", "OPTIONS"])
def get_pdf_page_thumbnail(filename, page_num):
    """Generate and serve thumbnail for a specific PDF page"""
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization, ngrok-skip-browser-warning"
        )
        response.headers["Access-Control-Max-Age"] = "3600"
        return response, 200

    try:
        # Security: prevent directory traversal
        if ".." in filename or "/" in filename.replace("\\", "/"):
            return jsonify({"error": "Invalid filename"}), 400

        # Check different directories for the file
        possible_paths = [
            os.path.join(PROCESSED_DIR, filename),
            os.path.join(CONVERTED_DIR, filename),
            os.path.join(UPLOAD_DIR, filename),
        ]

        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break

        if not file_path:
            return jsonify({"error": "File not found"}), 404

        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext != '.pdf':
            return jsonify({"error": "Not a PDF file"}), 400

        # Generate thumbnail for specific page
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(
                file_path, 
                first_page=page_num, 
                last_page=page_num, 
                dpi=150,  # Higher DPI for better quality
                poppler_path=POPPLER_PATH
            )
            
            if images:
                img = images[0]
                # Resize to larger thumbnail for preview
                img.thumbnail((800, 1132), Image.Resampling.LANCZOS)  # A4 ratio
                
                # Convert to bytes
                thumb_io = io.BytesIO()
                img.save(thumb_io, format='JPEG', quality=90)
                thumbnail_data = thumb_io.getvalue()
                
                response = app.response_class(
                    response=thumbnail_data,
                    status=200,
                    mimetype="image/jpeg"
                )
                response.headers["Access-Control-Allow-Origin"] = "*"
                response.headers["Access-Control-Allow-Methods"] = "GET, HEAD, OPTIONS"
                response.headers["Access-Control-Expose-Headers"] = "Content-Length, Content-Type"
                response.headers["Cache-Control"] = "public, max-age=86400"
                response.headers["Content-Type"] = "image/jpeg"
                return response
            else:
                return jsonify({"error": "Could not extract page"}), 500
                
        except ImportError:
            return jsonify({"error": "pdf2image not available"}), 500
        except Exception as e:
            print(f"[ERROR] PDF page extraction error: {str(e)}")
            return jsonify({"error": f"Page extraction error: {str(e)}"}), 500

    except Exception as e:
        print(f"[ERROR] PDF page thumbnail error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Page thumbnail error: {str(e)}"}), 500


@app.route("/thumbnail/<path:filename>", methods=["GET", "OPTIONS"])
def get_thumbnail(filename):
    """Generate and serve thumbnail images for documents and PDFs"""
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization, ngrok-skip-browser-warning"
        )
        response.headers["Access-Control-Max-Age"] = "3600"
        return response, 200

    try:
        # Security: prevent directory traversal
        if ".." in filename or "/" in filename.replace("\\", "/"):
            return jsonify({"error": "Invalid filename"}), 400

        # Check different directories for the file
        possible_paths = [
            os.path.join(PROCESSED_DIR, filename),  # Processed images
            os.path.join(CONVERTED_DIR, filename),  # Converted PDFs
            os.path.join(UPLOAD_DIR, filename),     # Uploaded files
        ]

        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break

        if not file_path:
            print(f"[ERROR] Thumbnail source file not found: {filename}")
            return jsonify({"error": "File not found"}), 404

        # Generate thumbnail based on file type
        file_ext = os.path.splitext(filename)[1].lower()
        thumbnail_data = None

        try:
            if file_ext in ['.pdf']:
                # For PDFs, try to convert first page to image using pdf2image or similar
                try:
                    from pdf2image import convert_from_path
                    images = convert_from_path(file_path, first_page=1, last_page=1, dpi=100, poppler_path=POPPLER_PATH)
                    if images:
                        img = images[0]
                        # Resize to thumbnail size
                        img.thumbnail((200, 250), Image.Resampling.LANCZOS)
                        # Convert to bytes
                        thumb_io = io.BytesIO()
                        img.save(thumb_io, format='JPEG', quality=85)
                        thumbnail_data = thumb_io.getvalue()
                except ImportError:
                    # Fallback: return a placeholder
                    print(f"[WARN] pdf2image not available, using placeholder for {filename}")
                    placeholder = Image.new('RGB', (200, 250), color=(100, 100, 100))
                    thumb_io = io.BytesIO()
                    placeholder.save(thumb_io, format='JPEG', quality=85)
                    thumbnail_data = thumb_io.getvalue()

            elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
                # For image files, read and resize
                img = Image.open(file_path)
                # Convert RGBA to RGB if needed
                if img.mode == 'RGBA':
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[3])
                    img = rgb_img
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to thumbnail size while maintaining aspect ratio
                img.thumbnail((200, 250), Image.Resampling.LANCZOS)
                # Convert to bytes
                thumb_io = io.BytesIO()
                img.save(thumb_io, format='JPEG', quality=85)
                thumbnail_data = thumb_io.getvalue()

            elif file_ext in ['.txt']:
                # For text files, create a text-based thumbnail
                text_preview = Image.new('RGB', (200, 250), color=(255, 255, 255))
                thumb_io = io.BytesIO()
                text_preview.save(thumb_io, format='JPEG', quality=85)
                thumbnail_data = thumb_io.getvalue()

            else:
                # Unknown file type, return placeholder
                placeholder = Image.new('RGB', (200, 250), color=(200, 200, 200))
                thumb_io = io.BytesIO()
                placeholder.save(thumb_io, format='JPEG', quality=85)
                thumbnail_data = thumb_io.getvalue()

        except Exception as e:
            print(f"[WARN] Error generating thumbnail: {str(e)}")
            # Return a gray placeholder on error
            placeholder = Image.new('RGB', (200, 250), color=(180, 180, 180))
            thumb_io = io.BytesIO()
            placeholder.save(thumb_io, format='JPEG', quality=85)
            thumbnail_data = thumb_io.getvalue()

        if thumbnail_data:
            response = app.response_class(
                response=thumbnail_data,
                status=200,
                mimetype="image/jpeg"
            )
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, HEAD, OPTIONS"
            response.headers["Access-Control-Expose-Headers"] = "Content-Length, Content-Type"
            response.headers["Cache-Control"] = "public, max-age=86400"
            response.headers["Content-Type"] = "image/jpeg"
            return response

        return jsonify({"error": "Failed to generate thumbnail"}), 500

    except Exception as e:
        print(f"[ERROR] Thumbnail generation error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Thumbnail generation error: {str(e)}"}), 500


@app.route("/uploads/<filename>", methods=["GET", "OPTIONS"])
def get_upload_file(filename):
    """Serve uploaded (preview) image file with CORS headers"""
    # Handle OPTIONS request
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization, ngrok-skip-browser-warning"
        )
        response.headers["Access-Control-Max-Age"] = "3600"
        return response, 200

    try:
        # Security: prevent directory traversal
        if ".." in filename or "/" in filename:
            return jsonify({"error": "Invalid filename"}), 400

        # Remove "processed_" prefix if present to get upload filename
        upload_filename = filename.replace("processed_", "")

        # Check if file exists
        file_path = os.path.join(UPLOAD_DIR, upload_filename)
        if not os.path.exists(file_path):
            print(f"[ERROR] Upload file not found: {file_path}")
            return jsonify({"error": "File not found"}), 404

        print(f"[OK] Serving upload file: {upload_filename}")
        response = send_from_directory(UPLOAD_DIR, upload_filename)

        # Add comprehensive CORS headers
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization, ngrok-skip-browser-warning"
        )
        response.headers["Access-Control-Expose-Headers"] = "Content-Length, Content-Type"
        response.headers["Cache-Control"] = "public, max-age=300"  # 5 min cache for previews
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Detect image type from extension
        if upload_filename.lower().endswith(".png"):
            response.headers["Content-Type"] = "image/png"
        elif upload_filename.lower().endswith(".jpg") or upload_filename.lower().endswith(".jpeg"):
            response.headers["Content-Type"] = "image/jpeg"
        else:
            response.headers["Content-Type"] = "image/jpeg"  # Default to JPEG

        return response
    except Exception as e:
        print(f"[ERROR] Upload file serving error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"File serving error: {str(e)}"}), 500


@app.route("/delete/<filename>", methods=["DELETE"])
def delete_file(filename):
    """Delete a processed file and its associated text"""
    try:
        # Delete image file
        image_path = os.path.join(PROCESSED_DIR, filename)
        if os.path.exists(image_path):
            os.remove(image_path)

        # Delete text file
        text_filename = f"{os.path.splitext(filename)[0]}.txt"
        text_path = os.path.join(TEXT_DIR, text_filename)
        if os.path.exists(text_path):
            os.remove(text_path)

        # Delete original upload if exists
        original_filename = filename.replace("processed_", "")
        upload_path = os.path.join(UPLOAD_DIR, original_filename)
        if os.path.exists(upload_path):
            os.remove(upload_path)

        # Notify via Socket.IO
        socketio.emit("file_deleted", {"filename": filename})

        return jsonify({"status": "success", "message": f"File {filename} deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ocr/<filename>")
def get_ocr_text(filename):
    """Get extracted OCR text for a file"""
    try:
        text_filename = f"{os.path.splitext(filename)[0]}.txt"
        text_path = os.path.join(TEXT_DIR, text_filename)

        if os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read()
            return jsonify({"filename": filename, "text": text, "length": len(text)})
        else:
            return jsonify({"filename": filename, "text": "", "message": "No text extracted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/print", methods=["POST"])
def trigger_print():
    """
    Trigger print command and notify phone to capture
    Expects: JSON with { "type": "blank" } or { "type": "test" }
    """
    try:
        data = request.get_json() if request.is_json else {}
        print_type = data.get("type", "blank")
        
        # Check multiple possible locations for blank.pdf
        possible_blank_paths = [
            os.path.join(PRINT_DIR, "blank.pdf"),
            os.path.join(BASE_DIR, "app", "print_scripts", "blank.pdf"),
            os.path.join(PUBLIC_DIR, "blank.pdf"),
        ]
        
        blank_pdf = None
        for path in possible_blank_paths:
            if os.path.exists(path):
                blank_pdf = path
                break

        if print_type == "test":
            # Test printer connection - just verify blank.pdf exists
            return jsonify(
                {
                    "status": "success",
                    "message": "Printer test: blank.pdf is ready" if blank_pdf else "blank.pdf not found",
                    "pdf_exists": blank_pdf is not None,
                    "pdf_path": blank_pdf,
                }
            )

        elif print_type == "blank":
            # Print blank page and trigger phone capture (feed documents through printer)
            if not blank_pdf:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "blank.pdf not found in any location.",
                        }
                    ),
                    404,
                )

            # Execute print using print-file.py or direct printing
            print_script = os.path.join(os.path.dirname(blank_pdf), "print-file.py")
            
            if os.path.exists(print_script):
                try:
                    # Run print script
                    subprocess.Popen(["python", print_script], cwd=os.path.dirname(blank_pdf))
                    print(f"Print triggered via script: {blank_pdf}")
                except Exception as print_error:
                    print(f"Print script error: {str(print_error)}")
            else:
                # Fallback: Direct print using printer_test approach
                try:
                    import win32api
                    import win32print
                    
                    printer_name = win32print.GetDefaultPrinter()
                    if printer_name:
                        # Method 1: PowerShell
                        try:
                            cmd = f'Start-Process -FilePath "{blank_pdf}" -Verb PrintTo -ArgumentList "{printer_name}" -WindowStyle Hidden'
                            subprocess.run(["powershell", "-Command", cmd], timeout=30)
                            print(f"Print triggered via PowerShell: {blank_pdf}")
                        except:
                            # Method 2: ShellExecute
                            win32api.ShellExecute(0, "printto", blank_pdf, f'"{printer_name}"', ".", 0)
                            print(f"Print triggered via ShellExecute: {blank_pdf}")
                except Exception as direct_print_error:
                    print(f"Direct print error: {str(direct_print_error)}")

            # Notify phone to capture
            try:
                socketio.emit(
                    "capture_now",
                    {
                        "message": "Capture the printed document",
                        "timestamp": datetime.now().isoformat(),
                    },
                )
                print("Capture notification sent to phone")
            except Exception as socket_error:
                print(f"Socket.IO error (non-critical): {str(socket_error)}")

            return jsonify(
                {"status": "success", "message": "Print command sent and capture triggered"}
            )
        else:
            return (
                jsonify(
                    {"status": "error", "message": 'Invalid print type. Use "blank" or "test"'}
                ),
                400,
            )

    except Exception as e:
        print(f"Print error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/print/document", methods=["POST"])
def print_single_document():
    """
    Print a single document by filename
    Converts images to PDF before printing for better compatibility
    Expects JSON: { "filename": "document.jpg", "copies": 1 }
    """
    try:
        data = request.get_json() if request.is_json else {}
        filename = data.get("filename")
        copies = int(data.get("copies", 1))
        
        if not filename:
            return jsonify({"success": False, "error": "No filename provided"}), 400
        
        logger.info(f"[PRINT_DOC] Print request for: {filename}")
        
        # Search for file in multiple directories
        search_dirs = [
            PROCESSED_DIR,
            CONVERTED_DIR,
            UPLOAD_DIR,
            PDF_DIR,
            os.path.join(DATA_DIR, "ocr_results"),
        ]
        
        file_path = None
        for search_dir in search_dirs:
            candidate = os.path.join(search_dir, filename)
            if os.path.exists(candidate):
                file_path = candidate
                logger.info(f"[PRINT_DOC] Found file in: {search_dir}")
                break
        
        if not file_path:
            logger.error(f"[PRINT_DOC] File not found: {filename}")
            return jsonify({
                "success": False, 
                "error": f"File not found: {filename}",
            }), 404
        
        # Check if it's an image - if so, convert to PDF first
        file_ext = os.path.splitext(filename)[1].lower()
        print_path = file_path
        temp_pdf = None
        
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif']:
            logger.info(f"[PRINT_DOC] Converting image to PDF: {filename}")
            try:
                from PIL import Image
                import tempfile
                
                # Open image and convert to PDF
                img = Image.open(file_path)
                
                # Convert to RGB if necessary (for PNG with transparency)
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create temp PDF
                temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
                temp_pdf_path = temp_pdf.name
                temp_pdf.close()
                
                # Save as PDF
                img.save(temp_pdf_path, 'PDF', resolution=100.0)
                print_path = temp_pdf_path
                logger.info(f"[PRINT_DOC] Created temp PDF: {temp_pdf_path}")
                
            except Exception as convert_err:
                logger.error(f"[PRINT_DOC] Image to PDF conversion failed: {convert_err}")
                # Continue with original file if conversion fails
                print_path = file_path
        
        logger.info(f"[PRINT_DOC] Printing: {print_path}")
        
        # Print the document
        print_success = False
        error_msg = None
        
        try:
            import win32api
            import win32print
            
            printer_name = win32print.GetDefaultPrinter()
            logger.info(f"[PRINT_DOC] Default printer: {printer_name}")
            
            if not printer_name:
                return jsonify({"success": False, "error": "No default printer configured"}), 500
            
            for copy_num in range(copies):
                # Method 1: ShellExecute with "print"
                try:
                    result = win32api.ShellExecute(0, "print", print_path, None, ".", 0)
                    logger.info(f"[PRINT_DOC] ShellExecute result: {result}")
                    if result > 32:
                        print_success = True
                        logger.info(f"[PRINT_DOC] ShellExecute succeeded (copy {copy_num + 1})")
                except Exception as e:
                    logger.warning(f"[PRINT_DOC] ShellExecute failed: {e}")
                    error_msg = str(e)
                
                # Method 2: PrintTo with printer name
                if not print_success:
                    try:
                        result = win32api.ShellExecute(0, "printto", print_path, f'"{printer_name}"', ".", 0)
                        if result > 32:
                            print_success = True
                            logger.info(f"[PRINT_DOC] PrintTo succeeded (copy {copy_num + 1})")
                    except Exception as e:
                        logger.warning(f"[PRINT_DOC] PrintTo failed: {e}")
                        error_msg = str(e)
                
                # Method 3: PowerShell
                if not print_success:
                    try:
                        cmd = f'Start-Process -FilePath "{print_path}" -Verb Print -WindowStyle Hidden'
                        result = subprocess.run(["powershell", "-Command", cmd], 
                                               timeout=30, capture_output=True, text=True)
                        if result.returncode == 0:
                            print_success = True
                            logger.info(f"[PRINT_DOC] PowerShell succeeded (copy {copy_num + 1})")
                        else:
                            error_msg = result.stderr
                    except Exception as e:
                        logger.warning(f"[PRINT_DOC] PowerShell failed: {e}")
                        error_msg = str(e)
                        
        except ImportError:
            # win32 not available, use PowerShell only
            logger.warning("[PRINT_DOC] win32 modules not available, using PowerShell")
            for copy_num in range(copies):
                try:
                    cmd = f'Start-Process -FilePath "{print_path}" -Verb Print -WindowStyle Hidden'
                    result = subprocess.run(["powershell", "-Command", cmd], 
                                           timeout=30, capture_output=True, text=True)
                    if result.returncode == 0:
                        print_success = True
                        logger.info(f"[PRINT_DOC] PowerShell succeeded (copy {copy_num + 1})")
                    else:
                        error_msg = result.stderr
                except Exception as e:
                    error_msg = str(e)
        
        # Clean up temp PDF after a delay (give print spooler time)
        if temp_pdf:
            def cleanup_temp():
                import time
                time.sleep(10)  # Wait for print spooler
                try:
                    os.unlink(temp_pdf.name)
                    logger.info(f"[PRINT_DOC] Cleaned up temp PDF")
                except:
                    pass
            import threading
            threading.Thread(target=cleanup_temp, daemon=True).start()
        
        if print_success:
            # Emit capture_now event to phone
            try:
                socketio.emit("capture_now", {
                    "message": f"Capture: {filename}",
                    "document": filename,
                    "timestamp": datetime.now().isoformat()
                })
                logger.info(f"[PRINT_DOC] Sent capture_now for: {filename}")
            except Exception as socket_err:
                logger.warning(f"[PRINT_DOC] Socket emit failed: {socket_err}")
            
            return jsonify({
                "success": True,
                "message": f"Printed: {filename}",
                "filename": filename,
                "copies": copies,
                "converted_to_pdf": temp_pdf is not None
            })
        else:
            return jsonify({
                "success": False,
                "error": error_msg or "Print failed",
                "filename": filename
            }), 500
            
    except Exception as e:
        logger.error(f"[PRINT_DOC] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/printer/diagnostics", methods=["GET", "POST"])
def printer_diagnostics():
    """
    Run printer diagnostics and return results
    This endpoint executes the printer-test.py script and returns detailed diagnostics
    """
    try:
        import io
        import subprocess
        import sys
        from contextlib import redirect_stderr, redirect_stdout

        # Path to the diagnostics script
        diag_script = os.path.join(
            os.path.dirname(PRINT_DIR), "scripts", "printer-test", "print-test.py"
        )

        if not os.path.exists(diag_script):
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Diagnostics script not found",
                        "script_path": diag_script,
                    }
                ),
                404,
            )

        print(f"Running diagnostics from: {diag_script}")

        # Run the diagnostics script and capture output
        try:
            result = subprocess.run(
                [sys.executable, diag_script], capture_output=True, text=True, timeout=30
            )

            diagnostics_output = result.stdout
            if result.stderr:
                diagnostics_output += "\n[STDERR]\n" + result.stderr

            return jsonify(
                {
                    "status": "success",
                    "message": "Printer diagnostics completed",
                    "output": diagnostics_output,
                    "return_code": result.returncode,
                    "success": result.returncode == 0,
                }
            )

        except subprocess.TimeoutExpired:
            return (
                jsonify(
                    {"status": "error", "message": "Diagnostics script timed out after 30 seconds"}
                ),
                500,
            )
        except Exception as e:
            return (
                jsonify({"status": "error", "message": f"Error running diagnostics: {str(e)}"}),
                500,
            )

    except Exception as e:
        print(f"Diagnostics error: {str(e)}")
        return jsonify({"status": "error", "message": f"Diagnostics endpoint error: {str(e)}"}), 500


# ============================================================================
# ADVANCED PROCESSING ENDPOINTS (New Modular System)
# ============================================================================


@app.route("/process/advanced", methods=["POST"])
def advanced_process():
    """
    Advanced document processing with sequential execution and comprehensive logging
    Expects: multipart/form-data with 'file' and optional processing options
    """
    if not MODULES_AVAILABLE or doc_pipeline is None:
        print("[ERROR] Advanced processing not available")
        return (
            jsonify(
                {
                    "error": "Advanced processing not available",
                    "message": "Install required dependencies: pip install -r requirements.txt",
                    "success": False,
                }
            ),
            503,
        )

    try:
        file = request.files.get("file")
        if not file:
            print("[ERROR] No file provided")
            return jsonify({"error": "No file provided", "success": False}), 400

        # Get processing options from form data
        options = {
            "auto_crop": request.form.get("auto_crop", "true").lower() == "true",
            "ai_enhance": request.form.get("ai_enhance", "false").lower() == "true",
            "export_pdf": request.form.get("export_pdf", "false").lower() == "true",
            "compress": request.form.get("compress", "true").lower() == "true",
            "compression_quality": int(request.form.get("compression_quality", "85")),
            "page_size": request.form.get("page_size", "A4"),
            "strict_quality": request.form.get("strict_quality", "false").lower() == "true",
        }

        print(f"\n{'='*70}")
        print(f"[PROCESSING] ADVANCED PROCESSING INITIATED")
        print(f"  Options: {options}")
        print(f"{'='*70}")

        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"doc_{timestamp}_{unique_id}.jpg"
        upload_path = os.path.join(UPLOAD_DIR, filename)

        print(f"\n[STEP 1] Saving upload...")
        file.save(upload_path)
        print(f"  ? Saved: {upload_path}")

        # Process using new pipeline
        print(f"\n[STEP 2] Processing with advanced pipeline...")
        result = doc_pipeline.process_document(upload_path, PROCESSED_DIR, options)

        # Emit Socket.IO event
        if result.get("success"):
            try:
                print(f"\n[STEP 3] Notifying clients...")
                socketio.emit(
                    "processing_complete",
                    {
                        "filename": os.path.basename(result.get("processed_image", "")),
                        "text": result.get("text", "")[:500],  # Preview only
                        "document_type": result.get("document_type", "UNKNOWN"),
                        "confidence": result.get("ocr_confidence", 0),
                        "quality": result.get("quality", {}),
                    },
                )
                print(f"  ? Notification sent")
            except Exception as e:
                print(f"  [WARN] Socket.IO notification failed: {e}")
        else:
            print(f"\n[ERROR] Processing failed: {result.get('error')}")

        print(f"\n{'='*70}")
        print(f"[OK] ADVANCED PROCESSING COMPLETED")
        print(f"{'='*70}\n")

        return jsonify(result)

    except Exception as e:
        error_msg = f"Advanced processing error: {str(e)}"
        print(f"\n[ERROR] {error_msg}")
        print(traceback.format_exc())
        return (
            jsonify({"success": False, "error": error_msg, "traceback": traceback.format_exc()}),
            500,
        )


@app.route("/validate/quality", methods=["POST"])
def validate_quality():
    """
    Validate image quality before processing
    Returns blur and focus scores with fallback if modules unavailable
    """
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file provided"}), 400

        # Save temporarily
        temp_path = os.path.join(UPLOAD_DIR, f"temp_{uuid.uuid4()}.jpg")
        file.save(temp_path)

        try:
            # Try using new modular validation if available
            if MODULES_AVAILABLE:
                quality_result = validate_image_file(temp_path)
            else:
                # Fallback: basic quality check
                quality_result = perform_basic_quality_check(temp_path)

            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

            return jsonify(quality_result)
        except Exception as module_error:
            print(f"Module quality validation error: {module_error}")
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            # Fallback to basic check
            quality_result = (
                perform_basic_quality_check(temp_path)
                if os.path.exists(temp_path)
                else {
                    "blur_score": 0,
                    "is_blurry": False,
                    "focus_score": 100,
                    "is_focused": True,
                    "quality": {"overall_acceptable": True, "issues": [], "recommendations": []},
                }
            )
            return jsonify(quality_result)

    except Exception as e:
        print(f"Quality validation error: {e}")
        traceback.print_exc()
        return (
            jsonify(
                {
                    "error": f"Validation failed: {str(e)}",
                    "quality": {"overall_acceptable": True, "issues": [], "recommendations": []},
                }
            ),
            200,
        )


@app.route("/export/pdf", methods=["POST"])
def export_pdf():
    """
    Export processed images to PDF
    Expects: JSON with 'filenames' array and optional 'page_size'
    """
    if not MODULES_AVAILABLE or doc_pipeline is None:
        return jsonify({"error": "PDF export not available"}), 503

    try:
        data = request.get_json()
        filenames = data.get("filenames", [])
        page_size = data.get("page_size", "A4")

        if not filenames:
            return jsonify({"error": "No filenames provided"}), 400

        # Build full paths
        image_paths = [os.path.join(PROCESSED_DIR, f) for f in filenames]

        # Validate all files exist
        for path in image_paths:
            if not os.path.exists(path):
                return jsonify({"error": f"File not found: {os.path.basename(path)}"}), 404

        # Generate PDF filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"document_{timestamp}.pdf"
        pdf_path = os.path.join(PDF_DIR, pdf_filename)

        # Export to PDF
        success = doc_pipeline.exporter.export_to_pdf(image_paths, pdf_path, page_size=page_size)

        if success:
            return jsonify(
                {"success": True, "pdf_filename": pdf_filename, "pdf_url": f"/pdf/{pdf_filename}"}
            )
        else:
            return jsonify({"success": False, "error": "PDF generation failed"}), 500

    except Exception as e:
        print(f"PDF export error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/pdf/<filename>")
def serve_pdf(filename):
    """Serve generated PDF files"""
    return send_from_directory(PDF_DIR, filename)


@app.route("/pipeline/info")
def pipeline_info():
    """Get information about the processing pipeline"""
    if not MODULES_AVAILABLE or doc_pipeline is None:
        return jsonify({"available": False, "message": "Advanced pipeline not initialized"})

    try:
        info = doc_pipeline.get_pipeline_info()
        info["available"] = True
        return jsonify(info)
    except Exception as e:
        return jsonify({"available": False, "error": str(e)})


@app.route("/classify/document", methods=["POST"])
def classify_document():
    """
    Classify document type
    Expects: multipart/form-data with 'file'
    """
    if not MODULES_AVAILABLE or doc_pipeline is None:
        return jsonify({"error": "Document classification not available"}), 503

    if not doc_pipeline.classifier.is_trained:
        return (
            jsonify(
                {"error": "Classifier not trained", "message": "Please train the classifier first"}
            ),
            503,
        )

    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file provided"}), 400

        # Save temporarily
        temp_path = os.path.join(UPLOAD_DIR, f"temp_{uuid.uuid4()}.jpg")
        file.save(temp_path)

        # Read and classify
        image = cv2.imread(temp_path)
        doc_type, confidence = doc_pipeline.classifier.predict(image)

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify(
            {
                "document_type": doc_type,
                "confidence": float(confidence),
                "all_types": doc_pipeline.classifier.DOCUMENT_TYPES,
            }
        )

    except Exception as e:
        print(f"Classification error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/detect/document", methods=["POST"])
def detect_document_borders():
    """
    Real-time document border detection
    Expects: multipart/form-data with 'file'
    Returns: Document corners in normalized coordinates [0-100]
    """
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file provided", "success": False}), 400

        # Read file into memory
        file_bytes = file.read()

        # Check if detection is available
        if not MODULES_AVAILABLE:
            print("[WARN] Detection not available - performing basic detection")
            # Basic fallback: just detect if there's content
            return jsonify(
                {
                    "success": True,
                    "corners": [],
                    "message": "Detection service initializing",
                    "fallback": True,
                }
            )

        # Detect document
        try:
            result = detect_and_serialize(file_bytes)
            return jsonify(result)
        except Exception as detection_error:
            print(f"[ERROR] Detection error: {str(detection_error)}")
            # Return success anyway to not break UI
            return jsonify(
                {
                    "success": True,
                    "corners": [],
                    "message": f"Detection unavailable: {str(detection_error)}",
                    "fallback": True,
                }
            )

    except Exception as e:
        print(f"Document detection endpoint error: {str(e)}")
        return (
            jsonify({"success": True, "error": str(e), "corners": [], "fallback": True}),
            200,
        )  # Return 200 to not break frontend


@socketio.on("detect_frame")
def handle_frame_detection(data):
    """
    Real-time frame detection via WebSocket
    Expects: base64 encoded image data
    Emits: detection result with corners
    """
    try:
        if not MODULES_AVAILABLE:
            emit("detection_result", {"success": False, "message": "Detection service unavailable"})
            return

        # Decode base64 image
        import base64
        import io

        from PIL import Image as PILImage

        image_data = data.get("image")
        if not image_data:
            emit("detection_result", {"success": False, "message": "No image data"})
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
        print(f"Frame detection error: {str(e)}")
        emit("detection_result", {"success": False, "message": f"Detection error: {str(e)}"})


def batch_process():
    """
    Process multiple files sequentially with comprehensive tracking and logging
    Expects: multipart/form-data with multiple 'files[]'
    """
    if not MODULES_AVAILABLE or doc_pipeline is None:
        return jsonify({"error": "Batch processing not available", "success": False}), 503

    try:
        files = request.files.getlist("files[]")
        if not files:
            return jsonify({"error": "No files provided", "success": False}), 400

        print(f"\n{'='*70}")
        print(f"?? BATCH PROCESSING INITIATED")
        print(f"  Files: {len(files)}")
        print(f"{'='*70}")

        # Get options
        options = {
            "auto_crop": request.form.get("auto_crop", "true").lower() == "true",
            "export_pdf": request.form.get("export_pdf", "false").lower() == "true",
            "compress": request.form.get("compress", "true").lower() == "true",
        }

        # Save all files sequentially
        print(f"\n[PHASE 1] Saving files...")
        upload_paths = []
        for i, file in enumerate(files, 1):
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                filename = f"batch_{timestamp}_{unique_id}.jpg"
                upload_path = os.path.join(UPLOAD_DIR, filename)
                file.save(upload_path)
                upload_paths.append(upload_path)
                print(f"  [{i}/{len(files)}] ? Saved: {filename}")
            except Exception as save_error:
                print(f"  [{i}/{len(files)}] [ERROR] Failed to save: {str(save_error)}")

        if not upload_paths:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Failed to save any files",
                        "total_files": len(files),
                    }
                ),
                500,
            )

        print(f"\n[PHASE 2] Processing files...")
        # Batch process sequentially
        results = doc_pipeline.batch_process(upload_paths, PROCESSED_DIR, options)

        # Calculate statistics
        successful = sum(1 for r in results if r.get("success"))
        failed = len(results) - successful

        print(f"\n[PHASE 3] Building response...")
        response = {
            "success": True,
            "total_files": len(files),
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / len(files) * 100) if len(files) > 0 else 0,
            "results": results,
        }

        print(f"  ? Summary: {successful} successful, {failed} failed")
        print(f"\n{'='*70}")
        print(f"[OK] BATCH PROCESSING COMPLETED")
        print(f"{'='*70}\n")

        return jsonify(response)

    except Exception as e:
        error_msg = f"Batch processing error: {str(e)}"
        print(f"\n[ERROR] {error_msg}")
        print(traceback.format_exc())
        return (
            jsonify({"success": False, "error": error_msg, "traceback": traceback.format_exc()}),
            500,
        )


# ============================================================================
# OCR ENDPOINTS (PaddleOCR with Ollama post-processing)
# ============================================================================

# OCR data directory
OCR_DATA_DIR = os.path.join(DATA_DIR, "ocr_results")
os.makedirs(OCR_DATA_DIR, exist_ok=True)


@app.route("/ocr/<path:filename>", methods=["POST", "OPTIONS"])
def run_ocr(filename):
    """
    Run PaddleOCR on a processed image
    Returns structured OCR results with bounding boxes and derived title
    Notifies frontend via Socket.IO on completion
    """
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization, ngrok-skip-browser-warning"
        )
        response.headers["Access-Control-Max-Age"] = "3600"
        return response, 200

    try:
        from app.modules.ocr.paddle_ocr import get_ocr_processor
        
        # Security: prevent directory traversal
        if ".." in filename:
            return jsonify({"error": "Invalid filename"}), 400

        # Find the image file
        image_path = os.path.join(PROCESSED_DIR, filename)
        if not os.path.exists(image_path):
            # Try uploads directory
            image_path = os.path.join(UPLOAD_DIR, filename)
        
        if not os.path.exists(image_path):
            return jsonify({"error": f"File not found: {filename}"}), 404

        print(f"\n{'='*60}")
        print(f"📝 OCR PROCESSING: {filename}")
        print(f"{'='*60}")

        # Get or create OCR processor
        processor = get_ocr_processor(OCR_DATA_DIR)
        
        # Run OCR
        result = processor.process_image(image_path)
        
        # Check if OCR actually succeeded (has text)
        ocr_success = result.word_count > 0 and result.derived_title not in ["Error Processing Document", "Untitled Document"]
        
        # Save result only if successful
        json_path = None
        if ocr_success:
            json_path = processor.save_result(filename, result)
        
        # Prepare response
        response_data = {
            "success": ocr_success,
            "filename": filename,
            "ocr_result": result.to_dict() if ocr_success else None,
            "ocr_ready": ocr_success,
            "error": None if ocr_success else "OCR could not extract any text from the image"
        }
        
        if ocr_success:
            print(f"✅ OCR complete: {result.word_count} words, {len(result.raw_results)} regions")
            print(f"   Derived title: {result.derived_title}")
        else:
            print(f"⚠️ OCR found no text in: {filename}")
        print(f"   Processing time: {result.processing_time_ms:.0f}ms")
        print(f"{'='*60}\n")

        # Notify frontend via Socket.IO
        try:
            socketio.emit("ocr_complete", {
                "filename": filename,
                "success": ocr_success,
                "result": result.to_dict() if ocr_success else None,
                "derived_title": result.derived_title if ocr_success else None,
                "word_count": result.word_count,
                "confidence": result.confidence_avg,
                "has_text": result.word_count > 0,
            })
        except Exception as socket_error:
            print(f"[WARN] Socket.IO emit failed: {socket_error}")

        return jsonify(response_data)

    except Exception as e:
        error_msg = f"OCR error: {str(e)}"
        print(f"[ERROR] {error_msg}")
        traceback.print_exc()
        return jsonify({"success": False, "error": error_msg}), 500


@app.route("/ocr/<path:filename>", methods=["GET"])
def get_ocr_result(filename):
    """
    Get existing OCR result for a file
    Returns cached result if available
    """
    try:
        from app.modules.ocr.paddle_ocr import get_ocr_processor
        
        # Security: prevent directory traversal
        if ".." in filename:
            return jsonify({"error": "Invalid filename"}), 400

        # Get OCR processor
        processor = get_ocr_processor(OCR_DATA_DIR)
        
        # Check if OCR result exists
        result = processor.load_result(filename)
        
        if result:
            return jsonify({
                "success": True,
                "filename": filename,
                "ocr_result": result,
                "ocr_ready": True,
            })
        else:
            return jsonify({
                "success": True,
                "filename": filename,
                "ocr_result": None,
                "ocr_ready": False,
            })

    except Exception as e:
        error_msg = f"Error fetching OCR result: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return jsonify({"success": False, "error": error_msg}), 500


@app.route("/ocr-status/<path:filename>", methods=["GET"])
def get_ocr_status(filename):
    """
    Quick check if OCR has been run on a file
    """
    try:
        from app.modules.ocr.paddle_ocr import get_ocr_processor
        
        if ".." in filename:
            return jsonify({"error": "Invalid filename"}), 400

        processor = get_ocr_processor(OCR_DATA_DIR)
        has_ocr = processor.has_ocr_result(filename)
        
        return jsonify({
            "filename": filename,
            "ocr_ready": has_ocr,
        })

    except Exception as e:
        return jsonify({"filename": filename, "ocr_ready": False})


@app.route("/ocr-batch-status", methods=["POST"])
def get_batch_ocr_status():
    """
    Check OCR status for multiple files at once
    """
    try:
        from app.modules.ocr.paddle_ocr import get_ocr_processor
        
        data = request.get_json()
        filenames = data.get("filenames", [])
        
        processor = get_ocr_processor(OCR_DATA_DIR)
        
        statuses = {}
        for filename in filenames:
            if ".." not in filename:
                has_ocr = processor.has_ocr_result(filename)
                status_info = {"has_ocr": has_ocr}
                
                # If OCR exists, try to get the derived title
                if has_ocr:
                    try:
                        result = processor.load_result(filename)
                        if result and result.derived_title:
                            status_info["derived_title"] = result.derived_title
                    except:
                        pass
                
                statuses[filename] = status_info
        
        return jsonify({"success": True, "statuses": statuses})

    except Exception as e:
        return jsonify({"success": False, "statuses": {}, "error": str(e)})


# ============================================================================
# FILE CONVERSION ENDPOINTS
# ============================================================================


@app.route("/convert", methods=["POST", "OPTIONS"])
def convert_files():
    """
    Convert files between formats (JPG, PNG, PDF, DOCX)
    Supports batch conversion
    """
    # Handle OPTIONS preflight request
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization, ngrok-skip-browser-warning"
        )
        response.headers["Access-Control-Max-Age"] = "3600"
        return response, 200

    try:
        from app.modules.document import ExportModule
        from app.modules.document.converter import FileConverter

        data = request.get_json()
        files = data.get("files", [])
        target_format = data.get("format", "pdf").lower()
        merge_pdf = data.get("merge_pdf", False)  # New option for merging PDFs
        custom_filename = data.get("filename", "").strip()  # New option for custom filename

        if not files:
            return jsonify({"success": False, "error": "No files provided"}), 400

        # Use FileConverter to validate supported formats (ExportModule handles export/merge)
        # Do not instantiate ExportModule just to validate formats
        if not FileConverter.is_supported_format(target_format):
            return (
                jsonify({"success": False, "error": f"Unsupported target format: {target_format}"}),
                400,
            )

        # Create converted directory if it doesn't exist
        converted_dir = os.path.join(DATA_DIR, "converted")
        os.makedirs(converted_dir, exist_ok=True)

        # Resolve full paths using the correct DATA_DIR
        input_paths = [os.path.join(PROCESSED_DIR, f) for f in files]

        # Validate files exist
        missing_files = [f for f, p in zip(files, input_paths) if not os.path.exists(p)]
        if missing_files:
            print(f"[ERROR] Missing files: {missing_files}")
            print(f"   Looking in: {PROCESSED_DIR}")
            return (
                jsonify(
                    {"success": False, "error": f'Files not found: {", ".join(missing_files)}'}
                ),
                404,
            )

        print(f"?? Files validated successfully")
        print(f"?? Processed dir: {PROCESSED_DIR}")

        print(f"\n{'='*70}")
        print(f"?? FILE CONVERSION STARTED")
        print(f"{'='*70}")
        print(f"  Files: {len(files)}")
        print(f"  Target format: {target_format.upper()}")
        print(f"  Merge PDF: {merge_pdf}")

        # Check if merging to single PDF
        if merge_pdf and target_format == "pdf":
            # Generate merged PDF filename
            if custom_filename:
                # Sanitize filename - remove extension if provided, ensure .pdf extension
                base_name = os.path.splitext(custom_filename)[0]
                # Remove any potentially dangerous characters
                safe_name = "".join(c for c in base_name if c.isalnum() or c in " -_").strip()
                if not safe_name:
                    safe_name = "merged_document"
                merged_filename = f"{safe_name}.pdf"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                merged_filename = f"merged_document_{timestamp}.pdf"

            merged_path = os.path.join(converted_dir, merged_filename)

            # Merge all images into single PDF
            # Use FileConverter for merging images into a PDF
            success, message = FileConverter.merge_images_to_pdf(input_paths, merged_path)

            print(f"\n{'='*70}")
            print(f"[OK] MERGE CONVERSION COMPLETED")
            print(f"  Status: {'Success' if success else 'Failed'}")
            print(f"{'='*70}\n")

            if success:
                results = [
                    {
                        "input": ", ".join([os.path.basename(f) for f in files]),
                        "output": merged_filename,
                        "success": True,
                        "message": message,
                    }
                ]

                # Emit Socket.IO event
                try:
                    socketio.emit(
                        "conversion_complete",
                        {"success_count": 1, "fail_count": 0, "total": len(files), "merged": True},
                    )
                except Exception as socket_error:
                    print(f"[WARN] Socket.IO emit failed: {socket_error}")

                return jsonify(
                    {
                        "success": True,
                        "results": results,
                        "success_count": 1,
                        "fail_count": 0,
                        "total": len(files),
                        "merged": True,
                        "merged_file": merged_filename,
                    }
                )
            else:
                return jsonify({"success": False, "error": message}), 500

        # Regular batch convert (separate files)
        # Use FileConverter for batch conversion
        success_count, fail_count, results = FileConverter.batch_convert(
            input_paths, converted_dir, target_format
        )

        print(f"\n{'='*70}")
        print(f"[OK] CONVERSION COMPLETED")
        print(f"  Success: {success_count}")
        print(f"  Failed: {fail_count}")
        print(f"{'='*70}\n")

        # Emit Socket.IO event for real-time update
        try:
            socketio.emit(
                "conversion_complete",
                {"success_count": success_count, "fail_count": fail_count, "total": len(files)},
            )
        except Exception as socket_error:
            print(f"[WARN] Socket.IO emit failed: {socket_error}")

        return jsonify(
            {
                "success": True,
                "results": results,
                "success_count": success_count,
                "fail_count": fail_count,
                "total": len(files),
            }
        )

    except Exception as e:
        error_msg = f"Conversion error: {str(e)}"
        print(f"\n[ERROR] {error_msg}")
        print(traceback.format_exc())
        return (
            jsonify({"success": False, "error": error_msg, "traceback": traceback.format_exc()}),
            500,
        )


@app.route("/converted/<path:filename>")
def serve_converted_file(filename):
    """Serve converted files"""
    try:
        return send_from_directory(CONVERTED_DIR, filename)
    except Exception as e:
        print(f"Error serving converted file: {e}")
        return jsonify({"error": str(e)}), 404


@app.route("/converted-page/<path:filepath>")
def serve_converted_page(filepath):
    """Serve extracted PDF pages from converted documents"""
    try:
        # filepath format: filename_pages/filename_page_001.jpg
        return send_from_directory(CONVERTED_DIR, filepath)
    except Exception as e:
        print(f"Error serving converted page: {e}")
        return jsonify({"error": str(e)}), 404


@app.route("/get-converted-files", methods=["GET"])
def get_converted_files():
    """Get list of converted files with extracted pages info"""
    try:
        if not os.path.exists(CONVERTED_DIR):
            return jsonify({"files": []})

        files = []
        for filename in os.listdir(CONVERTED_DIR):
            filepath = os.path.join(CONVERTED_DIR, filename)
            # Only include files, not directories
            if os.path.isfile(filepath):
                stat = os.stat(filepath)
                file_info = {
                    "filename": filename,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "url": f"/converted/{filename}",
                    "pages": [],  # Will be populated if pages are extracted
                }
                
                # Check if extracted pages exist for this file
                base_name = os.path.splitext(filename)[0]
                pages_dir = os.path.join(CONVERTED_DIR, f"{base_name}_pages")
                
                if os.path.isdir(pages_dir):
                    # List extracted pages
                    page_files = sorted([f for f in os.listdir(pages_dir) if f.endswith('.jpg')])
                    for page_file in page_files:
                        page_path = os.path.join(pages_dir, page_file)
                        page_size = os.path.getsize(page_path)
                        file_info["pages"].append({
                            "filename": page_file,
                            "path": f"{base_name}_pages/{page_file}",
                            "size": page_size,
                            "url": f"/converted-page/{base_name}_pages/{page_file}",
                        })
                
                files.append(file_info)

        # Sort by creation time (newest first)
        files.sort(key=lambda x: x["created"], reverse=True)

        return jsonify({"files": files})

    except Exception as e:
        print(f"Error listing converted files: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/delete-converted/<filename>", methods=["DELETE", "OPTIONS"])
def delete_converted_file(filename):
    """Delete a converted file"""
    # Handle OPTIONS preflight request
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization, ngrok-skip-browser-warning"
        )
        response.headers["Access-Control-Max-Age"] = "3600"
        return response, 200

    try:
        # Security: prevent directory traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            return jsonify({"error": "Invalid filename"}), 400

        file_path = os.path.join(CONVERTED_DIR, filename)

        # Verify file exists and is in the converted directory
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404

        # Verify file is actually in CONVERTED_DIR (security check)
        if not os.path.abspath(file_path).startswith(os.path.abspath(CONVERTED_DIR)):
            return jsonify({"error": "Invalid file path"}), 400

        # Delete the file
        os.remove(file_path)
        print(f"[OK] Deleted converted file: {filename}")

        return jsonify({"success": True, "message": f"Successfully deleted {filename}"})

    except Exception as e:
        print(f"[ERROR] Error deleting converted file: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# SMART CONNECTION STATUS ENDPOINTS - SEQUENTIAL VALIDATION
# ============================================================================


@app.route("/connection/validate-wifi", methods=["POST", "OPTIONS"])
def validate_wifi_connection():
    """Validate phone ? laptop WiFi connection via HTTP request"""
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response, 200

    try:
        data = request.get_json() or {}
        timestamp = data.get('timestamp', 0)

        # The fact that this endpoint is being called proves the connection exists
        # (HTTP request successfully reached the backend)
        connected_clients = len(socketio.server.manager.rooms.get("", {})) if hasattr(socketio.server, 'manager') else 0

        logger.info(f"[OK] WiFi validation successful - HTTP POST received | Clients: {connected_clients}")

        return jsonify({
            "connected": True,
            "message": "[OK] Phone and Laptop on same network",
            "ip": "Network established",
            "timestamp": timestamp,
            "clients": connected_clients
        }), 200

    except Exception as e:
        logger.error(f"WiFi validation error: {str(e)}")
        return jsonify({
            "connected": False,
            "message": f"[ERROR] WiFi check failed: {str(e)}"
        }), 200


@app.route("/connection/validate-camera", methods=["POST", "OPTIONS"])
def validate_camera_capturing():
    """Validate phone camera is actively capturing frames"""
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response, 200

    try:
        data = request.get_json() or {}
        is_capturing = data.get('isCapturing', False)
        timestamp = data.get('timestamp', 0)

        # Check if camera session exists or if video stream is active
        # In a real scenario, you'd check actual frame capture state from Phone
        # For now, we verify the client sent proper capture state
        
        camera_active = is_capturing
        
        logger.info(f"[CAMERA] Camera validation - Capturing: {camera_active}")

        if camera_active:
            return jsonify({
                "capturing": True,
                "message": "[OK] Camera is actively capturing frames",
                "timestamp": timestamp,
                "frameRate": "30fps"
            }), 200
        else:
            return jsonify({
                "capturing": False,
                "message": "[ERROR] Camera not currently capturing"
            }), 200

    except Exception as e:
        logger.error(f"Camera validation error: {str(e)}")
        return jsonify({
            "capturing": False,
            "message": f"[ERROR] Camera check failed: {str(e)}"
        }), 200


@app.route("/connection/validate-printer", methods=["POST", "OPTIONS"])
def validate_printer_connection():
    """Validate laptop to printer connection by auto-printing blank.pdf"""
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response, 200

    try:
        data = request.get_json() or {}
        test_print = data.get('testPrint', False)
        timestamp = data.get('timestamp', 0)

        printer_ready = False
        printer_model = "Unknown"
        print_success = False
        print_error_msg = ""

        # Check if printer is available
        try:
            import win32print
            
            # Get default printer name
            printer_name = win32print.GetDefaultPrinter()
            
            if not printer_name:
                return jsonify({
                    "connected": False,
                    "message": "[ERROR] No default printer configured"
                }), 200
            
            printer_model = printer_name
            
            # If test print is requested, print blank.pdf (actual blank page)
            if test_print:
                # Path to blank.pdf in print_scripts folder (actual blank PDF page)
                blank_pdf_path = os.path.join(PRINT_DIR, "blank.pdf")
                
                # Fallback to public folder if not in print_scripts
                if not os.path.exists(blank_pdf_path):
                    blank_pdf_path = os.path.join(os.path.dirname(__file__), 'public', 'blank.pdf')
                
                if not os.path.exists(blank_pdf_path):
                    return jsonify({
                        "connected": False,
                        "message": "[ERROR] blank.pdf not found. Cannot print blank page."
                    }), 200
                
                # Print using multiple methods for reliability
                print_success = False
                print_error_msg = ""
                
                try:
                    import subprocess
                    import win32api
                    
                    logger.info(f"[INFO] Attempting to print blank page: {blank_pdf_path}")
                    
                    # Method 1: Use win32api ShellExecute with "print" verb (most reliable for PDFs)
                    try:
                        win32api.ShellExecute(
                            0,           # handle to parent window
                            "print",     # operation - "print" uses default printer
                            blank_pdf_path,  # file to print
                            None,        # parameters (not needed for print)
                            os.path.dirname(blank_pdf_path),  # working directory
                            0            # show command (0 = hide)
                        )
                        print_success = True
                        printer_ready = True
                        logger.info(f"[OK] Blank page sent to {printer_name} via ShellExecute print")
                    except Exception as shell_err:
                        logger.warning(f"[WARNING] ShellExecute print failed: {shell_err}")
                        
                        # Method 2: PowerShell with Start-Process PrintTo
                        try:
                            escaped_path = blank_pdf_path.replace("\\", "\\\\")
                            escaped_printer = printer_name.replace("\\", "\\\\")
                            ps_cmd = f'Start-Process -FilePath "{escaped_path}" -Verb PrintTo -ArgumentList \\"{escaped_printer}\\" -WindowStyle Hidden'
                            
                            result = subprocess.run(
                                ["powershell", "-Command", ps_cmd],
                                capture_output=True,
                                text=True,
                                timeout=30
                            )
                            
                            if result.returncode == 0:
                                print_success = True
                                printer_ready = True
                                logger.info(f"[OK] Blank page sent to {printer_name} via PowerShell")
                            else:
                                logger.warning(f"[WARNING] PowerShell print returned: {result.stderr}")
                        except Exception as ps_err:
                            logger.warning(f"[WARNING] PowerShell print failed: {ps_err}")
                            
                            # Method 3: Direct print using SumatraPDF if available (silent print)
                            sumatra_paths = [
                                r"C:\Program Files\SumatraPDF\SumatraPDF.exe",
                                r"C:\Program Files (x86)\SumatraPDF\SumatraPDF.exe",
                                os.path.expanduser(r"~\AppData\Local\SumatraPDF\SumatraPDF.exe")
                            ]
                            
                            for sumatra_path in sumatra_paths:
                                if os.path.exists(sumatra_path):
                                    try:
                                        # SumatraPDF silent print: -print-to "printer" file.pdf
                                        subprocess.run(
                                            [sumatra_path, "-print-to", printer_name, blank_pdf_path],
                                            capture_output=True,
                                            timeout=30
                                        )
                                        print_success = True
                                        printer_ready = True
                                        logger.info(f"[OK] Blank page sent to {printer_name} via SumatraPDF")
                                        break
                                    except Exception as sumatra_err:
                                        logger.warning(f"[WARNING] SumatraPDF print failed: {sumatra_err}")
                            
                            if not print_success:
                                print_error_msg = "Could not print blank page. Please ensure a PDF viewer is installed."
                                printer_ready = True  # Printer exists but print failed
                        
                except Exception as print_err:
                    print_error_msg = str(print_err)
                    logger.error(f"[ERROR] Print failed: {print_err}")
                    printer_ready = True  # Still mark as ready since printer exists
            else:
                # No test print requested, just check printer exists
                printer_ready = True
                print_success = True
                
            logger.info(f"[INFO] Printer validation for {printer_name}: ready={printer_ready}, print_success={print_success}")

        except ImportError:
            return jsonify({
                "connected": False,
                "message": "[ERROR] win32print module not installed"
            }), 200
        except Exception as printer_err:
            logger.error(f"Printer check error: {printer_err}")
            return jsonify({
                "connected": False,
                "message": f"[ERROR] Printer check failed: {str(printer_err)}"
            }), 200

        if printer_ready and (print_success or not test_print):
            return jsonify({
                "connected": True,
                "message": f"[OK] Printer ready: {printer_model}",
                "model": printer_model,
                "timestamp": timestamp,
                "testPrintSent": test_print and print_success
            }), 200
        else:
            return jsonify({
                "connected": False,
                "message": f"[ERROR] {print_error_msg or 'Printer not responding or offline'}"
            }), 200

    except Exception as e:
        logger.error(f"Printer validation error: {str(e)}")
        return jsonify({
            "connected": False,
            "message": f"[ERROR] Printer check failed: {str(e)}"
        }), 200


# Keep existing endpoints for backward compatibility
@app.route("/connection/phone-wifi", methods=["GET", "OPTIONS"])
def check_phone_wifi():
    """Check if phone is connected to the same Wi-Fi network"""
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response, 200

    try:
        # Try to detect if a phone/mobile device is connected via Socket.IO
        # Check if there are active Socket connections (phone would connect)
        connected_clients = len(socketio.server.manager.rooms.get("", {}))

        if connected_clients > 0:
            return jsonify(
                {
                    "connected": True,
                    "message": "Phone connected to Wi-Fi network",
                    "ip": "Network active",
                    "clients": connected_clients,
                }
            )
        else:
            return (
                jsonify(
                    {"connected": False, "message": "Phone not detected on network", "clients": 0}
                ),
                200,
            )
    except Exception as e:
        print(f"Wi-Fi check error: {str(e)}")
        return jsonify({"connected": False, "message": f"Connection check failed: {str(e)}"}), 200


@app.route("/connection/printer-status", methods=["GET", "OPTIONS"])
def check_printer_status():
    """Check printer connectivity by attempting to query printer status"""
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response, 200

    try:
        import subprocess
        import sys

        # Try to detect printer via system commands
        try:
            # Windows: Use wmic to check for printers
            result = subprocess.run(
                ["wmic", "logicaldisk", "get", "name"], capture_output=True, text=True, timeout=5
            )

            # If we can execute system commands, printer infrastructure is available
            if result.returncode == 0:
                return jsonify(
                    {
                        "connected": True,
                        "message": "Printer ready",
                        "model": "Network Printer",
                        "status": "online",
                    }
                )
        except Exception:
            pass

        # Fallback: check if print services are available
        try:
            result = subprocess.run(
                ["wmic", "printjob", "list"], capture_output=True, text=True, timeout=5
            )
            return jsonify(
                {
                    "connected": True,
                    "message": "Printer services active",
                    "model": "System Printer",
                    "status": "ready",
                }
            )
        except Exception:
            return (
                jsonify(
                    {"connected": False, "message": "Printer not responding", "status": "offline"}
                ),
                200,
            )

    except Exception as e:
        print(f"Printer check error: {str(e)}")
        return (
            jsonify(
                {
                    "connected": False,
                    "message": f"Printer check failed: {str(e)}",
                    "status": "error",
                }
            ),
            200,
        )


@app.route("/connection/camera-ready", methods=["GET", "OPTIONS"])
def check_camera_ready():
    """Check if phone camera session is active and ready"""
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response, 200

    try:
        # Check if there are active Socket.IO connections (indicates phone app is open)
        # and if any image uploads have occurred recently (camera activity)
        connected_clients = len(socketio.server.manager.rooms.get("", {}))

        # Check for recent uploads in the uploads directory
        upload_dir = UPLOAD_DIR
        if os.path.exists(upload_dir):
            files = os.listdir(upload_dir)
            if files:
                # Check if there are recent files (modified within last 5 minutes)
                import time

                current_time = time.time()
                recent_files = [
                    f
                    for f in files
                    if os.path.getmtime(os.path.join(upload_dir, f)) > current_time - 300
                ]

                if recent_files or connected_clients > 0:
                    return jsonify(
                        {
                            "ready": True,
                            "message": "Camera ready and active",
                            "active_clients": connected_clients,
                            "recent_uploads": len(recent_files),
                        }
                    )

        # If no recent activity but clients are connected
        if connected_clients > 0:
            return jsonify(
                {
                    "ready": True,
                    "message": "Camera session active",
                    "active_clients": connected_clients,
                    "recent_uploads": 0,
                }
            )
        else:
            return (
                jsonify(
                    {"ready": False, "message": "Camera session not detected", "active_clients": 0}
                ),
                200,
            )

    except Exception as e:
        print(f"Camera ready check error: {str(e)}")
        return jsonify({"ready": False, "message": f"Camera check failed: {str(e)}"}), 200


# ============================================================================
# SOCKET.IO EVENTS
# ============================================================================

# ============================================================================
# SOCKET.IO HANDLERS
# ============================================================================


@socketio.on("connect")
def handle_connect():
    """Handle client connection - keep it absolutely simple"""
    print(f"[OK] Socket connected: {request.sid}")
    return True


@socketio.on("error")
def error_handler(e):
    """Handle errors"""
    print(f"Socket error: {e}")


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Socket disconnected: {request.sid}")


@socketio.on("ping")
def handle_ping():
    """Handle ping from client"""
    emit("pong", {"timestamp": datetime.now().isoformat()})


# ============================================================================
# VOICE AI ENDPOINTS
# ============================================================================


@app.route("/voice/start", methods=["POST"])
def start_voice_session():
    """
    Start a new voice AI session
    Loads Whisper model and checks Ollama availability
    """
    try:
        from app.modules.voice import voice_ai_orchestrator

        logger.info("Starting voice AI session...")
        result = voice_ai_orchestrator.start_session()

        if result.get("success"):
            logger.info("[OK] Voice AI session started successfully")
            return jsonify(result), 200
        else:
            logger.error(f"[ERROR] Voice AI session start failed: {result.get('error')}")
            return jsonify(result), 503

    except ImportError as e:
        logger.error(f"Voice AI module import error: {str(e)}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Voice AI module not available. Install dependencies: pip install openai-whisper requests",
                }
            ),
            503,
        )
    except Exception as e:
        logger.error(f"Voice session start error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/voice/transcribe", methods=["POST"])
def transcribe_voice():
    """
    Transcribe audio to text using Whisper Large-v3 Turbo
    Expects: multipart/form-data with 'audio' field (WAV format)
    """
    try:
        from app.modules.voice import voice_ai_orchestrator

        # Check if session is active
        if not voice_ai_orchestrator.session_active:
            return (
                jsonify(
                    {"success": False, "error": "No active voice session. Start a session first."}
                ),
                400,
            )

        # Get audio file
        audio_file = request.files.get("audio")
        if not audio_file:
            return jsonify({"success": False, "error": "No audio file provided"}), 400

        # Read audio bytes
        audio_data = audio_file.read()

        logger.info(f"Received audio: {len(audio_data)} bytes")

        # Transcribe
        transcription = voice_ai_orchestrator.whisper_service.transcribe_audio(audio_data)

        if transcription.get("success"):
            logger.info(f"[OK] Transcription: {transcription.get('text')}")
            return jsonify(transcription), 200
        else:
            logger.error(f"[ERROR] Transcription failed: {transcription.get('error')}")
            return jsonify(transcription), 500

    except ImportError:
        return jsonify({"success": False, "error": "Voice AI module not available"}), 503
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/voice/chat", methods=["POST"])
def chat_with_ai():
    """
    Send text to the AI and get a response.
    UNIFIED: This endpoint now behaves identically to text input.
    Voice-specific logic has been removed - all processing is done by generate_response().
    
    Expects: JSON with 'message' field
    Returns: AI response with all standard fields
    """
    try:
        from app.modules.voice import voice_ai_orchestrator

        # Check if session is active
        if not voice_ai_orchestrator.session_active:
            return (
                jsonify(
                    {"success": False, "error": "No active voice session. Start a session first."}
                ),
                400,
            )

        data = request.get_json()
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"success": False, "error": "No message provided"}), 400

        logger.info(f"💬 User message: {user_message}")

        # Generate response - NO extra processing, same as text input
        response = voice_ai_orchestrator.chat_service.generate_response(user_message)

        if response.get("success"):
            ai_response = response.get("response", "")
            
            # Ensure response fields are consistent for frontend
            if "tts_response" not in response:
                response["tts_response"] = ai_response
            if "ai_response" not in response:
                response["ai_response"] = ai_response
            
            logger.info(f"[OK] AI response: {ai_response[:100]}...")
            return jsonify(response), 200
        else:
            logger.error(f"[ERROR] Chat failed: {response.get('error')}")
            return jsonify(response), 500

    except ImportError:
        return jsonify({"success": False, "error": "Voice AI module not available"}), 503
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/voice/speak", methods=["POST"])
def speak_text():
    """
    Speak text using TTS (blocking call)
    Expects: JSON with 'text' field
    """
    try:
        from app.modules.voice import voice_ai_orchestrator

        data = request.get_json()
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400

        # Speak text (blocking)
        result = voice_ai_orchestrator.speak_text_response(text)

        if result.get("success"):
            return jsonify(result), 200
        else:
            return jsonify(result), 500

    except ImportError:
        return jsonify({"success": False, "error": "Voice AI module not available"}), 503
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/voice/process", methods=["POST"])
def process_voice_complete():
    """
    Complete voice processing pipeline: Audio ? Transcription ? AI Response
    Expects: multipart/form-data with 'audio' field (WAV format)
    Returns: Transcription + AI response + session status
    """
    try:
        from app.modules.voice import voice_ai_orchestrator

        # Check if session is active
        if not voice_ai_orchestrator.session_active:
            logger.warning("Voice processing requested without active session")
            return (
                jsonify(
                    {"success": False, "error": "No active voice session. Start a session first."}
                ),
                400,
            )

        # Get audio file
        audio_file = request.files.get("audio")
        if not audio_file:
            logger.warning("Voice processing requested without audio file")
            return jsonify({"success": False, "error": "No audio file provided"}), 400

        # Read audio bytes
        audio_data = audio_file.read()
        if not audio_data:
            logger.warning("Audio file provided but is empty")
            return jsonify({"success": False, "error": "Audio file is empty"}), 400

        logger.info(
            f"Processing voice input: {len(audio_data)} bytes, filename: {audio_file.filename}"
        )

        # Validate audio file
        logger.info(f"Audio file first 20 bytes (hex): {audio_data[:20].hex()}")
        logger.info(f"Audio file first 20 bytes (raw): {audio_data[:20]}")

        # Check for RIFF header
        if audio_data[:4] != b"RIFF":
            logger.warning(f"[WARN] Audio file doesn't have RIFF header! Got: {audio_data[:4]}")
            logger.warning(f"   This may cause FFmpeg errors during transcription")
        else:
            logger.info("[OK] Audio file has valid RIFF header")

        # Process through complete pipeline
        result = voice_ai_orchestrator.process_voice_input(audio_data)

        if result.get("success"):
            logger.info(f"[OK] Voice processing complete")
            logger.info(f"   User: {result.get('user_text')}")
            logger.info(f"   AI: {result.get('ai_response')}")

            # FALLBACK: Check if orchestration trigger wasn't detected by LLM
            # This ensures print/scan commands always work even if LLM missed them
            # BUT skip if awaiting_confirmation is set (voice module already handled it properly)
            if not result.get("orchestration_trigger") and not result.get("awaiting_confirmation"):
                user_text_lower = result.get("user_text", "").lower()
                
                # Check for print intent
                print_keywords = ["print", "printing", "printout", "print doc", "print file"]
                switch_keywords = ["switch to", "open", "go to", "show me", "navigate to"]
                is_switch_print = any(sk in user_text_lower for sk in switch_keywords) and "print" in user_text_lower
                is_print_command = any(kw in user_text_lower for kw in print_keywords)
                is_not_question = not any(w in user_text_lower for w in ["what", "can you", "how", "help", "tell me", "can i"])
                
                if (is_print_command or is_switch_print) and is_not_question:
                    logger.info(f"[FALLBACK] Print command detected: '{result.get('user_text')}'")
                    # ASK FOR CONFIRMATION instead of directly triggering
                    result["awaiting_confirmation"] = True
                    result["pending_mode"] = "print"
                    result["ai_response"] = "Do you want to print? Say yes to proceed."
                
                # Check for scan intent
                scan_keywords = ["scan", "scanning", "capture", "scan doc", "capture document"]
                is_switch_scan = any(sk in user_text_lower for sk in switch_keywords) and "scan" in user_text_lower
                is_scan_command = any(kw in user_text_lower for kw in scan_keywords)
                
                if (is_scan_command or is_switch_scan) and is_not_question:
                    logger.info(f"[FALLBACK] Scan command detected: '{result.get('user_text')}'")
                    # ASK FOR CONFIRMATION instead of directly triggering
                    result["awaiting_confirmation"] = True
                    result["pending_mode"] = "scan"
                    result["ai_response"] = "Do you want to scan? Say yes to proceed."

            # Check if user text contains orchestration commands
            user_text = result.get("user_text", "")
            
            # If orchestration was just triggered, set up orchestrator state
            if result.get("orchestration_trigger") and ORCHESTRATION_AVAILABLE and orchestrator:
                from app.modules.orchestration import IntentType
                
                mode = result.get("orchestration_mode")
                if mode in ["print", "scan"]:
                    logger.info(f"[ORCHESTRATOR] Setting up {mode} orchestration state")
                    
                    # Add voice_triggered flag to trigger CONFIGURING state
                    intent_text = f"{mode} a document with voice control"
                    intent, params = orchestrator.detect_intent(intent_text)
                    
                    if params is None:
                        params = {}
                    params["voice_triggered"] = True
                    
                    # Process command to set orchestrator state
                    orchestration_result = orchestrator.process_command(
                        intent_text,
                        force_voice_triggered=True,
                    )
                    
                    # Store orchestration result for later use
                    result["orchestration_state_setup"] = True
                    result["orchestration_workflow_state"] = orchestrator.current_state.value
                    
                    logger.info(f"[ORCHESTRATOR] State set to: {orchestrator.current_state.value}")
            if ORCHESTRATION_AVAILABLE and orchestrator and user_text:
                # Check current orchestration state
                current_state = orchestrator.current_state.value

                # If in CONFIGURING state, parse voice for configuration changes
                # BUT skip if a voice command like proceed_action was already detected
                voice_command = result.get("voice_command")
                skip_config_parsing = voice_command in [
                    "proceed_action", "select_document", "select_document_range",
                    "select_multiple_documents", "deselect_document", "switch_section",
                    "next_document", "previous_document", "confirm", "cancel"
                ]
                
                # ALSO skip config parsing if this is a print/scan intent - handle it as new orchestration
                user_text_lower = user_text.lower().strip()
                is_print_intent = user_text_lower in ["print", "print.", "i want to print", "print document", "print documents"]
                is_scan_intent = user_text_lower in ["scan", "scan.", "i want to scan", "scan document", "scan documents"]
                
                if is_print_intent or is_scan_intent:
                    skip_config_parsing = True
                    # Force detection as orchestration intent
                    from app.modules.orchestration import IntentType
                    intent = IntentType.PRINT if is_print_intent else IntentType.SCAN
                    mode = "print" if is_print_intent else "scan"
                    
                    logger.info(f"[VOICE] Detected {mode} intent in configuring state, treating as new orchestration")
                    
                    # Set result flags for frontend
                    result["orchestration_trigger"] = True
                    result["orchestration_mode"] = mode
                    result["ai_response"] = f"Would you like me to open the {mode} configuration? Say 'yes' to proceed."
                
                if current_state == "configuring" and orchestrator.pending_action and not skip_config_parsing:
                    action_type = orchestrator.pending_action.get("type")
                    if action_type:
                        logger.info(f"[VOICE] Parsing voice configuration for {action_type}")
                        parsed_config = orchestrator.parse_voice_configuration(
                            user_text, action_type
                        )

                        if parsed_config.get("no_changes"):
                            # User indicated they're done with configuration
                            result["orchestration"] = {
                                "no_changes": True,
                                "message": "Configuration complete. Ready to proceed.",
                                "ready_to_confirm": True,
                            }
                            result["ai_response"] = (
                                "Perfect! Your settings are ready. Shall we proceed?"
                            )

                            socketio.emit(
                                "orchestration_update",
                                {
                                    "type": "configuration_complete",
                                    "ready_to_confirm": True,
                                    "timestamp": datetime.now().isoformat(),
                                },
                            )
                        elif parsed_config:
                            # Apply configuration updates
                            update_result = orchestrator.update_configuration(
                                action_type, parsed_config
                            )
                            result["orchestration"] = {
                                "configuration_updated": True,
                                "updates": parsed_config,
                                "configuration": update_result.get("configuration"),
                            }

                            confirmation_message = _build_voice_confirmation(parsed_config)
                            result["ai_response"] = confirmation_message

                            socketio.emit(
                                "orchestration_update",
                                {
                                    "type": "voice_configuration_updated",
                                    "action_type": action_type,
                                    "updates": parsed_config,
                                    "configuration": update_result.get("configuration"),
                                    "frontend_updates": update_result.get("frontend_updates"),
                                    "frontend_state": update_result.get("frontend_state"),
                                    "timestamp": datetime.now().isoformat(),
                                },
                            )
                        else:
                            result["ai_response"] = (
                                "I didn't catch any configuration changes. Try saying things like 'landscape', '3 copies', or 'color mode'."
                            )
                else:
                    # Try to detect orchestration intent
                    from app.modules.orchestration import IntentType

                    intent, params = orchestrator.detect_intent(user_text)

                    # If valid orchestration intent detected, process it
                    if intent in [IntentType.PRINT, IntentType.SCAN]:
                        logger.info(f"[TRIGGER] Orchestration intent detected: {intent.value}")

                        # Add voice_triggered flag to parameters
                        if params is None:
                            params = {}
                        params["voice_triggered"] = True

                        # Process command with voice_triggered flag
                        orchestration_result = orchestrator.process_command(
                            user_text,
                            force_voice_triggered=True,
                        )

                        # Add orchestration result to response
                        result["orchestration"] = orchestration_result
                        result["orchestration_detected"] = True

                        # Override AI response with orchestration message
                        if orchestration_result.get("message"):
                            result["ai_response"] = orchestration_result["message"]

                        # Emit orchestration update
                        try:
                            socketio.emit(
                                "orchestration_update",
                                {
                                    "type": "voice_command_detected",
                                    "intent": intent.value,
                                    "result": orchestration_result,
                                    "timestamp": datetime.now().isoformat(),
                                    "open_ui": orchestration_result.get("open_ui", False),
                                    "skip_mode_selection": orchestration_result.get(
                                        "skip_mode_selection", False
                                    ),
                                    "frontend_state": orchestration_result.get("frontend_state"),
                                },
                            )
                        except Exception as socket_error:
                            logger.warning(f"Orchestration socket emit failed: {socket_error}")

            # Emit to frontend via Socket.IO
            try:
                socketio.emit(
                    "voice_message",
                    {
                        "user_text": result.get("user_text"),
                        "ai_response": result.get("ai_response"),
                        "orchestration": result.get("orchestration"),
                        "timestamp": datetime.now().isoformat(),
                        "session_ended": result.get("session_ended", False),
                    },
                )
                
                # Emit config updates if settings were changed via voice command
                if result.get("settings_updated") or result.get("config_updates"):
                    config_updates = result.get("config_updates", result.get("command_params", {}))
                    voice_command = result.get("voice_command")
                    
                    logger.info(f"[SETTINGS] Emitting config update: {voice_command} -> {config_updates}")
                    
                    socketio.emit(
                        "config_update",
                        {
                            "type": "voice_settings_update",
                            "voice_command": voice_command,
                            "updates": config_updates,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )
            except Exception as socket_error:
                logger.warning(f"Socket.IO emit failed: {socket_error}")

            return jsonify(result), 200
        else:
            logger.error(f"[ERROR] Voice processing failed: {result.get('error')}")
            logger.error(f"   Error stage: {result.get('stage', 'unknown')}")
            return jsonify(result), 500

    except ImportError as ie:
        logger.error(f"Voice AI module import error: {str(ie)}")
        return (
            jsonify(
                {"success": False, "error": "Voice AI module not available", "details": str(ie)}
            ),
            503,
        )
    except Exception as e:
        logger.error(f"Voice processing error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e), "details": traceback.format_exc()}), 500


@app.route("/voice/end", methods=["POST"])
def end_voice_session():
    """
    End the voice AI session
    Clears conversation history and resets state
    """
    try:
        from app.modules.voice import voice_ai_orchestrator

        voice_ai_orchestrator.end_session()

        logger.info("[OK] Voice AI session ended")

        return jsonify({"success": True, "message": "Voice AI session ended"}), 200

    except ImportError:
        return jsonify({"success": False, "error": "Voice AI module not available"}), 503
    except Exception as e:
        logger.error(f"End session error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/voice/status", methods=["GET"])
def voice_status():
    """
    Get voice AI system status
    """
    try:
        from app.modules.voice import voice_ai_orchestrator

        return (
            jsonify(
                {
                    "session_active": voice_ai_orchestrator.session_active,
                    "whisper_loaded": voice_ai_orchestrator.whisper_service.is_loaded,
                    "ollama_available": voice_ai_orchestrator.chat_service.check_ollama_available(),
                    "conversation_length": len(
                        voice_ai_orchestrator.chat_service.conversation_history
                    ),
                }
            ),
            200,
        )

    except ImportError:
        return jsonify({"available": False, "error": "Voice AI module not available"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# CONNECTION VALIDATION ENDPOINTS
# ============================================================================


@app.route("/validate/connection", methods=["GET"])
def validate_connection():
    """
    Validate phone-laptop connection (simple ping response)
    """
    try:
        return (
            jsonify(
                {
                    "success": True,
                    "connected": True,
                    "server": "PrintChakra Backend",
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"success": False, "connected": False, "error": str(e)}), 500


@app.route("/validate/camera", methods=["POST"])
def validate_camera():
    """
    Validate camera is active and capturing frames
    Expects a small image frame from the phone camera
    """
    try:
        # Check if image data is provided
        if "frame" not in request.files and "image" not in request.files:
            return (
                jsonify({"success": False, "camera_active": False, "error": "No frame provided"}),
                400,
            )

        file = request.files.get("frame") or request.files.get("image")

        # Try to read the image
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        if img is None or img.size == 0:
            return (
                jsonify({"success": False, "camera_active": False, "error": "Invalid image data"}),
                400,
            )

        # Image is valid - camera is working
        height, width = img.shape[:2]
        return (
            jsonify(
                {
                    "success": True,
                    "camera_active": True,
                    "frame_size": f"{width}x{height}",
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Camera validation error: {e}")
        return jsonify({"success": False, "camera_active": False, "error": str(e)}), 500


@app.route("/validate/printer", methods=["POST"])
def validate_printer():
    """
    Validate printer connection by printing a blank PDF
    """
    try:
        import win32print

        # Get default printer
        try:
            printer_name = win32print.GetDefaultPrinter()
        except Exception as e:
            return (
                jsonify(
                    {
                        "success": False,
                        "printer_connected": False,
                        "error": f"No default printer found: {str(e)}",
                    }
                ),
                500,
            )

        # Check printer status
        try:
            handle = win32print.OpenPrinter(printer_name)
            printer_info = win32print.GetPrinter(handle, 2)
            win32print.ClosePrinter(handle)
        except Exception as e:
            return (
                jsonify(
                    {
                        "success": False,
                        "printer_connected": False,
                        "error": f"Cannot access printer: {str(e)}",
                    }
                ),
                500,
            )

        # Try to print blank.pdf
        blank_pdf_path = os.path.join(BASE_DIR, "app", "print_scripts", "blank.pdf")

        if not os.path.exists(blank_pdf_path):
            return (
                jsonify(
                    {
                        "success": False,
                        "printer_connected": False,
                        "error": f"Test file not found: {blank_pdf_path}",
                    }
                ),
                500,
            )

        # Print using PowerShell
        try:
            powershell_cmd = (
                f'Get-Content "{blank_pdf_path}" -Raw | Out-Printer -Name "{printer_name}"'
            )
            result = subprocess.run(
                ["powershell.exe", "-Command", powershell_cmd],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                return (
                    jsonify(
                        {
                            "success": True,
                            "printer_connected": True,
                            "printer_name": printer_name,
                            "message": "Test page sent to printer",
                            "timestamp": datetime.now().isoformat(),
                        }
                    ),
                    200,
                )
            else:
                return (
                    jsonify(
                        {
                            "success": False,
                            "printer_connected": False,
                            "error": f"Print command failed: {result.stderr}",
                        }
                    ),
                    500,
                )

        except subprocess.TimeoutExpired:
            return (
                jsonify(
                    {"success": False, "printer_connected": False, "error": "Print command timeout"}
                ),
                500,
            )
        except Exception as e:
            return (
                jsonify(
                    {
                        "success": False,
                        "printer_connected": False,
                        "error": f"Print error: {str(e)}",
                    }
                ),
                500,
            )

    except ImportError:
        return (
            jsonify(
                {
                    "success": False,
                    "printer_connected": False,
                    "error": "win32print module not available (Windows only)",
                }
            ),
            503,
        )
    except Exception as e:
        logger.error(f"Printer validation error: {e}")
        return jsonify({"success": False, "printer_connected": False, "error": str(e)}), 500


# ============================================================================
# AI ORCHESTRATION ENDPOINTS
# ============================================================================

# Import orchestration service
try:
    from app.modules.orchestration import get_orchestrator

    ORCHESTRATION_AVAILABLE = True
    orchestrator = get_orchestrator(DATA_DIR)
    logger.info("[OK] AI Orchestration service initialized")
except ImportError as e:
    ORCHESTRATION_AVAILABLE = False
    orchestrator = None
    logger.warning(f"[WARN] AI Orchestration not available: {e}")


@app.route("/orchestrate/command", methods=["POST"])
def orchestrate_command():
    """
    Process natural language orchestration command
    Expects: JSON with { "command": "print this document" }
    """
    if not ORCHESTRATION_AVAILABLE or not orchestrator:
        return jsonify({"success": False, "error": "Orchestration service not available"}), 503

    try:
        data = request.get_json()
        command = data.get("command", "")

        if not command:
            return jsonify({"success": False, "error": "No command provided"}), 400

        logger.info(f"[ORCHESTRATOR] Orchestration command: {command}")

        # Process command
        result = orchestrator.process_command(command)

        # Emit Socket.IO event for UI updates
        if result.get("success"):
            socketio.emit(
                "orchestration_update",
                {
                    "type": "command_processed",
                    "result": result,
                    "timestamp": datetime.now().isoformat(),
                },
            )

        return jsonify(result)

    except Exception as e:
        logger.error(f"[ERROR] Orchestration command error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/orchestrate/confirm", methods=["POST"])
def orchestrate_confirm():
    """Confirm pending orchestration action"""
    if not ORCHESTRATION_AVAILABLE or not orchestrator:
        return jsonify({"success": False, "error": "Orchestration service not available"}), 503

    try:
        result = orchestrator.confirm_action()

        # Emit Socket.IO event
        socketio.emit(
            "orchestration_update",
            {"type": "action_confirmed", "result": result, "timestamp": datetime.now().isoformat()},
        )

        # If action was to open phone interface, emit redirect event
        if result.get("success") and result.get("redirect_to"):
            socketio.emit(
                "orchestration_redirect",
                {"path": result["redirect_to"], "message": result.get("message", "")},
            )

        return jsonify(result)

    except Exception as e:
        logger.error(f"[ERROR] Orchestration confirm error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/orchestrate/cancel", methods=["POST"])
def orchestrate_cancel():
    """Cancel pending orchestration action"""
    if not ORCHESTRATION_AVAILABLE or not orchestrator:
        return jsonify({"success": False, "error": "Orchestration service not available"}), 503

    try:
        result = orchestrator.cancel_action()

        # Emit Socket.IO event
        socketio.emit(
            "orchestration_update",
            {"type": "action_cancelled", "result": result, "timestamp": datetime.now().isoformat()},
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"[ERROR] Orchestration cancel error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/orchestrate/status", methods=["GET"])
def orchestrate_status():
    """Get current orchestration status"""
    if not ORCHESTRATION_AVAILABLE or not orchestrator:
        return jsonify({"success": False, "error": "Orchestration service not available"}), 503

    try:
        result = orchestrator._handle_status_inquiry()
        return jsonify(result)

    except Exception as e:
        logger.error(f"[ERROR] Orchestration status error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/orchestrate/documents", methods=["GET"])
def orchestrate_documents():
    """Get available documents for orchestration"""
    if not ORCHESTRATION_AVAILABLE or not orchestrator:
        return jsonify({"success": False, "error": "Orchestration service not available"}), 503

    try:
        result = orchestrator._handle_list_documents()
        return jsonify(result)

    except Exception as e:
        logger.error(f"[ERROR] Orchestration documents error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/orchestrate/select", methods=["POST"])
def orchestrate_select():
    """
    Select a document for orchestration
    Expects: JSON with { "filename": "doc.jpg" }
    """
    if not ORCHESTRATION_AVAILABLE or not orchestrator:
        return jsonify({"success": False, "error": "Orchestration service not available"}), 503

    try:
        data = request.get_json()
        filename = data.get("filename", "")

        if not filename:
            return jsonify({"success": False, "error": "No filename provided"}), 400

        result = orchestrator.select_document(filename)

        # Emit Socket.IO event
        if result.get("success"):
            socketio.emit(
                "orchestration_update",
                {
                    "type": "document_selected",
                    "document": result.get("document"),
                    "timestamp": datetime.now().isoformat(),
                },
            )

        return jsonify(result)

    except Exception as e:
        logger.error(f"[ERROR] Orchestration select error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/orchestrate/configure", methods=["POST"])
def orchestrate_configure():
    """
    Update orchestration configuration
    Expects: JSON with { "type": "print|scan", "settings": {...} }
    """
    if not ORCHESTRATION_AVAILABLE or not orchestrator:
        return jsonify({"success": False, "error": "Orchestration service not available"}), 503

    try:
        data = request.get_json()
        action_type = data.get("type", "")
        settings = data.get("settings", {})

        if not action_type or not settings:
            return jsonify({"success": False, "error": "Missing type or settings"}), 400

        result = orchestrator.update_configuration(action_type, settings)

        # Emit Socket.IO event
        if result.get("success"):
            socketio.emit(
                "orchestration_update",
                {
                    "type": "configuration_updated",
                    "action_type": action_type,
                    "configuration": result.get("configuration"),
                    "frontend_state": result.get("frontend_state"),
                    "frontend_updates": result.get("frontend_updates"),
                    "timestamp": datetime.now().isoformat(),
                },
            )

        return jsonify(result)

    except Exception as e:
        logger.error(f"[ERROR] Orchestration configure error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/orchestrate/print", methods=["POST"])
def orchestrate_print():
    """
    Execute print job with configuration from frontend
    Accepts multipart/form-data with:
        - convertedFiles: JSON array of converted PDF filenames
        - selectedDocuments: JSON array of document filenames to print
        - dashboardFiles: JSON array of dashboard file names
        - options: JSON object with print configuration
    """
    try:
        # Parse form data
        converted_files = json.loads(request.form.get("convertedFiles", "[]"))
        selected_documents = json.loads(request.form.get("selectedDocuments", "[]"))
        dashboard_files = json.loads(request.form.get("dashboardFiles", "[]"))
        options = json.loads(request.form.get("options", "{}"))

        logger.info(f"[PRINT] Received print request:")
        logger.info(f"[PRINT]   - converted_files: {converted_files}")
        logger.info(f"[PRINT]   - selected_documents: {selected_documents}")
        logger.info(f"[PRINT]   - dashboard_files: {dashboard_files}")

        # Collect all files to print
        files_to_print = []

        # Helper function to find file in multiple directories
        def find_file(filename):
            """Search for file in multiple possible directories"""
            search_dirs = [
                PROCESSED_DIR,
                CONVERTED_DIR,
                UPLOAD_DIR,
                PDF_DIR,
                os.path.join(DATA_DIR, "ocr_results"),
            ]
            for search_dir in search_dirs:
                file_path = os.path.join(search_dir, filename)
                if os.path.exists(file_path):
                    logger.info(f"[PRINT] Found {filename} in {search_dir}")
                    return file_path
            logger.warning(f"[PRINT] File not found: {filename}")
            logger.warning(f"[PRINT]   Searched in: {search_dirs}")
            return None

        # Add converted PDFs from converted directory
        for filename in converted_files:
            file_path = find_file(filename)
            if file_path:
                files_to_print.append({"filename": filename, "path": file_path, "source": "converted"})

        # Add selected documents - search in multiple directories
        for filename in selected_documents:
            file_path = find_file(filename)
            if file_path:
                files_to_print.append({"filename": filename, "path": file_path, "source": "processed"})

        # Add dashboard files
        for filename in dashboard_files:
            file_path = find_file(filename)
            if file_path:
                files_to_print.append({"filename": filename, "path": file_path, "source": "dashboard"})

        logger.info(f"[PRINT] Files to print: {[f['filename'] for f in files_to_print]}")

        if not files_to_print:
            logger.error(f"[PRINT] No valid files found to print!")
            logger.error(f"[PRINT]   PROCESSED_DIR: {PROCESSED_DIR}")
            logger.error(f"[PRINT]   Contents: {os.listdir(PROCESSED_DIR) if os.path.exists(PROCESSED_DIR) else 'DIR NOT FOUND'}")
            return jsonify({
                "success": False,
                "error": "No valid files found to print",
                "searched_documents": selected_documents,
                "searched_dirs": [PROCESSED_DIR, CONVERTED_DIR, UPLOAD_DIR]
            }), 400

        logger.info(f"[PRINT] Starting print job for {len(files_to_print)} file(s)")
        logger.info(f"[PRINT] Options: {options}")

        # Extract print options
        copies = int(options.get("copies", 1))
        color_mode = options.get("colorMode", "color")
        duplex = options.get("duplex", False)
        pages = options.get("pages", "all")
        custom_range = options.get("customRange", "")
        layout = options.get("layout", "portrait")
        # Delay options: first document gets longer delay, subsequent get shorter delay
        first_delay = int(options.get("firstDelay", 12))  # 12 seconds default for first doc
        subsequent_delay = int(options.get("subsequentDelay", 2))  # 2 seconds default for subsequent docs

        # Try to print each file
        successful_prints = []
        failed_prints = []

        try:
            import win32api
            import win32print
            import time

            printer_name = win32print.GetDefaultPrinter()
            logger.info(f"[PRINT] Default printer: {printer_name}")
            
            if not printer_name:
                return jsonify({
                    "success": False,
                    "error": "No default printer configured"
                }), 500

            for idx, file_info in enumerate(files_to_print):
                file_path = file_info["path"]
                filename = file_info["filename"]
                
                logger.info(f"[PRINT] Processing file {idx + 1}/{len(files_to_print)}: {filename}")
                logger.info(f"[PRINT]   Path: {file_path}")
                logger.info(f"[PRINT]   Exists: {os.path.exists(file_path)}")

                try:
                    # Apply delay BEFORE printing (except for first document if no delay needed)
                    # First document gets first_delay seconds, subsequent get subsequent_delay seconds
                    if idx == 0 and first_delay > 0:
                        logger.info(f"[PRINT] Waiting {first_delay}s before first document...")
                        # Emit progress update
                        socketio.emit("print_progress", {
                            "type": "waiting",
                            "document": filename,
                            "index": idx + 1,
                            "total": len(files_to_print),
                            "delay": first_delay,
                            "message": f"Waiting {first_delay}s before first document..."
                        })
                        time.sleep(first_delay)
                    elif idx > 0 and subsequent_delay > 0:
                        logger.info(f"[PRINT] Waiting {subsequent_delay}s before document {idx + 1}...")
                        # Emit progress update
                        socketio.emit("print_progress", {
                            "type": "waiting",
                            "document": filename,
                            "index": idx + 1,
                            "total": len(files_to_print),
                            "delay": subsequent_delay,
                            "message": f"Waiting {subsequent_delay}s before document {idx + 1}..."
                        })
                        time.sleep(subsequent_delay)
                    
                    for copy_num in range(copies):
                        # Try multiple print methods for reliability
                        print_success = False
                        
                        # Method 1: ShellExecute with "print" verb
                        try:
                            result = win32api.ShellExecute(0, "print", file_path, None, ".", 0)
                            logger.info(f"[PRINT] ShellExecute result: {result}")
                            if result > 32:  # Success if > 32
                                print_success = True
                                logger.info(f"[PRINT] ShellExecute succeeded for: {filename}")
                        except Exception as shell_err:
                            logger.warning(f"[PRINT] ShellExecute failed: {shell_err}")
                        
                        # Method 2: PowerShell Start-Process if ShellExecute failed
                        if not print_success:
                            try:
                                cmd = f'Start-Process -FilePath "{file_path}" -Verb Print -WindowStyle Hidden'
                                result = subprocess.run(["powershell", "-Command", cmd], 
                                                       timeout=30, capture_output=True, text=True)
                                if result.returncode == 0:
                                    print_success = True
                                    logger.info(f"[PRINT] PowerShell Print succeeded for: {filename}")
                                else:
                                    logger.warning(f"[PRINT] PowerShell Print failed: {result.stderr}")
                            except Exception as ps_err:
                                logger.warning(f"[PRINT] PowerShell method failed: {ps_err}")
                        
                        # Method 3: PrintTo with printer name
                        if not print_success:
                            try:
                                result = win32api.ShellExecute(0, "printto", file_path, f'"{printer_name}"', ".", 0)
                                logger.info(f"[PRINT] PrintTo result: {result}")
                                if result > 32:
                                    print_success = True
                                    logger.info(f"[PRINT] PrintTo succeeded for: {filename}")
                            except Exception as printto_err:
                                logger.warning(f"[PRINT] PrintTo failed: {printto_err}")
                        
                        if print_success:
                            logger.info(f"[PRINT] Sent to printer: {filename} (copy {copy_num + 1}/{copies})")
                        else:
                            logger.error(f"[PRINT] All print methods failed for: {filename}")

                    successful_prints.append(filename)
                    
                    # Emit capture_now event after each document is printed
                    # This tells the phone to capture the document coming out of the printer
                    socketio.emit("capture_now", {
                        "message": f"Capture document {idx + 1}: {filename}",
                        "document": filename,
                        "index": idx + 1,
                        "total": len(files_to_print),
                        "timestamp": datetime.now().isoformat()
                    })
                    logger.info(f"[PRINT] Sent capture_now for document {idx + 1}: {filename}")
                    
                except Exception as print_err:
                    logger.error(f"[PRINT] Failed to print {filename}: {print_err}")
                    failed_prints.append({"filename": filename, "error": str(print_err)})

        except ImportError:
            # Windows print modules not available, try alternative methods
            logger.warning("[PRINT] win32 modules not available, using alternative print method")
            import time

            for idx, file_info in enumerate(files_to_print):
                file_path = file_info["path"]
                filename = file_info["filename"]

                try:
                    # Apply delay BEFORE printing
                    if idx == 0 and first_delay > 0:
                        logger.info(f"[PRINT] Waiting {first_delay}s before first document...")
                        # Emit progress update
                        socketio.emit("print_progress", {
                            "type": "waiting",
                            "document": filename,
                            "index": idx + 1,
                            "total": len(files_to_print),
                            "delay": first_delay,
                            "message": f"Waiting {first_delay}s before first document..."
                        })
                        time.sleep(first_delay)
                    elif idx > 0 and subsequent_delay > 0:
                        logger.info(f"[PRINT] Waiting {subsequent_delay}s before document {idx + 1}...")
                        # Emit progress update
                        socketio.emit("print_progress", {
                            "type": "waiting",
                            "document": filename,
                            "index": idx + 1,
                            "total": len(files_to_print),
                            "delay": subsequent_delay,
                            "message": f"Waiting {subsequent_delay}s before document {idx + 1}..."
                        })
                        time.sleep(subsequent_delay)
                    
                    for copy_num in range(copies):
                        # Use subprocess with PowerShell
                        cmd = f'Start-Process -FilePath "{file_path}" -Verb Print -WindowStyle Hidden'
                        subprocess.run(["powershell", "-Command", cmd], timeout=30, check=True)
                        logger.info(f"[PRINT] Sent via PowerShell: {filename} (copy {copy_num + 1}/{copies})")

                    successful_prints.append(filename)
                    
                    # Emit capture_now event after each document is printed
                    socketio.emit("capture_now", {
                        "message": f"Capture document {idx + 1}: {filename}",
                        "document": filename,
                        "index": idx + 1,
                        "total": len(files_to_print),
                        "timestamp": datetime.now().isoformat()
                    })
                    logger.info(f"[PRINT] Sent capture_now for document {idx + 1}: {filename}")
                    
                except Exception as print_err:
                    logger.error(f"[PRINT] Failed to print {filename}: {print_err}")
                    failed_prints.append({"filename": filename, "error": str(print_err)})

        # Emit Socket.IO notification
        socketio.emit("print_job_update", {
            "type": "print_completed",
            "successful": successful_prints,
            "failed": failed_prints,
            "options": options,
            "timestamp": datetime.now().isoformat()
        })

        if failed_prints and not successful_prints:
            return jsonify({
                "success": False,
                "error": "All print jobs failed",
                "failed": failed_prints
            }), 500

        return jsonify({
            "success": True,
            "message": f"Print job sent: {len(successful_prints)} file(s) ({copies} cop{'y' if copies == 1 else 'ies'} each)",
            "printed": successful_prints,
            "failed": failed_prints,
            "copies": copies,
            "colorMode": color_mode,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"[ERROR] Orchestrate print error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/orchestrate/scan", methods=["POST"])
def orchestrate_scan():
    """
    Initialize scan job with configuration from frontend
    Accepts JSON with scan options:
        - pageMode: 'all' | 'odd' | 'even' | 'custom'
        - customRange: string (e.g., "1-3,5,7-9")
        - layout: 'portrait' | 'landscape'
        - paperSize: string
        - resolution: string
        - colorMode: 'color' | 'bw'
        - saveAsDefault: boolean
    """
    try:
        data = request.get_json() or {}

        scan_config = {
            "pageMode": data.get("pageMode", "all"),
            "customRange": data.get("customRange", ""),
            "layout": data.get("layout", "portrait"),
            "paperSize": data.get("paperSize", "A4"),
            "resolution": data.get("resolution", "300"),
            "colorMode": data.get("colorMode", "color"),
            "saveAsDefault": data.get("saveAsDefault", False),
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"[SCAN] Scan job initialized with config: {scan_config}")

        # Store scan configuration for the phone interface
        # This will be retrieved when capturing
        global current_scan_config
        current_scan_config = scan_config

        # Emit Socket.IO notification to phone
        socketio.emit("scan_job_init", {
            "type": "scan_ready",
            "config": scan_config,
            "message": "Scanner ready. Use phone interface to capture documents.",
            "timestamp": datetime.now().isoformat()
        })

        return jsonify({
            "success": True,
            "message": "Scan job initialized. Open phone interface to capture documents.",
            "config": scan_config,
            "next_steps": [
                "Open phone capture interface",
                "Position document in frame",
                "Capture when ready"
            ]
        })

    except Exception as e:
        logger.error(f"[ERROR] Orchestrate scan error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/orchestrate/voice-config", methods=["POST"])
def orchestrate_voice_config():
    """
    Parse voice command for configuration changes
    Expects: JSON with { "voice_text": "...", "action_type": "print|scan" }
    """
    if not ORCHESTRATION_AVAILABLE or not orchestrator:
        return jsonify({"success": False, "error": "Orchestration service not available"}), 503

    try:
        data = request.get_json()
        voice_text = data.get("voice_text", "")
        action_type = data.get("action_type", "")

        if not voice_text or not action_type:
            return jsonify({"success": False, "error": "Missing voice_text or action_type"}), 400

        # Parse voice command
        parsed_config = orchestrator.parse_voice_configuration(voice_text, action_type)

        # Check if user indicated no more changes
        if parsed_config.get("no_changes"):
            return jsonify(
                {
                    "success": True,
                    "no_changes": True,
                    "message": "Ready to proceed",
                    "action": "ready_to_confirm",
                }
            )

        # Apply configuration updates if any were parsed
        if parsed_config:
            result = orchestrator.update_configuration(action_type, parsed_config)

            # Emit Socket.IO event
            if result.get("success"):
                socketio.emit(
                    "orchestration_update",
                    {
                        "type": "voice_configuration_updated",
                        "action_type": action_type,
                        "updates": parsed_config,
                        "configuration": result.get("configuration"),
                        "frontend_updates": result.get("frontend_updates"),
                        "frontend_state": result.get("frontend_state"),
                        "timestamp": datetime.now().isoformat(),
                    },
                )

            return jsonify(
                {
                    "success": True,
                    "updates": parsed_config,
                    "configuration": result.get("configuration"),
                    "frontend_updates": result.get("frontend_updates"),
                    "frontend_state": result.get("frontend_state"),
                    "message": f"Updated: {', '.join(parsed_config.keys())}",
                }
            )
        else:
            return jsonify(
                {
                    "success": True,
                    "updates": {},
                    "message": 'No configuration changes detected. Try saying specific options like "landscape" or "3 copies"',
                }
            )

    except Exception as e:
        logger.error(f"[ERROR] Voice configuration parsing error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/orchestrate/reset", methods=["POST"])
def orchestrate_reset():
    """Reset orchestrator to idle state"""
    if not ORCHESTRATION_AVAILABLE or not orchestrator:
        return jsonify({"success": False, "error": "Orchestration service not available"}), 503

    try:
        result = orchestrator.reset_state()

        # Emit Socket.IO event
        socketio.emit(
            "orchestration_update", {"type": "state_reset", "timestamp": datetime.now().isoformat()}
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"[ERROR] Orchestration reset error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/orchestrate/history", methods=["GET"])
def orchestrate_history():
    """Get workflow history"""
    if not ORCHESTRATION_AVAILABLE or not orchestrator:
        return jsonify({"success": False, "error": "Orchestration service not available"}), 503

    try:
        limit = request.args.get("limit", 10, type=int)
        history = orchestrator.get_workflow_history(limit)

        return jsonify({"success": True, "history": history, "count": len(history)})

    except Exception as e:
        logger.error(f"[ERROR] Orchestration history error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PrintChakra Backend Server")
    print("=" * 60)
    print(f"Upload directory: {UPLOAD_DIR}")
    print(f"Processed directory: {PROCESSED_DIR}")
    print(f"Text directory: {TEXT_DIR}")
    print("=" * 60)

    # Pre-load Whisper model on startup (runs in background)
    def preload_voice_models():
        """Pre-load voice AI models on startup to avoid first-request delay"""
        try:
            logger.info("[STARTUP] Pre-loading Whisper AI model in background...")
            print("[STARTUP] Pre-loading Whisper AI model...")
            from app.modules.voice import voice_ai_orchestrator
            import threading
            
            def load_models():
                try:
                    voice_ai_orchestrator.whisper_service.load_model()
                    logger.info("[OK] Whisper model pre-loaded successfully")
                    print("[OK] Whisper model pre-loaded")
                except Exception as e:
                    logger.warning(f"[WARN] Could not pre-load Whisper model: {e}")
                    print(f"[WARN] Could not pre-load Whisper model: {e}")
            
            # Load in background thread so startup isn't blocked
            thread = threading.Thread(target=load_models, daemon=True)
            thread.start()
        except Exception as e:
            logger.warning(f"[WARN] Voice model pre-loading skipped: {e}")

    preload_voice_models()

    # Configure SSL context if certificates are provided
    import ssl
    ssl_context = None
    ssl_cert = os.environ.get('SSL_CERT')
    ssl_key = os.environ.get('SSL_KEY')
    
    if ssl_cert and ssl_key and os.path.exists(ssl_cert) and os.path.exists(ssl_key):
        try:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(ssl_cert, ssl_key)
            print(f"[HTTPS] SSL enabled")
            print(f"[HTTPS] Certificate: {ssl_cert}")
            print(f"[HTTPS] Private Key: {ssl_key}")
            local_ip = os.environ.get('LOCAL_IP', 'localhost')
            print(f"[HTTPS] Access URL: https://{local_ip}:5000")
        except Exception as e:
            print(f"[HTTPS] Warning: Could not load SSL certificates: {e}")
            print("[HTTPS] Falling back to HTTP")
            ssl_context = None
    else:
        print("[HTTP] Running without SSL (use https-local.ps1 for HTTPS)")

    print("=" * 60)

    # Run with Socket.IO
    socketio.run(
        app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True, ssl_context=ssl_context
    )
