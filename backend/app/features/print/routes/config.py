"""
PrintChakra Backend - Print Config Routes

Printer configuration and settings endpoints.
"""

import logging
from flask import jsonify, request
from app.features.print.routes import print_bp
from app.core.middleware.cors import create_options_response
from app.core.paths import IS_CLOUD

logger = logging.getLogger(__name__)


@print_bp.route("/printers", methods=["GET", "OPTIONS"])
def list_printers():
    """List all available printers."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        from app.features.print.services.printer_service import PrinterService
        
        printer_service = PrinterService()
        printers = printer_service.list_printers()
        
        return jsonify({
            "success": True,
            "printers": printers,
            "total": len(printers)
        })
    
    except ImportError:
        # Fallback to system call
        try:
            import subprocess
            result = subprocess.run(
                ["wmic", "printer", "get", "name"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            printers = [{"name": line.strip(), "status": "unknown"} for line in lines if line.strip()]
            
            return jsonify({
                "success": True,
                "printers": printers,
                "total": len(printers)
            })
        
        except Exception as e:
            return jsonify({
                "success": True,
                "printers": [],
                "total": 0,
                "error": f"Could not list printers: {e}"
            })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@print_bp.route("/printers/default", methods=["GET", "OPTIONS"])
def get_default_printer():
    """Get the default printer."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        from app.features.print.services.printer_service import PrinterService
        
        printer_service = PrinterService()
        default = printer_service.get_default_printer()
        
        return jsonify({
            "success": True,
            "printer": default
        })
    
    except ImportError:
        try:
            import subprocess
            result = subprocess.run(
                ["wmic", "printer", "where", "default=true", "get", "name"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            lines = result.stdout.strip().split("\n")[1:]
            default = lines[0].strip() if lines and lines[0].strip() else None
            
            return jsonify({
                "success": True,
                "printer": {"name": default} if default else None
            })
        
        except Exception:
            return jsonify({
                "success": True,
                "printer": None
            })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@print_bp.route("/printers/default", methods=["POST", "OPTIONS"])
def set_default_printer():
    """Set the default printer."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        data = request.get_json()
        
        if not data or "printer" not in data:
            return jsonify({"success": False, "error": "No printer name provided"}), 400
        
        printer_name = data.get("printer")

        if IS_CLOUD:
            # The browser print dialog lets the user pick the real printer; there is nothing to set here.
            return jsonify({"success": True, "message": "Printing uses the browser print dialog"})
        
        from app.features.print.services.printer_service import PrinterService
        
        printer_service = PrinterService()
        result = printer_service.set_default_printer(printer_name)
        
        if result:
            logger.info(f"[OK] Default printer set to: {printer_name}")
            return jsonify({"success": True, "message": "Default printer set"})
        else:
            return jsonify({"success": False, "error": "Failed to set default printer"}), 400
    
    except ImportError:
        return jsonify({
            "success": False,
            "error": "Printer service not available"
        }), 503
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@print_bp.route("/printers/<printer_name>/status", methods=["GET", "OPTIONS"])
def get_printer_status(printer_name):
    """Get printer status."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        from app.features.print.services.printer_service import PrinterService
        
        printer_service = PrinterService()
        status = printer_service.get_printer_status(printer_name)
        
        return jsonify({
            "success": True,
            "printer": printer_name,
            "status": status
        })
    
    except ImportError:
        return jsonify({
            "success": True,
            "printer": printer_name,
            "status": {
                "online": True,
                "state": "unknown",
                "jobs_pending": 0
            },
            "note": "Printer service not available"
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@print_bp.route("/printers/<printer_name>/capabilities", methods=["GET", "OPTIONS"])
def get_printer_capabilities(printer_name):
    """Get printer capabilities (color, duplex, paper sizes, etc.)."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        from app.features.print.services.printer_service import PrinterService
        
        printer_service = PrinterService()
        capabilities = printer_service.get_printer_capabilities(printer_name)
        
        return jsonify({
            "success": True,
            "printer": printer_name,
            "capabilities": capabilities
        })
    
    except ImportError:
        return jsonify({
            "success": True,
            "printer": printer_name,
            "capabilities": {
                "color": True,
                "duplex": False,
                "paper_sizes": ["A4", "Letter"],
                "default_paper_size": "A4"
            },
            "note": "Printer service not available"
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@print_bp.route("/settings", methods=["GET", "OPTIONS"])
def get_print_settings():
    """Get global print settings."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        from app.features.print.services.printer_service import PrinterService
        
        printer_service = PrinterService()
        settings = printer_service.get_settings()
        
        return jsonify({
            "success": True,
            "settings": settings
        })
    
    except ImportError:
        return jsonify({
            "success": True,
            "settings": {
                "default_copies": 1,
                "default_color": True,
                "default_duplex": False,
                "default_paper_size": "A4",
                "auto_fit_page": True
            }
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@print_bp.route("/settings", methods=["POST", "OPTIONS"])
def update_print_settings():
    """Update global print settings."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "error": "No settings provided"}), 400
        
        from app.features.print.services.printer_service import PrinterService
        
        printer_service = PrinterService()
        printer_service.update_settings(data)
        
        logger.info("[OK] Print settings updated")
        
        return jsonify({"success": True, "message": "Settings updated"})
    
    except ImportError:
        return jsonify({
            "success": False,
            "error": "Printer service not available"
        }), 503
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
