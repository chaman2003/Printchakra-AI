"""
PrintChakra Backend - Orchestration Command Routes

Voice and text command execution endpoints.
"""

import json
import logging
import os
from flask import jsonify, request, g
from app.features.orchestration.routes import orchestration_bp
from app.core.config import get_data_dirs
from app.core.middleware.cors import create_options_response
from app.core.paths import IS_CLOUD
from app.modules.orchestration import get_orchestrator

logger = logging.getLogger(__name__)

# Fields that make up the orchestrator's workflow state.
_STATE_FIELDS = ("current_state", "pending_action", "selected_document", "configuration", "workflow_history")


def _state_file() -> str:
    return os.path.join(get_data_dirs()["DATA_DIR"], "state", "orchestrator.json")


def _load_state(orchestrator) -> None:
    """Restore workflow state saved by an earlier request, possibly on another serverless instance."""
    from app.modules.orchestration.core_service import WorkflowState

    try:
        with open(_state_file(), "r", encoding="utf-8") as handle:
            saved = json.load(handle)
    except (OSError, ValueError):
        return
    try:
        orchestrator.current_state = WorkflowState(saved.get("current_state", "idle"))
    except ValueError:
        orchestrator.current_state = WorkflowState.IDLE
    for field in _STATE_FIELDS[1:]:
        if field in saved:
            setattr(orchestrator, field, saved[field])


def _save_state(orchestrator) -> None:
    state = {field: getattr(orchestrator, field, None) for field in _STATE_FIELDS}
    state["current_state"] = getattr(state["current_state"], "value", "idle")
    path = _state_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, default=str)


def _get_orchestrator():
    """Get shared orchestrator instance."""
    orchestrator = get_orchestrator(get_data_dirs()["DATA_DIR"])
    if IS_CLOUD:
        # Each request may land on a different instance, so state travels through the data directory.
        _load_state(orchestrator)
        g.orchestrator = orchestrator
    return orchestrator


@orchestration_bp.after_request
def _persist_orchestrator_state(response):
    orchestrator = g.pop("orchestrator", None)
    if IS_CLOUD and orchestrator is not None:
        try:
            _save_state(orchestrator)
        except OSError as exc:
            logger.warning("Could not save orchestrator state: %s", exc)
    return response


def _normalize_command_text(command: str, parameters: dict) -> str:
    """Build a basic natural-language command text from command + parameters."""
    cmd = (command or "").strip().lower()
    params = parameters or {}
    if not cmd:
        return ""

    if cmd in {"print", "scan", "status", "help"}:
        return cmd

    if cmd == "list_files":
        return "list documents"
    if cmd == "list_printers":
        return "printer status"

    if cmd == "ocr":
        document = (params.get("document") or "").strip()
        return f"extract text from {document}".strip()

    if cmd == "convert":
        document = (params.get("document") or "").strip()
        target_format = (params.get("target_format") or "pdf").strip()
        if document:
            return f"convert {document} to {target_format}"
        return f"convert to {target_format}"

    return cmd


def run_unified_orchestration(text: str, source: str = "api", force_voice_triggered: bool = False) -> dict:
    """Run orchestration through the shared orchestrator for both voice and text."""
    command_text = (text or "").strip()
    if not command_text:
        return {
            "success": False,
            "error": "No command text provided",
            "orchestration_trigger": False,
        }

    orchestrator = _get_orchestrator()
    result = orchestrator.process_command(
        command_text,
        force_voice_triggered=force_voice_triggered,
    )

    intent = (result.get("intent") or "").lower()
    trigger_mode = intent if intent in {"print", "scan"} else None

    return {
        "success": bool(result.get("success")),
        "source": source,
        "command_text": command_text,
        "intent": intent,
        "orchestration_trigger": bool(result.get("success") and trigger_mode),
        "orchestration_mode": trigger_mode,
        "awaiting_confirmation": bool(result.get("requires_confirmation", False)),
        "pending_mode": trigger_mode if result.get("requires_confirmation") else None,
        "orchestration": {
            "backend_result": result,
            "frontend_state": result.get("frontend_state"),
            "frontend_updates": result.get("frontend_updates"),
        },
        "result": result,
        "message": result.get("message", ""),
    }


@orchestration_bp.route("/command", methods=["POST", "OPTIONS"])
def execute_command():
    """Execute an orchestration command."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        data = request.get_json() or {}

        text = (data.get("text") or "").strip()
        command = (data.get("command") or "").strip()
        parameters = data.get("parameters", {}) or {}
        source = data.get("source", "api")
        force_voice_triggered = bool(data.get("force_voice_triggered", source == "voice"))

        if not text and command:
            text = _normalize_command_text(command, parameters)

        if not text:
            return jsonify({"success": False, "error": "No command or text provided"}), 400

        logger.info(f"Executing unified orchestration: '{text}' (source: {source})")
        unified = run_unified_orchestration(text, source, force_voice_triggered)

        return jsonify(unified), 200 if unified.get("success") else 400

    except Exception as e:
        logger.error(f"Command execution error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@orchestration_bp.route("/command/parse", methods=["POST", "OPTIONS"])
def parse_command():
    """Parse a natural language command into structured format."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        data = request.get_json() or {}
        text = (data.get("text") or "").strip()

        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400

        parsed = parse_natural_language_command(text)

        return jsonify({
            "success": True,
            "original": text,
            "parsed": parsed,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@orchestration_bp.route("/commands", methods=["GET", "OPTIONS"])
def list_available_commands():
    """List all available orchestration commands."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        commands = [
            {
                "name": "print",
                "description": "Print a document",
                "parameters": ["document", "printer", "copies", "color", "duplex"],
                "examples": ["print the document", "print 2 copies"]
            },
            {
                "name": "scan",
                "description": "Scan a document",
                "parameters": ["format", "resolution", "color"],
                "examples": ["scan document", "scan to PDF"]
            },
            {
                "name": "ocr",
                "description": "Extract text from document",
                "parameters": ["document", "language"],
                "examples": ["extract text", "OCR this document"]
            },
            {
                "name": "convert",
                "description": "Convert document format",
                "parameters": ["document", "target_format"],
                "examples": ["convert to PDF", "make this a PDF"]
            },
            {
                "name": "list_files",
                "description": "List documents",
                "parameters": ["folder", "type"],
                "examples": ["list documents", "show my files"]
            },
            {
                "name": "list_printers",
                "description": "List available printers",
                "parameters": [],
                "examples": ["list printers", "what printers are available"]
            },
            {
                "name": "status",
                "description": "Get system status",
                "parameters": ["service"],
                "examples": ["system status", "is the printer ready"]
            }
        ]
        
        return jsonify({
            "success": True,
            "commands": commands,
            "total": len(commands)
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@orchestration_bp.route("/command/history", methods=["GET", "OPTIONS"])
def get_command_history():
    """Get command execution history."""
    if request.method == "OPTIONS":
        return create_options_response()
    
    try:
        limit = max(1, min(request.args.get("limit", 50, type=int), 200))
        orchestrator = _get_orchestrator()
        history = orchestrator.get_history(limit)

        return jsonify({"success": True, "history": history, "total": len(history)})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def execute_orchestration_command(command: str, parameters: dict) -> dict:
    """Compatibility wrapper for legacy callers."""
    text = _normalize_command_text(command, parameters)
    return run_unified_orchestration(text, source="api", force_voice_triggered=False)


def execute_print_command(params: dict) -> dict:
    """Execute print command."""
    return execute_orchestration_command("print", params)


def execute_scan_command(params: dict) -> dict:
    """Execute scan command."""
    return execute_orchestration_command("scan", params)


def execute_ocr_command(params: dict) -> dict:
    """Execute OCR command."""
    return execute_orchestration_command("ocr", params)


def execute_convert_command(params: dict) -> dict:
    """Execute convert command."""
    return execute_orchestration_command("convert", params)


def execute_list_files_command(params: dict) -> dict:
    """Execute list files command."""
    return execute_orchestration_command("list_files", params)


def execute_list_printers_command(params: dict) -> dict:
    """Execute list printers command."""
    return execute_orchestration_command("list_printers", params)


def execute_status_command(params: dict) -> dict:
    """Execute status command."""
    return execute_orchestration_command("status", params)


def parse_natural_language_command(text: str) -> dict:
    """Parse natural language into structured command."""
    orchestrator = _get_orchestrator()
    intent, parameters = orchestrator.detect_intent(text)

    command = None
    if intent.value == "print":
        command = "print"
    elif intent.value == "scan":
        command = "scan"
    elif intent.value == "list_documents":
        command = "list_files"
    elif intent.value == "view_status":
        command = "status"
    elif intent.value == "help":
        command = "help"

    return {
        "command": command,
        "confidence": 0.9 if command else 0.0,
        "parameters": parameters,
        "intent": intent.value,
        "message": "Could not parse command" if not command else "",
    }


@orchestration_bp.route("/confirm", methods=["POST", "OPTIONS"])
def confirm_command():
    """Confirm pending orchestration action."""
    if request.method == "OPTIONS":
        return create_options_response()

    try:
        result = _get_orchestrator().confirm_action()
        return jsonify({"success": bool(result.get("success")), "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@orchestration_bp.route("/cancel", methods=["POST", "OPTIONS"])
def cancel_command():
    """Cancel pending orchestration action."""
    if request.method == "OPTIONS":
        return create_options_response()

    try:
        result = _get_orchestrator().cancel_action()
        return jsonify({"success": bool(result.get("success")), "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@orchestration_bp.route("/status", methods=["GET", "OPTIONS"])
def get_status():
    """Get orchestration status."""
    if request.method == "OPTIONS":
        return create_options_response()

    try:
        result = _get_orchestrator().process_command("status")
        return jsonify({"success": bool(result.get("success")), "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@orchestration_bp.route("/documents", methods=["GET", "OPTIONS"])
def list_documents():
    """List documents available for orchestration."""
    if request.method == "OPTIONS":
        return create_options_response()

    try:
        orchestrator = _get_orchestrator()
        documents = orchestrator._get_available_documents()
        return jsonify({"success": True, "documents": documents, "total": len(documents)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@orchestration_bp.route("/select", methods=["POST", "OPTIONS"])
def select_document():
    """Select document by filename or 1-based index."""
    if request.method == "OPTIONS":
        return create_options_response()

    try:
        data = request.get_json() or {}
        filename = (data.get("filename") or "").strip()
        index = data.get("index")

        orchestrator = _get_orchestrator()

        if not filename and index is None:
            return jsonify({"success": False, "error": "Provide filename or index"}), 400

        if index is not None and not filename:
            try:
                idx = int(index)
            except (TypeError, ValueError):
                return jsonify({"success": False, "error": "Index must be a number"}), 400

            documents = orchestrator._get_available_documents()
            if idx < 1 or idx > len(documents):
                return jsonify({"success": False, "error": "Index out of range"}), 400
            filename = documents[idx - 1]["filename"]

        result = orchestrator.select_document(filename)
        return jsonify({"success": bool(result.get("success")), "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@orchestration_bp.route("/configure", methods=["POST", "OPTIONS"])
def configure_action():
    """Update print or scan configuration."""
    if request.method == "OPTIONS":
        return create_options_response()

    try:
        data = request.get_json() or {}
        mode = (data.get("mode") or "").strip().lower()
        settings = data.get("settings", {}) or {}

        if mode not in {"print", "scan"}:
            return jsonify({"success": False, "error": "Mode must be print or scan"}), 400

        result = _get_orchestrator().update_configuration(mode, settings)
        return jsonify({"success": bool(result.get("success")), "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@orchestration_bp.route("/reset", methods=["POST", "OPTIONS"])
def reset_orchestration():
    """Reset orchestration state."""
    if request.method == "OPTIONS":
        return create_options_response()

    try:
        result = _get_orchestrator().reset_state()
        return jsonify({"success": bool(result.get("success")), "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@orchestration_bp.route("/history", methods=["GET", "OPTIONS"])
def orchestration_history():
    """Compatibility endpoint for orchestration history."""
    if request.method == "OPTIONS":
        return create_options_response()

    try:
        limit = max(1, min(request.args.get("limit", 50, type=int), 200))
        history = _get_orchestrator().get_history(limit)
        return jsonify({"success": True, "history": history, "total": len(history)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@orchestration_bp.route("/scan", methods=["POST", "OPTIONS"])
def start_scan_flow():
    """Compatibility endpoint used by Dashboard scan execution."""
    if request.method == "OPTIONS":
        return create_options_response()

    try:
        data = request.get_json() or {}
        orchestrator = _get_orchestrator()

        # Map Dashboard-style fields to orchestrator config keys.
        mapped_settings = {
            "mode": data.get("scanMode"),
            "text_mode": data.get("scanTextMode"),
            "page_mode": data.get("scanPageMode"),
            "custom_range": data.get("scanCustomRange"),
            "orientation": data.get("layout") or data.get("scanLayout"),
            "paper_size": data.get("paperSize") or data.get("scanPaperSize"),
            "resolution": int(data.get("scanResolution")) if str(data.get("scanResolution", "")).isdigit() else data.get("scanResolution"),
            "color_mode": data.get("scanColorMode"),
            "format": data.get("scanFormat"),
        }
        mapped_settings = {k: v for k, v in mapped_settings.items() if v not in (None, "")}

        if mapped_settings:
            orchestrator.update_configuration("scan", mapped_settings)

        result = orchestrator.process_command("scan", force_voice_triggered=True)
        return jsonify({
            "success": bool(result.get("success")),
            "message": result.get("message", "Scan workflow started"),
            "result": result,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
