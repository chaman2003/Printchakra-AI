"""
AI Orchestration Service - Intelligent Print & Scan Orchestration
Provides hands-free, context-aware document operations with autonomous workflow execution
"""

import logging
import os
import re
import threading
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .parsers import detect_intent_and_parameters
from .runtime_mixin import OrchestratorRuntimeMixin

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Supported orchestration intents"""

    PRINT = "print"
    SCAN = "scan"
    VIEW_STATUS = "view_status"
    CONFIGURE = "configure"
    LIST_DOCUMENTS = "list_documents"
    HELP = "help"
    UNKNOWN = "unknown"


class WorkflowState(Enum):
    """Workflow execution states"""

    IDLE = "idle"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    EXECUTING = "executing"
    CONFIGURING = "configuring"
    COMPLETED = "completed"
    FAILED = "failed"


class PrintScanOrchestrator(OrchestratorRuntimeMixin):
    """
    Intelligent orchestrator for print and scan operations
    Handles intent detection, workflow execution, and state management
    """

    def __init__(self, data_dir: str):
        """
        Initialize orchestrator

        Args:
            data_dir: Base data directory for documents
        """
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, "processed")
        self.upload_dir = os.path.join(data_dir, "uploads")

        # Workflow state
        self.current_state = WorkflowState.IDLE
        self.pending_action = None
        self.selected_document = None
        self.configuration = self._default_configuration()
        self.workflow_history = []
        self.state_lock = threading.Lock()

        logger.info("[ORCHESTRATOR] AI Orchestrator initialized")

    def _default_configuration(self) -> Dict[str, Any]:
        """Get default print/scan configuration"""
        return {
            "print": {
                "copies": 1,
                "paper_size": "A4",
                "paper_size_custom": "",
                "orientation": "portrait",
                "color_mode": "color",
                "resolution": 300,
                "pages": "all",
                "custom_range": "",
                "scale": 100,
                "scale_custom": "",
                "margins": "default",
                "margins_custom": "",
                "pages_per_sheet": "1",
                "pages_per_sheet_custom": "",
                "duplex": False,
                "quality": "normal",
            },
            "scan": {
                "mode": "single",
                "text_mode": False,
                "page_mode": "all",
                "custom_range": "",
                "resolution": 300,
                "resolution_custom": "",
                "color_mode": "color",
                "paper_size": "A4",
                "paper_size_custom": "",
                "orientation": "portrait",
                "format": "pdf",
                "quality": "normal",
            },
        }

    _PRINT_FRONTEND_FIELDS = {
        "color_mode": ("printColorMode", lambda v: v),
        "orientation": ("printLayout", lambda v: v),
        "paper_size": ("printPaperSize", lambda v: v),
        "paper_size_custom": ("printPaperSizeCustom", lambda v: v),
        "resolution": ("printResolution", lambda v: str(v)),
        "pages": ("printPages", lambda v: v),
        "custom_range": ("printCustomRange", lambda v: v),
        "scale": ("printScale", lambda v: str(v)),
        "scale_custom": ("printScaleCustom", lambda v: v),
        "margins": ("printMargins", lambda v: v),
        "margins_custom": ("printMarginsCustom", lambda v: v),
        "pages_per_sheet": ("printPagesPerSheet", lambda v: str(v)),
        "pages_per_sheet_custom": ("printPagesPerSheetCustom", lambda v: v),
        "copies": ("printCopies", lambda v: str(v)),
        "duplex": ("printDuplex", lambda v: bool(v)),
        "quality": ("printQuality", lambda v: v),
    }

    _SCAN_FRONTEND_FIELDS = {
        "mode": ("scanMode", lambda v: v),
        "text_mode": ("scanTextMode", lambda v: bool(v)),
        "page_mode": ("scanPageMode", lambda v: v),
        "custom_range": ("scanCustomRange", lambda v: v),
        "orientation": ("scanLayout", lambda v: v),
        "paper_size": ("scanPaperSize", lambda v: v),
        "paper_size_custom": ("scanPaperSizeCustom", lambda v: v),
        "resolution": ("scanResolution", lambda v: str(v)),
        "resolution_custom": ("scanResolutionCustom", lambda v: v),
        "color_mode": ("scanColorMode", lambda v: v),
        "format": ("scanFormat", lambda v: v),
        "quality": ("scanQuality", lambda v: v),
    }

    def _convert_to_frontend_options(self, action_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Map internal configuration to Dashboard orchestrateOptions fields"""
        if action_type not in ["print", "scan"]:
            return {}

        field_map = (
            self._PRINT_FRONTEND_FIELDS if action_type == "print" else self._SCAN_FRONTEND_FIELDS
        )

        source = dict(data or {})
        # Normalize synonyms that may appear during parsing
        if action_type == "scan" and "page_size" in source and "paper_size" not in source:
            source["paper_size"] = source.pop("page_size")

        options = {}
        for key, (target, converter) in field_map.items():
            if key in source and source[key] is not None:
                try:
                    options[target] = converter(source[key])
                except Exception:
                    options[target] = source[key]

        return options

    def _build_frontend_state(self, action_type: str) -> Dict[str, Any]:
        """Generate a frontend-friendly snapshot of the current configuration"""
        return {
            "mode": action_type,
            "options": self._convert_to_frontend_options(action_type, self.configuration[action_type]),
            "workflowState": self.current_state.value,
            "document": self.selected_document if action_type == "print" else None,
            "timestamp": datetime.now().isoformat(),
        }

    def _build_frontend_updates(self, action_type: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Map partial configuration updates to frontend option keys"""
        return self._convert_to_frontend_options(action_type, updates or {})

    def detect_intent(self, user_input: str) -> Tuple[IntentType, Dict[str, Any]]:
        """
        Detect user intent from natural language input

        Args:
            user_input: User's text or voice input

        Returns:
            Tuple of (intent_type, extracted_parameters)
        """
        intent_key, parameters = detect_intent_and_parameters(user_input)
        intent_map = {
            "print": IntentType.PRINT,
            "scan": IntentType.SCAN,
            "view_status": IntentType.VIEW_STATUS,
            "configure": IntentType.CONFIGURE,
            "list_documents": IntentType.LIST_DOCUMENTS,
            "help": IntentType.HELP,
            "unknown": IntentType.UNKNOWN,
        }
        return intent_map.get(intent_key, IntentType.UNKNOWN), parameters

    def process_command(self, user_input: str, force_voice_triggered: bool = False) -> Dict[str, Any]:
        """
        Process user command and execute workflow

        Args:
            user_input: Natural language command
            force_voice_triggered: When True, treat this command as coming from the
                hands-free voice pipeline so the workflow skips manual confirmation
                and enters configuration mode automatically.

        Returns:
            Response dictionary with action results
        """
        with self.state_lock:
            intent, parameters = self.detect_intent(user_input)

            if force_voice_triggered:
                parameters["voice_triggered"] = True

            logger.info(f"[ORCHESTRATOR] Detected intent: {intent.value}, Parameters: {parameters}")

            # Route to appropriate handler
            if intent == IntentType.PRINT:
                return self._handle_print_intent(parameters)
            elif intent == IntentType.SCAN:
                return self._handle_scan_intent(parameters)
            elif intent == IntentType.VIEW_STATUS:
                return self._handle_status_inquiry()
            elif intent == IntentType.CONFIGURE:
                return self._handle_configuration(parameters)
            elif intent == IntentType.LIST_DOCUMENTS:
                return self._handle_list_documents()
            elif intent == IntentType.HELP:
                return self._handle_help_request()
            else:
                return {
                    "success": False,
                    "intent": "unknown",
                    "message": 'I didn\'t understand that command. Try "help" to see what I can do.',
                    "suggestions": [
                        "Print a document",
                        "Scan a document",
                        "List available documents",
                        "Show status",
                        "Help",
                    ],
                }

    def _handle_print_intent(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle print intent

        Args:
            parameters: Extracted parameters

        Returns:
            Response dictionary
        """
        # Check if this is a voice-triggered orchestration request
        voice_triggered = parameters.pop("voice_triggered", False)

        # Get available documents
        documents = self._get_available_documents()

        if not documents:
            return {
                "success": False,
                "intent": "print",
                "message": "No documents available to print. Please upload or scan a document first.",
                "requires_action": "upload_or_scan",
            }
        
        # Handle relative document selection (e.g., "print last 2 documents")
        if parameters.get("document_selection") == "relative":
            doc_count = parameters.get("document_count", 1)
            doc_position = parameters.get("document_position", "last")
            
            if doc_position == "last":
                selected_docs = documents[:doc_count]
            else:  # first
                selected_docs = documents[-doc_count:][::-1]
            
            # For multiple documents, store list; for single, store the document
            if len(selected_docs) == 1:
                self.selected_document = selected_docs[0]
            else:
                # Multiple document handling - for now, select the most recent
                self.selected_document = selected_docs[0]
                parameters["multiple_documents"] = [doc["filename"] for doc in selected_docs]
            
            logger.info(f"[ORCHESTRATOR] Selected {len(selected_docs)} document(s): {[d['filename'] for d in selected_docs]}")

        # If only one document available, select it automatically
        elif len(documents) == 1:
            self.selected_document = documents[0]

        # Update configuration with parameters
        if parameters:
            self.configuration["print"].update(parameters)

        # Prepare confirmation message
        doc_name = self.selected_document["filename"] if self.selected_document else "a document"
        config = self.configuration["print"]

        config_summary = f"{config['copies']} cop{'y' if config['copies'] == 1 else 'ies'}"
        if config["color_mode"]:
            config_summary += f", {config['color_mode']}"
        if config["duplex"]:
            config_summary += ", duplex"
        if config["paper_size"]:
            config_summary += f", {config['paper_size']}"

        # Set state based on trigger type
        if voice_triggered:
            self.current_state = WorkflowState.CONFIGURING
            message = "What options would you like to change or edit?"
        else:
            self.current_state = WorkflowState.AWAITING_CONFIRMATION
            message = f"Ready to print {doc_name} ({config_summary}). Shall we proceed?"

        self.pending_action = {
            "type": "print",
            "document": self.selected_document,
            "configuration": self.configuration["print"].copy(),
            "voice_triggered": voice_triggered,
        }

        return {
            "success": True,
            "intent": "print",
            "requires_confirmation": not voice_triggered,
            "requires_options": voice_triggered,
            "message": message,
            "document": self.selected_document,
            "configuration": self.configuration["print"],
            "available_documents": documents if not self.selected_document else None,
            "workflow_state": self.current_state.value,
            "open_ui": voice_triggered,
            "skip_mode_selection": voice_triggered,
            "frontend_state": self._build_frontend_state("print"),
        }

    def _handle_scan_intent(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle scan intent

        Args:
            parameters: Extracted parameters

        Returns:
            Response dictionary
        """
        # Check if this is a voice-triggered orchestration request
        voice_triggered = parameters.pop("voice_triggered", False)

        # Update configuration with parameters
        if parameters:
            self.configuration["scan"].update(parameters)

        config = self.configuration["scan"]
        config_summary = (
            f"{config['resolution']} DPI, {config['color_mode']}, {config['format'].upper()}"
        )

        # Set state based on trigger type
        if voice_triggered:
            self.current_state = WorkflowState.CONFIGURING
            message = "What options would you like to change or edit?"
        else:
            self.current_state = WorkflowState.AWAITING_CONFIRMATION
            message = f"Ready to scan document ({config_summary}). Shall we proceed?"

        self.pending_action = {
            "type": "scan",
            "configuration": self.configuration["scan"].copy(),
            "voice_triggered": voice_triggered,
        }

        return {
            "success": True,
            "intent": "scan",
            "requires_confirmation": not voice_triggered,
            "requires_options": voice_triggered,
            "message": message,
            "configuration": self.configuration["scan"],
            "workflow_state": self.current_state.value,
            "next_step": (
                "Open phone capture interface or use connected scanner"
                if not voice_triggered
                else None
            ),
            "open_ui": voice_triggered,
            "skip_mode_selection": voice_triggered,
            "frontend_state": self._build_frontend_state("scan"),
        }

# One orchestrator per data directory, so each user's workflow state and documents stay separate.
_orchestrators: Dict[str, PrintScanOrchestrator] = {}
_orchestrator_lock = threading.Lock()


def get_orchestrator(data_dir: str) -> PrintScanOrchestrator:
    """
    Get or create the orchestrator for a data directory

    Args:
        data_dir: Data directory path (a user's own folder when logged in)

    Returns:
        PrintScanOrchestrator instance
    """
    with _orchestrator_lock:
        if data_dir not in _orchestrators:
            _orchestrators[data_dir] = PrintScanOrchestrator(data_dir)
        return _orchestrators[data_dir]
