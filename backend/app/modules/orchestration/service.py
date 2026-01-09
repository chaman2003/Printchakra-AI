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


class PrintScanOrchestrator:
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
        user_input_lower = user_input.lower()
        parameters = {}

        # Check for mode switching commands
        switch_keywords = ["switch to", "change to", "go to", "open", "show me", "navigate to"]
        if any(keyword in user_input_lower for keyword in switch_keywords):
            if "print" in user_input_lower:
                parameters["switch_mode"] = "print"
                parameters["voice_triggered"] = True
                return IntentType.PRINT, parameters
            elif "scan" in user_input_lower:
                parameters["switch_mode"] = "scan"
                parameters["voice_triggered"] = True
                return IntentType.SCAN, parameters

        # Print intent patterns
        print_keywords = ["print", "printing", "printout", "hard copy", "paper copy"]
        if any(keyword in user_input_lower for keyword in print_keywords):
            # Check for document selection patterns
            doc_patterns = [
                r"(?:print|printing)\s+(?:the\s+)?(?:last|latest|newest|most recent)\s+(\d+)\s+(?:documents?|files?)",
                r"(?:print|printing)\s+(?:the\s+)?(?:first|oldest)\s+(\d+)\s+(?:documents?|files?)",
                r"(?:print|printing)\s+(?:the\s+)?(?:last|latest|newest|recent)\s+(?:document|file|one)",
            ]
            
            for pattern in doc_patterns:
                match = re.search(pattern, user_input_lower)
                if match:
                    if match.lastindex and match.lastindex >= 1:
                        # Found number pattern like "last 2 documents"
                        count = int(match.group(1))
                        parameters["document_selection"] = "relative"
                        parameters["document_count"] = count
                        parameters["document_position"] = "last" if "last" in match.group(0) or "latest" in match.group(0) or "newest" in match.group(0) or "recent" in match.group(0) else "first"
                    else:
                        # Found single document pattern like "last document"
                        parameters["document_selection"] = "relative"
                        parameters["document_count"] = 1
                        parameters["document_position"] = "last"
                    break
            
            # Extract parameters
            if "color" in user_input_lower or "black" in user_input_lower or "grayscale" in user_input_lower:
                parameters["color_mode"] = "color"

            # Extract number of copies
            copy_match = re.search(r"(\d+)\s*cop(?:y|ies)", user_input_lower)
            if copy_match:
                parameters["copies"] = int(copy_match.group(1))

            # Extract paper size
            if "a4" in user_input_lower:
                parameters["paper_size"] = "A4"
            elif "letter" in user_input_lower:
                parameters["paper_size"] = "letter"

            # Extract duplex
            if (
                "both sides" in user_input_lower
                or "duplex" in user_input_lower
                or "double sided" in user_input_lower
            ):
                parameters["duplex"] = True

            return IntentType.PRINT, parameters

        # Scan intent patterns
        scan_keywords = ["scan", "scanning", "capture", "digitize", "photo"]
        if any(keyword in user_input_lower for keyword in scan_keywords):
            # Extract parameters
            if "high quality" in user_input_lower or "high resolution" in user_input_lower:
                parameters["resolution"] = 600
                parameters["quality"] = "high"
            elif "low quality" in user_input_lower or "low resolution" in user_input_lower:
                parameters["resolution"] = 150
                parameters["quality"] = "low"

            if "pdf" in user_input_lower:
                parameters["format"] = "pdf"
            elif "jpg" in user_input_lower or "jpeg" in user_input_lower:
                parameters["format"] = "jpg"

            return IntentType.SCAN, parameters

        # Status inquiry
        status_keywords = ["status", "what's happening", "progress", "how's it going"]
        if any(keyword in user_input_lower for keyword in status_keywords):
            return IntentType.VIEW_STATUS, {}

        # Configuration
        config_keywords = ["configure", "settings", "set up", "change settings", "options"]
        if any(keyword in user_input_lower for keyword in config_keywords):
            return IntentType.CONFIGURE, parameters

        # List documents
        list_keywords = [
            "list",
            "show documents",
            "what documents",
            "available files",
            "show files",
        ]
        if any(keyword in user_input_lower for keyword in list_keywords):
            return IntentType.LIST_DOCUMENTS, {}

        # Help
        help_keywords = ["help", "what can you do", "capabilities", "how to"]
        if any(keyword in user_input_lower for keyword in help_keywords):
            return IntentType.HELP, {}

        return IntentType.UNKNOWN, {}

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

    def _handle_status_inquiry(self) -> Dict[str, Any]:
        """Handle status inquiry"""
        return {
            "success": True,
            "intent": "status",
            "current_state": self.current_state.value,
            "selected_document": self.selected_document,
            "pending_action": self.pending_action,
            "configuration": self.configuration,
            "message": self._get_status_message(),
            "available_documents_count": len(self._get_available_documents()),
        }

    def _handle_configuration(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle configuration changes"""
        self.current_state = WorkflowState.CONFIGURING

        return {
            "success": True,
            "intent": "configure",
            "message": "What would you like to configure? (print settings / scan settings)",
            "current_configuration": self.configuration,
            "workflow_state": self.current_state.value,
        }

    def _handle_list_documents(self) -> Dict[str, Any]:
        """Handle document listing request"""
        documents = self._get_available_documents()

        if not documents:
            message = "No documents available. Upload or scan a document to get started."
        else:
            doc_list = ", ".join([doc["filename"] for doc in documents[:5]])
            message = f"Available documents ({len(documents)}): {doc_list}"
            if len(documents) > 5:
                message += f" and {len(documents) - 5} more..."

        return {
            "success": True,
            "intent": "list_documents",
            "message": message,
            "documents": documents,
            "count": len(documents),
        }

    def _handle_help_request(self) -> Dict[str, Any]:
        """Handle help request"""
        return {
            "success": True,
            "intent": "help",
            "message": "I can help you with document operations! Here's what I can do:",
            "capabilities": [
                {
                    "action": "Print",
                    "description": "Print documents with custom settings (copies, color, duplex)",
                    "examples": [
                        "Print this document",
                        "Print 2 copies in color",
                        "Print double-sided",
                    ],
                },
                {
                    "action": "Scan",
                    "description": "Scan or capture documents with quality settings",
                    "examples": ["Scan a document", "Scan in high quality", "Capture as PDF"],
                },
                {
                    "action": "Status",
                    "description": "Check current workflow status and progress",
                    "examples": ["What's the status?", "Show progress", "What's happening?"],
                },
                {
                    "action": "List",
                    "description": "View available documents",
                    "examples": ["List documents", "Show files", "What documents are available?"],
                },
                {
                    "action": "Configure",
                    "description": "Change default settings",
                    "examples": [
                        "Configure print settings",
                        "Change scan quality",
                        "Set up options",
                    ],
                },
            ],
        }

    def confirm_action(self) -> Dict[str, Any]:
        """
        Confirm and execute pending action

        Returns:
            Execution result
        """
        with self.state_lock:
            if self.current_state != WorkflowState.AWAITING_CONFIRMATION:
                return {
                    "success": False,
                    "error": "No action pending confirmation",
                    "current_state": self.current_state.value,
                }

            if not self.pending_action:
                return {"success": False, "error": "No action configured"}

            # Execute the pending action
            self.current_state = WorkflowState.EXECUTING
            action_type = self.pending_action["type"]

            try:
                if action_type == "print":
                    result = self._execute_print()
                elif action_type == "scan":
                    result = self._execute_scan()
                else:
                    result = {"success": False, "error": f"Unknown action type: {action_type}"}

                if result["success"]:
                    self.current_state = WorkflowState.COMPLETED
                else:
                    self.current_state = WorkflowState.FAILED

                # Log to history
                self.workflow_history.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "action": self.pending_action,
                        "result": result,
                    }
                )

                # Clear pending action
                self.pending_action = None

                return result

            except Exception as e:
                logger.error(f"[ERROR] Action execution failed: {e}")
                self.current_state = WorkflowState.FAILED
                return {"success": False, "error": str(e), "action": self.pending_action}

    def cancel_action(self) -> Dict[str, Any]:
        """Cancel pending action"""
        with self.state_lock:
            if self.current_state != WorkflowState.AWAITING_CONFIRMATION:
                return {"success": False, "message": "No action to cancel"}

            cancelled_action = self.pending_action
            self.pending_action = None
            self.current_state = WorkflowState.IDLE

            return {
                "success": True,
                "message": "Action cancelled",
                "cancelled_action": cancelled_action,
            }

    def _execute_print(self) -> Dict[str, Any]:
        """Execute print operation"""
        logger.info("[PRINT] Executing print operation...")

        document = self.pending_action.get("document")
        config = self.pending_action.get("configuration")

        if not document:
            return {"success": False, "error": "No document selected"}

        # Construct file path
        file_path = os.path.join(self.processed_dir, document["filename"])

        if not os.path.exists(file_path):
            return {"success": False, "error": f'Document not found: {document["filename"]}'}

        # Trigger print (this would interface with actual print module)
        # For now, return success with configuration
        return {
            "success": True,
            "action": "print",
            "message": f'Print job sent: {document["filename"]}',
            "document": document,
            "configuration": config,
            "job_id": f'print_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            "timestamp": datetime.now().isoformat(),
        }

    def _execute_scan(self) -> Dict[str, Any]:
        """Execute scan operation"""
        logger.info("📸 Executing scan operation...")

        config = self.pending_action.get("configuration")

        # Trigger scan interface
        return {
            "success": True,
            "action": "scan",
            "message": "Opening scan interface...",
            "configuration": config,
            "next_steps": [
                "Open phone capture interface",
                "Position document in frame",
                "Capture when ready",
            ],
            "redirect_to": "/phone",
            "timestamp": datetime.now().isoformat(),
        }

    def select_document(self, filename: str) -> Dict[str, Any]:
        """
        Select a document for operation

        Args:
            filename: Document filename

        Returns:
            Selection result
        """
        documents = self._get_available_documents()
        document = next((doc for doc in documents if doc["filename"] == filename), None)

        if not document:
            return {"success": False, "error": f"Document not found: {filename}"}

        self.selected_document = document

        return {"success": True, "message": f"Selected document: {filename}", "document": document}

    def parse_voice_configuration(self, voice_text: str, action_type: str) -> Dict[str, Any]:
        """
        Parse voice commands for configuration changes

        Args:
            voice_text: User's voice command text
            action_type: 'print' or 'scan'

        Returns:
            Parsed configuration updates
        """
        updates: Dict[str, Any] = {}
        text_lower = voice_text.lower()
        import re

        def contains_any(phrases):
            return any(phrase in text_lower for phrase in phrases)

        # Common stop phrases indicating no more changes
        stop_phrases = [
            "no changes",
            "that's all",
            "nothing else",
            "done",
            "proceed",
            "continue",
            "i'm good",
            "all set",
            "looks good",
        ]

        if any(phrase in text_lower for phrase in stop_phrases):
            return {"no_changes": True}

        if action_type == "print":
            # Orientation
            if contains_any(["landscape", "horizontal", "wide"]):
                updates["orientation"] = "landscape"
            elif contains_any(["portrait", "vertical", "tall"]):
                updates["orientation"] = "portrait"

            # Copies
            copies_match = re.search(r"(\d+)\s*(?:cop(?:y|ies)|prints|pages)", text_lower)
            if copies_match:
                updates["copies"] = int(copies_match.group(1))
            elif contains_any(["single copy", "one copy"]):
                updates["copies"] = 1

            # Color mode
            if contains_any(["full color", "print in color", "color mode", "black and white", "black & white", "bw", "monochrome", "mono", "gray scale", "grey scale", "grayscale", "greyscale"]):
                updates["color_mode"] = "color"
            elif "color" in text_lower:
                updates["color_mode"] = "color"

            # Duplex
            if contains_any(["double sided", "two sided", "both sides", "duplex", "front and back"]):
                updates["duplex"] = True
            elif contains_any(["single sided", "one sided", "front only", "simplex"]):
                updates["duplex"] = False

            # Page selection
            page_range_match = re.search(r"page(?:s)?\s+(\d+)(?:\s*-\s*|\s+to\s+)(\d+)", text_lower)
            if page_range_match:
                updates["pages"] = "custom"
                updates["custom_range"] = f"{page_range_match.group(1)}-{page_range_match.group(2)}"
            elif contains_any(["odd page", "odd pages", "only odd", "odd only", "just odd"]):
                updates["pages"] = "odd"
            elif contains_any(["even page", "even pages", "only even", "even only", "just even"]):
                updates["pages"] = "even"
            elif contains_any(["all pages", "entire document", "every page"]):
                updates["pages"] = "all"

            # Paper size
            paper_sizes = ["a5", "a4", "a3", "letter", "legal", "tabloid"]
            for size in paper_sizes:
                if size in text_lower:
                    updates["paper_size"] = size
                    break
            if "custom" in text_lower and re.search(r"(\d+(?:\.\d+)?)\s*(?:x|by)\s*(\d+(?:\.\d+)?)", text_lower):
                updates["paper_size"] = "custom"

            # Resolution / DPI
            dpi_match = re.search(r"(\d+)\s*dpi", text_lower)
            if dpi_match:
                updates["resolution"] = int(dpi_match.group(1))
            elif contains_any(["high resolution", "high quality", "ultra quality"]):
                updates["resolution"] = 600
                updates["quality"] = "high"
            elif contains_any(["draft quality", "draft mode", "low quality", "fast draft"]):
                updates["resolution"] = 150
                updates["quality"] = "draft"

            # Scale / zoom
            scale_match = re.search(r"(\d{2,3})\s*(?:%|percent)", text_lower)
            if scale_match:
                updates["scale"] = int(scale_match.group(1))
            elif contains_any(["fit to page", "fit page", "full size"]):
                updates["scale"] = 100

            # Pages per sheet
            pps_match = re.search(r"(\d+)\s*(?:per\s*(?:sheet|page|side))", text_lower)
            if pps_match:
                updates["pages_per_sheet"] = pps_match.group(1)
            elif contains_any(["two up", "2-up", "2 up"]):
                updates["pages_per_sheet"] = "2"
            elif contains_any(["four up", "4-up", "4 up"]):
                updates["pages_per_sheet"] = "4"

            # Margins
            if contains_any(["no margin", "borderless", "edge to edge", "full bleed"]):
                updates["margins"] = "none"
            elif contains_any(["narrow margin", "thin margin", "small margin"]):
                updates["margins"] = "narrow"
            elif contains_any(["default margin", "standard margin", "normal margin"]):
                updates["margins"] = "default"

            # Quality
            if contains_any(["ultra quality", "high quality", "best quality", "premium quality"]):
                updates["quality"] = "high"
            elif contains_any(["draft", "eco mode", "low quality", "economy mode"]):
                updates["quality"] = "draft"
            elif contains_any(["normal quality", "standard quality"]):
                updates["quality"] = "normal"

        elif action_type == "scan":
            # Resolution
            dpi_match = re.search(r"(\d+)\s*dpi", text_lower)
            if dpi_match:
                updates["resolution"] = int(dpi_match.group(1))
            elif contains_any(["1200", "twelve hundred"]):
                updates["resolution"] = 1200
            elif contains_any(["600", "six hundred"]):
                updates["resolution"] = 600
            elif contains_any(["300", "three hundred"]):
                updates["resolution"] = 300

            # Color mode
            if contains_any(["full color", "color scan", "scan in color"]):
                updates["color_mode"] = "color"
            elif contains_any(["black and white", "black & white", "grayscale", "greyscale", "mono", "monochrome"]) or "color" in text_lower:
                updates["color_mode"] = "color"

            # Format
            formats = ["pdf", "png", "jpg", "jpeg", "tiff"]
            for fmt in formats:
                if fmt in text_lower:
                    updates["format"] = fmt
                    break
            if "image" in text_lower and "pdf" not in text_lower:
                updates.setdefault("format", "png")

            # Page size / orientation
            page_sizes = ["a5", "a4", "letter", "legal", "a3"]
            for size in page_sizes:
                if size in text_lower:
                    updates["paper_size"] = size
                    break

            if contains_any(["landscape", "horizontal", "wide"]):
                updates["orientation"] = "landscape"
            elif contains_any(["portrait", "vertical", "tall"]):
                updates["orientation"] = "portrait"

            # Page selection
            page_range_match = re.search(r"page(?:s)?\s+(\d+)(?:\s*-\s*|\s+to\s+)(\d+)", text_lower)
            if page_range_match:
                updates["page_mode"] = "custom"
                updates["custom_range"] = f"{page_range_match.group(1)}-{page_range_match.group(2)}"
            elif contains_any(["odd page", "odd pages", "only odd", "odd only", "just odd"]):
                updates["page_mode"] = "odd"
            elif contains_any(["even page", "even pages", "only even", "even only", "just even"]):
                updates["page_mode"] = "even"
            elif contains_any(["all pages", "entire document", "everything"]):
                updates["page_mode"] = "all"

            # Text / OCR mode
            if contains_any(["text mode", "ocr", "extract text", "enable ocr", "turn on ocr"]):
                updates["text_mode"] = True
            elif contains_any(["disable ocr", "turn off ocr", "no text"]):
                updates["text_mode"] = False

            # Capture mode (single / multi)
            if contains_any(["multi", "batch", "continuous", "auto feed"]):
                updates["mode"] = "multi"
            elif contains_any(["single", "one page", "single shot"]):
                updates["mode"] = "single"

            # Quality cues
            if contains_any(["high quality", "best quality", "premium quality"]):
                updates["quality"] = "high"
            elif contains_any(["draft", "eco mode", "low quality"]):
                updates["quality"] = "draft"
            elif contains_any(["normal quality", "standard quality"]):
                updates["quality"] = "normal"

        return updates

    def update_configuration(self, action_type: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update configuration for print or scan

        Args:
            action_type: 'print' or 'scan'
            settings: Settings to update

        Returns:
            Update result
        """
        if action_type not in ["print", "scan"]:
            return {"success": False, "error": f"Invalid action type: {action_type}"}

        normalized_settings = dict(settings)

        if action_type == "scan" and "page_size" in normalized_settings and "paper_size" not in normalized_settings:
            normalized_settings["paper_size"] = normalized_settings.pop("page_size")

        self.configuration[action_type].update(normalized_settings)

        frontend_state = self._build_frontend_state(action_type)
        frontend_updates = self._build_frontend_updates(action_type, normalized_settings)

        return {
            "success": True,
            "message": f"{action_type.capitalize()} configuration updated",
            "configuration": self.configuration[action_type],
            "frontend_state": frontend_state,
            "frontend_updates": frontend_updates,
        }

    def _get_available_documents(self) -> List[Dict[str, Any]]:
        """Get list of available processed documents with index positions"""
        if not os.path.exists(self.processed_dir):
            return []

        documents = []
        for filename in os.listdir(self.processed_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".pdf")):
                file_path = os.path.join(self.processed_dir, filename)
                if os.path.exists(file_path):
                    stat = os.stat(file_path)
                    documents.append(
                        {
                            "filename": filename,
                            "size": stat.st_size,
                            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                            "path": file_path,
                        }
                    )

        # Sort by creation time (newest first)
        documents.sort(key=lambda x: x["created"], reverse=True)
        
        # Add index positions (1-based for user-friendly references)
        for idx, doc in enumerate(documents):
            doc["index"] = idx + 1
            doc["position"] = idx  # 0-based position

        return documents
    
    def get_documents_by_relative_position(self, position_desc: str) -> List[Dict[str, Any]]:
        """
        Get documents by relative position descriptions like 'last 2', 'first 3', 'latest'
        
        Args:
            position_desc: Description like 'last', 'last 2', 'first', 'first 3', 'latest', etc.
        
        Returns:
            List of matching documents
        """
        documents = self._get_available_documents()
        
        if not documents:
            return []
        
        position_lower = position_desc.lower().strip()
        
        # Single document references
        if position_lower in ["last", "latest", "newest", "most recent", "recent"]:
            return [documents[0]]  # First in list (sorted newest first)
        
        if position_lower in ["first", "oldest", "earliest"]:
            return [documents[-1]]  # Last in list
        
        # Multiple document references with numbers
        import re
        
        # Match patterns like "last 2", "first 3", "latest 5"
        last_match = re.search(r"(?:last|latest|recent|newest)\s+(\d+)", position_lower)
        if last_match:
            count = int(last_match.group(1))
            return documents[:min(count, len(documents))]
        
        first_match = re.search(r"(?:first|oldest|earliest)\s+(\d+)", position_lower)
        if first_match:
            count = int(first_match.group(1))
            return documents[-min(count, len(documents)):][::-1]  # Reverse to maintain chronological order
        
        # Default: return empty if no match
        return []

    def _get_status_message(self) -> str:
        """Generate human-readable status message"""
        if self.current_state == WorkflowState.IDLE:
            return "Ready for commands. What would you like to do?"
        elif self.current_state == WorkflowState.AWAITING_CONFIRMATION:
            action_type = (
                self.pending_action.get("type", "action") if self.pending_action else "action"
            )
            return f"Awaiting confirmation for {action_type} operation."
        elif self.current_state == WorkflowState.EXECUTING:
            return "Executing operation..."
        elif self.current_state == WorkflowState.CONFIGURING:
            return "In configuration mode."
        elif self.current_state == WorkflowState.COMPLETED:
            return "Last operation completed successfully."
        elif self.current_state == WorkflowState.FAILED:
            return "Last operation failed. Ready for new commands."
        else:
            return "Unknown state."

    def get_workflow_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get workflow history

        Args:
            limit: Maximum number of history items

        Returns:
            List of workflow history items
        """
        return self.workflow_history[-limit:]

    def reset_state(self) -> Dict[str, Any]:
        """Reset orchestrator to idle state"""
        with self.state_lock:
            self.current_state = WorkflowState.IDLE
            self.pending_action = None
            self.selected_document = None

            return {
                "success": True,
                "message": "Orchestrator reset to idle state",
                "current_state": self.current_state.value,
            }
    
    def reset(self) -> None:
        """Reset the orchestrator state (alias for reset_state)."""
        self.reset_state()
    
    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get orchestration history.
        
        Args:
            limit: Maximum number of history entries
            
        Returns:
            List of history entries
        """
        return self.workflow_history[-limit:]
    
    def execute_print(self, files: List[str], settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute print workflow for files.
        
        Args:
            files: List of filenames to print
            settings: Print settings
            
        Returns:
            Result dictionary
        """
        from datetime import datetime
        
        with self.state_lock:
            self.current_state = WorkflowState.EXECUTING
        
        try:
            # Log workflow entry
            workflow_entry = {
                "type": "print",
                "files": files,
                "settings": settings,
                "timestamp": datetime.now().isoformat(),
                "status": "executing"
            }
            
            # Import print service
            try:
                from app.features.print.services.print_job_service import PrintJobService
                job_service = PrintJobService()
                
                results = []
                for filename in files:
                    copies = settings.get("copies", 1)
                    result = job_service.print_document(filename, copies)
                    results.append({
                        "filename": filename,
                        "success": result.get("success", False),
                        "message": result.get("message") or result.get("error")
                    })
                
                success_count = sum(1 for r in results if r["success"])
                
                workflow_entry["status"] = "completed" if success_count > 0 else "failed"
                workflow_entry["results"] = results
                self.workflow_history.append(workflow_entry)
                
                with self.state_lock:
                    self.current_state = WorkflowState.COMPLETED if success_count > 0 else WorkflowState.FAILED
                
                return {
                    "success": success_count > 0,
                    "message": f"Printed {success_count}/{len(files)} files",
                    "results": results
                }
            
            except ImportError as e:
                workflow_entry["status"] = "failed"
                workflow_entry["error"] = "Print service not available"
                self.workflow_history.append(workflow_entry)
                
                with self.state_lock:
                    self.current_state = WorkflowState.FAILED
                
                return {
                    "success": False,
                    "error": "Print service not available"
                }
        
        except Exception as e:
            with self.state_lock:
                self.current_state = WorkflowState.FAILED
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def execute_scan(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute scan workflow.
        
        Args:
            settings: Scan settings
            
        Returns:
            Result dictionary
        """
        from datetime import datetime
        
        with self.state_lock:
            self.current_state = WorkflowState.EXECUTING
        
        try:
            workflow_entry = {
                "type": "scan",
                "settings": settings,
                "timestamp": datetime.now().isoformat(),
                "status": "executing"
            }
            
            # For now, scanning is typically triggered from a different endpoint
            # This is a placeholder for orchestrated scan operations
            
            workflow_entry["status"] = "completed"
            workflow_entry["message"] = "Scan operation initiated"
            self.workflow_history.append(workflow_entry)
            
            with self.state_lock:
                self.current_state = WorkflowState.COMPLETED
            
            return {
                "success": True,
                "message": "Scan operation initiated",
                "settings": settings
            }
        
        except Exception as e:
            with self.state_lock:
                self.current_state = WorkflowState.FAILED
            
            return {
                "success": False,
                "error": str(e)
            }


# Global orchestrator instance
_orchestrator_instance = None
_orchestrator_lock = threading.Lock()


def get_orchestrator(data_dir: str) -> PrintScanOrchestrator:
    """
    Get or create global orchestrator instance

    Args:
        data_dir: Data directory path

    Returns:
        PrintScanOrchestrator instance
    """
    global _orchestrator_instance

    with _orchestrator_lock:
        if _orchestrator_instance is None:
            _orchestrator_instance = PrintScanOrchestrator(data_dir)

        return _orchestrator_instance
