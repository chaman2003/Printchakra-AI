"""
Chat Flow Service - Main business logic for AI chat workflow (TEXT and VOICE)

This service handles BOTH text-based AI chat and voice-based AI chat.
The flow is IDENTICAL regardless of input method - only the I/O differs:
- Text: User types → gets text response
- Voice: User speaks (transcribed to text) → gets response (converted to speech via TTS)

State Machine Flow:
1. IDLE → User says/types "print" or "scan" → AWAITING_CONFIRM
2. AWAITING_CONFIRM → User says/types "yes" → SELECTING_DOCS
3. SELECTING_DOCS → User selects docs + says/types "proceed" → CONFIGURING
4. CONFIGURING → User adjusts settings + says/types "proceed" → REVIEWING
5. REVIEWING → User says/types "proceed" → EXECUTING
6. EXECUTING → Job completes → COMPLETED/IDLE

Interaction Modes:
- WORK mode (default): Print/scan workflow commands
- TALK mode: Free conversation with AI assistant
"""

import logging
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime

from .types import ChatFlowState, ChatFlowStep, ChatFlowConfig, ChatFlowResponse
from .commands import ChatFlowCommands

logger = logging.getLogger(__name__)

# Ollama API settings for talk mode
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "smollm2:135m"  # Default model, can be changed
OLLAMA_TIMEOUT = 60


class ChatFlowService:
    """
    Manages the AI-controlled print/scan workflow state machine.
    
    Single source of truth for the entire workflow - handles both text and voice input.
    The same process_text() method is used regardless of how input was received.
    """
    
    def __init__(self):
        self.state = ChatFlowState()
        self.session_active = False
        # Conversation history for talk mode
        self.talk_history: List[Dict[str, str]] = []
        self.talk_system_prompt = """You are PrintChakra AI, a friendly and intelligent voice assistant. 
PrintChakra is a voice-controlled document management app that helps users print and scan documents hands-free.

In TALK MODE, you can discuss ANY topic - technology, science, general knowledge, creative ideas, current events, etc.
Be conversational, helpful, and engaging. Give thoughtful answers.
Keep responses concise but informative (2-4 sentences is ideal for voice).

If the user asks about PrintChakra or what you can do:
- PrintChakra lets users print and scan documents using voice commands
- Users can select documents, adjust settings (copies, color, orientation), and execute jobs - all by voice
- Features include document conversion, OCR, and a full voice-controlled workflow

If the user wants to go back to printing/scanning, they can say 'let's work'."""
    
    def start_session(self) -> Dict[str, Any]:
        """Start a new voice session"""
        self.session_active = True
        self.state.reset()
        logger.info("[CHAT_FLOW] Session started")
        return {
            "success": True,
            "message": "Voice session started. Say 'print' or 'scan' to begin."
        }
    
    def end_session(self) -> Dict[str, Any]:
        """End the voice session"""
        self.session_active = False
        self.state.reset()
        logger.info("[CHAT_FLOW] Session ended")
        return {
            "success": True,
            "message": "Voice session ended. Goodbye!"
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current workflow state"""
        return self.state.to_dict()
    
    def process_text(self, text: str) -> ChatFlowResponse:
        """
        Process voice input text and return appropriate response.
        
        This is the main entry point for all voice commands.
        """
        if not self.session_active:
            return ChatFlowResponse(
                success=False,
                ai_response="Voice session not active. Please start a session first.",
                current_step=ChatFlowStep.IDLE
            )
        
        text_lower = text.lower().strip()
        logger.info(f"[CHAT_FLOW] Processing: '{text}' | Current step: {self.state.step.value}")
        
        # Parse the command
        command, params = ChatFlowCommands.parse_command(text)
        logger.info(f"[CHAT_FLOW] Parsed command: {command} | Params: {params}")
        
        # Handle based on current step
        response = self._handle_command(command, params, text)
        
        self.state.last_updated = datetime.now()
        return response
    
    def _handle_command(
        self, 
        command: Optional[str], 
        params: Dict[str, Any],
        original_text: str
    ) -> ChatFlowResponse:
        """Handle parsed command based on current state"""
        
        step = self.state.step
        
        # === HANDLE EXIT ANYTIME ===
        if command == "exit_session":
            self.state.reset()
            return ChatFlowResponse(
                success=True,
                ai_response="Goodbye! Voice session ended.",
                tts_response="Goodbye!",
                current_step=ChatFlowStep.IDLE,
                voice_command="exit_session",
                interaction_mode=self.state.interaction_mode
            )
        
        # === HANDLE MODE SWITCH (talk/work) ANYTIME ===
        if command == "switch_to_talk":
            self.state.interaction_mode = "talk"
            self.state.reset()  # Reset workflow but keep interaction_mode
            self.state.interaction_mode = "talk"  # Restore after reset
            response_text = "Switched to talk mode! I'm here to chat about anything. Ask me questions, discuss topics, or just have a conversation. Say 'let's work' when you're ready to print or scan."
            return ChatFlowResponse(
                success=True,
                ai_response=response_text,
                tts_response=response_text,
                current_step=ChatFlowStep.IDLE,
                voice_command="switch_to_talk",
                interaction_mode="talk"
            )
        
        if command == "switch_to_work":
            self.state.interaction_mode = "work"
            self.state.reset()  # Reset workflow
            self.state.interaction_mode = "work"  # Restore after reset
            response_text = "Back to work mode! Say 'print' to print documents or 'scan' to scan documents."
            return ChatFlowResponse(
                success=True,
                ai_response=response_text,
                tts_response=response_text,
                current_step=ChatFlowStep.IDLE,
                voice_command="switch_to_work",
                interaction_mode="work"
            )
        
        # === IF IN TALK MODE, HANDLE CONVERSATIONALLY ===
        if self.state.interaction_mode == "talk":
            return self._handle_talk_mode(command, params, original_text)
        
        # === HANDLE CANCEL ANYTIME ===
        if command == "cancel":
            previous_step = self.state.step
            self.state.reset()
            return ChatFlowResponse(
                success=True,
                ai_response="Cancelled. Say 'print' or 'scan' to start again.",
                tts_response="Cancelled.",
                current_step=ChatFlowStep.IDLE,
                voice_command="cancel",
                interaction_mode=self.state.interaction_mode
            )
        
        # === STATE: IDLE or AWAITING_MODE ===
        if step in [ChatFlowStep.IDLE, ChatFlowStep.AWAITING_MODE]:
            return self._handle_idle(command, params, original_text)
        
        # === STATE: AWAITING_CONFIRM ===
        if step == ChatFlowStep.AWAITING_CONFIRM:
            return self._handle_awaiting_confirm(command, params, original_text)
        
        # === STATE: SELECTING_DOCS ===
        if step == ChatFlowStep.SELECTING_DOCS:
            return self._handle_selecting_docs(command, params, original_text)
        
        # === STATE: CONFIGURING ===
        if step == ChatFlowStep.CONFIGURING:
            return self._handle_configuring(command, params, original_text)
        
        # === STATE: REVIEWING ===
        if step == ChatFlowStep.REVIEWING:
            return self._handle_reviewing(command, params, original_text)
        
        # Fallback
        return ChatFlowResponse(
            success=True,
            ai_response="I didn't understand that. Say 'print' or 'scan' to begin.",
            current_step=self.state.step
        )
    
    def _handle_idle(
        self,
        command: Optional[str],
        params: Dict[str, Any],
        original_text: str
    ) -> ChatFlowResponse:
        """Handle commands in IDLE state"""
        
        # Handle greetings
        if command == "greeting":
            response_text = "Hi there! I'm PrintChakra AI, your voice-controlled document assistant. I can help you print or scan documents. Just say 'print' or 'scan' to get started."
            return ChatFlowResponse(
                success=True,
                ai_response=response_text,
                tts_response=response_text,
                current_step=ChatFlowStep.IDLE
            )
        
        if command == "start_print":
            self.state.step = ChatFlowStep.AWAITING_CONFIRM
            self.state.pending_mode = "print"
            response_text = "Do you want to print? Say yes to proceed."
            return ChatFlowResponse(
                success=True,
                ai_response=response_text,
                tts_response=response_text,
                current_step=ChatFlowStep.AWAITING_CONFIRM
            )
        
        if command == "start_scan":
            self.state.step = ChatFlowStep.AWAITING_CONFIRM
            self.state.pending_mode = "scan"
            response_text = "Do you want to scan? Say yes to proceed."
            return ChatFlowResponse(
                success=True,
                ai_response=response_text,
                tts_response=response_text,
                current_step=ChatFlowStep.AWAITING_CONFIRM
            )
        
        # Default response - for unrecognized commands in work mode, 
        # use AI to give a helpful response but remind about print/scan
        return self._handle_unrecognized_in_work_mode(original_text)
    
    def _handle_unrecognized_in_work_mode(self, original_text: str) -> ChatFlowResponse:
        """Handle unrecognized input in work mode - use AI but guide to print/scan"""
        try:
            work_system_prompt = """You are PrintChakra AI, a voice-controlled document assistant. 
Your PRIMARY job is to help users PRINT and SCAN documents.
The user said something that isn't a print/scan command.
Give a brief, helpful response to their question if relevant, but ALWAYS end by reminding them:
"Say 'print' to print documents or 'scan' to scan. Or say 'let's talk' for free conversation."
Keep responses concise (1-2 sentences max before the reminder)."""
            
            messages = [
                {"role": "system", "content": work_system_prompt},
                {"role": "user", "content": original_text}
            ]
            
            query = {
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 100,
                    "num_ctx": 1024,
                },
            }
            
            response = requests.post(OLLAMA_URL, json=query, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get("message", {}).get("content", "").strip()
                import re
                ai_response = ai_response.replace("**", "").replace("*", "")
                ai_response = re.sub(r'\s+', ' ', ai_response).strip()
                
                return ChatFlowResponse(
                    success=True,
                    ai_response=ai_response,
                    tts_response=ai_response,
                    current_step=ChatFlowStep.IDLE,
                    interaction_mode="work"
                )
        except Exception as e:
            logger.error(f"[WORK_MODE] AI fallback failed: {e}")
        
        # Fallback if AI fails
        response_text = "I'm here to help with printing and scanning. Say 'print' or 'scan' to begin, or 'let's talk' for conversation."
        return ChatFlowResponse(
            success=True,
            ai_response=response_text,
            tts_response=response_text,
            current_step=ChatFlowStep.IDLE,
            interaction_mode="work"
        )
    
    def _handle_awaiting_confirm(
        self,
        command: Optional[str],
        params: Dict[str, Any],
        original_text: str
    ) -> ChatFlowResponse:
        """Handle commands in AWAITING_CONFIRM state"""
        
        mode = self.state.pending_mode
        
        if command == "confirm":
            self.state.step = ChatFlowStep.SELECTING_DOCS
            self.state.config.mode = mode
            self.state.pending_mode = None
            
            response_text = f"Opening {mode} interface. Select your documents, then say 'proceed'."
            return ChatFlowResponse(
                success=True,
                ai_response=response_text,
                tts_response=response_text,
                current_step=ChatFlowStep.SELECTING_DOCS,
                open_document_selector=True,
                voice_command="open_document_selector",
                command_params={"mode": mode}
            )
        
        # User said something else - might be changing mode
        if command == "start_print":
            self.state.pending_mode = "print"
            response_text = "Do you want to print? Say yes to proceed."
            return ChatFlowResponse(
                success=True,
                ai_response=response_text,
                tts_response=response_text,
                current_step=ChatFlowStep.AWAITING_CONFIRM
            )
        
        if command == "start_scan":
            self.state.pending_mode = "scan"
            response_text = "Do you want to scan? Say yes to proceed."
            return ChatFlowResponse(
                success=True,
                ai_response=response_text,
                tts_response=response_text,
                current_step=ChatFlowStep.AWAITING_CONFIRM
            )
        
        # Not a confirmation - remind user
        response_text = f"Would you like to {mode}? Say 'yes' to continue or 'cancel' to abort."
        return ChatFlowResponse(
            success=True,
            ai_response=response_text,
            tts_response=response_text,
            current_step=ChatFlowStep.AWAITING_CONFIRM
        )
    
    def _handle_selecting_docs(
        self,
        command: Optional[str],
        params: Dict[str, Any],
        original_text: str
    ) -> ChatFlowResponse:
        """Handle commands in SELECTING_DOCS state"""
        
        mode = self.state.config.mode
        
        # Document selection commands - generate descriptive response and pass to frontend
        doc_commands = [
            "select_document", "select_documents", "select_document_range",
            "select_all_documents", "deselect_document", "deselect_documents", "deselect_all"
        ]
        
        if command in doc_commands:
            # Generate appropriate response text based on command
            if command == "select_document":
                idx = params.get("index", 1)
                response_text = f"Document {idx} selected."
            elif command == "select_documents":
                indices = params.get("indices", [])
                if len(indices) <= 3:
                    response_text = f"Documents {', '.join(map(str, indices))} selected."
                else:
                    response_text = f"{len(indices)} documents selected."
            elif command == "select_document_range":
                start = params.get("start", 1)
                end = params.get("end", 1)
                response_text = f"Documents {start} to {end} selected."
            elif command == "select_all_documents":
                response_text = "All documents selected."
            elif command == "deselect_document":
                idx = params.get("index", 1)
                response_text = f"Document {idx} deselected."
            elif command == "deselect_documents":
                indices = params.get("indices", [])
                if len(indices) <= 3:
                    response_text = f"Documents {', '.join(map(str, indices))} deselected."
                else:
                    response_text = f"{len(indices)} documents deselected."
            elif command == "deselect_all":
                response_text = "All documents deselected."
            else:
                response_text = "Got it."
            
            return ChatFlowResponse(
                success=True,
                ai_response=response_text,
                tts_response=response_text,
                current_step=ChatFlowStep.SELECTING_DOCS,
                voice_command=command,
                command_params=params
            )
        
        # Switch section command (converted files vs upload)
        if command == "switch_section":
            section = params.get("section", "upload")
            section_name = "converted files" if section == "converted" else "upload files" if section == "upload" else section
            response_text = f"Switched to {section_name}."
            return ChatFlowResponse(
                success=True,
                ai_response=response_text,
                tts_response=response_text,
                current_step=ChatFlowStep.SELECTING_DOCS,
                voice_command=command,
                command_params=params
            )
        
        # Allow settings commands during document selection (user can say "landscape" anytime)
        settings_commands = ["set_orientation", "set_color_mode", "set_copies", 
                           "set_paper_size", "set_duplex", "set_pages", "set_quality", "set_scan_dpi"]
        if command in settings_commands:
            # Forward to configuring handler but stay in current step
            result = self._handle_configuring(command, params, original_text)
            result.current_step = ChatFlowStep.SELECTING_DOCS  # Stay in selecting state
            return result
        
        # Proceed to configuration
        if command == "proceed" or command == "confirm":
            self.state.step = ChatFlowStep.CONFIGURING
            
            response_text = "Documents selected. You can adjust settings like 'landscape', '2 copies', 'color'. Say 'proceed' when ready."
            
            return ChatFlowResponse(
                success=True,
                ai_response=response_text,
                tts_response="Documents selected. Adjust settings or say proceed.",
                current_step=ChatFlowStep.CONFIGURING,
                open_config_panel=True,
                voice_command="open_config_panel",
                command_params={"mode": mode}
            )
        
        # Go back
        if command == "go_back":
            self.state.step = ChatFlowStep.AWAITING_CONFIRM
            response_text = f"Going back. Do you want to {mode}? Say yes to proceed."
            return ChatFlowResponse(
                success=True,
                ai_response=response_text,
                tts_response=response_text,
                current_step=ChatFlowStep.AWAITING_CONFIRM
            )
        
        # Default help message
        response_text = "Select documents by saying 'select document 1' or 'select documents 1 to 5'. Say 'proceed' when done."
        return ChatFlowResponse(
            success=True,
            ai_response=response_text,
            tts_response=response_text,
            current_step=ChatFlowStep.SELECTING_DOCS
        )
    
    def _handle_configuring(
        self,
        command: Optional[str],
        params: Dict[str, Any],
        original_text: str
    ) -> ChatFlowResponse:
        """Handle commands in CONFIGURING state"""
        
        mode = self.state.config.mode
        
        # Settings commands
        settings_commands = {
            "set_orientation": "orientation",
            "set_color_mode": "color_mode",
            "set_copies": "copies",
            "set_paper_size": "paper_size",
            "set_duplex": "duplex",
            "set_pages": "pages",
            "set_quality": "quality",
            "set_scan_dpi": "scan_dpi",
        }
        
        if command in settings_commands:
            # Update config
            config_key = settings_commands[command]
            config_updates = {}
            
            if command == "set_orientation":
                self.state.config.orientation = params.get("orientation", "portrait")
                config_updates["orientation"] = self.state.config.orientation
                response_text = f"{self.state.config.orientation.capitalize()}."
                
            elif command == "set_color_mode":
                self.state.config.color_mode = params.get("color_mode", "color")
                config_updates["color_mode"] = self.state.config.color_mode
                mode_display = "Color" if self.state.config.color_mode == "color" else \
                              "Grayscale" if self.state.config.color_mode == "grayscale" else "Black and white"
                response_text = f"{mode_display}."
                
            elif command == "set_copies":
                self.state.config.copies = params.get("copies", 1)
                config_updates["copies"] = self.state.config.copies
                response_text = f"{self.state.config.copies} copies."
                
            elif command == "set_paper_size":
                self.state.config.paper_size = params.get("paper_size", "A4")
                config_updates["paper_size"] = self.state.config.paper_size
                response_text = f"{self.state.config.paper_size}."
                
            elif command == "set_duplex":
                self.state.config.duplex = params.get("duplex", False)
                config_updates["duplex"] = self.state.config.duplex
                response_text = "Double-sided." if self.state.config.duplex else "Single-sided."
                
            elif command == "set_pages":
                self.state.config.pages = params.get("pages", "all")
                self.state.config.custom_range = params.get("custom_range")
                config_updates["pages"] = self.state.config.pages
                if self.state.config.custom_range:
                    config_updates["custom_range"] = self.state.config.custom_range
                    response_text = f"Pages {self.state.config.custom_range}."
                else:
                    response_text = f"{self.state.config.pages.capitalize()} pages."
                    
            elif command == "set_quality":
                self.state.config.quality = params.get("quality", "normal")
                config_updates["quality"] = self.state.config.quality
                response_text = f"{self.state.config.quality.capitalize()} quality."
                
            elif command == "set_scan_dpi":
                self.state.config.scan_dpi = params.get("dpi", 300)
                config_updates["scan_dpi"] = self.state.config.scan_dpi
                response_text = f"{self.state.config.scan_dpi} DPI."
            
            else:
                response_text = "Setting updated."
            
            return ChatFlowResponse(
                success=True,
                ai_response=response_text,
                tts_response=response_text,
                current_step=ChatFlowStep.CONFIGURING,
                voice_command=command,
                command_params=params,
                config_updates=config_updates
            )
        
        # Proceed to review
        if command == "proceed" or command == "confirm":
            self.state.step = ChatFlowStep.REVIEWING
            
            # Build summary
            config = self.state.config
            summary = f"{config.copies} cop{'ies' if config.copies != 1 else 'y'}, {config.orientation}, {config.color_mode}"
            
            response_text = f"Ready to {mode}: {summary}. Say 'proceed' to start or 'back' to change settings."
            return ChatFlowResponse(
                success=True,
                ai_response=response_text,
                tts_response=response_text,
                current_step=ChatFlowStep.REVIEWING,
                open_review_panel=True,
                voice_command="open_review_panel",
                command_params={"config": config.to_dict()}
            )
        
        # Go back to document selection
        if command == "go_back":
            self.state.step = ChatFlowStep.SELECTING_DOCS
            response_text = "Going back. Select your documents, then say 'proceed'."
            return ChatFlowResponse(
                success=True,
                ai_response=response_text,
                tts_response=response_text,
                current_step=ChatFlowStep.SELECTING_DOCS,
                open_document_selector=True
            )
        
        response_text = "Adjust settings like 'landscape', '2 copies', or 'color'. Say 'proceed' when ready."
        return ChatFlowResponse(
            success=True,
            ai_response=response_text,
            tts_response=response_text,
            current_step=ChatFlowStep.CONFIGURING
        )
    
    def _handle_reviewing(
        self,
        command: Optional[str],
        params: Dict[str, Any],
        original_text: str
    ) -> ChatFlowResponse:
        """Handle commands in REVIEWING state"""
        
        mode = self.state.config.mode
        
        # Execute job
        if command == "proceed" or command == "confirm":
            self.state.step = ChatFlowStep.EXECUTING
            
            response_text = f"Starting {mode} job now!"
            return ChatFlowResponse(
                success=True,
                ai_response=response_text,
                tts_response=response_text,
                current_step=ChatFlowStep.EXECUTING,
                execute_job=True,
                voice_command="execute_job",
                command_params={"mode": mode, "config": self.state.config.to_dict()}
            )
        
        # Go back to configuration
        if command == "go_back":
            self.state.step = ChatFlowStep.CONFIGURING
            response_text = "Going back to settings. Adjust as needed, then say 'proceed'."
            return ChatFlowResponse(
                success=True,
                ai_response=response_text,
                tts_response=response_text,
                current_step=ChatFlowStep.CONFIGURING,
                open_config_panel=True
            )
        
        # Allow settings changes from review
        settings_commands = [
            "set_orientation", "set_color_mode", "set_copies",
            "set_paper_size", "set_duplex", "set_pages", "set_quality"
        ]
        if command in settings_commands:
            # Go back to configuring and apply the setting
            self.state.step = ChatFlowStep.CONFIGURING
            return self._handle_configuring(command, params, original_text)
        
        response_text = f"Review your settings. Say 'proceed' to {mode} or 'back' to change settings."
        return ChatFlowResponse(
            success=True,
            ai_response=response_text,
            tts_response=response_text,
            current_step=ChatFlowStep.REVIEWING
        )
    
    def job_completed(self, success: bool = True, message: str = "") -> ChatFlowResponse:
        """Called when print/scan job completes"""
        mode = self.state.config.mode
        
        if success:
            response_text = f"{mode.capitalize()} job completed successfully! Say 'print' or 'scan' to start another job."
            self.state.reset()
            
            return ChatFlowResponse(
                success=True,
                ai_response=response_text,
                tts_response=response_text,
                current_step=ChatFlowStep.IDLE
            )
        else:
            self.state.step = ChatFlowStep.REVIEWING
            response_text = f"{mode.capitalize()} job failed: {message}. Say 'proceed' to retry or 'cancel' to abort."
            return ChatFlowResponse(
                success=False,
                ai_response=response_text,
                tts_response=response_text,
                current_step=ChatFlowStep.REVIEWING
            )
    
    def update_selected_documents(self, documents: list) -> None:
        """Update the list of selected documents (called from frontend)"""
        self.state.config.selected_documents = documents
        logger.info(f"[CHAT_FLOW] Updated selected documents: {len(documents)} documents")
    
    def _handle_talk_mode(
        self,
        command: Optional[str],
        params: Dict[str, Any],
        original_text: str
    ) -> ChatFlowResponse:
        """Handle conversation in talk mode - free chat with AI"""
        
        # Handle greetings in talk mode too
        if command == "greeting":
            response_text = "Hello! We're in talk mode - I'm happy to chat about anything! What's on your mind?"
            return ChatFlowResponse(
                success=True,
                ai_response=response_text,
                tts_response=response_text,
                current_step=ChatFlowStep.IDLE,
                interaction_mode="talk"
            )
        
        # For any other input, send to Ollama for intelligent response
        try:
            # Add user message to history
            self.talk_history.append({"role": "user", "content": original_text})
            
            # Build messages for Ollama
            messages = [
                {"role": "system", "content": self.talk_system_prompt}
            ] + self.talk_history[-16:]  # Keep last 8 exchanges
            
            # Call Ollama API
            query = {
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": 150,
                    "num_ctx": 2048,
                    "repeat_penalty": 1.2,
                },
            }
            
            logger.info(f"[TALK_MODE] Sending to Ollama: {original_text[:50]}...")
            
            response = requests.post(
                OLLAMA_URL,
                json=query,
                timeout=OLLAMA_TIMEOUT,
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get("message", {}).get("content", "").strip()
                
                # Clean up response
                import re
                ai_response = ai_response.replace("**", "").replace("*", "")
                ai_response = re.sub(r'\s+', ' ', ai_response).strip()
                if ai_response and ai_response[-1] not in ".!?":
                    ai_response += "."
                
                # Add to history
                self.talk_history.append({"role": "assistant", "content": ai_response})
                
                # Keep history manageable
                if len(self.talk_history) > 16:
                    self.talk_history = self.talk_history[-16:]
                
                logger.info(f"[TALK_MODE] AI response: {ai_response[:100]}...")
                
                return ChatFlowResponse(
                    success=True,
                    ai_response=ai_response,
                    tts_response=ai_response,
                    current_step=ChatFlowStep.IDLE,
                    interaction_mode="talk"
                )
            else:
                logger.error(f"[TALK_MODE] Ollama API error: {response.status_code}")
                response_text = "I'm having trouble connecting to my brain right now. Please try again."
                return ChatFlowResponse(
                    success=False,
                    ai_response=response_text,
                    tts_response=response_text,
                    current_step=ChatFlowStep.IDLE,
                    interaction_mode="talk"
                )
                
        except requests.exceptions.Timeout:
            logger.error("[TALK_MODE] Ollama request timed out")
            response_text = "I took too long to think. Let me try a simpler response."
            return ChatFlowResponse(
                success=False,
                ai_response=response_text,
                tts_response=response_text,
                current_step=ChatFlowStep.IDLE,
                interaction_mode="talk"
            )
        except Exception as e:
            logger.error(f"[TALK_MODE] Error: {str(e)}")
            response_text = "Something went wrong. Say 'let's work' to go back to print/scan mode."
            return ChatFlowResponse(
                success=False,
                ai_response=response_text,
                tts_response=response_text,
                current_step=ChatFlowStep.IDLE,
                interaction_mode="talk"
            )


# Global instance
chat_flow_service = ChatFlowService()
