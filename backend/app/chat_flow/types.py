"""
Chat Flow Types - Data structures for voice workflow
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime


class ChatFlowStep(Enum):
    """Steps in the voice-controlled workflow"""
    IDLE = "idle"                      # No active workflow
    AWAITING_MODE = "awaiting_mode"    # Waiting for "print" or "scan"
    AWAITING_CONFIRM = "awaiting_confirm"  # "Do you want to print? Say yes"
    SELECTING_DOCS = "selecting_docs"  # User selecting documents
    CONFIGURING = "configuring"        # Setting options (copies, color, etc.)
    REVIEWING = "reviewing"            # Review before execution
    EXECUTING = "executing"            # Print/scan in progress
    COMPLETED = "completed"            # Workflow finished


@dataclass
class ChatFlowConfig:
    """Configuration for print/scan job"""
    # Common
    mode: Optional[str] = None  # "print" or "scan"
    selected_documents: List[str] = field(default_factory=list)
    
    # Print settings
    copies: int = 1
    orientation: str = "portrait"  # portrait, landscape
    color_mode: str = "color"      # color, grayscale, bw
    paper_size: str = "A4"         # A4, Letter, Legal, etc.
    duplex: bool = False
    pages: str = "all"             # all, odd, even, custom
    custom_range: Optional[str] = None  # e.g., "1-3, 5"
    quality: str = "normal"        # draft, normal, high
    
    # Scan settings
    scan_dpi: int = 300
    scan_format: str = "pdf"       # pdf, jpg, png
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "selected_documents": self.selected_documents,
            "copies": self.copies,
            "orientation": self.orientation,
            "color_mode": self.color_mode,
            "paper_size": self.paper_size,
            "duplex": self.duplex,
            "pages": self.pages,
            "custom_range": self.custom_range,
            "quality": self.quality,
            "scan_dpi": self.scan_dpi,
            "scan_format": self.scan_format,
        }


@dataclass
class ChatFlowState:
    """Current state of the voice workflow"""
    step: ChatFlowStep = ChatFlowStep.IDLE
    config: ChatFlowConfig = field(default_factory=ChatFlowConfig)
    pending_mode: Optional[str] = None  # "print" or "scan" awaiting confirmation
    interaction_mode: str = "work"  # "work" or "talk"
    last_updated: datetime = field(default_factory=datetime.now)
    
    def reset(self):
        """Reset to initial state"""
        self.step = ChatFlowStep.IDLE
        self.config = ChatFlowConfig()
        self.pending_mode = None
        # Note: interaction_mode is NOT reset - user stays in their chosen mode
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step.value,
            "config": self.config.to_dict(),
            "pending_mode": self.pending_mode,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class ChatFlowResponse:
    """Response from chat flow processing"""
    success: bool
    ai_response: str
    tts_response: Optional[str] = None  # Shorter version for TTS
    
    # State info
    current_step: ChatFlowStep = ChatFlowStep.IDLE
    
    # Triggers for frontend
    open_document_selector: bool = False
    open_config_panel: bool = False
    open_review_panel: bool = False
    execute_job: bool = False
    
    # Voice command detected
    voice_command: Optional[str] = None
    command_params: Optional[Dict[str, Any]] = None
    
    # Config updates (for frontend to apply)
    config_updates: Optional[Dict[str, Any]] = None
    
    # Interaction mode indicator
    interaction_mode: str = "work"  # "work" or "talk"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "ai_response": self.ai_response,
            "tts_response": self.tts_response or self.ai_response,
            "current_step": self.current_step.value,
            "open_document_selector": self.open_document_selector,
            "open_config_panel": self.open_config_panel,
            "open_review_panel": self.open_review_panel,
            "execute_job": self.execute_job,
            "voice_command": self.voice_command,
            "command_params": self.command_params,
            "config_updates": self.config_updates,
            "interaction_mode": self.interaction_mode,
        }
