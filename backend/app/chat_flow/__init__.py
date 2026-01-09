"""
Chat Flow Module - Unified AI Chat Workflow for Print/Scan Operations

This module handles BOTH text-based and voice-based AI chat interactions.
The workflow is IDENTICAL regardless of input method:

Text Chat: User types → process_text() → text response
Voice Chat: Audio → transcribe → process_text() → TTS response

Flow Steps:
1. "print" or "scan" → "Do you want to [print/scan]? Say yes to proceed."
2. "yes" → Opens document selector
3. "select document 1 to 5" → Selects documents
4. "proceed" → Opens configuration
5. "landscape", "2 copies", "color" → Adjusts settings
6. "proceed" → Review page
7. "proceed" → Execute print/scan

All AI-based print/scan orchestration flows through this single module.
"""

from .service import ChatFlowService, chat_flow_service
from .commands import ChatFlowCommands
from .types import (
    ChatFlowState,
    ChatFlowStep,
    ChatFlowConfig,
    ChatFlowResponse,
)
from .routes import chat_flow_bp

__all__ = [
    'ChatFlowService',
    'chat_flow_service',
    'ChatFlowCommands',
    'ChatFlowState',
    'ChatFlowStep',
    'ChatFlowConfig',
    'ChatFlowResponse',
    'chat_flow_bp',
]
