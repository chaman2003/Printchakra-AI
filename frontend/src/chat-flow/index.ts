/**
 * Chat Flow Module - Unified Voice-Controlled Print/Scan Workflow
 *
 * This module handles the complete AI interaction flow for both TEXT and VOICE input:
 * 
 * The flow is IDENTICAL regardless of input method:
 * 1. "print" or "scan" → "Do you want to [print/scan]? Say yes to proceed."
 * 2. "yes" → Opens document selector
 * 3. "select document 1 to 5" → Selects documents
 * 4. "proceed" → Opens configuration
 * 5. "landscape", "2 copies", "color" → Adjusts settings
 * 6. "proceed" → Review page
 * 7. "proceed" → Execute print/scan
 *
 * Usage:
 * - Text Chat: User types commands, gets text responses
 * - Voice Chat: User speaks commands, gets spoken responses (via TTS)
 * 
 * Both modes use the same ChatFlowService backend and same state machine.
 */

// Context and Provider - main entry point
export { ChatFlowProvider, useChatFlow } from './ChatFlowContext';

// Component - renders both text and voice UI
export { ChatFlowComponent } from './ChatFlowComponent';

// API client - for direct API calls if needed
export { chatFlowApi } from './chatFlowApi';

// Types - for type checking
export * from './chatFlowTypes';

// Convenience re-export
export { ChatFlowStep, DEFAULT_CONFIG } from './chatFlowTypes';
export type { ChatFlowConfig, ChatFlowResponse, ChatFlowMessage, ChatFlowState } from './chatFlowTypes';
