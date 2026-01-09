/**
 * Chat Flow Types - TypeScript interfaces for voice workflow
 */

export enum ChatFlowStep {
  IDLE = 'idle',
  AWAITING_MODE = 'awaiting_mode',
  AWAITING_CONFIRM = 'awaiting_confirm',
  SELECTING_DOCS = 'selecting_docs',
  CONFIGURING = 'configuring',
  REVIEWING = 'reviewing',
  EXECUTING = 'executing',
  COMPLETED = 'completed',
}

export interface ChatFlowConfig {
  mode: 'print' | 'scan' | null;
  selectedDocuments: string[];

  // Print settings
  copies: number;
  orientation: 'portrait' | 'landscape';
  colorMode: 'color' | 'grayscale' | 'bw';
  paperSize: string;
  duplex: boolean;
  pages: 'all' | 'odd' | 'even' | 'custom';
  customRange?: string;
  quality: 'draft' | 'normal' | 'high';

  // Scan settings
  scanDpi: number;
  scanFormat: 'pdf' | 'jpg' | 'png';
}

export interface ChatFlowState {
  step: ChatFlowStep;
  config: ChatFlowConfig;
  pendingMode: 'print' | 'scan' | null;
  lastUpdated: string;
}

export interface ChatFlowResponse {
  success: boolean;
  ai_response: string;
  tts_response?: string;
  current_step: string;

  // Frontend triggers
  open_document_selector?: boolean;
  open_config_panel?: boolean;
  open_review_panel?: boolean;
  execute_job?: boolean;

  // Voice command info
  voice_command?: string;
  command_params?: Record<string, any>;
  config_updates?: Record<string, any>;
  
  // Interaction mode: "work" or "talk"
  interaction_mode?: 'work' | 'talk';
}

export interface ChatFlowMessage {
  id: string;
  type: 'user' | 'ai' | 'system';
  text: string;
  timestamp: Date;
}

export const DEFAULT_CONFIG: ChatFlowConfig = {
  mode: null,
  selectedDocuments: [],
  copies: 1,
  orientation: 'portrait',
  colorMode: 'color',
  paperSize: 'A4',
  duplex: false,
  pages: 'all',
  quality: 'normal',
  scanDpi: 300,
  scanFormat: 'pdf',
};
