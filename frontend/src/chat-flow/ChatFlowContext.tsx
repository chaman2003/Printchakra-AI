/**
 * Chat Flow Context - State management for voice workflow
 */

import React, { createContext, useContext, useState, useCallback, useRef, ReactNode } from 'react';
import { useToast } from '@chakra-ui/react';
import {
  ChatFlowStep,
  ChatFlowConfig,
  ChatFlowMessage,
  ChatFlowResponse,
  DEFAULT_CONFIG,
} from './chatFlowTypes';
import { chatFlowApi } from './chatFlowApi';

interface ChatFlowContextValue {
  // State
  isSessionActive: boolean;
  currentStep: ChatFlowStep;
  config: ChatFlowConfig;
  messages: ChatFlowMessage[];
  isRecording: boolean;
  isProcessing: boolean;
  isSpeaking: boolean;
  interactionMode: 'work' | 'talk';

  // Actions
  startSession: () => Promise<void>;
  endSession: () => Promise<void>;
  processText: (text: string) => Promise<ChatFlowResponse | null>;
  processAudio: (audioBlob: Blob) => Promise<void>;
  speak: (text: string) => Promise<void>;
  updateConfig: (updates: Partial<ChatFlowConfig>) => void;
  updateSelectedDocuments: (documents: string[]) => void;
  addMessage: (type: 'user' | 'ai' | 'system', text: string) => void;
  setIsRecording: (value: boolean) => void;

  // Refs for external access
  documentSelectorRef: React.RefObject<any>;
}

const ChatFlowContext = createContext<ChatFlowContextValue | null>(null);

export function useChatFlow() {
  const context = useContext(ChatFlowContext);
  if (!context) {
    throw new Error('useChatFlow must be used within a ChatFlowProvider');
  }
  return context;
}

interface ChatFlowProviderProps {
  children: ReactNode;
  onOpenDocumentSelector?: () => void;
  onOpenConfigPanel?: () => void;
  onOpenReviewPanel?: () => void;
  onExecuteJob?: (config: ChatFlowConfig) => void;
  onVoiceCommand?: (command: string, params: Record<string, any>) => void;
}

export function ChatFlowProvider({
  children,
  onOpenDocumentSelector,
  onOpenConfigPanel,
  onOpenReviewPanel,
  onExecuteJob,
  onVoiceCommand,
}: ChatFlowProviderProps) {
  const toast = useToast();

  // State
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [currentStep, setCurrentStep] = useState<ChatFlowStep>(ChatFlowStep.IDLE);
  const [config, setConfig] = useState<ChatFlowConfig>(DEFAULT_CONFIG);
  const [messages, setMessages] = useState<ChatFlowMessage[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [interactionMode, setInteractionMode] = useState<'work' | 'talk'>('work');

  // Refs
  const documentSelectorRef = useRef<any>(null);
  const isSpeakingRef = useRef(false);
  const cooldownUntilRef = useRef<number>(0);
  const lastAIResponseRef = useRef<string>('');

  // Add message with deduplication
  const addMessage = useCallback((type: 'user' | 'ai' | 'system', text: string) => {
    const id = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    setMessages(prev => {
      // Check for duplicate
      const lastMsg = prev[prev.length - 1];
      if (lastMsg && lastMsg.type === type && lastMsg.text === text) {
        return prev;
      }
      return [...prev, { id, type, text, timestamp: new Date() }];
    });
  }, []);

  // Start session
  const startSession = useCallback(async () => {
    try {
      const result = await chatFlowApi.startSession();
      if (result.success) {
        setIsSessionActive(true);
        setCurrentStep(ChatFlowStep.IDLE);
        setConfig(DEFAULT_CONFIG);
        setMessages([]);
        addMessage('system', "Voice AI Ready! Say 'print' or 'scan' to begin. Say 'bye printchakra' to end.");
        toast({
          title: 'Voice AI Ready',
          description: "Say 'print' or 'scan' to begin",
          status: 'success',
          duration: 3000,
        });
      }
    } catch (error: any) {
      console.error('Failed to start session:', error);
      toast({
        title: 'Session Start Failed',
        description: error.message || 'Could not start voice session',
        status: 'error',
        duration: 5000,
      });
    }
  }, [addMessage, toast]);

  // End session
  const endSession = useCallback(async () => {
    try {
      await chatFlowApi.endSession();
      setIsSessionActive(false);
      setCurrentStep(ChatFlowStep.IDLE);
      setConfig(DEFAULT_CONFIG);
      addMessage('system', 'Voice session ended. Thank you!');
    } catch (error) {
      console.error('Failed to end session:', error);
    }
  }, [addMessage]);

  // Handle response from backend
  const handleResponse = useCallback(
    (response: ChatFlowResponse) => {
      // Update step
      if (response.current_step) {
        setCurrentStep(response.current_step as ChatFlowStep);
      }
      
      // Update interaction mode
      if (response.interaction_mode) {
        setInteractionMode(response.interaction_mode);
      }

      // Store AI response for echo detection
      lastAIResponseRef.current = (response.tts_response || response.ai_response || '').toLowerCase();

      // Apply config updates
      if (response.config_updates) {
        setConfig(prev => ({ ...prev, ...response.config_updates }));
      }

      // Handle triggers
      if (response.open_document_selector) {
        onOpenDocumentSelector?.();
      }
      if (response.open_config_panel) {
        onOpenConfigPanel?.();
      }
      if (response.open_review_panel) {
        onOpenReviewPanel?.();
      }
      if (response.execute_job && response.command_params?.config) {
        onExecuteJob?.(response.command_params.config);
      }

      // Forward voice commands to parent
      if (response.voice_command && onVoiceCommand) {
        onVoiceCommand(response.voice_command, response.command_params || {});
      }
    },
    [onOpenDocumentSelector, onOpenConfigPanel, onOpenReviewPanel, onExecuteJob, onVoiceCommand]
  );

  // Process text
  const processText = useCallback(
    async (text: string): Promise<ChatFlowResponse | null> => {
      if (!isSessionActive) return null;

      try {
        setIsProcessing(true);
        addMessage('user', text);

        const response = await chatFlowApi.processText(text);

        if (response.success) {
          addMessage('ai', response.ai_response);
          handleResponse(response);
        }

        return response;
      } catch (error: any) {
        console.error('Error processing text:', error);
        addMessage('system', 'Error processing your request. Please try again.');
        return null;
      } finally {
        setIsProcessing(false);
      }
    },
    [isSessionActive, addMessage, handleResponse]
  );

  // Speak text
  const speak = useCallback(async (text: string) => {
    try {
      setIsSpeaking(true);
      isSpeakingRef.current = true;

      const result = await chatFlowApi.speak(text);

      // Dynamic cooldown: 50% of actual TTS duration, minimum 500ms, max 1500ms
      // This allows recording to start shortly after TTS finishes
      const ttsDuration = result.estimatedDuration || 1000;
      const cooldownTime = Math.min(Math.max(ttsDuration * 0.5, 500), 1500);
      cooldownUntilRef.current = Date.now() + cooldownTime;
      
      console.log(`[ChatFlow] TTS took ${ttsDuration}ms, cooldown set to ${cooldownTime}ms`);
    } catch (error) {
      console.error('TTS error:', error);
    } finally {
      setIsSpeaking(false);
      isSpeakingRef.current = false;
    }
  }, []);

  // Process audio
  const processAudio = useCallback(
    async (audioBlob: Blob) => {
      if (!isSessionActive || isSpeakingRef.current) return;

      // Check cooldown
      if (Date.now() < cooldownUntilRef.current) {
        console.log('[ChatFlow] Still in cooldown, skipping');
        return;
      }

      try {
        setIsProcessing(true);

        const result = await chatFlowApi.processAudio(audioBlob);

        if (result.success && result.user_text && result.chat_flow_response) {
          // Echo detection
          const userTextLower = result.user_text.toLowerCase();
          if (lastAIResponseRef.current && isEcho(userTextLower, lastAIResponseRef.current)) {
            console.log('[ChatFlow] Echo detected, ignoring');
            return;
          }

          addMessage('user', result.user_text);
          addMessage('ai', result.chat_flow_response.ai_response);
          handleResponse(result.chat_flow_response);

          // Speak response
          const textToSpeak = result.chat_flow_response.tts_response || result.chat_flow_response.ai_response;
          await speak(textToSpeak);
        }
      } catch (error) {
        console.error('Error processing audio:', error);
      } finally {
        setIsProcessing(false);
      }
    },
    [isSessionActive, addMessage, handleResponse, speak]
  );

  // Update config
  const updateConfig = useCallback((updates: Partial<ChatFlowConfig>) => {
    setConfig(prev => ({ ...prev, ...updates }));
  }, []);

  // Update selected documents
  const updateSelectedDocuments = useCallback(async (documents: string[]) => {
    setConfig(prev => ({ ...prev, selectedDocuments: documents }));
    try {
      await chatFlowApi.updateDocuments(documents);
    } catch (error) {
      console.error('Error updating documents:', error);
    }
  }, []);

  const value: ChatFlowContextValue = {
    isSessionActive,
    currentStep,
    config,
    messages,
    isRecording,
    isProcessing,
    isSpeaking,
    interactionMode,
    startSession,
    endSession,
    processText,
    processAudio,
    speak,
    updateConfig,
    updateSelectedDocuments,
    addMessage,
    setIsRecording,
    documentSelectorRef,
  };

  return <ChatFlowContext.Provider value={value}>{children}</ChatFlowContext.Provider>;
}

// Echo detection helper
function isEcho(transcribed: string, lastResponse: string): boolean {
  if (!lastResponse || !transcribed) return false;

  // Exact match
  if (transcribed.trim() === lastResponse.trim()) return true;

  // Word overlap check
  const responseWords = lastResponse.split(/\s+/).filter(w => w.length > 2);
  const transcribedWords = transcribed.split(/\s+/).filter(w => w.length > 2);

  if (responseWords.length >= 3 && transcribedWords.length >= 3) {
    const matchingWords = responseWords.filter(w => transcribedWords.includes(w));
    const matchRatio = matchingWords.length / Math.min(responseWords.length, transcribedWords.length);
    return matchRatio > 0.6;
  }

  return false;
}
