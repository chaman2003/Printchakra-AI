/**
 * Chat Flow API - Backend communication for voice workflow
 */

import apiClient from '../apiClient';
import { ChatFlowResponse, ChatFlowState } from './chatFlowTypes';

const BASE_URL = '/chat-flow';

export const chatFlowApi = {
  /**
   * Start a new voice chat session
   */
  async startSession(): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post(`${BASE_URL}/start`);
    return response.data;
  },

  /**
   * End the voice chat session
   */
  async endSession(): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post(`${BASE_URL}/end`);
    return response.data;
  },

  /**
   * Get current workflow state
   */
  async getState(): Promise<{ success: boolean; state: ChatFlowState }> {
    const response = await apiClient.get(`${BASE_URL}/state`);
    return response.data;
  },

  /**
   * Process transcribed voice text
   */
  async processText(text: string): Promise<ChatFlowResponse> {
    const response = await apiClient.post(`${BASE_URL}/process`, { text });
    return response.data;
  },

  /**
   * Update selected documents
   */
  async updateDocuments(documents: string[]): Promise<{ success: boolean }> {
    const response = await apiClient.post(`${BASE_URL}/update-documents`, { documents });
    return response.data;
  },

  /**
   * Notify job completion
   */
  async jobCompleted(success: boolean, message?: string): Promise<ChatFlowResponse> {
    const response = await apiClient.post(`${BASE_URL}/job-completed`, {
      success,
      message: message || '',
    });
    return response.data;
  },

  /**
   * Transcribe audio and process through chat flow
   * Uses the existing voice/process endpoint for transcription
   */
  async processAudio(audioBlob: Blob): Promise<{
    success: boolean;
    user_text?: string;
    chat_flow_response?: ChatFlowResponse;
    error?: string;
  }> {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');

    try {
      // Use existing transcription endpoint
      const transcribeResponse = await apiClient.post('/voice/transcribe', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 60000,
      });

      if (!transcribeResponse.data.success || !transcribeResponse.data.text) {
        return {
          success: false,
          error: transcribeResponse.data.error || 'Transcription failed',
        };
      }

      const userText = transcribeResponse.data.text;

      // Process the transcribed text through chat flow
      const chatResponse = await chatFlowApi.processText(userText);

      return {
        success: true,
        user_text: userText,
        chat_flow_response: chatResponse,
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.message || 'Audio processing failed',
      };
    }
  },

  /**
   * Speak text using TTS
   * Returns estimated duration based on text length for cooldown calculation
   */
  async speak(text: string): Promise<{ success: boolean; estimatedDuration?: number }> {
    try {
      const startTime = Date.now();
      const response = await apiClient.post('/voice/speak', { text }, { timeout: 60000 });
      const actualDuration = Date.now() - startTime;
      
      // Return actual duration (how long TTS took) for dynamic cooldown
      return { 
        success: response.data.success, 
        estimatedDuration: actualDuration 
      };
    } catch (error) {
      console.error('TTS error:', error);
      return { success: false, estimatedDuration: 0 };
    }
  },
};
