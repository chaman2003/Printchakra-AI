/**
 * VoiceAIChat Component
 * Hands-free voice AI chat interface with automatic transcription and AI responses
 * Uses Whisper Large-v3 Turbo + Voice AI
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Box,
  Button,
  Flex,
  Heading,
  IconButton,
  Spinner,
  Text,
  VStack,
  HStack,
  useToast,
  Avatar,
  Badge,
  useColorModeValue,
  Input,
  InputGroup,
  InputRightElement,
} from '@chakra-ui/react';
import { FiMic, FiMicOff, FiX, FiSend } from 'react-icons/fi';
import apiClient from '../../apiClient';
import { convertToWAV, isValidAudioBlob, getAudioDuration, VoiceMessage, addMessageWithDedup } from './utils';
import Iconify from '../common/Iconify';

interface VoiceAIChatProps {
  isOpen: boolean;
  // onClose accepts an optional { force: boolean } parameter to allow force-closing
  onClose?: (opts?: { force?: boolean }) => void;
  onOrchestrationTrigger?: (mode: 'print' | 'scan', config?: any) => void;
  isMinimized?: boolean;
  onToggleMinimize?: () => void;
  onVoiceCommand?: (payload: { command: string; params?: Record<string, any> }) => void;
  autoStartRecording?: boolean;
}

const VoiceAIChat: React.FC<VoiceAIChatProps> = ({
  isOpen,
  onClose,
  onOrchestrationTrigger,
  isMinimized = false,
  onToggleMinimize,
  onVoiceCommand,
  autoStartRecording = false,
}) => {
  const [messages, setMessages] = useState<VoiceMessage[]>([]);
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false); // Track if TTS is playing
  const [sessionStatus, setSessionStatus] = useState<string>('');
  const [chatInput, setChatInput] = useState<string>('');
  const [isTextSending, setIsTextSending] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatInputRef = useRef<HTMLInputElement>(null);
  const sessionStartedRef = useRef<boolean>(false);
  const recordingRestartTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isSpeakingRef = useRef(false);
  const isSessionActiveRef = useRef(false);
  // When the user manually stops recording, prevent automatic restarts until they manually start again
  const userStoppedRef = useRef(false);
  const activeToastIdsRef = useRef<Map<string, { timestamp: number; id: string | number }>>(new Map());
  const toastTimeoutRef = useRef<{ [key: string]: ReturnType<typeof setTimeout> }>({});
  // Track last AI response to detect echo (AI hearing itself)
  const lastAIResponseRef = useRef<string>('');
  // Cooldown timestamp - don't record until this time has passed
  const cooldownUntilRef = useRef<number>(0);

  const toast = useToast();

  /**
   * Show toast with proper management - prevents duplicate toasts within time window
   * Uses both deduplication and throttling for maximum protection
   */
  const showToast = useCallback((options: any) => {
    // Validate required properties
    if (!options || typeof options !== 'object') {
      console.warn('Invalid toast options provided');
      return;
    }

    // Create a toast key for deduplication
    const toastKey = `${options.title || ''}_${options.description || ''}_${options.status || 'info'}`;

    // Sanitize content first
    const safeOptions = {
      title: options.title?.toString()?.slice(0, 100) || '',
      description: options.description?.toString()?.slice(0, 200) || '',
      status: options.status || 'info',
      duration: options.duration ?? 3000,
      isClosable: options.isClosable ?? true,
      position: options.position || 'bottom-right',
    };

    // Skip toast if both title and description are empty
    if (!safeOptions.title && !safeOptions.description) {
      console.warn('Toast has no content - skipping');
      return;
    }

    // Check if this exact toast is already being shown
    const existingToast = activeToastIdsRef.current.get(toastKey);
    if (existingToast) {
      const timeSinceShown = Date.now() - existingToast.timestamp;
      // Prevent toast from showing again within 500ms
      if (timeSinceShown < 500) {
        console.log(`Toast throttled (shown ${timeSinceShown}ms ago):`, toastKey);
        return existingToast.id;
      } else {
        // Enough time has passed, remove old tracking
        activeToastIdsRef.current.delete(toastKey);
      }
    }

    // Clear any pending timeout for this toast key
    if (toastTimeoutRef.current[toastKey]) {
      clearTimeout(toastTimeoutRef.current[toastKey]);
      delete toastTimeoutRef.current[toastKey];
    }

    // Limit visible toasts to avoid overflowing the UI
    const VISIBLE_TOAST_LIMIT = 4;
    if (activeToastIdsRef.current.size >= VISIBLE_TOAST_LIMIT) {
      // Find and remove the oldest toast
      let oldestKey: string | null = null;
      let oldestTimestamp = Infinity;
      activeToastIdsRef.current.forEach((val, key) => {
        if (val.timestamp < oldestTimestamp) {
          oldestTimestamp = val.timestamp;
          oldestKey = key;
        }
      });
      if (oldestKey) {
        const oldest = activeToastIdsRef.current.get(oldestKey);
        if (oldest) {
          try {
            toast.close(oldest.id as any);
          } catch (err) {
            // Some Chakra versions do not expose close - ignore safely
            console.warn('Could not close oldest toast programmatically', err);
          }
        }
        activeToastIdsRef.current.delete(oldestKey);
      }
    }

    // Show toast
    const toastId = toast({
      ...safeOptions,
      onCloseComplete: () => {
        // Remove from tracking after a small delay to ensure it's fully closed
        if (toastTimeoutRef.current[toastKey]) {
          clearTimeout(toastTimeoutRef.current[toastKey]);
        }
        toastTimeoutRef.current[toastKey] = setTimeout(() => {
          activeToastIdsRef.current.delete(toastKey);
          delete toastTimeoutRef.current[toastKey];
        }, 100);
        options.onCloseComplete?.();
      },
    });

    // Track this toast with timestamp
    activeToastIdsRef.current.set(toastKey, {
      timestamp: Date.now(),
      id: toastId,
    });

    return toastId;
  }, [toast]);

  // Color mode values - Updated to match SurfaceCard theme
  const bgColor = useColorModeValue('rgba(255, 248, 240, 0.95)', 'rgba(12, 16, 35, 0.92)');
  const borderColor = useColorModeValue('rgba(121, 95, 238, 0.18)', 'rgba(69, 202, 255, 0.25)');
  const userMessageBg = useColorModeValue('brand.50', 'rgba(121, 95, 238, 0.15)');
  const aiMessageBg = useColorModeValue('orange.50', 'rgba(255, 255, 255, 0.05)');
  const systemMessageBg = useColorModeValue('yellow.50', 'rgba(236, 201, 75, 0.1)');
  const chatBoxBg = useColorModeValue('orange.50', 'rgba(0, 0, 0, 0.2)');
  const shadow = useColorModeValue(
    '0 20px 35px rgba(121, 95, 238, 0.08)',
    '0 25px 50px rgba(0, 0, 0, 0.8)'
  );

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Cleanup on unmount - stop recording and end session
  useEffect(() => {
    return () => {
      // Stop recording if active
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        mediaRecorderRef.current.stop();
        if (mediaRecorderRef.current.stream) {
          mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
        }
      }
      if (recordingRestartTimeoutRef.current) {
        clearTimeout(recordingRestartTimeoutRef.current);
        recordingRestartTimeoutRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    isSpeakingRef.current = isSpeaking;
    
    // When speaking stops, schedule clearing of last response after 10 seconds
    // This prevents false echo detection on unrelated future speech
    if (!isSpeaking && lastAIResponseRef.current) {
      const clearTimer = setTimeout(() => {
        lastAIResponseRef.current = '';
      }, 10000);
      return () => clearTimeout(clearTimer);
    }
  }, [isSpeaking]);

  useEffect(() => {
    isSessionActiveRef.current = isSessionActive;
    if (!isSessionActive && recordingRestartTimeoutRef.current) {
      clearTimeout(recordingRestartTimeoutRef.current);
      recordingRestartTimeoutRef.current = null;
    }
  }, [isSessionActive]);

  // Start voice AI session when drawer opens
  useEffect(() => {
    if (isOpen && !isSessionActive && !sessionStartedRef.current) {
      sessionStartedRef.current = true;
      startSession();
    }
    if (!isOpen) {
      sessionStartedRef.current = false;
    }
    // Focus chat input when drawer opens
    if (isOpen) {
      setTimeout(() => {
        chatInputRef.current?.focus();
      }, 300); // Wait for drawer animation
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, isSessionActive]);

  // Auto-start recording when chat opens if autoStartRecording is true
  useEffect(() => {
    if (isOpen && autoStartRecording && isSessionActive && !isRecording) {
      // Delay to ensure session is fully initialized
      const timer = setTimeout(() => {
        startRecording();
      }, 500);
      return () => clearTimeout(timer);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, autoStartRecording, isSessionActive]);

  // Ensure chat input can receive focus when clicked, even with modal open
  useEffect(() => {
    const inputElement = chatInputRef.current;
    if (!inputElement) return;

    const handleFocus = () => {
      inputElement.focus();
    };

    inputElement.addEventListener('click', handleFocus, true);
    return () => inputElement.removeEventListener('click', handleFocus, true);
  }, []);

  const addMessage = useCallback((type: 'user' | 'ai' | 'system', text: string) => {
    setMessages(prev => addMessageWithDedup(prev, type, text));
  }, []);

  const stopRecording = useCallback((manual = true) => {
    // Mark that the user explicitly stopped recording so we don't auto-restart
    // Only set this if it was a manual stop (clicking the button)
    if (manual) {
      userStoppedRef.current = true;
    }

    // Clear any pending restart timers
    if (recordingRestartTimeoutRef.current) {
      clearTimeout(recordingRestartTimeoutRef.current);
      recordingRestartTimeoutRef.current = null;
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      try {
        mediaRecorderRef.current.stop();
      } catch (e) {
        console.warn('[stopRecording] Error stopping media recorder:', e);
      }
      setIsRecording(false);
    }

    // Ensure we stop all media tracks to fully release microphone
    try {
      if (mediaRecorderRef.current?.stream) {
        mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      }
    } catch (e) {
      console.warn('[stopRecording] Error stopping media tracks:', e);
    }
  }, []);


  const startRecording = useCallback(async () => {
    console.log('[startRecording] Starting... isSessionActive:', isSessionActiveRef.current);

    // Clear a manual stop flag when user actively starts recording
    userStoppedRef.current = false;

    if (!isSessionActiveRef.current) {
      console.warn('[startRecording] Voice session inactive');
      return;
    }

    if (recordingRestartTimeoutRef.current) {
      clearTimeout(recordingRestartTimeoutRef.current);
      recordingRestartTimeoutRef.current = null;
    }

    if (isSpeakingRef.current) {
      console.log('TTS is speaking - delaying recording restart');
      scheduleRecordingStart(200);
      return;
    }

    // Check cooldown period (prevents mic from picking up speaker audio)
    const now = Date.now();
    if (now < cooldownUntilRef.current) {
      const remainingCooldown = cooldownUntilRef.current - now;
      console.log(`[startRecording] Still in cooldown, waiting ${remainingCooldown}ms`);
      scheduleRecordingStart(remainingCooldown + 100);
      return;
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      console.log('Recorder already active - keeping current stream');
      return;
    }

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Microphone access not supported in this browser');
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 48000,
          channelCount: 1,
          sampleSize: 16,
        },
      });

      const mimeTypes = ['audio/wav', 'audio/webm', 'audio/webm;codecs=opus', 'audio/mp4'];

      let selectedMimeType = 'audio/webm';
      for (const mimeType of mimeTypes) {
        if (MediaRecorder.isTypeSupported(mimeType)) {
          selectedMimeType = mimeType;
          console.log(`Using MIME type: ${mimeType}`);
          break;
        }
      }

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: selectedMimeType,
        audioBitsPerSecond: 256000,
      });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = event => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        try {
          let audioBlob = new Blob(audioChunksRef.current, { type: mediaRecorder.mimeType });

          if (!isValidAudioBlob(audioBlob)) {
            console.error('Invalid audio blob generated');
            throw new Error('Audio blob is invalid');
          }

          console.log(`Raw audio blob: ${audioBlob.size} bytes, type: ${audioBlob.type}`);

          if (audioBlob.type !== 'audio/wav' && !audioBlob.type.includes('wav')) {
            console.log('Converting audio to WAV format...');
            audioBlob = await convertToWAV(audioBlob);
            console.log(`Converted to WAV: ${audioBlob.size} bytes`);
          }

          console.log(`Final audio blob: ${audioBlob.size} bytes, type: ${audioBlob.type}`);

          const duration = await getAudioDuration(audioBlob);
          console.log(`Audio duration: ${duration.toFixed(2)}s`);

          // Skip only extremely short audio (less than 0.3 seconds)
          if (duration < 0.3) {
            console.log('Audio too short (< 0.3s) - skipping processing');
            scheduleRecordingStart(300);
            return;
          }

          // For very short audio (0.3-1 second), do basic voice detection
          // Longer audio goes straight to Whisper
          if (duration < 1.0) {
            const { hasVoiceActivity } = await import('../../utils/audioUtils');
            // Very lenient threshold for detecting ANY speech
            const hasVoice = await hasVoiceActivity(audioBlob, 0.003);

            if (!hasVoice) {
              console.log('No voice detected in short audio - skipping processing');
              scheduleRecordingStart(300);
              return;
            }
          }

          // Process audio - let Whisper decide what it hears
          await processAudio(audioBlob);
        } catch (error: any) {
          console.error('Error in onstop handler:', error);
          const _errMsg = error?.response?.data?.error || error?.message || String(error);
          // Suppress decode-related errors to avoid alarming users
          if (typeof _errMsg === 'string' && /decode/i.test(_errMsg)) {
            console.warn('Audio decode failure suppressed (not shown to user):', _errMsg);
          } else {
            showToast({
              title: 'Audio Processing Error',
              description: _errMsg,
              status: 'error',
              duration: 5000,
            });
          }

          scheduleRecordingStart(600);
        } finally {
          stream.getTracks().forEach(track => track.stop());
        }
      };

      mediaRecorder.onerror = (event: any) => {
        console.error('MediaRecorder error:', event.error);
        showToast({
          title: 'Recording Error',
          description: event.error || 'An error occurred during recording',
          status: 'error',
          duration: 5000,
        });
        scheduleRecordingStart(600);
      };

      mediaRecorder.start();
      setIsRecording(true);
      console.log('[startRecording] Recording started');

      const audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.8;
      source.connect(analyser);

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      let silenceStart: number | null = null;
      let speechDetected = false;
      let audioContextClosed = false;
      // Increased threshold to detect speech better (was 8, now 20 - more robust)
      const SILENCE_THRESHOLD = 20;
      // Reduced silence duration for faster response (was 1500ms, now 800ms)
      const SILENCE_DURATION = 800;

      const closeAudioContext = () => {
        if (!audioContextClosed && audioContext.state !== 'closed') {
          try {
            audioContext.close();
            audioContextClosed = true;
          } catch (e) {
            // Already closed
          }
        }
      };

      const checkAudioLevel = () => {
        if (mediaRecorderRef.current?.state !== 'recording' || audioContextClosed) {
          closeAudioContext();
          return;
        }

        analyser.getByteTimeDomainData(dataArray);

        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
          const normalized = (dataArray[i] - 128) / 128;
          sum += normalized * normalized;
        }
        const rms = Math.sqrt(sum / bufferLength);
        const audioLevel = rms * 100;

        if (audioLevel > SILENCE_THRESHOLD) {
          speechDetected = true;
          silenceStart = null;
        } else if (speechDetected && audioLevel <= SILENCE_THRESHOLD) {
          if (silenceStart === null) {
            silenceStart = Date.now();
          } else if (Date.now() - silenceStart > SILENCE_DURATION) {
            console.log('Silence detected after speech - stopping recording');
            stopRecording(false);
            closeAudioContext();
            return;
          }
        }

        requestAnimationFrame(checkAudioLevel);
      };

      checkAudioLevel();

      // Increased max recording duration (was 15 seconds, now 20 seconds for safety)
      setTimeout(() => {
        if (mediaRecorderRef.current?.state === 'recording') {
          console.log('Max duration reached - stopping recording');
          stopRecording(false);
          closeAudioContext();
        }
      }, 20000);

    } catch (error: any) {
      console.error('Recording error:', error);
      showToast({
        title: 'Microphone Error',
        description: error.message || 'Could not access microphone',
        status: 'error',
        duration: 5000,
      });

      scheduleRecordingStart(1000);
    }
  }, [showToast]);

  // Ensure we only have one pending restart timer and we always wait for TTS to finish
  const scheduleRecordingStart = useCallback((delay = 0) => {
    if (!isSessionActiveRef.current) {
      return;
    }

    // Respect user's explicit stop - do not auto-restart until they manually start again
    if (userStoppedRef.current) {
      console.log('[scheduleRecordingStart] User has manually stopped recording - not auto-restarting');
      return;
    }

    if (recordingRestartTimeoutRef.current) {
      clearTimeout(recordingRestartTimeoutRef.current);
    }

    // Check cooldown and adjust delay if needed
    const now = Date.now();
    let adjustedDelay = delay;
    if (now < cooldownUntilRef.current) {
      const cooldownRemaining = cooldownUntilRef.current - now;
      adjustedDelay = Math.max(delay, cooldownRemaining + 100);
      console.log(`[scheduleRecordingStart] Adjusting delay for cooldown: ${adjustedDelay}ms`);
    }

    recordingRestartTimeoutRef.current = setTimeout(() => {
      recordingRestartTimeoutRef.current = null;

      if (!isSessionActiveRef.current) {
        return;
      }

      if (isSpeakingRef.current) {
        scheduleRecordingStart(200);
        return;
      }
      
      // Double-check cooldown before starting
      if (Date.now() < cooldownUntilRef.current) {
        const remaining = cooldownUntilRef.current - Date.now();
        scheduleRecordingStart(remaining + 100);
        return;
      }

      if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        return;
      }

      startRecording();
    }, Math.max(0, adjustedDelay));
  }, []);

  const startSession = async () => {
    try {
      setSessionStatus('Loading Whisper AI model (smaller model, faster loading)...');

      const response = await apiClient.post(
        '/voice/start',
        {},
        {
          timeout: 120000,
        }
      );

      if (response.data.success) {
        setIsSessionActive(true);
        setSessionStatus('Ready - Just speak naturally');
        addMessage(
          'system',
          'Voice AI Ready! Just speak naturally - Say "bye printchakra" to end.'
        );

        showToast({
          title: 'Voice AI Ready',
          description: 'Recording started - Speak now',
          status: 'success',
          duration: 4000,
        });

        // Start recording immediately
        startRecording();
      } else {
        throw new Error(response.data.error || 'Failed to start session');
      }
    } catch (error: any) {
      console.error('Session start error:', error);
      setSessionStatus('Failed');

      const errorMsg =
        error.response?.data?.error || error.message || 'Could not start voice AI session';

      showToast({
        title: 'Session Start Failed',
        description: errorMsg,
        status: 'error',
        duration: 5000,
      });
    }
  };

  /**
   * Check if transcribed text is likely an echo of the last AI response
   * This prevents the AI from responding to its own speech
   */
  const isEchoOfLastResponse = (transcribedText: string): boolean => {
    if (!lastAIResponseRef.current || !transcribedText) return false;
    
    const lastResponse = lastAIResponseRef.current.toLowerCase().trim();
    const transcribed = transcribedText.toLowerCase().trim();
    
    // If the transcription is very similar to what AI just said, it's an echo
    // Check for exact match or high similarity
    if (transcribed === lastResponse) {
      console.log('[ECHO DETECTED] Exact match with last AI response');
      return true;
    }
    
    // Check if transcription starts with or contains AI's response
    if (transcribed.includes(lastResponse) || lastResponse.includes(transcribed)) {
      console.log('[ECHO DETECTED] Transcription contains AI response or vice versa');
      return true;
    }
    
    // Check for common AI phrases that shouldn't be repeated by user
    const aiOnlyPhrases = [
      "i'm glad you asked",
      "the following is a sample",
      "please say",
      "insert text here",
      "what options would you like",
      "opening scan interface",
      "opening print interface",
      "do you want to print",
      "do you want to scan",
      "say yes to proceed",
      "printchakra ai",
      "voice session",
      "i can help you",
    ];
    
    for (const phrase of aiOnlyPhrases) {
      if (transcribed.includes(phrase)) {
        console.log(`[ECHO DETECTED] Transcription contains AI-only phrase: "${phrase}"`);
        return true;
      }
    }
    
    // Check if transcription contains most of the AI response (echo pickup)
    const lastResponseWords = lastResponse.split(/\s+/).filter(w => w.length > 2);
    const transcribedWords = transcribed.split(/\s+/).filter(w => w.length > 2);
    
    if (lastResponseWords.length >= 3 && transcribedWords.length >= 3) {
      // Count matching words
      const matchingWords = lastResponseWords.filter(w => transcribedWords.includes(w));
      const matchRatio = matchingWords.length / Math.min(lastResponseWords.length, transcribedWords.length);
      
      // Lower threshold to catch more echoes (was 0.6, now 0.5)
      if (matchRatio > 0.5) {
        console.log(`[ECHO DETECTED] ${(matchRatio * 100).toFixed(0)}% word match with last AI response`);
        return true;
      }
    }
    
    // Check if very long transcription overlaps significantly with AI response
    if (transcribedWords.length > 10) {
      const overlapCount = transcribedWords.filter(w => lastResponseWords.includes(w)).length;
      if (overlapCount > 5) {
        console.log(`[ECHO DETECTED] Long transcription has ${overlapCount} overlapping words`);
        return true;
      }
    }
    
    return false;
  };

  const processAudio = async (audioBlob: Blob) => {
    if (isSpeaking) {
      console.log('TTS is speaking - skipping audio processing');
      scheduleRecordingStart(200);
      return;
    }
    
    // Check cooldown (extra safety against echo)
    if (Date.now() < cooldownUntilRef.current) {
      console.log('[processAudio] Still in cooldown, skipping');
      scheduleRecordingStart(500);
      return;
    }

    try {
      setIsProcessing(true);
      setSessionStatus('Transcribing audio...');

      console.log(`Processing audio blob: ${audioBlob.size} bytes, type: ${audioBlob.type}`);

      // Create form data
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.wav');

      console.log('Sending to backend...');

      // Send to backend for processing (Whisper → Voice AI)
      // Use longer timeout for voice processing (model loading can take time)
      const response = await apiClient.post('/voice/process', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 120000, // 2 minutes for voice processing
      });

      console.log('Backend response:', response.data);

      // Debug: Log exactly what Whisper heard vs what AI processed
      if (response.data.user_text) {
        console.log(`🎤 Whisper transcribed: "${response.data.user_text}"`);
        console.log(`🤖 AI response: "${response.data.ai_response}"`);
        console.log(`📋 Voice command: ${response.data.voice_command || 'none'}`);
        console.log(`🎯 Orchestration: ${response.data.orchestration_trigger ? response.data.orchestration_mode : 'none'}`);
        
        // Echo detection - check if AI is hearing itself
        if (isEchoOfLastResponse(response.data.user_text)) {
          console.log('🔇 ECHO DETECTED - AI heard itself, ignoring and restarting recording');
          setIsProcessing(false);
          setSessionStatus('Ready - Just speak naturally');
          scheduleRecordingStart(1500);
          return;
        }
      }

      // Check for no speech detected (auto-retry)
      if (response.data.auto_retry && response.data.no_speech_detected) {
        console.log('No human speech detected - auto-retrying in 1 second');

        addMessage('system', '⚠️ No human speech detected. Retrying...');

        setSessionStatus('No speech detected, retrying...');
        setIsProcessing(false);

        scheduleRecordingStart(600);
        return;
      }

      if (response.data.success) {
        const {
          user_text,
          full_text,
          ai_response,
          tts_response,
          session_ended,
          keyword_detected,
          orchestration_trigger,
          orchestration_mode,
          config_params,
          voice_command,
          command_params,
          orchestration,
          // New ChatFlow fields
          current_step,
          open_document_selector,
          open_config_panel,
          open_review_panel,
          execute_job,
          config_updates,
        } = response.data;

        // Store AI response for echo detection
        lastAIResponseRef.current = (tts_response || ai_response || '').toLowerCase();

        // Add user message only (no duplicate "heard" message)
        addMessage('user', user_text);

        // AI response should already be complete and under 25 words (no truncation needed)
        addMessage('ai', ai_response);

        setIsProcessing(false);

        // Handle voice commands (document selector control, navigation, settings)
        if (voice_command && voice_command.trim()) {
          const payload = {
            command: voice_command,
            params: command_params || {},
          };

          console.log(`[ChatFlow] Voice command: ${voice_command}`, payload.params);
          onVoiceCommand?.(payload);

          showToast({
            title: `Voice Command: ${voice_command.replace('_', ' ').toUpperCase()}`,
            description:
              Object.keys(payload.params || {}).length > 0
                ? `Executing: ${JSON.stringify(payload.params)}`
                : 'Processing command...',
            status: 'info',
            duration: 2000,
          });
        }

        // Handle ChatFlow UI triggers
        if (open_document_selector) {
          console.log('[ChatFlow] Opening document selector');
          showToast({
            title: 'Document Selector',
            description: 'Select your documents',
            status: 'info',
            duration: 3000,
          });
          // Trigger orchestration for document selection
          if (onOrchestrationTrigger) {
            const mode = orchestration_mode || 'print';
            onOrchestrationTrigger(mode as 'print' | 'scan', { step: 'selecting_docs' });
          }
        }
        
        if (open_config_panel) {
          console.log('[ChatFlow] Opening configuration panel');
          showToast({
            title: 'Configuration',
            description: 'Adjust your settings',
            status: 'info',
            duration: 3000,
          });
        }
        
        if (open_review_panel) {
          console.log('[ChatFlow] Opening review panel');
          showToast({
            title: 'Review',
            description: 'Review your settings',
            status: 'info',
            duration: 3000,
          });
        }
        
        if (execute_job) {
          console.log('[ChatFlow] Executing job', command_params);
          showToast({
            title: 'Executing Job',
            description: `Starting ${command_params?.mode || 'operation'}...`,
            status: 'success',
            duration: 3000,
          });
        }
        
        // Apply configuration updates if provided
        if (config_updates && Object.keys(config_updates).length > 0) {
          console.log('[ChatFlow] Configuration updates:', config_updates);
          // Emit config_update event for Dashboard to handle
          onVoiceCommand?.({
            command: 'config_update',
            params: config_updates,
          });
        }

        // Check for orchestration trigger BEFORE TTS (legacy support)
        if (orchestration_trigger && orchestration_mode) {
          const frontendState = orchestration?.frontend_state;
          const orchestrationPayload = frontendState
            ? frontendState
            : config_params
              ? { mode: orchestration_mode, options: config_params }
              : undefined;

          console.log(
            `🎯 Orchestration triggered: ${orchestration_mode}`,
            orchestrationPayload || config_params || {}
          );

          // Show notification about orchestration
          showToast({
            title: `Opening ${orchestration_mode === 'print' ? 'Print' : 'Scan'} Interface`,
            description: 'Orchestration system activated',
            status: 'info',
            duration: 3000,
          });

          // Trigger orchestration in parent component
          if (onOrchestrationTrigger) {
            // Delay slightly to allow TTS to start
            setTimeout(() => {
              onOrchestrationTrigger(
                orchestration_mode as 'print' | 'scan',
                orchestrationPayload || config_params || undefined
              );
            }, 500);
          }
        }
        // 2. STOP RECORDING before TTS begins (mutually exclusive states)
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
          stopRecording(false);
        }

        // 3. THEN play TTS (blocking - no input allowed)
        setIsSpeaking(true);
        setIsRecording(false); // Ensure recording badge is off
        setSessionStatus('Speaking response...');

        try {
          // Use shortened TTS response if available, otherwise use full response
          const textToSpeak = tts_response || ai_response;
          await apiClient.post(
            '/voice/speak',
            {
              text: textToSpeak,
            },
            {
              timeout: 60000,
            }
          );
          
          // Set cooldown period after TTS completes (3.5 seconds to let audio fully dissipate)
          // This prevents the mic from picking up residual speaker audio
          // Increased from 2.5s to 3.5s for better echo prevention
          cooldownUntilRef.current = Date.now() + 3500;
          console.log('[TTS] Speech complete, cooldown until:', new Date(cooldownUntilRef.current).toISOString());
        } catch (ttsError) {
          console.error('TTS error:', ttsError);
          // Continue even if TTS fails
        } finally {
          setIsSpeaking(false);
          setIsRecording(false); // Ensure recording is fully stopped
          setSessionStatus('Ready - Just speak naturally');
          // Focus chat input after TTS completes
          setTimeout(() => {
            chatInputRef.current?.focus();
          }, 100);

          // Check if session should end
          if (session_ended) {
            setIsSessionActive(false);
            addMessage('system', 'Voice session ended. Thank you!');

            showToast({
              title: 'Session Ended',
              description: 'Goodbye!',
              status: 'info',
              duration: 3000,
            });
          } else {
            // Auto-restart recording with 3.5 second delay after TTS completes
            // This gives time for speaker audio to fully dissipate before mic listens again
            // Increased from 2.5s to 3.5s for better echo prevention
            scheduleRecordingStart(3500);
          }
        }
      } else {
        throw new Error(response.data.error || 'Processing failed');
      }
    } catch (error: any) {
      console.error('Audio processing error:', error);

      // Provide detailed error information
      let errorMessage = error.message || 'Could not process audio';
      if (error.response?.data?.error) {
        errorMessage = error.response.data.error;
        if (error.response.data.details) {
          console.error('Backend details:', error.response.data.details);
        }
      }

      // Check if it's a keyword-related error
      if (error.response?.data?.requires_keyword) {
        // Silent retry for keyword errors - don't spam user
        console.log(`Keyword detection error, retrying...`);
        setSessionStatus('Ready - Just speak naturally');

        scheduleRecordingStart(300);
      } else {
        // Only show non-retryable errors
        console.error('Transcription error:', errorMessage);

        // Don't show file access errors - just retry silently
        if (errorMessage.includes('process cannot access') || errorMessage.includes('being used')) {
          console.log('File access error, retrying...');
          setSessionStatus('Ready - Just speak naturally');

          scheduleRecordingStart(500);
        } else {
          // Show only critical errors
          showToast({
            title: 'Processing Error',
            description: errorMessage,
            status: 'error',
            duration: 3000,
          });

          setSessionStatus('Ready - Just speak naturally');

          scheduleRecordingStart(700);
        }
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const endSession = async () => {
    try {
      // Stop recording if active
      if (isRecording && mediaRecorderRef.current) {
        if (mediaRecorderRef.current.state === 'recording') {
          mediaRecorderRef.current.stop();
        }
        setIsRecording(false);

        // Stop all media tracks
        if (mediaRecorderRef.current.stream) {
          mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
        }
      }

      await apiClient.post('/voice/end');

      setIsSessionActive(false);
      addMessage('system', '🛑 Session ended manually.');

      showToast({
        title: 'Session Ended',
        description: 'Recording stopped and session closed',
        status: 'info',
        duration: 2000,
      });
    } catch (error) {
      console.error('End session error:', error);
      // Force cleanup even on error
      setIsRecording(false);
      setIsSessionActive(false);
      if (mediaRecorderRef.current?.stream) {
        mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      }
    }
  };

  const sendTextMessage = async (text: string) => {
    if (!text.trim() || isSpeaking) return;

    try {
      setIsTextSending(true);
      setChatInput('');

      // STOP recording when text message is sent
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        mediaRecorderRef.current.stop();
        setIsRecording(false);
        if (recordingRestartTimeoutRef.current) {
          clearTimeout(recordingRestartTimeoutRef.current);
          recordingRestartTimeoutRef.current = null;
        }
      }

      // Add user message to chat immediately
      addMessage('user', text);

      // Get AI response (text only, no TTS yet)
      setIsProcessing(true);

      const response = await apiClient.post(
        '/voice/chat',
        {
          message: text,
        },
        {
          timeout: 30000,
        }
      );

      if (response.data.success) {
        const aiResponse = response.data.response;
        const ttsResponse = response.data.tts_response;
        const orchestrationTrigger = response.data.orchestration_trigger;
        const orchestrationMode = response.data.orchestration_mode;
        const configParams = response.data.config_params;
        const voiceCommand = response.data.voice_command;
        const commandParams = response.data.command_params;

        console.log('Backend response:', {
          aiResponse,
          ttsResponse,
          orchestrationTrigger,
          orchestrationMode,
          configParams,
        });

        // 1. Display AI message FIRST
        addMessage('ai', aiResponse);
        setIsProcessing(false);

        // Handle voice commands (document selector control)
        if (voiceCommand && voiceCommand.trim()) {
          const payload = {
            command: voiceCommand,
            params: commandParams || {},
          };

          console.log(`Voice command detected (text): ${voiceCommand}`, payload.params);
          onVoiceCommand?.(payload);

          showToast({
            title: `Voice Command: ${voiceCommand.replace('_', ' ').toUpperCase()}`,
            description:
              Object.keys(payload.params || {}).length > 0
                ? `Executing: ${JSON.stringify(payload.params)}`
                : 'Processing command...',
            status: 'info',
            duration: 2000,
          });
        }

        // Check for orchestration trigger BEFORE TTS
        if (orchestrationTrigger && orchestrationMode) {
          console.log(`Orchestration triggered: ${orchestrationMode}`, configParams || {});

          // Show notification
          showToast({
            title: `Opening ${orchestrationMode === 'print' ? 'Print' : 'Scan'} Interface`,
            description: 'Orchestration system activated',
            status: 'info',
            duration: 3000,
          });

          // Trigger orchestration
          if (onOrchestrationTrigger) {
            setTimeout(() => {
              onOrchestrationTrigger(orchestrationMode as 'print' | 'scan', configParams);
            }, 500);
          }
        }

        // 2. THEN play TTS (blocking - no input allowed until complete)
        setIsSpeaking(true);
        isSpeakingRef.current = true;
        setSessionStatus('Speaking response...');

        try {
          // Use shortened TTS response if available, otherwise use full response
          const textToSpeak = ttsResponse || aiResponse;
          await apiClient.post(
            '/voice/speak',
            {
              text: textToSpeak,
            },
            {
              timeout: 60000, // TTS can take time for long responses
            }
          );
        } catch (ttsError) {
          console.error('TTS error:', ttsError);
          // Don't show error to user, TTS failure is non-critical
        } finally {
          setIsSpeaking(false);
          isSpeakingRef.current = false;
          setSessionStatus('Ready - Speak naturally');
          // Focus chat input after TTS completes
          setTimeout(() => {
            chatInputRef.current?.focus();
          }, 100);
          // Auto-restart recording with 1 second delay
          scheduleRecordingStart(1000);
        }
      } else {
        setIsProcessing(false);
        const errorMsg = response.data.error || 'Unknown error occurred';
        addMessage('system', `❌ Error: ${errorMsg}`);
        showToast({
          title: 'Message Error',
          description: errorMsg,
          status: 'error',
          duration: 5000,
        });
        // Auto-restart recording with 1 second delay after error
        scheduleRecordingStart(1000);
      }
    } catch (error: any) {
      console.error('Text message error:', error);
      setIsProcessing(false);
      setIsSpeaking(false);
      isSpeakingRef.current = false;
      const errorMsg = error.response?.data?.error || error.message || 'Failed to process message';
      addMessage('system', `❌ Error: ${errorMsg}`);
      showToast({
        title: 'Message Error',
        description: errorMsg,
        status: 'error',
        duration: 5000,
      });
      // Resume recording after error
      scheduleRecordingStart(600);
    } finally {
      setIsTextSending(false);
    }
  };

  const handleChatInputKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendTextMessage(chatInput);
    }
  };

  const handleClose = async () => {
    try {
      // Stop recording if active
      if (isRecording && mediaRecorderRef.current) {
        if (mediaRecorderRef.current.state === 'recording') {
          mediaRecorderRef.current.stop();
        }
        setIsRecording(false);

        // Stop all media tracks
        if (mediaRecorderRef.current.stream) {
          mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
        }
      }

      // End session if active
      if (isSessionActive) {
        try {
          await apiClient.post('/voice/end');
        } catch (error) {
          console.error('Error ending session on close:', error);
        }
        setIsSessionActive(false);
      }

      // Clear messages and close
      setMessages([]);
      setSessionStatus('');
      // Force-close the chat on user request (if provided)
      onClose?.({ force: true });

      showToast({
        title: 'Voice AI Closed',
        description: 'Recording stopped and session ended',
        status: 'info',
        duration: 2000,
      });
    } catch (error) {
      console.error('Error closing drawer:', error);
      // Force close anyway
      setMessages([]);
      setSessionStatus('');
      setIsRecording(false);
      setIsSessionActive(false);

      // Force stop media tracks
      if (mediaRecorderRef.current?.stream) {
        mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      }

      onClose?.();
    }
  };

  return (
    <VStack flex="1" spacing={0} align="stretch" h="100%" w="100%" bg={bgColor} position="relative" zIndex={2000} pointerEvents="auto">
      {/* Header */}
      <Box
        borderBottomWidth="1px"
        borderColor={borderColor}
        p={4}
        bg={bgColor}
      >
        <Flex align="center" gap={3} justify="space-between">
          <Flex align="center" gap={3} flex="1">
            {!isMinimized && (
              <>
                <Avatar size="sm" name="PrintChakra AI" bg="brand.500" />
                <Box flex="1">
                  <Heading size="sm" fontWeight="700">PrintChakra AI</Heading>
                  <HStack spacing={2} mt={1} flexWrap="wrap">
                    <Badge colorScheme={isSessionActive ? 'green' : 'gray'} fontSize="xs" variant="subtle" borderRadius="full" px={2}>{sessionStatus}</Badge>
                    {/* Mutually exclusive state badges - only show one at a time */}
                    {isSpeaking ? (
                      <Badge colorScheme="purple" fontSize="xs" variant="subtle" borderRadius="full" px={2}>
                        <HStack spacing={1}>
                          <Spinner size="xs" />
                          <Text>Speaking</Text>
                        </HStack>
                      </Badge>
                    ) : isProcessing ? (
                      <Badge colorScheme="blue" fontSize="xs" variant="subtle" borderRadius="full" px={2}>
                        <HStack spacing={1}>
                          <Spinner size="xs" />
                          <Text>Processing</Text>
                        </HStack>
                      </Badge>
                    ) : isRecording ? (
                      <Badge colorScheme="red" variant="solid" fontSize="xs" borderRadius="full" px={2}>
                        <HStack spacing={1}>
                          <Box
                            w={1.5}
                            h={1.5}
                            borderRadius="full"
                            bg="white"
                            animation="pulse 1.5s infinite"
                          />
                          <Text>RECORDING</Text>
                        </HStack>
                      </Badge>
                    ) : null}
                  </HStack>
                </Box>
              </>
            )}
          </Flex>
          <IconButton
            aria-label="Close chat"
            icon={<Iconify icon={FiX} boxSize={5} />}
            size="sm"
            variant="ghost"
            colorScheme="red"
            onClick={handleClose}
            borderRadius="full"
          />
        </Flex>
      </Box>

      {/* Chat Body */}
      {!isMinimized && (
        <>
          <Box flex="1" overflowY="auto" p={4} css={{
            '&::-webkit-scrollbar': { width: '4px' },
            '&::-webkit-scrollbar-track': { width: '6px' },
            '&::-webkit-scrollbar-thumb': { background: borderColor, borderRadius: '24px' },
          }}>
            <VStack spacing={4} align="stretch">
              {messages.length === 0 && (
                <Flex
                  direction="column"
                  align="center"
                  justify="center"
                  height="100%"
                  color="gray.500"
                  py={10}
                >
                  <Box p={4} bg={userMessageBg} borderRadius="full" mb={4}>
                    <Iconify icon={FiMic} boxSize={8} color="brand.500" />
                  </Box>
                  <Text fontWeight="600" textAlign="center">
                    Start talking with PrintChakra AI
                  </Text>
                  <Text fontSize="sm" mt={2} textAlign="center" color="gray.400">
                    Say "bye printchakra" to end
                  </Text>
                </Flex>
              )}

              {messages.map(message => (
                <Box
                  key={message.id}
                  alignSelf={message.type === 'user' ? 'flex-end' : 'flex-start'}
                  maxW="85%"
                >
                  {message.type === 'system' ? (
                    <Box bg={systemMessageBg} px={4} py={2} borderRadius="xl" textAlign="center" border="1px solid" borderColor="yellow.200">
                      <Text fontSize="xs" fontStyle="italic" color="yellow.800">
                        {message.text}
                        {message.count && message.count > 1 && (
                          <Badge ml={2} colorScheme="orange" variant="solid" borderRadius="full">
                            ×{message.count}
                          </Badge>
                        )}
                      </Text>
                    </Box>
                  ) : (
                    <Flex
                      direction={message.type === 'user' ? 'row-reverse' : 'row'}
                      gap={3}
                      align="flex-end"
                    >
                      <Avatar
                        size="xs"
                        name={message.type === 'user' ? 'You' : 'PrintChakra AI'}
                        bg={message.type === 'user' ? 'brand.500' : 'green.500'}
                      />
                      <Box
                        bg={message.type === 'user' ? userMessageBg : aiMessageBg}
                        px={4}
                        py={3}
                        borderRadius="2xl"
                        borderTopRightRadius={message.type === 'user' ? 'sm' : '2xl'}
                        borderTopLeftRadius={message.type === 'user' ? '2xl' : 'sm'}
                        boxShadow="sm"
                      >
                        <Text fontSize="sm" lineHeight="tall">{message.text}</Text>
                        <Text fontSize="10px" color="gray.500" mt={1} textAlign={message.type === 'user' ? 'right' : 'left'}>
                          {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </Text>
                      </Box>
                    </Flex>
                  )}
                </Box>
              ))}

              <div ref={messagesEndRef} />
            </VStack>
          </Box>

          <Box borderTopWidth="1px" borderColor={borderColor} p={4} bg={chatBoxBg}>
            <InputGroup size="md" mb={2}>
              <Input
                ref={chatInputRef}
                placeholder={isSpeaking ? "Wait for AI to finish speaking..." : "Type your message..."}
                value={chatInput}
                onChange={e => setChatInput(e.target.value)}
                onKeyPress={handleChatInputKeyPress}
                borderRadius="xl"
                bg={useColorModeValue('white', 'rgba(0,0,0,0.2)')}
                border="1px solid"
                borderColor={borderColor}
                _focus={{ borderColor: 'brand.400', boxShadow: '0 0 0 1px var(--chakra-colors-brand-400)' }}
                autoComplete="off"
              />
              <InputRightElement width="3rem">
                <IconButton
                  h="1.75rem"
                  size="sm"
                  aria-label="Send message"
                  icon={<Iconify icon={FiSend} boxSize={4} />}
                  colorScheme="brand"
                  onClick={() => sendTextMessage(chatInput)}
                  isDisabled={!chatInput.trim() || isTextSending || isProcessing || isSpeaking}
                  variant="ghost"
                  borderRadius="full"
                />
              </InputRightElement>
            </InputGroup>
            <Text fontSize="xs" color="gray.500" textAlign="center">
              {isSpeaking ? '🔊 AI is speaking... type now, send when ready' : '💬 Type a message to chat with the AI'}
            </Text>
          </Box>

          <Box borderTopWidth="1px" p={4}>
            <Flex gap={2} justify="center">
              {isSessionActive ? (
                <>
                  <Button
                    leftIcon={<Iconify icon={isRecording ? FiMicOff : FiMic} boxSize={5} />}
                    colorScheme={isRecording ? 'red' : 'green'}
                    onClick={isRecording ? () => stopRecording(true) : startRecording}
                    isDisabled={isProcessing || isTextSending || isSpeaking}
                    flex={1}
                  >
                    {isRecording ? 'Stop Recording' : 'Start Recording'}
                  </Button>
                  <IconButton
                    aria-label="End session"
                    icon={<Iconify icon={FiX} boxSize={5} />}
                    colorScheme="red"
                    variant="outline"
                    onClick={endSession}
                  />
                </>
              ) : (
                <Button
                  leftIcon={<Iconify icon={FiMic} boxSize={5} />}
                  colorScheme="blue"
                  onClick={startSession}
                  flex={1}
                >
                  Start Voice Session
                </Button>
              )}
            </Flex>

            <VStack spacing={1} mt={2}>
              <Text fontSize="xs" color="gray.500" textAlign="center">
                Powered by Whisper & Voice AI + English TTS
              </Text>
              <Text fontSize="xs" color="gray.500" textAlign="center">
                Voice: Microsoft Ravi • Hands-free
              </Text>
            </VStack>
          </Box>
        </>
      )}
    </VStack>
  );
};

export default VoiceAIChat;
