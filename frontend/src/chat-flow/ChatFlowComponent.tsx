/**
 * Chat Flow Component - Main UI for voice workflow
 */

import React, { useEffect, useRef, useCallback, useState } from 'react';
import {
  Box,
  VStack,
  HStack,
  Text,
  Button,
  IconButton,
  Card,
  CardBody,
  Badge,
  Flex,
  Spacer,
  useColorModeValue,
  Spinner,
  Tooltip,
  Collapse,
  Divider,
} from '@chakra-ui/react';
import { MdMic, MdMicOff, MdClose, MdVolumeUp, MdRefresh, MdSettings, MdWork, MdChat } from 'react-icons/md';
import { useChatFlow } from './ChatFlowContext';
import { ChatFlowStep } from './chatFlowTypes';

// Wrapper to fix React 18 type compatibility with react-icons
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const Icon: React.FC<{ as: any; size?: number | string }> = ({ as: IconComponent, size }) => {
  return React.createElement(IconComponent, { size });
};

const STEP_LABELS: Record<ChatFlowStep, string> = {
  [ChatFlowStep.IDLE]: 'Ready',
  [ChatFlowStep.AWAITING_MODE]: 'Awaiting Mode',
  [ChatFlowStep.AWAITING_CONFIRM]: 'Confirming',
  [ChatFlowStep.SELECTING_DOCS]: 'Selecting Documents',
  [ChatFlowStep.CONFIGURING]: 'Configuring',
  [ChatFlowStep.REVIEWING]: 'Reviewing',
  [ChatFlowStep.EXECUTING]: 'Processing',
  [ChatFlowStep.COMPLETED]: 'Completed',
};

const STEP_COLORS: Record<ChatFlowStep, string> = {
  [ChatFlowStep.IDLE]: 'gray',
  [ChatFlowStep.AWAITING_MODE]: 'teal',
  [ChatFlowStep.AWAITING_CONFIRM]: 'blue',
  [ChatFlowStep.SELECTING_DOCS]: 'purple',
  [ChatFlowStep.CONFIGURING]: 'orange',
  [ChatFlowStep.REVIEWING]: 'cyan',
  [ChatFlowStep.EXECUTING]: 'green',
  [ChatFlowStep.COMPLETED]: 'green',
};

interface ChatFlowComponentProps {
  isOpen: boolean;
  onClose: () => void;
  onOpenDocumentSelector: () => void;
  onOpenConfigPanel?: () => void;
  onExecuteJob: (config: any) => void;
  /** Default input mode - user can switch between them */
  defaultMode?: 'text' | 'voice';
  /** Title to display */
  title?: string;
}

export function ChatFlowComponent({
  isOpen,
  onClose,
  onOpenDocumentSelector,
  onOpenConfigPanel,
  onExecuteJob,
  defaultMode = 'text',
  title = 'AI Assistant',
}: ChatFlowComponentProps) {
  const {
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
    setIsRecording,
  } = useChatFlow();

  // Input mode state - text or voice
  const [inputMode, setInputMode] = useState<'text' | 'voice'>(defaultMode);
  // Auto-speak AI responses in voice mode
  const [autoSpeak, setAutoSpeak] = useState(defaultMode === 'voice');

  // Styling
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  const userMsgBg = useColorModeValue('blue.50', 'blue.900');
  const aiMsgBg = useColorModeValue('gray.50', 'gray.700');
  const systemMsgBg = useColorModeValue('yellow.50', 'yellow.900');

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  // Local state
  const [inputText, setInputText] = useState('');

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Initialize session when opened
  useEffect(() => {
    if (isOpen && !isSessionActive) {
      startSession();
    }
  }, [isOpen, isSessionActive, startSession]);

  // Handle close
  const handleClose = useCallback(async () => {
    if (isSessionActive) {
      await endSession();
    }
    onClose();
  }, [isSessionActive, endSession, onClose]);

  // Start recording
  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
      });

      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        
        // Process audio through chat flow context
        await processAudio(audioBlob);

        // Clean up stream
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Error starting recording:', error);
    }
  }, [setIsRecording, processAudio]);

  // Stop recording
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [isRecording, setIsRecording]);

  // Handle text input
  const handleSubmitText = useCallback(async () => {
    if (!inputText.trim()) return;
    const response = await processText(inputText.trim());
    setInputText('');
    
    // Auto-speak response in voice mode
    if (autoSpeak && response?.ai_response) {
      const textToSpeak = response.tts_response || response.ai_response;
      await speak(textToSpeak);
    }
  }, [inputText, processText, autoSpeak, speak]);

  // Render message
  const renderMessage = (msg: { id: string; type: string; text: string; timestamp: Date }) => {
    let bg = aiMsgBg;
    let align: 'flex-start' | 'flex-end' | 'center' = 'flex-start';
    let label = 'AI';

    if (msg.type === 'user') {
      bg = userMsgBg;
      align = 'flex-end';
      label = 'You';
    } else if (msg.type === 'system') {
      bg = systemMsgBg;
      align = 'center';
      label = 'System';
    }

    return (
      <Flex key={msg.id} justify={align} w="100%">
        <Box
          maxW={msg.type === 'system' ? '100%' : '80%'}
          bg={bg}
          p={3}
          borderRadius="lg"
          boxShadow="sm"
        >
          <Text fontSize="xs" color="gray.500" mb={1}>
            {label}
          </Text>
          <Text fontSize="sm">{msg.text}</Text>
        </Box>
      </Flex>
    );
  };

  if (!isOpen) return null;

  return (
    <Card
      position="fixed"
      bottom="80px"
      right="20px"
      w="400px"
      maxH="600px"
      bg={bgColor}
      borderColor={borderColor}
      borderWidth={1}
      boxShadow="xl"
      zIndex={1000}
    >
      <CardBody p={0}>
        {/* Header */}
        <Flex
          p={3}
          borderBottomWidth={1}
          borderColor={borderColor}
          alignItems="center"
        >
          <HStack spacing={2}>
            {interactionMode === 'talk' ? <Icon as={MdChat} size={20} /> : <Icon as={MdWork} size={20} />}
            <Text fontWeight="bold">{title}</Text>
            {/* Mode Badge */}
            <Badge 
              colorScheme={interactionMode === 'talk' ? 'purple' : 'blue'} 
              fontSize="xs"
              variant="solid"
            >
              {interactionMode === 'talk' ? '💬 Talk' : '🔧 Work'}
            </Badge>
          </HStack>
          <Spacer />
          <HStack spacing={2}>
            {/* Input mode toggle */}
            <Tooltip label={inputMode === 'text' ? 'Switch to Voice' : 'Switch to Text'}>
              <IconButton
                aria-label="Toggle mode"
                icon={inputMode === 'text' ? <Icon as={MdMic} /> : <Icon as={MdRefresh} />}
                size="sm"
                variant="ghost"
                onClick={() => {
                  const newMode = inputMode === 'text' ? 'voice' : 'text';
                  setInputMode(newMode);
                  setAutoSpeak(newMode === 'voice');
                }}
              />
            </Tooltip>
            {interactionMode === 'work' && (
              <Badge colorScheme={STEP_COLORS[currentStep]} fontSize="xs">
                {STEP_LABELS[currentStep]}
              </Badge>
            )}
            <IconButton
              aria-label="Close"
              icon={<Icon as={MdClose} />}
              size="sm"
              variant="ghost"
              onClick={handleClose}
            />
          </HStack>
        </Flex>

        {/* Config Summary */}
        <Collapse in={currentStep !== ChatFlowStep.IDLE}>
          <Box p={2} bg={useColorModeValue('gray.50', 'gray.700')} fontSize="xs">
            <HStack justify="space-between">
              <Text>
                Mode: <strong>{config.mode || 'N/A'}</strong>
              </Text>
              <Text>
                Docs: <strong>{config.selectedDocuments.length}</strong>
              </Text>
              <Text>
                Copies: <strong>{config.copies}</strong>
              </Text>
              {config.mode === 'print' && (
                <Text>
                  Color: <strong>{config.colorMode}</strong>
                </Text>
              )}
            </HStack>
          </Box>
        </Collapse>

        {/* Messages */}
        <VStack
          p={3}
          spacing={3}
          align="stretch"
          overflowY="auto"
          maxH="350px"
          minH="200px"
        >
          {messages.map(renderMessage)}
          {isProcessing && (
            <Flex justify="center">
              <Spinner size="sm" />
            </Flex>
          )}
          <div ref={messagesEndRef} />
        </VStack>

        <Divider />

        {/* Input Area */}
        <Box p={3}>
          <HStack spacing={2}>
            {/* Text input */}
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSubmitText()}
              placeholder="Type or speak..."
              style={{
                flex: 1,
                padding: '8px 12px',
                borderRadius: '8px',
                border: `1px solid ${borderColor}`,
                background: bgColor,
                outline: 'none',
              }}
              disabled={isProcessing || isSpeaking}
            />

            {/* Record button */}
            <Tooltip label={isRecording ? 'Stop recording' : 'Start recording'}>
              <IconButton
                aria-label={isRecording ? 'Stop' : 'Record'}
                icon={isRecording ? <Icon as={MdMicOff} /> : <Icon as={MdMic} />}
                colorScheme={isRecording ? 'red' : 'blue'}
                isRound
                onClick={isRecording ? stopRecording : startRecording}
                isDisabled={isProcessing || isSpeaking}
              />
            </Tooltip>

            {/* Submit button */}
            <Button
              colorScheme="blue"
              size="md"
              onClick={handleSubmitText}
              isDisabled={!inputText.trim() || isProcessing || isSpeaking}
            >
              Send
            </Button>
          </HStack>

          {/* Status indicators */}
          <HStack mt={2} justify="space-between" fontSize="xs" color="gray.500">
            <Text>
              Mode: {inputMode === 'voice' ? '🎤 Voice' : '⌨️ Text'}
              {autoSpeak && ' (auto-speak on)'}
            </Text>
            {isSpeaking && (
              <HStack color="blue.500">
                <Icon as={MdVolumeUp} />
                <Text>Speaking...</Text>
              </HStack>
            )}
          </HStack>
        </Box>

        {/* Quick Actions */}
        {currentStep === ChatFlowStep.SELECTING_DOCS && (
          <Box px={3} pb={3}>
            <Button
              size="sm"
              variant="outline"
              colorScheme="purple"
              leftIcon={<Icon as={MdSettings} />}
              onClick={onOpenDocumentSelector}
              w="100%"
            >
              Open Document Selector
            </Button>
          </Box>
        )}

        {currentStep === ChatFlowStep.CONFIGURING && onOpenConfigPanel && (
          <Box px={3} pb={3}>
            <Button
              size="sm"
              variant="outline"
              colorScheme="orange"
              leftIcon={<Icon as={MdSettings} />}
              onClick={onOpenConfigPanel}
              w="100%"
            >
              Open Configuration Panel
            </Button>
          </Box>
        )}
      </CardBody>
    </Card>
  );
}

export default ChatFlowComponent;
