import React, { useEffect, useState, useCallback } from 'react';
import {
  Box,
  Button,
  Flex,
  Grid,
  HStack,
  Image,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalHeader,
  ModalOverlay,
  NumberDecrementStepper,
  NumberIncrementStepper,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  Progress,
  Slider,
  SliderFilledTrack,
  SliderThumb,
  SliderTrack,
  Tag,
  Text,
  useColorModeValue,
  useToast,
  VStack,
} from '@chakra-ui/react';
import { FiCamera, FiClock, FiPlay, FiSettings, FiTrash2 } from 'react-icons/fi';
import { Iconify } from '../common';
import { useCalibration } from '../../context/CalibrationContext';
import { useSocket } from '../../context/SocketContext';
import { API_BASE_URL } from '../../config';
import apiClient from '../../apiClient';

interface TestCapture {
  filename: string;
  url: string;
  size: number;
  timestamp: string;
}

interface CalibrationModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const CalibrationModal: React.FC<CalibrationModalProps> = ({ isOpen, onClose }) => {
  const {
    initialDelay,
    setInitialDelay,
    interCaptureDelay,
    setInterCaptureDelay,
    isCalibrated,
    startDelayCountdown,
    countdownValue,
    isCountingDown,
    cancelCountdown,
    resetCalibration,
  } = useCalibration();

  const { socket } = useSocket();
  const [tempDelay, setTempDelay] = useState<number>(initialDelay);
  const [tempInterCaptureDelay, setTempInterCaptureDelay] = useState<number>(interCaptureDelay);
  const [isTestingDelay, setIsTestingDelay] = useState(false);
  const [testCaptures, setTestCaptures] = useState<TestCapture[]>([]);
  const toast = useToast();

  // Sync tempDelay with initialDelay when it changes externally
  useEffect(() => {
    setTempDelay(initialDelay);
  }, [initialDelay]);

  // Sync tempInterCaptureDelay with interCaptureDelay when it changes externally
  useEffect(() => {
    setTempInterCaptureDelay(interCaptureDelay);
  }, [interCaptureDelay]);

  // Start test mode when modal opens - emit event to phone
  useEffect(() => {
    if (isOpen && socket) {
      // Notify phone that test mode started
      socket.emit('calibration_test_mode', { enabled: true });
    }
  }, [isOpen, socket]);

  // Listen for test captures from phone
  useEffect(() => {
    if (!socket) return;

    const handleTestCapture = (data: TestCapture) => {
      console.log('[Calibration] Test capture received:', data);
      setTestCaptures(prev => [data, ...prev]);
    };

    const handleTestCapturesCleared = () => {
      console.log('[Calibration] Test captures cleared');
      setTestCaptures([]);
    };

    socket.on('test_capture_received', handleTestCapture);
    socket.on('test_captures_cleared', handleTestCapturesCleared);

    return () => {
      socket.off('test_capture_received', handleTestCapture);
      socket.off('test_captures_cleared', handleTestCapturesCleared);
    };
  }, [socket]);

  // Load existing test captures on mount
  useEffect(() => {
    if (isOpen) {
      loadTestCaptures();
    }
  }, [isOpen]);

  const loadTestCaptures = async () => {
    try {
      const response = await apiClient.get('/test-captures');
      if (response.data.success) {
        setTestCaptures(response.data.captures || []);
      }
    } catch (error) {
      console.error('[Calibration] Failed to load test captures:', error);
    }
  };

  // Cleanup test captures when modal closes
  const handleClose = useCallback(async () => {
    // Stop any running test
    if (isCountingDown) {
      cancelCountdown();
    }
    setIsTestingDelay(false);

    // Notify phone that test mode ended
    if (socket) {
      socket.emit('calibration_test_mode', { enabled: false });
    }

    // Clear test captures from backend
    if (testCaptures.length > 0) {
      try {
        await apiClient.post('/test-captures/clear');
        console.log('[Calibration] Test captures cleared on close');
      } catch (error) {
        console.error('[Calibration] Failed to clear test captures:', error);
      }
    }

    setTestCaptures([]);
    onClose();
  }, [cancelCountdown, isCountingDown, onClose, socket, testCaptures.length]);

  const bgCard = useColorModeValue('rgba(255, 248, 240, 0.95)', 'rgba(12, 16, 35, 0.92)');
  const borderColor = useColorModeValue('rgba(121, 95, 238, 0.08)', 'rgba(255, 255, 255, 0.08)');
  const shadow = useColorModeValue(
    '0 20px 35px rgba(121, 95, 238, 0.08)',
    '0 25px 50px rgba(0, 0, 0, 0.8)'
  );
  const textMuted = useColorModeValue('gray.600', 'gray.400');

  return (
    <Modal isOpen={isOpen} onClose={handleClose} size="xl" isCentered scrollBehavior="inside">
      <ModalOverlay bg="blackAlpha.600" />
      <ModalContent
        bg={bgCard}
        borderRadius="2xl"
        border="1px solid"
        borderColor={borderColor}
        boxShadow={shadow}
        maxH="90vh"
      >
        <ModalHeader borderBottom="1px solid" borderColor={borderColor} py={4}>
          <HStack spacing={3}>
            <Iconify icon={FiSettings} boxSize={6} color="brand.400" />
            <Text fontWeight="700" fontSize="lg">
              Auto-Capture Settings
            </Text>
            {isCalibrated && (
              <Tag size="md" colorScheme="green" variant="subtle" borderRadius="full">
                Configured
              </Tag>
            )}
          </HStack>
          <ModalCloseButton top={3} right={3} />
        </ModalHeader>

        <ModalBody py={6} px={6}>
          <VStack align="stretch" spacing={6}>
            {/* Current Delay Display */}
            <Box
              bg={useColorModeValue('white', 'rgba(255,255,255,0.03)')}
              px={4}
              py={4}
              borderRadius="lg"
              border="1px solid"
              borderColor={borderColor}
            >
              <Flex align="center" justify="space-between" mb={3}>
                <HStack spacing={2}>
                  <Iconify icon={FiClock} boxSize={4} color="brand.400" />
                  <Text fontSize="sm" fontWeight="600">Auto-Capture Startup Delay</Text>
                </HStack>
                <Tag size="lg" colorScheme="brand" variant="solid" borderRadius="full">
                  {initialDelay}s
                </Tag>
              </Flex>
              <Text fontSize="xs" color={textMuted}>
                Wait time for your smartphone's auto-capture feature to fully turn on and initialize
                before starting the document scanning process.
              </Text>
            </Box>

            {/* Set Startup Delay */}
            <Box>
              <Text fontSize="sm" fontWeight="600" mb={3}>
                Set Startup Delay (seconds)
              </Text>
              <HStack spacing={4} mb={3}>
                <Slider
                  value={tempDelay}
                  onChange={(val) => setTempDelay(val)}
                  min={0}
                  max={30}
                  step={1}
                  flex={1}
                >
                  <SliderTrack bg={useColorModeValue('gray.200', 'whiteAlpha.300')}>
                    <SliderFilledTrack bg="brand.400" />
                  </SliderTrack>
                  <SliderThumb boxSize={5} />
                </Slider>
                <NumberInput
                  value={tempDelay}
                  onChange={(_, val) => setTempDelay(isNaN(val) ? 0 : val)}
                  min={0}
                  max={60}
                  w="80px"
                  size="sm"
                >
                  <NumberInputField />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
              </HStack>
              <Button
                size="sm"
                colorScheme="brand"
                w="full"
                onClick={() => {
                  setInitialDelay(tempDelay);
                  // Broadcast delay update to all connected devices
                  if (socket) {
                    socket.emit('delay_settings_updated', {
                      initialDelay: tempDelay,
                      interCaptureDelay: interCaptureDelay,
                      timestamp: Date.now()
                    });
                  }
                  toast({
                    title: 'Delay Updated',
                    description: `Auto-capture startup delay set to ${tempDelay} seconds`,
                    status: 'success',
                    duration: 3000,
                    isClosable: true,
                  });
                }}
                isDisabled={tempDelay === initialDelay}
              >
                Save Delay
              </Button>

              {/* Inter-Capture Delay */}
              <Text fontSize="sm" fontWeight="600" mt={4} mb={2}>
                Inter-Capture Delay (seconds)
              </Text>
              <Text fontSize="xs" color="gray.500" mb={2}>
                Wait time between consecutive document captures
              </Text>
              <HStack spacing={4} mb={3}>
                <Slider
                  value={tempInterCaptureDelay}
                  onChange={(val) => setTempInterCaptureDelay(val)}
                  min={0}
                  max={15}
                  step={0.5}
                  flex={1}
                >
                  <SliderTrack bg={useColorModeValue('gray.200', 'whiteAlpha.300')}>
                    <SliderFilledTrack bg="orange.400" />
                  </SliderTrack>
                  <SliderThumb boxSize={5} />
                </Slider>
                <NumberInput
                  value={tempInterCaptureDelay}
                  onChange={(_, val) => setTempInterCaptureDelay(isNaN(val) ? 0 : val)}
                  min={0}
                  max={30}
                  step={0.5}
                  w="80px"
                  size="sm"
                >
                  <NumberInputField />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
              </HStack>
              <Button
                size="sm"
                colorScheme="orange"
                w="full"
                onClick={() => {
                  setInterCaptureDelay(tempInterCaptureDelay);
                  // Broadcast delay update to all connected devices
                  if (socket) {
                    socket.emit('delay_settings_updated', {
                      initialDelay: initialDelay,
                      interCaptureDelay: tempInterCaptureDelay,
                      timestamp: Date.now()
                    });
                  }
                  toast({
                    title: 'Inter-Capture Delay Updated',
                    description: `${tempInterCaptureDelay}s delay between captures`,
                    status: 'success',
                    duration: 3000,
                    isClosable: true,
                  });
                }}
                isDisabled={tempInterCaptureDelay === interCaptureDelay}
              >
                Save Inter-Capture Delay
              </Button>
            </Box>

            {/* Test Delay */}
            <Box
              bg={useColorModeValue('gray.50', 'rgba(255,255,255,0.03)')}
              px={4}
              py={4}
              borderRadius="lg"
              border="1px solid"
              borderColor={borderColor}
            >
              <Text fontSize="sm" fontWeight="600" mb={2}>
                Test Startup Delay
              </Text>
              <Text fontSize="xs" color={textMuted} mb={3}>
                Run a test countdown. Images captured during the test will appear below to verify timing.
              </Text>
              {isCountingDown ? (
                <VStack spacing={3}>
                  <Box
                    w="100%"
                    bg={useColorModeValue('brand.50', 'rgba(121, 95, 238, 0.1)')}
                    borderRadius="lg"
                    p={4}
                    textAlign="center"
                  >
                    <Text fontSize="4xl" fontWeight="bold" color="brand.400">
                      {countdownValue}
                    </Text>
                    <Text fontSize="sm" color={textMuted}>seconds remaining</Text>
                  </Box>
                  <Progress
                    value={((initialDelay - (countdownValue || 0)) / initialDelay) * 100}
                    size="sm"
                    colorScheme="brand"
                    borderRadius="full"
                    w="100%"
                  />
                  <Button
                    size="sm"
                    colorScheme="red"
                    variant="outline"
                    onClick={() => {
                      cancelCountdown();
                      setIsTestingDelay(false);
                    }}
                  >
                    Cancel Test
                  </Button>
                </VStack>
              ) : (
                <Button
                  size="sm"
                  colorScheme="green"
                  variant="outline"
                  w="full"
                  leftIcon={<Iconify icon={FiPlay} boxSize={4} />}
                  onClick={async () => {
                    setIsTestingDelay(true);
                    // Notify phone to start test capture countdown and auto-capture
                    if (socket) {
                      socket.emit('start_test_capture', { 
                        delay: initialDelay,
                        autoCapture: true  // Tell phone to auto-capture after countdown
                      });
                    }
                    try {
                      await startDelayCountdown();
                      toast({
                        title: 'Startup Delay Complete!',
                        description: 'Phone should capture now. Check captured images below.',
                        status: 'success',
                        duration: 3000,
                        isClosable: true,
                      });
                    } catch (e) {
                      // Cancelled - notify phone to stop
                      if (socket) {
                        socket.emit('cancel_test_capture');
                      }
                    }
                    setIsTestingDelay(false);
                  }}
                  isLoading={isTestingDelay && !isCountingDown}
                >
                  Test {initialDelay}s Delay
                </Button>
              )}
            </Box>

            {/* Test Captures Preview */}
            {testCaptures.length > 0 && (
              <Box
                bg={useColorModeValue('green.50', 'rgba(34, 197, 94, 0.1)')}
                px={4}
                py={4}
                borderRadius="lg"
                border="1px solid"
                borderColor={useColorModeValue('green.200', 'green.700')}
              >
                <Flex align="center" justify="space-between" mb={3}>
                  <HStack spacing={2}>
                    <Iconify icon={FiCamera} boxSize={4} color="green.500" />
                    <Text fontSize="sm" fontWeight="600" color="green.600">
                      Test Captures ({testCaptures.length})
                    </Text>
                  </HStack>
                  <Button
                    size="xs"
                    colorScheme="red"
                    variant="ghost"
                    leftIcon={<Iconify icon={FiTrash2} boxSize={3} />}
                    onClick={async () => {
                      try {
                        await apiClient.post('/test-captures/clear');
                        setTestCaptures([]);
                        toast({
                          title: 'Cleared',
                          description: 'Test captures deleted',
                          status: 'info',
                          duration: 2000,
                        });
                      } catch (error) {
                        console.error('Failed to clear test captures:', error);
                      }
                    }}
                  >
                    Clear All
                  </Button>
                </Flex>
                <Text fontSize="xs" color={textMuted} mb={3}>
                  These images were captured during testing and will be automatically deleted when you close this screen.
                </Text>
                <Grid templateColumns="repeat(3, 1fr)" gap={2}>
                  {testCaptures.slice(0, 9).map((capture) => (
                    <Box
                      key={capture.filename}
                      borderRadius="md"
                      overflow="hidden"
                      border="1px solid"
                      borderColor={borderColor}
                      position="relative"
                    >
                      <Image
                        src={`${API_BASE_URL}${capture.url}`}
                        alt={capture.filename}
                        w="100%"
                        h="80px"
                        objectFit="cover"
                        fallback={
                          <Flex w="100%" h="80px" bg="gray.200" align="center" justify="center">
                            <Text fontSize="xs" color="gray.500">Loading...</Text>
                          </Flex>
                        }
                      />
                      <Box
                        position="absolute"
                        bottom={0}
                        left={0}
                        right={0}
                        bg="blackAlpha.600"
                        px={1}
                        py={0.5}
                      >
                        <Text fontSize="8px" color="white" noOfLines={1}>
                          {new Date(capture.timestamp).toLocaleTimeString()}
                        </Text>
                      </Box>
                    </Box>
                  ))}
                </Grid>
                {testCaptures.length > 9 && (
                  <Text fontSize="xs" color={textMuted} mt={2} textAlign="center">
                    +{testCaptures.length - 9} more capture(s)
                  </Text>
                )}
              </Box>
            )}

            {/* Footer Actions */}
            <HStack spacing={2} justify="space-between" pt={2}>
              <Button
                size="sm"
                variant="ghost"
                colorScheme="red"
                onClick={() => {
                  resetCalibration();
                  setTempDelay(10);
                  toast({
                    title: 'Settings Reset',
                    description: 'Startup delay reset to default (10 seconds)',
                    status: 'info',
                    duration: 2000,
                  });
                }}
              >
                Reset to Default
              </Button>
              <Button onClick={handleClose}>
                Done
              </Button>
            </HStack>
          </VStack>
        </ModalBody>
      </ModalContent>
    </Modal>
  );
};

export default CalibrationModal;
