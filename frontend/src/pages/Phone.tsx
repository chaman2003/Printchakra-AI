import React, { useCallback, useEffect, useRef, useState } from 'react';
import apiClient from '../apiClient';
import { useSocket } from '../context/SocketContext';
import {
  Alert,
  AlertDescription,
  AlertIcon,
  Box,
  Button,
  ButtonGroup,
  Card,
  CardBody,
  Flex,
  Heading,
  Spinner,
  Stack,
  Text,
  Tooltip,
  useColorModeValue,
  useToast,
  VStack,
} from '@chakra-ui/react';
import { FiCamera, FiEye, FiEyeOff, FiMaximize2, FiMinimize2, FiUpload, FiWifi } from 'react-icons/fi';
import { API_ENDPOINTS } from '../config';
import { shrinkForUpload } from '../utils/imageUpload';
import { Iconify, ConnectionValidator } from '../components/common';

const Phone: React.FC = () => {
  const [connected, setConnected] = useState(false);
  const [captureMode, setCaptureMode] = useState<'file' | 'camera'>('file');
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [message, setMessage] = useState<string>('');
  const [showControls, setShowControls] = useState(true);
  const [showConnectionValidator, setShowConnectionValidator] = useState(false);
  const [isFullScreen, setIsFullScreen] = useState(false);
  const [isProcessingQueue, setIsProcessingQueue] = useState(false);
  const [uploadQueue, setUploadQueue] = useState<Array<{ id: string; blob: Blob; filename: string }>>([]);
  const [stream, setStream] = useState<MediaStream | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cameraContainerRef = useRef<HTMLDivElement>(null);

  const toast = useToast();
  const { socket, connected: socketConnected } = useSocket();

  const panelBg = useColorModeValue('whiteAlpha.900', 'rgba(12, 16, 35, 0.95)');
  const muted = useColorModeValue('gray.600', 'whiteAlpha.700');

  const showMessage = useCallback((msg: string) => {
    setMessage(msg);
    setTimeout(() => setMessage(''), 5000);
  }, []);

  const toggleFullScreen = useCallback(() => {
    if (!cameraContainerRef.current) return;
    if (!document.fullscreenElement) {
      cameraContainerRef.current.requestFullscreen().then(() => setIsFullScreen(true)).catch(() => {});
    } else {
      document.exitFullscreen().then(() => setIsFullScreen(false)).catch(() => {});
    }
  }, []);

  useEffect(() => {
    const handler = () => setIsFullScreen(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', handler);
    return () => document.removeEventListener('fullscreenchange', handler);
  }, []);

  const startCamera = useCallback(async () => {
    try {
      if (stream) {
        stream.getTracks().forEach((track: MediaStreamTrack) => track.stop());
      }
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play().catch((e) => console.error('Video play failed:', e));
        };
      }
    } catch (err) {
      console.error('Camera start error:', err);
      toast({ title: 'Camera Error', description: (err as Error).message, status: 'error', duration: 3000 });
    }
  }, [stream, toast]);

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach((track: MediaStreamTrack) => track.stop());
      setStream(null);
    }
  }, [stream]);

  const handleCaptureMode = useCallback((mode: 'file' | 'camera') => {
    setCaptureMode(mode);
    setPreviewImage(null);
    if (mode === 'camera') {
      startCamera();
    } else {
      stopCamera();
    }
  }, [startCamera, stopCamera]);

  const uploadImage = useCallback((file: Blob, filename: string) => {
    setUploadQueue((prev) => [...prev, { id: Date.now().toString() + Math.random(), blob: file, filename }]);
    showMessage('Added to processing queue');
  }, [showMessage]);

  const captureFromCamera = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context || !video.videoWidth || !video.videoHeight) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);

    canvas.toBlob(
      (blob: Blob | null) => {
        if (blob) {
          uploadImage(blob, `capture_${Date.now()}.jpg`);
        }
      },
      'image/jpeg',
      0.9
    );
  }, [uploadImage]);

  const handleFileSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      toast({ title: 'Invalid file', description: 'Please select an image file', status: 'warning', duration: 2500 });
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewImage(url);
    uploadImage(file, file.name);
  }, [toast, uploadImage]);

  useEffect(() => {
    const processQueue = async () => {
      if (isProcessingQueue || uploadQueue.length === 0) return;

      setIsProcessingQueue(true);
      const item = uploadQueue[0];
      try {
        const { blob, filename } = await shrinkForUpload(item.blob, item.filename);
        const formData = new FormData();
        formData.append('file', blob, filename);
        await apiClient.post(API_ENDPOINTS.upload, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        toast({
          title: 'Upload complete',
          description: `${item.filename} sent`,
          status: 'success',
          duration: 2000,
          position: 'top-right',
        });
      } catch (err) {
        console.error('Queue upload failed:', err);
        toast({ title: 'Upload failed', status: 'error', duration: 3000, position: 'top-right' });
      } finally {
        setUploadQueue((prev) => prev.slice(1));
        setIsProcessingQueue(false);
      }
    };

    processQueue();
  }, [uploadQueue, isProcessingQueue, toast]);

  useEffect(() => {
    setConnected(socketConnected);
  }, [socketConnected]);

  useEffect(() => {
    if (!socket) return;

    socket.on('capture_now', () => {
      showMessage('Capture triggered from Dashboard');
      setTimeout(() => {
        if (captureMode === 'camera' && stream) {
          captureFromCamera();
        } else {
          showMessage('Switch to Camera mode to capture');
        }
      }, 400);
    });

    return () => {
      socket.off('capture_now');
    };
  }, [socket, captureMode, stream, captureFromCamera, showMessage]);

  useEffect(() => {
    return () => {
      if (stream) stream.getTracks().forEach((track) => track.stop());
    };
  }, [stream]);

  return (
    <VStack align="stretch" spacing={{ base: 8, md: 10 }} pb={16}>
      <Flex direction={{ base: 'column', md: 'row' }} justify="space-between" gap={6}>
        <Stack spacing={2}>
          <Heading size="lg" display="flex" alignItems="center" gap={3}>
            Phone Capture
          </Heading>
          <Text color={muted} maxW="lg">
            Capture documents and send them to the Dashboard for processing.
          </Text>
        </Stack>
        <Flex align="center" gap={3} bg="surface.blur" borderRadius="full" border="1px solid rgba(17, 17, 17,0.22)" px={5} py={2}>
          <Box w={3} h={3} borderRadius="full" bg={connected ? 'green.400' : 'red.400'} />
          <Text fontWeight="600" color={muted} display="flex" alignItems="center" gap={2}>
            <Iconify icon={FiWifi} boxSize={5} />
            {connected ? 'Connected' : 'Offline'}
          </Text>
        </Flex>

        {(uploadQueue.length > 0 || isProcessingQueue) && (
          <Flex align="center" gap={2} bg="blue.500" color="white" borderRadius="full" px={4} py={2}>
            <Spinner size="xs" />
            <Text fontWeight="bold" fontSize="sm">Queue: {uploadQueue.length + (isProcessingQueue ? 1 : 0)}</Text>
          </Flex>
        )}
      </Flex>

      {message && (
        <Alert status="info" borderRadius="xl" bg="rgba(17, 17, 17,0.1)" border="1px solid rgba(17, 17, 17,0.25)">
          <AlertIcon />
          <AlertDescription>{message}</AlertDescription>
        </Alert>
      )}

      <Card bg={panelBg} border="1px solid rgba(17, 17, 17,0.18)" boxShadow="subtle">
        <CardBody>
          <Stack spacing={6}>
            <ButtonGroup isAttached variant="ghost" alignSelf="center">
              <Button leftIcon={<Iconify icon={FiUpload} boxSize={5} />} colorScheme={captureMode === 'file' ? 'brand' : undefined} onClick={() => handleCaptureMode('file')}>
                Choose File
              </Button>
              <Button leftIcon={<Iconify icon={FiCamera} boxSize={5} />} colorScheme={captureMode === 'camera' ? 'brand' : undefined} onClick={() => handleCaptureMode('camera')}>
                Live Camera
              </Button>
            </ButtonGroup>

            {captureMode === 'file' ? (
              <Stack spacing={4} align="center">
                <input type="file" ref={fileInputRef} onChange={handleFileSelect} accept="image/*,image/jpeg,image/jpg,image/png" capture="environment" style={{ display: 'none' }} />
                <Button size="lg" colorScheme="brand" leftIcon={<Iconify icon={FiUpload} boxSize={5} />} onClick={() => fileInputRef.current?.click()}>
                  Select Image
                </Button>
                <Text fontSize="sm" color={muted} textAlign="center">
                  Upload a document photo from your device.
                </Text>
              </Stack>
            ) : (
              <Stack spacing={4}>
                <Box
                  ref={cameraContainerRef}
                  position="relative"
                  borderRadius="xl"
                  overflow="hidden"
                  bg="black"
                  border="2px solid"
                  borderColor="gray.600"
                >
                  <video ref={videoRef} autoPlay playsInline muted style={{ width: '100%', height: isFullScreen ? '100vh' : 'auto', maxHeight: isFullScreen ? '100vh' : '70vh', display: 'block', backgroundColor: 'black', objectFit: 'contain' }} />
                  <canvas ref={canvasRef} style={{ display: 'none' }} />

                  <Tooltip label={showControls ? 'Hide controls' : 'Show controls'} hasArrow placement="left">
                    <Button position="absolute" top={3} right={3} size="sm" colorScheme="brand" variant={showControls ? 'solid' : 'ghost'} onClick={() => setShowControls(!showControls)} zIndex={10} borderRadius="full" p={2} minWidth="auto">
                      <Iconify icon={showControls ? FiEye : FiEyeOff} boxSize={4} />
                    </Button>
                  </Tooltip>
                </Box>

                {showControls && (
                  <Flex wrap="wrap" gap={2} justify="center">
                    <Button colorScheme="brand" leftIcon={<Iconify icon={FiCamera} boxSize={5} />} onClick={captureFromCamera} isDisabled={!stream || isProcessingQueue}>
                      Capture
                    </Button>
                    <Button variant="outline" colorScheme="brand" onClick={toggleFullScreen} isDisabled={!stream} leftIcon={<Iconify icon={isFullScreen ? FiMinimize2 : FiMaximize2} boxSize={5} />}>
                      Fullscreen
                    </Button>
                  </Flex>
                )}
              </Stack>
            )}

            {previewImage && (
              <Stack spacing={3}>
                <Heading size="sm">Preview</Heading>
                <Box borderRadius="2xl" overflow="hidden" border="1px solid rgba(17, 17, 17,0.25)" boxShadow="subtle">
                  <img src={previewImage} alt="Preview" style={{ width: '100%', display: 'block' }} />
                </Box>
              </Stack>
            )}
          </Stack>
        </CardBody>
      </Card>

      <ConnectionValidator
        isOpen={showConnectionValidator}
        onClose={() => setShowConnectionValidator(false)}
        videoRef={videoRef as React.RefObject<HTMLVideoElement>}
      />
    </VStack>
  );
};

export default Phone;
