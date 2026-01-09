import React, { useCallback, useEffect, useRef, useState } from 'react';
import apiClient from '../apiClient';
import { useSocket } from '../context/SocketContext';
import { useCalibration } from '../context/CalibrationContext';
import {
  Alert,
  AlertDescription,
  AlertIcon,
  Badge,
  Box,
  Button,
  ButtonGroup,
  Card,
  CardBody,
  CardHeader,
  Flex,
  Grid,
  Heading,
  Stack,
  Spinner,
  Switch,
  Tag,
  Text,
  Tooltip,
  useColorModeValue,
  useToast,
  VStack,
} from '@chakra-ui/react';
import {
  FiAlertTriangle,
  FiAperture,
  FiCamera,
  FiCheckCircle,
  FiCpu,
  FiEye,
  FiEyeOff,
  FiMaximize2,
  FiMinimize2,
  FiUpload,
  FiWifi,
} from 'react-icons/fi';
import {
  API_BASE_URL,
  API_ENDPOINTS,
  SOCKET_CONFIG,
  SOCKET_IO_ENABLED,
  getDefaultHeaders,
} from '../config';
import { Iconify, ConnectionValidator } from '../components/common';

interface QualityCheck {
  blur_score: number;
  is_blurry: boolean;
  focus_score: number;
  is_focused: boolean;
  quality: {
    overall_acceptable: boolean;
    issues: string[];
    recommendations: string[];
  };
}

type DetectedQuad = {
  topLeft: { x: number; y: number };
  topRight: { x: number; y: number };
  bottomRight: { x: number; y: number };
  bottomLeft: { x: number; y: number };
};

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

const Phone: React.FC = () => {
  const [connected, setConnected] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [captureMode, setCaptureMode] = useState<'file' | 'camera'>('file');
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [message, setMessage] = useState<string>('');
  const [qualityCheck, setQualityCheck] = useState<QualityCheck | null>(null);
  const [validateQuality, setValidateQuality] = useState(true);
  const [processingOptions, setProcessingOptions] = useState({
    autoCrop: true,
    aiEnhance: true,
    strictQuality: true,
  });
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isFullScreen, setIsFullScreen] = useState(false);
  const cameraContainerRef = useRef<HTMLDivElement>(null);
  const [autoCapture, setAutoCapture] = useState(false);
  const [autoCaptureCount, setAutoCaptureCount] = useState(0);
  const autoCaptureIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const canvasOverlayRef = useRef<HTMLCanvasElement>(null);
  const docDetectionCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const detectionRafRef = useRef<number | null>(null);
  const [showControls, setShowControls] = useState(true);
  const [showConnectionValidator, setShowConnectionValidator] = useState(false);
  const [frameChangeStatus, setFrameChangeStatus] = useState<'waiting' | 'detecting' | 'ready' | 'captured'>('waiting');
  const [cameraOrientation, setCameraOrientation] = useState<'portrait' | 'landscape'>('portrait');
  const [detectedQuad, setDetectedQuad] = useState<DetectedQuad | null>(null);

  // Countdown state for auto-capture from dashboard
  const [countdown, setCountdown] = useState<number | null>(null);
  const countdownIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Sync state - tracks if auto-capture was triggered remotely (from dashboard)
  const [autoCaptureSource, setAutoCaptureSource] = useState<'local' | 'dashboard' | null>(null);
  const [pendingDocumentCount, setPendingDocumentCount] = useState<number>(0);

  // Auto-capture delays from calibration
  const { initialDelay, interCaptureDelay, startDelayCountdown, countdownValue, isCountingDown, cancelCountdown, setInitialDelay, setInterCaptureDelay } = useCalibration();
  const [isWaitingForInitialDelay, setIsWaitingForInitialDelay] = useState(false);
  const hasAppliedInitialDelayRef = useRef(false);

  // Track last capture time for inter-capture delay
  const lastCaptureTimeRef = useRef<number>(0);
  // Ref to access current interCaptureDelay value in callbacks without stale closure
  const interCaptureDelayRef = useRef<number>(interCaptureDelay);

  // Keep ref in sync with current value
  useEffect(() => {
    interCaptureDelayRef.current = interCaptureDelay;
  }, [interCaptureDelay]);

  // Test capture mode (for calibration testing from Dashboard)
  const [isTestCaptureMode, setIsTestCaptureMode] = useState(false);

  // Upload Queue State
  const [uploadQueue, setUploadQueue] = useState<Array<{
    id: string;
    blob: Blob;
    filename: string;
    options: typeof processingOptions;
    isTestCapture?: boolean;  // Flag to route to test-capture endpoint
  }>>([]);
  const [isProcessingQueue, setIsProcessingQueue] = useState(false);

  const toast = useToast();

  // Use shared Socket from context instead of creating new connections
  const { socket, connected: socketConnected } = useSocket();

  // Frame comparison based auto-capture (no API calls for detection)
  const lastCapturedImageDataRef = useRef<ImageData | null>(null);
  const lastFrameImageDataRef = useRef<ImageData | null>(null);
  const isCapturingRef = useRef<boolean>(false);
  const stableFrameCountRef = useRef<number>(0);
  const comparisonCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const captureCooldownRef = useRef<boolean>(false);

  // Theme values with insane visual enhancements
  const panelBg = useColorModeValue('whiteAlpha.900', 'rgba(12, 16, 35, 0.95)');
  const muted = useColorModeValue('gray.600', 'whiteAlpha.700');

  const showMessage = useCallback((msg: string) => {
    setMessage(msg);
    setTimeout(() => setMessage(''), 5000);
  }, []);

  const toggleFullScreen = async () => {
    if (!isFullScreen) {
      try {
        if (cameraContainerRef.current?.requestFullscreen) {
          await cameraContainerRef.current.requestFullscreen();
          setIsFullScreen(true);
          // Show connection validator after entering fullscreen
          setTimeout(() => {
            setShowConnectionValidator(true);
          }, 500);
        }
      } catch (err) {
        console.error('Failed to enter fullscreen:', err);
      }
    } else {
      if (document.fullscreenElement) {
        document.exitFullscreen();
      }
      setIsFullScreen(false);
    }
  };

  // Listen for fullscreen change events
  useEffect(() => {
    const handleFullscreenChange = () => {
      if (!document.fullscreenElement) {
        setIsFullScreen(false);
      }
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  // Initialize comparison canvas on mount
  useEffect(() => {
    comparisonCanvasRef.current = document.createElement('canvas');
    return () => {
      comparisonCanvasRef.current = null;
    };
  }, []);

  // Compare two ImageData objects and return difference percentage (0-100)
  const compareFrames = (frame1: ImageData, frame2: ImageData): number => {
    if (frame1.width !== frame2.width || frame1.height !== frame2.height) {
      console.warn(`Frame size mismatch: ${frame1.width}x${frame1.height} vs ${frame2.width}x${frame2.height}`);
      return 100; // Different sizes = completely different
    }

    if (frame1.data.length === 0 || frame2.data.length === 0) {
      console.warn('Empty frame data detected');
      return 100; // Empty = treat as different
    }

    const data1 = frame1.data;
    const data2 = frame2.data;
    let diffCount = 0;
    const pixelCount = data1.length / 4;

    // Sample every 4th pixel for performance (still accurate enough)
    for (let i = 0; i < data1.length; i += 16) {
      const r1 = data1[i], g1 = data1[i + 1], b1 = data1[i + 2];
      const r2 = data2[i], g2 = data2[i + 1], b2 = data2[i + 2];

      // Calculate color distance
      const diff = Math.abs(r1 - r2) + Math.abs(g1 - g2) + Math.abs(b1 - b2);

      // Threshold for considering a pixel as "different"
      if (diff > 60) {
        diffCount++;
      }
    }

    // Return percentage of different pixels
    return (diffCount / (pixelCount / 4)) * 100;
  };

  // Check if an image is blank/uniform (like gray captures with no document)
  // Returns true if image appears to be blank/invalid
  const isBlankImage = (imageData: ImageData): boolean => {
    const data = imageData.data;
    const pixelCount = data.length / 4;

    if (pixelCount === 0) return true;

    // Calculate mean and variance of grayscale values
    let sum = 0;
    let sumSq = 0;
    let minVal = 255;
    let maxVal = 0;

    // Sample pixels for performance
    const step = 16;
    let sampledCount = 0;

    for (let i = 0; i < data.length; i += step * 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];

      // Convert to grayscale
      const gray = (r + g + b) / 3;

      sum += gray;
      sumSq += gray * gray;
      sampledCount++;

      if (gray < minVal) minVal = gray;
      if (gray > maxVal) maxVal = gray;
    }

    const mean = sum / sampledCount;
    const variance = (sumSq / sampledCount) - (mean * mean);
    const stdDev = Math.sqrt(variance);
    const range = maxVal - minVal;

    // Log for debugging
    console.log(`Image analysis: mean=${mean.toFixed(1)}, stdDev=${stdDev.toFixed(1)}, range=${range}`);

    // Image is considered blank if:
    // 1. Very low variance (uniform color) - stdDev < 15
    // 2. Narrow color range - range < 50
    // 3. OR if it's mostly gray (mean between 80-180) with low contrast
    const isLowContrast = stdDev < 15 && range < 50;
    const isUniformGray = mean > 80 && mean < 180 && stdDev < 20 && range < 60;

    if (isLowContrast || isUniformGray) {
      console.log('⚠️ Blank/uniform image detected - will be rejected');
      return true;
    }

    return false;
  };

  // Get current frame as ImageData (scaled down for comparison)
  const getCurrentFrameData = (): ImageData | null => {
    try {
      if (!videoRef.current || !comparisonCanvasRef.current) return null;

      const video = videoRef.current;
      const canvas = comparisonCanvasRef.current;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });

      if (!ctx || !video.videoWidth || video.paused || video.ended) {
        if (video.paused) {
          console.warn('Video paused - attempting resume...');
          video.play().catch(e => console.warn('Resume failed:', e));
        }
        return null;
      }

      // Use small size for fast comparison (160x120)
      canvas.width = 160;
      canvas.height = 120;

      try {
        ctx.drawImage(video, 0, 0, 160, 120);
      } catch (drawError) {
        console.warn('Frame draw error:', drawError);
        return null;
      }

      const imageData = ctx.getImageData(0, 0, 160, 120);

      // Validate frame data
      if (!imageData || !imageData.data || imageData.data.length === 0) {
        console.warn('Invalid frame data');
        return null;
      }

      return imageData;
    } catch (error) {
      console.error('getCurrentFrameData error:', error);
      return null;
    }
  };

  // Ref to track auto capture count without triggering re-renders/dependency changes
  const autoCaptureCountRef = useRef(0);

  // Main frame comparison loop - runs every 500ms (MUST be defined BEFORE startAutoCapture)
  const startFrameComparisonLoop = useCallback(() => {
    if (autoCaptureIntervalRef.current) {
      clearInterval(autoCaptureIntervalRef.current);
    }

    autoCaptureIntervalRef.current = setInterval(() => {
      try {
        if (isCapturingRef.current || captureCooldownRef.current || !videoRef.current) return;

        // Check if video stream is still active
        const video = videoRef.current;
        if (!video.videoWidth || video.videoWidth === 0 || video.paused || video.ended) {
          if (video.paused) {
            console.debug('Video paused - attempting resume...');
            video.play().catch(e => console.warn('Auto-resume failed:', e));
          }
          return;
        }

        const currentFrame = getCurrentFrameData();
        if (!currentFrame) return;

        // First capture: no previous frame to compare
        if (!lastCapturedImageDataRef.current) {
          // Wait for document to be stable for 3 consecutive frames
          if (lastFrameImageDataRef.current) {
            const frameDiff = compareFrames(currentFrame, lastFrameImageDataRef.current);

            if (frameDiff < 5) {
              // Frame is stable (< 5% change)
              stableFrameCountRef.current++;
              setFrameChangeStatus('detecting');

              if (stableFrameCountRef.current >= 2) {
                // Stable for 1 second - capture first document
                console.log('📷 First document stable - capturing...');
                captureInBackgroundRef.current?.();
                // Reset stability counter and start cooldown
                stableFrameCountRef.current = 0;
                captureCooldownRef.current = true;
                setTimeout(() => { captureCooldownRef.current = false; }, 1500);
              }
            } else {
              // Frame changed, reset stability counter
              stableFrameCountRef.current = 0;
              setFrameChangeStatus('waiting');
            }
          }

          lastFrameImageDataRef.current = currentFrame;
          return;
        }

        // Compare current frame with last CAPTURED frame
        const diffFromCaptured = compareFrames(currentFrame, lastCapturedImageDataRef.current);

        // Also compare with previous frame to detect stability
        const diffFromLastFrame = lastFrameImageDataRef.current
          ? compareFrames(currentFrame, lastFrameImageDataRef.current)
          : 100;

        // Document changed significantly from captured (> 25% difference)
        // AND current frame is stable (< 5% change from previous frame)
        if (diffFromCaptured > 25) {
          // New document detected
          if (diffFromLastFrame < 5) {
            // And it's stable
            stableFrameCountRef.current++;
            setFrameChangeStatus('ready');

            if (stableFrameCountRef.current >= 2) {
              // New stable document - capture it!
              console.log(`📷 New document detected (${diffFromCaptured.toFixed(1)}% different) - capturing...`);
              captureInBackgroundRef.current?.();
              // Reset stability counter and start cooldown using inter-capture delay
              stableFrameCountRef.current = 0;
              captureCooldownRef.current = true;
              // Use configured inter-capture delay via ref (to avoid stale closure)
              const cooldownMs = Math.max(500, (interCaptureDelayRef.current || 2) * 1000);
              setTimeout(() => { captureCooldownRef.current = false; }, cooldownMs);
            }
          } else {
            // Document still moving
            stableFrameCountRef.current = 0;
            setFrameChangeStatus('detecting');
          }
        } else {
          // Same document as before
          stableFrameCountRef.current = 0;
          setFrameChangeStatus('waiting');
        }

        lastFrameImageDataRef.current = currentFrame;

      } catch (error) {
        console.error('Frame comparison loop error:', error);
        // Continue loop even on error - don't stop the interval
      }
    }, 500); // Check every 500ms
  }, []);

  // Start continuous auto-capture mode with frame comparison
  // NOTE: Delay is now applied in socket.on('start_auto_capture') handler, not here
  const startAutoCapture = useCallback(async (source: 'local' | 'dashboard' = 'local', documentCount?: number) => {
    // Reset the initial delay flag when starting a new capture session
    hasAppliedInitialDelayRef.current = false;

    setAutoCapture(true);
    setAutoCaptureCount(0);
    autoCaptureCountRef.current = 0;
    lastCapturedImageDataRef.current = null;
    lastFrameImageDataRef.current = null;
    stableFrameCountRef.current = 0;
    setFrameChangeStatus('waiting');
    setAutoCaptureSource(source);
    if (documentCount) setPendingDocumentCount(documentCount);

    // Start frame comparison loop
    startFrameComparisonLoop();

    // Notify dashboard if started locally
    if (source === 'local' && socket) {
      socket.emit('auto_capture_state_changed', { enabled: true, source: 'phone' });
    }

    toast({
      title: source === 'dashboard' ? '📱 Auto-Capture Started!' : 'Auto-Capture Enabled',
      description: source === 'dashboard'
        ? `Ready to capture ${documentCount || 'multiple'} document(s). Place documents one by one.`
        : 'Place documents one by one. Each new document will be captured automatically.',
      status: 'success',
      duration: 3000,
    });
  }, [socket, toast, startFrameComparisonLoop]);

  // Stop continuous auto-capture mode
  const stopAutoCapture = useCallback(() => {
    if (autoCaptureIntervalRef.current) {
      clearInterval(autoCaptureIntervalRef.current);
      autoCaptureIntervalRef.current = null;
    }
    // Clear countdown if running
    if (countdownIntervalRef.current) {
      clearInterval(countdownIntervalRef.current);
      countdownIntervalRef.current = null;
    }
    // Cancel startup delay countdown if running
    cancelCountdown();
    setIsWaitingForInitialDelay(false);
    hasAppliedInitialDelayRef.current = false;

    setCountdown(null);
    setAutoCapture(false);
    setFrameChangeStatus('waiting');
    lastCapturedImageDataRef.current = null;
    lastFrameImageDataRef.current = null;
    stableFrameCountRef.current = 0;

    // Notify dashboard that auto-capture stopped
    if (socket) {
      socket.emit('auto_capture_state_changed', { enabled: false, source: 'phone', capturedCount: autoCaptureCountRef.current });
    }

    const wasRemote = autoCaptureSource === 'dashboard';
    setAutoCaptureSource(null);
    setPendingDocumentCount(0);

    if (autoCaptureCountRef.current > 0) {
      toast({
        title: 'Auto-Capture Stopped',
        description: `Captured ${autoCaptureCountRef.current} document(s)${wasRemote ? ' - Dashboard notified' : ''}`,
        status: 'success',
        duration: 3000,
      });
    }
  }, [toast, socket, autoCaptureSource]);

  // Process upload queue
  useEffect(() => {
    const processQueue = async () => {
      if (isProcessingQueue || uploadQueue.length === 0) return;

      setIsProcessingQueue(true);
      const item = uploadQueue[0];

      try {
        const formData = new FormData();
        formData.append('file', item.blob, item.filename);

        // Route to test-capture endpoint if in test mode
        if (item.isTestCapture) {
          await apiClient.post('/test-capture', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
          });
          toast({
            title: '🧪 Test Capture Uploaded',
            description: 'Check calibration settings to see preview',
            status: 'info',
            duration: 2000,
            isClosable: true,
            position: 'top-right',
          });
        } else {
          // Normal upload with processing options
          formData.append('auto_crop', item.options.autoCrop.toString());
          formData.append('ai_enhance', item.options.aiEnhance.toString());
          formData.append('strict_quality', item.options.strictQuality.toString());

          await apiClient.post(API_ENDPOINTS.upload, formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
          });
          toast({
            title: 'Upload complete',
            description: `${item.filename} processed`,
            status: 'success',
            duration: 2000,
            isClosable: true,
            position: 'top-right',
          });
        }
      } catch (err) {
        console.error('Queue upload failed:', err);
        toast({
          title: 'Upload failed',
          description: 'Failed to upload image',
          status: 'error',
          duration: 3000,
          position: 'top-right',
        });
      } finally {
        setUploadQueue(prev => prev.slice(1));
        setIsProcessingQueue(false);
      }
    };

    processQueue();
  }, [uploadQueue, isProcessingQueue, toast]);

  // Ref to hold latest captureInBackground function to avoid stale closures in interval
  const captureInBackgroundRef = useRef<(() => Promise<void>) | null>(null);

  const startAsyncUpload = useCallback((blob: Blob, filename: string, optionsSnapshot: typeof processingOptions, isTestCapture: boolean = false) => {
    // Add to queue instead of immediate upload
    setUploadQueue(prev => [...prev, {
      id: Date.now().toString() + Math.random(),
      blob,
      filename,
      options: optionsSnapshot,
      isTestCapture,  // Pass test capture flag
    }]);

    // Only increment capture count for non-test captures
    if (!isTestCapture) {
      setAutoCaptureCount(prev => {
        const newCount = prev + 1;
        autoCaptureCountRef.current = newCount;
        return newCount;
      });
    }

    // Toast removed to prevent re-renders during rapid capture
    console.log(`📸 Queued: ${filename}${isTestCapture ? ' (TEST)' : ''}`);
  }, []);

  // Background capture without freezing camera
  const captureInBackground = useCallback(async () => {
    if (isCapturingRef.current || !videoRef.current || !canvasRef.current) return;

    isCapturingRef.current = true;
    setFrameChangeStatus('captured');

    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d', { willReadFrequently: true });

      if (!context || !video.videoWidth || video.videoWidth === 0) {
        console.warn('Video not ready for capture, skipping...');
        setFrameChangeStatus('waiting');
        return;
      }

      // Capture full resolution frame
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0);

      // Store the captured frame for comparison
      const capturedFrame = getCurrentFrameData();
      if (capturedFrame) {
        lastCapturedImageDataRef.current = capturedFrame;

        // Check if the image is blank/uniform (gray, no document content)
        if (isBlankImage(capturedFrame)) {
          console.log('⚠️ Blank image detected - skipping upload');
          toast({
            title: 'Blank Image Detected',
            description: 'No document content found - capture skipped',
            status: 'warning',
            duration: 2000,
            position: 'top',
          });
          // Reset and wait for next document
          stableFrameCountRef.current = 0;
          lastFrameImageDataRef.current = null;
          setFrameChangeStatus('waiting');
          return;
        }
      }

      // Convert to blob
      const blob = await new Promise<Blob | null>((resolve) => {
        canvas.toBlob(
          (b) => resolve(b),
          'image/jpeg',
          0.9
        );
      });

      if (!blob) {
        console.warn('Failed to create blob from canvas');
        setFrameChangeStatus('waiting');
        return;
      }

      const filename = `auto_capture_${Date.now()}.jpg`;
      const optionsSnapshot = { ...processingOptions };

      // Kick off upload without blocking future captures
      // Pass isTestCaptureMode to route to test endpoint if in calibration test mode
      startAsyncUpload(blob, filename, optionsSnapshot, isTestCaptureMode);

      // Reset frame tracking so next document can be detected immediately
      stableFrameCountRef.current = 0;
      lastFrameImageDataRef.current = null;
      setFrameChangeStatus('waiting');

    } catch (err) {
      console.error('Background capture error:', err);
      lastCapturedImageDataRef.current = null;
      setFrameChangeStatus('waiting');
    } finally {
      // Allow next capture immediately (uploads continue async)
      isCapturingRef.current = false;
    }
  }, [processingOptions, startAsyncUpload, isTestCaptureMode, toast]);

  // Keep ref updated with latest captureInBackground
  useEffect(() => {
    captureInBackgroundRef.current = captureInBackground;
  }, [captureInBackground]);

  const detectDocumentQuad = useCallback((): DetectedQuad | null => {
    const video = videoRef.current;
    if (!video || video.readyState < 2 || !video.videoWidth) return null;

    // Use offscreen canvas for detection - higher resolution for better accuracy
    const targetWidth = 320;  // Increased from 200 for better edge detection
    const aspectRatio = video.videoHeight / video.videoWidth;
    const targetHeight = Math.round(targetWidth * aspectRatio);

    const canvas = docDetectionCanvasRef.current ?? document.createElement('canvas');
    docDetectionCanvasRef.current = canvas;
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return null;

    ctx.drawImage(video, 0, 0, targetWidth, targetHeight);
    const imageData = ctx.getImageData(0, 0, targetWidth, targetHeight);
    const { data, width, height } = imageData;

    // Convert to grayscale with edge-aware enhancement
    const gray: number[][] = Array(height).fill(null).map(() => Array(width).fill(0));
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        gray[y][x] = data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114;
      }
    }

    // Apply Gaussian blur (5x5 for better noise reduction)
    const blurred: number[][] = Array(height).fill(null).map(() => Array(width).fill(0));
    const kernel5x5 = [
      1, 4, 6, 4, 1,
      4, 16, 24, 16, 4,
      6, 24, 36, 24, 6,
      4, 16, 24, 16, 4,
      1, 4, 6, 4, 1
    ];
    const kernel5x5Sum = 256;

    for (let y = 2; y < height - 2; y++) {
      for (let x = 2; x < width - 2; x++) {
        let sum = 0;
        let k = 0;
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            sum += gray[y + dy][x + dx] * kernel5x5[k++];
          }
        }
        blurred[y][x] = sum / kernel5x5Sum;
      }
    }

    // Apply CLAHE-like local contrast enhancement for better edge detection
    const enhanced: number[][] = Array(height).fill(null).map(() => Array(width).fill(0));
    const blockSize = 16;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        // Calculate local mean and std in a block
        let localSum = 0, localSumSq = 0, count = 0;
        const halfBlock = Math.floor(blockSize / 2);
        for (let dy = -halfBlock; dy <= halfBlock; dy++) {
          for (let dx = -halfBlock; dx <= halfBlock; dx++) {
            const ny = Math.max(0, Math.min(height - 1, y + dy));
            const nx = Math.max(0, Math.min(width - 1, x + dx));
            localSum += blurred[ny][nx];
            localSumSq += blurred[ny][nx] * blurred[ny][nx];
            count++;
          }
        }
        const localMean = localSum / count;
        const localVar = Math.max(1, (localSumSq / count) - (localMean * localMean));
        const localStd = Math.sqrt(localVar);
        // Enhance contrast locally
        enhanced[y][x] = Math.max(0, Math.min(255, 128 + (blurred[y][x] - localMean) * 2 / Math.max(localStd / 64, 1)));
      }
    }

    // Compute gradients using Sobel operator with direction on enhanced image
    const gradMag: number[][] = Array(height).fill(null).map(() => Array(width).fill(0));
    const gradDir: number[][] = Array(height).fill(null).map(() => Array(width).fill(0));

    for (let y = 2; y < height - 2; y++) {
      for (let x = 2; x < width - 2; x++) {
        const gx = -enhanced[y - 1][x - 1] + enhanced[y - 1][x + 1]
          - 2 * enhanced[y][x - 1] + 2 * enhanced[y][x + 1]
          - enhanced[y + 1][x - 1] + enhanced[y + 1][x + 1];
        const gy = -enhanced[y - 1][x - 1] - 2 * enhanced[y - 1][x] - enhanced[y - 1][x + 1]
          + enhanced[y + 1][x - 1] + 2 * enhanced[y + 1][x] + enhanced[y + 1][x + 1];
        gradMag[y][x] = Math.sqrt(gx * gx + gy * gy);
        gradDir[y][x] = Math.atan2(gy, gx);
      }
    }

    // Non-maximum suppression for thinner, more precise edges
    const suppressed: number[][] = Array(height).fill(null).map(() => Array(width).fill(0));
    for (let y = 2; y < height - 2; y++) {
      for (let x = 2; x < width - 2; x++) {
        const angle = gradDir[y][x];
        const mag = gradMag[y][x];

        // Quantize angle to nearest 45 degrees
        let dx1 = 0, dy1 = 0, dx2 = 0, dy2 = 0;
        const angleNorm = ((angle * 180 / Math.PI) + 180) % 180;

        if (angleNorm < 22.5 || angleNorm >= 157.5) {
          dx1 = 1; dy1 = 0; dx2 = -1; dy2 = 0;
        } else if (angleNorm < 67.5) {
          dx1 = 1; dy1 = 1; dx2 = -1; dy2 = -1;
        } else if (angleNorm < 112.5) {
          dx1 = 0; dy1 = 1; dx2 = 0; dy2 = -1;
        } else {
          dx1 = -1; dy1 = 1; dx2 = 1; dy2 = -1;
        }

        const n1 = gradMag[y + dy1]?.[x + dx1] ?? 0;
        const n2 = gradMag[y + dy2]?.[x + dx2] ?? 0;

        // Keep only local maxima
        if (mag >= n1 && mag >= n2) {
          suppressed[y][x] = mag;
        }
      }
    }

    // Adaptive dual threshold (Canny-style hysteresis) with improved percentiles
    const histogram = new Array(256).fill(0);
    let maxMag = 0;
    for (let y = 2; y < height - 2; y++) {
      for (let x = 2; x < width - 2; x++) {
        const bin = Math.min(255, Math.floor(suppressed[y][x]));
        histogram[bin]++;
        if (suppressed[y][x] > maxMag) maxMag = suppressed[y][x];
      }
    }

    // Find thresholds using percentiles - tuned for document edges
    const totalPixels = (width - 4) * (height - 4);
    let cumulative = 0;
    let lowThreshold = 25, highThreshold = 60;

    // Use Otsu-like method for better threshold selection
    for (let i = 0; i < 256; i++) {
      cumulative += histogram[i];
      if (cumulative > totalPixels * 0.75 && lowThreshold === 25) {
        lowThreshold = Math.max(20, Math.min(i, maxMag * 0.15));
      }
      if (cumulative > totalPixels * 0.92) {
        highThreshold = Math.max(lowThreshold * 2.5, Math.min(i, maxMag * 0.35));
        break;
      }
    }

    // Apply hysteresis thresholding
    const edges: boolean[][] = Array(height).fill(null).map(() => Array(width).fill(false));
    const strong: boolean[][] = Array(height).fill(null).map(() => Array(width).fill(false));

    // Mark strong edges
    for (let y = 2; y < height - 2; y++) {
      for (let x = 2; x < width - 2; x++) {
        if (suppressed[y][x] >= highThreshold) {
          strong[y][x] = true;
          edges[y][x] = true;
        }
      }
    }

    // Connect weak edges to strong edges with limited iterations
    let changed = true;
    let iterations = 0;
    const maxIterations = 5;
    while (changed && iterations < maxIterations) {
      changed = false;
      iterations++;
      for (let y = 2; y < height - 2; y++) {
        for (let x = 2; x < width - 2; x++) {
          if (!edges[y][x] && suppressed[y][x] >= lowThreshold) {
            // Check if connected to strong edge
            for (let dy = -1; dy <= 1; dy++) {
              for (let dx = -1; dx <= 1; dx++) {
                if (edges[y + dy]?.[x + dx]) {
                  edges[y][x] = true;
                  changed = true;
                  break;
                }
              }
              if (edges[y][x]) break;
            }
          }
        }
      }
    }

    // Morphological closing to connect nearby edges
    const closed: boolean[][] = Array(height).fill(null).map(() => Array(width).fill(false));
    // Dilate
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        if (edges[y][x] || edges[y-1][x] || edges[y+1][x] || edges[y][x-1] || edges[y][x+1]) {
          closed[y][x] = true;
        }
      }
    }
    // Erode back
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        if (closed[y][x] && closed[y-1][x] && closed[y+1][x] && closed[y][x-1] && closed[y][x+1]) {
          edges[y][x] = true;
        }
      }
    }

    // Find contours using boundary tracing
    const visited: boolean[][] = Array(height).fill(null).map(() => Array(width).fill(false));
    const contours: { x: number; y: number }[][] = [];

    const traceContour = (startX: number, startY: number): { x: number; y: number }[] => {
      const contour: { x: number; y: number }[] = [];
      const directions = [
        { dx: 1, dy: 0 }, { dx: 1, dy: 1 }, { dx: 0, dy: 1 }, { dx: -1, dy: 1 },
        { dx: -1, dy: 0 }, { dx: -1, dy: -1 }, { dx: 0, dy: -1 }, { dx: 1, dy: -1 }
      ];

      let x = startX, y = startY;
      let dir = 0;

      do {
        contour.push({ x, y });
        visited[y][x] = true;

        let found = false;
        for (let i = 0; i < 8; i++) {
          const checkDir = (dir + 6 + i) % 8;
          const nx = x + directions[checkDir].dx;
          const ny = y + directions[checkDir].dy;

          if (nx >= 0 && nx < width && ny >= 0 && ny < height && edges[ny][nx] && !visited[ny][nx]) {
            x = nx;
            y = ny;
            dir = checkDir;
            found = true;
            break;
          }
        }

        if (!found) break;
      } while (!(x === startX && y === startY) && contour.length < 8000);

      return contour;
    };

    // Find all contours with adaptive minimum length based on image size
    const minContourLength = Math.max(60, Math.min(width, height) * 0.25);
    for (let y = 2; y < height - 2; y++) {
      for (let x = 2; x < width - 2; x++) {
        if (edges[y][x] && !visited[y][x]) {
          const contour = traceContour(x, y);
          if (contour.length > minContourLength) {
            contours.push(contour);
          }
        }
      }
    }

    if (contours.length === 0) return null;

    // Douglas-Peucker simplification with adaptive epsilon
    const simplifyContour = (points: { x: number; y: number }[], epsilon: number): { x: number; y: number }[] => {
      if (points.length < 3) return points;

      let maxDist = 0;
      let maxIdx = 0;
      const first = points[0];
      const last = points[points.length - 1];

      for (let i = 1; i < points.length - 1; i++) {
        const p = points[i];
        const num = Math.abs((last.y - first.y) * p.x - (last.x - first.x) * p.y + last.x * first.y - last.y * first.x);
        const den = Math.sqrt((last.y - first.y) ** 2 + (last.x - first.x) ** 2);
        const dist = den > 0 ? num / den : 0;

        if (dist > maxDist) {
          maxDist = dist;
          maxIdx = i;
        }
      }

      if (maxDist > epsilon) {
        const left = simplifyContour(points.slice(0, maxIdx + 1), epsilon);
        const right = simplifyContour(points.slice(maxIdx), epsilon);
        return [...left.slice(0, -1), ...right];
      }

      return [first, last];
    };

    // Helper: check if quadrilateral is convex
    const isConvex = (pts: { x: number; y: number }[]): boolean => {
      if (pts.length !== 4) return false;
      let sign = 0;
      for (let i = 0; i < 4; i++) {
        const p1 = pts[i];
        const p2 = pts[(i + 1) % 4];
        const p3 = pts[(i + 2) % 4];
        const cross = (p2.x - p1.x) * (p3.y - p2.y) - (p2.y - p1.y) * (p3.x - p2.x);
        if (cross !== 0) {
          if (sign === 0) sign = cross > 0 ? 1 : -1;
          else if ((cross > 0 ? 1 : -1) !== sign) return false;
        }
      }
      return true;
    };

    // Helper: calculate angle at corner (in degrees)
    const cornerAngle = (p1: { x: number, y: number }, p2: { x: number, y: number }, p3: { x: number, y: number }): number => {
      const v1 = { x: p1.x - p2.x, y: p1.y - p2.y };
      const v2 = { x: p3.x - p2.x, y: p3.y - p2.y };
      const dot = v1.x * v2.x + v1.y * v2.y;
      const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y);
      const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y);
      if (mag1 === 0 || mag2 === 0) return 0;
      return Math.acos(Math.max(-1, Math.min(1, dot / (mag1 * mag2)))) * 180 / Math.PI;
    };

    // Helper: calculate edge lengths of quad
    const edgeLengths = (pts: { x: number; y: number }[]): number[] => {
      const lengths: number[] = [];
      for (let i = 0; i < 4; i++) {
        const p1 = pts[i];
        const p2 = pts[(i + 1) % 4];
        lengths.push(Math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2));
      }
      return lengths;
    };

    // Helper: score a quadrilateral for document-likeness
    const scoreQuad = (quad: { x: number; y: number }[]): number => {
      // Area score
      const area = 0.5 * Math.abs(
        (quad[0].x * quad[1].y - quad[1].x * quad[0].y) +
        (quad[1].x * quad[2].y - quad[2].x * quad[1].y) +
        (quad[2].x * quad[3].y - quad[3].x * quad[2].y) +
        (quad[3].x * quad[0].y - quad[0].x * quad[3].y)
      );
      const imageArea = width * height;
      const areaRatio = area / imageArea;

      // Reject too small or too large
      if (areaRatio < 0.05 || areaRatio > 0.95) return 0;

      // Area score (prefer 15-80% coverage)
      let areaScore = 0;
      if (areaRatio >= 0.15 && areaRatio <= 0.80) {
        areaScore = 100;
      } else if (areaRatio >= 0.08 && areaRatio < 0.15) {
        areaScore = 50;
      } else if (areaRatio > 0.80 && areaRatio <= 0.90) {
        areaScore = 30;
      } else {
        areaScore = 10;
      }

      // Angle score (prefer near 90 degrees)
      const angles = [
        cornerAngle(quad[3], quad[0], quad[1]),
        cornerAngle(quad[0], quad[1], quad[2]),
        cornerAngle(quad[1], quad[2], quad[3]),
        cornerAngle(quad[2], quad[3], quad[0]),
      ];
      const avgAngleDeviation = angles.reduce((s, a) => s + Math.abs(a - 90), 0) / 4;
      let angleScore = 0;
      if (avgAngleDeviation < 8) angleScore = 100;
      else if (avgAngleDeviation < 15) angleScore = 70;
      else if (avgAngleDeviation < 25) angleScore = 40;
      else if (avgAngleDeviation < 40) angleScore = 15;
      else return 0; // Too skewed

      // Edge ratio score (prefer rectangular proportions)
      const edges = edgeLengths(quad);
      const minEdge = Math.min(...edges);
      const maxEdge = Math.max(...edges);
      const edgeRatio = minEdge / maxEdge;
      let edgeScore = 0;
      if (edgeRatio > 0.4) edgeScore = 80;
      else if (edgeRatio > 0.25) edgeScore = 50;
      else if (edgeRatio > 0.15) edgeScore = 20;
      else return 0; // Too elongated

      // Margin score - penalize if too close to image edges
      const margin = 0.03; // 3% margin
      const minX = Math.min(quad[0].x, quad[1].x, quad[2].x, quad[3].x) / width;
      const maxX = Math.max(quad[0].x, quad[1].x, quad[2].x, quad[3].x) / width;
      const minY = Math.min(quad[0].y, quad[1].y, quad[2].y, quad[3].y) / height;
      const maxY = Math.max(quad[0].y, quad[1].y, quad[2].y, quad[3].y) / height;

      let marginScore = 100;
      if (minX < margin || minY < margin || maxX > (1 - margin) || maxY > (1 - margin)) {
        marginScore = 20; // Edge detection might be picking up frame boundaries
      }

      // Parallel edges score (opposite edges should be roughly parallel)
      const parallelScore = (() => {
        const angle01 = Math.atan2(quad[1].y - quad[0].y, quad[1].x - quad[0].x);
        const angle32 = Math.atan2(quad[2].y - quad[3].y, quad[2].x - quad[3].x);
        const angle12 = Math.atan2(quad[2].y - quad[1].y, quad[2].x - quad[1].x);
        const angle03 = Math.atan2(quad[3].y - quad[0].y, quad[3].x - quad[0].x);

        const diff1 = Math.abs(angle01 - angle32) * 180 / Math.PI;
        const diff2 = Math.abs(angle12 - angle03) * 180 / Math.PI;
        const avgDiff = (Math.min(diff1, 180 - diff1) + Math.min(diff2, 180 - diff2)) / 2;

        if (avgDiff < 10) return 80;
        if (avgDiff < 20) return 50;
        if (avgDiff < 35) return 25;
        return 0;
      })();

      // Weighted total score
      return (areaScore * 0.25) + (angleScore * 0.30) + (edgeScore * 0.15) + (marginScore * 0.15) + (parallelScore * 0.15);
    };

    // Find the best quadrilateral
    let bestQuad: { x: number; y: number }[] | null = null;
    let bestScore = 0;

    // Sort contours by length (longer first - more likely to be document boundary)
    const sortedContours = [...contours].sort((a, b) => b.length - a.length);

    for (const contour of sortedContours.slice(0, 15)) { // Check top 15 contours
      // Try multiple epsilon values for simplification
      for (const epsilonFactor of [0.015, 0.02, 0.025, 0.03]) {
        const epsilon = Math.max(contour.length, 100) * epsilonFactor;
        const simplified = simplifyContour(contour, epsilon);

        if (simplified.length >= 4 && simplified.length <= 12) {
          // Find 4 corners using distance from centroid + angle
          const cx = simplified.reduce((s, p) => s + p.x, 0) / simplified.length;
          const cy = simplified.reduce((s, p) => s + p.y, 0) / simplified.length;

          // Score each point by distance from center
          const scored = simplified.map(p => ({
            ...p,
            dist: Math.sqrt((p.x - cx) ** 2 + (p.y - cy) ** 2),
            angle: Math.atan2(p.y - cy, p.x - cx)
          }));

          // Sort by angle and pick 4 points from 4 quadrants
          scored.sort((a, b) => a.angle - b.angle);

          // Find points in each quadrant (TL, TR, BR, BL)
          const quadrants: { x: number; y: number; dist: number }[] = [
            { x: 0, y: 0, dist: 0 }, // TL: angle ~135 to 180 or -180 to -90
            { x: 0, y: 0, dist: 0 }, // TR: angle -90 to 0
            { x: 0, y: 0, dist: 0 }, // BR: angle 0 to 90
            { x: 0, y: 0, dist: 0 }, // BL: angle 90 to 135
          ];

          for (const p of scored) {
            const deg = p.angle * 180 / Math.PI;
            let qi = -1;
            if (deg >= -180 && deg < -90) qi = 0; // TL
            else if (deg >= -90 && deg < 0) qi = 1; // TR
            else if (deg >= 0 && deg < 90) qi = 2; // BR
            else qi = 3; // BL

            if (p.dist > quadrants[qi].dist) {
              quadrants[qi] = { x: p.x, y: p.y, dist: p.dist };
            }
          }

          // Check if all quadrants have valid points
          if (quadrants.every(q => q.dist > 0)) {
            const quad = quadrants.map(q => ({ x: q.x, y: q.y }));

            // Validate convexity
            if (!isConvex(quad)) continue;

            // Score this quad
            const score = scoreQuad(quad);

            if (score > bestScore) {
              bestScore = score;
              bestQuad = quad;
            }
          }
        }
      }
    }

    // Fallback: use edge point extremes if no good quad from contours
    if (!bestQuad) {
      const allEdgePoints: { x: number; y: number }[] = [];
      for (let y = 2; y < height - 2; y++) {
        for (let x = 2; x < width - 2; x++) {
          if (edges[y][x]) allEdgePoints.push({ x, y });
        }
      }

      if (allEdgePoints.length < 150) return null;

      // Use centroid-based quadrant selection
      const cx = allEdgePoints.reduce((s, p) => s + p.x, 0) / allEdgePoints.length;
      const cy = allEdgePoints.reduce((s, p) => s + p.y, 0) / allEdgePoints.length;

      const quadrants: { x: number; y: number; dist: number }[] = [
        { x: 0, y: 0, dist: 0 },
        { x: 0, y: 0, dist: 0 },
        { x: 0, y: 0, dist: 0 },
        { x: 0, y: 0, dist: 0 },
      ];

      for (const p of allEdgePoints) {
        const angle = Math.atan2(p.y - cy, p.x - cx) * 180 / Math.PI;
        const dist = Math.sqrt((p.x - cx) ** 2 + (p.y - cy) ** 2);

        let qi = -1;
        if (angle >= -180 && angle < -90) qi = 0;
        else if (angle >= -90 && angle < 0) qi = 1;
        else if (angle >= 0 && angle < 90) qi = 2;
        else qi = 3;

        if (dist > quadrants[qi].dist) {
          quadrants[qi] = { x: p.x, y: p.y, dist };
        }
      }

      if (quadrants.every(q => q.dist > 0)) {
        const quad = quadrants.map(q => ({ x: q.x, y: q.y }));

        const area = 0.5 * Math.abs(
          (quad[0].x * quad[1].y - quad[1].x * quad[0].y) +
          (quad[1].x * quad[2].y - quad[2].x * quad[1].y) +
          (quad[2].x * quad[3].y - quad[3].x * quad[2].y) +
          (quad[3].x * quad[0].y - quad[0].x * quad[3].y)
        );

        if (area > width * height * 0.08) {
          bestQuad = quad;
        }
      }
    }

    if (!bestQuad) return null;

    // Normalize to 0-1 range
    const norm = (val: number, max: number) => clamp(val / max, 0, 1);
    const [tl, tr, br, bl] = bestQuad;

    return {
      topLeft: { x: norm(tl.x, width), y: norm(tl.y, height) },
      topRight: { x: norm(tr.x, width), y: norm(tr.y, height) },
      bottomRight: { x: norm(br.x, width), y: norm(br.y, height) },
      bottomLeft: { x: norm(bl.x, width), y: norm(bl.y, height) },
    };
  }, []);

  useEffect(() => {
    if (!stream) {
      setDetectedQuad(null);
      return;
    }

    let lastQuad: DetectedQuad | null = null;
    let frameCount = 0;
    let stableCount = 0; // Track how many stable frames
    let noDetectionCount = 0; // Track frames without detection
    let recentQuads: DetectedQuad[] = []; // Buffer for temporal averaging
    const maxRecentQuads = 5;

    // Adaptive smoothing - smoother when stable, more responsive when changing
    const getSmoothing = () => {
      if (stableCount > 10) return 0.08; // Very stable - slow smoothing
      if (stableCount > 5) return 0.15;  // Stable - moderate smoothing
      return 0.30; // Unstable - responsive
    };

    // Calculate weighted average of recent quads
    const averageQuads = (quads: DetectedQuad[]): DetectedQuad | null => {
      if (quads.length === 0) return null;
      
      // More recent quads get higher weight
      let totalWeight = 0;
      let avgTL = { x: 0, y: 0 };
      let avgTR = { x: 0, y: 0 };
      let avgBR = { x: 0, y: 0 };
      let avgBL = { x: 0, y: 0 };
      
      quads.forEach((q, i) => {
        const weight = Math.pow(1.5, i); // Exponential weights
        totalWeight += weight;
        avgTL.x += q.topLeft.x * weight;
        avgTL.y += q.topLeft.y * weight;
        avgTR.x += q.topRight.x * weight;
        avgTR.y += q.topRight.y * weight;
        avgBR.x += q.bottomRight.x * weight;
        avgBR.y += q.bottomRight.y * weight;
        avgBL.x += q.bottomLeft.x * weight;
        avgBL.y += q.bottomLeft.y * weight;
      });
      
      return {
        topLeft: { x: avgTL.x / totalWeight, y: avgTL.y / totalWeight },
        topRight: { x: avgTR.x / totalWeight, y: avgTR.y / totalWeight },
        bottomRight: { x: avgBR.x / totalWeight, y: avgBR.y / totalWeight },
        bottomLeft: { x: avgBL.x / totalWeight, y: avgBL.y / totalWeight },
      };
    };

    // Calculate movement magnitude between two quads
    const quadMovement = (q1: DetectedQuad, q2: DetectedQuad): number => {
      return Math.abs(q1.topLeft.x - q2.topLeft.x) +
        Math.abs(q1.topLeft.y - q2.topLeft.y) +
        Math.abs(q1.topRight.x - q2.topRight.x) +
        Math.abs(q1.topRight.y - q2.topRight.y) +
        Math.abs(q1.bottomRight.x - q2.bottomRight.x) +
        Math.abs(q1.bottomRight.y - q2.bottomRight.y) +
        Math.abs(q1.bottomLeft.x - q2.bottomLeft.x) +
        Math.abs(q1.bottomLeft.y - q2.bottomLeft.y);
    };

    const tick = () => {
      frameCount++;

      // Process every frame for smoother detection (was skipping every other)
      const quad = detectDocumentQuad();

      if (quad) {
        noDetectionCount = 0;

        // Add to recent quads buffer
        recentQuads.push(quad);
        if (recentQuads.length > maxRecentQuads) {
          recentQuads.shift();
        }

        if (lastQuad) {
          // Calculate movement from last frame
          const movement = quadMovement(quad, lastQuad);

          // If very stable, increase stability counter
          if (movement < 0.015) {
            stableCount = Math.min(stableCount + 1, 25);
          } else if (movement < 0.04) {
            stableCount = Math.max(0, stableCount - 1);
          } else {
            stableCount = Math.max(0, stableCount - 3);
            // Clear buffer on large movement for faster response
            if (movement > 0.1) {
              recentQuads = [quad];
            }
          }

          // Use temporal averaging when stable
          let targetQuad = quad;
          if (stableCount > 3 && recentQuads.length >= 3) {
            const avgQuad = averageQuads(recentQuads);
            if (avgQuad) targetQuad = avgQuad;
          }

          const smoothingFactor = getSmoothing();
          const smoothed: DetectedQuad = {
            topLeft: {
              x: lastQuad.topLeft.x + (targetQuad.topLeft.x - lastQuad.topLeft.x) * smoothingFactor,
              y: lastQuad.topLeft.y + (targetQuad.topLeft.y - lastQuad.topLeft.y) * smoothingFactor,
            },
            topRight: {
              x: lastQuad.topRight.x + (targetQuad.topRight.x - lastQuad.topRight.x) * smoothingFactor,
              y: lastQuad.topRight.y + (targetQuad.topRight.y - lastQuad.topRight.y) * smoothingFactor,
            },
            bottomRight: {
              x: lastQuad.bottomRight.x + (targetQuad.bottomRight.x - lastQuad.bottomRight.x) * smoothingFactor,
              y: lastQuad.bottomRight.y + (targetQuad.bottomRight.y - lastQuad.bottomRight.y) * smoothingFactor,
            },
            bottomLeft: {
              x: lastQuad.bottomLeft.x + (targetQuad.bottomLeft.x - lastQuad.bottomLeft.x) * smoothingFactor,
              y: lastQuad.bottomLeft.y + (targetQuad.bottomLeft.y - lastQuad.bottomLeft.y) * smoothingFactor,
            },
          };
          lastQuad = smoothed;
          setDetectedQuad(smoothed);
        } else {
          lastQuad = quad;
          setDetectedQuad(quad);
        }
      } else {
        // Hysteresis: require multiple no-detection frames before hiding
        noDetectionCount++;
        if (noDetectionCount > 8) { // Increased from 5 for more stability
          lastQuad = null;
          stableCount = 0;
          recentQuads = [];
          setDetectedQuad(null);
        }
      }

      detectionRafRef.current = requestAnimationFrame(tick);
    };

    detectionRafRef.current = requestAnimationFrame(tick);
    return () => {
      if (detectionRafRef.current) {
        cancelAnimationFrame(detectionRafRef.current);
        detectionRafRef.current = null;
      }
    };
  }, [stream, detectDocumentQuad]);

  const startCamera = async (orientation?: 'portrait' | 'landscape') => {
    const targetOrientation = orientation || cameraOrientation;
    try {
      // Stop existing stream first
      if (stream) {
        stream.getTracks().forEach((track: MediaStreamTrack) => track.stop());
      }

      // Get best available camera resolution
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
          const v = videoRef.current;
          if (!v) return;
          v.play().catch(e => console.error("Video play failed:", e));

          // Sync orientation based on actual stream dimensions
          if (v.videoWidth && v.videoHeight) {
            const isActualPortrait = v.videoHeight > v.videoWidth;
            setCameraOrientation(isActualPortrait ? 'portrait' : 'landscape');
          }
        };
      }
    } catch (err) {
      console.error('Camera start error:', err);
      toast({
        title: 'Camera Error',
        description: (err as Error).message,
        status: 'error',
        duration: 3000,
      });
    }
  };

  const toggleCameraOrientation = async () => {
    const newOrientation = cameraOrientation === 'portrait' ? 'landscape' : 'portrait';
    setCameraOrientation(newOrientation);
    // Restart camera with new constraints
    if (stream) {
      await startCamera(newOrientation);
    }
  };

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach((track: MediaStreamTrack) => track.stop());
      setStream(null);
    }
    if (detectionRafRef.current) {
      cancelAnimationFrame(detectionRafRef.current);
      detectionRafRef.current = null;
    }
    // Stop auto-capture if running
    if (autoCaptureIntervalRef.current) {
      clearInterval(autoCaptureIntervalRef.current);
      autoCaptureIntervalRef.current = null;
    }
    setAutoCapture(false);
  }, [stream]);

  const handleCaptureMode = (mode: 'file' | 'camera') => {
    setCaptureMode(mode);
    setPreviewImage(null);
    setQualityCheck(null);
    if (mode === 'camera') {
      startCamera();
    } else {
      stopCamera();
    }
  };

  const checkImageQuality = useCallback(
    async (file: Blob): Promise<QualityCheck | null> => {
      if (!validateQuality) return null;

      try {
        const formData = new FormData();
        formData.append('file', file, 'temp.jpg');

        const response = await apiClient.post(API_ENDPOINTS.validateQuality, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });

        const quality: QualityCheck = response.data;
        setQualityCheck(quality);

        // Show warnings if quality issues detected
        if (!quality.quality.overall_acceptable) {
          const warnings = quality.quality.issues.join('\n');
          const confirm = window.confirm(
            `⚠️ Quality Issues Detected:\n\n${warnings}\n\nRecommendations:\n${quality.quality.recommendations.join('\n')}\n\nDo you want to upload anyway?`
          );
          if (!confirm) {
            return null;
          }
        } else {
          showMessage(
            `✓ Quality: Blur ${quality.blur_score.toFixed(1)}, Focus ${quality.focus_score.toFixed(1)}`
          );
        }

        return quality;
      } catch (err: any) {
        console.error('Quality check failed:', err);
        // Handle service unavailable error gracefully
        if (err.response?.status === 503) {
          showMessage('⚠️ Quality check service unavailable - uploading without validation');
          console.warn('Quality validation service is unavailable (503)');
        } else {
          showMessage('⚠️ Quality check failed - uploading without validation');
        }
        return null; // Continue without quality check on error
      }
    },
    [showMessage, validateQuality]
  );

  const uploadImage = useCallback(
    async (file: Blob, filename: string) => {
      // Add to queue instead of blocking upload
      // Pass isTestCaptureMode to route to test endpoint if in calibration test mode
      setUploadQueue(prev => [...prev, {
        id: Date.now().toString() + Math.random(),
        blob: file,
        filename,
        options: { ...processingOptions },
        isTestCapture: isTestCaptureMode,  // Route to test endpoint in test mode
      }]);

      if (isTestCaptureMode) {
        showMessage(`🧪 Test capture added`);
        toast({
          title: '🧪 Test Capture',
          description: 'Image will appear in calibration settings',
          status: 'info',
          duration: 2000,
        });
      } else {
        showMessage(`✅ Added to processing queue`);
        toast({
          title: 'Queued for processing',
          description: 'You can continue capturing.',
          status: 'success',
          duration: 2000,
        });

        // Show additional message about checking dashboard
        setTimeout(() => {
          showMessage(
            '📊 Processing in background... Check Dashboard for results.'
          );
        }, 1500);
      }

      // Clear quality check
      setQualityCheck(null);
    },
    [
      processingOptions,
      isTestCaptureMode,
      showMessage,
      toast
    ]
  );

  const captureFromCamera = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d', { willReadFrequently: true });

    if (!context) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);

    // Check if the image is blank before processing
    const frameData = getCurrentFrameData();
    if (frameData && isBlankImage(frameData)) {
      toast({
        title: 'Blank Image Detected',
        description: 'No document content found - capture skipped',
        status: 'warning',
        duration: 3000,
      });
      return;
    }

    canvas.toBlob(
      async (blob: Blob | null) => {
        if (blob) {
          // NOTE: We do NOT set previewImage here to allow continuous capture
          // setPreviewImage(url);

          // Check quality before uploading (optional - could be skipped for speed)
          if (validateQuality) {
            const quality = await checkImageQuality(blob);
            if (
              quality === null &&
              qualityCheck &&
              !qualityCheck.quality.overall_acceptable
            ) {
              // If quality check failed and user cancelled, stop.
              return;
            }
          }

          uploadImage(blob, `capture_${Date.now()}.jpg`);
        }
      },
      'image/jpeg',
      0.9
    );
  }, [checkImageQuality, qualityCheck, uploadImage, validateQuality, toast]);

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      alert('Please select an image file');
      return;
    }

    const url = URL.createObjectURL(file);
    setPreviewImage(url);

    // Check quality before uploading
    const quality = await checkImageQuality(file);
    if (
      quality === null &&
      validateQuality &&
      qualityCheck &&
      !qualityCheck.quality.overall_acceptable
    ) {
      setPreviewImage(null);
      return;
    }

    uploadImage(file, file.name);
  };

  useEffect(() => {
    // Sync local connected state with socket context
    setConnected(socketConnected);
  }, [socketConnected]);

  useEffect(() => {
    console.log('Setting up Socket.IO event listeners');

    if (!socket) {
      return;
    }

    socket.on('capture_now', (data: any) => {
      console.log('Received capture command:', data);
      showMessage('📸 Capture triggered from Dashboard!');
      setTimeout(() => {
        if (captureMode === 'camera' && stream) {
          captureFromCamera();
        } else {
          showMessage('Switch to Camera mode to auto-capture');
        }
      }, 500);
    });



    // Handle auto-capture start - Phone applies the delay
    // This centralizes ALL delay logic in the phone view
    socket.on('start_auto_capture', (data: any) => {
      console.log('Received auto-capture command from Dashboard:', data);
      const documentCount = data?.documentCount || 1;
      // Use delay from Dashboard if provided, otherwise use local initialDelay
      const delaySeconds = data?.delaySeconds ?? initialDelay ?? 10;
      const delayMs = delaySeconds * 1000;

      console.log(`[AUTO-CAPTURE] delaySeconds from data: ${data?.delaySeconds}, initialDelay: ${initialDelay}, final: ${delaySeconds}`);

      // Ensure camera mode is active FIRST
      if (captureMode !== 'camera' || !stream) {
        showMessage('💡 Switching to Camera mode...');
        handleCaptureMode('camera');
      }

      // If delay is 0 or less, start immediately
      if (delaySeconds <= 0) {
        toast({
          title: '📱 Auto-Capture Enabled!',
          description: `Ready to capture ${documentCount} document${documentCount !== 1 ? 's' : ''}`,
          status: 'success',
          duration: 3000,
          isClosable: true,
          position: 'top',
        });
        setTimeout(() => startAutoCapture('dashboard', documentCount), 100);
        return;
      }

      // Show countdown toast
      toast({
        title: '📱 Auto-Capture Starting...',
        description: `Will enable in ${delaySeconds} seconds`,
        status: 'info',
        duration: delaySeconds * 1000,
        isClosable: true,
        position: 'top',
      });

      // Start countdown display
      setCountdown(delaySeconds);
      setPendingDocumentCount(documentCount);

      // Clear any existing countdown
      if (countdownIntervalRef.current) {
        clearInterval(countdownIntervalRef.current);
      }

      // Countdown interval for visual feedback
      countdownIntervalRef.current = setInterval(() => {
        setCountdown(prev => {
          if (prev === null || prev <= 1) {
            clearInterval(countdownIntervalRef.current!);
            countdownIntervalRef.current = null;
            return null;
          }
          return prev - 1;
        });
      }, 1000);

      // After delay, start auto-capture
      setTimeout(() => {
        // Clear countdown display
        setCountdown(null);
        if (countdownIntervalRef.current) {
          clearInterval(countdownIntervalRef.current);
          countdownIntervalRef.current = null;
        }

        // Show success toast
        toast({
          title: '📱 Auto-Capture Enabled!',
          description: `Ready to capture ${documentCount} document${documentCount !== 1 ? 's' : ''}`,
          status: 'success',
          duration: 3000,
          isClosable: true,
          position: 'top',
        });

        // Start auto-capture
        startAutoCapture('dashboard', documentCount);
      }, delayMs);
    });

    // Handle stop auto-capture from dashboard
    socket.on('stop_auto_capture', () => {
      console.log('Received stop auto-capture from Dashboard');
      stopAutoCapture();
      toast({
        title: '⏹️ Auto-Capture Stopped',
        description: 'Dashboard disabled auto-capture',
        status: 'info',
        duration: 2000,
      });
    });

    // Handle sync request from dashboard
    socket.on('request_auto_capture_state', () => {
      socket.emit('auto_capture_state_changed', {
        enabled: autoCapture,
        source: 'phone',
        capturedCount: autoCaptureCountRef.current
      });
    });

    // Handle calibration test mode from dashboard
    socket.on('calibration_test_mode', (data: { enabled: boolean }) => {
      console.log('[Phone] Calibration test mode:', data.enabled);
      setIsTestCaptureMode(data.enabled);
      if (data.enabled) {
        toast({
          title: '🧪 Test Mode Active',
          description: 'Captures will be used for calibration testing',
          status: 'info',
          duration: 3000,
        });
      }
    });

    // Handle test capture start from dashboard
    socket.on('start_test_capture', async (data: { delay: number; autoCapture?: boolean }) => {
      console.log('[Phone] Starting test capture countdown:', data);
      setIsTestCaptureMode(true);
      
      if (data.autoCapture) {
        // Switch to camera mode if not already
        if (captureMode !== 'camera' || !stream) {
          console.log('[Phone] Switching to camera mode for test');
          handleCaptureMode('camera');
          // Give camera time to initialize
          await new Promise(resolve => setTimeout(resolve, 500));
        }
        
        toast({
          title: '📸 Test Capture Starting',
          description: `Auto-capturing in ${data.delay}s to test calibration`,
          status: 'info',
          duration: data.delay * 1000,
        });
        
        // Use the calibration countdown
        setIsWaitingForInitialDelay(true);
        try {
          await startDelayCountdown();
          // Countdown complete - capture now
          console.log('[Phone] Test countdown complete - capturing');
          await captureFromCamera();
          toast({
            title: '✅ Test Capture Complete',
            description: 'Check calibration settings for preview',
            status: 'success',
            duration: 3000,
          });
          // Exit test mode after successful capture
          setTimeout(() => {
            setIsTestCaptureMode(false);
          }, 1000);
        } catch (e) {
          console.log('[Phone] Test capture cancelled');
          setIsTestCaptureMode(false);
        } finally {
          setIsWaitingForInitialDelay(false);
        }
      } else {
        toast({
          title: '📸 Test Capture Mode',
          description: `Capture any document now to test ${data.delay}s delay`,
          status: 'success',
          duration: 3000,
        });
      }
    });
    
    // Handle test capture cancellation
    socket.on('cancel_test_capture', () => {
      console.log('[Phone] Test capture cancelled');
      cancelCountdown();
      setIsWaitingForInitialDelay(false);
      setIsTestCaptureMode(false);
    });

    // Handle delay settings updates from Dashboard in real-time
    socket.on('delay_settings_updated', (data: { initialDelay: number; interCaptureDelay: number; timestamp: number }) => {
      console.log('[Phone] Delay settings updated from Dashboard:', data);
      // Update local context with new values
      setInitialDelay(data.initialDelay);
      setInterCaptureDelay(data.interCaptureDelay);
      toast({
        title: '⚙️ Settings Updated',
        description: `Delays: ${data.initialDelay}s startup, ${data.interCaptureDelay}s between captures`,
        status: 'info',
        duration: 3000,
        isClosable: true,
      });
    });

    return () => {
      socket.off('capture_now');
      socket.off('start_auto_capture');
      socket.off('stop_auto_capture');
      socket.off('request_auto_capture_state');
      socket.off('calibration_test_mode');
      socket.off('start_test_capture');
      socket.off('cancel_test_capture');
      socket.off('delay_settings_updated');
    };
  }, [
    socket,
    captureFromCamera,
    captureMode,
    showMessage,
    startAutoCapture,
    stopAutoCapture,
    handleCaptureMode,
    stream,
    toast,
    autoCapture,
    initialDelay,  // Added to ensure delay value is current
  ]);

  // Cleanup camera on unmount
  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      // Ensure auto-capture stops when component unmounts
      if (autoCaptureIntervalRef.current) {
        clearInterval(autoCaptureIntervalRef.current);
      }
      if (detectionRafRef.current) {
        cancelAnimationFrame(detectionRafRef.current);
        detectionRafRef.current = null;
      }
    };
  }, [stream]);

  return (
    <VStack align="stretch" spacing={10} pb={16}>
      <Flex direction={{ base: 'column', md: 'row' }} justify="space-between" gap={6}>
        <Stack spacing={2}>
          <Heading size="lg" display="flex" alignItems="center" gap={3}>
            📱 Phone Interface
          </Heading>
          <Text color={muted} maxW="lg">
            Deploy the mobile capture cockpit to ingest pristine documents with AI guidance.
          </Text>
        </Stack>
        <Flex
          align="center"
          gap={3}
          bg="surface.blur"
          borderRadius="full"
          border="1px solid rgba(121,95,238,0.22)"
          px={5}
          py={2}
        >
          <Box
            w={3}
            h={3}
            borderRadius="full"
            bg={connected ? 'green.400' : 'red.400'}
            boxShadow={`0 0 12px ${connected ? 'rgba(72,187,120,0.6)' : 'rgba(245,101,101,0.6)'}`}
          />
          <Text fontWeight="600" color={muted} display="flex" alignItems="center" gap={2}>
            <Iconify icon={FiWifi} boxSize={5} />
            {connected ? 'Connected to processing hub' : 'Link offline'}
          </Text>
        </Flex>

        {/* Queue Indicator */}
        {(uploadQueue.length > 0 || isProcessingQueue) && (
          <Flex
            align="center"
            gap={2}
            bg="blue.500"
            color="white"
            borderRadius="full"
            px={4}
            py={2}
            boxShadow="0 0 12px rgba(66, 153, 225, 0.6)"
            animation="pulse 2s infinite"
          >
            <Spinner size="xs" />
            <Text fontWeight="bold" fontSize="sm">
              Processing: {uploadQueue.length + (isProcessingQueue ? 1 : 0)}
            </Text>
          </Flex>
        )}
      </Flex>

      {message && (
        <Alert
          status="info"
          borderRadius="xl"
          bg="rgba(69,202,255,0.1)"
          border="1px solid rgba(69,202,255,0.25)"
        >
          <AlertIcon />
          <AlertDescription>{message}</AlertDescription>
        </Alert>
      )}

      <Card bg={panelBg} border="1px solid rgba(121,95,238,0.18)" boxShadow="subtle">
        <CardHeader>
          <Heading size="sm">Capture Preferences</Heading>
        </CardHeader>
        <CardBody>
          <Grid templateColumns={{ base: '1fr', md: 'repeat(3, 1fr)' }} gap={4}>
            <Flex
              align="center"
              justify="space-between"
              bg="surface.blur"
              borderRadius="lg"
              px={4}
              py={3}
              border="1px solid rgba(69,202,255,0.18)"
            >
              <Stack spacing={1}>
                <Text fontWeight="600">Quality validation</Text>
                <Text fontSize="sm" color={muted}>
                  Detect blur and focus issues before upload.
                </Text>
              </Stack>
              <Switch
                colorScheme="brand"
                isChecked={validateQuality}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                  setValidateQuality(e.target.checked)
                }
              />
            </Flex>

            <Flex
              align="center"
              justify="space-between"
              bg="surface.blur"
              borderRadius="lg"
              px={4}
              py={3}
              border="1px solid rgba(69,202,255,0.18)"
            >
              <Stack spacing={1}>
                <Text fontWeight="600">Auto crop</Text>
                <Text fontSize="sm" color={muted}>
                  Automatically align document edges.
                </Text>
              </Stack>
              <Switch
                colorScheme="brand"
                isChecked={processingOptions.autoCrop}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                  setProcessingOptions({ ...processingOptions, autoCrop: e.target.checked })
                }
              />
            </Flex>

            <Flex
              align="center"
              justify="space-between"
              bg="surface.blur"
              borderRadius="lg"
              px={4}
              py={3}
              border="1px solid rgba(69,202,255,0.18)"
            >
              <Stack spacing={1}>
                <Text fontWeight="600">AI enhancement</Text>
                <Text fontSize="sm" color={muted}>
                  Boost clarity with AI-driven retouching.
                </Text>
              </Stack>
              <Switch
                colorScheme="brand"
                isChecked={processingOptions.aiEnhance}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                  setProcessingOptions({ ...processingOptions, aiEnhance: e.target.checked })
                }
              />
            </Flex>
          </Grid>
        </CardBody>
      </Card>

      {qualityCheck && (
        <Card
          bg={panelBg}
          border={`1px solid ${qualityCheck.quality.overall_acceptable ? 'rgba(72,187,120,0.25)' : 'rgba(246,173,85,0.35)'}`}
          boxShadow="subtle"
        >
          <CardHeader display="flex" alignItems="center" justifyContent="space-between">
            <Heading size="sm" display="flex" alignItems="center" gap={2}>
              <Iconify
                icon={qualityCheck.quality.overall_acceptable ? FiCheckCircle : FiAlertTriangle}
                boxSize={5}
              />
              Quality Analysis
            </Heading>
            <Tag
              colorScheme={qualityCheck.quality.overall_acceptable ? 'green' : 'orange'}
              borderRadius="full"
            >
              {qualityCheck.quality.overall_acceptable ? 'Ready to upload' : 'Review suggested'}
            </Tag>
          </CardHeader>
          <CardBody>
            <Grid templateColumns={{ base: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' }} gap={4} mb={4}>
              <Flex
                direction="column"
                bg="surface.blur"
                borderRadius="lg"
                p={3}
                border="1px solid rgba(69,202,255,0.22)"
              >
                <Text fontSize="xs" textTransform="uppercase" color={muted}>
                  Blur score
                </Text>
                <Text fontWeight="700">{qualityCheck.blur_score.toFixed(1)}</Text>
                <Text fontSize="sm" color={qualityCheck.is_blurry ? 'orange.300' : 'green.300'}>
                  {qualityCheck.is_blurry ? 'Requires attention' : 'Crystal clear'}
                </Text>
              </Flex>
              <Flex
                direction="column"
                bg="surface.blur"
                borderRadius="lg"
                p={3}
                border="1px solid rgba(69,202,255,0.22)"
              >
                <Text fontSize="xs" textTransform="uppercase" color={muted}>
                  Focus score
                </Text>
                <Text fontWeight="700">{qualityCheck.focus_score.toFixed(1)}</Text>
                <Text fontSize="sm" color={qualityCheck.is_focused ? 'green.300' : 'orange.300'}>
                  {qualityCheck.is_focused ? 'Focused' : 'Adjust focus'}
                </Text>
              </Flex>
            </Grid>

            {qualityCheck.quality.issues.length > 0 && (
              <Stack spacing={2}>
                <Text fontWeight="600">Detected issues</Text>
                {qualityCheck.quality.issues.map((issue: string, index: number) => (
                  <Flex
                    key={index}
                    align="center"
                    gap={2}
                    bg="rgba(246,173,85,0.1)"
                    borderRadius="md"
                    px={3}
                    py={2}
                  >
                    <Iconify icon={FiAlertTriangle} color="orange.300" />
                    <Text color={muted}>{issue}</Text>
                  </Flex>
                ))}
              </Stack>
            )}
          </CardBody>
        </Card>
      )}

      <Card bg={panelBg} border="1px solid rgba(121,95,238,0.18)">
        <CardBody>
          <Stack spacing={8}>
            {/* Current Delay Settings Display */}
            <Flex
              bg={useColorModeValue('rgba(121, 95, 238, 0.08)', 'rgba(121, 95, 238, 0.15)')}
              borderRadius="lg"
              px={4}
              py={3}
              border="1px solid"
              borderColor={useColorModeValue('rgba(121, 95, 238, 0.2)', 'rgba(121, 95, 238, 0.3)')}
              alignItems="center"
              justifyContent="space-between"
              flexWrap="wrap"
              gap={3}
            >
              <Flex alignItems="center" gap={2}>
                <Iconify icon="solar:clock-circle-bold" boxSize={5} color="brand.400" />
                <Text fontSize="sm" fontWeight="600" color={useColorModeValue('gray.700', 'gray.200')}>
                  Auto-Capture Delays
                </Text>
              </Flex>
              <Flex gap={3} flexWrap="wrap">
                <Badge
                  colorScheme="purple"
                  fontSize="xs"
                  px={3}
                  py={1}
                  borderRadius="full"
                  display="flex"
                  alignItems="center"
                  gap={1}
                >
                  <Text>Startup:</Text>
                  <Text fontWeight="bold">{initialDelay}s</Text>
                </Badge>
                <Badge
                  colorScheme="orange"
                  fontSize="xs"
                  px={3}
                  py={1}
                  borderRadius="full"
                  display="flex"
                  alignItems="center"
                  gap={1}
                >
                  <Text>Between:</Text>
                  <Text fontWeight="bold">{interCaptureDelay}s</Text>
                </Badge>
              </Flex>
            </Flex>

            <ButtonGroup isAttached variant="ghost" alignSelf="center">
              <Button
                leftIcon={<Iconify icon={FiUpload} boxSize={5} />}
                colorScheme={captureMode === 'file' ? 'brand' : undefined}
                onClick={() => handleCaptureMode('file')}
              >
                Choose File
              </Button>
              <Button
                leftIcon={<Iconify icon={FiCamera} boxSize={5} />}
                colorScheme={captureMode === 'camera' ? 'brand' : undefined}
                onClick={() => handleCaptureMode('camera')}
              >
                Live Camera
              </Button>
            </ButtonGroup>

            {captureMode === 'file' ? (
              <Stack spacing={4} align="center">
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileSelect}
                  accept="image/*,image/jpeg,image/jpg,image/png"
                  capture="environment"
                  style={{ display: 'none' }}
                />
                <Button
                  size="lg"
                  colorScheme="brand"
                  leftIcon={<Iconify icon={FiUpload} boxSize={5} />}
                  onClick={() => fileInputRef.current?.click()}
                  isLoading={uploading}
                  loadingText="Uploading"
                >
                  Select Image
                </Button>
                <Text fontSize="sm" color={muted} textAlign="center">
                  Upload an existing document or snap a fresh capture from your device.
                </Text>
              </Stack>
            ) : (
              <Stack spacing={4}>
                {/* Camera Feed Container */}
                <Box
                  ref={cameraContainerRef}
                  position="relative"
                  borderRadius="xl"
                  overflow="hidden"
                  bg="black"
                  border="2px solid"
                  borderColor={detectedQuad ? "green.400" : "gray.600"}
                  transition="border-color 0.3s"
                  sx={{
                    width: '100%',
                    maxWidth: isFullScreen ? '100vw' : '100%',
                    height: isFullScreen ? '100vh' : 'auto',
                    aspectRatio: 'auto',
                    mx: 'auto',
                  }}
                >
                  {/* Video Element */}
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    style={{
                      width: '100%',
                      height: isFullScreen ? '100vh' : 'auto',
                      maxHeight: isFullScreen ? '100vh' : '70vh',
                      display: 'block',
                      backgroundColor: 'black',
                      objectFit: 'contain',
                    }}
                  />
                  <canvas ref={canvasRef} style={{ display: 'none' }} />

                  {/* TEST MODE BANNER - Visible when in calibration test mode */}
                  {isTestCaptureMode && !autoCapture && (
                    <Flex
                      position="absolute"
                      top={0}
                      left={0}
                      right={0}
                      bg="linear-gradient(90deg, rgba(139, 92, 246, 0.95) 0%, rgba(109, 40, 217, 0.95) 100%)"
                      color="white"
                      px={4}
                      py={3}
                      alignItems="center"
                      justifyContent="center"
                      zIndex={30}
                      boxShadow="0 4px 20px rgba(139, 92, 246, 0.4)"
                    >
                      <Flex alignItems="center" gap={3}>
                        <Box
                          w={4}
                          h={4}
                          borderRadius="full"
                          bg="white"
                          animation="pulse 1s infinite"
                          boxShadow="0 0 10px rgba(255,255,255,0.8)"
                        />
                        <Box textAlign="center">
                          <Text fontSize="sm" fontWeight="bold" lineHeight="1.2">
                            🧪 TEST CAPTURE MODE
                          </Text>
                          <Text fontSize="xs" opacity={0.9}>
                            Testing calibration delay settings
                          </Text>
                        </Box>
                      </Flex>
                    </Flex>
                  )}

                  {/* PROMINENT AUTO-CAPTURE ACTIVE BANNER - Always visible when auto-capture is ON */}
                  {autoCapture && (
                    <Flex
                      position="absolute"
                      top={0}
                      left={0}
                      right={0}
                      bg="linear-gradient(90deg, rgba(34, 197, 94, 0.95) 0%, rgba(22, 163, 74, 0.95) 100%)"
                      color="white"
                      px={4}
                      py={3}
                      alignItems="center"
                      justifyContent="space-between"
                      zIndex={30}
                      boxShadow="0 4px 20px rgba(34, 197, 94, 0.4)"
                    >
                      <Flex alignItems="center" gap={3}>
                        <Box
                          w={4}
                          h={4}
                          borderRadius="full"
                          bg="white"
                          animation="pulse 1s infinite"
                          boxShadow="0 0 10px rgba(255,255,255,0.8)"
                        />
                        <Box>
                          <Text fontSize="sm" fontWeight="bold" lineHeight="1.2">
                            🎯 AUTO-CAPTURE ACTIVE
                          </Text>
                          <Text fontSize="xs" opacity={0.9}>
                            {autoCaptureSource === 'dashboard' ? '📡 From Dashboard • ' : ''}
                            Place documents - they capture automatically
                          </Text>
                        </Box>
                      </Flex>
                      <Flex alignItems="center" gap={2}>
                        <Box
                          bg="whiteAlpha.300"
                          px={3}
                          py={1}
                          borderRadius="full"
                          fontSize="sm"
                          fontWeight="bold"
                        >
                          📷 {autoCaptureCount}
                        </Box>
                        <Button
                          size="sm"
                          colorScheme="red"
                          variant="solid"
                          onClick={stopAutoCapture}
                          leftIcon={<Text>⏹</Text>}
                        >
                          Stop
                        </Button>
                      </Flex>
                    </Flex>
                  )}

                  {/* Detection Status Badge - Moved down when auto-capture banner is showing */}
                  <Box
                    position="absolute"
                    top={autoCapture ? 16 : 3}
                    left={3}
                    bg={detectedQuad ? "green.500" : "gray.600"}
                    color="white"
                    px={3}
                    py={1}
                    borderRadius="full"
                    fontSize="xs"
                    fontWeight="bold"
                    display="flex"
                    alignItems="center"
                    gap={2}
                    transition="all 0.3s"
                  >
                    <Box
                      w={2}
                      h={2}
                      borderRadius="full"
                      bg={detectedQuad ? "green.200" : "gray.400"}
                      animation={detectedQuad ? "pulse 1s infinite" : "none"}
                    />
                    {detectedQuad ? "Document Detected" : "Searching..."}
                  </Box>

                  {/* Compact Delay Settings Display - Bottom left corner, always visible in fullscreen */}
                  {isFullScreen && (
                    <Flex
                      position="absolute"
                      bottom={3}
                      left={3}
                      bg="rgba(0, 0, 0, 0.7)"
                      backdropFilter="blur(10px)"
                      color="white"
                      px={3}
                      py={2}
                      borderRadius="lg"
                      fontSize="xs"
                      gap={2}
                      alignItems="center"
                      zIndex={20}
                      border="1px solid rgba(255, 255, 255, 0.1)"
                    >
                      <Iconify icon="solar:clock-circle-bold" boxSize={3} />
                      <Text fontWeight="600">{initialDelay}s</Text>
                      <Text opacity={0.7}>•</Text>
                      <Text fontWeight="600">{interCaptureDelay}s</Text>
                    </Flex>
                  )}

                  {/* Eye Toggle Button */}
                  <Tooltip label={showControls ? 'Hide controls' : 'Show controls'} hasArrow placement="left">
                    <Button
                      position="absolute"
                      top={autoCapture ? 16 : 3}
                      right={3}
                      size="sm"
                      colorScheme="brand"
                      variant={showControls ? 'solid' : 'ghost'}
                      onClick={() => setShowControls(!showControls)}
                      zIndex={10}
                      borderRadius="full"
                      p={2}
                      minWidth="auto"
                    >
                      <Iconify icon={showControls ? FiEye : FiEyeOff} boxSize={4} />
                    </Button>
                  </Tooltip>

                  {/* Countdown Overlay */}
                  {countdown !== null && (
                    <Flex
                      position="absolute"
                      top={0}
                      left={0}
                      right={0}
                      bottom={0}
                      align="center"
                      justify="center"
                      bg="rgba(0,0,0,0.8)"
                      zIndex={20}
                      flexDirection="column"
                    >
                      <Text
                        fontSize="8xl"
                        fontWeight="bold"
                        color="white"
                        textShadow="0 0 40px rgba(66, 153, 225, 1)"
                      >
                        {countdown}
                      </Text>
                      <Text fontSize="lg" color="white" mt={4}>📱 Auto-Capture Starting...</Text>
                      <Text fontSize="sm" color="whiteAlpha.700" mt={2}>
                        {pendingDocumentCount} document{pendingDocumentCount !== 1 ? 's' : ''} queued
                      </Text>
                      <Button
                        mt={4}
                        colorScheme="red"
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          if (countdownIntervalRef.current) {
                            clearInterval(countdownIntervalRef.current);
                            countdownIntervalRef.current = null;
                          }
                          setCountdown(null);
                          setPendingDocumentCount(0);
                          toast({ title: 'Cancelled', status: 'warning', duration: 2000 });
                        }}
                      >
                        Cancel
                      </Button>
                    </Flex>
                  )}

                  {/* Auto-Capture Startup Delay Overlay - Shows when waiting for phone's auto-capture to turn on */}
                  {isWaitingForInitialDelay && isCountingDown && (
                    <Flex
                      position="absolute"
                      top={0}
                      left={0}
                      right={0}
                      bottom={0}
                      align="center"
                      justify="center"
                      bg="rgba(121, 95, 238, 0.9)"
                      zIndex={25}
                      flexDirection="column"
                    >
                      <Box
                        w={32}
                        h={32}
                        borderRadius="full"
                        border="4px solid"
                        borderColor="white"
                        display="flex"
                        alignItems="center"
                        justifyContent="center"
                        mb={4}
                      >
                        <Text
                          fontSize="6xl"
                          fontWeight="bold"
                          color="white"
                        >
                          {countdownValue}
                        </Text>
                      </Box>
                      <Text fontSize="xl" color="white" fontWeight="bold">
                        ⏳ Auto-Capture Starting...
                      </Text>
                      <Text fontSize="md" color="whiteAlpha.800" mt={2} textAlign="center" maxW="80%">
                        Waiting for your phone's auto-capture feature to fully turn on...
                      </Text>
                      <Button
                        mt={6}
                        colorScheme="red"
                        variant="solid"
                        size="md"
                        onClick={() => {
                          cancelCountdown();
                          setIsWaitingForInitialDelay(false);
                        }}
                      >
                        Cancel
                      </Button>
                    </Flex>
                  )}

                  {/* Auto-capture Status */}
                  {autoCapture && (
                    <Box
                      position="absolute"
                      bottom={3}
                      left={3}
                      bg={
                        frameChangeStatus === 'captured' ? 'green.500' :
                          frameChangeStatus === 'ready' ? 'green.400' :
                            frameChangeStatus === 'detecting' ? 'orange.400' :
                              'blue.500'
                      }
                      color="white"
                      px={3}
                      py={1}
                      borderRadius="full"
                      fontSize="xs"
                      fontWeight="bold"
                      display="flex"
                      alignItems="center"
                      gap={2}
                    >
                      <Box w={2} h={2} borderRadius="full" bg="white" animation="pulse 1s infinite" />
                      {frameChangeStatus === 'captured' ? '✓ Captured!' :
                        frameChangeStatus === 'ready' ? '📸 Ready...' :
                          frameChangeStatus === 'detecting' ? '👀 Detecting...' :
                            `📷 ${autoCaptureCount} captured`}
                      {autoCaptureSource === 'dashboard' && (
                        <Tag size="sm" colorScheme="blue" ml={1}>Dashboard</Tag>
                      )}
                    </Box>
                  )}

                  {/* Fullscreen Controls */}
                  {isFullScreen && showControls && (
                    <VStack
                      position="absolute"
                      bottom={6}
                      left="50%"
                      transform="translateX(-50%)"
                      spacing={2}
                      zIndex={5}
                    >
                      <Flex gap={2}>
                        <Button
                          colorScheme={autoCapture ? 'red' : 'brand'}
                          size="lg"
                          onClick={() => autoCapture ? stopAutoCapture() : startAutoCapture('local')}
                          isDisabled={!stream || uploading}
                          leftIcon={<Iconify icon={FiAperture} boxSize={5} />}
                        >
                          {autoCapture ? `Stop (${autoCaptureCount})` : 'Auto'}
                        </Button>
                        <Button
                          colorScheme="brand"
                          size="lg"
                          onClick={captureFromCamera}
                          isDisabled={!stream || uploading || autoCapture}
                          isLoading={uploading}
                          leftIcon={<Iconify icon={FiCamera} boxSize={5} />}
                        >
                          Capture
                        </Button>
                      </Flex>
                    </VStack>
                  )}
                </Box>

                {/* Control Buttons - Normal Mode */}
                {!isFullScreen && showControls && (
                  <Flex wrap="wrap" gap={2} justify="center">
                    <Button
                      colorScheme="brand"
                      leftIcon={<Iconify icon={FiCamera} boxSize={5} />}
                      onClick={captureFromCamera}
                      isDisabled={!stream || uploading || autoCapture}
                      isLoading={uploading}
                      loadingText="Uploading"
                    >
                      Capture
                    </Button>
                    <Button
                      variant={autoCapture ? 'solid' : 'outline'}
                      colorScheme={autoCapture ? 'red' : 'orange'}
                      onClick={() => autoCapture ? stopAutoCapture() : startAutoCapture('local')}
                      isDisabled={!stream || uploading}
                      leftIcon={<Iconify icon={FiAperture} boxSize={5} />}
                    >
                      {autoCapture ? `Stop (${autoCaptureCount})` : 'Auto Capture'}
                    </Button>
                    <Button
                      variant="outline"
                      colorScheme="brand"
                      onClick={toggleFullScreen}
                      isDisabled={!stream}
                      leftIcon={<Iconify icon={isFullScreen ? FiMinimize2 : FiMaximize2} boxSize={5} />}
                    >
                      Fullscreen
                    </Button>
                  </Flex>
                )}

                {/* Auto-Capture Status Panel */}
                {autoCapture && showControls && !isFullScreen && (
                  <Flex
                    align="center"
                    gap={3}
                    bg="surface.blur"
                    borderRadius="lg"
                    border="1px solid rgba(69,202,255,0.2)"
                    px={4}
                    py={3}
                  >
                    <Iconify icon={FiCpu} color="brand.300" />
                    <Stack spacing={0}>
                      <Text fontWeight="600" fontSize="sm">
                        {frameChangeStatus === 'ready' ? '📸 New document detected - capturing...' :
                          frameChangeStatus === 'detecting' ? '👀 Waiting for stable document...' :
                            frameChangeStatus === 'captured' ? '✓ Captured! Place next document.' :
                              `📷 Captured ${autoCaptureCount} documents`}
                      </Text>
                      <Text fontSize="xs" color={muted}>
                        Place each document in view • Auto-captures when stable
                      </Text>
                    </Stack>
                  </Flex>
                )}
              </Stack>
            )}

            {previewImage && (
              <Stack spacing={3}>
                <Heading size="sm">Preview</Heading>
                <Box
                  borderRadius="2xl"
                  overflow="hidden"
                  border="1px solid rgba(69,202,255,0.25)"
                  boxShadow="subtle"
                >
                  <img
                    src={previewImage}
                    alt="Preview"
                    style={{ width: '100%', display: 'block' }}
                  />
                </Box>
              </Stack>
            )}

            <Stack spacing={3}>
              <Heading size="sm">How it works</Heading>
              <Stack spacing={2} color={muted} fontSize="sm">
                <Text>1. Choose between live capture or file upload for your documents.</Text>
                <Text>2. Enable quality validation to enforce blur and focus standards.</Text>
                <Text>3. Leverage auto-detection to align documents before sending.</Text>
                <Text>4. Once captured, files upload automatically to the Dashboard pipeline.</Text>
                <Text>5. Monitor processing progress and results in the Dashboard experience.</Text>
              </Stack>
            </Stack>
          </Stack>
        </CardBody>
      </Card>
      {/* Connection Validator Modal */}
      <ConnectionValidator
        isOpen={showConnectionValidator}
        onClose={() => setShowConnectionValidator(false)}
        videoRef={videoRef as React.RefObject<HTMLVideoElement>}
      />
    </VStack>
  );
};

export default Phone;
