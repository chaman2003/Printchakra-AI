import React, { useEffect, useState, useCallback, useMemo, useRef } from 'react';
import apiClient from '../apiClient';
import { useSocket } from '../context/SocketContext';
import { useCalibration } from '../context/CalibrationContext';
import {
  Badge,
  Box,
  Button,
  ButtonGroup,
  Card,
  CardBody,
  CardFooter,
  CardHeader,
  Checkbox,
  Drawer,
  DrawerBody,
  DrawerContent,
  DrawerFooter,
  DrawerHeader,
  DrawerOverlay,
  Flex,
  Grid,
  Heading,
  HStack,
  IconButton,
  Input,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  Progress,
  Radio,
  RadioGroup,
  Select,
  SimpleGrid,
  Spinner,
  Stack,
  Tag,
  Text,
  Tooltip,
  useColorModeValue,
  useDisclosure,
  useToast,
  VStack,
} from '@chakra-ui/react';
import {
  FiDownload,
  FiFileText,
  FiRefreshCw,
  FiTrash2,
  FiZoomIn,
  FiActivity,
} from 'react-icons/fi';
import { motion } from 'framer-motion';
import { API_BASE_URL, API_ENDPOINTS, getImageUrl } from '../config';
import { Iconify, FancySelect, ConnectionValidator } from '../components/common';
import { SmartConnectionStatusHandle } from '../components/common/SmartConnectionStatus';
import { VoiceAIChat, DocumentPreview, DashboardHeroCard, DashboardActionPanel, DeviceInfoPanel } from '../components';
import DocumentSelector, {
  DocumentSelectorHandle,
} from '../components/document/DocumentSelector';
import PageShell from '../components/layout/PageShell';
import { DashboardShell } from '../components/layout/DashboardRegions';
import { FileInfo, Document, OCRResult } from '../types';
import { processFileForPreview } from '../utils/pdfUtils';
import { parseVoiceDocumentSelection } from '../utils/voiceDocumentSelectionParser';
import { runOCR, getOCRResult, getBatchOCRStatus } from '../ocrApi';
import { OCRReadyBadge } from '../components/document/OCROverlay';
import OCRStructuredView from '../components/document/OCRStructuredView';

// ==================== MODAL & PREVIEW CONFIGURATION ====================
// 🔧 ADJUST THESE VALUES TO CONTROL MODAL AND PREVIEW SIZING
//
// If the preview is cut off or you want to change the modal size:
// 1. Adjust modal.maxHeight/maxWidth to change the overall modal size
// 2. Adjust modalBody.maxHeight to change scrollable content area
// 3. Adjust previewBox.maxHeight to change the sticky preview container height
//
const MODAL_CONFIG = {
  modal: {
    // Use calc with the viewport height minus top/bottom chrome so modal never exceeds viewport
    maxHeight: 'calc(100vh - 4rem)', // Maximum modal height (adjust to leave room for chrome)
    maxWidth: '95vw', // Maximum modal width (increase to make modal wider)
  },
  modalBody: {
    // Will be used as `calc(${MODAL_CONFIG.modalBody.maxHeight})` in ModalBody
    maxHeight: '100vh - 14rem', // Modal body max height (leave room for header/footer)
  },
  previewBox: {
    // Sticky preview box inside modal body
    maxHeight: '100vh - 18rem', // Preview box max height (increase if preview is cut off)
  },
};
// =======================================================================

const includeIfDefined = (value: any, key: string) =>
  value === undefined || value === null ? {} : { [key]: value };

const describeVoiceUpdates = (updates: Record<string, any>) => {
  if (!updates) {
    return '';
  }

  const labels: Record<string, string> = {
    color_mode: 'Color Mode',
    orientation: 'Orientation',
    paper_size: 'Paper Size',
    page_size: 'Page Size',
    paper_size_custom: 'Paper Size',
    margins: 'Margins',
    margins_custom: 'Margins',
    scale: 'Scale',
    scale_custom: 'Scale',
    copies: 'Copies',
    duplex: 'Duplex',
    quality: 'Quality',
    resolution: 'Resolution',
    format: 'Format',
    custom_range: 'Custom Range',
    pages: 'Pages',
    page_mode: 'Pages',
    pages_per_sheet: 'Pages Per Sheet',
    pages_per_sheet_custom: 'Pages Per Sheet',
    mode: 'Scan Mode',
    text_mode: 'Text Mode',
    printColorMode: 'Color Mode',
    printLayout: 'Orientation',
    printPaperSize: 'Paper Size',
    printPaperSizeCustom: 'Paper Size',
    printResolution: 'Resolution',
    printPages: 'Pages',
    printCustomRange: 'Custom Range',
    printScale: 'Scale',
    printScaleCustom: 'Scale',
    printMargins: 'Margins',
    printMarginsCustom: 'Margins',
    printPagesPerSheet: 'Pages Per Sheet',
    printPagesPerSheetCustom: 'Pages Per Sheet',
    printCopies: 'Copies',
    printDuplex: 'Duplex',
    printQuality: 'Quality',
    scanColorMode: 'Color Mode',
    scanLayout: 'Orientation',
    scanResolution: 'Resolution',
    scanResolutionCustom: 'Resolution',
    scanPaperSize: 'Paper Size',
    scanPaperSizeCustom: 'Paper Size',
    scanPageMode: 'Pages',
    scanCustomRange: 'Custom Range',
    scanFormat: 'Format',
    scanTextMode: 'Text Mode',
    scanMode: 'Scan Mode',
    scanQuality: 'Quality',
  };

  return Object.entries(updates)
    .map(([key, value]) => {
      const label = labels[key] || key;
      if (typeof value === 'boolean') {
        return `${label}: ${value ? 'On' : 'Off'}`;
      }
      return `${label}: ${value}`;
    })
    .join(', ');
};

// Motion components
const MotionBox = motion.create(Box);
const MotionCard = motion.create(Card);
const MotionModalContent = motion.create(ModalContent);
const MotionButton = motion.create(Button);
const MotionFlex = motion.create(Flex);

interface ProcessingProgress {
  filename?: string;
  step: number;
  total_steps: number;
  stage_name: string;
  message?: string;
}

type PreviewControlSource = 'manual' | 'voice';

interface PreviewControlState {
  docIndex: number;
  page: number;
  source: PreviewControlSource;
  token: number;
}

type OrchestrationAction = 'print' | 'scan';

type ColorModeOption = 'color' | 'bw' | 'grayscale';

const useImageWithHeaders = (imageUrl: string, refreshToken?: number) => {
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (!imageUrl) {
      return;
    }

    let revokeUrl: string | null = null;
    setLoading(true);
    setError(null);
    setBlobUrl(null); // Clear previous blob to show loading state

    const loadImage = async () => {
      try {
        const headers: Record<string, string> = {
          'X-Requested-With': 'XMLHttpRequest',
        };

        // Add ngrok bypass header if using ngrok tunnel
        if (API_BASE_URL.includes('ngrok') || API_BASE_URL.includes('loca.lt')) {
          headers['ngrok-skip-browser-warning'] = 'true';
        }

        // Add cache-buster to force fresh fetch (timestamp + refresh token)
        const cacheBuster = `_t=${Date.now()}_r=${refreshToken || 0}`;
        const url = imageUrl.includes('?')
          ? `${imageUrl}&${cacheBuster}`
          : `${imageUrl}?${cacheBuster}`;

        const response = await fetch(url, {
          headers,
          cache: 'no-store',
          mode: 'cors',
        });

        if (!response.ok) {
          console.error(`Failed to fetch image: ${response.status} ${response.statusText}`, url);
          throw new Error(`Failed to fetch image: ${response.status}`);
        }

        const blob = await response.blob();

        // Validate blob
        if (!blob || blob.size === 0) {
          throw new Error('Empty image response');
        }

        const objectUrl = URL.createObjectURL(blob);
        revokeUrl = objectUrl;
        setBlobUrl(objectUrl);
      } catch (err) {
        console.error('Image load error:', err, imageUrl);
        setError(err as Error);
        setBlobUrl(null);
      } finally {
        setLoading(false);
      }
    };

    loadImage();

    return () => {
      if (revokeUrl) {
        URL.revokeObjectURL(revokeUrl);
      }
    };
  }, [imageUrl, refreshToken]);

  return { blobUrl, loading, error };
};

// Image component that loads with proper headers
interface SecureImageProps {
  filename: string;
  alt: string;
  className?: string;
  onClick?: () => void;
  style?: React.CSSProperties;
  refreshToken?: number;
}

const SecureImage: React.FC<SecureImageProps> = ({ filename, alt, className, onClick, style, refreshToken }) => {
  // Try /processed first, fallback to /thumbnail
  const imageUrl = `${API_BASE_URL}${API_ENDPOINTS.processed}/${filename}`;
  const { blobUrl, loading, error } = useImageWithHeaders(imageUrl, refreshToken);
  const [thumbnailError, setThumbnailError] = useState(false);
  const [thumbnailLoaded, setThumbnailLoaded] = useState(false);

  // Build thumbnail URL with headers bypass for ngrok
  const thumbnailUrl = useMemo(() => {
    const base = `${API_BASE_URL}/thumbnail/${filename}`;
    const params = new URLSearchParams();
    params.set('_t', Date.now().toString());
    if (refreshToken) params.set('_r', refreshToken.toString());
    return `${base}?${params.toString()}`;
  }, [filename, refreshToken]);

  // Reset thumbnail states when filename or refreshToken changes
  useEffect(() => {
    setThumbnailError(false);
    setThumbnailLoaded(false);
  }, [filename, refreshToken]);

  if (loading) {
    return (
      <Flex
        direction="column"
        align="center"
        justify="center"
        className={className}
        style={style}
        minH="160px"
        bg="surface.blur"
        borderRadius="lg"
      >
        <Spinner size="lg" color="brand.400" />
        <Text mt={3} fontSize="sm" color="text.muted">
          Loading preview…
        </Text>
      </Flex>
    );
  }

  // If primary image loaded successfully, show it
  if (blobUrl && !error) {
    return (
      <Box
        as="img"
        src={blobUrl}
        alt={alt}
        className={className}
        onClick={onClick}
        style={style}
        cursor={onClick ? 'pointer' : 'default'}
        objectFit="cover"
        w="100%"
        h="100%"
        borderRadius="lg"
      />
    );
  }

  // Fallback: try thumbnail endpoint
  if (thumbnailError) {
    // Both failed - show placeholder
    return (
      <Flex
        direction="column"
        align="center"
        justify="center"
        className={className}
        style={style}
        minH="160px"
        bg="rgba(0,0,0,0.2)"
        borderRadius="lg"
        onClick={onClick}
        cursor={onClick ? 'pointer' : 'default'}
      >
        <Iconify icon={FiFileText} boxSize={8} color="text.muted" />
        <Text mt={2} fontSize="xs" color="text.muted">
          Preview unavailable
        </Text>
      </Flex>
    );
  }

  // Try loading thumbnail
  return (
    <Box position="relative" w="100%" h="100%">
      {!thumbnailLoaded && (
        <Flex
          position="absolute"
          inset={0}
          direction="column"
          align="center"
          justify="center"
          bg="surface.blur"
          borderRadius="lg"
          zIndex={1}
        >
          <Spinner size="md" color="brand.400" />
        </Flex>
      )}
      <Box
        as="img"
        src={thumbnailUrl}
        alt={alt}
        className={className}
        onClick={onClick}
        style={style}
        cursor={onClick ? 'pointer' : 'default'}
        objectFit="cover"
        w="100%"
        h="100%"
        borderRadius="lg"
        opacity={thumbnailLoaded ? 1 : 0}
        transition="opacity 0.2s"
        onLoad={() => setThumbnailLoaded(true)}
        onError={() => setThumbnailError(true)}
      />
    </Box>
  );
};

// Modal image component with proper header-based loading for ngrok/deployment
interface ModalImageWithHeadersProps {
  filename: string;
  alt: string;
  refreshToken?: number;
}

const ModalImageWithHeaders: React.FC<ModalImageWithHeadersProps> = ({ filename, alt, refreshToken }) => {
  const imageUrl = `${API_BASE_URL}${API_ENDPOINTS.processed}/${filename}`;
  const { blobUrl, loading, error } = useImageWithHeaders(imageUrl, refreshToken);

  if (loading) {
    return (
      <Flex
        direction="column"
        align="center"
        justify="center"
        minH="200px"
        minW="200px"
      >
        <Spinner size="xl" color="brand.400" thickness="4px" />
        <Text mt={4} fontSize="md" color="text.muted">
          Loading image…
        </Text>
      </Flex>
    );
  }

  if (error || !blobUrl) {
    return (
      <Flex
        direction="column"
        align="center"
        justify="center"
        minH="200px"
        minW="300px"
        bg="rgba(0,0,0,0.1)"
        borderRadius="lg"
        p={8}
      >
        <Text fontSize="lg" color="text.muted" textAlign="center">
          Image not found or deleted
        </Text>
        <Text fontSize="sm" color="text.muted" mt={2}>
          {filename}
        </Text>
      </Flex>
    );
  }

  return (
    <Box
      as="img"
      src={blobUrl}
      alt={alt}
      maxW="100%"
      maxH={{ base: 'calc(95vh - 120px)', md: 'calc(90vh - 120px)' }}
      w="auto"
      h="auto"
      objectFit="contain"
      borderRadius="lg"
    />
  );
};

const Dashboard: React.FC = () => {
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [refreshToken, setRefreshToken] = useState<number>(0); // Forces thumbnail re-renders
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [ocrText, setOcrText] = useState<string>('');
  const [selectedImageFile, setSelectedImageFile] = useState<string | null>(null);
  const [processingProgress, setProcessingProgress] = useState<ProcessingProgress | null>(null);
  const [connectionRetries, setConnectionRetries] = useState(0);

  // Auto-capture delays from calibration
  const { initialDelay, interCaptureDelay, isCalibrated, setInitialDelay, setInterCaptureDelay } = useCalibration();

  // PaddleOCR state - initialize from localStorage for persistence across page reloads
  const [ocrResults, setOcrResults] = useState<Record<string, OCRResult>>(() => {
    try {
      const saved = localStorage.getItem('printchakra_ocr_results');
      return saved ? JSON.parse(saved) : {};
    } catch (e) {
      console.warn('Failed to load OCR results from localStorage:', e);
      return {};
    }
  });
  const [ocrLoading, setOcrLoading] = useState<Record<string, boolean>>({});
  const [ocrProgress, setOcrProgress] = useState<Record<string, { elapsed: number; status: string }>>({});
  const [activeOCRView, setActiveOCRView] = useState<string | null>(null);

  // Persist OCR results to localStorage whenever they change
  useEffect(() => {
    try {
      localStorage.setItem('printchakra_ocr_results', JSON.stringify(ocrResults));
    } catch (e) {
      console.warn('Failed to save OCR results to localStorage:', e);
    }
  }, [ocrResults]);

  // File conversion state
  const [selectionMode, setSelectionMode] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [targetFormat, setTargetFormat] = useState<string>('pdf');
  const [converting, setConverting] = useState(false);
  const [conversionProgress, setConversionProgress] = useState<string>('');
  const [mergePdf, setMergePdf] = useState<boolean>(true); // New: merge PDF option
  const [customFilename, setCustomFilename] = useState<string>(''); // New: custom filename for merged PDF

  // Converted files state
  const [convertedFiles, setConvertedFiles] = useState<any[]>([]);
  const [processedConvertedDocs, setProcessedConvertedDocs] = useState<Document[]>([]);

  const currentDocumentOptions = useMemo(
    () =>
      files.map(file => ({
        filename: file.filename,
        size: file.size,
        type: file.mime_type?.includes('pdf') ? 'pdf' : 'image',
        thumbnailUrl: `${API_BASE_URL}/thumbnail/${file.filename}`,
        isProcessed: file.has_text,
      })),
    [files]
  );

  const convertedDocumentOptions = useMemo(
    () => processedConvertedDocs.length > 0 ? processedConvertedDocs :
      convertedFiles.map((file: any) => ({
        filename: file.filename,
        size: file.size || 0,
        type: 'pdf',
        thumbnailUrl: `${API_BASE_URL}/thumbnail/${file.filename}`,
        isProcessed: true,
      })),
    [processedConvertedDocs, convertedFiles]
  );

  // Smart selection state - simple multi-select
  const [lastClickedIndex, setLastClickedIndex] = useState<number | null>(null);
  const [rangeStart, setRangeStart] = useState<number | null>(null);
  const [rangeEnd, setRangeEnd] = useState<number | null>(null);

  // Orchestrate Print & Capture state
  const [orchestrateStep, setOrchestrateStep] = useState<number>(1); // 1=mode, 2=options, 3=confirm
  const [orchestrateMode, setOrchestrateMode] = useState<'scan' | 'print' | null>(null);
  const orchestrateModeRef = React.useRef<'scan' | 'print' | null>(null);
  // Keep ref in sync with state
  React.useEffect(() => {
    orchestrateModeRef.current = orchestrateMode;
  }, [orchestrateMode]);
  // For scan mode: track whether user wants to select/upload documents or use feed tray
  const [scanDocumentSource, setScanDocumentSource] = useState<'select' | 'feed' | null>(null);
  // Track if user manually closed document selector to prevent auto-reopening
  const [userClosedDocSelector, setUserClosedDocSelector] = useState(false);
  const [orchestrateOptions, setOrchestrateOptions] = useState({
    // Scan options
    scanMode: 'single' as 'single' | 'multi',
    scanTextMode: false,
    scanPageMode: 'all' as 'all' | 'odd' | 'even' | 'custom',
    scanCustomRange: '',
    scanLayout: 'portrait' as 'portrait' | 'landscape',
    scanPaperSize: 'A4' as string,
    scanPaperSizeCustom: '',
    scanResolution: '300' as string,
    scanResolutionCustom: '',
    scanColorMode: 'color' as ColorModeOption,
    scanFormat: 'pdf' as string,
    scanQuality: 'normal' as string,
    // Print options
    printPages: 'all' as 'all' | 'odd' | 'even' | 'custom',
    printCustomRange: '',
    printLayout: 'portrait' as 'portrait' | 'landscape',
    printPaperSize: 'A4' as string,
    printPaperSizeCustom: '',
    printResolution: '300' as string,
    printColorMode: 'color' as ColorModeOption,
    printScale: '100' as string,
    printScaleCustom: '',
    printMargins: 'default' as 'default' | 'narrow' | 'none',
    printMarginsCustom: '',
    printPagesPerSheet: '1' as string,
    printPagesPerSheetCustom: '',
    printCopies: '1' as string,
    printDuplex: false,
    printQuality: 'normal' as string,
    printFiles: [] as File[],
    printConvertedFiles: [] as string[],
    // Default settings
    saveAsDefault: false,
  });

  const toast = useToast();
  const imageModal = useDisclosure();
  const conversionModal = useDisclosure();
  const convertedDrawer = useDisclosure();
  const orchestrateModal = useDisclosure();
  const documentSelectorModal = useDisclosure(); // Document selector modal
  const deviceInfoModal = useDisclosure(); // Device info modal
  const justConfirmedSelectionRef = useRef(false);
  const confirmationTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const markSelectionConfirmed = useCallback(() => {
    justConfirmedSelectionRef.current = true;
    if (confirmationTimerRef.current) {
      clearTimeout(confirmationTimerRef.current);
    }
    confirmationTimerRef.current = setTimeout(() => {
      justConfirmedSelectionRef.current = false;
      confirmationTimerRef.current = null;
    }, 800);
  }, []);

  // Socket connection for real-time updates
  const { socket, connected: socketConnected, reconnect: socketReconnect } = useSocket();
  const socketRef = React.useRef(socket);
  useEffect(() => {
    socketRef.current = socket;
  }, [socket]);

  // Document feeder state for scan workflow
  const [isFeedingDocuments, setIsFeedingDocuments] = useState(false);
  const [documentsFed, setDocumentsFed] = useState(false);
  const [feedCount, setFeedCount] = useState(0);
  const [documentsToFeed, setDocumentsToFeed] = useState(1); // Number of documents to feed
  const [autoCaptureEnabled, setAutoCaptureEnabled] = useState(false); // Track auto-capture state

  // Selected documents for orchestrate modal
  const [selectedDocuments, setSelectedDocuments] = useState<any[]>([]);
  const [previewControl, setPreviewControl] = useState<PreviewControlState>(() => ({
    docIndex: 0,
    page: 1,
    source: 'manual',
    token: Date.now(),
  }));

  const bumpPreviewFocus = useCallback(
    (updates: Partial<Omit<PreviewControlState, 'token'>>) => {
      setPreviewControl(prev => ({
        docIndex: updates.docIndex ?? prev.docIndex,
        page: updates.page ?? prev.page,
        source: updates.source ?? prev.source,
        token: Date.now(),
      }));
    },
    []
  );

  useEffect(() => {
    if (selectedDocuments.length === 0) {
      setPreviewControl({
        docIndex: 0,
        page: 1,
        source: 'manual',
        token: Date.now(),
      });
      return;
    }

    setPreviewControl(prev => {
      if (prev.docIndex <= selectedDocuments.length - 1) {
        return prev;
      }
      return {
        ...prev,
        docIndex: 0,
        page: 1,
        token: Date.now(),
      };
    });
  }, [selectedDocuments.length]);

  // Connection status modal state
  const connectivityModal = useDisclosure();
  const connectionStatusRef = React.useRef<SmartConnectionStatusHandle>(null);

  // Ref for scrolling the modal body
  const modalBodyRef = React.useRef<HTMLDivElement>(null);
  const voiceOrchestrationTriggerRef = React.useRef<
    ((mode: OrchestrationAction, config?: any) => void) | null
  >(null);
  const lastOrchestrationUpdateRef = React.useRef<string | null>(null);
  const documentSelectorRef = React.useRef<DocumentSelectorHandle | null>(null);
  const voiceDocumentSelectionRef = React.useRef<{ section: 'current' | 'converted' | 'upload'; index: number }>(
    {
      section: 'current',
      index: 1,
    }
  );
  // Refs for async callbacks that are defined later
  const executePrintJobRef = React.useRef<(() => Promise<void>) | null>(null);
  const executeScanJobRef = React.useRef<(() => Promise<void>) | null>(null);

  const [isChatVisible, setIsChatVisible] = useState(false); // Chat hidden by default
  const [chatWidth, setChatWidth] = useState<number>(() => {
    // Load saved width from localStorage or default to 380px
    const saved = localStorage.getItem('printchakra_chat_width');
    return saved ? Math.min(Math.max(parseInt(saved, 10), 280), 600) : 380;
  });
  const [isResizingChat, setIsResizingChat] = useState(false);
  const chatResizeRef = React.useRef<{ startX: number; startWidth: number } | null>(null);
  const [orchestrationContext, setOrchestrationContext] = useState<'manual' | 'voice'>('manual');
  const isVoiceOrchestration = orchestrationContext === 'voice';

  useEffect(() => {
    if (!orchestrateModal.isOpen && orchestrationContext === 'voice') {
      setOrchestrationContext('manual');
    }
  }, [orchestrateModal.isOpen, orchestrationContext]);

  useEffect(() => {
    if (orchestrationContext === 'voice' && !isChatVisible) {
      setIsChatVisible(true);
    }
  }, [orchestrationContext, isChatVisible]);

  // Process converted files with extracted pages or PDF.js fallback
  useEffect(() => {
    const processConvertedFiles = async () => {
      if (!convertedFiles || convertedFiles.length === 0) {
        setProcessedConvertedDocs([]);
        return;
      }

      try {
        const processed: Document[] = [];

        for (const file of convertedFiles) {
          try {
            // Check if this file has extracted pages
            const baseName = file.filename.replace(/\.[^.]+$/, '');
            const pagesInfo: any[] = [];
            let thumbnailUrl = '';

            // If file has pages info from backend, use it
            if (file.pages && Array.isArray(file.pages) && file.pages.length > 0) {
              console.log(`[Dashboard] Using pre-extracted pages for: ${file.filename}`);

              // Use extracted pages directly
              for (const page of file.pages) {
                pagesInfo.push({
                  pageNumber: parseInt(page.filename.match(/page_(\d+)/)?.[1] || '1'),
                  thumbnailUrl: page.url || `${API_BASE_URL}/converted-page/${page.path}`,
                  width: 0,
                  height: 0,
                });
              }

              // First page as thumbnail
              thumbnailUrl = pagesInfo[0]?.thumbnailUrl || '';

              processed.push({
                filename: file.filename,
                size: file.size || 0,
                type: 'pdf',
                thumbnailUrl,
                pages: pagesInfo,
              });

              console.log(`[Dashboard] Loaded ${pagesInfo.length} extracted pages for: ${file.filename}`);
            } else {
              // Fallback: try to fetch and convert with PDF.js
              console.log(`[Dashboard] No extracted pages found, using PDF.js for: ${file.filename}`);

              const response = await fetch(`${API_BASE_URL}/converted/${file.filename}`);
              if (!response.ok) {
                console.warn(`Failed to fetch converted file: ${file.filename}`);
                // Use backend thumbnail as last resort
                processed.push({
                  filename: file.filename,
                  size: file.size || 0,
                  type: 'pdf',
                  thumbnailUrl: `${API_BASE_URL}/thumbnail/${file.filename}`,
                });
                continue;
              }

              const blob = await response.blob();
              const processedFile = new File([blob], file.filename, { type: 'application/pdf' });

              // Process with PDF.js
              const result = await processFileForPreview(processedFile);

              processed.push({
                filename: result.filename,
                size: result.size,
                type: result.type,
                fileObject: result.fileObject,
                thumbnailUrl: result.thumbnailUrl,
                pages: result.pages,
              });

              console.log(`[Dashboard] Processed with PDF.js: ${file.filename}`);
            }
          } catch (error) {
            console.error(`[Dashboard] Error processing converted file ${file.filename}:`, error);
            // Fallback to backend thumbnail
            processed.push({
              filename: file.filename,
              size: file.size || 0,
              type: 'pdf',
              thumbnailUrl: `${API_BASE_URL}/thumbnail/${file.filename}`,
            });
          }
        }

        setProcessedConvertedDocs(processed);
      } catch (error) {
        console.error('[Dashboard] Error processing converted files:', error);
      }
    };

    processConvertedFiles();
  }, [convertedFiles]);

  const legacyVoiceKeyMap = useMemo<Record<OrchestrationAction, Record<string, string>>>(
    () => ({
      print: {
        colorMode: 'color_mode',
        color_mode: 'color_mode',
        layout: 'orientation',
        orientation: 'orientation',
        paperSize: 'paper_size',
        pageSize: 'paper_size',
        paper_size: 'paper_size',
        paperSizeCustom: 'paper_size_custom',
        paper_size_custom: 'paper_size_custom',
        resolution: 'resolution',
        pages: 'pages',
        customRange: 'custom_range',
        custom_range: 'custom_range',
        scale: 'scale',
        scaleCustom: 'scale_custom',
        scale_custom: 'scale_custom',
        margins: 'margins',
        marginsCustom: 'margins_custom',
        pagesPerSheet: 'pages_per_sheet',
        pages_per_sheet: 'pages_per_sheet',
        pagesPerSheetCustom: 'pages_per_sheet_custom',
        pages_per_sheet_custom: 'pages_per_sheet_custom',
        copies: 'copies',
        duplex: 'duplex',
        quality: 'quality',
      },
      scan: {
        colorMode: 'color_mode',
        color_mode: 'color_mode',
        layout: 'orientation',
        orientation: 'orientation',
        paperSize: 'paper_size',
        pageSize: 'paper_size',
        page_size: 'paper_size',
        paper_size: 'paper_size',
        paperSizeCustom: 'paper_size_custom',
        paper_size_custom: 'paper_size_custom',
        resolution: 'resolution',
        scanResolution: 'resolution',
        resolutionCustom: 'resolution_custom',
        resolution_custom: 'resolution_custom',
        format: 'format',
        scanFormat: 'format',
        mode: 'mode',
        scanMode: 'mode',
        pageMode: 'page_mode',
        scanPageMode: 'page_mode',
        customRange: 'custom_range',
        scanCustomRange: 'custom_range',
        textMode: 'text_mode',
        scanTextMode: 'text_mode',
        quality: 'quality',
      },
    }),
    []
  );

  const normalizeVoiceUpdates = useCallback(
    (actionType: OrchestrationAction | undefined, updates: Record<string, any> = {}) => {
      if (!actionType || !updates) {
        return {};
      }

      if (actionType === 'print') {
        return {
          ...includeIfDefined(updates.color_mode, 'printColorMode'),
          ...includeIfDefined(updates.orientation, 'printLayout'),
          ...includeIfDefined(updates.paper_size, 'printPaperSize'),
          ...includeIfDefined(updates.paper_size_custom, 'printPaperSizeCustom'),
          ...includeIfDefined(updates.margins, 'printMargins'),
          ...includeIfDefined(updates.margins_custom, 'printMarginsCustom'),
          ...includeIfDefined(
            updates.scale !== undefined && updates.scale !== null
              ? String(updates.scale)
              : undefined,
            'printScale'
          ),
          ...includeIfDefined(updates.scale_custom, 'printScaleCustom'),
          ...includeIfDefined(
            updates.resolution !== undefined && updates.resolution !== null
              ? String(updates.resolution)
              : undefined,
            'printResolution'
          ),
          ...includeIfDefined(updates.pages, 'printPages'),
          ...includeIfDefined(updates.custom_range, 'printCustomRange'),
          ...includeIfDefined(
            updates.copies !== undefined && updates.copies !== null
              ? String(updates.copies)
              : undefined,
            'printCopies'
          ),
          ...includeIfDefined(
            typeof updates.duplex === 'boolean' ? updates.duplex : undefined,
            'printDuplex'
          ),
          ...includeIfDefined(
            updates.pages_per_sheet !== undefined && updates.pages_per_sheet !== null
              ? String(updates.pages_per_sheet)
              : undefined,
            'printPagesPerSheet'
          ),
          ...includeIfDefined(updates.pages_per_sheet_custom, 'printPagesPerSheetCustom'),
          ...includeIfDefined(updates.quality, 'printQuality'),
        };
      }

      return {
        ...includeIfDefined(updates.color_mode, 'scanColorMode'),
        ...includeIfDefined(updates.orientation, 'scanLayout'),
        ...includeIfDefined(
          updates.resolution !== undefined && updates.resolution !== null
            ? String(updates.resolution)
            : undefined,
          'scanResolution'
        ),
        ...includeIfDefined(updates.page_size || updates.paper_size, 'scanPaperSize'),
        ...includeIfDefined(updates.paper_size_custom, 'scanPaperSizeCustom'),
        ...includeIfDefined(updates.format, 'scanFormat'),
        ...includeIfDefined(updates.page_mode, 'scanPageMode'),
        ...includeIfDefined(updates.custom_range, 'scanCustomRange'),
        ...includeIfDefined(
          updates.resolution_custom !== undefined && updates.resolution_custom !== null
            ? String(updates.resolution_custom)
            : undefined,
          'scanResolutionCustom'
        ),
        ...includeIfDefined(
          typeof updates.text_mode === 'boolean' ? updates.text_mode : undefined,
          'scanTextMode'
        ),
        ...includeIfDefined(updates.mode, 'scanMode'),
        ...includeIfDefined(updates.quality, 'scanQuality'),
      };
    },
    []
  );

  const applyVoiceConfigurationUpdates = useCallback(
    (
      actionType: OrchestrationAction | undefined,
      updates: Record<string, any>,
      normalizedOverride?: Record<string, any>
    ) => {
      if (
        !actionType ||
        ((!updates || Object.keys(updates).length === 0) &&
          (!normalizedOverride || Object.keys(normalizedOverride).length === 0))
      ) {
        return false;
      }

      const normalized =
        normalizedOverride && Object.keys(normalizedOverride).length > 0
          ? normalizedOverride
          : normalizeVoiceUpdates(actionType, updates);

      if (!normalized || Object.keys(normalized).length === 0) {
        return false;
      }

      setOrchestrateOptions(prev => ({
        ...prev,
        ...normalized,
      }));

      const summarySource =
        updates && Object.keys(updates).length > 0 ? updates : normalized;
      const summary = describeVoiceUpdates(summarySource);
      if (summary) {
        setVoiceAdjustmentLog(prev => {
          const next = [
            {
              timestamp: new Date().toISOString(),
              summary,
              action: actionType,
            },
            ...prev,
          ];
          return next.slice(0, 6);
        });
      }

      return true;
    },
    [normalizeVoiceUpdates, setOrchestrateOptions]
  );

  const resolveInitialVoiceConfigOptions = useCallback(
    (mode: OrchestrationAction, config?: any) => {
      if (!config || typeof config !== 'object') {
        return {};
      }

      if (config.options && typeof config.options === 'object') {
        return config.options;
      }

      const prefixedKey = Object.keys(config).some(key =>
        key.startsWith(mode === 'print' ? 'print' : 'scan')
      );
      if (prefixedKey) {
        return config;
      }

      const mapping = legacyVoiceKeyMap[mode];
      const snakeSource: Record<string, any> = {};

      Object.entries(config).forEach(([key, value]) => {
        if (value === undefined || value === null) {
          return;
        }
        const mappedKey = mapping?.[key];
        if (mappedKey) {
          snakeSource[mappedKey] = value;
        }
      });

      if (Object.keys(snakeSource).length === 0) {
        return {};
      }

      return normalizeVoiceUpdates(mode, snakeSource);
    },
    [legacyVoiceKeyMap, normalizeVoiceUpdates]
  );

  const handleOrchestrationUpdate = useCallback(
    (payload: any) => {
      if (!payload?.type) {
        return;
      }

      const eventId =
        payload.timestamp || `${payload.type}-${JSON.stringify(payload.updates ?? payload.result ?? '')}`;
      if (lastOrchestrationUpdateRef.current === eventId) {
        return;
      }
      lastOrchestrationUpdateRef.current = eventId;

      switch (payload.type) {
        case 'voice_configuration_updated': {
          const actionType = payload.action_type as OrchestrationAction | undefined;
          const updates = payload.updates || {};
          const applied = applyVoiceConfigurationUpdates(
            actionType,
            updates,
            payload.frontend_updates
          );
          if (applied) {
            if (!orchestrateModal.isOpen) {
              orchestrateModal.onOpen();
            }
            const described =
              describeVoiceUpdates(
                Object.keys(updates).length > 0 ? updates : payload.frontend_updates || {}
              ) || 'settings updated';
            toast({
              title: 'Voice Settings Applied',
              description: `Updated via voice: ${described}`,
              status: 'success',
              duration: 3000,
            });
          }
          break;
        }
        case 'configuration_updated': {
          const actionType = payload.action_type as OrchestrationAction | undefined;
          const normalized =
            payload.frontend_updates || payload.frontend_state?.options || {};
          applyVoiceConfigurationUpdates(actionType, payload.updates || {}, normalized);
          break;
        }
        case 'configuration_complete': {
          // Only advance to step 3 if we're actually at step 2 configuration
          // AND we haven't just confirmed document selection
          if (justConfirmedSelectionRef.current) {
            console.log('[configuration_complete] Ignoring - just confirmed document selection');
            // Don't advance, user is just confirming documents not configuration
            break;
          }
          if (orchestrateStep !== 2) {
            console.log('[configuration_complete] Ignoring - not at step 2, current step:', orchestrateStep);
            break;
          }
          console.log('[configuration_complete] Advancing to step 3 review');
          setOrchestrateStep(3);
          if (!orchestrateModal.isOpen) {
            orchestrateModal.onOpen();
          }
          toast({
            title: 'Configuration Complete',
            description: 'Voice assistant finished adjusting your settings.',
            status: 'info',
            duration: 3000,
          });
          break;
        }
        case 'voice_command_detected': {
          const mode = payload.intent as OrchestrationAction | undefined;
          if (mode === 'print' || mode === 'scan') {
            const shouldOpen = payload.open_ui ?? payload.result?.open_ui;
            if (shouldOpen && voiceOrchestrationTriggerRef.current) {
              const configPayload =
                payload.frontend_state ||
                payload.result?.frontend_state ||
                payload.result?.configuration;
              voiceOrchestrationTriggerRef.current(mode, configPayload);
            }
            if (payload.skip_mode_selection || payload.result?.skip_mode_selection) {
              setOrchestrateStep(2);
            }
          }
          break;
        }
        default:
          break;
      }
    },
    [applyVoiceConfigurationUpdates, orchestrateModal, setOrchestrateStep, toast]
  );

  // Load saved defaults from localStorage on mount
  useEffect(() => {
    const savedDefaults = localStorage.getItem('printchakra_orchestrate_defaults');
    if (savedDefaults) {
      try {
        const parsed = JSON.parse(savedDefaults);
        setOrchestrateOptions((prev: typeof orchestrateOptions) => ({
          ...prev,
          ...parsed,
          // Don't restore files and saveAsDefault flag
          printFiles: [],
          printConvertedFiles: [],
          saveAsDefault: false,
        }));
      } catch (e) {
        console.error('Failed to parse saved defaults:', e);
      }
    }
  }, []);

  useEffect(() => {
    return () => {
      if (confirmationTimerRef.current) {
        clearTimeout(confirmationTimerRef.current);
      }
    };
  }, []);

  // Auto-open document selector when entering Print Step 2 with no documents selected
  // Only auto-open once, not if user manually closed it
  useEffect(() => {
    if (
      orchestrateStep === 2 &&
      orchestrateMode === 'print' &&
      selectedDocuments.length === 0 &&
      orchestrateModal.isOpen &&
      !userClosedDocSelector &&
      !documentSelectorModal.isOpen &&
      !justConfirmedSelectionRef.current
    ) {
      // Small delay to allow the modal body to render
      const timer = setTimeout(() => {
        documentSelectorModal.onOpen();
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [orchestrateStep, orchestrateMode, selectedDocuments.length, orchestrateModal.isOpen, userClosedDocSelector, documentSelectorModal]);

  // Reset userClosedDocSelector when leaving print mode or step 2
  useEffect(() => {
    if (orchestrateStep !== 2 || orchestrateMode !== 'print') {
      setUserClosedDocSelector(false);
    }
  }, [orchestrateStep, orchestrateMode]);

  // Save defaults to localStorage when saveAsDefault is checked
  const saveDefaultSettings = () => {
    const settingsToSave = {
      scanMode: orchestrateOptions.scanMode,
      scanTextMode: orchestrateOptions.scanTextMode,
      scanPageMode: orchestrateOptions.scanPageMode,
      scanLayout: orchestrateOptions.scanLayout,
      scanPaperSize: orchestrateOptions.scanPaperSize,
      scanFormat: orchestrateOptions.scanFormat,
      scanResolution: orchestrateOptions.scanResolution,
      scanColorMode: orchestrateOptions.scanColorMode,
      printPages: orchestrateOptions.printPages,
      printLayout: orchestrateOptions.printLayout,
      printPaperSize: orchestrateOptions.printPaperSize,
      printScale: orchestrateOptions.printScale,
      printMargins: orchestrateOptions.printMargins,
      printCopies: orchestrateOptions.printCopies,
      printDuplex: orchestrateOptions.printDuplex,
      printQuality: orchestrateOptions.printQuality,
    };
    localStorage.setItem('printchakra_orchestrate_defaults', JSON.stringify(settingsToSave));
    toast({
      title: '💾 Settings Saved',
      description: 'Your default settings have been saved locally',
      status: 'success',
      duration: 3000,
      isClosable: true,
    });
  };

  // Fetch document info including pages for PDFs
  const fetchDocumentInfo = useCallback(async (filename: string) => {
    try {
      const response = await apiClient.get(`/document/info/${filename}`);
      const docInfo = response.data;

      // Prepend API_BASE_URL to thumbnail URLs if they're relative paths
      if (docInfo.pages) {
        docInfo.pages = docInfo.pages.map((page: any) => ({
          ...page,
          thumbnailUrl: page.thumbnailUrl.startsWith('http')
            ? page.thumbnailUrl
            : `${API_BASE_URL}${page.thumbnailUrl}`
        }));
      }

      return docInfo;
    } catch (error) {
      console.error('Error fetching document info:', error);
      // Return basic structure as fallback
      return {
        filename: filename,
        file_type: filename.split('.').pop()?.toLowerCase() || 'unknown',
        pages: [{
          pageNumber: 1,
          thumbnailUrl: `${API_BASE_URL}/thumbnail/${filename}`
        }]
      };
    }
  }, []);

  // Enhance selected documents with page information
  const enhanceDocumentsWithPages = useCallback(async (docs: any[]) => {
    const enhanced = await Promise.all(
      docs.map(async (doc) => {
        // If document already has pages (from client-side PDF.js conversion), use them
        if (doc.pages && doc.pages.length > 0) {
          return doc;
        }

        // For server-side documents, fetch page information
        try {
          const docInfo = await fetchDocumentInfo(doc.filename);
          return {
            ...doc,
            pages: docInfo.pages || [{
              pageNumber: 1,
              thumbnailUrl: doc.thumbnailUrl || `${API_BASE_URL}/thumbnail/${doc.filename}`
            }]
          };
        } catch (error) {
          console.error(`Error fetching page info for ${doc.filename}:`, error);
          return {
            ...doc,
            pages: [{
              pageNumber: 1,
              thumbnailUrl: doc.thumbnailUrl || `${API_BASE_URL}/thumbnail/${doc.filename}`
            }]
          };
        }
      })
    );
    return enhanced;
  }, [fetchDocumentInfo]);

  // File cache system - stores file metadata and counts
  // Use refs to avoid rerunning useEffect hooks when cache metadata updates
  const filesCacheRef = React.useRef<{ data: any[] | null; lastCount: number; timestamp: number }>(
    {
      data: null,
      lastCount: 0,
      timestamp: 0,
    }
  );

  const convertedFilesCacheRef = React.useRef<{ data: any[] | null; lastCount: number; timestamp: number }>(
    {
      data: null,
      lastCount: 0,
      timestamp: 0,
    }
  );

  const [voiceAdjustmentLog, setVoiceAdjustmentLog] = useState<
    Array<{ timestamp: string; summary: string; action: OrchestrationAction }>
  >([]);

  const renderVoiceAdjustmentPanel = useCallback(
    (mode: OrchestrationAction) => {
      const entries = voiceAdjustmentLog
        .filter(entry => entry.action === mode)
        .slice(0, 4);

      if (entries.length === 0) {
        return null;
      }

      return (
        <Box
          border="1px solid"
          borderColor="rgba(121,95,238,0.35)"
          bg="rgba(121,95,238,0.08)"
          borderRadius="xl"
          p={4}
          mb={4}
        >
          <HStack justify="space-between" mb={3}>
            <Heading size="sm">Recent AI adjustments</Heading>
            <Badge colorScheme="brand" variant="subtle">
              {mode === 'print' ? 'Print' : 'Scan'}
            </Badge>
          </HStack>
          <Stack spacing={2}>
            {entries.map(entry => (
              <Flex
                key={`${entry.timestamp}-${entry.summary}`}
                align="baseline"
                justify="space-between"
                gap={4}
              >
                <Text fontSize="sm" flex={1} color="text.muted">
                  {entry.summary}
                </Text>
                <Text fontSize="xs" color="text.secondary">
                  {new Date(entry.timestamp).toLocaleTimeString()}
                </Text>
              </Flex>
            ))}
          </Stack>
        </Box>
      );
    },
    [voiceAdjustmentLog]
  );

  const getSectionFromParam = useCallback((section?: string): 'current' | 'converted' | 'upload' => {
    if (!section) {
      return 'current';
    }
    const normalized = section.toLowerCase();
    if (normalized.includes('convert')) {
      return 'converted';
    }
    if (normalized.includes('upload')) {
      return 'upload';
    }
    return 'current';
  }, []);

  const selectDocumentForVoice = useCallback(
    async (section: 'current' | 'converted' | 'upload', documentNumber?: number) => {
      if (!documentNumber || Number.isNaN(documentNumber) || documentNumber < 1) {
        return false;
      }

      // Handle uploads via the selector ref directly since uploaded files live inside the component
      if (section === 'upload') {
        documentSelectorModal.onOpen();
        documentSelectorRef.current?.focusSection('upload');
        const target = documentSelectorRef.current?.selectDocumentByIndex('upload', documentNumber);
        if (!target) {
          toast({
            title: 'Document not found',
            description: `Upload section does not have item #${documentNumber}.`,
            status: 'warning',
            duration: 3000,
          });
          return false;
        }

        voiceDocumentSelectionRef.current = { section, index: documentNumber };
        const enhancedDocs = await enhanceDocumentsWithPages([target]);
        setSelectedDocuments(enhancedDocs);
        bumpPreviewFocus({ docIndex: 0, page: 1, source: 'voice' });

        toast({
          title: 'Document ready',
          description: `${target.filename} selected via voice`,
          status: 'success',
          duration: 2500,
        });
        return true;
      }

      const docs = section === 'converted' ? convertedDocumentOptions : currentDocumentOptions;
      const targetDoc = docs[documentNumber - 1];
      if (!targetDoc) {
        toast({
          title: 'Document not found',
          description: `Section ${section} does not have item #${documentNumber}.`,
          status: 'warning',
          duration: 3000,
        });
        return false;
      }

      documentSelectorModal.onOpen();
      documentSelectorRef.current?.focusSection(section);
      documentSelectorRef.current?.selectDocumentByIndex(section, documentNumber);
      voiceDocumentSelectionRef.current = { section, index: documentNumber };

      const enhancedDocs = await enhanceDocumentsWithPages([targetDoc]);
      setSelectedDocuments(enhancedDocs);
      bumpPreviewFocus({ docIndex: 0, page: 1, source: 'voice' });

      toast({
        title: 'Document ready',
        description: `${targetDoc.filename} selected via voice`,
        status: 'success',
        duration: 2500,
      });
      return true;
    },
    [
      convertedDocumentOptions,
      currentDocumentOptions,
      documentSelectorModal,
      enhanceDocumentsWithPages,
      bumpPreviewFocus,
      toast,
    ]
  );

  const forceCloseChat = useCallback(() => {
    setIsChatVisible(false);
    setOrchestrationContext('manual');
  }, []);

  // Allows optional force-close when the voice orchestration is active
  const handleDockedChatClose = useCallback((opts?: { force?: boolean }) => {
    const force = opts?.force === true;

    if (orchestrationContext === 'voice') {
      if (force) {
        // Force-stop orchestration and close chat
        setOrchestrationContext('manual');
        setIsChatVisible(false);
        toast({
          title: 'Voice assistant stopped',
          description: 'Voice assistant stopped and chat closed.',
          status: 'info',
          duration: 3000,
          isClosable: true,
        });
        return;
      }

      // Non-forced close: inform user that voice assistant is active and keep chat open
      toast({
        title: 'Voice assistant active',
        description: 'Chat stays visible while AI configures your settings.',
        status: 'info',
        duration: 3000,
        isClosable: true,
      });
      setIsChatVisible(true);
      return;
    }

    setIsChatVisible(false);
  }, [orchestrationContext, toast]);

  const handleChatVisibilityToggle = useCallback(() => {
    if (isChatVisible) {
      handleDockedChatClose();
    } else {
      setIsChatVisible(true);
    }
  }, [handleDockedChatClose, isChatVisible]);

  // Chat resize handlers
  const handleChatResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizingChat(true);
    chatResizeRef.current = { startX: e.clientX, startWidth: chatWidth };
  }, [chatWidth]);

  const handleChatResizeMove = useCallback((e: MouseEvent) => {
    if (!isResizingChat || !chatResizeRef.current) return;

    // Calculate new width (resize from left edge, so subtract delta)
    const delta = chatResizeRef.current.startX - e.clientX;
    const newWidth = Math.min(Math.max(chatResizeRef.current.startWidth + delta, 280), 600);
    setChatWidth(newWidth);
  }, [isResizingChat]);

  const handleChatResizeEnd = useCallback(() => {
    if (isResizingChat) {
      setIsResizingChat(false);
      chatResizeRef.current = null;
      // Save to localStorage
      localStorage.setItem('printchakra_chat_width', chatWidth.toString());
    }
  }, [isResizingChat, chatWidth]);

  // Effect to add/remove mouse listeners for resize
  useEffect(() => {
    if (isResizingChat) {
      document.addEventListener('mousemove', handleChatResizeMove);
      document.addEventListener('mouseup', handleChatResizeEnd);
      document.body.style.cursor = 'ew-resize';
      document.body.style.userSelect = 'none';
    }

    return () => {
      document.removeEventListener('mousemove', handleChatResizeMove);
      document.removeEventListener('mouseup', handleChatResizeEnd);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isResizingChat, handleChatResizeMove, handleChatResizeEnd]);

  const handleVoiceCommand = useCallback(
    async (data: any) => {
      const { command, params } = data;
      console.log(`Handling voice command: ${command}`, params);

      const sectionParam = getSectionFromParam(params?.section);
      const parsedDocumentNumber = params?.document_number
        ? parseInt(params.document_number, 10)
        : undefined;

      switch (command) {
        // ==================== New Commands ====================
        case 'clear_print_queue': {
          try {
            await apiClient.post(API_ENDPOINTS.printerClearQueue);
            toast({
              title: 'Print Queue Cleared',
              description: 'All print jobs have been cancelled',
              status: 'success',
              duration: 3000,
            });
          } catch (error) {
            toast({
              title: 'Error',
              description: 'Failed to clear print queue',
              status: 'error',
              duration: 3000,
            });
          }
          break;
        }

        case 'check_connectivity': {
          connectivityModal.onOpen();
          connectionStatusRef.current?.runCheck();
          toast({
            title: 'Connectivity Check',
            description: 'Checking device connections...',
            status: 'info',
            duration: 2000,
          });
          break;
        }

        case 'device_info': {
          deviceInfoModal.onOpen();
          toast({
            title: 'Device Information',
            description: 'Opening device info panel',
            status: 'info',
            duration: 2000,
          });
          break;
        }

        case 'show_converted': {
          convertedDrawer.onOpen();
          toast({
            title: 'Converted Files',
            description: 'Showing converted documents',
            status: 'info',
            duration: 2000,
          });
          break;
        }

        case 'convert_file': {
          if (selectedFiles.length > 0) {
            conversionModal.onOpen();
            toast({
              title: 'File Conversion',
              description: `Ready to convert ${selectedFiles.length} file(s)`,
              status: 'info',
              duration: 2000,
            });
          } else {
            toast({
              title: 'No Files Selected',
              description: 'Please select files first by saying "selection mode on"',
              status: 'warning',
              duration: 3000,
            });
          }
          break;
        }

        case 'selection_mode_on': {
          setSelectionMode(true);
          toast({
            title: 'Selection Mode',
            description: 'Selection mode enabled. Tap files to select them.',
            status: 'info',
            duration: 2000,
          });
          break;
        }

        case 'selection_mode_off': {
          setSelectionMode(false);
          setSelectedFiles([]);
          toast({
            title: 'Selection Mode',
            description: 'Selection disabled, all files deselected',
            status: 'info',
            duration: 2000,
          });
          break;
        }

        case 'select_range': {
          // Parse range params like "select 1 to 5" or "select 1 and 4"
          const rangeText = params?.range || params?.text || '';
          const rangeMatch = rangeText.match(/(\d+)\s*(?:to|-)\s*(\d+)/i);
          const andMatch = rangeText.match(/(\d+)\s+and\s+(\d+)/i);
          const singleMatch = rangeText.match(/select\s+(\d+)/i);

          if (!selectionMode) {
            setSelectionMode(true);
          }

          if (rangeMatch) {
            // Range selection: "select 1 to 5"
            const start = parseInt(rangeMatch[1], 10) - 1;
            const end = parseInt(rangeMatch[2], 10) - 1;
            const newSelection = files
              .slice(Math.min(start, end), Math.max(start, end) + 1)
              .map(f => f.filename);
            setSelectedFiles(newSelection);
            toast({
              title: 'Range Selected',
              description: `Selected files ${rangeMatch[1]} to ${rangeMatch[2]}`,
              status: 'success',
              duration: 2000,
            });
          } else if (andMatch) {
            // Multiple selection: "select 1 and 4"
            const idx1 = parseInt(andMatch[1], 10) - 1;
            const idx2 = parseInt(andMatch[2], 10) - 1;
            const newSelection: string[] = [];
            if (files[idx1]) newSelection.push(files[idx1].filename);
            if (files[idx2]) newSelection.push(files[idx2].filename);
            setSelectedFiles(prev => Array.from(new Set([...prev, ...newSelection])));
            toast({
              title: 'Files Selected',
              description: `Selected files ${andMatch[1]} and ${andMatch[2]}`,
              status: 'success',
              duration: 2000,
            });
          } else if (singleMatch) {
            // Single selection: "select 1"
            const idx = parseInt(singleMatch[1], 10) - 1;
            if (files[idx]) {
              setSelectedFiles(prev =>
                prev.includes(files[idx].filename)
                  ? prev.filter(f => f !== files[idx].filename)
                  : [...prev, files[idx].filename]
              );
              toast({
                title: 'File Toggled',
                description: `Toggled file ${singleMatch[1]}`,
                status: 'info',
                duration: 2000,
              });
            }
          } else if (rangeText.toLowerCase().includes('all')) {
            // Select all
            setSelectedFiles(files.map(f => f.filename));
            toast({
              title: 'All Selected',
              description: `Selected all ${files.length} files`,
              status: 'success',
              duration: 2000,
            });
          }
          break;
        }

        case 'toggle_auto_capture': {
          // Toggle auto-capture for scan
          const currentSocket = socketRef.current;
          if (currentSocket) {
            if (autoCaptureEnabled) {
              currentSocket.emit('stop_auto_capture');
              setAutoCaptureEnabled(false);
              toast({
                title: 'Auto Capture',
                description: 'Auto capture stopped',
                status: 'info',
                duration: 2000,
              });
            } else {
              currentSocket.emit('start_auto_capture', {
                session_id: 'voice-capture',
                settings: orchestrateOptions,
              });
              setAutoCaptureEnabled(true);
              toast({
                title: 'Auto Capture',
                description: 'Auto capture started on phone',
                status: 'success',
                duration: 2000,
              });
            }
          }
          break;
        }

        case 'use_feed_tray': {
          // Set scan document source to feed tray
          setScanDocumentSource('feed');
          toast({
            title: 'Feed Tray Selected',
            description: 'Using printer feed tray for scanning',
            status: 'info',
            duration: 2000,
          });
          break;
        }

        case 'select_documents': {
          // Open document selector for selecting/uploading
          if (orchestrateMode === 'scan') {
            setScanDocumentSource('select');
          }
          documentSelectorModal.onOpen();
          toast({
            title: 'Document Selection',
            description: 'Opening document selector',
            status: 'info',
            duration: 2000,
          });
          break;
        }

        case 'select_documents_advanced': {
          // Advanced voice selection: "Select the first 2 documents", "Select documents 2 to 5", etc.
          const selectionText = params?.selection_text || params?.raw_input || '';
          const targetSection = (sectionParam as 'current' | 'converted') || 'current';
          const docs = targetSection === 'converted' ? convertedDocumentOptions : currentDocumentOptions;

          if (docs.length === 0) {
            toast({
              title: 'No documents',
              description: `No documents available in ${targetSection} section`,
              status: 'warning',
              duration: 2000,
            });
            break;
          }

          // Parse the advanced selection pattern
          const result = parseVoiceDocumentSelection(selectionText, docs.length);
          if (!result || result.indices.length === 0) {
            toast({
              title: 'Selection Error',
              description: 'Could not parse document selection. Try "select documents 1 to 3".',
              status: 'warning',
              duration: 3000,
            });
            break;
          }

          // Open modal and select the documents
          documentSelectorModal.onOpen();
          documentSelectorRef.current?.focusSection(targetSection);

          if (documentSelectorRef.current?.selectMultipleDocuments) {
            documentSelectorRef.current.selectMultipleDocuments(targetSection, result.indices);

            // Also update Dashboard's selectedDocuments state
            const selectedDocs = result.indices.map(idx => docs[idx]).filter(Boolean);
            if (selectedDocs.length > 0) {
              enhanceDocumentsWithPages(selectedDocs).then(enhanced => {
                setSelectedDocuments(enhanced);
              });
            }

            toast({
              title: 'Documents Selected',
              description: result.description,
              status: 'success',
              duration: 2500,
            });
          }
          break;
        }

        case 'close_panel': {
          // Close any open modal or panel
          if (orchestrateModal.isOpen) orchestrateModal.onClose();
          if (documentSelectorModal.isOpen) documentSelectorModal.onClose();
          if (conversionModal.isOpen) conversionModal.onClose();
          if (convertedDrawer.isOpen) convertedDrawer.onClose();
          if (deviceInfoModal.isOpen) deviceInfoModal.onClose();
          if (connectivityModal.isOpen) connectivityModal.onClose();
          if (imageModal.isOpen) imageModal.onClose();
          if (isChatVisible) {
            forceCloseChat();
          }
          toast({
            title: 'Panels Closed',
            description: 'All panels and modals closed',
            status: 'info',
            duration: 2000,
          });
          break;
        }

        // ==================== Existing Commands ====================
        case 'select_document': {
          if (sectionParam === 'upload') {
            documentSelectorModal.onOpen();
            documentSelectorRef.current?.focusSection('upload');
            toast({
              title: 'Upload',
              description: 'Switching to upload tab.',
              status: 'info',
              duration: 2000,
            });
            break;
          }

          // Open document selector first
          documentSelectorModal.onOpen();

          // Get active section from document selector or use specified section
          // Only call focusSection if user explicitly specified a section
          const activeSection = documentSelectorRef.current?.getActiveSection?.() || 'current';
          const userSpecifiedSection = sectionParam && sectionParam !== 'current' ? sectionParam : null;
          const targetSection = userSpecifiedSection || (activeSection === 'upload' ? 'current' : activeSection as 'current' | 'converted');

          // Only switch sections if user explicitly specified one
          if (userSpecifiedSection) {
            documentSelectorRef.current?.focusSection(targetSection);
          }

          if (parsedDocumentNumber) {
            await selectDocumentForVoice(targetSection, parsedDocumentNumber);

          } else {
            voiceDocumentSelectionRef.current = { section: targetSection, index: 1 };
            toast({
              title: 'Document Selection',
              description: `Browsing ${targetSection} documents`,
              status: 'info',
              duration: 2000,
            });
          }
          break;
        }

        case 'select_multiple_documents': {
          // Open document selector first
          documentSelectorModal.onOpen();

          // Use active section if no section specified, fallback to current
          const activeSection = documentSelectorRef.current?.getActiveSection?.() || 'current';
          const userSpecifiedSection = sectionParam && sectionParam !== 'current' ? sectionParam : null;
          const targetSection = userSpecifiedSection || (activeSection === 'upload' ? 'current' : activeSection as 'current' | 'converted');
          const count = params?.count || 2;

          const selectionType = params?.selection_type || 'first_n';
          const documentNumbers = params?.document_numbers;

          // Only switch sections if user explicitly specified one
          if (userSpecifiedSection) {
            documentSelectorRef.current?.focusSection(targetSection);
          }


          // Get the right document list
          const docs = targetSection === 'converted' ? convertedDocumentOptions : currentDocumentOptions;

          if (docs.length === 0) {
            toast({
              title: 'No documents',
              description: `No documents available in ${targetSection} section`,
              status: 'warning',
              duration: 2000,
            });
            break;
          }

          // Determine which documents to select
          let indicesToSelect: number[] = [];

          if (selectionType === 'all' || count === -1) {
            // Select all documents
            indicesToSelect = docs.map((_, idx) => idx);
          } else if (selectionType === 'specific' && documentNumbers) {
            // Specific document numbers (1-based from user)
            indicesToSelect = documentNumbers
              .filter((n: number) => n >= 1 && n <= docs.length)
              .map((n: number) => n - 1);
          } else if (selectionType === 'last_n') {
            // Last N documents
            const numToSelect = Math.min(count, docs.length);
            const startIdx = docs.length - numToSelect;
            indicesToSelect = Array.from({ length: numToSelect }, (_, i) => startIdx + i);
          } else {
            // First N documents (default)
            const numToSelect = Math.min(count, docs.length);
            indicesToSelect = Array.from({ length: numToSelect }, (_, i) => i);
          }


          // Select the documents via ref method
          if (documentSelectorRef.current?.selectMultipleDocuments) {
            documentSelectorRef.current.selectMultipleDocuments(targetSection, indicesToSelect);
            toast({
              title: 'Documents Selected',
              description: `Selected ${indicesToSelect.length} document(s) from ${targetSection}`,
              status: 'success',
              duration: 2000,
            });
          } else {
            // Fallback: select first document if method not available
            await selectDocumentForVoice(targetSection, 1);
            toast({
              title: 'Document Selected',
              description: `Selected first document from ${targetSection}`,
              status: 'info',
              duration: 2000,
            });
          }
          break;
        }

        case 'select_document_range': {
          // Handle range selection (e.g., "select documents 1 to 5")
          // Open document selector first
          documentSelectorModal.onOpen();

          const activeSection = documentSelectorRef.current?.getActiveSection?.() || 'current';
          const userSpecifiedSection = sectionParam && sectionParam !== 'current' ? sectionParam : null;
          const targetSection = userSpecifiedSection || (activeSection === 'upload' ? 'current' : activeSection as 'current' | 'converted');
          const startIdx = (params?.start || 1) - 1; // Convert to 0-based
          const endIdx = (params?.end || 5) - 1;

          // Only switch sections if user explicitly specified one
          if (userSpecifiedSection) {
            documentSelectorRef.current?.focusSection(targetSection);
          }


          // Get the right document list
          const docs = targetSection === 'converted' ? convertedDocumentOptions : currentDocumentOptions;

          if (docs.length === 0) {
            toast({
              title: 'No documents',
              description: `No documents available in ${targetSection} section`,
              status: 'warning',
              duration: 2000,
            });
            break;
          }

          // Create range of indices
          const indicesToSelect: number[] = [];
          for (let i = startIdx; i <= Math.min(endIdx, docs.length - 1); i++) {
            if (i >= 0) indicesToSelect.push(i);
          }

          // Select the documents via ref method
          if (documentSelectorRef.current?.selectMultipleDocuments && indicesToSelect.length > 0) {
            documentSelectorRef.current.selectMultipleDocuments(targetSection, indicesToSelect);
            toast({
              title: 'Range Selected',
              description: `Selected documents ${params?.start || 1} to ${Math.min(params?.end || 5, docs.length)}`,
              status: 'success',
              duration: 2000,
            });
          } else {
            toast({
              title: 'Selection Error',
              description: 'Could not select documents in range',
              status: 'warning',
              duration: 2000,
            });
          }
          break;
        }

        case 'deselect_document': {
          // Handle deselection - mirrors select patterns (last N, first N, range, specific, all)
          const deselectAll = params?.deselect_all ?? true;
          const deselectType = params?.deselect_type || 'all';
          const targetSection = (sectionParam as 'current' | 'converted') || 'current';
          const docs = targetSection === 'converted' ? convertedDocumentOptions : currentDocumentOptions;

          if (deselectAll || deselectType === 'all') {
            // Clear all selections
            if (documentSelectorRef.current?.clearSelections) {
              documentSelectorRef.current.clearSelections();
            }
            toast({
              title: 'Selection Cleared',
              description: 'All documents deselected',
              status: 'info',
              duration: 2000,
            });
          } else {
            let indicesToRemove: number[] = [];
            let description = '';

            if (deselectType === 'last_n') {
              const count = params?.count || 2;
              const startIdx = Math.max(0, docs.length - count);
              indicesToRemove = Array.from({ length: count }, (_, i) => startIdx + i);
              description = `Deselected last ${count} documents`;
            } else if (deselectType === 'first_n') {
              const count = params?.count || 2;
              indicesToRemove = Array.from({ length: Math.min(count, docs.length) }, (_, i) => i);
              description = `Deselected first ${count} documents`;
            } else if (deselectType === 'range') {
              const start = (params?.start || 1) - 1; // Convert to 0-based
              const end = (params?.end || 5) - 1;
              indicesToRemove = Array.from({ length: end - start + 1 }, (_, i) => start + i);
              description = `Deselected documents ${params?.start} to ${params?.end}`;
            } else {
              // Specific documents
              const docNumbers = params?.document_numbers || [params?.document_number || 1];
              indicesToRemove = docNumbers.map((n: number) => n - 1); // Convert to 0-based
              description = docNumbers.length > 1
                ? `Deselected documents ${docNumbers.join(', ')}`
                : `Deselected document ${docNumbers[0]}`;
            }

            if (documentSelectorRef.current?.removeDocumentsFromSelection && indicesToRemove.length > 0) {
              documentSelectorRef.current.removeDocumentsFromSelection(targetSection, indicesToRemove);
              toast({
                title: 'Documents Deselected',
                description,
                status: 'info',
                duration: 2000,
              });
            }
          }
          break;
        }


        case 'undo_action': {
          if (documentSelectorRef.current?.undoLastSelection) {
            documentSelectorRef.current.undoLastSelection();
            toast({
              title: 'Undo',
              description: 'Reverted last selection action',
              status: 'info',
              duration: 2000,
            });
          }
          break;
        }

        case 'switch_section': {
          // Voice command to switch between document selector sections
          const section = params?.section as string;
          documentSelectorModal.onOpen();

          if (section === 'converted' || section === 'converted files') {
            documentSelectorRef.current?.focusSection('converted');
            toast({
              title: 'Switched to Converted Files',
              description: 'Now viewing converted documents',
              status: 'info',
              duration: 2000,
            });
          } else if (section === 'upload' || section === 'local' || section === 'local files') {
            documentSelectorRef.current?.focusSection('upload');
            toast({
              title: 'Switched to Upload',
              description: 'Now viewing local file upload',
              status: 'info',
              duration: 2000,
            });
          } else if (section === 'current' || section === 'current documents') {
            documentSelectorRef.current?.focusSection('current');
            toast({
              title: 'Switched to Current Documents',
              description: 'Now viewing current documents',
              status: 'info',
              duration: 2000,
            });
          }
          break;
        }

        case 'proceed_action': {
          // Context-aware navigation - proceed to next step based on current state
          console.log('[PROCEED_ACTION] State check:', {
            documentSelectorModalOpen: documentSelectorModal.isOpen,
            orchestrateModalOpen: orchestrateModal.isOpen,
            orchestrateMode,
            orchestrateStep,
            selectedDocumentsCount: selectedDocuments.length,
          });

          // Priority 1: If document selector is open, confirm selection and show config
          if (documentSelectorModal.isOpen) {
            console.log('[PROCEED_ACTION] Document selector open - confirming selection');

            // Get the currently selected documents from the DocumentSelector component
            const internallySelectedDocs = documentSelectorRef.current?.getSelectedDocuments?.() || [];
            console.log('[PROCEED_ACTION] Internally selected docs:', internallySelectedDocs);

            // Close the selector first
            documentSelectorModal.onClose();

            if (orchestrateMode && internallySelectedDocs.length > 0) {
              // Update selectedDocuments with what was internally selected
              const enhancedDocs = await enhanceDocumentsWithPages(internallySelectedDocs);
              console.log('[PROCEED_ACTION] Setting selected documents:', enhancedDocs);
              setSelectedDocuments(enhancedDocs);

              // Ensure we're at step 2 (configuration)
              if (orchestrateStep !== 2) {
                setOrchestrateStep(2);
              }

              // Open orchestration modal if not already open to show configuration
              if (!orchestrateModal.isOpen) {
                orchestrateModal.onOpen();
              }

              markSelectionConfirmed();

              toast({
                title: 'Selection Confirmed',
                description: `${enhancedDocs.length} document(s) selected. Configure settings below.`,
                status: 'success',
                duration: 2000,
              });
            } else if (!orchestrateMode) {
              toast({
                title: 'Select Mode First',
                description: 'Please say "print" or "scan" to continue',
                status: 'info',
                duration: 2000,
              });
            } else {
              toast({
                title: 'Select Documents',
                description: 'Please select at least one document',
                status: 'info',
                duration: 2000,
              });
            }
            break;
          }

          // Priority 2: At step 1 with documents selected → move to step 2 (configuration)
          if (orchestrateStep === 1 && selectedDocuments.length > 0 && orchestrateMode) {
            console.log('[PROCEED_ACTION] Step 1 with documents → moving to step 2');
            setOrchestrateStep(2);
            if (!orchestrateModal.isOpen) {
              orchestrateModal.onOpen();
            }
            markSelectionConfirmed();
            toast({
              title: 'Configuration Page',
              description: 'Adjust your print settings, then say "proceed" to review.',
              status: 'info',
              duration: 2000,
            });
            break;
          }

          // Priority 3: At step 1 without documents → prompt for mode/documents
          if (orchestrateStep === 1) {
            toast({
              title: 'Select Mode',
              description: 'Please say "print" or "scan" to continue',
              status: 'info',
              duration: 3000,
            });
            break;
          }

          // Priority 4: At step 2 (configuration page) with print mode → move to review (step 3)
          if (orchestrateStep === 2 && orchestrateMode === 'print' && selectedDocuments.length > 0 && !justConfirmedSelectionRef.current) {
            console.log('[PROCEED_ACTION] Step 2 configuration → moving to step 3 review');
            setOrchestrateStep(3);
            toast({
              title: 'Review & Confirm',
              description: 'Review your settings. Say "proceed" to start printing.',
              status: 'info',
              duration: 2000,
            });
            break;
          }

          // Priority 5: At step 2 (configuration page) with scan mode → start scanning
          if (orchestrateStep === 2 && orchestrateMode === 'scan' && selectedDocuments.length > 0) {
            console.log('[PROCEED_ACTION] Step 2 scan → starting feed and scan');
            feedDocumentsThroughPrinter();
            toast({
              title: 'Starting Feed & Scan',
              description: `Feeding ${documentsToFeed} document(s)...`,
              status: 'info',
              duration: 2000,
            });
            break;
          }

          // Priority 6: At step 3 with print mode → execute print job
          if (orchestrateStep === 3 && orchestrateMode === 'print') {
            console.log('[PROCEED_ACTION] Step 3 review → starting print job');
            executePrintJob();
            break;
          }

          // Default fallback
          console.log('[PROCEED_ACTION] No matching condition, showing default message');
          toast({
            title: 'Proceed',
            description: 'Ready to continue',
            status: 'info',
            duration: 2000,
          });
          break;
        }

        case 'set_feed_count': {
          // Set number of documents to feed via voice
          const count = params?.count || 1;
          setDocumentsToFeed(count);
          toast({
            title: 'Feed Count Set',
            description: `Will feed ${count} document${count !== 1 ? 's' : ''}`,
            status: 'success',
            duration: 2000,
          });
          break;
        }

        case 'set_pages': {
          // Set page selection via voice (odd, even, custom, all)
          const pages = params?.pages || 'all';
          const customRange = params?.customRange;

          if (orchestrateMode === 'print') {
            setOrchestrateOptions(prev => ({
              ...prev,
              printPages: pages as any,
              ...(customRange && { printCustomRange: customRange }),
            }));
          } else if (orchestrateMode === 'scan') {
            setOrchestrateOptions(prev => ({
              ...prev,
              scanPageMode: pages as any,
              ...(customRange && { scanCustomRange: customRange }),
            }));
          }

          toast({
            title: 'Page Selection Updated',
            description: pages === 'custom' && customRange
              ? `Pages ${customRange}`
              : `${pages.charAt(0).toUpperCase() + pages.slice(1)} pages`,
            status: 'success',
            duration: 2000,
          });
          break;
        }

        case 'select_specific_documents': {


          // Handle specific document list selection (e.g., "select documents 4, 6, 8")
          const targetSection = (sectionParam as 'current' | 'converted') || 'current';
          const documentNumbers = params?.document_numbers || [];

          // Open document selector and focus on correct section
          documentSelectorModal.onOpen();
          documentSelectorRef.current?.focusSection(targetSection);

          const docs = targetSection === 'converted' ? convertedDocumentOptions : currentDocumentOptions;

          if (docs.length === 0) {
            toast({
              title: 'No documents',
              description: `No documents available in ${targetSection} section`,
              status: 'warning',
              duration: 2000,
            });
            break;
          }

          // Convert 1-based to 0-based indices
          const indicesToSelect = documentNumbers
            .filter((n: number) => n >= 1 && n <= docs.length)
            .map((n: number) => n - 1);

          if (documentSelectorRef.current?.addDocumentsToSelection && indicesToSelect.length > 0) {
            documentSelectorRef.current.addDocumentsToSelection(targetSection, indicesToSelect);
            toast({
              title: 'Documents Selected',
              description: `Added documents ${documentNumbers.join(', ')} to selection`,
              status: 'success',
              duration: 2000,
            });
          } else if (indicesToSelect.length === 0) {
            toast({
              title: 'Invalid Selection',
              description: 'No valid document numbers provided',
              status: 'warning',
              duration: 2000,
            });
          }
          break;
        }

        case 'switch_section': {


          documentSelectorModal.onOpen();
          documentSelectorRef.current?.focusSection(sectionParam);
          if (sectionParam !== 'upload') {
            voiceDocumentSelectionRef.current = {
              section: sectionParam,
              index: 1,
            };
          }
          toast({
            title: 'Section Switch',
            description: `Switching to ${sectionParam} section`,
            status: 'info',
            duration: 2000,
          });
          break;
        }

        case 'next_document': {
          const ctx = voiceDocumentSelectionRef.current;
          if (ctx.section === 'upload') {
            const nextIndex = ctx.index + 1;
            await selectDocumentForVoice(ctx.section, nextIndex);
            break;
          }

          const docs = ctx.section === 'converted' ? convertedDocumentOptions : currentDocumentOptions;
          if (docs.length === 0) {
            toast({
              title: 'No documents',
              description: `No documents available in ${ctx.section} section`,
              status: 'warning',
              duration: 2000,
            });
            break;
          }
          const nextIndex = Math.min(docs.length, ctx.index + 1);
          await selectDocumentForVoice(ctx.section, nextIndex);
          break;
        }

        case 'previous_document': {
          const ctx = voiceDocumentSelectionRef.current;
          if (ctx.section === 'upload') {
            const prevIndex = Math.max(1, ctx.index - 1);
            await selectDocumentForVoice(ctx.section, prevIndex);
            break;
          }

          const docs = ctx.section === 'converted' ? convertedDocumentOptions : currentDocumentOptions;
          if (docs.length === 0) {
            toast({
              title: 'No documents',
              description: `No documents available in ${ctx.section} section`,
              status: 'warning',
              duration: 2000,
            });
            break;
          }
          const prevIndex = Math.max(1, ctx.index - 1);
          await selectDocumentForVoice(ctx.section, prevIndex);
          break;
        }

        case 'upload_document': {
          const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
          if (fileInput) {
            fileInput.click();
          }
          toast({
            title: 'Upload',
            description: 'Opening upload dialog...',
            status: 'info',
            duration: 2000,
          });
          break;
        }

        case 'confirm': {
          if (orchestrateMode === 'print') {
            executePrintJobRef.current?.();
          } else if (orchestrateMode === 'scan') {
            executeScanJobRef.current?.();
          }
          toast({
            title: 'Executing',
            description: `Starting ${orchestrateMode} operation...`,
            status: 'success',
            duration: 2000,
          });
          break;
        }

        case 'cancel': {
          orchestrateModal.onClose();
          toast({
            title: 'Cancelled',
            description: 'Operation cancelled',
            status: 'warning',
            duration: 2000,
          });
          break;
        }

        case 'status': {
          toast({
            title: 'Current Status',
            description: orchestrateMode
              ? `${orchestrateMode.toUpperCase()} mode active`
              : 'No active operation',
            status: 'info',
            duration: 3000,
          });
          break;
        }

        case 'repeat_settings': {
          const currentSettings = orchestrateMode === 'print'
            ? `Print settings: ${orchestrateOptions.printColorMode}, ${orchestrateOptions.printLayout}, ${orchestrateOptions.printPaperSize}`
            : orchestrateMode === 'scan'
              ? `Scan settings: ${orchestrateOptions.scanColorMode}, ${orchestrateOptions.scanLayout}, ${orchestrateOptions.scanResolution} DPI`
              : 'No settings configured';

          toast({
            title: 'Current Settings',
            description: currentSettings,
            status: 'info',
            duration: 4000,
          });
          break;
        }

        case 'stop_recording': {
          if (isChatVisible) {
            forceCloseChat();
          }
          toast({
            title: 'Voice Stopped',
            description: 'Recording stopped',
            status: 'info',
            duration: 2000,
          });
          break;
        }

        // ==================== Settings Commands ====================
        case 'set_layout': {
          const layout = params?.layout || params?.orientation;
          if (layout && orchestrateMode) {
            const key = orchestrateMode === 'print' ? 'printLayout' : 'scanLayout';
            setOrchestrateOptions(prev => ({ ...prev, [key]: layout }));
            toast({
              title: 'Layout Updated',
              description: `Set to ${layout}`,
              status: 'success',
              duration: 2000,
            });
          }
          break;
        }

        case 'set_color':
        case 'set_color_mode': {
          const colorMode = params?.colorMode || params?.color_mode;
          if (colorMode && orchestrateMode) {
            const key = orchestrateMode === 'print' ? 'printColorMode' : 'scanColorMode';
            setOrchestrateOptions(prev => ({ ...prev, [key]: colorMode }));
            const displayMode = colorMode === 'bw' ? 'Black & White' : colorMode;
            toast({
              title: 'Color Mode Updated',
              description: `Set to ${displayMode}`,
              status: 'success',
              duration: 2000,
            });
          }
          break;
        }

        case 'set_paper_size': {
          const paperSize = params?.paperSize || params?.paper_size;
          if (paperSize && orchestrateMode) {
            const key = orchestrateMode === 'print' ? 'printPaperSize' : 'scanPaperSize';
            setOrchestrateOptions(prev => ({ ...prev, [key]: paperSize }));
            toast({
              title: 'Paper Size Updated',
              description: `Set to ${paperSize}`,
              status: 'success',
              duration: 2000,
            });
          }
          break;
        }

        case 'set_resolution': {
          const resolution = params?.resolution || params?.dpi;
          if (resolution && orchestrateMode) {
            const key = orchestrateMode === 'print' ? 'printResolution' : 'scanResolution';
            setOrchestrateOptions(prev => ({ ...prev, [key]: String(resolution) }));
            toast({
              title: 'Resolution Updated',
              description: `Set to ${resolution} DPI`,
              status: 'success',
              duration: 2000,
            });
          }
          break;
        }

        case 'set_copies': {
          const copies = params?.copies || params?.count;
          if (copies && orchestrateMode === 'print') {
            setOrchestrateOptions(prev => ({ ...prev, printCopies: String(copies) }));
            toast({
              title: 'Copies Updated',
              description: `Set to ${copies} copies`,
              status: 'success',
              duration: 2000,
            });
          }
          break;
        }

        case 'set_duplex': {
          const duplex = params?.duplex ?? params?.double_sided ?? true;
          if (orchestrateMode === 'print') {
            setOrchestrateOptions(prev => ({ ...prev, printDuplex: duplex }));
            toast({
              title: 'Duplex Updated',
              description: `Double-sided ${duplex ? 'enabled' : 'disabled'}`,
              status: 'success',
              duration: 2000,
            });
          }
          break;
        }

        case 'set_margins': {
          const margins = params?.margins;
          if (margins && orchestrateMode === 'print') {
            setOrchestrateOptions(prev => ({ ...prev, printMargins: margins }));
            toast({
              title: 'Margins Updated',
              description: `Set to ${margins}`,
              status: 'success',
              duration: 2000,
            });
          }
          break;
        }

        case 'set_scale': {
          const scale = params?.scale;
          if (scale && orchestrateMode === 'print') {
            setOrchestrateOptions(prev => ({ ...prev, printScale: String(scale) }));
            toast({
              title: 'Scale Updated',
              description: `Set to ${scale}%`,
              status: 'success',
              duration: 2000,
            });
          }
          break;
        }

        case 'set_quality': {
          const quality = params?.quality;
          if (quality && orchestrateMode) {
            const key = orchestrateMode === 'print' ? 'printQuality' : 'scanQuality';
            setOrchestrateOptions(prev => ({ ...prev, [key]: quality }));
            toast({
              title: 'Quality Updated',
              description: `Set to ${quality}`,
              status: 'success',
              duration: 2000,
            });
          }
          break;
        }

        case 'set_pages': {
          const pages = params?.pages || params?.page_mode;
          if (pages && orchestrateMode) {
            const key = orchestrateMode === 'print' ? 'printPages' : 'scanPageMode';
            setOrchestrateOptions(prev => ({ ...prev, [key]: pages }));
            if (params?.customRange || params?.custom_range) {
              const rangeKey = orchestrateMode === 'print' ? 'printCustomRange' : 'scanCustomRange';
              setOrchestrateOptions(prev => ({
                ...prev,
                [rangeKey]: params?.customRange || params?.custom_range
              }));
            }
            toast({
              title: 'Page Selection Updated',
              description: `Set to ${pages}`,
              status: 'success',
              duration: 2000,
            });
          }
          break;
        }

        case 'toggle_ocr':
        case 'toggle_text_mode': {
          const enabled = params?.enabled ?? params?.textMode ?? true;
          if (orchestrateMode === 'scan') {
            setOrchestrateOptions(prev => ({ ...prev, scanTextMode: enabled }));
            toast({
              title: 'OCR Mode Updated',
              description: `OCR ${enabled ? 'enabled' : 'disabled'}`,
              status: 'success',
              duration: 2000,
            });
          }
          break;
        }

        case 'set_format': {
          const format = params?.format;
          if (format && orchestrateMode === 'scan') {
            setOrchestrateOptions(prev => ({ ...prev, scanFormat: format }));
            toast({
              title: 'Format Updated',
              description: `Set to ${format.toUpperCase()}`,
              status: 'success',
              duration: 2000,
            });
          }
          break;
        }

        case 'set_scan_mode': {
          const scanMode = params?.scanMode || params?.scan_mode;
          if (scanMode && orchestrateMode === 'scan') {
            setOrchestrateOptions(prev => ({ ...prev, scanMode: scanMode }));
            toast({
              title: 'Scan Mode Updated',
              description: `Set to ${scanMode === 'multi' ? 'Multi-page' : 'Single page'}`,
              status: 'success',
              duration: 2000,
            });
          }
          break;
        }

        case 'scroll_down': {

          // Dispatch scroll event for orchestration modal
          const modalBody = document.querySelector('[data-orchestration-scroll]');
          if (modalBody) {
            modalBody.scrollBy({ top: 300, behavior: 'smooth' });
          }
          break;
        }

        case 'scroll_up': {
          const modalBody = document.querySelector('[data-orchestration-scroll]');
          if (modalBody) {
            modalBody.scrollBy({ top: -300, behavior: 'smooth' });
          }
          break;
        }

        case 'apply_settings':
        case 'go_next': {
          if (orchestrateStep < 3) {
            setOrchestrateStep(prev => prev + 1);
            toast({
              title: 'Moving Forward',
              description: `Step ${orchestrateStep + 1} of 3`,
              status: 'info',
              duration: 2000,
            });
          }
          break;
        }

        case 'go_back': {
          if (orchestrateStep > 1) {
            setOrchestrateStep(prev => prev - 1);
            toast({
              title: 'Going Back',
              description: `Step ${orchestrateStep - 1} of 3`,
              status: 'info',
              duration: 2000,
            });
          }
          break;
        }

        case 'apply_settings': {
          // Handle multiple settings from a single voice command
          // e.g., "print in landscape with grayscale at 300 dpi"
          if (!orchestrateMode) {
            toast({
              title: 'No Active Mode',
              description: 'Please start a print or scan operation first',
              status: 'warning',
              duration: 2000,
            });
            break;
          }

          const newSettings: Record<string, any> = {};
          const appliedChanges: string[] = [];

          // Map settings to orchestrate options
          const settingsMapping: Record<string, { key: string; label: string; format?: (v: any) => string }> = {
            colorMode: {
              key: orchestrateMode === 'print' ? 'printColorMode' : 'scanColorMode',
              label: 'Color mode',
              format: (v) => v === 'bw' ? 'Black & White' : 'Color'
            },
            color_mode: {
              key: orchestrateMode === 'print' ? 'printColorMode' : 'scanColorMode',
              label: 'Color mode',
              format: (v) => v === 'bw' ? 'Black & White' : 'Color'
            },
            layout: {
              key: orchestrateMode === 'print' ? 'printLayout' : 'scanLayout',
              label: 'Layout'
            },
            paperSize: {
              key: orchestrateMode === 'print' ? 'printPaperSize' : 'scanPaperSize',
              label: 'Paper size'
            },
            paper_size: {
              key: orchestrateMode === 'print' ? 'printPaperSize' : 'scanPaperSize',
              label: 'Paper size'
            },
            resolution: {
              key: orchestrateMode === 'print' ? 'printResolution' : 'scanResolution',
              label: 'Resolution',
              format: (v) => `${v} DPI`
            },
            quality: {
              key: orchestrateMode === 'print' ? 'printQuality' : 'scanQuality',
              label: 'Quality'
            },
            copies: { key: 'printCopies', label: 'Copies' },
            duplex: {
              key: 'printDuplex',
              label: 'Double-sided',
              format: (v) => v ? 'On' : 'Off'
            },
            margins: { key: 'printMargins', label: 'Margins' },
            format: { key: 'scanFormat', label: 'Format', format: (v) => v?.toUpperCase() },
            scanTextMode: {
              key: 'scanTextMode',
              label: 'OCR',
              format: (v) => v ? 'Enabled' : 'Disabled'
            },
            scanMode: {
              key: 'scanMode',
              label: 'Scan mode',
              format: (v) => v === 'multi' ? 'Multi-page' : 'Single page'
            },
            pages: {
              key: orchestrateMode === 'print' ? 'printPages' : 'scanPageMode',
              label: 'Pages'
            },
            scale: {
              key: 'printScale',
              label: 'Scale',
              format: (v) => `${v}%`
            },
            pagesPerSheet: { key: 'printPagesPerSheet', label: 'Pages per sheet' },
          };

          // Apply each setting
          if (params) {
            Object.entries(params).forEach(([key, value]) => {
              const mapping = settingsMapping[key];
              if (mapping && value !== undefined) {
                newSettings[mapping.key] = value;
                const formattedValue = mapping.format ? mapping.format(value) : String(value);
                appliedChanges.push(`${mapping.label}: ${formattedValue}`);
              }
            });
          }

          if (Object.keys(newSettings).length > 0) {
            setOrchestrateOptions(prev => ({ ...prev, ...newSettings }));

            const description = appliedChanges.length > 3
              ? `Updated ${appliedChanges.length} settings`
              : appliedChanges.join(', ');

            toast({
              title: 'Settings Applied',
              description: description,
              status: 'success',
              duration: 3000,
            });

            console.log('[Voice] Applied multiple settings:', newSettings);
          }
          break;
        }

        default:
          console.warn(`Unknown voice command: ${command}`);
      }

    },
    [
      connectivityModal,
      connectionStatusRef,
      convertedDrawer,
      conversionModal,
      convertedDocumentOptions,
      currentDocumentOptions,
      deviceInfoModal,
      documentSelectorModal,
      documentSelectorRef,
      files,
      forceCloseChat,
      getSectionFromParam,
      imageModal,
      isChatVisible,
      orchestrateMode,
      orchestrateOptions,
      orchestrateOptions.printColorMode,
      orchestrateOptions.printLayout,
      orchestrateOptions.printPaperSize,
      orchestrateOptions.scanColorMode,
      orchestrateOptions.scanLayout,
      orchestrateOptions.scanResolution,
      orchestrateModal,
      orchestrateStep,
      selectDocumentForVoice,
      selectedDocuments,
      selectedFiles.length,
      selectionMode,
      setOrchestrateOptions,
      setOrchestrateStep,
      toast,
      autoCaptureEnabled,
    ]
  );

  const handleOrchestrationUpdateRef = React.useRef(handleOrchestrationUpdate);
  useEffect(() => {
    handleOrchestrationUpdateRef.current = handleOrchestrationUpdate;
  }, [handleOrchestrationUpdate]);

  const surfaceCard = useColorModeValue('whiteAlpha.900', 'rgba(12, 16, 35, 0.95)');
  const dockedChatBg = useColorModeValue('rgba(248, 250, 255, 0.95)', 'rgba(9, 14, 26, 0.96)');
  const dockedBorderColor = useColorModeValue('rgba(121,95,238,0.12)', 'rgba(121,95,238,0.35)');
  const statusDotColor = connected ? 'green.400' : 'red.400';
  const statusTextColor = useColorModeValue('gray.600', 'gray.300');
  const statusText = error
    ? `Connection issues (retry ${connectionRetries})`
    : connected
      ? 'Live link established'
      : 'Disconnected';
  const showReopenOrchestrateButton = Boolean(orchestrateMode && !orchestrateModal.isOpen);

  // Derived color values to avoid calling hooks conditionally inside JSX
  const systemModalBgGradient = useColorModeValue(
    'linear-gradient(135deg, #ffffff 0%, #f8f7ff 100%)',
    'linear-gradient(135deg, #1a202c 0%, #2d3748 100%)'
  );
  const validatorBoxBg = useColorModeValue('rgba(255,255,255,0.5)', 'rgba(0,0,0,0.2)');
  const chatSidebarBg = useColorModeValue('white', 'gray.800');
  const chatSidebarBorderColor = useColorModeValue('gray.200', 'gray.700');

  useEffect(() => {
    setConnected(socketConnected);
  }, [socketConnected]);

  // Handle reconnect attempt from UI
  const handleReconnect = useCallback(() => {
    console.log('Attempting manual reconnection...');
    if (socketReconnect) {
      socketReconnect();
    }
    // Also refresh files after a short delay
    setTimeout(() => loadFiles(false), 1000);
  }, [socketReconnect]);

  useEffect(() => {
    if (!socket) {
      return;
    }

    const newFileListener = (data: any) => {
      console.log('New file uploaded:', data);
      // Immediately add placeholder file to state for instant UI feedback
      if (data?.filename) {
        setFiles((prevFiles: FileInfo[]) => {
          // Check if file already exists
          const exists = prevFiles.some(f => f.filename === data.filename);
          if (exists) {
            // Update existing file's processing status
            return prevFiles.map(f =>
              f.filename === data.filename
                ? { ...f, processing: true, processing_stage: 'Processing...' }
                : f
            );
          }
          // Add new file at the beginning (newest first)
          const newFile: FileInfo = {
            filename: data.filename,
            size: 0,
            created: new Date().toISOString(),
            has_text: false,
            processing: true,
            processing_stage: 'Processing...',
          };
          return [newFile, ...prevFiles];
        });
        // Force thumbnail refresh
        setRefreshToken(prev => prev + 1);
      }
      // Also refresh from server to get complete data
      loadFiles(false);
    };
    socket.on('new_file', newFileListener);

    const fileDeletedListener = (data: any) => {
      console.log('File deleted:', data);
      // Immediately remove from state
      if (data?.filename) {
        setFiles((prevFiles: FileInfo[]) =>
          prevFiles.filter(f => f.filename !== data.filename)
        );
      }
      loadFiles(false);
    };
    socket.on('file_deleted', fileDeletedListener);

    const processingProgressListener = (data: ProcessingProgress) => {
      console.log(`Processing: Step ${data.step}/${data.total_steps} - ${data.stage_name}`);
      setProcessingProgress(data);
      // Update the file's processing status in the list
      if (data?.filename) {
        setFiles((prevFiles: FileInfo[]) =>
          prevFiles.map(f =>
            f.filename === data.filename
              ? {
                ...f,
                processing: true,
                processing_step: data.step,
                processing_total: data.total_steps,
                processing_stage: data.stage_name
              }
              : f
          )
        );
      }
    };
    socket.on('processing_progress', processingProgressListener);

    const processingCompleteListener = (data: any) => {
      console.log('Processing complete:', data);
      setProcessingProgress(null);

      // Handle renamed files - match by original_filename if file was renamed
      const filenameToMatch = data?.original_filename || data?.filename;
      const newFilename = data?.filename;

      if (filenameToMatch) {
        setFiles((prevFiles: FileInfo[]) =>
          prevFiles.map(f => {
            // Match against original filename (before rename) or current filename
            if (f.filename === filenameToMatch || f.filename === newFilename) {
              return {
                ...f,
                filename: newFilename, // Update to new filename if renamed
                processing: false,
                has_text: data.has_text || false
              };
            }
            return f;
          })
        );
        // Force thumbnail refresh by updating the refresh token
        setRefreshToken(prev => prev + 1);
      }
      // Refresh from server to get complete data (with slight delay to ensure file is saved)
      setTimeout(() => loadFiles(false), 300);
    };
    socket.on('processing_complete', processingCompleteListener);

    const processingErrorListener = (data: any) => {
      console.error('Processing error:', data);
      setProcessingProgress(null);
      // Update file state to show error
      if (data?.filename) {
        setFiles((prevFiles: FileInfo[]) =>
          prevFiles.map(f =>
            f.filename === data.filename
              ? { ...f, processing: false, processing_error: data.error }
              : f
          )
        );
      }
    };
    socket.on('processing_error', processingErrorListener);

    // Listen for orchestration updates from voice commands
    const orchestrationUpdateListener = (payload: any) => {
      handleOrchestrationUpdateRef.current?.(payload);
    };
    socket.on('orchestration_update', orchestrationUpdateListener);

    // Listen for auto-capture state changes from phone
    const autoCaptureStateListener = (data: any) => {
      console.log('Auto-capture state changed from phone:', data);
      setAutoCaptureEnabled(data?.enabled || false);
      if (data?.enabled) {
        toast({
          title: '📱 Phone Auto-Capture Active',
          description: data?.source === 'phone' ? 'Started from Phone interface' : 'Auto-capture is running',
          status: 'success',
          duration: 3000,
        });
      } else if (data?.capturedCount > 0) {
        toast({
          title: '✅ Auto-Capture Complete',
          description: `Phone captured ${data.capturedCount} document(s)`,
          status: 'success',
          duration: 4000,
        });
      }
    };
    socket.on('auto_capture_state_changed', autoCaptureStateListener);

    // Listen for OCR completion events
    const ocrCompleteListener = (data: any) => {
      console.log('OCR complete:', data);
      if (data?.filename && data?.success && data?.result) {
        const newFilename = data.filename;
        const oldFilename = data.original_filename || data.filename;
        const wasRenamed = data.renamed || (oldFilename !== newFilename);
        
        // Store the OCR result with new filename
        setOcrResults(prev => ({ ...prev, [newFilename]: data.result }));

        // Update file list - rename if needed
        setFiles((prevFiles: FileInfo[]) => {
          if (wasRenamed) {
            // Remove old file, add renamed file
            return prevFiles.map(f =>
              f.filename === oldFilename
                ? { ...f, filename: newFilename, has_text: true }
                : f
            );
          } else {
            // Just update has_text status
            return prevFiles.map(f =>
              f.filename === newFilename ? { ...f, has_text: true } : f
            );
          }
        });

        // Clear loading state for both old and new filenames
        setOcrLoading(prev => {
          const updated = { ...prev };
          delete updated[oldFilename];
          delete updated[newFilename];
          return updated;
        });

        // If OCR view is open for the old filename, update it
        if (activeOCRView === oldFilename && wasRenamed) {
          setActiveOCRView(newFilename);
        }

        toast({
          title: wasRenamed ? '📄 OCR Complete - File Renamed' : '📄 OCR Complete',
          description: data.result.derived_title
            ? `"${data.result.derived_title}"${wasRenamed ? '\n✨ File automatically renamed' : ''}`
            : `Processed ${data.result.word_count} words`,
          status: 'success',
          duration: 4000,
        });
      }
    };
    socket.on('ocr_complete', ocrCompleteListener);

    // Listen for config updates from voice commands (settings like layout, color mode, etc.)
    const configUpdateListener = (data: any) => {
      console.log('Config update from voice:', data);
      if (data?.updates) {
        const updates = data.updates;
        const currentMode = orchestrateModeRef.current;
        
        setOrchestrateOptions(prev => {
          const newOptions = { ...prev };
          
          // Map voice config keys to orchestrate options
          if (updates.layout) {
            if (currentMode === 'print') {
              newOptions.printLayout = updates.layout;
            } else if (currentMode === 'scan') {
              newOptions.scanLayout = updates.layout;
            }
          }
          if (updates.colorMode || updates.color_mode) {
            const colorMode = updates.colorMode || updates.color_mode;
            if (currentMode === 'print') {
              newOptions.printColorMode = colorMode;
            } else if (currentMode === 'scan') {
              newOptions.scanColorMode = colorMode;
            }
          }
          if (updates.paperSize || updates.paper_size) {
            const paperSize = updates.paperSize || updates.paper_size;
            if (currentMode === 'print') {
              newOptions.printPaperSize = paperSize;
            } else if (currentMode === 'scan') {
              newOptions.scanPaperSize = paperSize;
            }
          }
          if (updates.resolution) {
            if (currentMode === 'print') {
              newOptions.printResolution = String(updates.resolution);
            } else if (currentMode === 'scan') {
              newOptions.scanResolution = String(updates.resolution);
            }
          }
          if (updates.copies) {
            newOptions.printCopies = String(updates.copies);
          }
          if (updates.duplex !== undefined) {
            newOptions.printDuplex = updates.duplex;
          }
          if (updates.quality) {
            newOptions.printQuality = updates.quality;
          }
          
          return newOptions;
        });
        
        // Show toast for settings update
        const updateDesc = Object.entries(updates)
          .map(([key, value]) => `${key}: ${value}`)
          .join(', ');
        toast({
          title: 'Settings Updated',
          description: updateDesc,
          status: 'success',
          duration: 2000,
        });
      }
    };
    socket.on('config_update', configUpdateListener);

    // Listen for delay settings updates from other devices
    const delaySettingsListener = (data: { initialDelay: number; interCaptureDelay: number; timestamp: number }) => {
      console.log('Delay settings updated from another device:', data);
      setInitialDelay(data.initialDelay);
      setInterCaptureDelay(data.interCaptureDelay);
      toast({
        title: '⚙️ Delay Settings Synced',
        description: `Updated: ${data.initialDelay}s startup, ${data.interCaptureDelay}s between captures`,
        status: 'info',
        duration: 3000,
      });
    };
    socket.on('delay_settings_updated', delaySettingsListener);

    loadFiles();
    loadConvertedFiles(); // Load converted files on component mount

    let pollInterval = 60000;
    const maxInterval = 300000;
    let timeoutId: ReturnType<typeof setTimeout>;
    // local counter removed - avoid poll debug logging in production

    const startPolling = () => {
      // -- startPolling invoked
      timeoutId = setTimeout(async () => {
        console.log('Safety net polling for new files...');
        try {
          await loadFiles(false);
          if (pollInterval > 60000) {
            console.log('Connection restored, resetting poll interval');
            pollInterval = 60000;
            setConnectionRetries(0);
          }
        } catch (err) {
          pollInterval = Math.min(pollInterval * 1.5, maxInterval);
          setConnectionRetries((prev: number) => prev + 1);
          console.log(`Poll failed, backing off to ${pollInterval}ms`);
        }
        startPolling();
      }, pollInterval);
    };

    startPolling();

    const handleVisibilityChange = () => {
      if (!document.hidden) {
        console.log('Page became visible - refreshing files');
        loadFiles(false);
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      clearTimeout(timeoutId);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      socket.off('new_file', newFileListener);
      socket.off('file_deleted', fileDeletedListener);
      socket.off('processing_progress', processingProgressListener);
      socket.off('processing_complete', processingCompleteListener);
      socket.off('processing_error', processingErrorListener);
      socket.off('orchestration_update', orchestrationUpdateListener);
      socket.off('auto_capture_state_changed', autoCaptureStateListener);
      socket.off('ocr_complete', ocrCompleteListener);
      socket.off('config_update', configUpdateListener);
      socket.off('delay_settings_updated', delaySettingsListener);
      // -- startPolling cleaned up
    };
  }, [socket, toast]);

  // Safety mechanism: Clear stuck processing states after 2 minutes
  useEffect(() => {
    const PROCESSING_TIMEOUT_MS = 120000; // 2 minutes
    const processingStartTimes = new Map<string, number>();

    const checkStuckProcessing = () => {
      const now = Date.now();
      setFiles((prevFiles: FileInfo[]) => {
        let hasChanges = false;
        const updatedFiles = prevFiles.map(f => {
          if (f.processing) {
            // Track when processing started
            if (!processingStartTimes.has(f.filename)) {
              processingStartTimes.set(f.filename, now);
            }

            const startTime = processingStartTimes.get(f.filename) || now;
            if (now - startTime > PROCESSING_TIMEOUT_MS) {
              console.warn(`Processing timeout for ${f.filename} - clearing stuck state`);
              hasChanges = true;
              processingStartTimes.delete(f.filename);
              return { ...f, processing: false };
            }
          } else {
            // Clear tracking when processing is done
            processingStartTimes.delete(f.filename);
          }
          return f;
        });

        return hasChanges ? updatedFiles : prevFiles;
      });
    };

    const intervalId = setInterval(checkStuckProcessing, 10000); // Check every 10 seconds

    return () => {
      clearInterval(intervalId);
    };
  }, []);

  const lastLoadFilesRunRef = React.useRef<number>(0);
  const minLoadFilesInterval = 200; // ms - reduced for faster real-time updates during capture

  const loadFiles = useCallback(async (showLoading = true) => {
    const now = Date.now();
    if (now - lastLoadFilesRunRef.current < minLoadFilesInterval) {
      console.log('Skipping loadFiles due to rate limit');
      return;
    }
    lastLoadFilesRunRef.current = now;
    // Prevent overlapping calls that can cause duplicate requests
    if ((loadFiles as any)._isRunning) {
      console.log('loadFiles: already running - skipping');
      return;
    }
    (loadFiles as any)._isRunning = true;
    try {
      if (showLoading) {
        setLoading(true);
      }
      const response = await apiClient.get(API_ENDPOINTS.files, {
        timeout: 10000, // 10 second timeout
      });
      const filesData = Array.isArray(response.data) ? response.data : response.data.files || [];

      // Smart cache: check if files actually changed (count OR content/processing status)
      const newCount = filesData.length;
      const cachedData = filesCacheRef.current.data || [];

      // Create a fingerprint of the files list to detect any changes
      const getFilesFingerprint = (files: any[]) => {
        return files.map(f => `${f.filename}:${f.processing}:${f.has_text}`).sort().join('|');
      };

      const newFingerprint = getFilesFingerprint(filesData);
      const cachedFingerprint = getFilesFingerprint(cachedData);

      if (newCount !== filesCacheRef.current.lastCount || newFingerprint !== cachedFingerprint) {
        console.log(
          `Files changed: count ${filesCacheRef.current.lastCount} → ${newCount}, updating cache`
        );
        setFiles(filesData);
        filesCacheRef.current = {
          data: filesData,
          lastCount: newCount,
          timestamp: Date.now(),
        };
      } else {
        console.log(`Files unchanged (${newCount}), using cached data`);
        // Use cached data if available
        if (filesCacheRef.current.data) {
          setFiles(filesCacheRef.current.data);
        } else {
          setFiles(filesData);
        }
      }
      setError(null);
    } catch (err: any) {
      console.error('Failed to load files:', err);
      const errorMsg =
        err.code === 'ERR_NETWORK' || err.code === 'ERR_CONNECTION_CLOSED'
          ? 'Backend connection lost. Retrying...'
          : err.message || 'Failed to load files';
      setError(errorMsg);

      // Don't show persistent error on background polls
      if (!showLoading) {
        setTimeout(() => setError(null), 5000); // Clear error after 5s
      }
    } finally {
      (loadFiles as any)._isRunning = false;
      if (showLoading) {
        setLoading(false);
      }
    }
  }, []);

  const deleteFile = async (filename: string) => {
    if (!window.confirm('Are you sure you want to delete this file?')) {
      return;
    }

    try {
      await apiClient.delete(`${API_ENDPOINTS.delete}/${filename}`);
      setFiles(files.filter((f: FileInfo) => f.filename !== filename));
      // Also remove OCR results for deleted file
      setOcrResults(prev => {
        const updated = { ...prev };
        delete updated[filename];
        return updated;
      });
      if (selectedFile === filename) {
        setSelectedFile(null);
        setOcrText('');
      }
    } catch (err: any) {
      toast({
        title: 'Delete failed',
        description: err.message,
        status: 'error',
      });
    }
  };

  const viewOCR = async (filename: string) => {
    try {
      const response = await apiClient.get(`${API_ENDPOINTS.ocr}/${filename}`);
      setOcrText(response.data.text || 'No text found');
      setSelectedFile(filename);
    } catch (err: any) {
      toast({
        title: 'Unable to load OCR',
        description: err.message,
        status: 'error',
      });
    }
  };

  // PaddleOCR functions
  const handleRunOCR = async (filename: string) => {
    // Mark as loading
    setOcrLoading(prev => ({ ...prev, [filename]: true }));
    setOcrProgress(prev => ({ ...prev, [filename]: { elapsed: 0, status: 'Starting OCR...' } }));

    // Start a timer to show elapsed time
    const progressInterval = setInterval(() => {
      setOcrProgress(prev => {
        const current = prev[filename] || { elapsed: 0, status: 'Processing...' };
        return {
          ...prev,
          [filename]: { ...current, elapsed: current.elapsed + 500 }
        };
      });
    }, 500);

    try {
      // Show initial toast
      const toastId = toast({
        title: 'OCR Processing',
        description: 'Starting OCR on document... This may take a moment.',
        status: 'info',
        duration: null, // Keep open until dismissed
        isClosable: true,
      });

      // Start OCR processing
      const response = await runOCR(filename);

      // Close the progress toast
      if (toastId !== undefined) {
        toast.close(toastId);
      }

      if (response.success && response.ocr_result) {
        const newFilename = response.filename || filename;
        const wasRenamed = newFilename !== filename;
        
        // Store the OCR result with new filename
        setOcrResults(prev => ({ ...prev, [newFilename]: response.ocr_result! }));

        // Update file list - rename if needed
        setFiles(prev => {
          if (wasRenamed) {
            return prev.map(f =>
              f.filename === filename
                ? { ...f, filename: newFilename, has_text: true }
                : f
            );
          } else {
            return prev.map(f =>
              f.filename === filename ? { ...f, has_text: true } : f
            );
          }
        });

        const elapsedSeconds = (response.ocr_result.processing_time_ms / 1000).toFixed(1);

        toast({
          title: wasRenamed ? '✅ OCR Complete - File Renamed' : '✅ OCR Complete',
          description: response.ocr_result.derived_title
            ? `Document: "${response.ocr_result.derived_title}"\n${response.ocr_result.word_count} words • ${elapsedSeconds}s${wasRenamed ? '\n✨ File automatically renamed' : ''}`
            : `Processed ${response.ocr_result.word_count} words in ${elapsedSeconds}s`,
          status: 'success',
          duration: 5000,
        });
      } else {
        toast({
          title: '❌ OCR Failed',
          description: response.error || 'Unknown error',
          status: 'error',
          duration: 5000,
        });
      }
    } catch (err: any) {
      toast({
        title: '❌ OCR Error',
        description: err.message || 'An unexpected error occurred',
        status: 'error',
        duration: 5000,
      });
    } finally {
      clearInterval(progressInterval);
      setOcrLoading(prev => ({ ...prev, [filename]: false }));
      setOcrProgress(prev => {
        const newProgress = { ...prev };
        delete newProgress[filename];
        return newProgress;
      });
    }
  };

  const handleViewOCRResult = async (filename: string) => {
    // If we already have the result cached, just open the view
    if (ocrResults[filename]) {
      setActiveOCRView(filename);
      return;
    }

    // Otherwise, fetch it
    try {
      const response = await getOCRResult(filename);
      if (response.success && response.ocr_result) {
        setOcrResults(prev => ({ ...prev, [filename]: response.ocr_result! }));
        setActiveOCRView(filename);
      } else {
        toast({
          title: 'No OCR Result',
          description: 'Run OCR first to process this document',
          status: 'info',
        });
      }
    } catch (err: any) {
      toast({
        title: 'Failed to load OCR',
        description: err.message,
        status: 'error',
      });
    }
  };

  const closeOCRView = () => {
    setActiveOCRView(null);
  };

  // Load OCR statuses for all files on mount and when files change
  const loadOCRStatuses = useCallback(async (filenames: string[]) => {
    if (filenames.length === 0) return;

    try {
      const response = await getBatchOCRStatus(filenames);
      if (response.success && response.statuses) {
        // For files with OCR ready, fetch their full results
        const ocrReadyFiles = Object.entries(response.statuses)
          .filter(([_, status]) => status.has_ocr)
          .map(([filename]) => filename);

        // Batch load OCR results for files that have them
        for (const filename of ocrReadyFiles) {
          if (!ocrResults[filename]) {
            try {
              const ocrResponse = await getOCRResult(filename);
              if (ocrResponse.success && ocrResponse.ocr_result) {
                setOcrResults(prev => ({ ...prev, [filename]: ocrResponse.ocr_result! }));
              }
            } catch (err) {
              console.warn(`Failed to load OCR for ${filename}:`, err);
            }
          }
        }
      }
    } catch (err) {
      console.warn('Failed to load OCR statuses:', err);
    }
  }, [ocrResults]);

  // Load OCR statuses when files change
  useEffect(() => {
    if (files.length > 0) {
      const filenames = files.map(f => f.filename);
      loadOCRStatuses(filenames);
    }
  }, [files, loadOCRStatuses]);

  const openOrchestrateModal = () => {
    setOrchestrationContext('manual');
    // Reset state
    setOrchestrateStep(1);
    setOrchestrateMode(null);
    setScanDocumentSource('feed'); // Default to feed tray for scanning
    // Reset document feeder state
    setDocumentsFed(false);
    setFeedCount(0);
    setDocumentsToFeed(1);
    // Reset the "user closed doc selector" flag to allow auto-opening
    setUserClosedDocSelector(false);

    // If files are selected, auto-select print mode
    if (selectedFiles.length > 0) {
      setOrchestrateMode('print');
      setOrchestrateStep(2);
    }

    orchestrateModal.onOpen();
  };

  const reopenOrchestrateModal = () => {
    // Always treat reopening as a voice context (no blur, transparent background)
    setOrchestrationContext('voice');
    setIsChatVisible(true);

    // Reopen with current mode and step 2 (keep same settings)
    if (orchestrateMode) {
      setOrchestrateStep(2);
      orchestrateModal.onOpen();
    } else {
      // If no mode set, open from beginning
      openOrchestrateModal();
    }
  };

  const handleVoiceOrchestrationTrigger = (mode: OrchestrationAction, config?: any) => {
    console.log('Dashboard: Orchestration triggered', { mode, config });

    setOrchestrationContext('voice');

    // Keep the voice chat drawer OPEN during orchestration for Jarvis-like experience
    // User can interact with AI while configuring print/scan settings
    if (!isChatVisible) {
      setIsChatVisible(true);
    }

    setOrchestrateMode(mode);

    // For scan mode, default directly to feed tray (skip document source selection)
    if (mode === 'scan') {
      setScanDocumentSource('feed');
    }

    // For print mode, reset userClosedDocSelector so document selector auto-opens
    if (mode === 'print') {
      setUserClosedDocSelector(false);
      // Clear any previously selected documents to show fresh selection screen
      setSelectedDocuments([]);
    }

    const resolvedVoiceOptions = resolveInitialVoiceConfigOptions(mode, config);
    if (Object.keys(resolvedVoiceOptions).length > 0) {
      setOrchestrateOptions(prev => ({
        ...prev,
        ...resolvedVoiceOptions,
      }));
    }

    setOrchestrateStep(2);

    if (!orchestrateModal.isOpen) {
      orchestrateModal.onOpen();
      toast({
        title: `${mode === 'print' ? '🖨️ Print' : '📸 Scan'} Mode Activated`,
        description: mode === 'scan'
          ? 'Configure scanning settings. Feed documents through printer tray.'
          : 'Select documents to print.',
        status: 'success',
        duration: 4000,
        isClosable: true,
      });
    }
  };

  voiceOrchestrationTriggerRef.current = handleVoiceOrchestrationTrigger;

  const executePrintJob = async () => {
    try {
      console.log('[PRINT] executePrintJob called');
      console.log('[PRINT] selectedDocuments:', selectedDocuments);

      // Collect all documents to print
      const documentsToprint: string[] = [];

      if (selectedDocuments.length > 0) {
        selectedDocuments.forEach(doc => documentsToprint.push(doc.filename));
      }

      if (orchestrateOptions.printConvertedFiles.length > 0) {
        orchestrateOptions.printConvertedFiles.forEach((f: string) => documentsToprint.push(f));
      }

      console.log('[PRINT] documentsToprint:', documentsToprint);

      if (documentsToprint.length === 0) {
        toast({
          title: 'No Documents Selected',
          description: 'Please select documents to print first.',
          status: 'warning',
          duration: 3000,
        });
        return;
      }

      // Close modal immediately
      orchestrateModal.onClose();

      // Emit auto-capture start IMMEDIATELY - Phone handles the delay
      // This centralizes ALL delay logic in the phone view
      if (socket && !autoCaptureEnabled) {
        console.log(`[DASHBOARD] Sending start_auto_capture with delaySeconds=${initialDelay}`);
        socket.emit('start_auto_capture', {
          documentCount: documentsToprint.length,
          delaySeconds: initialDelay,  // Tell phone what delay to use
          timestamp: Date.now(),
        });
        setAutoCaptureEnabled(true);
        toast({
          title: '📱 Auto-Capture Sent to Phone',
          description: `Phone will enable auto-capture in ${initialDelay}s`,
          status: 'info',
          duration: 3000,
        });
      }

      // Print ALL documents INSTANTLY (no delays)
      toast({
        title: '🖨️ Printing Started',
        description: `Sending ${documentsToprint.length} document(s) to printer...`,
        status: 'info',
        duration: 2000,
      });

      let successCount = 0;
      let failCount = 0;

      // Print all documents immediately (no waiting)
      for (let i = 0; i < documentsToprint.length; i++) {
        const filename = documentsToprint[i];

        try {
          console.log(`[PRINT] Printing document ${i + 1}/${documentsToprint.length}: ${filename}`);

          // Backend will convert images to PDF if needed
          const response = await apiClient.post('/print/document', {
            filename,
            copies: orchestrateOptions.printCopies || 1,
          });

          if (response.data.success) {
            successCount++;
            console.log(`[PRINT] Success: ${filename}`);
          } else {
            failCount++;
            console.error(`[PRINT] Failed: ${filename}`, response.data.error);
          }
        } catch (err: any) {
          failCount++;
          console.error(`[PRINT] Error printing ${filename}:`, err);
        }
      }

      // Final summary
      if (successCount > 0) {
        toast({
          title: `🖨️ Printed ${successCount} Document${successCount !== 1 ? 's' : ''}`,
          description: failCount > 0
            ? `${successCount} printed, ${failCount} failed.`
            : 'All documents sent to printer!',
          status: failCount > 0 ? 'warning' : 'success',
          duration: 4000,
        });
      } else {
        toast({
          title: 'Print Failed',
          description: 'Could not print any documents.',
          status: 'error',
          duration: 5000,
        });
      }
    } catch (err: any) {
      toast({
        title: 'Print Failed',
        description: err.response?.data?.error || err.message,
        status: 'error',
        duration: 5000,
      });
    }
  };

  const executeScanJob = async () => {
    try {
      const response = await apiClient.post('/orchestrate/scan', {
        pageMode: orchestrateOptions.scanPageMode,
        customRange: orchestrateOptions.scanCustomRange,
        layout: orchestrateOptions.scanLayout,
        paperSize: orchestrateOptions.scanPaperSize,
        resolution: orchestrateOptions.scanResolution,
        colorMode: orchestrateOptions.scanColorMode,
        saveAsDefault: orchestrateOptions.saveAsDefault,
      });

      toast({
        title: 'Scan Job Initiated',
        description: response.data.message || 'Scanner is ready',
        status: 'success',
      });

      orchestrateModal.onClose();
    } catch (err: any) {
      toast({
        title: 'Scan Failed',
        description: err.response?.data?.error || err.message,
        status: 'error',
      });
    }
  };

  // Assign refs for voice command handlers
  useEffect(() => {
    executePrintJobRef.current = executePrintJob;
    executeScanJobRef.current = executeScanJob;
  });

  // Feed documents through printer (uses printer as document feeder)
  // Also triggers auto-capture on phone with 5-second countdown
  // First document has 12-second delay BEFORE feeding, subsequent documents have 2-second delay
  const feedDocumentsThroughPrinter = async () => {
    if (documentsToFeed < 1) {
      toast({
        title: 'Invalid Count',
        description: 'Please enter at least 1 document to feed.',
        status: 'warning',
        duration: 3000,
      });
      return;
    }

    setIsFeedingDocuments(true);
    let successCount = 0;
    let failCount = 0;

    // Emit auto-capture start IMMEDIATELY - Phone handles the delay
    // This centralizes ALL delay logic in the phone view
    if (socket && !autoCaptureEnabled) {
      console.log(`[DASHBOARD] Sending start_auto_capture with delaySeconds=${initialDelay}`);
      socket.emit('start_auto_capture', {
        documentCount: documentsToFeed,
        delaySeconds: initialDelay,  // Tell phone what delay to use
        timestamp: Date.now(),
      });
      setAutoCaptureEnabled(true);
      toast({
        title: '📱 Auto-Capture Sent to Phone',
        description: `Phone will enable auto-capture in ${initialDelay}s`,
        status: 'info',
        duration: 3000,
      });
    }

    // Feed documents INSTANTLY (no waiting)
    toast({
      title: '🖨️ Feeding Started',
      description: `Feeding ${documentsToFeed} document(s) through printer...`,
      status: 'info',
      duration: 2000,
    });

    try {
      // Loop through the number of documents to feed - NO DELAYS
      for (let i = 0; i < documentsToFeed; i++) {
        try {
          // Use the /print endpoint with type: "blank" - this triggers the printer to feed paper
          const response = await apiClient.post('/print', { type: 'blank' });


          if (response.data.status === 'success') {
            successCount++;
            setFeedCount(prev => prev + 1);
            console.log(`[FEED] Document ${i + 1} fed successfully`);
          } else {
            console.error(`[FEED] Document ${i + 1} failed: ${response.data.message}`);
            failCount++;
          }
        } catch (innerErr) {
          console.error(`[FEED] Error feeding document ${i + 1}:`, innerErr);
          failCount++;
        }
      }

      // Show results
      if (successCount > 0) {
        setDocumentsFed(true);
        toast({
          title: `🖨️ Fed ${successCount} Document${successCount !== 1 ? 's' : ''}`,
          description: failCount > 0
            ? `${successCount} fed, ${failCount} failed. Auto-capture will enable soon.`
            : 'All documents fed! Auto-capture will enable in a moment.',
          status: failCount > 0 ? 'warning' : 'success',
          duration: 5000,
        });
      } else {
        toast({
          title: 'Feed Failed',
          description: 'Could not feed any documents through printer.',
          status: 'error',
          duration: 5000,
        });
      }
    } catch (err: any) {
      toast({
        title: 'Feed Failed',
        description: err.response?.data?.message || err.message || 'Could not feed documents through printer',
        status: 'error',
        duration: 5000,
      });
    } finally {
      setIsFeedingDocuments(false);
    }
  };

  // Voice command handler for orchestration modal
  const triggerPrint = async () => {
    // Use the new orchestrate modal
    openOrchestrateModal();
  };

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const testPrinter = async () => {
    try {
      // Show a loading toast
      const loadingToast = toast({
        title: 'Running Diagnostics...',
        description: 'Checking printer configuration and connectivity',
        status: 'info',
        duration: null,
        isClosable: true,
      });

      // Call the diagnostics endpoint
      const response = await apiClient.get('/printer/diagnostics');

      // Close loading toast
      toast.close(loadingToast);

      // Display results
      if (response.data.success) {
        toast({
          title: 'Diagnostics Complete ✅',
          description: 'Printer is functioning properly',
          status: 'success',
          duration: 5,
        });
      } else {
        toast({
          title: 'Diagnostics Complete ⚠️',
          description: 'Printer may have issues - check details',
          status: 'warning',
          duration: 5,
        });
      }

      setSelectedFile(null);
      console.log('Full Diagnostics Output:', response.data.output);
      console.log('Printer Status:', response.data);
    } catch (err: any) {
      toast({
        title: 'Diagnostics Failed',
        description: err.message || 'Could not run printer diagnostics',
        status: 'error',
        duration: 5,
      });
      console.error('Diagnostics error:', err);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const handleRefreshClick = (e?: React.MouseEvent<HTMLButtonElement>) => {
    if (e) {
      e.preventDefault();
    }
    loadFiles(true);
  };

  const openImageModal = useCallback(
    (filename: string) => {
      setSelectedImageFile(filename);
      imageModal.onOpen();
    },
    [imageModal]
  );

  const closeImageModal = useCallback(() => {
    setSelectedImageFile(null);
    imageModal.onClose();
  }, [imageModal]);

  // Conversion handlers
  const toggleSelectionMode = () => {
    setSelectionMode(!selectionMode);
    setSelectedFiles([]);
    setLastClickedIndex(null);
    setRangeStart(null);
    setRangeEnd(null);
  };

  const handleFileClick = (index: number, filename: string) => {
    if (!selectionMode) return;

    // Simply toggle the file selection
    setSelectedFiles((prev: string[]) => {
      const isSelected = prev.includes(filename);
      const newSelected = isSelected
        ? prev.filter((f: string) => f !== filename)
        : [...prev, filename];

      // Update range based on selected files
      if (newSelected.length >= 2) {
        const selectedIndices = files
          .map((f: FileInfo, idx: number) => ({ filename: f.filename, index: idx }))
          .filter((f: { filename: string; index: number }) => newSelected.includes(f.filename))
          .map((f: { filename: string; index: number }) => f.index)
          .sort((a: number, b: number) => a - b);

        setRangeStart(selectedIndices[0]);
        setRangeEnd(selectedIndices[selectedIndices.length - 1]);
      } else {
        setRangeStart(null);
        setRangeEnd(null);
      }

      return newSelected;
    });

    setLastClickedIndex(index);
  };

  const selectRange = () => {
    if (rangeStart !== null && rangeEnd !== null) {
      const start = Math.min(rangeStart, rangeEnd);
      const end = Math.max(rangeStart, rangeEnd);
      const rangeFiles = files.slice(start, end + 1).map((f: FileInfo) => f.filename);
      setSelectedFiles(rangeFiles);
    }
  };

  const selectOdd = () => {
    if (rangeStart !== null && rangeEnd !== null) {
      const start = Math.min(rangeStart, rangeEnd);
      const end = Math.max(rangeStart, rangeEnd);
      const oddFiles = files
        .slice(start, end + 1)
        .filter((_: FileInfo, idx: number) => idx % 2 === 0) // 0-indexed, so even idx = odd position (1st, 3rd, 5th...)
        .map((f: FileInfo) => f.filename);
      setSelectedFiles(oddFiles);
    }
  };

  const selectEven = () => {
    if (rangeStart !== null && rangeEnd !== null) {
      const start = Math.min(rangeStart, rangeEnd);
      const end = Math.max(rangeStart, rangeEnd);
      const evenFiles = files
        .slice(start, end + 1)
        .filter((_: FileInfo, idx: number) => idx % 2 === 1) // 0-indexed, so odd idx = even position (2nd, 4th, 6th...)
        .map((f: FileInfo) => f.filename);
      setSelectedFiles(evenFiles);
    }
  };

  const selectAll = () => {
    setSelectedFiles(files.map((f: FileInfo) => f.filename));
    setLastClickedIndex(null);
    setRangeStart(null);
    setRangeEnd(null);
  };

  const clearSelection = () => {
    setSelectedFiles([]);
    setLastClickedIndex(null);
    setRangeStart(null);
    setRangeEnd(null);
  };

  const invertSelection = () => {
    const currentlySelected = new Set(selectedFiles);
    const inverted = files
      .filter((f: FileInfo) => !currentlySelected.has(f.filename))
      .map((f: FileInfo) => f.filename);
    setSelectedFiles(inverted);
    setLastClickedIndex(null);
    setRangeStart(null);
    setRangeEnd(null);
  };

  const openConversionModal = () => {
    if (selectedFiles.length === 0) {
      toast({
        title: 'Select files first',
        description: 'Pick at least one document before starting a conversion.',
        status: 'info',
      });
      return;
    }
    conversionModal.onOpen();
  };

  const closeConversionModal = () => {
    setTargetFormat('pdf');
    setConversionProgress('');
    setCustomFilename(''); // Reset custom filename
    conversionModal.onClose();
  };

  const handleConvert = async () => {
    try {
      setConverting(true);
      setConversionProgress('Starting conversion...');

      const payload = {
        files: selectedFiles,
        format: targetFormat,
        merge_pdf: mergePdf && targetFormat === 'pdf',
        filename: customFilename.trim() || undefined,
      };
      console.log('Sending convert request', payload);
      const response = await apiClient.post('/convert', payload);

      if (response.data.success) {
        const { success_count, fail_count, results, merged } = response.data;

        if (merged) {
          setConversionProgress(
            `Merged into single PDF!\n${selectedFiles.length} files combined`
          );
        } else {
          setConversionProgress(
            `Conversion complete!\nSuccess: ${success_count}\nFailed: ${fail_count}`
          );
        }

        // Show detailed results
        const successFiles = results.filter((r: any) => r.success).map((r: any) => r.output);
        if (successFiles.length > 0) {
          setTimeout(() => {
            toast({
              title: merged ? 'Merged PDF ready' : 'Conversion complete',
              description: merged
                ? `${selectedFiles.length} files combined successfully.`
                : `${successFiles.length} files available in converted folder.`,
              status: 'success',
              duration: 6000,
            });
            closeConversionModal();
            setSelectionMode(false);
            setSelectedFiles([]);
            loadConvertedFiles(); // Refresh converted files list
          }, 1500);
        }
      } else {
        setConversionProgress(`Conversion failed: ${response.data.error}`);
      }
    } catch (err: any) {
      console.error('Conversion error:', err);
      // Surface more detailed server error message if available
      const serverMessage =
        err?.response?.data?.error || err?.response?.data?.message || err?.message || 'Unknown error';
      setConversionProgress(`Conversion error: ${serverMessage}`);
      // Also display a toast for user visibility
      toast({
        title: 'Conversion error',
        description: serverMessage,
        status: 'error',
        duration: 6000,
      });
    } finally {
      setConverting(false);
    }
  };

  // Load converted files
  const lastLoadConvertedRunRef = React.useRef<number>(0);
  const minLoadConvertedInterval = 1000; // ms

  const loadConvertedFiles = useCallback(async () => {
    const now = Date.now();
    if (now - lastLoadConvertedRunRef.current < minLoadConvertedInterval) {
      console.log('Skipping loadConvertedFiles due to rate limit');
      return;
    }
    lastLoadConvertedRunRef.current = now;
    if ((loadConvertedFiles as any)._isRunning) {
      console.log('loadConvertedFiles: already running - skipping');
      return;
    }
    (loadConvertedFiles as any)._isRunning = true;
    try {
      const response = await apiClient.get('/get-converted-files');
      if (response.data.files) {
        // Smart cache: only update if file count changed
        const newCount = response.data.files.length;
        if (newCount !== convertedFilesCacheRef.current.lastCount) {
          console.log(
            `Converted file count changed: ${convertedFilesCacheRef.current.lastCount} → ${newCount}, updating cache`
          );
          setConvertedFiles(response.data.files);
          convertedFilesCacheRef.current = {
            data: response.data.files,
            lastCount: newCount,
            timestamp: Date.now(),
          };
        } else {
          console.log(`Converted file count unchanged (${newCount}), using cached data`);
          // Use cached data if available
          if (convertedFilesCacheRef.current.data) {
            setConvertedFiles(convertedFilesCacheRef.current.data);
          } else {
            setConvertedFiles(response.data.files);
          }
        }
      }
    } catch (err) {
      console.error('Failed to load converted files:', err);
    }
    finally {
      (loadConvertedFiles as any)._isRunning = false;
    }
  }, []);

  const handleConvertedDrawerToggle = useCallback(() => {
    if (!convertedDrawer.isOpen) {
      loadConvertedFiles();
    }
    convertedDrawer.onToggle();
  }, [convertedDrawer, loadConvertedFiles]);
  // Prevent scrolling when chat visibility changes
  return (
    <Box position="relative" minH="100dvh">

      <PageShell>
        <DashboardShell
          styleOverrides={{
            mr: isChatVisible ? { base: 0, lg: `${chatWidth + 16}px` } : 0,
            transition: isResizingChat ? 'none' : 'margin-right 0.3s ease-out',
            minH: '100vh',
            pt: { base: 0, md: 2 },
            pb: 4,
            px: { base: 0, md: 2 },
            pr: isChatVisible ? { base: 0, lg: 0 } : { base: 0, md: 2 },
          }}
        >
          {/* Main Content Area */}
          <VStack align="stretch" spacing={{ base: 3, md: 4 }} pb={4}>
            <DashboardHeroCard
              statusDotColor={statusDotColor}
              statusTextColor={statusTextColor}
              statusText={statusText}
              error={error}
              onRefresh={handleRefreshClick}
              onCheckConnectivity={!connected ? handleReconnect : connectivityModal.onOpen}
            >
              <DashboardActionPanel
                isChatVisible={isChatVisible}
                selectionMode={selectionMode}
                selectedFilesCount={selectedFiles.length}
                orchestrateMode={orchestrateMode}
                showReopenOrchestrate={showReopenOrchestrateButton}
                onTriggerPrint={triggerPrint}
                onToggleChat={handleChatVisibilityToggle}
                onToggleSelectionMode={toggleSelectionMode}
                onOpenConversionModal={openConversionModal}
                onCheckConnectivity={connectivityModal.onOpen}
                onToggleConvertedDrawer={handleConvertedDrawerToggle}
                isConvertedDrawerOpen={convertedDrawer.isOpen}
                onReopenOrchestrate={reopenOrchestrateModal}
                embedded={true}
              />
            </DashboardHeroCard>
          </VStack>

          {error && (
            <Box
              borderRadius="xl"
              bg="rgba(255, 170, 0, 0.08)"
              border="1px solid rgba(255,170,0,0.2)"
              p={4}
            >
              <Text color="orange.300" fontWeight="600">
                {error}
              </Text>
            </Box>
          )}

          {processingProgress && (
            <Card bg={surfaceCard} border="1px solid rgba(121,95,238,0.15)" boxShadow="subtle">
              <CardHeader>
                <Heading size="sm">
                  Processing · Step {processingProgress.step}/{processingProgress.total_steps} ·{' '}
                  {processingProgress.stage_name}
                </Heading>
              </CardHeader>
              <CardBody>
                <Stack spacing={3}>
                  <Progress
                    value={(processingProgress.step / processingProgress.total_steps) * 100}
                    colorScheme="brand"
                    borderRadius="full"
                    height="10px"
                  />
                  <Text color="text.muted">{processingProgress.message}</Text>
                </Stack>
              </CardBody>
            </Card>
          )}

          {loading ? (
            <Flex align="center" justify="center" py={12}>
              <Spinner size="xl" thickness="5px" color="brand.400" />
            </Flex>
          ) : (
            <Stack spacing={10}>
              <Box>
                <Flex justify="space-between" align="baseline" mb={4}>
                  <Heading size="md">Files · {files.length}</Heading>
                  {selectionMode && (
                    <Tag size="lg" colorScheme="purple" borderRadius="full">
                      {selectedFiles.length} selected
                    </Tag>
                  )}
                </Flex>

                {/* Selection Buttons - Simple multi-select */}
                {selectionMode && files.length > 0 && (
                  <Stack spacing={3} mb={6}>
                    <Flex gap={2} wrap="wrap" align="center">
                      {/* Always show basic buttons */}
                      <Button size="sm" colorScheme="blue" variant="outline" onClick={selectAll}>
                        Select All
                      </Button>

                      <Button
                        size="sm"
                        colorScheme="gray"
                        variant="outline"
                        onClick={clearSelection}
                        isDisabled={selectedFiles.length === 0}
                      >
                        Clear All
                      </Button>

                      {/* Show Select Range button when user has selected 2+ different items */}
                      {selectedFiles.length >= 2 && rangeStart !== null && rangeEnd !== null && (
                        <Button
                          size="sm"
                          colorScheme="green"
                          variant="solid"
                          onClick={selectRange}
                          leftIcon={<Box as="span">📍</Box>}
                          boxShadow="0 0 20px rgba(72, 187, 120, 0.4)"
                        >
                          Select Range ({rangeEnd - rangeStart + 1} items)
                        </Button>
                      )}

                      {/* Show Odd/Even buttons after a range is selected */}
                      {selectedFiles.length > 1 && rangeStart !== null && rangeEnd !== null && (
                        <>
                          <Button size="sm" colorScheme="cyan" variant="outline" onClick={selectOdd}>
                            Select Odd
                          </Button>

                          <Button size="sm" colorScheme="cyan" variant="outline" onClick={selectEven}>
                            Select Even
                          </Button>
                        </>
                      )}
                    </Flex>
                  </Stack>
                )}

                {files.length === 0 ? (
                  <Card
                    border="1px solid rgba(121,95,238,0.2)"
                    bg={surfaceCard}
                    textAlign="center"
                    py={10}
                  >
                    <CardBody>
                      <Stack spacing={3} align="center">
                        <Iconify icon={FiFileText} boxSize={10} color="brand.300" />
                        <Heading size="sm">No files yet</Heading>
                        <Text color="text.muted" maxW="md">
                          Initiate a capture from the Phone interface or trigger a print job to start
                          populating this space.
                        </Text>
                      </Stack>
                    </CardBody>
                  </Card>
                ) : (
                  <SimpleGrid columns={{ base: 1, md: 2, xl: 3 }} spacing={6}>
                    {files.map((file: FileInfo, index: number) => {
                      const isSelected = selectedFiles.includes(file.filename);
                      const documentIndex = index + 1;
                      return (
                        <MotionCard
                          key={file.filename}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.3, delay: index * 0.05 }}
                          whileHover={{
                            y: -8,
                            scale: 1.02,
                            transition: { duration: 0.2 },
                          }}
                          borderRadius="2xl"
                          border={`2px solid ${isSelected ? 'rgba(72, 187, 120, 0.6)' : 'rgba(121,95,238,0.18)'}`}
                          boxShadow={isSelected ? '0 0 20px rgba(72, 187, 120, 0.4)' : 'subtle'}
                          bg={surfaceCard}
                          position="relative"
                          overflow="hidden"
                          cursor={selectionMode ? 'pointer' : 'default'}
                          onClick={(e: React.MouseEvent) => {
                            if (selectionMode && !file.processing) {
                              handleFileClick(index, file.filename);
                            }
                          }}
                        >
                          {selectionMode && !file.processing && (
                            <Checkbox
                              position="absolute"
                              top={4}
                              left={4}
                              colorScheme="brand"
                              size="lg"
                              borderRadius="md"
                              isChecked={isSelected}
                              onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                                e.stopPropagation();
                                handleFileClick(index, file.filename);
                              }}
                            />
                          )}

                          <CardHeader>
                            <Stack spacing={2}>
                              <HStack spacing={2} align="center">
                                <Badge colorScheme="purple" borderRadius="full" px={2} py={0.5}>
                                  #{documentIndex}
                                </Badge>
                                <Heading size="sm" noOfLines={1} title={file.filename}>
                                  {file.filename}
                                </Heading>
                              </HStack>
                            </Stack>
                          </CardHeader>

                          <CardBody>
                            <Box
                              position="relative"
                              borderRadius="xl"
                              overflow="hidden"
                              border="1px solid rgba(121,95,238,0.18)"
                              _hover={{
                                borderColor: 'brand.300',
                                transform: file.processing ? undefined : 'translateY(-4px)',
                                transition: 'all 0.3s ease',
                              }}
                            >
                              <Box h="220px" bg="surface.blur">
                                <SecureImage
                                  filename={file.filename}
                                  alt={file.filename}
                                  onClick={() => !file.processing && openImageModal(file.filename)}
                                  refreshToken={refreshToken}
                                />
                              </Box>

                              {file.processing && (
                                <Flex
                                  position="absolute"
                                  inset={0}
                                  align="center"
                                  justify="center"
                                  direction="column"
                                  bg="rgba(4,7,19,0.76)"
                                  color="white"
                                  gap={3}
                                >
                                  <Spinner size="lg" thickness="4px" color="brand.200" />
                                  <Text fontWeight="600">
                                    Processing {file.processing_step}/{file.processing_total}
                                  </Text>
                                  <Text fontSize="sm" color="rgba(255,255,255,0.75)">
                                    {file.processing_stage}
                                  </Text>
                                </Flex>
                              )}
                            </Box>
                          </CardBody>

                          <CardFooter>
                            <Flex w="100%" justify="space-between" align="center">
                              <Stack spacing={1}>
                                {file.processing && (
                                  <Tag size="md" colorScheme="purple" borderRadius="full">
                                    Active pipeline
                                  </Tag>
                                )}
                                {!file.processing && ocrResults[file.filename] && (
                                  <OCRReadyBadge
                                    title={ocrResults[file.filename].derived_title}
                                    onClick={() => handleViewOCRResult(file.filename)}
                                  />
                                )}

                                {/* Badge for text available */}
                                {!file.processing && !ocrResults[file.filename] && file.has_text && (
                                  <Badge colorScheme="blue" borderRadius="full" px={2}>
                                    Text available
                                  </Badge>
                                )}
                              </Stack>

                              <ButtonGroup size="sm" variant="ghost" spacing={1}>
                                {!file.processing && (
                                  <Tooltip label="View" hasArrow>
                                    <IconButton
                                      aria-label="View"
                                      icon={<Iconify icon={FiZoomIn} boxSize={5} />}
                                      onClick={() => openImageModal(file.filename)}
                                    />
                                  </Tooltip>
                                )}
                                {/* Run PaddleOCR button - shows when pipeline complete but no OCR yet */}
                                {!file.processing && !ocrResults[file.filename] && (
                                  <Tooltip
                                    label={
                                      ocrLoading[file.filename]
                                        ? `OCR Processing... ${ocrProgress[file.filename]?.elapsed ? `(${(ocrProgress[file.filename].elapsed / 1000).toFixed(1)}s)` : ''}`
                                        : 'Run OCR (PaddleOCR)'
                                    }
                                    hasArrow
                                  >
                                    <Button
                                      aria-label="Run OCR"
                                      onClick={() => handleRunOCR(file.filename)}
                                      isLoading={!!ocrLoading[file.filename]}
                                      colorScheme={ocrLoading[file.filename] ? 'blue' : 'brown'}
                                      variant="solid"
                                      size="sm"
                                      borderRadius="full"
                                      minW="44px"
                                      h="38px"
                                      px={3}
                                      loadingText={ocrLoading[file.filename] ? `${(ocrProgress[file.filename]?.elapsed / 1000).toFixed(0)}s` : undefined}
                                    >
                                      {ocrLoading[file.filename] ? '⏳' : 'OCR'}
                                    </Button>
                                  </Tooltip>
                                )}
                                {/* View OCR result button - shows when OCR is complete */}
                                {ocrResults[file.filename] && !file.processing && (
                                  <Tooltip label="View OCR Result" hasArrow>
                                    <IconButton
                                      aria-label="View OCR"
                                      icon={<Iconify icon={FiFileText} boxSize={5} />}
                                      onClick={() => handleViewOCRResult(file.filename)}
                                      colorScheme="green"
                                    />
                                  </Tooltip>
                                )}
                                {/* Legacy OCR view for files with existing text */}
                                {file.has_text && !ocrResults[file.filename] && !file.processing && (
                                  <Tooltip label="View Legacy OCR" hasArrow>
                                    <IconButton
                                      aria-label="Legacy OCR"
                                      icon={<Iconify icon={FiFileText} boxSize={5} />}
                                      onClick={() => viewOCR(file.filename)}
                                      variant="outline"
                                    />
                                  </Tooltip>
                                )}
                                <Tooltip label="Delete" hasArrow>
                                  <IconButton
                                    aria-label="Delete"
                                    icon={<Iconify icon={FiTrash2} boxSize={5} />}
                                    onClick={() => deleteFile(file.filename)}
                                    isDisabled={file.processing}
                                  />
                                </Tooltip>
                              </ButtonGroup>
                            </Flex>
                          </CardFooter>
                        </MotionCard>
                      );
                    })}
                  </SimpleGrid>
                )}
              </Box>

              {selectedFile && ocrText && (
                <Card border="1px solid rgba(69,202,255,0.25)" bg={surfaceCard} boxShadow="subtle">
                  <CardHeader display="flex" alignItems="center" justifyContent="space-between">
                    <Heading size="sm">OCR · {selectedFile}</Heading>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => {
                        setSelectedFile(null);
                        setOcrText('');
                      }}
                    >
                      Close
                    </Button>
                  </CardHeader>
                  <CardBody>
                    <Box
                      bg="rgba(15,20,42,0.88)"
                      color="white"
                      borderRadius="2xl"
                      p={5}
                      fontFamily="mono"
                      fontSize="sm"
                      maxH="320px"
                      overflowY="auto"
                      boxShadow="inset 0 0 0 1px rgba(69,202,255,0.18)"
                    >
                      {ocrText}
                    </Box>
                  </CardBody>
                </Card>
              )}
            </Stack>
          )}

          <Modal
            isOpen={imageModal.isOpen && Boolean(selectedImageFile)}
            onClose={closeImageModal}
            size={{ base: 'full', sm: 'xl', md: '2xl', lg: '3xl', xl: '4xl' }}
            scrollBehavior="inside"
            isCentered
          >
            <ModalOverlay bg="blackAlpha.600" />
            <ModalContent
              bg={surfaceCard}
              borderRadius="2xl"
              border="1px solid rgba(121,95,238,0.25)"
              boxShadow="halo"
              maxH={{ base: '95vh', md: '90vh' }}
              m={{ base: 2, md: 4 }}
              overflow="hidden"
            >
              <ModalHeader>{selectedImageFile}</ModalHeader>
              <ModalCloseButton borderRadius="full" />
              <ModalBody
                p={{ base: 2, md: 4 }}
                display="flex"
                alignItems="center"
                justifyContent="center"
                overflow="hidden"
              >
                {selectedImageFile && (
                  <ModalImageWithHeaders
                    filename={selectedImageFile}
                    alt={selectedImageFile}
                    refreshToken={refreshToken}
                  />
                )}
              </ModalBody>
              <ModalFooter>
                <Button variant="ghost" mr={3} onClick={closeImageModal}>
                  Close
                </Button>
                <Button
                  colorScheme="brand"
                  leftIcon={<Iconify icon={FiDownload} boxSize={5} />}
                  onClick={async () => {
                    if (!selectedImageFile) return;
                    try {
                      const imageUrl = `${API_BASE_URL}${API_ENDPOINTS.processed}/${selectedImageFile}`;
                      const headers: Record<string, string> = {};

                      // Add ngrok bypass header if using ngrok tunnel
                      if (API_BASE_URL.includes('ngrok') || API_BASE_URL.includes('loca.lt')) {
                        headers['ngrok-skip-browser-warning'] = 'true';
                      }

                      const response = await fetch(imageUrl, { headers });

                      if (!response.ok) {
                        throw new Error(`Failed to fetch: ${response.status}`);
                      }

                      const blob = await response.blob();
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = selectedImageFile;
                      document.body.appendChild(a);
                      a.click();
                      document.body.removeChild(a);
                      URL.revokeObjectURL(url);
                    } catch (err: any) {
                      console.error('Download failed:', err);
                      toast({
                        title: 'Download failed',
                        description: err.message || 'Unable to download file',
                        status: 'error',
                        duration: 4000,
                      });
                    }
                  }}
                >
                  Download
                </Button>
              </ModalFooter>
            </ModalContent>
          </Modal>

          {/* PaddleOCR Structured View Modal */}
          <Modal
            isOpen={Boolean(activeOCRView)}
            onClose={closeOCRView}
            size="6xl"
            scrollBehavior="inside"
            isCentered
          >
            <ModalOverlay bg="blackAlpha.600" />
            <ModalContent
              bg={surfaceCard}
              borderRadius="2xl"
              border="1px solid rgba(72, 187, 120, 0.25)"
              boxShadow="halo"
              maxH={{ base: '95vh', md: '90vh' }}
              m={{ base: 2, md: 4 }}
              overflow="hidden"
            >
              <ModalHeader>
                <HStack spacing={3}>
                  <Iconify icon="mdi:text-recognition" boxSize={6} color="green.400" />
                  <Text>
                    OCR Result: {activeOCRView && ocrResults[activeOCRView]?.derived_title
                      ? ocrResults[activeOCRView].derived_title
                      : activeOCRView}
                  </Text>
                </HStack>
              </ModalHeader>
              <ModalCloseButton borderRadius="full" />
              <ModalBody p={{ base: 2, md: 4 }} overflow="auto">
                {activeOCRView && ocrResults[activeOCRView] && (
                  <OCRStructuredView
                    result={ocrResults[activeOCRView]}
                    filename={activeOCRView}
                    imageUrl={`${API_BASE_URL}${API_ENDPOINTS.processed}/${activeOCRView}`}
                    onRetry={() => {
                      if (activeOCRView) {
                        handleRunOCR(activeOCRView);
                      }
                    }}
                    isRetrying={!!(activeOCRView && ocrLoading[activeOCRView])}
                    onClose={closeOCRView}
                  />
                )}
              </ModalBody>
              <ModalFooter>
                <Button variant="ghost" mr={3} onClick={closeOCRView}>
                  Close
                </Button>
                {/* Show retry button in footer if OCR failed */}
                {activeOCRView && ocrResults[activeOCRView] && 
                  (ocrResults[activeOCRView].word_count === 0 || !ocrResults[activeOCRView].full_text) && (
                  <Button
                    colorScheme="blue"
                    leftIcon={<Iconify icon="mdi:refresh" boxSize={5} />}
                    mr={3}
                    onClick={() => {
                      if (activeOCRView) {
                        handleRunOCR(activeOCRView);
                      }
                    }}
                    isLoading={!!(activeOCRView && ocrLoading[activeOCRView])}
                    loadingText="Retrying..."
                  >
                    Retry OCR
                  </Button>
                )}
                {activeOCRView && ocrResults[activeOCRView]?.full_text && (
                  <Button
                    colorScheme="brand"
                    leftIcon={<Iconify icon="mdi:content-copy" boxSize={5} />}
                    onClick={() => {
                      if (activeOCRView && ocrResults[activeOCRView]?.full_text) {
                        navigator.clipboard.writeText(ocrResults[activeOCRView].full_text);
                        toast({
                          title: 'Copied',
                          description: 'OCR text copied to clipboard',
                          status: 'success',
                          duration: 2000,
                        });
                      }
                    }}
                  >
                    Copy Text
                  </Button>
                )}
              </ModalFooter>
            </ModalContent>
          </Modal>

          <Modal isOpen={conversionModal.isOpen} onClose={closeConversionModal} size="lg">
            <ModalOverlay bg="blackAlpha.600" />
            <ModalContent bg={surfaceCard} borderRadius="2xl" border="1px solid rgba(121,95,238,0.25)">
              <ModalHeader>Convert Files</ModalHeader>
              <ModalCloseButton />
              <ModalBody>
                <Stack spacing={5}>
                  <Box>
                    <Text fontWeight="600">Selected Files: {selectedFiles.length}</Text>
                    <Box
                      mt={2}
                      maxH="160px"
                      overflowY="auto"
                      bg="surface.blur"
                      borderRadius="lg"
                      p={3}
                      border="1px solid rgba(121,95,238,0.2)"
                    >
                      <Stack spacing={2} fontSize="sm">
                        {selectedFiles.map((filename: string, index: number) => (
                          <Text key={filename}>
                            #{index + 1} · {filename}
                          </Text>
                        ))}
                      </Stack>
                    </Box>
                  </Box>

                  <Box>
                    <Text fontWeight="600" mb={2}>
                      Convert to
                    </Text>
                    <Select
                      value={targetFormat}
                      onChange={(e: React.ChangeEvent<HTMLSelectElement>) =>
                        setTargetFormat(e.target.value)
                      }
                      isDisabled={converting}
                    >
                      <option value="pdf">PDF</option>
                      <option value="png">PNG</option>
                      <option value="jpg">JPG</option>
                      <option value="docx">Word Document (DOCX)</option>
                    </Select>
                  </Box>

                  {targetFormat === 'pdf' && selectedFiles.length > 1 && (
                    <Flex
                      align="center"
                      justify="space-between"
                      bg="surface.blur"
                      borderRadius="lg"
                      p={3}
                      border="1px solid rgba(69,202,255,0.25)"
                    >
                      <Stack spacing={1}>
                        <Text fontWeight="600">Merge into single PDF</Text>
                        <Text fontSize="sm" color="text.muted">
                          {mergePdf
                            ? 'All files will merge into one premium PDF output.'
                            : 'Each file becomes an individual PDF.'}
                        </Text>
                      </Stack>
                      <Checkbox
                        isChecked={mergePdf}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                          setMergePdf(e.target.checked)
                        }
                        isDisabled={converting}
                        colorScheme="brand"
                      />
                    </Flex>
                  )}

                  {targetFormat === 'pdf' && mergePdf && selectedFiles.length > 1 && (
                    <Box>
                      <Text fontWeight="600" mb={2}>
                        PDF Filename (optional)
                      </Text>
                      <Input
                        placeholder="Enter custom filename (without .pdf extension)"
                        value={customFilename}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                          setCustomFilename(e.target.value)
                        }
                        isDisabled={converting}
                        maxLength={50}
                      />
                      <Text fontSize="xs" color="text.muted" mt={1}>
                        If empty, will use auto-generated name like
                        "merged_document_20251021_123456.pdf"
                      </Text>
                    </Box>
                  )}

                  {conversionProgress && (
                    <Box
                      bg="rgba(69,202,255,0.1)"
                      borderRadius="lg"
                      p={3}
                      border="1px solid rgba(69,202,255,0.2)"
                    >
                      <Text fontFamily="mono" whiteSpace="pre-wrap" color="brand.200">
                        {conversionProgress}
                      </Text>
                    </Box>
                  )}
                </Stack>
              </ModalBody>
              <ModalFooter>
                <Button variant="ghost" mr={3} onClick={closeConversionModal} isDisabled={converting}>
                  Cancel
                </Button>
                <Button
                  colorScheme="brand"
                  onClick={handleConvert}
                  isLoading={converting}
                  loadingText="Converting"
                >
                  Convert
                </Button>
              </ModalFooter>
            </ModalContent>
          </Modal>

          <Drawer
            isOpen={convertedDrawer.isOpen}
            placement="right"
            onClose={convertedDrawer.onClose}
            size="md"
          >
            <DrawerOverlay bg="blackAlpha.600" />
            <DrawerContent bg={surfaceCard} borderColor="rgba(121,95,238,0.25)" borderLeftWidth="1px">
              <DrawerHeader display="flex" alignItems="center" justifyContent="space-between">
                <Heading size="sm">Converted Files</Heading>
                <Button size="sm" variant="ghost" onClick={convertedDrawer.onClose}>
                  Close
                </Button>
              </DrawerHeader>
              <DrawerBody>
                {convertedFiles.length === 0 ? (
                  <Flex direction="column" align="center" justify="center" h="full" color="text.muted">
                    <Iconify icon={FiFileText} boxSize={12} mb={4} />
                    <Text>No converted artifacts yet.</Text>
                  </Flex>
                ) : (
                  <Stack spacing={4}>
                    {convertedFiles.map((file: any, index: number) => {
                      const convertedIndex = index + 1;
                      return (
                        <Card
                          key={file.filename}
                          borderRadius="xl"
                          border="1px solid rgba(69,202,255,0.18)"
                        >
                          <CardBody>
                            <Stack spacing={3}>
                              <Stack spacing={1}>
                                <HStack spacing={2} align="center">
                                  <Badge colorScheme="cyan" borderRadius="full" px={2} py={0.5}>
                                    #{convertedIndex}
                                  </Badge>
                                  <Heading size="sm">{file.filename}</Heading>
                                </HStack>
                                <Text fontSize="xs" color="text.muted">
                                  {(file.size / 1024).toFixed(2)} KB ·{' '}
                                  {new Date(file.created).toLocaleString()}
                                </Text>
                              </Stack>
                              <Flex gap={2} wrap="wrap">
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  flex={1}
                                  minW="120px"
                                  leftIcon={<Iconify icon={FiDownload} boxSize={4} />}
                                  onClick={async () => {
                                    try {
                                      const response = await apiClient.get(
                                        `/converted/${file.filename}`,
                                        {
                                          responseType: 'blob',
                                        }
                                      );
                                      const blob = response.data;
                                      const url = URL.createObjectURL(blob);
                                      const a = document.createElement('a');
                                      a.href = url;
                                      a.download = file.filename;
                                      document.body.appendChild(a);
                                      a.click();
                                      document.body.removeChild(a);
                                      URL.revokeObjectURL(url);
                                    } catch (err: any) {
                                      console.error('Download failed:', err);
                                      toast({
                                        title: 'Download failed',
                                        description: err.message,
                                        status: 'error',
                                      });
                                    }
                                  }}
                                >
                                  Download
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  flex={1}
                                  minW="120px"
                                  colorScheme="red"
                                  leftIcon={<Iconify icon={FiTrash2} boxSize={4} />}
                                  onClick={async () => {
                                    if (!window.confirm(`Delete ${file.filename}?`)) return;

                                    try {
                                      await apiClient.delete(`/delete-converted/${file.filename}`);
                                      setConvertedFiles(
                                        convertedFiles.filter((f: any) => f.filename !== file.filename)
                                      );
                                      toast({
                                        title: 'File deleted',
                                        description: `${file.filename} has been deleted.`,
                                        status: 'success',
                                      });
                                    } catch (err: any) {
                                      console.error('Delete failed:', err);
                                      toast({
                                        title: 'Delete failed',
                                        description: err.message,
                                        status: 'error',
                                      });
                                    }
                                  }}
                                >
                                  Delete
                                </Button>
                              </Flex>
                            </Stack>
                          </CardBody>
                        </Card>
                      );
                    })}
                  </Stack>
                )}
              </DrawerBody>
              <DrawerFooter>
                <Button variant="outline" width="full" onClick={convertedDrawer.onClose}>
                  Close Drawer
                </Button>
              </DrawerFooter>
            </DrawerContent>
          </Drawer>

          {/* Orchestrate Print & Capture Modal - Only render when DocumentSelector is closed */}
          {!documentSelectorModal.isOpen && (
            <Modal
              isOpen={orchestrateModal.isOpen}
              onClose={orchestrateModal.onClose}
              size={isVoiceOrchestration ? 'full' : '6xl'}
              isCentered={!isVoiceOrchestration}
              scrollBehavior={isVoiceOrchestration ? 'outside' : 'inside'}
              motionPreset={isVoiceOrchestration ? 'slideInBottom' : 'scale'}
              trapFocus={!isChatVisible}
              preserveScrollBarGap
              blockScrollOnMount={true}
            >
              <ModalOverlay
                backdropFilter="none"
                bg={isVoiceOrchestration ? 'transparent' : 'blackAlpha.500'}
                zIndex={isChatVisible ? 2001 : undefined}
                pointerEvents="none"
              />
              <MotionModalContent
                bg={surfaceCard}
                borderRadius={isVoiceOrchestration ? '2xl' : { base: 'xl', md: '2xl', lg: '3xl' }}
                border="1px solid"
                borderColor="brand.300"
                boxShadow="0 25px 60px rgba(121, 95, 238, 0.4)"
                maxH={MODAL_CONFIG.modal.maxHeight}
                maxW={isChatVisible ? `calc(100vw - ${chatWidth + 20}px)` : MODAL_CONFIG.modal.maxWidth}
                w={isChatVisible ? `calc(100vw - ${chatWidth + 20}px)` : 'auto'}
                h="auto"
                ml={isChatVisible ? '8px' : 'auto'}
                mr={isChatVisible ? `${chatWidth + 12}px` : 'auto'}
                mt={isChatVisible ? '8px' : 'auto'}
                my={isVoiceOrchestration ? '5vh' : 'auto'}
                overflow="hidden"
                display="flex"
                flexDirection="column"
                zIndex={2050}
                pointerEvents="auto"
                initial={{ opacity: 0, scale: 0.9, y: 20 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.9, y: 20 }}
                transition={{ duration: 0.3, ease: 'easeOut' }}
              >
                <Flex
                  h="100%"
                  w="100%"
                  direction={{ base: 'column', xl: isVoiceOrchestration ? 'row' : 'column' }}
                  overflow="hidden"
                  pointerEvents="auto"
                >
                  <Box flex="1" display="flex" flexDirection="column" overflow="hidden" pointerEvents="auto">
                    {/* STEP 1: Choose Mode */}
                    {orchestrateStep === 1 && (
                      <>
                        <ModalHeader
                          fontSize="3xl"
                          fontWeight="700"
                          py="1.5rem"
                          bgGradient="linear(to-r, brand.500, nebula.500)"
                          bgClip="text"
                          display="flex"
                          alignItems="center"
                          gap={3}
                        >
                          <Iconify icon="solar:settings-bold-duotone" width={32} height={32} />
                          Orchestrate Print & Capture
                        </ModalHeader>
                        <ModalCloseButton
                          size="lg"
                          top={6}
                          right={6}
                          _hover={{ bg: 'red.500', color: 'white' }}
                        />
                        <ModalBody py={8} px={10}>
                          <Stack spacing={8}>
                            <Text fontSize="lg" color="text.muted" textAlign="center" fontWeight="500">
                              Choose your operation to get started
                            </Text>
                            <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
                              <MotionBox
                                p={8}
                                borderRadius="2xl"
                                border="3px solid"
                                borderColor={orchestrateMode === 'scan' ? 'brand.400' : 'whiteAlpha.200'}
                                bg={orchestrateMode === 'scan' ? 'rgba(121,95,238,0.12)' : 'whiteAlpha.50'}
                                cursor="pointer"
                                onClick={() => setOrchestrateMode('scan')}
                                position="relative"
                                overflow="hidden"
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ duration: 0.3, delay: 0.1 }}
                                whileHover={{
                                  borderColor: 'brand.400',
                                  y: -8,
                                  scale: 1.02,
                                  boxShadow: '0 12px 30px rgba(121,95,238,0.35)',
                                  transition: { duration: 0.2 },
                                }}
                                whileTap={{ scale: 0.98 }}
                              >
                                {orchestrateMode === 'scan' && (
                                  <Box
                                    position="absolute"
                                    top={4}
                                    right={4}
                                    bg="brand.500"
                                    color="white"
                                    borderRadius="full"
                                    p={1}
                                  >
                                    <Iconify icon="solar:check-circle-bold" width={24} height={24} />
                                  </Box>
                                )}
                                <VStack align="start" spacing={4}>
                                  <Box p={4} bg="brand.500" borderRadius="xl" color="white">
                                    <Iconify icon="solar:document-add-bold-duotone" width={40} height={40} />
                                  </Box>
                                  <Heading size="lg">Scan Mode</Heading>
                                  <Text fontSize="md" color="text.muted" lineHeight="1.6">
                                    Capture physical documents from your scanner with advanced options for
                                    quality, layout, and text detection
                                  </Text>
                                </VStack>
                              </MotionBox>
                              <MotionBox
                                p={8}
                                borderRadius="2xl"
                                border="3px solid"
                                borderColor={orchestrateMode === 'print' ? 'brand.400' : 'whiteAlpha.200'}
                                bg={orchestrateMode === 'print' ? 'rgba(121,95,238,0.12)' : 'whiteAlpha.50'}
                                cursor="pointer"
                                onClick={() => setOrchestrateMode('print')}
                                position="relative"
                                overflow="hidden"
                                initial={{ opacity: 0, x: 20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ duration: 0.3, delay: 0.2 }}
                                whileHover={{
                                  borderColor: 'brand.400',
                                  y: -8,
                                  scale: 1.02,
                                  boxShadow: '0 12px 30px rgba(121,95,238,0.35)',
                                  transition: { duration: 0.2 },
                                }}
                                whileTap={{ scale: 0.98 }}
                              >
                                {orchestrateMode === 'print' && (
                                  <Box
                                    position="absolute"
                                    top={4}
                                    right={4}
                                    bg="brand.500"
                                    color="white"
                                    borderRadius="full"
                                    p={1}
                                  >
                                    <Iconify icon="solar:check-circle-bold" width={24} height={24} />
                                  </Box>
                                )}
                                <VStack align="start" spacing={4}>
                                  <Box p={4} bg="nebula.500" borderRadius="xl" color="white">
                                    <Iconify icon="solar:printer-bold-duotone" width={40} height={40} />
                                  </Box>
                                  <Heading size="lg">Print Mode</Heading>
                                  <Text fontSize="md" color="text.muted" lineHeight="1.6">
                                    Print documents from your collection with precise control over layout,
                                    margins, scaling, and page selection
                                  </Text>
                                </VStack>
                              </MotionBox>
                            </SimpleGrid>
                          </Stack>
                        </ModalBody>
                        <ModalFooter
                          py="1.5rem"
                          px="2.5rem"
                          borderTop="1px solid"
                          borderColor="whiteAlpha.200"
                        >
                          <Button variant="ghost" mr={3} onClick={orchestrateModal.onClose} size="lg">
                            Cancel
                          </Button>
                          <Button
                            colorScheme="brand"
                            onClick={() => {
                              // Default to feed tray for scan mode (skip document source selection)
                              if (orchestrateMode === 'scan') {
                                setScanDocumentSource('feed');
                              }
                              setOrchestrateStep(2);
                            }}

                            isDisabled={!orchestrateMode}
                            size="lg"
                            px={8}
                            rightIcon={<Iconify icon="solar:arrow-right-bold" width={20} height={20} />}
                          >
                            Continue
                          </Button>
                        </ModalFooter>
                      </>
                    )}

                    {/* STEP 2: Scan Options */}
                    {orchestrateStep === 2 && orchestrateMode === 'scan' && (
                      <>
                        <ModalHeader
                          fontSize="2xl"
                          fontWeight="700"
                          py="1.5rem"
                          display="flex"
                          alignItems="center"
                          gap={3}
                          borderBottom="1px solid"
                          borderColor="whiteAlpha.200"
                        >
                          <Box p={2} bg="brand.500" borderRadius="lg" color="white">
                            <Iconify icon="solar:document-add-bold-duotone" width={24} height={24} />
                          </Box>
                          {scanDocumentSource === null ? 'Select Document Source' : 'Scan Configuration'}
                          <Badge colorScheme="purple" fontSize="sm" px={3} py={1}>
                            Step 2 of 3
                          </Badge>
                        </ModalHeader>
                        <ModalCloseButton
                          size="lg"
                          top={6}
                          right={6}
                          _hover={{ bg: 'red.500', color: 'white' }}
                        />

                        {/* Sub-step: Choose document source (select/upload OR feed tray) */}
                        {scanDocumentSource === null && (
                          <>
                            <ModalBody py={8} px={6} pb={{ base: 24, md: 12 }}>
                              <VStack spacing={6} align="stretch">
                                <Text fontSize="lg" color="text.secondary" textAlign="center" mb={4}>
                                  How would you like to provide documents for scanning?
                                </Text>

                                <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
                                  {/* Option 1: Select/Upload Documents */}
                                  <Box
                                    as="button"
                                    p={8}
                                    borderRadius="2xl"
                                    border="3px solid"
                                    borderColor="brand.400"
                                    bg="rgba(121,95,238,0.08)"
                                    _hover={{
                                      borderColor: 'brand.500',
                                      bg: 'rgba(121,95,238,0.15)',
                                      transform: 'translateY(-4px)',
                                      boxShadow: '0 12px 30px rgba(121,95,238,0.3)',
                                    }}
                                    transition="all 0.3s"
                                    onClick={() => {
                                      setScanDocumentSource('select');
                                      documentSelectorModal.onOpen();
                                    }}
                                    textAlign="center"
                                  >
                                    <VStack spacing={4}>
                                      <Box
                                        p={4}
                                        borderRadius="full"
                                        bg="brand.500"
                                        color="white"
                                      >
                                        <Iconify icon="solar:upload-bold-duotone" width={48} height={48} />
                                      </Box>
                                      <Heading size="md" color="text.primary">
                                        Select or Upload Document
                                      </Heading>
                                      <Text fontSize="sm" color="text.muted">
                                        Choose from existing documents or upload new files from your device
                                      </Text>
                                      <HStack spacing={2} flexWrap="wrap" justify="center">
                                        <Badge colorScheme="purple">PDF</Badge>
                                        <Badge colorScheme="blue">JPG</Badge>
                                        <Badge colorScheme="green">PNG</Badge>
                                      </HStack>
                                    </VStack>
                                  </Box>

                                  {/* Option 2: Use Feed Tray */}
                                  <Box
                                    as="button"
                                    p={8}
                                    borderRadius="2xl"
                                    border="3px solid"
                                    borderColor="orange.400"
                                    bg="rgba(237,137,54,0.08)"
                                    _hover={{
                                      borderColor: 'orange.500',
                                      bg: 'rgba(237,137,54,0.15)',
                                      transform: 'translateY(-4px)',
                                      boxShadow: '0 12px 30px rgba(237,137,54,0.3)',
                                    }}
                                    transition="all 0.3s"
                                    onClick={() => setScanDocumentSource('feed')}
                                    textAlign="center"
                                  >
                                    <VStack spacing={4}>
                                      <Box
                                        p={4}
                                        borderRadius="full"
                                        bg="orange.500"
                                        color="white"
                                      >
                                        <Iconify icon="solar:printer-bold-duotone" width={48} height={48} />
                                      </Box>
                                      <Heading size="md" color="text.primary">
                                        Use Printer Feed Tray
                                      </Heading>
                                      <Text fontSize="sm" color="text.muted">
                                        Feed physical documents through the printer's input tray for scanning
                                      </Text>
                                      <HStack spacing={2}>
                                        <Badge colorScheme="orange">Physical Documents</Badge>
                                      </HStack>
                                    </VStack>
                                  </Box>
                                </SimpleGrid>
                              </VStack>
                            </ModalBody>
                            <ModalFooter
                              py="1.5rem"
                              px="2.5rem"
                              borderTop="1px solid"
                              borderColor="whiteAlpha.200"
                              position="sticky"
                              bottom={0}
                              bg={surfaceCard}
                              zIndex={30}
                            >
                              <Button
                                variant="ghost"
                                onClick={() => setOrchestrateStep(1)}
                                size="lg"
                                leftIcon={<Iconify icon="solar:arrow-left-bold" width={20} height={20} />}
                              >
                                Back to Mode Selection
                              </Button>
                            </ModalFooter>
                          </>
                        )}

                        {/* Scan options after document source is selected */}
                        {scanDocumentSource !== null && (
                          <>
                            <ModalBody
                              ref={modalBodyRef}
                              data-orchestration-scroll="scan"
                              py="1rem"
                              px="1.5rem"
                              pb={{ base: 28, md: 12 }}
                              maxH={`calc(${MODAL_CONFIG.modalBody.maxHeight})`}
                              overflowY="auto"
                              overflowX="hidden"
                              css={{
                                '&::-webkit-scrollbar': { width: '8px' },
                                '&::-webkit-scrollbar-track': { background: 'transparent' },
                                '&::-webkit-scrollbar-thumb': {
                                  background: 'rgba(121,95,238,0.4)',
                                  borderRadius: '10px',
                                },
                                '&::-webkit-scrollbar-thumb:hover': {
                                  background: 'rgba(121,95,238,0.6)',
                                },
                              }}
                            >
                              {renderVoiceAdjustmentPanel('scan')}
                              {renderVoiceAdjustmentPanel('print')}
                              <Grid
                                templateColumns={{ base: '1fr', lg: scanDocumentSource === 'feed' ? '1fr' : '1fr 1fr' }}
                                gap="1.5rem"
                                alignItems="start"
                              >
                                {/* Live Preview - Scan Mode (LEFT SIDE) - Hidden for feed tray mode */}
                                {scanDocumentSource === 'select' && (
                                  <Box
                                    display={{ base: 'none', lg: 'flex' }}
                                    flexDirection="column"
                                    position="sticky"
                                    top={0}
                                    alignSelf="start"
                                    maxH={`calc(${MODAL_CONFIG.previewBox.maxHeight})`}
                                    order={{ base: 2, lg: 1 }}
                                  >
                                    <DocumentPreview
                                      documents={
                                        selectedDocuments.length > 0
                                          ? selectedDocuments.map(doc => ({
                                            filename: doc.filename,
                                            thumbnailUrl:
                                              doc.thumbnailUrl ||
                                              `${API_BASE_URL}${API_ENDPOINTS.processed}/${doc.filename}`,
                                            pages: doc.pages || [
                                              {
                                                pageNumber: 1,
                                                thumbnailUrl:
                                                  doc.thumbnailUrl ||
                                                  `${API_BASE_URL}${API_ENDPOINTS.processed}/${doc.filename}`,
                                              },
                                            ]
                                          }))
                                          : []
                                      }
                                      previewSettings={{
                                        layout: orchestrateOptions.scanLayout,
                                        paperSize: orchestrateOptions.scanPaperSize,
                                        colorMode: orchestrateOptions.scanColorMode,
                                      }}
                                      activeDocIndex={previewControl.docIndex}
                                      activePage={previewControl.page}
                                      focusToken={previewControl.token}
                                      highlightSource={previewControl.source}
                                      onRequestDocChange={nextIndex =>
                                        bumpPreviewFocus({ docIndex: nextIndex, page: 1, source: 'manual' })
                                      }
                                      onRequestPageChange={nextPage =>
                                        bumpPreviewFocus({ page: nextPage, source: 'manual' })
                                      }
                                      ocrResult={selectedDocuments[previewControl.docIndex ?? 0] ? ocrResults[selectedDocuments[previewControl.docIndex ?? 0].filename] : undefined}
                                      textSelectionEnabled={true}
                                    />
                                  </Box>
                                )}

                                {/* Options Panel (RIGHT SIDE) */}
                                <Stack spacing={3} order={{ base: 1, lg: 2 }}>
                                  {/* Auto-Capture Delay Settings Display */}
                                  <Box
                                    p={4}
                                    borderRadius="lg"
                                    border="2px solid"
                                    borderColor={isCalibrated ? 'green.400' : 'orange.400'}
                                    bg={useColorModeValue(
                                      isCalibrated ? 'green.50' : 'orange.50',
                                      isCalibrated ? 'rgba(34, 197, 94, 0.1)' : 'rgba(237, 137, 54, 0.1)'
                                    )}
                                    transition="all 0.3s"
                                  >
                                    <Flex align="center" justify="space-between" gap={3} flexWrap="wrap">
                                      <HStack spacing={2}>
                                        <Iconify
                                          icon="solar:clock-circle-bold"
                                          width={20}
                                          height={20}
                                          color={isCalibrated ? 'var(--chakra-colors-green-500)' : 'var(--chakra-colors-orange-500)'}
                                        />
                                        <Heading size="sm" color={useColorModeValue('gray.700', 'gray.200')}>
                                          Auto-Capture Timing
                                        </Heading>
                                        <Badge
                                          colorScheme={isCalibrated ? 'green' : 'orange'}
                                          fontSize="xs"
                                        >
                                          {isCalibrated ? 'Configured' : 'Default'}
                                        </Badge>
                                      </HStack>
                                      <HStack spacing={3} flexWrap="wrap">
                                        <Tooltip
                                          label="Time to wait for smartphone camera to initialize before capturing first document"
                                          placement="top"
                                          hasArrow
                                        >
                                          <Badge
                                            colorScheme="purple"
                                            fontSize="sm"
                                            px={3}
                                            py={2}
                                            borderRadius="lg"
                                            display="flex"
                                            alignItems="center"
                                            gap={2}
                                          >
                                            <Text fontWeight="500">Startup:</Text>
                                            <Text fontWeight="bold" fontSize="md">{initialDelay}s</Text>
                                          </Badge>
                                        </Tooltip>
                                        <Tooltip
                                          label="Delay between consecutive document captures"
                                          placement="top"
                                          hasArrow
                                        >
                                          <Badge
                                            colorScheme="orange"
                                            fontSize="sm"
                                            px={3}
                                            py={2}
                                            borderRadius="lg"
                                            display="flex"
                                            alignItems="center"
                                            gap={2}
                                          >
                                            <Text fontWeight="500">Between:</Text>
                                            <Text fontWeight="bold" fontSize="md">{interCaptureDelay}s</Text>
                                          </Badge>
                                        </Tooltip>
                                      </HStack>
                                    </Flex>
                                  </Box>

                                  {/* Select Document Button - Only show if user chose 'select' source */}
                                  {scanDocumentSource === 'select' && (
                                    <Button
                                      size="lg"
                                      colorScheme="brand"
                                      variant="solid"
                                      leftIcon={
                                        <Iconify icon="solar:document-add-bold-duotone" width={20} height={20} />
                                      }
                                      onClick={documentSelectorModal.onOpen}
                                      w="full"
                                      py="1.5rem"
                                      fontSize="md"
                                      fontWeight="600"
                                      _hover={{
                                        transform: 'translateY(-2px)',
                                        boxShadow: '0 8px 20px rgba(121,95,238,0.3)',
                                      }}
                                      transition="all 0.3s"
                                    >
                                      {selectedDocuments.length > 0
                                        ? `${selectedDocuments.length} Document${selectedDocuments.length > 1 ? 's' : ''} Selected`
                                        : 'Select Document to Scan'}
                                    </Button>
                                  )}

                                  {/* Feed Documents Through Printer - Only show if user chose 'feed' source */}
                                  {scanDocumentSource === 'feed' && (
                                    <Box
                                      p="1.25rem"
                                      borderRadius="xl"
                                      border="2px solid"
                                      borderColor={documentsFed ? 'green.400' : 'orange.400'}
                                      bg={documentsFed ? 'rgba(72,187,120,0.08)' : 'rgba(237,137,54,0.08)'}
                                      transition="all 0.3s"
                                      _hover={{
                                        borderColor: documentsFed ? 'green.500' : 'orange.500',
                                        transform: 'translateY(-2px)',
                                        boxShadow: 'lg',
                                      }}
                                    >
                                      <Flex direction="column" gap={3}>
                                        <Flex justify="space-between" align="center" gap={4}>
                                          <Box flex="1">
                                            <Heading size="md" mb={2} display="flex" alignItems="center" gap={2}>
                                              <Iconify
                                                icon="solar:printer-bold-duotone"
                                                width={24}
                                                height={24}
                                                color={documentsFed ? 'var(--chakra-colors-green-500)' : 'var(--chakra-colors-orange-400)'}
                                              />
                                              Feed Documents
                                              {documentsFed && (
                                                <Badge colorScheme="green" fontSize="xs">
                                                  ✓ {feedCount} page{feedCount !== 1 ? 's' : ''} fed
                                                </Badge>
                                              )}
                                            </Heading>
                                            <Text fontSize="sm" color="text.muted">
                                              Place documents in printer input tray, enter count, then feed through to output tray
                                            </Text>
                                          </Box>
                                        </Flex>

                                        {/* Number of documents input */}
                                        <Box>
                                          <Text fontSize="sm" fontWeight="600" mb={1} color="text.secondary">
                                            Number of Documents
                                          </Text>
                                          <Input
                                            type="number"
                                            min={1}
                                            max={50}
                                            value={documentsToFeed}
                                            onChange={(e) => {
                                              const val = e.target.value;
                                              if (val === '' || val === '-' || val === '+') {
                                                setDocumentsToFeed(1);
                                              } else {
                                                const num = parseInt(val);
                                                if (!isNaN(num)) {
                                                  setDocumentsToFeed(Math.max(1, Math.min(50, num)));
                                                }
                                              }
                                            }}
                                            size="lg"
                                            borderColor="orange.300"
                                            _hover={{ borderColor: 'orange.400' }}
                                            _focus={{ borderColor: 'orange.500', boxShadow: '0 0 0 3px rgba(237,137,54,0.2)' }}
                                            textAlign="center"
                                            fontWeight="bold"
                                            fontSize="xl"
                                          />
                                        </Box>


                                        <Text fontSize="xs" color="text.muted" fontStyle="italic">
                                          💡 Tip: The printer will pull {documentsToFeed} document{documentsToFeed !== 1 ? 's' : ''} from the input tray and output them for scanning
                                        </Text>

                                      </Flex>
                                    </Box>
                                  )}

                                  {/* Select Page Scan Mode - Only show for 'select' source */}
                                  {scanDocumentSource === 'select' && (
                                    <Box
                                      p="1.25rem"
                                      borderRadius="xl"
                                      border="2px solid"
                                      borderColor="whiteAlpha.200"
                                      bg="whiteAlpha.50"
                                      _dark={{ bg: 'whiteAlpha.50' }}
                                      transition="all 0.3s"
                                      _hover={{
                                        borderColor: 'brand.400',
                                        transform: 'translateY(-2px)',
                                        boxShadow: 'lg',
                                      }}
                                    >
                                      <Heading size="md" mb={4} display="flex" alignItems="center" gap={2}>
                                        <Iconify
                                          icon="solar:document-bold-duotone"
                                          width={24}
                                          height={24}
                                          color="var(--chakra-colors-brand-500)"
                                        />
                                        Page Scan Mode
                                      </Heading>
                                      <Select
                                        value={orchestrateOptions.scanMode}
                                        onChange={(e: React.ChangeEvent<HTMLSelectElement>) =>
                                          setOrchestrateOptions({
                                            ...orchestrateOptions,
                                            scanMode: e.target.value as any,
                                          })
                                        }
                                        bg="whiteAlpha.50"
                                        borderColor="brand.300"
                                        size="lg"
                                        _hover={{ borderColor: 'brand.400' }}
                                        _focus={{
                                          borderColor: 'brand.500',
                                          boxShadow: '0 0 0 3px rgba(121,95,238,0.2)',
                                        }}
                                      >
                                        <option value="single">📄 Single Page</option>
                                        <option value="multi">📚 Multi-Page Document</option>
                                      </Select>
                                    </Box>
                                  )}

                                  {/* Text Detection - Only show for 'select' source */}
                                  {scanDocumentSource === 'select' && (
                                    <Box
                                      p="1.25rem"
                                      borderRadius="xl"
                                      border="2px solid"
                                      borderColor={orchestrateOptions.scanTextMode ? 'brand.400' : 'whiteAlpha.200'}
                                      bg={
                                        orchestrateOptions.scanTextMode ? 'rgba(121,95,238,0.08)' : 'whiteAlpha.50'
                                      }
                                      transition="all 0.3s"
                                      _hover={{
                                        borderColor: 'brand.400',
                                        transform: 'translateY(-2px)',
                                        boxShadow: 'lg',
                                      }}
                                    >
                                      <Flex justify="space-between" align="center">
                                        <Box flex="1">
                                          <Heading size="md" mb={2} display="flex" alignItems="center" gap={2}>
                                            <Iconify
                                              icon="solar:text-bold-duotone"
                                              width={24}
                                              height={24}
                                              color="var(--chakra-colors-brand-500)"
                                            />
                                            Text Detection
                                          </Heading>
                                          <Text fontSize="sm" color="text.muted">
                                            Extract text from scanned documents using OCR
                                          </Text>
                                        </Box>
                                        <Checkbox
                                          size="lg"
                                          colorScheme="brand"
                                          isChecked={orchestrateOptions.scanTextMode}
                                          onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                            setOrchestrateOptions({
                                              ...orchestrateOptions,
                                              scanTextMode: e.target.checked,
                                            })
                                          }
                                        >
                                          <Text fontWeight="600">Enable OCR</Text>
                                        </Checkbox>
                                      </Flex>
                                    </Box>
                                  )}

                                  {/* Layout - Only show for 'select' source */}

                                  {scanDocumentSource === 'select' && (
                                    <Box
                                      p="1.25rem"
                                      borderRadius="xl"
                                      border="2px solid"
                                      borderColor="whiteAlpha.200"
                                      bg="whiteAlpha.50"
                                      transition="all 0.3s"
                                      _hover={{
                                        borderColor: 'brand.400',
                                        transform: 'translateY(-2px)',
                                        boxShadow: 'lg',
                                      }}
                                    >
                                      <Heading size="md" mb={4} display="flex" alignItems="center" gap={2}>
                                        <Iconify
                                          icon="solar:tablet-bold-duotone"
                                          width={24}
                                          height={24}
                                          color="var(--chakra-colors-brand-500)"
                                        />
                                        Scan Layout
                                      </Heading>
                                      <ButtonGroup isAttached width="full" size="lg">
                                        <Button
                                          flex={1}
                                          variant={
                                            orchestrateOptions.scanLayout === 'portrait' ? 'solid' : 'outline'
                                          }
                                          colorScheme={
                                            orchestrateOptions.scanLayout === 'portrait' ? 'brand' : 'gray'
                                          }
                                          onClick={async () => {
                                            if (!selectedImageFile) return;
                                            try {
                                              const imageUrl = getImageUrl(API_ENDPOINTS.processed, selectedImageFile);
                                              const response = await apiClient.get(imageUrl, {
                                                responseType: 'blob',
                                              });

                                              const url = URL.createObjectURL(response.data);
                                              const a = document.createElement('a');
                                              a.href = url;
                                              a.download = selectedImageFile;
                                              document.body.appendChild(a);
                                              a.click();
                                              document.body.removeChild(a);
                                              URL.revokeObjectURL(url);
                                            } catch (err: any) {
                                              console.error('Download failed:', err);
                                              toast({
                                                title: 'Download failed',
                                                description: err.message || 'Unable to download file',
                                                status: 'error',
                                                duration: 4000,
                                              });
                                            }
                                          }}
                                        >
                                          Landscape
                                        </Button>
                                      </ButtonGroup>
                                    </Box>
                                  )}

                                  {/* Paper Size - Only show for 'select' source */}
                                  {scanDocumentSource === 'select' && (
                                    <Box
                                      p={5}
                                      borderRadius="xl"
                                      border="2px solid"
                                      borderColor="whiteAlpha.200"
                                      bg="whiteAlpha.50"
                                      transition="all 0.3s"
                                      _hover={{
                                        borderColor: 'brand.400',
                                        transform: 'translateY(-2px)',
                                        boxShadow: 'lg',
                                      }}
                                    >
                                      <FancySelect
                                        label="📏 Paper Size"
                                        options={[
                                          { value: 'A4', label: 'A4 (210×297 mm)' },
                                          { value: 'Letter', label: 'Letter (8.5×11 in)' },
                                          { value: 'Legal', label: 'Legal (8.5×14 in)' },
                                        ]}
                                        value={orchestrateOptions.scanPaperSize}
                                        onChange={(value: string) =>
                                          setOrchestrateOptions({ ...orchestrateOptions, scanPaperSize: value })
                                        }
                                      />
                                    </Box>
                                  )}

                                  {/* Resolution - Only show for 'select' source */}
                                  {scanDocumentSource === 'select' && (
                                    <Box
                                      p={5}
                                      borderRadius="xl"
                                      border="2px solid"
                                      borderColor="whiteAlpha.200"
                                      bg="whiteAlpha.50"
                                      transition="all 0.3s"
                                      _hover={{
                                        borderColor: 'brand.400',
                                        transform: 'translateY(-2px)',
                                        boxShadow: 'lg',
                                      }}
                                    >
                                      <Heading size="md" mb={4} display="flex" alignItems="center" gap={2}>
                                        <Iconify
                                          icon="solar:magnifer-zoom-in-bold-duotone"
                                          width={24}
                                          height={24}
                                          color="var(--chakra-colors-brand-500)"
                                        />
                                        Scan Resolution (DPI)
                                      </Heading>
                                      <FancySelect
                                        label=""
                                        options={[
                                          { value: '150', label: '150 DPI - Draft Quality' },
                                          { value: '300', label: '300 DPI - Standard (Recommended)' },
                                          { value: '600', label: '600 DPI - High Quality' },
                                          { value: '1200', label: '1200 DPI - Professional' },
                                        ]}
                                        value={orchestrateOptions.scanResolution}
                                        onChange={(value: string) =>
                                          setOrchestrateOptions({ ...orchestrateOptions, scanResolution: value })
                                        }
                                      />
                                    </Box>
                                  )}

                                  {/* Color Mode */}
                                  <Box
                                    p={5}
                                    borderRadius="xl"
                                    border="2px solid"
                                    borderColor="whiteAlpha.200"
                                    bg="whiteAlpha.50"
                                    transition="all 0.3s"
                                    _hover={{
                                      borderColor: 'brand.400',
                                      transform: 'translateY(-2px)',
                                      boxShadow: 'lg',
                                    }}
                                  >
                                    <Heading size="md" mb={4} display="flex" alignItems="center" gap={2}>
                                      <Iconify
                                        icon="solar:palette-2-bold-duotone"
                                        width={24}
                                        height={24}
                                        color="var(--chakra-colors-brand-500)"
                                      />
                                      Color Mode
                                    </Heading>
                                    <ButtonGroup isAttached width="full" size="lg">
                                      <Button
                                        flex={1}
                                        variant={
                                          orchestrateOptions.scanColorMode === 'color' ? 'solid' : 'outline'
                                        }
                                        colorScheme={
                                          orchestrateOptions.scanColorMode === 'color' ? 'brand' : 'gray'
                                        }
                                        onClick={() =>
                                          setOrchestrateOptions({ ...orchestrateOptions, scanColorMode: 'color' })
                                        }
                                        leftIcon={<Iconify icon="solar:pallete-bold" width={18} height={18} />}
                                        _hover={{ transform: 'scale(1.02)' }}
                                        transition="all 0.2s"
                                      >
                                        Color
                                      </Button>
                                      <Button
                                        flex={1}
                                        variant={orchestrateOptions.scanColorMode === 'bw' ? 'solid' : 'outline'}
                                        colorScheme={orchestrateOptions.scanColorMode === 'bw' ? 'brand' : 'gray'}
                                        onClick={() =>
                                          setOrchestrateOptions({ ...orchestrateOptions, scanColorMode: 'bw' })
                                        }
                                        leftIcon={<Iconify icon="solar:contrast-bold" width={18} height={18} />}
                                        _hover={{ transform: 'scale(1.02)' }}
                                        transition="all 0.2s"
                                      >
                                        B&W
                                      </Button>
                                    </ButtonGroup>
                                  </Box>

                                  {/* Layout / Rotation */}
                                  <Box
                                    p={5}
                                    borderRadius="xl"
                                    border="2px solid"
                                    borderColor="whiteAlpha.200"
                                    bg="whiteAlpha.50"
                                    transition="all 0.3s"
                                    _hover={{
                                      borderColor: 'brand.400',
                                      transform: 'translateY(-2px)',
                                      boxShadow: 'lg',
                                    }}
                                  >
                                    <Heading size="md" mb={4} display="flex" alignItems="center" gap={2}>
                                      <Iconify
                                        icon="solar:rotate-left-bold-duotone"
                                        width={24}
                                        height={24}
                                        color="var(--chakra-colors-brand-500)"
                                      />
                                      Layout / Rotation
                                    </Heading>
                                    <ButtonGroup isAttached width="full" size="lg">
                                      <Button
                                        flex={1}
                                        variant={
                                          orchestrateOptions.scanLayout === 'portrait' ? 'solid' : 'outline'
                                        }
                                        colorScheme={
                                          orchestrateOptions.scanLayout === 'portrait' ? 'brand' : 'gray'
                                        }
                                        onClick={() =>
                                          setOrchestrateOptions({ ...orchestrateOptions, scanLayout: 'portrait' })
                                        }
                                        leftIcon={<Iconify icon="solar:smartphone-bold" width={18} height={18} />}
                                        _hover={{ transform: 'scale(1.02)' }}
                                        transition="all 0.2s"
                                      >
                                        Portrait
                                      </Button>
                                      <Button
                                        flex={1}
                                        variant={orchestrateOptions.scanLayout === 'landscape' ? 'solid' : 'outline'}
                                        colorScheme={orchestrateOptions.scanLayout === 'landscape' ? 'brand' : 'gray'}
                                        onClick={() =>
                                          setOrchestrateOptions({ ...orchestrateOptions, scanLayout: 'landscape' })
                                        }
                                        leftIcon={<Iconify icon="solar:tablet-bold" width={18} height={18} />}
                                        _hover={{ transform: 'scale(1.02)' }}
                                        transition="all 0.2s"
                                      >
                                        Landscape
                                      </Button>
                                    </ButtonGroup>
                                  </Box>

                                  {/* Save as Default */}

                                  <Box
                                    p={6}
                                    borderRadius="xl"
                                    border="2px solid"
                                    borderColor="brand.400"
                                    bg="rgba(121,95,238,0.12)"
                                    transition="all 0.3s"
                                    _hover={{
                                      transform: 'scale(1.02)',
                                      borderColor: 'brand.500',
                                      boxShadow: '0 8px 20px rgba(121,95,238,0.3)',
                                    }}
                                  >
                                    <Checkbox
                                      size="lg"
                                      colorScheme="brand"
                                      isChecked={orchestrateOptions.saveAsDefault}
                                      onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                                        setOrchestrateOptions({
                                          ...orchestrateOptions,
                                          saveAsDefault: e.target.checked,
                                        });
                                        if (e.target.checked) {
                                          saveDefaultSettings();
                                        }
                                      }}
                                    >
                                      <HStack spacing={2}>
                                        <Iconify icon="solar:save-bold-duotone" width={20} height={20} />
                                        <Text fontWeight="600" fontSize="md">
                                          Save as Default Settings
                                        </Text>
                                      </HStack>
                                    </Checkbox>
                                    <Text fontSize="sm" color="text.muted" mt={2} ml={8}>
                                      Your preferences will be remembered for future scans
                                    </Text>
                                  </Box>
                                </Stack>
                              </Grid>
                            </ModalBody>
                            <ModalFooter
                              py="1.5rem"
                              px="2.5rem"
                              borderTop="1px solid"
                              borderColor="whiteAlpha.200"
                              display="flex"
                              justifyContent="space-between"
                              alignItems="center"
                            >
                              <HStack spacing={4}>
                                {selectedDocuments.length > 0 && (
                                  <>
                                    <HStack spacing={2} fontSize="sm">
                                      <Iconify icon="solar:document-bold" width={16} height={16} />
                                      <Text fontWeight="500">
                                        {selectedDocuments.length}{' '}
                                        {selectedDocuments.length === 1 ? 'sheet' : 'sheets'}
                                      </Text>
                                    </HStack>
                                    <HStack spacing={2}>
                                      <Badge colorScheme="blue" fontSize="xs">
                                        {orchestrateOptions.scanPaperSize}
                                      </Badge>
                                      <Badge colorScheme="purple" fontSize="xs" textTransform="uppercase">
                                        {orchestrateOptions.scanLayout}
                                      </Badge>
                                      {orchestrateOptions.scanColorMode !== 'color' && (
                                        <Badge colorScheme="gray" fontSize="xs" textTransform="uppercase">
                                          {orchestrateOptions.scanColorMode}
                                        </Badge>
                                      )}
                                    </HStack>
                                  </>
                                )}
                                {/* Calibration Delay Indicator */}
                                <Tooltip
                                  label={`Phone will wait ${initialDelay}s before capturing first document (calibrated for printer warmup)`}
                                  placement="top"
                                  hasArrow
                                >
                                  <Badge
                                    colorScheme={isCalibrated ? 'green' : 'orange'}
                                    fontSize="xs"
                                    px={2}
                                    py={1}
                                    borderRadius="md"
                                    display="flex"
                                    alignItems="center"
                                    gap={1}
                                  >
                                    <Iconify icon="solar:clock-circle-bold" width={12} height={12} />
                                    {initialDelay}s delay
                                  </Badge>
                                </Tooltip>
                              </HStack>
                              <HStack>
                                <Button
                                  variant="ghost"
                                  onClick={() => setOrchestrateStep(1)}
                                  size="lg"
                                  leftIcon={<Iconify icon="solar:arrow-left-bold" width={20} height={20} />}
                                >
                                  Back to Mode Selection
                                </Button>

                                <Button
                                  colorScheme="orange"
                                  onClick={feedDocumentsThroughPrinter}
                                  isLoading={isFeedingDocuments}
                                  loadingText="Feeding..."
                                  size="lg"
                                  px={8}
                                  rightIcon={<Iconify icon="solar:printer-bold" width={20} height={20} />}
                                >
                                  Feed and Scan
                                </Button>

                              </HStack>
                            </ModalFooter>
                          </>
                        )}
                      </>
                    )}

                    {/* STEP 2: Print Options */}
                    {orchestrateStep === 2 && orchestrateMode === 'print' && (
                      <>
                        <ModalHeader
                          fontSize="2xl"
                          fontWeight="700"
                          py="1.5rem"
                          display="flex"
                          alignItems="center"
                          gap={3}
                          borderBottom="1px solid"
                          borderColor="whiteAlpha.200"
                        >
                          <Box p={2} bg="nebula.500" borderRadius="lg" color="white">
                            <Iconify icon="solar:printer-bold-duotone" width={24} height={24} />
                          </Box>
                          Print Configuration
                          <Badge colorScheme="blue" fontSize="sm" px={3} py={1}>
                            Step 2 of 3
                          </Badge>
                        </ModalHeader>
                        <ModalCloseButton
                          size="lg"
                          top={6}
                          right={6}
                          _hover={{ bg: 'red.500', color: 'white' }}
                        />
                        <ModalBody
                          ref={modalBodyRef}
                          data-orchestration-scroll="print"
                          py={4}
                          px={6}
                          maxH="calc(85vh - 140px)"
                          overflowY="auto"
                          overflowX="hidden"
                          css={{
                            '&::-webkit-scrollbar': { width: '8px' },
                            '&::-webkit-scrollbar-track': { background: 'transparent' },
                            '&::-webkit-scrollbar-thumb': {
                              background: 'rgba(69,202,255,0.4)',
                              borderRadius: '10px',
                            },
                            '&::-webkit-scrollbar-thumb:hover': {
                              background: 'rgba(69,202,255,0.6)',
                            },
                          }}
                        >

                          <Grid templateColumns={{ base: '1fr', lg: '1fr 1fr' }} gap={6} alignItems="start">
                            {/* Live Preview - Print Mode (LEFT SIDE) */}
                            <Box
                              display={{ base: 'none', lg: 'flex' }}
                              flexDirection="column"
                              position="sticky"
                              top={0}
                              alignSelf="start"
                              maxH={`calc(${MODAL_CONFIG.previewBox.maxHeight})`}
                              order={{ base: 2, lg: 1 }}
                            >
                              <DocumentPreview
                                documents={
                                  selectedDocuments.length > 0
                                    ? selectedDocuments.map(doc => ({
                                      filename: doc.filename,
                                      thumbnailUrl:
                                        doc.thumbnailUrl ||
                                        `${API_BASE_URL}${API_ENDPOINTS.processed}/${doc.filename}`,
                                      pages: doc.pages || [
                                        {
                                          pageNumber: 1,
                                          thumbnailUrl:
                                            doc.thumbnailUrl ||
                                            `${API_BASE_URL}${API_ENDPOINTS.processed}/${doc.filename}`,
                                        },
                                      ]
                                    }))
                                    : []
                                }
                                previewSettings={{
                                  layout: orchestrateOptions.printLayout,
                                  scale: parseInt(orchestrateOptions.printScale) || 100,
                                  paperSize: orchestrateOptions.printPaperSize,
                                  colorMode: orchestrateOptions.printColorMode,
                                }}
                                activeDocIndex={previewControl.docIndex}
                                activePage={previewControl.page}
                                focusToken={previewControl.token}
                                highlightSource={previewControl.source}
                                onRequestDocChange={nextIndex =>
                                  bumpPreviewFocus({ docIndex: nextIndex, page: 1, source: 'manual' })
                                }
                                onRequestPageChange={nextPage =>
                                  bumpPreviewFocus({ page: nextPage, source: 'manual' })
                                }
                                ocrResult={selectedDocuments[previewControl.docIndex ?? 0] ? ocrResults[selectedDocuments[previewControl.docIndex ?? 0].filename] : undefined}
                                textSelectionEnabled={true}
                              />
                            </Box>

                            {/* Options Panel (RIGHT SIDE) */}
                            <Stack spacing={3} order={{ base: 1, lg: 2 }}>
                              {/* Auto-Capture Delay Settings Display */}
                              <Box
                                p={4}
                                borderRadius="lg"
                                border="2px solid"
                                borderColor={isCalibrated ? 'green.400' : 'orange.400'}
                                bg={useColorModeValue(
                                  isCalibrated ? 'green.50' : 'orange.50',
                                  isCalibrated ? 'rgba(34, 197, 94, 0.1)' : 'rgba(237, 137, 54, 0.1)'
                                )}
                                transition="all 0.3s"
                              >
                                <Flex align="center" justify="space-between" gap={3} flexWrap="wrap">
                                  <HStack spacing={2}>
                                    <Iconify
                                      icon="solar:clock-circle-bold"
                                      width={20}
                                      height={20}
                                      color={isCalibrated ? 'var(--chakra-colors-green-500)' : 'var(--chakra-colors-orange-500)'}
                                    />
                                    <Heading size="sm" color={useColorModeValue('gray.700', 'gray.200')}>
                                      Auto-Capture Timing
                                    </Heading>
                                    <Badge
                                      colorScheme={isCalibrated ? 'green' : 'orange'}
                                      fontSize="xs"
                                    >
                                      {isCalibrated ? 'Configured' : 'Default'}
                                    </Badge>
                                  </HStack>
                                  <HStack spacing={3} flexWrap="wrap">
                                    <Tooltip
                                      label="Time to wait for smartphone camera to initialize before capturing first document"
                                      placement="top"
                                      hasArrow
                                    >
                                      <Badge
                                        colorScheme="purple"
                                        fontSize="sm"
                                        px={3}
                                        py={2}
                                        borderRadius="lg"
                                        display="flex"
                                        alignItems="center"
                                        gap={2}
                                      >
                                        <Text fontWeight="500">Startup:</Text>
                                        <Text fontWeight="bold" fontSize="md">{initialDelay}s</Text>
                                      </Badge>
                                    </Tooltip>
                                    <Tooltip
                                      label="Delay between consecutive document captures"
                                      placement="top"
                                      hasArrow
                                    >
                                      <Badge
                                        colorScheme="orange"
                                        fontSize="sm"
                                        px={3}
                                        py={2}
                                        borderRadius="lg"
                                        display="flex"
                                        alignItems="center"
                                        gap={2}
                                      >
                                        <Text fontWeight="500">Between:</Text>
                                        <Text fontWeight="bold" fontSize="md">{interCaptureDelay}s</Text>
                                      </Badge>
                                    </Tooltip>
                                  </HStack>
                                </Flex>
                              </Box>

                              {/* Select Document Button */}
                              <Button
                                size="lg"
                                colorScheme="nebula"
                                variant="solid"
                                leftIcon={
                                  <Iconify icon="solar:printer-bold-duotone" width={20} height={20} />
                                }
                                onClick={documentSelectorModal.onOpen}
                                w="full"
                                py="1.5rem"
                                fontSize="md"
                                fontWeight="600"
                                _hover={{
                                  transform: 'translateY(-2px)',
                                  boxShadow: '0 8px 20px rgba(69,202,255,0.3)',
                                }}
                                transition="all 0.3s"
                              >
                                {selectedDocuments.length > 0
                                  ? `${selectedDocuments.length} Document${selectedDocuments.length > 1 ? 's' : ''} Selected for Print`
                                  : 'Select Documents to Print'}
                              </Button>

                              {/* Number of Copies - Moved to top */}
                              <Box
                                p={4}
                                borderRadius="lg"
                                border="1px solid"
                                borderColor="whiteAlpha.200"
                                bg="whiteAlpha.50"
                                transition="all 0.2s"
                                _hover={{ borderColor: 'brand.400' }}
                              >
                                <Heading size="sm" mb={3}>
                                  📄 Number of Copies
                                </Heading>
                                <Flex align="center" gap={3}>
                                  <Input
                                    type="number"
                                    min="1"
                                    max="999"
                                    value={orchestrateOptions.printCopies}
                                    onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                      setOrchestrateOptions({
                                        ...orchestrateOptions,
                                        printCopies: e.target.value,
                                      })
                                    }
                                    bg="whiteAlpha.100"
                                    borderColor="brand.300"
                                    _hover={{ borderColor: 'brand.400' }}
                                    w="100px"
                                    textAlign="center"
                                    fontWeight="bold"
                                  />
                                  <Text>copies</Text>
                                </Flex>
                              </Box>

                              {/* Pages to Print */}
                              <Box
                                p={5}
                                borderRadius="xl"
                                border="2px solid"
                                borderColor="whiteAlpha.200"
                                bg="whiteAlpha.50"
                                transition="all 0.3s"
                                _hover={{
                                  borderColor: 'nebula.400',
                                  transform: 'translateY(-2px)',
                                  boxShadow: 'lg',
                                }}
                              >
                                <Heading size="md" mb={4} display="flex" alignItems="center" gap={2}>
                                  <Iconify
                                    icon="solar:documents-bold-duotone"
                                    width={24}
                                    height={24}
                                    color="var(--chakra-colors-nebula-500)"
                                  />
                                  Pages to Print
                                </Heading>
                                <RadioGroup
                                  value={orchestrateOptions.printPages}
                                  onChange={(value: string) =>
                                    setOrchestrateOptions({ ...orchestrateOptions, printPages: value as any })
                                  }
                                >
                                  <Stack spacing={3}>
                                    <Radio value="all" colorScheme="brand">
                                      All Pages
                                    </Radio>
                                    <Radio value="odd" colorScheme="brand">
                                      Odd Pages Only (1, 3, 5...)
                                    </Radio>
                                    <Radio value="even" colorScheme="brand">
                                      Even Pages Only (2, 4, 6...)
                                    </Radio>
                                    <Radio value="custom" colorScheme="brand">
                                      <Flex
                                        direction="column"
                                        gap={2}
                                        mt={orchestrateOptions.printPages === 'custom' ? 2 : 0}
                                      >
                                        <Text>Custom Pages</Text>
                                        {orchestrateOptions.printPages === 'custom' && (
                                          <Input
                                            size="sm"
                                            placeholder="e.g., 1-5,7,9"
                                            value={orchestrateOptions.printCustomRange}
                                            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                              setOrchestrateOptions({
                                                ...orchestrateOptions,
                                                printCustomRange: e.target.value,
                                              })
                                            }
                                            bg="whiteAlpha.100"
                                            borderColor="brand.300"
                                            _hover={{ borderColor: 'brand.400' }}
                                          />
                                        )}
                                      </Flex>
                                    </Radio>
                                  </Stack>
                                </RadioGroup>
                              </Box>

                              {/* Layout */}
                              <Box
                                p={4}
                                borderRadius="lg"
                                border="1px solid"
                                borderColor="whiteAlpha.200"
                                bg="whiteAlpha.50"
                                transition="all 0.2s"
                                _hover={{ borderColor: 'brand.400' }}
                              >
                                <Heading size="sm" mb={3}>
                                  📐 Layout
                                </Heading>
                                <ButtonGroup isAttached width="full">
                                  <Button
                                    flex={1}
                                    variant={
                                      orchestrateOptions.printLayout === 'portrait' ? 'solid' : 'outline'
                                    }
                                    colorScheme={
                                      orchestrateOptions.printLayout === 'portrait' ? 'brand' : 'gray'
                                    }
                                    onClick={() =>
                                      setOrchestrateOptions({
                                        ...orchestrateOptions,
                                        printLayout: 'portrait',
                                      })
                                    }
                                  >
                                    📄 Portrait
                                  </Button>
                                  <Button
                                    flex={1}
                                    variant={
                                      orchestrateOptions.printLayout === 'landscape' ? 'solid' : 'outline'
                                    }
                                    colorScheme={
                                      orchestrateOptions.printLayout === 'landscape' ? 'brand' : 'gray'
                                    }
                                    onClick={() =>
                                      setOrchestrateOptions({
                                        ...orchestrateOptions,
                                        printLayout: 'landscape',
                                      })
                                    }
                                  >
                                    📐 Landscape
                                  </Button>
                                </ButtonGroup>
                              </Box>

                              {/* Paper Size & Color Mode - Side by Side */}
                              <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
                                {/* Paper Size */}
                                <Box
                                  p={4}
                                  borderRadius="lg"
                                  border="1px solid"
                                  borderColor="whiteAlpha.200"
                                  bg="whiteAlpha.50"
                                  transition="all 0.2s"
                                  _hover={{ borderColor: 'brand.400' }}
                                >
                                  <FancySelect
                                    label="📏 Paper Size"
                                    options={[
                                      { value: 'A4', label: 'A4 (210×297 mm)' },
                                      { value: 'Letter', label: 'Letter (8.5×11 in)' },
                                      { value: 'Legal', label: 'Legal (8.5×14 in)' },
                                    ]}
                                    value={orchestrateOptions.printPaperSize}
                                    onChange={(value: string) =>
                                      setOrchestrateOptions({ ...orchestrateOptions, printPaperSize: value })
                                    }
                                  />
                                </Box>

                                {/* Color Mode */}
                                <Box
                                  p={4}
                                  borderRadius="lg"
                                  border="1px solid"
                                  borderColor="whiteAlpha.200"
                                  bg="whiteAlpha.50"
                                  transition="all 0.2s"
                                  _hover={{ borderColor: 'brand.400' }}
                                >
                                  <Heading size="sm" mb={3}>
                                    Color Mode
                                  </Heading>
                                  <ButtonGroup isAttached width="full" size="md">
                                    <Button
                                      flex={1}
                                      variant={
                                        orchestrateOptions.printColorMode === 'color' ? 'solid' : 'outline'
                                      }
                                      colorScheme={
                                        orchestrateOptions.printColorMode === 'color' ? 'nebula' : 'gray'
                                      }
                                      onClick={() =>
                                        setOrchestrateOptions({
                                          ...orchestrateOptions,
                                          printColorMode: 'color',
                                        })
                                      }
                                      leftIcon={<Iconify icon="solar:pallete-bold" width={18} height={18} />}
                                    >
                                      Color
                                    </Button>
                                    <Button
                                      flex={1}
                                      variant={orchestrateOptions.printColorMode === 'bw' ? 'solid' : 'outline'}
                                      colorScheme={
                                        orchestrateOptions.printColorMode === 'bw' ? 'nebula' : 'gray'
                                      }
                                      onClick={() =>
                                        setOrchestrateOptions({ ...orchestrateOptions, printColorMode: 'bw' })
                                      }
                                      leftIcon={<Iconify icon="solar:contrast-bold" width={18} height={18} />}
                                    >
                                      B&W
                                    </Button>
                                  </ButtonGroup>
                                </Box>
                              </SimpleGrid>

                              {/* Double-sided (Duplex) */}
                              <Box
                                p={4}
                                borderRadius="lg"
                                border="1px solid"
                                borderColor="whiteAlpha.200"
                                bg="whiteAlpha.50"
                                transition="all 0.2s"
                                _hover={{ borderColor: 'brand.400' }}
                              >
                                <Checkbox
                                  size="lg"
                                  colorScheme="nebula"
                                  isChecked={orchestrateOptions.printDuplex}
                                  onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                    setOrchestrateOptions({
                                      ...orchestrateOptions,
                                      printDuplex: e.target.checked,
                                    })
                                  }
                                >
                                  <Text fontWeight="600">📑 Double-sided (Duplex) Printing</Text>
                                  <Text fontSize="xs" color="text.muted">
                                    Print on both sides of the paper
                                  </Text>
                                </Checkbox>
                              </Box>

                              {/* Save as Default */}
                              <Box
                                p={4}
                                borderRadius="lg"
                                border="1px solid"
                                borderColor="brand.400"
                                bg="rgba(121,95,238,0.1)"
                                transition="all 0.3s"
                                _hover={{ transform: 'scale(1.02)', borderColor: 'brand.500' }}
                              >
                                <Checkbox
                                  size="lg"
                                  colorScheme="brand"
                                  isChecked={orchestrateOptions.saveAsDefault}
                                  onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                    setOrchestrateOptions({
                                      ...orchestrateOptions,
                                      saveAsDefault: e.target.checked,
                                    })
                                  }
                                >
                                  <Text fontWeight="600">💾 Save as Default Settings</Text>
                                </Checkbox>
                              </Box>
                            </Stack>
                          </Grid>
                        </ModalBody>
                        <ModalFooter
                          py="1.5rem"
                          px="2.5rem"
                          borderTop="1px solid"
                          borderColor="whiteAlpha.200"
                          display="flex"
                          justifyContent="space-between"
                          alignItems="center"
                          position="sticky"
                          bottom={0}
                          bg={surfaceCard}
                          zIndex={30}
                        >
                          <HStack spacing={4}>
                            {selectedDocuments.length > 0 && (
                              <>
                                <HStack spacing={2} fontSize="sm">
                                  <Iconify icon="solar:document-bold" width={16} height={16} />
                                  <Text fontWeight="500">
                                    {selectedDocuments.length}{' '}
                                    {selectedDocuments.length === 1 ? 'sheet' : 'sheets'}
                                  </Text>
                                </HStack>
                                <HStack spacing={2}>
                                  <Badge colorScheme="blue" fontSize="xs">
                                    {orchestrateOptions.printPaperSize}
                                  </Badge>
                                  <Badge colorScheme="purple" fontSize="xs" textTransform="uppercase">
                                    {orchestrateOptions.printLayout}
                                  </Badge>
                                </HStack>
                              </>
                            )}
                            {/* Calibration Delay Indicator */}
                            <Tooltip
                              label={`Phone will wait ${initialDelay}s before capturing first document (calibrated for printer warmup)`}
                              placement="top"
                              hasArrow
                            >
                              <Badge
                                colorScheme={isCalibrated ? 'green' : 'orange'}
                                fontSize="xs"
                                px={2}
                                py={1}
                                borderRadius="md"
                                display="flex"
                                alignItems="center"
                                gap={1}
                              >
                                <Iconify icon="solar:clock-circle-bold" width={12} height={12} />
                                {initialDelay}s delay
                              </Badge>
                            </Tooltip>
                          </HStack>
                          <HStack>
                            <Button
                              variant="ghost"
                              onClick={() => setOrchestrateStep(1)}
                              size="lg"
                              leftIcon={<Iconify icon="solar:arrow-left-bold" width={20} height={20} />}
                            >
                              Back
                            </Button>
                            <Button
                              colorScheme="nebula"
                              onClick={() => setOrchestrateStep(3)}
                              size="lg"
                              px={8}
                              rightIcon={<Iconify icon="solar:arrow-right-bold" width={20} height={20} />}
                            >
                              Review Settings
                            </Button>
                          </HStack>
                        </ModalFooter>
                      </>
                    )}

                    {/* STEP 3: Confirmation */}
                    {orchestrateStep === 3 && (
                      <>
                        <ModalHeader
                          fontSize="2xl"
                          fontWeight="700"
                          py="1.5rem"
                          display="flex"
                          alignItems="center"
                          gap={3}
                          borderBottom="1px solid"
                          borderColor="whiteAlpha.200"
                        >
                          <Box p={2} bg="green.500" borderRadius="lg" color="white">
                            <Iconify icon="solar:check-circle-bold-duotone" width={24} height={24} />
                          </Box>
                          Review & Confirm
                          <Badge ml="auto" colorScheme="green" fontSize="sm" px={3} py={1}>
                            Step 3 of 3
                          </Badge>
                        </ModalHeader>
                        <ModalCloseButton
                          size="lg"
                          top={6}
                          right={6}
                          _hover={{ bg: 'red.500', color: 'white' }}
                        />
                        <ModalBody py={8} px={10}>
                          <Stack spacing={6}>
                            <Box
                              bg="linear-gradient(135deg, rgba(121,95,238,0.15), rgba(69,202,255,0.15))"
                              p={8}
                              borderRadius="2xl"
                              border="2px solid"
                              borderColor={orchestrateMode === 'scan' ? 'brand.300' : 'nebula.300'}
                              boxShadow="0 8px 25px rgba(121,95,238,0.2)"
                            >
                              <Heading size="lg" mb={6} display="flex" alignItems="center" gap={3}>
                                <Iconify
                                  icon={
                                    orchestrateMode === 'scan'
                                      ? 'solar:document-add-bold-duotone'
                                      : 'solar:printer-bold-duotone'
                                  }
                                  width={32}
                                  height={32}
                                  color={
                                    orchestrateMode === 'scan'
                                      ? 'var(--chakra-colors-brand-500)'
                                      : 'var(--chakra-colors-nebula-500)'
                                  }
                                />
                                {orchestrateMode === 'scan' ? 'Scan' : 'Print'} Configuration Summary
                              </Heading>
                              <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
                                {orchestrateMode === 'scan' ? (
                                  <>
                                    <HStack
                                      justify="space-between"
                                      p={4}
                                      bg="whiteAlpha.200"
                                      borderRadius="lg"
                                    >
                                      <Text fontWeight="600">Operation Mode:</Text>
                                      <Badge colorScheme="purple" fontSize="md" px={3} py={1}>
                                        Scan
                                      </Badge>
                                    </HStack>
                                    <HStack
                                      justify="space-between"
                                      p={4}
                                      bg="whiteAlpha.200"
                                      borderRadius="lg"
                                    >
                                      <Text fontWeight="600">Page Mode:</Text>
                                      <Badge colorScheme="blue" fontSize="md" px={3} py={1}>
                                        {orchestrateOptions.scanPageMode}
                                      </Badge>
                                    </HStack>
                                    <HStack
                                      justify="space-between"
                                      p={4}
                                      bg="whiteAlpha.200"
                                      borderRadius="lg"
                                    >
                                      <Text fontWeight="600">Layout:</Text>
                                      <Badge colorScheme="cyan" fontSize="md" px={3} py={1}>
                                        {orchestrateOptions.scanLayout}
                                      </Badge>
                                    </HStack>
                                    <HStack
                                      justify="space-between"
                                      p={4}
                                      bg="whiteAlpha.200"
                                      borderRadius="lg"
                                    >
                                      <Text fontWeight="600">Paper Size:</Text>
                                      <Badge colorScheme="teal" fontSize="md" px={3} py={1}>
                                        {orchestrateOptions.scanPaperSize}
                                      </Badge>
                                    </HStack>
                                    <HStack
                                      justify="space-between"
                                      p={4}
                                      bg="whiteAlpha.200"
                                      borderRadius="lg"
                                    >
                                      <Text fontWeight="600">Resolution:</Text>
                                      <Badge colorScheme="green" fontSize="md" px={3} py={1}>
                                        {orchestrateOptions.scanResolution} DPI
                                      </Badge>
                                    </HStack>
                                    <HStack
                                      justify="space-between"
                                      p={4}
                                      bg="whiteAlpha.200"
                                      borderRadius="lg"
                                    >
                                      <Text fontWeight="600">Color Mode:</Text>
                                      <Badge colorScheme="pink" fontSize="md" px={3} py={1}>
                                        {orchestrateOptions.scanColorMode}
                                      </Badge>
                                    </HStack>
                                    {orchestrateOptions.scanTextMode && (
                                      <HStack
                                        justify="space-between"
                                        p={4}
                                        bg="green.100"
                                        borderRadius="lg"
                                        gridColumn={{ base: 'span 1', md: 'span 2' }}
                                      >
                                        <HStack>
                                          <Iconify
                                            icon="solar:check-circle-bold"
                                            width={20}
                                            height={20}
                                            color="green.600"
                                          />
                                          <Text fontWeight="600" color="green.700">
                                            OCR Text Detection:
                                          </Text>
                                        </HStack>
                                        <Badge colorScheme="green" fontSize="md" px={3} py={1}>
                                          Enabled
                                        </Badge>
                                      </HStack>
                                    )}
                                  </>
                                ) : (
                                  <>
                                    <HStack
                                      justify="space-between"
                                      p={4}
                                      bg="whiteAlpha.200"
                                      borderRadius="lg"
                                    >
                                      <Text fontWeight="600">Operation Mode:</Text>
                                      <Badge colorScheme="blue" fontSize="md" px={3} py={1}>
                                        Print
                                      </Badge>
                                    </HStack>
                                    <HStack
                                      justify="space-between"
                                      p={4}
                                      bg="whiteAlpha.200"
                                      borderRadius="lg"
                                    >
                                      <Text fontWeight="600">Pages:</Text>
                                      <Badge colorScheme="purple" fontSize="md" px={3} py={1}>
                                        {orchestrateOptions.printPages}
                                      </Badge>
                                    </HStack>
                                    <HStack
                                      justify="space-between"
                                      p={4}
                                      bg="whiteAlpha.200"
                                      borderRadius="lg"
                                    >
                                      <Text fontWeight="600">Layout:</Text>
                                      <Badge colorScheme="cyan" fontSize="md" px={3} py={1}>
                                        {orchestrateOptions.printLayout}
                                      </Badge>
                                    </HStack>
                                    <HStack
                                      justify="space-between"
                                      p={4}
                                      bg="whiteAlpha.200"
                                      borderRadius="lg"
                                    >
                                      <Text fontWeight="600">Paper Size:</Text>
                                      <Badge colorScheme="teal" fontSize="md" px={3} py={1}>
                                        {orchestrateOptions.printPaperSize}
                                      </Badge>
                                    </HStack>
                                    <HStack
                                      justify="space-between"
                                      p={4}
                                      bg="whiteAlpha.200"
                                      borderRadius="lg"
                                    >
                                      <Text fontWeight="600">Color Mode:</Text>
                                      <Badge colorScheme="pink" fontSize="md" px={3} py={1}>
                                        {orchestrateOptions.printColorMode}
                                      </Badge>
                                    </HStack>
                                    {orchestrateOptions.printConvertedFiles.length > 0 && (
                                      <HStack justify="space-between" p={4} bg="blue.100" borderRadius="lg">
                                        <HStack>
                                          <Iconify
                                            icon="solar:document-text-bold"
                                            width={20}
                                            height={20}
                                            color="blue.600"
                                          />
                                          <Text fontWeight="600" color="blue.700">
                                            Selected PDFs:
                                          </Text>
                                        </HStack>
                                        <Badge colorScheme="blue" fontSize="md" px={3} py={1}>
                                          {orchestrateOptions.printConvertedFiles.length} file(s)
                                        </Badge>
                                      </HStack>
                                    )}
                                    {selectedDocuments.length > 0 && (
                                      <HStack justify="space-between" p={4} bg="green.100" borderRadius="lg" gridColumn={{ base: 'span 1', md: 'span 2' }}>
                                        <HStack>
                                          <Iconify
                                            icon="solar:documents-bold-duotone"
                                            width={20}
                                            height={20}
                                            color="green.600"
                                          />
                                          <Text fontWeight="600" color="green.700">
                                            Documents to Print:
                                          </Text>
                                        </HStack>
                                        <Badge colorScheme="green" fontSize="md" px={3} py={1}>
                                          {selectedDocuments.length} document(s) • {selectedDocuments.reduce((total, doc) => total + (doc.pages?.length || 1), 0)} page(s)
                                        </Badge>
                                      </HStack>
                                    )}
                                    <HStack
                                      justify="space-between"
                                      p={4}
                                      bg="whiteAlpha.200"
                                      borderRadius="lg"
                                    >
                                      <Text fontWeight="600">Copies:</Text>
                                      <Badge colorScheme="indigo" fontSize="md" px={3} py={1}>
                                        {orchestrateOptions.printCopies}
                                      </Badge>
                                    </HStack>
                                    <HStack
                                      justify="space-between"
                                      p={4}
                                      bg="whiteAlpha.200"
                                      borderRadius="lg"
                                    >
                                      <Text fontWeight="600">Double-sided:</Text>
                                      <Badge colorScheme={orchestrateOptions.printDuplex ? 'green' : 'gray'} fontSize="md" px={3} py={1}>
                                        {orchestrateOptions.printDuplex ? 'Yes' : 'No'}
                                      </Badge>
                                    </HStack>
                                    <HStack
                                      justify="space-between"
                                      p={4}
                                      bg="whiteAlpha.200"
                                      borderRadius="lg"
                                    >
                                      <Text fontWeight="600">Color Mode:</Text>
                                      <Badge colorScheme="pink" fontSize="md" px={3} py={1}>
                                        {orchestrateOptions.printColorMode}
                                      </Badge>
                                    </HStack>
                                  </>
                                )}
                                {orchestrateOptions.saveAsDefault && (
                                  <HStack
                                    justify="space-between"
                                    p={4}
                                    bg="green.100"
                                    borderRadius="lg"
                                    gridColumn={{ base: 'span 1', md: 'span 2' }}
                                  >
                                    <HStack>
                                      <Iconify
                                        icon="solar:save-bold-duotone"
                                        width={20}
                                        height={20}
                                        color="green.600"
                                      />
                                      <Text fontWeight="600" color="green.700">
                                        Save as Default:
                                      </Text>
                                    </HStack>
                                    <Badge colorScheme="green" fontSize="md" px={3} py={1}>
                                      Yes
                                    </Badge>
                                  </HStack>
                                )}
                              </SimpleGrid>
                            </Box>
                            <Box
                              p={5}
                              bg="yellow.50"
                              borderRadius="xl"
                              border="2px solid"
                              borderColor="yellow.300"
                              _dark={{ bg: 'rgba(255,193,7,0.15)', borderColor: 'yellow.600' }}
                            >
                              <HStack spacing={3} mb={2}>
                                <Iconify
                                  icon="solar:info-circle-bold"
                                  width={24}
                                  height={24}
                                  color="orange.500"
                                />
                                <Text
                                  fontWeight="700"
                                  fontSize="lg"
                                  color="orange.700"
                                  _dark={{ color: 'orange.300' }}
                                >
                                  Ready to Proceed
                                </Text>
                              </HStack>
                              <Text color="text.muted" fontSize="sm">
                                Review your settings above. Click "Start{' '}
                                {orchestrateMode === 'scan' ? 'Scanning' : 'Printing'}" to begin the operation
                                with these configurations.
                              </Text>
                            </Box>
                          </Stack>
                        </ModalBody>
                        <ModalFooter
                          py="1.5rem"
                          px="2.5rem"
                          borderTop="1px solid"
                          borderColor="whiteAlpha.200"
                        >
                          <Button
                            variant="ghost"
                            mr={3}
                            onClick={() => setOrchestrateStep(2)}
                            size="lg"
                            leftIcon={<Iconify icon="solar:arrow-left-bold" width={20} height={20} />}
                          >
                            Back
                          </Button>
                          <Button
                            colorScheme={orchestrateMode === 'scan' ? 'brand' : 'blue'}
                            onClick={async () => {
                              if (orchestrateMode === 'scan') {
                                await executeScanJob();
                              } else {
                                await executePrintJob();
                              }
                              setOrchestrateStep(1);
                              setOrchestrateMode(null);
                            }}
                            size="lg"
                            px={8}
                            rightIcon={<Iconify icon="solar:play-circle-bold" width={24} height={24} />}
                          >
                            Start {orchestrateMode === 'scan' ? 'Scanning' : 'Printing'}
                          </Button>
                        </ModalFooter>
                      </>
                    )}
                  </Box>
                </Flex>
              </MotionModalContent>
            </Modal>
          )}

          {/* Document Selector Modal */}
          <DocumentSelector
            ref={documentSelectorRef}
            isOpen={documentSelectorModal.isOpen}
            onClose={() => {
              setUserClosedDocSelector(true);
              documentSelectorModal.onClose();
            }}
            onSelect={async (docs) => {
              console.log('[DOC_SELECT] onSelect called with docs:', docs);
              // Enhance documents with page information before setting
              const enhancedDocs = await enhanceDocumentsWithPages(docs);
              console.log('[DOC_SELECT] Enhanced docs:', enhancedDocs);
              setSelectedDocuments(enhancedDocs);
              if (enhancedDocs.length > 0) {
                bumpPreviewFocus({ docIndex: 0, page: 1, source: 'manual' });
              }
              documentSelectorModal.onClose();
            }}
            currentDocuments={currentDocumentOptions}
            convertedDocuments={convertedDocumentOptions}
            allowMultiple={true}
            mode={orchestrateMode || 'print'}
            isChatVisible={isChatVisible || orchestrationContext === 'voice'}
            chatWidth={chatWidth}
            selectedFilenames={selectedDocuments.map(d => d.filename)}
          />
        </DashboardShell>
      </PageShell>

      {/* Connectivity Status Modal */}
      <Modal
        isOpen={connectivityModal.isOpen}
        onClose={connectivityModal.onClose}
        size="lg"
        isCentered
      >
        <ModalOverlay bg="blackAlpha.600" />
        <MotionModalContent
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.8 }}
          transition={{ type: 'spring', stiffness: 400, damping: 30 }}
          bg={systemModalBgGradient}
          borderRadius="2xl"
          border="1px solid"
          borderColor="rgba(121,95,238,0.2)"
          boxShadow="0 20px 60px rgba(0,0,0,0.3)"
        >
          <ModalHeader
            fontSize="xl"
            fontWeight="bold"
            display="flex"
            alignItems="center"
            gap={2}
            pb={2}
          >
            <Iconify icon={FiActivity} boxSize={5} color="brand.400" />
            System Status
          </ModalHeader>
          <ModalCloseButton colorScheme="brand" size="md" top={3} right={3} />

          <ModalBody py={4}>
            <Box
              bg={validatorBoxBg}
              borderRadius="xl"
              p={4}
              border="1px solid"
              borderColor="rgba(121,95,238,0.15)"
            >
              <ConnectionValidator
                ref={connectionStatusRef}
                variant="minimal"
                autoRun={true}
                onStatusComplete={(allConnected) => {
                  if (allConnected) {
                    toast({
                      title: '✅ All Systems Connected',
                      description: 'PrintChakra is ready to go!',
                      status: 'success',
                      duration: 3000,
                      isClosable: true,
                    });
                  }
                }}
              />
            </Box>
          </ModalBody>

          <ModalFooter pt={2} pb={4}>
            <HStack spacing={3} w="100%">
              <Button
                variant="ghost"
                flex={1}
                onClick={connectivityModal.onClose}
                colorScheme="gray"
                size="sm"
              >
                Close
              </Button>
              <Button
                colorScheme="brand"
                flex={1}
                size="sm"
                leftIcon={<Iconify icon={FiRefreshCw} />}
                onClick={() => {
                  connectionStatusRef.current?.runCheck();
                }}
              >
                Re-check
              </Button>
            </HStack>
          </ModalFooter>
        </MotionModalContent>
      </Modal>

      {/* Device Info Modal - Updated for Calibration */}
      <DeviceInfoPanel
        isOpen={deviceInfoModal.isOpen}
        onClose={deviceInfoModal.onClose}
        showButton={false}
      />

      {/* AI Chat Sidebar - Independent Fixed Position */}
      {isChatVisible && (
        <Box
          position="fixed"
          top="0"
          bottom="0"
          right="0"
          w={{ base: '100%', lg: `${chatWidth}px` }}
          bg={chatSidebarBg}
          boxShadow="-4px 0 16px rgba(0,0,0,0.3)"
          display="flex"
          flexDirection="column"
          borderLeft="1px solid"
          borderColor={chatSidebarBorderColor}
          overflowY="auto"
          zIndex={orchestrateModal.isOpen ? 2003 : 2000}
          transition={isResizingChat ? 'none' : 'transform 0.3s ease-out, z-index 0.1s'}
        >
          {/* Resize Handle */}
          <Box
            position="absolute"
            left="0"
            top="0"
            bottom="0"
            w="6px"
            cursor="ew-resize"
            bg="transparent"
            _hover={{ bg: 'blue.400', opacity: 0.5 }}
            _active={{ bg: 'blue.500', opacity: 0.7 }}
            onMouseDown={handleChatResizeStart}
            zIndex={2004}
            display={{ base: 'none', lg: 'block' }}
          />
          <VoiceAIChat
            isOpen={isChatVisible}
            onClose={handleDockedChatClose}
            onOrchestrationTrigger={handleVoiceOrchestrationTrigger}
            isMinimized={false}
            onToggleMinimize={handleDockedChatClose}
            onVoiceCommand={handleVoiceCommand}
            autoStartRecording={isChatVisible}
          />
        </Box>
      )}
    </Box>
  );
};

export default Dashboard;
