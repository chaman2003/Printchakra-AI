/**
 * DocumentPreview Component
 * Responsive document preview with real-time settings updates
 *
 * 📐 TO ADJUST PREVIEW SIZE: See PREVIEW_SIZE configuration below (lines 22-32)
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Box,
  Flex,
  VStack,
  HStack,
  IconButton,
  Text,
  Button,
  ButtonGroup,
  useColorModeValue,
  Spinner,
  Badge,
  Tooltip,
  SimpleGrid,
  useBreakpointValue,
} from '@chakra-ui/react';
import Iconify from '../common/Iconify';
import SelectableTextOverlay from './SelectableTextOverlay';
import { OCRResult } from '../../types';

// ==================== PREVIEW SIZE CONFIGURATION ====================
// A4 paper aspect ratio: 1:1.414 (210mm x 297mm)
// Preview dimensions are calculated to maintain A4 proportions
const PREVIEW_SIZE = {
  // Portrait mode dimensions (A4 ratio: 210 x 297)
  portrait: {
    width: 28, // vw units
    height: 39.6, // vh units (28 * 1.414 for A4 ratio)
  },
  // Landscape mode dimensions (A4 rotated: 297 x 210)
  landscape: {
    width: 39.6, // vw units (28 * 1.414)
    height: 28, // vh units (matches portrait width for proper A4 landscape)
  },
  // Container minimum height
  containerMinHeight: '50vh', // Increase if preview gets cut off
};
// ===================================================================

interface DocumentPage {
  pageNumber: number;
  thumbnailUrl?: string;
  fullUrl?: string;
}

interface DocumentPreviewProps {
  documents: Array<{
    filename: string;
    pages?: DocumentPage[];
    thumbnailUrl?: string;
  }>;
  previewSettings?: {
    layout?: 'portrait' | 'landscape';
    scale?: number;
    colorMode?: 'color' | 'grayscale' | 'bw';
    paperSize?: string;
  };
  isLoading?: boolean;
  activeDocIndex?: number;
  activePage?: number;
  focusToken?: number;
  highlightSource?: 'manual' | 'voice';
  onRequestDocChange?: (nextIndex: number) => void;
  onRequestPageChange?: (nextPage: number) => void;
  /** OCR results for the current document - enables text selection on image */
  ocrResult?: OCRResult | null;
  /** Whether text selection mode is enabled */
  textSelectionEnabled?: boolean;
}

const DocumentPreview: React.FC<DocumentPreviewProps> = ({
  documents,
  previewSettings,
  isLoading = false,
  activeDocIndex,
  activePage,
  focusToken,
  highlightSource = 'manual',
  onRequestDocChange,
  onRequestPageChange,
  ocrResult,
  textSelectionEnabled = true,
}) => {
  const [currentDocIndex, setCurrentDocIndex] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [zoomLevel, setZoomLevel] = useState(100);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showThumbnails, setShowThumbnails] = useState(false);
  const [rotation, setRotation] = useState(0);
  const [viewMode, setViewMode] = useState<'single' | 'all'>('single'); // New: toggle between single page and all pages
  const [focusBadgeVisible, setFocusBadgeVisible] = useState(false);
  const [previewDimensions, setPreviewDimensions] = useState({ width: 0, height: 0 });

  const containerRef = useRef<HTMLDivElement | null>(null);
  const previewBoxRef = useRef<HTMLDivElement | null>(null);

  // Updated to match SurfaceCard theme
  const bgColor = useColorModeValue('rgba(255, 248, 240, 0.95)', 'rgba(12, 16, 35, 0.92)');
  const borderColor = useColorModeValue('rgba(121, 95, 238, 0.18)', 'rgba(69, 202, 255, 0.25)');
  const thumbnailBg = useColorModeValue('rgba(255, 245, 235, 0.5)', 'rgba(30, 34, 50, 0.9)');
  const paperBg = useColorModeValue('white', '#ffffff');
  const toolbarBg = useColorModeValue('rgba(255, 248, 240, 0.9)', 'rgba(12,16,35,0.95)');
  const hasActiveDoc = typeof activeDocIndex === 'number';
  const hasActivePage = typeof activePage === 'number';

  const currentDoc = documents[currentDocIndex];
  const totalPages = currentDoc?.pages?.length || 1;
  
  // Calculate total pages across ALL documents
  const totalPagesAllDocs = documents.reduce((sum, doc) => sum + (doc.pages?.length || 1), 0);

  // Get current page URL (if pages exist, use page-specific URL, otherwise use document thumbnail)
  const getCurrentPageUrl = () => {
    if (currentDoc?.pages && currentDoc.pages.length > 0) {
      const page = currentDoc.pages.find(p => p.pageNumber === currentPage);
      return page?.thumbnailUrl || page?.fullUrl || currentDoc.thumbnailUrl;
    }
    return currentDoc?.thumbnailUrl;
  };

  const currentPageUrl = getCurrentPageUrl();
  
  // Get all pages across all documents for "all pages" view
  const getAllPages = () => {
    const allPages: Array<{ docIndex: number; pageNumber: number; url: string; filename: string }> = [];
    documents.forEach((doc, docIndex) => {
      if (doc.pages && doc.pages.length > 0) {
        doc.pages.forEach(page => {
          allPages.push({
            docIndex,
            pageNumber: page.pageNumber,
            url: page.thumbnailUrl || page.fullUrl || doc.thumbnailUrl || '',
            filename: doc.filename,
          });
        });
      } else {
        allPages.push({
          docIndex,
          pageNumber: 1,
          url: doc.thumbnailUrl || '',
          filename: doc.filename,
        });
      }
    });
    return allPages;
  };
  
  const allPages = getAllPages();

  // Determine paper orientation and size based on settings
  const layout = previewSettings?.layout || 'portrait';
  const isLandscape = layout === 'landscape';

  const responsiveWidth = useBreakpointValue({ base: 85, md: 50, lg: 28 }) || 28;

  // Calculate responsive paper dimensions using configurable values - Like Chrome print preview
  const getPaperDimensions = () => {
    const ratio = 1.414; // A4 aspect ratio
    
    // Start with a base size that fits in most containers
    let baseWidth = 600; // pixels as base
    let baseHeight = baseWidth * ratio;

    if (isLandscape) {
      // Swap for landscape
      const temp = baseWidth;
      baseWidth = baseHeight;
      baseHeight = temp;
    }

    // Apply zoom and scale settings
    const scale = (zoomLevel / 100) * ((previewSettings?.scale || 100) / 100);
    
    return {
      width: `${baseWidth * scale}px`,
      height: `${baseHeight * scale}px`,
    };
  };

  const paperDimensions = getPaperDimensions();

  // Check if rotated sideways (90° or 270°)
  const isRotatedSideways = rotation === 90 || rotation === 270;
  const focusBorderColor = focusBadgeVisible
    ? highlightSource === 'voice'
      ? 'brand.300'
      : 'cyan.200'
    : 'gray.200';
  const focusShadow = focusBadgeVisible
    ? highlightSource === 'voice'
      ? '0 0 20px rgba(121,95,238,0.35), 0 4px 20px rgba(0,0,0,0.2)'
      : '0 0 20px rgba(56,189,248,0.25), 0 4px 20px rgba(0,0,0,0.2)'
    : '0 4px 20px rgba(0,0,0,0.15), 0 1px 4px rgba(0,0,0,0.1)';

  const handleZoomIn = useCallback(() => {
    setZoomLevel(v => Math.min(v + 10, 200));
  }, []);

  const handleZoomOut = useCallback(() => {
    setZoomLevel(v => Math.max(v - 10, 50));
  }, []);

  const handleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      containerRef.current?.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  }, []);

  const handleRotate = useCallback(() => {
    setRotation(prev => (prev + 90) % 360);
  }, []);

  const updateDocIndex = useCallback(
    (next: number | ((prev: number) => number)) => {
      setCurrentDocIndex(prev => {
        const target = typeof next === 'function' ? next(prev) : next;
        const clamped = Math.max(0, Math.min(documents.length ? documents.length - 1 : 0, target));
        onRequestDocChange?.(clamped);
        return clamped;
      });
    },
    [documents.length, onRequestDocChange]
  );

  const updatePageNumber = useCallback(
    (next: number | ((prev: number) => number)) => {
      setCurrentPage(prev => {
        const target = typeof next === 'function' ? next(prev) : next;
        const clamped = Math.max(1, Math.min(totalPages || 1, target));
        onRequestPageChange?.(clamped);
        return clamped;
      });
    },
    [onRequestPageChange, totalPages]
  );

  useEffect(() => {
    if (!documents.length) {
      setCurrentDocIndex(0);
      setCurrentPage(1);
      return;
    }
    if (hasActiveDoc) {
      return;
    }
    setCurrentDocIndex(prev => Math.min(prev, documents.length - 1));
  }, [documents.length, hasActiveDoc]);

  useEffect(() => {
    if (!hasActiveDoc || !documents.length) {
      return;
    }
    const safeIndex = Math.max(0, Math.min(activeDocIndex!, documents.length - 1));
    setCurrentDocIndex(safeIndex);
  }, [activeDocIndex, documents.length, hasActiveDoc]);

  useEffect(() => {
    if (hasActivePage) {
      return;
    }
    setCurrentPage(prev => Math.min(prev, totalPages));
  }, [totalPages, hasActivePage]);

  useEffect(() => {
    if (!hasActivePage || !documents.length) {
      return;
    }
    const safePage = Math.max(1, Math.min(activePage!, totalPages));
    setCurrentPage(safePage);
  }, [activePage, totalPages, hasActivePage, documents.length]);

  // Reset zoom and rotation when documents change
  useEffect(() => {
    setZoomLevel(100);
    setRotation(0);
    setCurrentPage(1);
    setViewMode('single'); // Reset to single page view
  }, [documents]);

  // Reset to page 1 when switching documents
  useEffect(() => {
    if (hasActivePage) {
      return;
    }
    setCurrentPage(1);
  }, [currentDocIndex, hasActivePage]);

  useEffect(() => {
    if (!focusToken) {
      return;
    }
    setFocusBadgeVisible(true);
    const timeoutId = setTimeout(() => setFocusBadgeVisible(false), 2400);
    return () => clearTimeout(timeoutId);
  }, [focusToken]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Arrow keys for page navigation
      if (e.key === 'ArrowLeft' && currentPage > 1) {
        updatePageNumber(prev => prev - 1);
      } else if (e.key === 'ArrowRight' && currentPage < totalPages) {
        updatePageNumber(prev => prev + 1);
      }
      // Ctrl/Cmd + Arrow keys for document navigation
      else if ((e.ctrlKey || e.metaKey) && e.key === 'ArrowLeft' && currentDocIndex > 0) {
        updateDocIndex(prev => prev - 1);
      } else if (
        (e.ctrlKey || e.metaKey) &&
        e.key === 'ArrowRight' &&
        currentDocIndex < documents.length - 1
      ) {
        updateDocIndex(prev => prev + 1);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentPage, totalPages, currentDocIndex, documents.length, updatePageNumber, updateDocIndex]);

  if (documents.length === 0) {
    return (
      <Flex
        direction="column"
        align="center"
        justify="center"
        h="100%"
        minH="500px"
        bg={bgColor}
        borderRadius="lg"
        border="1px solid"
        borderColor={borderColor}
        p={6}
        position="relative"
      >
        <Box
          width="380px"
          height="520px"
          bg={paperBg}
          borderRadius="sm"
          boxShadow="0 4px 16px rgba(0,0,0,0.1)"
          display="flex"
          alignItems="center"
          justifyContent="center"
          border="1px solid"
          borderColor="gray.200"
        >
          <VStack spacing={3}>
            <Iconify icon="solar:document-bold" width={48} height={48} color="gray.300" />
            <Text fontSize={{ base: 'sm', md: 'md' }} fontWeight="500" color="gray.500">
              No Document Selected
            </Text>
            <Text fontSize="xs" color="gray.400" textAlign="center" maxW="220px">
              Select a document to see preview
            </Text>
          </VStack>
        </Box>

        <Text mt={4} fontSize="xs" color="text.muted" fontWeight="500">
          Ready to preview
        </Text>
      </Flex>
    );
  }

  return (
    <Box
      ref={containerRef}
      bg={bgColor}
      borderRadius="2xl"
      border="1px solid"
      borderColor={borderColor}
      overflow="hidden"
      position="relative"
      h="100%"
      display="flex"
      flexDirection="column"
      boxShadow="2xl"
    >
      {/* Top Toolbar */}
      <Flex
        px={{ base: '0.75rem', md: '1rem' }}
        py="0.625rem"
        borderBottom="1px solid"
        borderColor={borderColor}
        bg="rgba(12,16,35,0.85)"
        justify="space-between"
        align="center"
        flexShrink={0}
        gap={2}
        flexWrap="wrap"
      >
        <HStack spacing={2} minW={0} flex={1}>
          <Text
            fontSize={{ base: 'xs', md: 'sm' }}
            fontWeight="600"
            noOfLines={1}
            color="whiteAlpha.900"
            flex={1}
            minW={0}
          >
            {currentDoc?.filename}
          </Text>
          {documents.length > 1 && (
            <Badge colorScheme="purple" fontSize="xs" flexShrink={0}>
              {currentDocIndex + 1}/{documents.length}
            </Badge>
          )}
          {totalPagesAllDocs > 1 && (
            <Badge colorScheme="cyan" fontSize="xs" flexShrink={0}>
              {totalPagesAllDocs} {totalPagesAllDocs === 1 ? 'page' : 'pages'} total
            </Badge>
          )}
          {focusBadgeVisible && (
            <Badge
              colorScheme={highlightSource === 'voice' ? 'pink' : 'blue'}
              fontSize="xs"
              flexShrink={0}
            >
              {highlightSource === 'voice' ? 'Voice selection' : 'Selection updated'}
            </Badge>
          )}
        </HStack>

        <HStack spacing={1} flexShrink={0} flexWrap="wrap">
          {/* View Mode Toggle */}
          {totalPagesAllDocs > 1 && (
            <ButtonGroup size="sm" isAttached variant="outline">
              <Tooltip label="Single Page View">
                <IconButton
                  aria-label="Single page view"
                  icon={<Iconify icon="solar:document-bold" width={14} height={14} />}
                  onClick={() => setViewMode('single')}
                  bg={viewMode === 'single' ? 'brand.500' : 'whiteAlpha.200'}
                  color="white"
                  _hover={{ bg: viewMode === 'single' ? 'brand.600' : 'whiteAlpha.300' }}
                  borderColor={viewMode === 'single' ? 'brand.500' : 'whiteAlpha.300'}
                />
              </Tooltip>
              <Tooltip label="All Pages View">
                <IconButton
                  aria-label="All pages view"
                  icon={<Iconify icon="solar:documents-bold" width={14} height={14} />}
                  onClick={() => setViewMode('all')}
                  bg={viewMode === 'all' ? 'brand.500' : 'whiteAlpha.200'}
                  color="white"
                  _hover={{ bg: viewMode === 'all' ? 'brand.600' : 'whiteAlpha.300' }}
                  borderColor={viewMode === 'all' ? 'brand.500' : 'whiteAlpha.300'}
                />
              </Tooltip>
            </ButtonGroup>
          )}

          <ButtonGroup size="sm" isAttached variant="outline">
            <Tooltip label="Zoom Out">
              <IconButton
                aria-label="Zoom out"
                icon={<Iconify icon="solar:magnifer-zoom-out-bold" width={14} height={14} />}
                onClick={handleZoomOut}
                isDisabled={zoomLevel <= 50}
                bg="whiteAlpha.200"
                color="white"
                _hover={{ bg: 'whiteAlpha.300' }}
                _dark={{ bg: 'whiteAlpha.100' }}
              />
            </Tooltip>
            <Button
              minW={{ base: '50px', md: '60px' }}
              fontSize="xs"
              bg="whiteAlpha.200"
              color="white"
              fontWeight="500"
              _hover={{ bg: 'whiteAlpha.300' }}
              _dark={{ bg: 'whiteAlpha.100' }}
            >
              {zoomLevel}%
            </Button>
            <Tooltip label="Zoom In">
              <IconButton
                aria-label="Zoom in"
                icon={<Iconify icon="solar:magnifer-zoom-in-bold" width={14} height={14} />}
                onClick={handleZoomIn}
                isDisabled={zoomLevel >= 200}
                bg="whiteAlpha.200"
                color="white"
                _hover={{ bg: 'whiteAlpha.300' }}
                _dark={{ bg: 'whiteAlpha.100' }}
              />
            </Tooltip>
          </ButtonGroup>

          {/* Rotate Button */}
          {viewMode === 'single' && (
            <Tooltip label="Rotate 90°">
              <IconButton
                aria-label="Rotate"
                icon={<Iconify icon="solar:refresh-bold" width={14} height={14} />}
                size="sm"
                onClick={handleRotate}
                bg="whiteAlpha.200"
                color="white"
                _hover={{ bg: 'whiteAlpha.300' }}
                _dark={{ bg: 'whiteAlpha.100' }}
              />
            </Tooltip>
          )}

          {/* Document Navigation (if multiple documents) - Only in single page mode */}
          {documents.length > 1 && viewMode === 'single' && (
            <ButtonGroup size="sm" isAttached variant="outline" ml={2}>
              <Tooltip label="Previous Document">
                <IconButton
                  aria-label="Previous document"
                  icon={<Iconify icon="solar:alt-arrow-left-bold" width={14} height={14} />}
                  onClick={() => updateDocIndex(prev => Math.max(0, prev - 1))}
                  isDisabled={currentDocIndex === 0}
                  bg="whiteAlpha.200"
                  color="white"
                  _hover={{ bg: 'whiteAlpha.300' }}
                  _dark={{ bg: 'whiteAlpha.100' }}
                />
              </Tooltip>
              <Button
                minW={{ base: '60px', md: '70px' }}
                fontSize="xs"
                bg="whiteAlpha.200"
                color="white"
                fontWeight="500"
                _hover={{ bg: 'whiteAlpha.300' }}
                _dark={{ bg: 'whiteAlpha.100' }}
                cursor="default"
              >
                Doc {currentDocIndex + 1}/{documents.length}
              </Button>
              <Tooltip label="Next Document">
                <IconButton
                  aria-label="Next document"
                  icon={<Iconify icon="solar:alt-arrow-right-bold" width={14} height={14} />}
                  onClick={() =>
                    updateDocIndex(prev => Math.min(documents.length - 1, prev + 1))
                  }
                  isDisabled={currentDocIndex === documents.length - 1}
                  bg="whiteAlpha.200"
                  color="white"
                  _hover={{ bg: 'whiteAlpha.300' }}
                  _dark={{ bg: 'whiteAlpha.100' }}
                />
              </Tooltip>
            </ButtonGroup>
          )}

          {/* Page Navigation (if multiple pages in current document) - Only in single page mode */}
          {totalPages > 1 && viewMode === 'single' && (
            <ButtonGroup size="sm" isAttached variant="outline" ml={2}>
              <Tooltip label="Previous Page">
                <IconButton
                  aria-label="Previous page"
                  icon={<Iconify icon="solar:alt-arrow-left-bold" width={14} height={14} />}
                  onClick={() => updatePageNumber(prev => Math.max(1, prev - 1))}
                  isDisabled={currentPage === 1}
                  bg="whiteAlpha.200"
                  color="white"
                  _hover={{ bg: 'whiteAlpha.300' }}
                  _dark={{ bg: 'whiteAlpha.100' }}
                />
              </Tooltip>
              <Button
                minW={{ base: '50px', md: '60px' }}
                fontSize="xs"
                bg="whiteAlpha.200"
                color="white"
                fontWeight="500"
                _hover={{ bg: 'whiteAlpha.300' }}
                _dark={{ bg: 'whiteAlpha.100' }}
                cursor="default"
              >
                {currentPage}/{totalPages}
              </Button>
              <Tooltip label="Next Page">
                <IconButton
                  aria-label="Next page"
                  icon={<Iconify icon="solar:alt-arrow-right-bold" width={14} height={14} />}
                  onClick={() => updatePageNumber(prev => Math.min(totalPages, prev + 1))}
                  isDisabled={currentPage === totalPages}
                  bg="whiteAlpha.200"
                  color="white"
                  _hover={{ bg: 'whiteAlpha.300' }}
                  _dark={{ bg: 'whiteAlpha.100' }}
                />
              </Tooltip>
            </ButtonGroup>
          )}
        </HStack>
      </Flex>

      {/* Content Area */}
      <Flex flex={1} overflow="hidden" position="relative">
        {/* Thumbnails Sidebar */}
        {showThumbnails && (
          <VStack
            w="120px"
            borderRight="1px solid"
            borderColor={borderColor}
            bg={thumbnailBg}
            p="0.5rem"
            spacing={2}
            overflowY="auto"
            flexShrink={0}
            display={{ base: 'none', lg: 'flex' }}
            css={{
              '&::-webkit-scrollbar': { width: '6px' },
              '&::-webkit-scrollbar-thumb': {
                background: 'rgba(121,95,238,0.4)',
                borderRadius: '10px',
              },
            }}
          >
            {currentDoc?.pages?.map(page => (
              <Box
                key={page.pageNumber}
                w="100%"
                cursor="pointer"
                onClick={() => updatePageNumber(page.pageNumber)}
                
                border="2px solid"
                borderColor={currentPage === page.pageNumber ? 'brand.400' : 'transparent'}
                borderRadius="md"
                overflow="hidden"
                bg={bgColor}
                transition="all 0.2s"
                _hover={{ borderColor: 'brand.300' }}
              >
                <Box h="80px" bg="gray.100">
                  {page.thumbnailUrl ? (
                    <img
                      src={page.thumbnailUrl}
                      alt={`Page ${page.pageNumber}`}
                      style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                    />
                  ) : (
                    <Flex align="center" justify="center" h="100%">
                      <Text fontSize="lg" color="gray.400">
                        {page.pageNumber}
                      </Text>
                    </Flex>
                  )}
                </Box>
                <Text
                  fontSize="xs"
                  textAlign="center"
                  py={1}
                  fontWeight={currentPage === page.pageNumber ? '600' : '400'}
                >
                  {page.pageNumber}
                </Text>
              </Box>
            ))}
          </VStack>
        )}

        {/* Main Preview */}
        <Flex
          flex={1}
          align="center"
          justify="center"
          p={{ base: '0.5rem', md: '1rem', lg: '1.5rem' }}
          minH={PREVIEW_SIZE.containerMinHeight}
          overflow="auto"
          bg={bgColor}
          css={{
            '&::-webkit-scrollbar': { width: '8px', height: '8px' },
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
          {isLoading ? (
            <VStack spacing={4}>
              <Spinner size="xl" color="brand.500" thickness="4px" />
              <Text color="text.muted">Loading preview...</Text>
            </VStack>
          ) : viewMode === 'all' ? (
            // ALL PAGES VIEW - Show all pages in a grid
            <VStack spacing={6} w="100%" maxW="1200px" p={4}>
              <Text fontSize="lg" fontWeight="600" color="whiteAlpha.900" alignSelf="start">
                All Pages ({allPages.length})
              </Text>
              <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={6} w="100%">
                {allPages.map((page, index) => (
                  <Box
                    key={`${page.docIndex}-${page.pageNumber}`}
                    borderRadius="md"
                    overflow="hidden"
                    bg={paperBg}
                    boxShadow="0 2px 12px rgba(0,0,0,0.1)"
                    border="2px solid"
                    borderColor="whiteAlpha.300"
                    transition="all 0.3s"
                    _hover={{
                      transform: 'translateY(-4px)',
                      boxShadow: '0 8px 24px rgba(121,95,238,0.3)',
                      borderColor: 'brand.400',
                    }}
                    cursor="pointer"
                    onClick={() => {
                      updateDocIndex(page.docIndex);
                      updatePageNumber(page.pageNumber);
                      setViewMode('single');
                    }}
                  >
                    <Box position="relative" w="100%" h="350px" bg="gray.100">
                      {page.url ? (
                        <img
                          src={page.url}
                          alt={`${page.filename} - Page ${page.pageNumber}`}
                          style={{
                            width: '100%',
                            height: '100%',
                            objectFit: 'contain',
                            filter: 'none',
                          }}
                        />
                      ) : (
                        <Flex align="center" justify="center" h="100%">
                          <VStack spacing={2}>
                            <Iconify
                              icon="solar:document-bold"
                              width={48}
                              height={48}
                              color="gray.300"
                            />
                            <Text color="gray.500" fontSize="sm">
                              Preview not available
                            </Text>
                          </VStack>
                        </Flex>
                      )}
                      <Badge
                        position="absolute"
                        top={2}
                        left={2}
                        colorScheme="brand"
                        fontSize="xs"
                        px={2}
                        py={1}
                      >
                        Page {index + 1}
                      </Badge>
                    </Box>
                    <Box p={3} bg="whiteAlpha.100">
                      <Text fontSize="sm" fontWeight="600" noOfLines={1} color="whiteAlpha.900">
                        {page.filename}
                      </Text>
                      <HStack spacing={2} mt={1}>
                        <Badge colorScheme="purple" fontSize="xs">
                          Doc {page.docIndex + 1}
                        </Badge>
                        <Badge colorScheme="cyan" fontSize="xs">
                          Page {page.pageNumber}
                        </Badge>
                      </HStack>
                    </Box>
                  </Box>
                ))}
              </SimpleGrid>
            </VStack>
          ) : (
            // SINGLE PAGE VIEW - Show one page at a time (Like Chrome Print Preview)
            <Box
              width={paperDimensions.width}
              height={paperDimensions.height}
              maxW="100%"
              maxH="100%"
              transition="width 0.3s ease, height 0.3s ease, border-color 0.2s ease, box-shadow 0.2s ease"
              borderRadius="sm"
              boxShadow={focusShadow}
              bg={paperBg}
              border="2px solid"
              borderColor={focusBorderColor}
              position="relative"
              display="flex"
              alignItems="center"
              justifyContent="center"
              overflow="hidden"
              flexShrink={0}
              ref={previewBoxRef}
            >
              {currentPageUrl ? (
                <>
                  <img
                    key={`${currentDocIndex}-${currentPage}-${rotation}`}
                    src={currentPageUrl}
                    alt={`${currentDoc.filename} - Page ${currentPage}`}
                    onLoad={(e) => {
                      const img = e.target as HTMLImageElement;
                      const container = previewBoxRef.current;
                      if (container) {
                        setPreviewDimensions({
                          width: container.clientWidth,
                          height: container.clientHeight,
                        });
                      }
                    }}
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'contain',
                      display: 'block',
                      transformOrigin: 'center center',
                      transform: `rotate(${rotation}deg)`,
                      filter: 'none',
                      transition: 'transform 0.3s ease-out',
                    }}
                  />
                  {/* OCR Text Overlay for copy/paste */}
                  {textSelectionEnabled && ocrResult && ocrResult.raw_results && ocrResult.image_dimensions && rotation === 0 && (
                    <SelectableTextOverlay
                      rawResults={ocrResult.raw_results}
                      imageDimensions={ocrResult.image_dimensions}
                      containerWidth={previewDimensions.width}
                      containerHeight={previewDimensions.height}
                      isActive={true}
                    />
                  )}
                </>
              ) : (
                <VStack spacing={3}>
                  <Iconify icon="solar:document-bold" width={48} height={48} color="gray.300" />
                  <Text color="gray.500" fontSize="sm">
                    Preview not available
                  </Text>
                </VStack>
              )}
            </Box>
          )}
        </Flex>
      </Flex>
    </Box>
  );
};

export default DocumentPreview;
