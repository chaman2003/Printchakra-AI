/**
 * OCR Structured View Component
 * Displays OCR results in a structured, readable format with confidence indicators
 */

import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  VStack,
  HStack,
  Text,
  Badge,
  Divider,
  Button,
  Collapse,
  Progress,
  useColorModeValue,
  IconButton,
  Tooltip,
  Flex,
  Heading,
  Image,
  Spinner,
} from '@chakra-ui/react';
import Iconify from '../common/Iconify';
import { SelectableTextOverlay } from './SelectableTextOverlay';
import { OCRResult, OCRStructuredUnit, OCRRawResult } from '../../types';

interface OCRStructuredViewProps {
  ocrResult?: OCRResult;
  result?: OCRResult;  // Alternative prop name for compatibility
  filename?: string;
  imageUrl?: string;  // URL to the document image for image view
  onClose?: () => void;
  onRetry?: () => void;  // Callback to retry OCR
  isRetrying?: boolean;  // Show loading state during retry
  maxHeight?: string;
}

const UnitTypeBadge: React.FC<{ type: string }> = ({ type }) => {
  const colorSchemes: Record<string, string> = {
    title: 'purple',
    heading: 'blue',
    paragraph: 'green',
    list_item: 'cyan',
    table_cell: 'orange',
    footer: 'gray',
    other: 'pink',
  };

  return (
    <Badge
      colorScheme={colorSchemes[type] || 'gray'}
      variant="subtle"
      fontSize="xx-small"
      textTransform="uppercase"
    >
      {type.replace('_', ' ')}
    </Badge>
  );
};

const ConfidenceIndicator: React.FC<{ confidence: number }> = ({ confidence }) => {
  const percent = confidence * 100;
  let colorScheme = 'green';
  if (percent < 60) colorScheme = 'red';
  else if (percent < 80) colorScheme = 'orange';
  else if (percent < 95) colorScheme = 'yellow';

  return (
    <Tooltip label={`Confidence: ${percent.toFixed(1)}%`}>
      <HStack spacing={1} fontSize="xs">
        <Progress
          value={percent}
          size="xs"
          colorScheme={colorScheme}
          w="40px"
          borderRadius="full"
        />
        <Text color="text.muted" fontSize="xx-small">
          {percent.toFixed(0)}%
        </Text>
      </HStack>
    </Tooltip>
  );
};

const StructuredUnitCard: React.FC<{ unit: OCRStructuredUnit; index: number }> = ({ unit, index }) => {
  const bgColor = useColorModeValue('gray.50', 'whiteAlpha.50');
  const borderColor = useColorModeValue('gray.200', 'whiteAlpha.100');

  return (
    <Box
      p={3}
      bg={bgColor}
      borderRadius="md"
      border="1px solid"
      borderColor={borderColor}
      transition="all 0.2s"
      _hover={{ borderColor: 'brand.400', bg: useColorModeValue('gray.100', 'whiteAlpha.100') }}
    >
      <HStack justify="space-between" mb={2}>
        <HStack spacing={2}>
          <Text fontSize="xs" color="text.muted">#{index + 1}</Text>
          <UnitTypeBadge type={unit.type} />
        </HStack>
        <ConfidenceIndicator confidence={unit.confidence} />
      </HStack>
      <Text fontSize="sm" lineHeight="tall">
        {unit.text}
      </Text>
    </Box>
  );
};

const RawResultItem: React.FC<{ result: OCRRawResult; index: number }> = ({ result, index }) => {
  const bgColor = useColorModeValue('gray.50', 'whiteAlpha.50');

  return (
    <HStack
      p={2}
      bg={bgColor}
      borderRadius="md"
      spacing={3}
      fontSize="sm"
    >
      <Text color="text.muted" fontSize="xs" minW="24px">
        {index + 1}
      </Text>
      <Text flex={1} noOfLines={1}>
        {result.text}
      </Text>
      <ConfidenceIndicator confidence={result.confidence} />
    </HStack>
  );
};

export const OCRStructuredView: React.FC<OCRStructuredViewProps> = ({
  ocrResult: ocrResultProp,
  result,
  filename,
  imageUrl,
  onClose,
  onRetry,
  isRetrying = false,
  maxHeight = '70vh',
}) => {
  const [viewMode, setViewMode] = useState<'structured' | 'raw' | 'full' | 'image'>('structured');
  const [showStats, setShowStats] = useState(true);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const [zoom, setZoom] = useState(1);
  const imageContainerRef = useRef<HTMLDivElement>(null);
  const imageWrapperRef = useRef<HTMLDivElement>(null);

  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'whiteAlpha.200');

  // Support both prop names
  const ocrResult = ocrResultProp || result;
  
  // Check if OCR failed (no results)
  const isOCRFailed = ocrResult && (ocrResult.word_count === 0 || ocrResult.raw_results.length === 0);
  
  if (!ocrResult) {
    return (
      <Box p={4} textAlign="center">
        <Text color="text.muted">No OCR result available</Text>
        {onRetry && (
          <Button
            mt={3}
            colorScheme="blue"
            size="sm"
            leftIcon={<Iconify icon="mdi:refresh" />}
            onClick={onRetry}
            isLoading={isRetrying}
            loadingText="Running OCR..."
          >
            Run OCR
          </Button>
        )}
      </Box>
    );
  }
  
  // Show retry UI if OCR failed
  if (isOCRFailed) {
    return (
      <Box
        bg={bgColor}
        borderRadius="lg"
        border="1px solid"
        borderColor="orange.300"
        overflow="hidden"
        p={6}
        textAlign="center"
      >
        <VStack spacing={4}>
          <Iconify icon="mdi:alert-circle-outline" boxSize={12} color="orange.400" />
          <Heading size="md" color="orange.400">OCR Could Not Extract Text</Heading>
          <Text color="text.muted" fontSize="sm">
            The OCR process completed but could not detect any text in the image.
            This may happen if the image quality is low, text is too small, or the document is handwritten.
          </Text>
          <HStack spacing={3}>
            {onRetry && (
              <Button
                colorScheme="blue"
                leftIcon={<Iconify icon="mdi:refresh" />}
                onClick={onRetry}
                isLoading={isRetrying}
                loadingText="Retrying OCR..."
              >
                Retry OCR
              </Button>
            )}
            {onClose && (
              <Button variant="ghost" onClick={onClose}>
                Close
              </Button>
            )}
          </HStack>
          <Text fontSize="xs" color="text.muted">
            Processing time: {ocrResult.processing_time_ms.toFixed(0)}ms • 
            Image size: {ocrResult.image_dimensions[0]}×{ocrResult.image_dimensions[1]}
          </Text>
        </VStack>
      </Box>
    );
  }

  const hasStructured = ocrResult.structured_units && ocrResult.structured_units.length > 0;

  return (
    <Box
      bg={bgColor}
      borderRadius="lg"
      border="1px solid"
      borderColor={borderColor}
      overflow="hidden"
    >
      {/* Header */}
      <Flex
        p={3}
        borderBottom="1px solid"
        borderColor={borderColor}
        align="center"
        justify="space-between"
        bg={useColorModeValue('gray.50', 'whiteAlpha.50')}
      >
        <HStack spacing={3}>
          <Iconify icon="solar:document-text-bold" boxSize={5} color="brand.400" />
          <Box>
            <Heading size="sm">{ocrResult.derived_title || 'OCR Results'}</Heading>
            <Text fontSize="xs" color="text.muted">
              {ocrResult.word_count} words • {ocrResult.raw_results.length} text regions
            </Text>
          </Box>
        </HStack>
        <HStack>
          {onClose && (
            <IconButton
              aria-label="Close"
              icon={<Iconify icon="solar:close-circle-bold" />}
              size="sm"
              variant="ghost"
              onClick={onClose}
            />
          )}
        </HStack>
      </Flex>

      {/* Stats bar */}
      <Collapse in={showStats}>
        <HStack
          p={2}
          borderBottom="1px solid"
          borderColor={borderColor}
          spacing={4}
          fontSize="xs"
          flexWrap="wrap"
        >
          <HStack>
            <Text color="text.muted">Confidence:</Text>
            <Badge colorScheme={ocrResult.confidence_avg >= 0.8 ? 'green' : 'orange'}>
              {(ocrResult.confidence_avg * 100).toFixed(1)}%
            </Badge>
          </HStack>
          <HStack>
            <Text color="text.muted">Processing:</Text>
            <Text>{ocrResult.processing_time_ms.toFixed(0)}ms</Text>
          </HStack>
          <HStack>
            <Text color="text.muted">Size:</Text>
            <Text>{ocrResult.image_dimensions[0]}×{ocrResult.image_dimensions[1]}</Text>
          </HStack>
        </HStack>
      </Collapse>

      {/* View mode tabs */}
      <HStack p={2} borderBottom="1px solid" borderColor={borderColor} spacing={1}>
        <Button
          size="xs"
          variant={viewMode === 'structured' ? 'solid' : 'ghost'}
          colorScheme={viewMode === 'structured' ? 'brand' : 'gray'}
          onClick={() => setViewMode('structured')}
          isDisabled={!hasStructured}
        >
          Structured
        </Button>
        <Button
          size="xs"
          variant={viewMode === 'raw' ? 'solid' : 'ghost'}
          colorScheme={viewMode === 'raw' ? 'brand' : 'gray'}
          onClick={() => setViewMode('raw')}
        >
          Raw Regions
        </Button>
        <Button
          size="xs"
          variant={viewMode === 'full' ? 'solid' : 'ghost'}
          colorScheme={viewMode === 'full' ? 'brand' : 'gray'}
          onClick={() => setViewMode('full')}
        >
          Full Text
        </Button>
        {imageUrl && (
          <Button
            size="xs"
            variant={viewMode === 'image' ? 'solid' : 'ghost'}
            colorScheme={viewMode === 'image' ? 'blue' : 'gray'}
            onClick={() => setViewMode('image')}
            leftIcon={<Iconify icon="mdi:image-text" boxSize={3} />}
          >
            Image
          </Button>
        )}
        <Box flex={1} />
        <IconButton
          aria-label="Toggle stats"
          icon={<Iconify icon={showStats ? 'solar:eye-closed-bold' : 'solar:eye-bold'} />}
          size="xs"
          variant="ghost"
          onClick={() => setShowStats(!showStats)}
        />
      </HStack>

      {/* Content */}
      <Box maxH={viewMode === 'image' ? 'none' : maxHeight} overflowY="auto" p={3}>
        {viewMode === 'structured' && hasStructured && (
          <VStack spacing={2} align="stretch">
            {ocrResult.structured_units.map((unit, i) => (
              <StructuredUnitCard key={i} unit={unit} index={i} />
            ))}
          </VStack>
        )}

        {viewMode === 'raw' && (
          <VStack spacing={1} align="stretch">
            {ocrResult.raw_results.map((result, i) => (
              <RawResultItem key={i} result={result} index={i} />
            ))}
          </VStack>
        )}

        {viewMode === 'full' && (
          <Box>
            <Text
              fontSize="sm"
              lineHeight="tall"
              whiteSpace="pre-wrap"
              fontFamily="mono"
              p={3}
              bg={useColorModeValue('gray.50', 'whiteAlpha.50')}
              borderRadius="md"
            >
              {ocrResult.full_text || 'No text extracted'}
            </Text>
          </Box>
        )}

        {viewMode === 'structured' && !hasStructured && (
          <Text fontSize="sm" color="text.muted" textAlign="center" py={4}>
            No structured units available. View raw regions or full text.
          </Text>
        )}

        {viewMode === 'image' && imageUrl && (
          <Box>
            <HStack mb={2} justify="space-between">
              <Text fontSize="xs" color="text.muted">
                💡 Select text directly from the image to copy
              </Text>
              <HStack spacing={1}>
                <Button
                  size="xs"
                  variant="ghost"
                  leftIcon={<Iconify icon="mdi:minus" boxSize={4} />}
                  onClick={() => setZoom(Math.max(0.5, zoom - 0.1))}
                  isDisabled={zoom <= 0.5}
                >
                  Zoom Out
                </Button>
                <Text fontSize="xs" minW="45px" textAlign="center">
                  {Math.round(zoom * 100)}%
                </Text>
                <Button
                  size="xs"
                  variant="ghost"
                  leftIcon={<Iconify icon="mdi:plus" boxSize={4} />}
                  onClick={() => setZoom(Math.min(3, zoom + 0.1))}
                  isDisabled={zoom >= 3}
                >
                  Zoom In
                </Button>
                <Button
                  size="xs"
                  variant="ghost"
                  onClick={() => setZoom(1)}
                  isDisabled={zoom === 1}
                >
                  Reset
                </Button>
              </HStack>
            </HStack>
            <Box
              ref={imageContainerRef}
              display="flex"
              justifyContent="center"
              alignItems="center"
              bg={useColorModeValue('gray.100', 'whiteAlpha.100')}
              borderRadius="md"
              overflow="auto"
              h="calc(70vh - 160px)"
              onWheel={(e: React.WheelEvent) => {
                e.preventDefault();
                const delta = e.deltaY > 0 ? -0.05 : 0.05;
                setZoom((prev) => Math.max(0.5, Math.min(3, prev + delta)));
              }}
            >
              {!imageLoaded && (
                <Spinner size="lg" color="brand.400" />
              )}
              {/* Image wrapper with overlay positioned on top */}
              <Box
                ref={imageWrapperRef}
                position="relative"
                display={imageLoaded ? 'inline-block' : 'none'}
                transform={`scale(${zoom})`}
                transformOrigin="top center"
                transition="transform 0.2s ease-out"
              >
                <Image
                  src={imageUrl}
                  alt={filename || 'Document'}
                  maxW="100%"
                  maxH="calc(70vh - 160px)"
                  objectFit="contain"
                  display="block"
                  draggable={false}
                  onDragStart={(e) => e.preventDefault()}
                  sx={{ WebkitUserDrag: 'none' }}
                  onLoad={(e) => {
                    setImageLoaded(true);
                    const img = e.currentTarget;
                    // Use setTimeout to get accurate dimensions after render
                    setTimeout(() => {
                      setContainerSize({
                        width: img.clientWidth,
                        height: img.clientHeight,
                      });
                    }, 100);
                  }}
                />
                {imageLoaded && containerSize.width > 0 && ocrResult && (
                  <Box
                    position="absolute"
                    top={0}
                    left={0}
                    width={`${containerSize.width}px`}
                    height={`${containerSize.height}px`}
                  >
                    <SelectableTextOverlay
                      rawResults={ocrResult.raw_results}
                      imageDimensions={ocrResult.image_dimensions}
                      containerWidth={containerSize.width}
                      containerHeight={containerSize.height}
                      isActive={true}
                    />
                  </Box>
                )}
              </Box>
            </Box>
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default OCRStructuredView;
