/**
 * SelectableTextOverlay Component
 * Renders OCR text as invisible but selectable text over an image
 * Allows users to select and copy text from document images
 */

import React, { useMemo, useState } from 'react';
import { Box, Text } from '@chakra-ui/react';
import { OCRRawResult } from '../../types';

interface SelectableTextOverlayProps {
  /** Raw OCR results with bounding boxes and text */
  rawResults: OCRRawResult[];
  /** Original image dimensions [width, height] */
  imageDimensions: [number, number];
  /** Container dimensions for scaling */
  containerWidth: number;
  containerHeight: number;
  /** Whether the overlay is active/visible */
  isActive?: boolean;
}

interface TextRegion {
  text: string;
  left: number;
  top: number;
  width: number;
  height: number;
  fontSize: number;
}

export const SelectableTextOverlay: React.FC<SelectableTextOverlayProps> = ({
  rawResults,
  imageDimensions,
  containerWidth,
  containerHeight,
  isActive = true,
}) => {
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  // Calculate scaled text regions
  const textRegions = useMemo<TextRegion[]>(() => {
    if (!rawResults || rawResults.length === 0) return [];
    if (imageDimensions[0] === 0 || imageDimensions[1] === 0) return [];
    if (containerWidth === 0 || containerHeight === 0) return [];

    const scaleX = containerWidth / imageDimensions[0];
    const scaleY = containerHeight / imageDimensions[1];

    return rawResults.map((result) => {
      const { bbox, text } = result;
      
      // Scale coordinates
      const left = bbox.x * scaleX;
      const top = bbox.y * scaleY;
      const width = bbox.width * scaleX;
      const height = bbox.height * scaleY;
      
      // Calculate font size to fit text in box
      // Estimate based on height, adjusted for typical character width
      const estimatedFontSize = Math.max(8, Math.min(height * 0.85, 24));

      return {
        text,
        left,
        top,
        width,
        height,
        fontSize: estimatedFontSize,
      };
    });
  }, [rawResults, imageDimensions, containerWidth, containerHeight]);

  if (!isActive || textRegions.length === 0) {
    return null;
  }

  return (
    <Box
      position="absolute"
      top={0}
      left={0}
      width={`${containerWidth}px`}
      height={`${containerHeight}px`}
      overflow="hidden"
      zIndex={10}
      pointerEvents="none"
      userSelect="none"
    >
      {textRegions.map((region, index) => (
        <Text
          key={index}
          position="absolute"
          left={`${region.left}px`}
          top={`${region.top}px`}
          width={`${region.width}px`}
          height={`${region.height}px`}
          fontSize={`${region.fontSize}px`}
          lineHeight="1.2"
          overflow="hidden"
          whiteSpace="normal"
          wordBreak="break-word"
          cursor="text"
          pointerEvents="auto"
          userSelect={activeIndex === index ? 'text' : 'none'}
          color="transparent"
          bg="transparent"
          onMouseEnter={() => setActiveIndex(index)}
          onMouseLeave={() => setActiveIndex(null)}
          _hover={{
            bg: 'rgba(59, 130, 246, 0.15)',
            color: activeIndex === index ? 'rgba(0, 0, 0, 0.85)' : 'transparent',
          }}
          _selection={{
            bg: 'rgba(59, 130, 246, 0.5)',
            color: '#000',
          }}
        >
          {region.text}
        </Text>
      ))}
    </Box>
  );
};

export default SelectableTextOverlay;
