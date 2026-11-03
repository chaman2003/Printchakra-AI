import React from 'react';
import { withToken } from '../../auth/session';
import { Box, Button, Flex, Grid, HStack, Image, Text, useColorModeValue } from '@chakra-ui/react';
import { FiCamera, FiTrash2 } from 'react-icons/fi';
import { Iconify } from '../common';
import { API_BASE_URL } from '../../config';

export interface TestCapture {
  filename: string;
  url: string;
  size: number;
  timestamp: string;
}

interface TestCapturesPreviewProps {
  captures: TestCapture[];
  textMuted: string;
  borderColor: string;
  onClearAll: () => Promise<void>;
}

export const TestCapturesPreview: React.FC<TestCapturesPreviewProps> = ({
  captures,
  textMuted,
  borderColor,
  onClearAll,
}) => {
  if (captures.length === 0) {
    return null;
  }

  return (
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
            Test Captures ({captures.length})
          </Text>
        </HStack>
        <Button
          size="xs"
          colorScheme="red"
          variant="ghost"
          leftIcon={<Iconify icon={FiTrash2} boxSize={3} />}
          onClick={onClearAll}
        >
          Clear All
        </Button>
      </Flex>
      <Text fontSize="xs" color={textMuted} mb={3}>
        These images were captured during testing and will be automatically deleted when you close this screen.
      </Text>
      <Grid templateColumns="repeat(3, 1fr)" gap={2}>
        {captures.slice(0, 9).map((capture) => (
          <Box
            key={capture.filename}
            borderRadius="md"
            overflow="hidden"
            border="1px solid"
            borderColor={borderColor}
            position="relative"
          >
            <Image
              src={withToken(`${API_BASE_URL}${capture.url}`)}
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
            <Box position="absolute" bottom={0} left={0} right={0} bg="blackAlpha.600" px={1} py={0.5}>
              <Text fontSize="8px" color="white" noOfLines={1}>
                {new Date(capture.timestamp).toLocaleTimeString()}
              </Text>
            </Box>
          </Box>
        ))}
      </Grid>
      {captures.length > 9 && (
        <Text fontSize="xs" color={textMuted} mt={2} textAlign="center">
          +{captures.length - 9} more capture(s)
        </Text>
      )}
    </Box>
  );
};

export default TestCapturesPreview;
