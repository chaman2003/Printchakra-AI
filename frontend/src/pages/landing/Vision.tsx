import React from 'react';
import { Badge, Box, Flex, Grid, Heading, HStack, Text, VStack } from '@chakra-ui/react';
import { LuLayers, LuPrinter } from 'react-icons/lu';
import { Reveal } from '../../components/brand/Reveal';

const Layers = LuLayers as any;
const Printer = LuPrinter as any;

// Real figures from a test scan (an invoice photo) run on the live app.
const LOG = [
  { text: '> POST /phone/upload  invoice-photo.jpg', color: '#7ee787' },
  { text: '[INFO] 12-step cleanup: edges, perspective, shadows', color: '#f6f6f6' },
  { text: '[INFO] Gemini OCR: 89 words in 40 regions', color: '#f6f6f6' },
  { text: '[READY] processing_complete -> dashboard', color: '#f6f6f6' },
];

/** "Project Vision": statement and two points on the left, a terminal mock on the right. */
export const Vision: React.FC = () => (
  <Box as="section" id="vision" scrollMarginTop="80px" px={{ base: 5, md: 10 }} py={{ base: 20, md: 32 }}>
    <Box maxW="1600px" mx="auto">
      <Grid
        templateColumns={{ base: '1fr', lg: '1fr 1fr' }}
        gap={{ base: 12, lg: 16 }}
        p={{ base: 7, md: 14 }}
        border="1px solid rgba(30,30,30,0.12)"
        borderRadius="sm"
        bg="#ffffff"
      >
        <VStack align="start" spacing={6}>
          <Reveal>
            <Badge bg="#111111" color="#f2f2f2">Project Vision</Badge>
          </Reveal>
          <Reveal delay={100}>
            <Heading as="h2" fontSize="clamp(36px, 4.6vw, 68px)" letterSpacing="-0.05em" lineHeight={0.95}>
              Document Intelligence{' '}
              <Box as="em" fontFamily="serif" fontStyle="italic" fontWeight={400} letterSpacing="-0.02em" color="#838282">
                without
              </Box>{' '}
              the friction.
            </Heading>
          </Reveal>
          <Reveal delay={200}>
            <Text fontSize="18px" color="#838282" lineHeight={1.7}>
              We built PrintChakra to solve one problem: the "Physical-to-Digital" gap. Traditional scanners are slow, and moving a
              page from paper to a searchable file takes too many steps. PrintChakra combines <b>AI cleanup and text reading</b>{' '}
              with <b>live sync between your phone and your desktop</b> to make that a single motion.
            </Text>
          </Reveal>

          <Grid templateColumns="1fr 1fr" gap={8} w="100%" pt={4}>
            <Reveal delay={250}>
              <VStack align="start" spacing={2}>
                <Layers size={24} />
                <Text fontWeight={700}>Distributed Architecture</Text>
                <Text fontSize="14px" color="#838282">Frontend, backend and phone capture, all in sync.</Text>
              </VStack>
            </Reveal>
            <Reveal delay={320}>
              <VStack align="start" spacing={2}>
                <Printer size={24} />
                <Text fontWeight={700}>Print Anywhere</Text>
                <Text fontSize="14px" color="#838282">Uses your browser's print dialog, so any printer works.</Text>
              </VStack>
            </Reveal>
          </Grid>
        </VStack>

        <Reveal kind="clip" delay={150}>
          <Box position="relative" bg="#111111" borderRadius="sm" p={6} minH="320px" h="100%" overflow="hidden">
            <HStack spacing={2} mb={8}>
              {['#838282', '#838282', '#838282'].map((c, i) => (
                <Box key={i} w={3} h={3} borderRadius="full" border="1px solid #838282" />
              ))}
            </HStack>
            <VStack align="start" spacing={4} fontFamily="mono">
              {LOG.map((line) => (
                <Text key={line.text} fontSize="13px" color={line.color}>{line.text}</Text>
              ))}
              <Text color="rgba(246,246,246,0.35)" fontSize="13px">------------------------------------------</Text>
              <Flex w="100%" justify="space-between" fontSize="13px" color="#f6f6f6">
                <span>Cleanup</span><span style={{ color: '#7ee787' }}>~4s</span>
              </Flex>
              <Flex w="100%" justify="space-between" fontSize="13px" color="#f6f6f6">
                <span>Text recognition</span><span style={{ color: '#7ee787' }}>~6s</span>
              </Flex>
              <Text color="rgba(246,246,246,0.6)" fontSize="13px">&gt; Awaiting voice command...</Text>
            </VStack>
          </Box>
        </Reveal>
      </Grid>
    </Box>
  </Box>
);

export default Vision;
