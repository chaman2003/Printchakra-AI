import React from 'react';
import { Badge, Box, Flex, Grid, Heading, HStack, Text, VStack } from '@chakra-ui/react';
import { LuBox, LuCloudLightning, LuDatabase, LuEye, LuMic, LuServer, LuShield, LuTriangle, LuWorkflow } from 'react-icons/lu';
import { Reveal } from '../../components/brand/Reveal';

const asIcon = (icon: any) => icon as React.ElementType;

// Same three highlights as the original page. The wording now matches how the live app works.
const HIGHLIGHTS = [
  {
    title: 'Live Sync',
    desc: 'Captures from your phone reach your dashboard within seconds. No refresh, no cables, no pairing.',
    icon: LuCloudLightning,
    shape: 'square',
  },
  {
    title: 'AI Extraction',
    desc: 'Gemini vision reads every line with its position and type, so text can be searched, selected and copied. Pages are sent to Google for reading.',
    icon: LuShield,
    shape: 'circle',
  },
  {
    title: 'Voice Orchestration',
    desc: 'A Gemini Live voice assistant that operates the app: open a page, run OCR, start a print.',
    icon: LuMic,
    shape: 'triangle',
  },
];

const STACK = [
  { name: 'Gemini', icon: LuEye },
  { name: 'Flask', icon: LuServer },
  { name: 'Neon Postgres', icon: LuDatabase },
  { name: 'OpenCV', icon: LuWorkflow },
  { name: 'Vercel', icon: LuTriangle },
  { name: 'React 19', icon: LuBox },
];

const Mark: React.FC<{ shape: string }> = ({ shape }) => {
  const shapes: Record<string, React.CSSProperties> = {
    square: { width: 24, height: 24, background: '#111' },
    circle: { width: 26, height: 26, borderRadius: '50%', background: '#111' },
    triangle: { width: 28, height: 26, background: '#111', clipPath: 'polygon(50% 0, 100% 100%, 0 100%)' },
  };
  return <span aria-hidden="true" style={shapes[shape]} />;
};

/** "Inside the Engine": heading, three highlight columns, then the core stack row. */
export const Engine: React.FC = () => (
  <Box as="section" id="engine" scrollMarginTop="80px" bg="#ffffff" borderTop="1px solid rgba(30,30,30,0.1)" borderBottom="1px solid rgba(30,30,30,0.1)" px={{ base: 5, md: 10 }} py={{ base: 20, md: 32 }}>
    <Box maxW="1600px" mx="auto">
      <VStack spacing={5} textAlign="center" mb={{ base: 14, md: 24 }}>
        <Reveal>
          <Badge variant="outline" border="1px solid #1e1e1e" bg="transparent" px={4} py={1}>
            Inside the Engine
          </Badge>
        </Reveal>
        <Reveal delay={100}>
          <Heading as="h2" fontSize="clamp(40px, 6vw, 92px)" letterSpacing="-0.05em" lineHeight={0.92} maxW="1000px">
            Built for{' '}
            <Box as="em" fontFamily="serif" fontStyle="italic" fontWeight={400} letterSpacing="-0.02em" color="#838282">
              industrial
            </Box>{' '}
            performance
          </Heading>
        </Reveal>
        <Reveal delay={200}>
          <Text maxW="640px" color="#838282" fontSize="18px" lineHeight={1.7}>
            PrintChakra isn't just a web app. It's an orchestration layer running on a Python AI backend, with a pipeline that
            straightens, cleans and reads every page you capture.
          </Text>
        </Reveal>
      </VStack>

      <Grid templateColumns={{ base: '1fr', md: 'repeat(3, 1fr)' }} gap="32px">
        {HIGHLIGHTS.map((item, index) => (
          <Reveal key={item.title} delay={index * 120}>
            <Flex
              role="group"
              direction="column"
              h="100%"
              p={{ base: 7, md: 9 }}
              border="1px solid rgba(30,30,30,0.1)"
              borderRadius="sm"
              bg="transparent"
              transition="background-color 300ms ease, border-color 300ms ease"
              _hover={{ bg: '#f2f2f2', borderColor: 'rgba(30,30,30,0.25)' }}
            >
              <Flex
                w="64px"
                h="64px"
                align="center"
                justify="center"
                border="1px solid rgba(30,30,30,0.15)"
                borderRadius="sm"
                mb={8}
                transition="transform 400ms cubic-bezier(0.77, 0, 0.175, 1)"
                _groupHover={{ transform: 'rotate(12deg)' }}
              >
                <Mark shape={item.shape} />
              </Flex>
              <Heading as="h3" fontSize="28px" letterSpacing="-0.04em" mb={3}>
                {item.title}
              </Heading>
              <Text color="#838282" fontWeight={400} lineHeight={1.7}>
                {item.desc}
              </Text>
            </Flex>
          </Reveal>
        ))}
      </Grid>

      <Box pt={{ base: 16, md: 24 }}>
        <Text fontSize="12px" fontWeight={700} letterSpacing="0.2em" textTransform="uppercase" color="#838282" textAlign="center" mb={10}>
          The Core Stack
        </Text>
        <Flex wrap="wrap" justify="center" gap={{ base: 8, md: 16 }}>
          {STACK.map((tech) => (
            <HStack
              key={tech.name}
              spacing={3}
              filter="grayscale(1)"
              opacity={0.6}
              transition="opacity 300ms ease"
              _hover={{ opacity: 1 }}
            >
              <Box as={asIcon(tech.icon)} size={22} />
              <Text fontWeight={700} fontSize="15px">{tech.name}</Text>
            </HStack>
          ))}
        </Flex>
      </Box>
    </Box>
  </Box>
);

export default Engine;
