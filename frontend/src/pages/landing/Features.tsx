import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import { Box, Flex, Grid, Heading, Text } from '@chakra-ui/react';
import { LuArrowRight } from 'react-icons/lu';
import { Reveal } from '../../components/brand/Reveal';

const Arrow = LuArrowRight as any;

// Small geometric marks instead of pictograms: the design system avoids illustrative icons.
const shapes: Record<string, React.CSSProperties> = {
  circle: { width: 26, height: 26, borderRadius: '50%', background: '#111' },
  square: { width: 24, height: 24, background: '#111' },
  triangle: { width: 28, height: 26, background: '#111', clipPath: 'polygon(50% 0, 100% 100%, 0 100%)' },
  half: { width: 28, height: 14, background: '#111', borderRadius: '28px 28px 0 0' },
  ring: { width: 26, height: 26, borderRadius: '50%', border: '5px solid #111' },
  diamond: { width: 24, height: 24, background: '#111', transform: 'rotate(45deg)' },
};

const FEATURES = [
  {
    shape: 'square',
    title: 'Smart cleanup',
    body: 'Finds the page edges, fixes the perspective, lifts shadows and balances contrast in 12 automatic steps.',
  },
  {
    shape: 'circle',
    title: 'Text recognition',
    body: 'Reads every line and marks where it sits on the page, so the text can be searched, selected and copied.',
  },
  {
    shape: 'triangle',
    title: 'Voice control',
    body: 'Say "open phone capture" or "run OCR on my latest document" and the app does it.',
  },
  {
    shape: 'half',
    title: 'Your phone is the scanner',
    body: 'Open the site on your phone, take the photo, and it shows up on your computer within seconds.',
  },
  {
    shape: 'diamond',
    title: 'Convert and organise',
    body: 'Turn images into PDFs and back, rename files and keep them in folders.',
  },
  {
    shape: 'ring',
    title: 'Print from the browser',
    body: 'Use the print dialog you already know, with any printer your computer can reach.',
  },
];

/** Three-column cards: hairline border, white on hover, 64px mark that turns 12 degrees. */
export const Features: React.FC = () => (
  <Box as="section" id="features" scrollMarginTop="80px" px={{ base: 5, md: 10 }} pb={{ base: 20, md: 32 }}>
    <Box maxW="1600px" mx="auto">
      <Reveal>
        <Heading as="h2" fontSize="clamp(40px, 5.6vw, 84px)" letterSpacing="-0.05em" lineHeight={0.92} maxW="900px" mb={{ base: 10, md: 16 }}>
          Everything a paper trail needs
        </Heading>
      </Reveal>

      <Grid templateColumns={{ base: '1fr', md: 'repeat(2, 1fr)', lg: 'repeat(3, 1fr)' }} gap="32px">
        {FEATURES.map((feature, index) => (
          <Reveal key={feature.title} delay={(index % 3) * 100}>
            <Flex
              role="group"
              direction="column"
              justify="space-between"
              h="100%"
              minH="340px"
              p={{ base: 7, md: 9 }}
              border="1px solid rgba(30, 30, 30, 0.1)"
              borderRadius="sm"
              bg="transparent"
              transition="background-color 300ms ease, border-color 300ms ease"
              _hover={{ bg: '#ffffff', borderColor: 'rgba(30, 30, 30, 0.25)' }}
            >
              <Box>
                <Flex
                  w="64px"
                  h="64px"
                  align="center"
                  justify="center"
                  border="1px solid rgba(30, 30, 30, 0.15)"
                  borderRadius="sm"
                  mb={8}
                  transition="transform 400ms cubic-bezier(0.77, 0, 0.175, 1)"
                  _groupHover={{ transform: 'rotate(12deg)' }}
                >
                  <span aria-hidden="true" style={shapes[feature.shape]} />
                </Flex>
                <Heading as="h3" fontSize="28px" letterSpacing="-0.04em" mb={3}>
                  {feature.title}
                </Heading>
                <Text color="#838282" fontWeight={400} fontSize="16px" lineHeight={1.7}>
                  {feature.body}
                </Text>
              </Box>
              <Flex
                as={RouterLink}
                to="/signup"
                mt={8}
                align="center"
                gap={2}
                fontWeight={700}
                fontSize="13px"
                letterSpacing="0.12em"
                textTransform="uppercase"
                _hover={{ textDecoration: 'none' }}
              >
                Try it
                <Box as="span" transition="transform 200ms ease" _groupHover={{ transform: 'translateX(4px)' }}>
                  <Arrow size={16} />
                </Box>
              </Flex>
            </Flex>
          </Reveal>
        ))}
      </Grid>
    </Box>
  </Box>
);

export default Features;
