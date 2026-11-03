import React from 'react';
import { Box, Flex, Grid, GridItem, Heading, Text } from '@chakra-ui/react';
import { Reveal, REVEAL_EASE } from '../../components/brand/Reveal';

const publicUrl = (process.env.PUBLIC_URL || '').trim();
const base = !publicUrl || publicUrl === '.' ? '' : publicUrl.replace(/\/$/, '');
const img = (name: string) => `${base}/showcase/${name}`;

// design.md: images start at 20% grayscale and go to full colour at 1.05x scale on hover.
const imageStyles = {
  w: '100%',
  h: '100%',
  objectFit: 'cover' as const,
  filter: 'grayscale(20%)',
  transition: `filter 700ms ${REVEAL_EASE}, transform 700ms ${REVEAL_EASE}`,
};

interface FrameProps {
  src: string;
  alt: string;
  caption: string;
  title: string;
  radius: string;
  aspect?: string;
  height?: string;
  overlay?: string;
  objectPosition?: string;
}

const Frame: React.FC<FrameProps> = ({ src, alt, caption, title, radius, aspect, height, overlay, objectPosition = 'top center' }) => (
  <Box role="group">
    <Reveal kind="clip">
      <Box
        position="relative"
        overflow="hidden"
        borderRadius={radius}
        bg="#d9d9d9"
        border="1px solid rgba(30, 30, 30, 0.1)"
        aspectRatio={aspect}
        h={height}
        sx={{ isolation: 'isolate' }}
      >
        <Box
          as="img"
          src={src}
          alt={alt}
          loading="lazy"
          {...imageStyles}
          objectPosition={objectPosition}
          _groupHover={{ filter: 'grayscale(0%)', transform: 'scale(1.05)' }}
        />
        {overlay && (
          <Flex
            position="absolute"
            inset={0}
            align="center"
            justify="center"
            pointerEvents="none"
            opacity={0}
            transition="opacity 400ms ease"
            _groupHover={{ opacity: 1 }}
          >
            <Flex
              w="132px"
              h="132px"
              borderRadius="full"
              border="1px solid rgba(246, 246, 246, 0.8)"
              color="#f6f6f6"
              bg="rgba(17, 17, 17, 0.35)"
              backdropFilter="blur(6px)"
              align="center"
              justify="center"
              textAlign="center"
              px={4}
              fontSize="12px"
              fontWeight={700}
              letterSpacing="0.14em"
              textTransform="uppercase"
              lineHeight={1.4}
            >
              {overlay}
            </Flex>
          </Flex>
        )}
      </Box>
    </Reveal>
    <Box mt={4} px={1}>
      <Text fontSize="11px" fontWeight={700} letterSpacing="0.18em" textTransform="uppercase" color="#838282">
        {caption}
      </Text>
      <Text fontFamily="heading" fontWeight={700} fontSize="22px" letterSpacing="-0.03em" sx={{ wordSpacing: '0.14em' }} mt={1}>
        {title}
      </Text>
    </Box>
  </Box>
);

/** Asymmetrical 12-column grid: wide rectangle, tall pill, circle, wide rectangle. */
export const Showcase: React.FC = () => (
  <Box as="section" id="showcase" px={{ base: 5, md: 10 }} pb={{ base: 20, md: 32 }}>
    <Box maxW="1600px" mx="auto">
      <Flex justify="space-between" align="flex-end" mb={{ base: 10, md: 16 }} gap={6} direction={{ base: 'column', md: 'row' }}>
        <Reveal>
          <Heading as="h2" fontSize="clamp(40px, 5.6vw, 84px)" letterSpacing="-0.05em" lineHeight={0.92} maxW="760px">
            One workspace, from camera to printer
          </Heading>
        </Reveal>
        <Reveal delay={120}>
          <Text maxW="360px" color="#838282" fontWeight={400} fontSize="17px" lineHeight={1.7}>
            These are real screens from the app. Hover over any of them.
          </Text>
        </Reveal>
      </Flex>

      <Grid templateColumns="repeat(12, 1fr)" gap={{ base: 8, md: 8 }} alignItems="start">
        <GridItem colSpan={{ base: 12, md: 8 }}>
          <Frame
            src={img('dashboard.jpg')}
            alt="The PrintChakra dashboard listing scanned documents"
            caption="Dashboard"
            title="Every scan, sorted and searchable"
            radius="2px"
            aspect="16 / 10"
          />
        </GridItem>
        <GridItem colSpan={{ base: 12, md: 4 }} display="flex" justifyContent="center">
          <Box w="100%" maxW="340px">
            <Frame
              src={img('phone.jpg')}
              alt="The phone capture screen"
              caption="Phone capture"
              title="No app to install"
              radius="9999px"
              height="500px"
              overlay="Open on your phone"
              objectPosition="center top"
            />
          </Box>
        </GridItem>

        <GridItem colSpan={{ base: 12, md: 5 }} display="flex" justifyContent={{ base: 'center', md: 'stretch' }}>
          <Box w="100%" maxW={{ base: '420px', md: 'none' }}>
            <Frame
              src={img('document.jpg')}
              alt="A document after cleanup"
              caption="Cleanup"
              title="Flat, straight and readable"
              radius="9999px"
              aspect="1 / 1"
              objectPosition="center"
            />
          </Box>
        </GridItem>
        <GridItem colSpan={{ base: 12, md: 7 }}>
          <Frame
            src={img('ocr.jpg')}
            alt="Text recognition results beside the scanned page"
            caption="Text recognition"
            title="Select, search and copy every line"
            radius="2px"
            aspect="16 / 10"
          />
        </GridItem>
      </Grid>
    </Box>
  </Box>
);

export default Showcase;
