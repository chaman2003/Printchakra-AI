import React from 'react';
import { Box, Grid, Heading, Text } from '@chakra-ui/react';
import { Reveal } from '../../components/brand/Reveal';

const PILLARS = [
  {
    title: 'Capture',
    body: 'Point your phone at any page. PrintChakra finds the edges, straightens the perspective and lifts the shadows.',
  },
  {
    title: 'Read',
    body: 'Every line is read and boxed, so you can search the page, select the text and copy it out.',
  },
  {
    title: 'Print',
    body: 'Send the clean page to any printer your computer can see, through the same print dialog you already know.',
  },
];

/** Hairline into a large quote with one serif-italic word, then three short columns. */
export const Philosophy: React.FC = () => (
  <Box as="section" px={{ base: 5, md: 10 }} pt={{ base: 16, md: 24 }} pb={{ base: 20, md: 32 }}>
    <Box maxW="1200px" mx="auto" textAlign="center">
      <Box mx="auto" w="1px" h={{ base: '72px', md: '112px' }} bg="rgba(30, 30, 30, 0.1)" mb={{ base: 10, md: 14 }} />

      <Reveal>
        <Heading
          as="h2"
          fontSize="clamp(40px, 6.4vw, 92px)"
          letterSpacing="-0.05em"
          lineHeight={0.95}
          maxW="1000px"
          mx="auto"
        >
          Paper should be{' '}
          <Box as="em" fontFamily="serif" fontStyle="italic" fontWeight={400} letterSpacing="-0.02em" color="#838282">
            effortless
          </Box>
          .
        </Heading>
      </Reveal>

      <Grid templateColumns={{ base: '1fr', md: 'repeat(3, 1fr)' }} gap="32px" mt={{ base: 14, md: 24 }} textAlign="left">
        {PILLARS.map((pillar, index) => (
          <Reveal key={pillar.title} delay={index * 120}>
            <Heading as="h3" fontSize="30px" letterSpacing="-0.04em" mb={4}>
              {pillar.title}
            </Heading>
            <Text color="#838282" fontWeight={400} fontSize="17px" lineHeight={1.7}>
              {pillar.body}
            </Text>
          </Reveal>
        ))}
      </Grid>
    </Box>
  </Box>
);

export default Philosophy;
