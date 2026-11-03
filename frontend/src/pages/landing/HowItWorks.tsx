import React from 'react';
import { Box, Grid, Heading, Text } from '@chakra-ui/react';
import { Reveal } from '../../components/brand/Reveal';

const STEPS = [
  {
    number: '01',
    title: 'Open it on your phone',
    body: 'Sign in on your phone and open Phone capture. Allow the camera once.',
  },
  {
    number: '02',
    title: 'Take the photo',
    body: 'Frame the page and tap capture, or pick a photo you already have. Large photos are shrunk before they upload.',
  },
  {
    number: '03',
    title: 'Review and print',
    body: 'The cleaned-up page appears on your dashboard with its text. Print it, convert it or run text recognition again.',
  },
];

export const HowItWorks: React.FC = () => (
  <Box as="section" id="how" scrollMarginTop="80px" px={{ base: 5, md: 10 }} pb={{ base: 20, md: 32 }}>
    <Box maxW="1600px" mx="auto">
      <Reveal>
        <Heading as="h2" fontSize="clamp(40px, 5.6vw, 84px)" letterSpacing="-0.05em" lineHeight={0.92} mb={{ base: 10, md: 16 }}>
          Three steps, about thirty seconds
        </Heading>
      </Reveal>
      <Grid templateColumns={{ base: '1fr', md: 'repeat(3, 1fr)' }} gap="32px">
        {STEPS.map((step, index) => (
          <Reveal key={step.number} delay={index * 120}>
            <Box borderTop="1px solid #1e1e1e" pt={6}>
              <Text fontFamily="heading" fontWeight={700} fontSize={{ base: '72px', md: '104px' }} letterSpacing="-0.06em" lineHeight={0.85} color="#bfbfbf">
                {step.number}
              </Text>
              <Heading as="h3" fontSize="28px" letterSpacing="-0.04em" mt={8} mb={3}>
                {step.title}
              </Heading>
              <Text color="#838282" fontWeight={400} fontSize="17px" lineHeight={1.7}>
                {step.body}
              </Text>
            </Box>
          </Reveal>
        ))}
      </Grid>
    </Box>
  </Box>
);

export default HowItWorks;
