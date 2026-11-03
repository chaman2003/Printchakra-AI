import React from 'react';
import { Box, Grid, Text, VStack } from '@chakra-ui/react';
import { Reveal } from '../../components/brand/Reveal';

// Same four-card strip as the original page; the values describe the cloud version.
const SPECS = [
  { label: 'Sync', value: 'Live', sub: 'No refresh needed' },
  { label: 'Cleanup', value: '12 steps', sub: 'On every scan' },
  { label: 'Power', value: 'Gemini', sub: 'Vision OCR' },
  { label: 'Access', value: 'Login', sub: 'Required to use the app' },
];

export const Stats: React.FC = () => (
  <Box as="section" px={{ base: 5, md: 10 }} pb={{ base: 16, md: 28 }}>
    <Grid
      maxW="1600px"
      mx="auto"
      templateColumns={{ base: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' }}
      borderTop="1px solid rgba(30,30,30,0.12)"
      borderBottom="1px solid rgba(30,30,30,0.12)"
    >
      {SPECS.map((spec, index) => (
        <Reveal key={spec.label} delay={index * 100}>
          <VStack
            align="flex-start"
            spacing={2}
            py={{ base: 7, md: 12 }}
            px={{ base: 4, md: 8 }}
            h="100%"
            borderLeft={{ base: index % 2 === 0 ? 'none' : '1px solid rgba(30,30,30,0.12)', md: index === 0 ? 'none' : '1px solid rgba(30,30,30,0.12)' }}
            borderTop={{ base: index > 1 ? '1px solid rgba(30,30,30,0.12)' : 'none', md: 'none' }}
          >
            <Text fontFamily="heading" fontWeight={700} fontSize={{ base: '36px', md: '56px' }} letterSpacing="-0.05em" lineHeight={0.9}>
              {spec.value}
            </Text>
            <Text fontSize="12px" fontWeight={700} letterSpacing="0.16em" textTransform="uppercase" pt={2}>
              {spec.label}
            </Text>
            <Text fontSize="14px" color="#838282">
              {spec.sub}
            </Text>
          </VStack>
        </Reveal>
      ))}
    </Grid>
  </Box>
);

export default Stats;
