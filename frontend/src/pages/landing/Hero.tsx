import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import { Box, Flex, Grid, HStack, Text, VStack } from '@chakra-ui/react';
import { LuArrowRight } from 'react-icons/lu';
import { EchoText } from '../../components/brand/EchoText';
import { Reveal } from '../../components/brand/Reveal';
import { PillButton } from '../../components/layout/SiteHeader';
import { useAuth } from '../../auth/AuthContext';

const Arrow = LuArrowRight as any;

const STATS = [
  { value: '12', label: 'cleanup steps on every scan' },
  { value: '~25s', label: 'from photo to searchable text' },
  { value: '0', label: 'apps to install' },
];

/** Pure typography: the echo-stacked wordmark, one line of promise, two buttons. No imagery. */
export const Hero: React.FC = () => {
  const { user } = useAuth();
  return (
    <Box as="section" id="product" scrollMarginTop="80px">
      <Flex
        direction="column"
        align="center"
        justify="center"
        textAlign="center"
        minH={{ base: '78vh', md: '86vh' }}
        px={{ base: 5, md: 10 }}
        pt={{ base: 16, md: 10 }}
        pb={{ base: 16, md: 20 }}
      >
        <Reveal>
          <Text
            fontSize="12px"
            fontWeight={700}
            letterSpacing="0.2em"
            textTransform="uppercase"
            color="#838282"
            border="1px solid rgba(30,30,30,0.15)"
            borderRadius="full"
            px={4}
            py={2}
            mb={{ base: 12, md: 16 }}
          >
            Document scanning, now in beta
          </Text>
        </Reveal>

        <Box pl="0.16em" pt="0.16em">
          <EchoText text="printchakra" fontSize="clamp(52px, 11vw, 180px)" />
        </Box>

        <Reveal delay={150}>
          <Text
            mt={{ base: 12, md: 16 }}
            maxW="640px"
            fontSize={{ base: '18px', md: '22px' }}
            lineHeight={1.5}
            color="#111111"
          >
            Turn your phone into a document scanner. Capture a page, clean it up, read the text and print it from one
            workspace.
          </Text>
        </Reveal>

        <Reveal delay={250}>
          <HStack mt={10} spacing={4} flexWrap="wrap" justify="center">
            <PillButton as={RouterLink} to={user ? '/dashboard' : '/signup'} rightIcon={<Arrow size={16} />}>
              {user ? 'Open the app' : 'Start free'}
            </PillButton>
            <PillButton as="a" href="#how" variant="outline">
              See how it works
            </PillButton>
          </HStack>
        </Reveal>
      </Flex>

      <Grid
        templateColumns={{ base: '1fr', md: 'repeat(3, 1fr)' }}
        borderTop="1px solid rgba(30,30,30,0.1)"
        borderBottom="1px solid rgba(30,30,30,0.1)"
        mx={{ base: 5, md: 10 }}
        maxW="1600px"
        marginInline={{ base: 5, md: 10, '2xl': 'auto' }}
      >
        {STATS.map((stat, index) => (
          <VStack
            key={stat.label}
            align="flex-start"
            spacing={2}
            py={{ base: 7, md: 10 }}
            px={{ base: 0, md: 8 }}
            borderLeft={{ base: 'none', md: index === 0 ? 'none' : '1px solid rgba(30,30,30,0.1)' }}
            borderTop={{ base: index === 0 ? 'none' : '1px solid rgba(30,30,30,0.1)', md: 'none' }}
          >
            <Text fontFamily="heading" fontWeight={700} fontSize={{ base: '48px', md: '64px' }} letterSpacing="-0.05em" lineHeight={0.9}>
              {stat.value}
            </Text>
            <Text color="#838282" fontSize="15px">
              {stat.label}
            </Text>
          </VStack>
        ))}
      </Grid>
    </Box>
  );
};

export default Hero;
