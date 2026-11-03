import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import { Box, Flex, Heading } from '@chakra-ui/react';
import { LuArrowRight } from 'react-icons/lu';
import { Reveal } from '../../components/brand/Reveal';
import { PillButton } from '../../components/layout/SiteHeader';
import { useAuth } from '../../auth/AuthContext';

const Arrow = LuArrowRight as any;

/** "Ready to Upgrade your document game?" with the original "Get Started for Free" button. */
export const FinalCta: React.FC = () => {
  const { user } = useAuth();
  return (
    <Box as="section" px={{ base: 5, md: 10 }} pb={{ base: 24, md: 36 }}>
      <Flex direction="column" align="center" textAlign="center" maxW="1200px" mx="auto" pt={{ base: 4, md: 8 }}>
        <Reveal>
          <Heading as="h2" fontSize="clamp(44px, 8vw, 128px)" letterSpacing="-0.05em" lineHeight={0.92}>
            Ready to{' '}
            <Box as="em" fontFamily="serif" fontStyle="italic" fontWeight={400} letterSpacing="-0.02em" color="#838282">
              upgrade
            </Box>{' '}
            your document game?
          </Heading>
        </Reveal>
        <Reveal delay={150}>
          <Box mt={{ base: 12, md: 16 }}>
            <PillButton as={RouterLink} to={user ? '/dashboard' : '/signup'} h="56px" px={10} rightIcon={<Arrow size={18} />}>
              Get Started for Free
            </PillButton>
          </Box>
        </Reveal>
      </Flex>
    </Box>
  );
};

export default FinalCta;
