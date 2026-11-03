import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import { Box, Flex, HStack } from '@chakra-ui/react';
import { LuArrowRight } from 'react-icons/lu';
import { EchoText } from '../../components/brand/EchoText';
import { Reveal } from '../../components/brand/Reveal';
import { PillButton } from '../../components/layout/SiteHeader';
import { useAuth } from '../../auth/AuthContext';

const Arrow = LuArrowRight as any;

export const FinalCta: React.FC = () => {
  const { user } = useAuth();
  return (
    <Box as="section" px={{ base: 5, md: 10 }} pb={{ base: 24, md: 36 }}>
      <Flex direction="column" align="center" textAlign="center" borderTop="1px solid rgba(30,30,30,0.1)" pt={{ base: 16, md: 28 }}>
        <Box pl="0.16em" pt="0.16em">
          <EchoText text="scan it." fontSize="clamp(56px, 12vw, 190px)" />
        </Box>
        <Reveal delay={150}>
          <HStack mt={{ base: 12, md: 16 }} spacing={4} flexWrap="wrap" justify="center">
            <PillButton as={RouterLink} to={user ? '/dashboard' : '/signup'} rightIcon={<Arrow size={16} />}>
              {user ? 'Open the app' : 'Create a free account'}
            </PillButton>
            {!user && (
              <PillButton as={RouterLink} to="/login" variant="outline">
                Log in
              </PillButton>
            )}
          </HStack>
        </Reveal>
      </Flex>
    </Box>
  );
};

export default FinalCta;
