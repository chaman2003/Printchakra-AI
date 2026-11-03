import React from 'react';
import { Box, HStack, Image, Text } from '@chakra-ui/react';
import { Link as RouterLink } from 'react-router-dom';

const logoSrc = (() => {
  const publicUrl = (process.env.PUBLIC_URL || '').trim();
  const base = !publicUrl || publicUrl === '.' ? '' : publicUrl.replace(/\/$/, '');
  return `${base}/logo.png`;
})();

interface LogoProps {
  to?: string;
  light?: boolean;
}

/** Wordmark with the mark in a hairline square. */
export const Logo: React.FC<LogoProps> = ({ to = '/', light = false }) => (
  <HStack as={RouterLink} to={to} spacing={3} aria-label="PrintChakra home" _hover={{ textDecoration: 'none' }}>
    <Box
      w="36px"
      h="36px"
      border="1px solid"
      borderColor={light ? 'rgba(246,246,246,0.4)' : '#1e1e1e'}
      borderRadius="sm"
      display="flex"
      alignItems="center"
      justifyContent="center"
      bg={light ? 'transparent' : '#ffffff'}
    >
      <Image src={logoSrc} alt="" w="22px" h="22px" objectFit="contain" />
    </Box>
    <Text
      fontFamily="heading"
      fontWeight={700}
      fontSize="22px"
      letterSpacing="-0.04em"
      lineHeight={1}
      color={light ? '#f6f6f6' : '#111111'}
    >
      printchakra
    </Text>
  </HStack>
);

export default Logo;
