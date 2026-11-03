import React from 'react';
import { Box, Flex, Heading, Text, VStack } from '@chakra-ui/react';
import { Logo } from '../../components/brand/Logo';
import { EchoText } from '../../components/brand/EchoText';

interface AuthShellProps {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}

const POINTS = [
  'Scan any page with the camera you already carry',
  'Text is read and boxed on every line',
  'Print through your browser to any printer',
];

/** Two-panel layout: a dark editorial panel on the left (desktop only) and the form on the right. */
export const AuthShell: React.FC<AuthShellProps> = ({ title, subtitle, children }) => (
  <Flex minH="100dvh" bg="#f2f2f2">
    <Flex
      display={{ base: 'none', lg: 'flex' }}
      direction="column"
      justify="space-between"
      flex="1"
      bg="#111111"
      color="#f6f6f6"
      p={14}
      position="relative"
      overflow="hidden"
    >
      <Logo light />
      <Box>
        <EchoText
          text="welcome."
          fontSize="clamp(64px, 7.4vw, 128px)"
          color="#f6f6f6"
          echoes={['#3a3a3a', '#2f2f2f', '#262626', '#1d1d1d']}
        />
        <VStack align="flex-start" spacing={4} mt={14} color="rgba(246,246,246,0.6)" fontSize="17px">
          {POINTS.map((point) => (
            <Flex key={point} align="center" gap={4}>
              <Box w="24px" h="1px" bg="rgba(246,246,246,0.5)" />
              <Text>{point}</Text>
            </Flex>
          ))}
        </VStack>
      </Box>
      <Text fontSize="13px" color="rgba(246,246,246,0.4)">
        Free while in beta.
      </Text>
    </Flex>

    <Flex flex="1" align="center" justify="center" px={{ base: 5, md: 10 }} py={10} direction="column">
      <Box w="100%" maxW="440px">
        <Box display={{ base: 'block', lg: 'none' }} mb={12}>
          <Logo />
        </Box>
        <Heading as="h1" fontSize={{ base: '48px', md: '64px' }} letterSpacing="-0.05em" lineHeight={0.92} mb={4}>
          {title}
        </Heading>
        <Text color="#838282" fontSize="17px" mb={10}>
          {subtitle}
        </Text>
        {children}
      </Box>
    </Flex>
  </Flex>
);

export default AuthShell;
