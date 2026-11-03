import React from 'react';
import { Box, Flex, Heading, IconButton, Stack, Text, Divider, useColorModeValue, HStack, Badge, Image } from '@chakra-ui/react';
import { keyframes } from '@emotion/react';
import { FiRefreshCw } from 'react-icons/fi';
import SurfaceCard from '../layout/SurfaceCard';
import { Iconify } from '../common';
import DeviceAndConnectivityPanel from './DeviceAndConnectivityPanel';

const pulseGreen = keyframes`
  0%, 100% { box-shadow: 0 0 6px rgba(72, 187, 120, 0.5); }
  50% { box-shadow: 0 0 12px rgba(72, 187, 120, 0.9); }
`;

const pulseRed = keyframes`
  0%, 100% { box-shadow: 0 0 6px rgba(245, 101, 101, 0.5); }
  50% { box-shadow: 0 0 12px rgba(245, 101, 101, 0.9); }
`;

interface DashboardHeroCardProps {
  statusDotColor: string;
  statusTextColor: string;
  statusText: string;
  description?: string;
  error?: string | null;
  onRefresh: React.MouseEventHandler<HTMLButtonElement>;
  onCheckConnectivity?: () => void;
  children?: React.ReactNode;
}

const DEFAULT_DESCRIPTION =
  'Monitor document ingestion, inspect OCR output, and orchestrate conversions in real time.';

const logoSrc = (() => {
  const publicUrl = (process.env.PUBLIC_URL || '').trim();
  const base = !publicUrl || publicUrl === '.' ? '' : publicUrl.replace(/\/$/, '');
  return `${base}/logo.png`;
})();

export const DashboardHeroCard: React.FC<DashboardHeroCardProps> = ({
  statusDotColor,
  statusTextColor,
  statusText,
  description = DEFAULT_DESCRIPTION,
  error,
  onRefresh,
  onCheckConnectivity,
  children,
}) => {
  const dividerColor = useColorModeValue('rgba(0,0,0,0.06)', 'rgba(17, 17, 17, 0.08)');
  const isConnected = statusDotColor === 'green.400';
  const pulseAnimation = isConnected ? `${pulseGreen} 2s ease-in-out infinite` : `${pulseRed} 1.5s ease-in-out infinite`;

  return (
    <SurfaceCard overflow="hidden" position="relative">
      {/* Decorative gradient accent bar */}
      <Box
        position="absolute"
        top={0}
        left={0}
        right={0}
        height="2px"
        bgGradient={isConnected
          ? 'linear(to-r, brand.400, nebula.400, cyber.400)'
          : 'linear(to-r, orange.400, red.400)'
        }
      />

      <Flex 
        direction={{ base: 'column', lg: 'row' }} 
        justify="space-between" 
        align={{ base: 'stretch', lg: 'center' }} 
        gap={{ base: 3, md: 4 }}
        flexWrap="wrap"
        pt={1}
      >
        <Stack spacing={1.5} flex="1" minW="0">
          <HStack spacing={3}>
            <Flex
              w={12}
              h={12}
              borderRadius="xl"
              bgGradient="linear(to-br, brand.400, nebula.500)"
              boxShadow="0 4px 12px rgba(17, 17, 17,0.3)"
              align="center"
              justify="center"
              flexShrink={0}
            >
              <Image 
                src={logoSrc} 
                alt="Logo" 
                h={9} 
                w={9} 
                objectFit="contain" 
                filter="brightness(1.1) drop-shadow(0 2px 4px rgba(0,0,0,0.2))"
              />
            </Flex>
            <Heading 
              size={{ base: 'sm', md: 'md' }} 
              fontWeight="800"
              letterSpacing="-0.01em"
            >
              Dashboard
            </Heading>
            <Badge
              colorScheme={isConnected ? "green" : "red"}
              variant="subtle"
              fontSize="9px"
              borderRadius="md"
              px={2}
              py={0.5}
              fontWeight="700"
            >
              {isConnected ? 'Live' : 'Offline'}
            </Badge>
          </HStack>
          <Text color="text.muted" maxW="xl" fontSize={{ base: 'xs', md: 'sm' }} lineHeight="short">
            {description}
          </Text>
        </Stack>

        <Flex 
          direction={{ base: 'column', sm: 'row' }}
          gap={{ base: 2, md: 2 }} 
          align={{ base: 'stretch', sm: 'center' }}
          flexWrap="wrap"
          justify="flex-end"
        >
          <DeviceAndConnectivityPanel onCheckConnectivity={onCheckConnectivity} />
          <Flex
            align="center"
            gap={2}
            px={3}
            py={1.5}
            borderRadius="lg"
            bg="surface.blur"
            border="1px solid"
            borderColor={isConnected ? "rgba(72, 187, 120, 0.2)" : "rgba(245, 101, 101, 0.2)"}
            flexShrink={0}
            whiteSpace="nowrap"
            cursor={!isConnected ? "pointer" : "default"}
            onClick={!isConnected ? onCheckConnectivity : undefined}
            _hover={!isConnected ? { borderColor: "rgba(245, 101, 101, 0.4)" } : undefined}
            title={!isConnected ? "Click to check connectivity" : "Connected to backend"}
          >
            <Box
              w={2}
              h={2}
              borderRadius="full"
              bg={error ? 'orange.400' : statusDotColor}
              animation={pulseAnimation}
              flexShrink={0}
            />
            <Text fontWeight="600" color={statusTextColor} fontSize="xs" whiteSpace="nowrap">
              {statusText}
            </Text>
          </Flex>
          <IconButton
            aria-label="Refresh files"
            icon={<Iconify icon={FiRefreshCw} boxSize={4} />}
            onClick={onRefresh}
            variant="ghost"
            colorScheme="brand"
            size="sm"
            borderRadius="lg"
            _hover={{ transform: 'rotate(180deg)', transition: 'transform 0.3s' }}
            flexShrink={0}
          />
        </Flex>
      </Flex>
      
      {children && (
        <>
          <Divider my={{ base: 3, md: 4 }} borderColor={dividerColor} />
          <Box>
            {children}
          </Box>
        </>
      )}
    </SurfaceCard>
  );
};
