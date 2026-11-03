import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import { Box, Flex, HStack, Image, Stack, Text, VStack } from '@chakra-ui/react';
import { motion } from 'framer-motion';
import { LuActivity, LuArrowRight, LuCheck, LuSmartphone, LuZap } from 'react-icons/lu';
import { EchoText } from '../../components/brand/EchoText';
import { Reveal } from '../../components/brand/Reveal';
import { PillButton } from '../../components/layout/SiteHeader';

const MotionBox = motion.create(Box as any);
const Arrow = LuArrowRight as any;
const Phone = LuSmartphone as any;
const Zap = LuZap as any;
const Check = LuCheck as any;
const Activity = LuActivity as any;

const logoSrc = (() => {
  const publicUrl = (process.env.PUBLIC_URL || '').trim();
  const base = !publicUrl || publicUrl === '.' ? '' : publicUrl.replace(/\/$/, '');
  return `${base}/logo.png`;
})();

const chip = {
  bg: '#ffffff',
  border: '1px solid rgba(30, 30, 30, 0.12)',
  borderRadius: 'sm',
  p: 4,
  position: 'absolute' as const,
  zIndex: 2,
};

/** Product frame: logo tile on a dark panel with two status chips and a label, as in the original page. */
const ProductFrame: React.FC = () => (
  <Reveal delay={200} w="100%" maxW="860px" mx="auto" mt={{ base: 16, md: 24 }} px={{ base: 3, md: 0 }}>
    <Box position="relative" p={{ base: 3, md: 4 }} border="1px solid rgba(30,30,30,0.12)" borderRadius="sm" bg="#ffffff">
      <Flex
        position="relative"
        overflow="hidden"
        bg="#111111"
        minH={{ base: '260px', md: '420px' }}
        align="center"
        justify="center"
        borderRadius="sm"
      >
        <Box position="absolute" inset={0} opacity={0.08} backgroundImage="linear-gradient(#f6f6f6 1px, transparent 1px), linear-gradient(90deg, #f6f6f6 1px, transparent 1px)" backgroundSize="40px 40px" />
        <MotionBox
          animate={{ y: [0, -12, 0] }}
          transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
          zIndex={1}
        >
          <Flex w={{ base: '120px', md: '168px' }} h={{ base: '120px', md: '168px' }} bg="#f2f2f2" border="1px solid #f6f6f6" borderRadius="sm" align="center" justify="center">
            <Image src={logoSrc} alt="PrintChakra" w="64%" h="64%" objectFit="contain" />
          </Flex>
        </MotionBox>
      </Flex>

      <MotionBox
        {...chip}
        top="14%"
        right={{ base: '-6px', md: '-28px' }}
        initial={{ x: 30, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ delay: 0.8 }}
      >
        <HStack spacing={3}>
          <Flex w={8} h={8} borderRadius="full" bg="#111111" color="#f2f2f2" align="center" justify="center">
            <Check size={16} />
          </Flex>
          <VStack align="start" spacing={0}>
            <Text fontSize="12px" fontWeight={700}>OCR Complete</Text>
            <Text fontSize="11px" color="#838282">89 words, 40 regions</Text>
          </VStack>
        </HStack>
      </MotionBox>

      <MotionBox
        {...chip}
        bottom="18%"
        left={{ base: '-6px', md: '-28px' }}
        initial={{ x: -30, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ delay: 1 }}
      >
        <HStack spacing={3}>
          <Flex w={8} h={8} borderRadius="full" border="1px solid #111111" align="center" justify="center">
            <Activity size={16} />
          </Flex>
          <VStack align="start" spacing={1.5}>
            <Text fontSize="12px" fontWeight={700}>Processing</Text>
            <Box w="100px" h="3px" bg="rgba(17,17,17,0.1)" overflow="hidden">
              <MotionBox h="100%" w="100%" bg="#111111" animate={{ x: ['-100%', '0%'] }} transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }} />
            </Box>
          </VStack>
        </HStack>
      </MotionBox>

      <MotionBox
        position="absolute"
        top="-14px"
        left="8%"
        bg="#111111"
        color="#f2f2f2"
        px={3}
        py={1}
        borderRadius="full"
        zIndex={2}
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 1.2 }}
      >
        <Text fontSize="10px" fontWeight={700} letterSpacing="0.14em">GEMINI VISION</Text>
      </MotionBox>
    </Box>
  </Reveal>
);

export const Hero: React.FC = () => (
  <Box as="section" id="overview" scrollMarginTop="80px" px={{ base: 5, md: 10 }} pt={{ base: 14, md: 20 }} pb={{ base: 16, md: 28 }}>
    <Flex direction="column" align="center" textAlign="center" maxW="1600px" mx="auto">
      <Reveal>
        <HStack
          spacing={2}
          px={4}
          py={2}
          border="1px solid rgba(30,30,30,0.15)"
          borderRadius="full"
          bg="#ffffff"
          mb={{ base: 12, md: 16 }}
        >
          <Zap size={14} />
          <Text fontSize="12px" fontWeight={700} letterSpacing="0.14em" textTransform="uppercase">
            V2.0 is now live in the cloud
          </Text>
        </HStack>
      </Reveal>

      {/* "The OS for" stays plain; the second line takes the echo stack. */}
      <Box pl="0.16em" pt="0.16em">
        <Text
          as="h1"
          fontFamily="heading"
          fontWeight={700}
          letterSpacing="-0.05em"
          lineHeight={0.9}
          fontSize="clamp(44px, 9.5vw, 150px)"
          sx={{ wordSpacing: '0.14em' }}
          srOnly
        >
          The OS for Hardware Intelligence
        </Text>
        <Text
          aria-hidden="true"
          fontFamily="heading"
          fontWeight={700}
          letterSpacing="-0.05em"
          lineHeight={0.9}
          fontSize="clamp(44px, 9.5vw, 150px)"
          sx={{ wordSpacing: '0.14em' }}
          mb="0.16em"
        >
          The OS for
        </Text>
        <EchoText text="Hardware" fontSize="clamp(44px, 9.5vw, 150px)" display="block" mb="0.2em" />
        <EchoText text="Intelligence" fontSize="clamp(44px, 9.5vw, 150px)" display="block" />
      </Box>

      <Reveal delay={150}>
        <Text mt={{ base: 12, md: 16 }} maxW="680px" mx="auto" fontSize={{ base: '18px', md: '22px' }} lineHeight={1.5}>
          Bridge the gap between physical documents and digital workflows. Scan, OCR, and Orchestrate from your phone and your
          browser, with nothing to install.
        </Text>
      </Reveal>

      <Reveal delay={250}>
        <Stack direction={{ base: 'column', sm: 'row' }} mt={10} spacing={4} justify="center">
          <PillButton as={RouterLink} to="/dashboard" rightIcon={<Arrow size={16} />}>
            Launch Dashboard
          </PillButton>
          <PillButton as={RouterLink} to="/phone" variant="outline" leftIcon={<Phone size={16} />}>
            Quick Capture
          </PillButton>
        </Stack>
      </Reveal>

      <ProductFrame />
    </Flex>
  </Box>
);

export default Hero;
