import React from 'react';
import { Box, BoxProps, useColorModeValue } from '@chakra-ui/react';
import { motion } from 'framer-motion';
import { StyledSection } from '../../ui/primitives';

const MotionBox = motion.create(Box);

interface PageShellProps extends BoxProps {
  commands?: string | string[];
}

const PageShell: React.FC<PageShellProps> = ({ children, commands, ...rest }) => {
  const background = useColorModeValue(
    'linear-gradient(135deg, #f2f2f2 0%, #f2f2f2 50%, #ececec 100%)',
    'linear-gradient(180deg, #080c18 0%, #0a0f1c 100%)'
  );

  return (
    <StyledSection
      componentKey="PageShell.Container"
      commands={commands}
      styleOverrides={{
        bg: background,
        minH: '100dvh',
        px: { base: 3, sm: 4, md: 6, lg: 8 },
        py: { base: 4, md: 6 },
        position: 'relative',
        overflow: 'hidden',
        w: '100%',
      }}
      {...rest}
    >
      <MotionBox
        w="100%"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
      >
        {children}
      </MotionBox>
    </StyledSection>
  );
};

export default PageShell;
