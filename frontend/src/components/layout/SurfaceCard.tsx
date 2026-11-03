import React from 'react';
import { BoxProps, useColorModeValue } from '@chakra-ui/react';
import { StyledSection } from '../../ui/primitives';

interface SurfaceCardProps extends BoxProps {
  commands?: string | string[];
}

const SurfaceCard = React.forwardRef<HTMLDivElement, SurfaceCardProps>((props, ref) => {
  const { children, commands, ...rest } = props;
  const background = useColorModeValue(
    'rgba(255, 255, 255, 0.92)',
    'linear-gradient(145deg, rgba(14, 18, 32, 0.96) 0%, rgba(11, 15, 28, 0.98) 100%)'
  );
  const borderColor = useColorModeValue(
    'rgba(0, 0, 0, 0.06)',
    'rgba(255, 255, 255, 0.06)'
  );
  const shadow = useColorModeValue(
    '0 1px 3px rgba(0, 0, 0, 0.04), 0 6px 24px rgba(17, 17, 17, 0.06)',
    '0 1px 3px rgba(0, 0, 0, 0.3), 0 6px 24px rgba(0, 0, 0, 0.2)'
  );

  return (
    <StyledSection
      ref={ref}
      componentKey="SurfaceCard.Root"
      commands={commands}
      styleOverrides={{
        bg: background,
        borderColor,
        boxShadow: shadow,
        borderWidth: '1px',
        borderRadius: 'xl',
        p: { base: 4, md: 5 },
        position: 'relative',
        transition: 'box-shadow 0.2s ease, border-color 0.2s ease',
      }}
      {...rest}
    >
      {children}
    </StyledSection>
  );
});

SurfaceCard.displayName = 'SurfaceCard';

export default SurfaceCard;
