import React from 'react';
import { Box, BoxProps } from '@chakra-ui/react';
import { palette } from '../../theme';

interface EchoTextProps extends Omit<BoxProps, 'children'> {
  text: string;
  /** Distance between echoes, in em. design.md uses 0.04em steps. */
  step?: number;
  /** Colour of the foreground word. */
  color?: string;
  /** Echo colours, nearest layer first. Defaults to design.md's #bfbfbf to #d9d9d9. */
  echoes?: string[];
}

/**
 * Typographic echo stack: the word is repeated behind itself, each copy shifted up and left by one more
 * step and a little lighter. The copies are decoration only, so they are hidden from assistive tech and
 * ignore the pointer.
 */
export const EchoText: React.FC<EchoTextProps> = ({
  text,
  step = 0.04,
  color = palette.ink,
  echoes = palette.echo,
  ...rest
}) => (
  <Box
    position="relative"
    display="inline-block"
    isolation="isolate"
    fontFamily="heading"
    fontWeight={700}
    letterSpacing="-0.05em"
    lineHeight={0.9}
    sx={{ wordSpacing: '0.14em' }}
    whiteSpace="nowrap"
    {...rest}
  >
    <Box as="span" srOnly>
      {text}
    </Box>
    {echoes.map((echo, index) => (
      <Box
        key={index}
        as="span"
        aria-hidden="true"
        position="absolute"
        top={`-${(index + 1) * step}em`}
        left={`-${(index + 1) * step}em`}
        color={echo}
        pointerEvents="none"
        userSelect="none"
        zIndex={-(index + 1)}
      >
        {text}
      </Box>
    ))}
    <Box as="span" aria-hidden="true" position="relative" color={color}>
      {text}
    </Box>
  </Box>
);

export default EchoText;
