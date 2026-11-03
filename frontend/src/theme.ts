import { extendTheme, ThemeConfig } from '@chakra-ui/react';
import { StyleFunctionProps } from '@chakra-ui/theme-tools';

/**
 * Swiss editorial design system (see design.md at the repo root).
 *
 * Off-white #f2f2f2 canvas, #111111 text, hairline borders, square corners, pill buttons.
 * The old brand scales (brand / nebula / cyber / neon, plus blue / purple / cyan / teal) all resolve to
 * neutral grays, so every existing screen that says colorScheme="brand" picks up the new look.
 * Status colors (red, green, orange, yellow) stay as they are because they carry meaning.
 */

export const palette = {
  bg: '#f2f2f2',
  ink: '#111111',
  line: 'rgba(30, 30, 30, 0.1)',
  lineStrong: 'rgba(30, 30, 30, 0.22)',
  muted: '#838282',
  soft: '#b6b5b5',
  card: '#ffffff',
  footer: '#1e1e1e',
  echo: ['#bfbfbf', '#c9c9c9', '#d1d1d1', '#d9d9d9'],
};

// Light only: the design system has no dark variant.
const config: ThemeConfig = {
  initialColorMode: 'light',
  useSystemColorMode: false,
  disableTransitionOnChange: true,
};

const fonts = {
  heading: '"Clash Display", "Satoshi", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  body: '"Satoshi", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  mono: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace',
  serif: '"Instrument Serif", Georgia, "Times New Roman", serif',
};

const neutral = {
  50: '#f6f6f6',
  100: '#ececec',
  200: '#d9d9d9',
  300: '#bfbfbf',
  400: '#838282',
  500: '#1e1e1e',
  600: '#111111',
  700: '#0b0b0b',
  800: '#050505',
  900: '#000000',
};

const colors = {
  brand: neutral,
  nebula: neutral,
  cyber: neutral,
  neon: neutral,
  // Remap the accent hues the screens use for emphasis. Status colors are left alone.
  blue: neutral,
  purple: neutral,
  cyan: neutral,
  teal: neutral,
  pink: neutral,
  gray: {
    50: '#f6f6f6',
    100: '#ececec',
    200: '#dcdcdc',
    300: '#bfbfbf',
    400: '#a3a2a2',
    500: '#838282',
    600: '#5f5e5e',
    700: '#3f3f3f',
    800: '#1e1e1e',
    900: '#111111',
  },
};

const radii = {
  none: '0',
  sm: '2px',
  base: '3px',
  md: '4px',
  lg: '6px',
  xl: '8px',
  '2xl': '10px',
  '3xl': '14px',
  full: '9999px',
  glass: '6px',
};

// Flat by design: borders do the work that shadows usually do.
const shadows = {
  outline: '0 0 0 2px rgba(17, 17, 17, 0.25)',
  halo: 'none',
  haloDark: 'none',
  subtle: '0 1px 0 rgba(30, 30, 30, 0.06)',
  subtleDark: '0 1px 0 rgba(30, 30, 30, 0.06)',
  neon: 'none',
  glow: 'none',
  sm: '0 1px 0 rgba(30, 30, 30, 0.08)',
  base: '0 1px 0 rgba(30, 30, 30, 0.08)',
  md: '0 6px 24px rgba(17, 17, 17, 0.08)',
  lg: '0 12px 40px rgba(17, 17, 17, 0.1)',
  xl: '0 18px 50px rgba(17, 17, 17, 0.12)',
  '2xl': '0 24px 60px rgba(17, 17, 17, 0.16)',
};

const semanticTokens = {
  colors: {
    'surface.bg': palette.bg,
    'surface.card': palette.card,
    'surface.blur': 'rgba(242, 242, 242, 0.9)',
    'surface.glass': 'rgba(255, 255, 255, 0.7)',
    'border.focus': palette.ink,
    'border.default': palette.lineStrong,
    'border.subtle': palette.line,
    'text.primary': palette.ink,
    'text.muted': palette.muted,
    'text.inverse': palette.bg,
    'accent.primary': palette.ink,
    'accent.secondary': palette.muted,
    'accent.tertiary': palette.soft,
  },
  radii: { glass: '6px' },
};

const isNeutralScheme = (scheme?: string) =>
  !scheme || ['brand', 'nebula', 'cyber', 'neon', 'blue', 'purple', 'cyan', 'teal', 'pink'].includes(scheme);

const components = {
  Heading: {
    baseStyle: {
      fontFamily: 'heading',
      fontWeight: 700,
      letterSpacing: '-0.03em',
      lineHeight: 1,
    },
  },
  Button: {
    baseStyle: {
      fontWeight: 700,
      borderRadius: 'full',
      letterSpacing: '0.02em',
      transition: 'background-color 160ms ease, color 160ms ease, border-color 160ms ease, transform 160ms ease',
      _focusVisible: { boxShadow: 'outline' },
    },
    variants: {
      // Solid black pill that inverts to an outlined pill on hover.
      solid: (props: StyleFunctionProps) =>
        isNeutralScheme(props.colorScheme)
          ? {
              bg: palette.ink,
              color: palette.bg,
              border: '1px solid',
              borderColor: palette.ink,
              _hover: { bg: 'transparent', color: palette.ink, _disabled: { bg: palette.ink, color: palette.bg } },
              _active: { bg: palette.line, transform: 'scale(0.98)' },
            }
          : {},
      outline: (props: StyleFunctionProps) =>
        isNeutralScheme(props.colorScheme)
          ? {
              border: '1px solid',
              borderColor: '#1e1e1e',
              color: palette.ink,
              bg: 'transparent',
              _hover: { bg: palette.ink, color: palette.bg },
              _active: { transform: 'scale(0.98)' },
            }
          : {},
      ghost: (props: StyleFunctionProps) =>
        isNeutralScheme(props.colorScheme)
          ? {
              color: palette.ink,
              _hover: { bg: 'rgba(17, 17, 17, 0.06)' },
              _active: { bg: 'rgba(17, 17, 17, 0.1)' },
            }
          : {},
      link: (props: StyleFunctionProps) =>
        isNeutralScheme(props.colorScheme) ? { color: palette.ink, _hover: { color: palette.muted, textDecoration: 'none' } } : {},
    },
    defaultProps: { colorScheme: 'brand' },
  },
  IconButton: {
    baseStyle: { borderRadius: 'full' },
  },
  Card: {
    baseStyle: {
      container: {
        bg: palette.card,
        borderRadius: 'md',
        boxShadow: 'none',
        border: '1px solid',
        borderColor: palette.line,
        transition: 'border-color 200ms ease, background-color 200ms ease',
        _hover: { borderColor: palette.lineStrong },
      },
    },
  },
  Input: {
    baseStyle: { field: { borderRadius: 'md', transition: 'border-color 160ms ease, background-color 160ms ease' } },
    variants: {
      filled: {
        field: {
          bg: palette.card,
          border: '1px solid',
          borderColor: palette.lineStrong,
          _hover: { bg: palette.card, borderColor: palette.ink },
          _focus: { bg: palette.card, borderColor: palette.ink, boxShadow: 'none' },
          _focusVisible: { borderColor: palette.ink, boxShadow: 'none' },
          _placeholder: { color: palette.soft },
        },
      },
    },
    defaultProps: { variant: 'filled' },
  },
  Textarea: {
    variants: {
      filled: {
        bg: palette.card,
        border: '1px solid',
        borderColor: palette.lineStrong,
        borderRadius: 'md',
        _hover: { bg: palette.card, borderColor: palette.ink },
        _focus: { bg: palette.card, borderColor: palette.ink, boxShadow: 'none' },
      },
    },
    defaultProps: { variant: 'filled' },
  },
  Select: {
    baseStyle: { field: { borderRadius: 'md' } },
    defaultProps: { variant: 'filled' },
  },
  Menu: {
    baseStyle: {
      list: {
        borderRadius: 'md',
        boxShadow: 'md',
        border: '1px solid',
        borderColor: palette.line,
        bg: palette.card,
        py: 1,
      },
      item: {
        borderRadius: 'sm',
        fontWeight: 500,
        _focus: { bg: 'rgba(17, 17, 17, 0.06)' },
        _hover: { bg: 'rgba(17, 17, 17, 0.06)' },
      },
    },
  },
  Modal: {
    baseStyle: {
      dialog: {
        bg: palette.card,
        borderRadius: 'md',
        border: '1px solid',
        borderColor: palette.line,
        boxShadow: '2xl',
      },
      overlay: { backdropFilter: 'blur(4px)', bg: 'rgba(17, 17, 17, 0.45)' },
      header: { fontFamily: 'heading', fontWeight: 700, letterSpacing: '-0.02em' },
    },
  },
  Drawer: {
    baseStyle: {
      dialog: { bg: palette.card, borderLeft: '1px solid', borderColor: palette.line },
      overlay: { backdropFilter: 'blur(4px)', bg: 'rgba(17, 17, 17, 0.45)' },
    },
  },
  Popover: {
    baseStyle: {
      content: { borderRadius: 'md', borderColor: palette.line, boxShadow: 'md' },
    },
  },
  Tooltip: {
    baseStyle: { bg: palette.ink, color: palette.bg, borderRadius: 'sm', fontWeight: 500, px: 3, py: 1.5 },
  },
  Progress: {
    baseStyle: {
      track: { bg: 'rgba(17, 17, 17, 0.08)', borderRadius: 'full' },
      filledTrack: { bg: palette.ink },
    },
  },
  Badge: {
    baseStyle: {
      borderRadius: 'full',
      px: 3,
      py: 0.5,
      fontWeight: 700,
      fontSize: '10px',
      letterSpacing: '0.08em',
      textTransform: 'uppercase',
    },
    variants: {
      subtle: (props: StyleFunctionProps) =>
        isNeutralScheme(props.colorScheme)
          ? { bg: 'rgba(17, 17, 17, 0.07)', color: palette.ink }
          : {},
    },
  },
  Tag: {
    baseStyle: { container: { borderRadius: 'full', fontWeight: 700 } },
  },
  Switch: {
    baseStyle: {
      track: { bg: '#d1d1d1', _checked: { bg: palette.ink }, _focusVisible: { boxShadow: 'outline' } },
    },
  },
  Checkbox: {
    baseStyle: {
      control: {
        borderRadius: 'sm',
        borderColor: palette.lineStrong,
        _checked: { bg: palette.ink, borderColor: palette.ink, _hover: { bg: palette.ink, borderColor: palette.ink } },
      },
    },
  },
  Tabs: {
    variants: {
      line: (props: StyleFunctionProps) =>
        isNeutralScheme(props.colorScheme)
          ? { tab: { fontWeight: 700, color: palette.muted, _selected: { color: palette.ink, borderColor: palette.ink } } }
          : {},
    },
  },
  Alert: {
    baseStyle: { container: { borderRadius: 'md' } },
  },
  Divider: {
    baseStyle: { borderColor: palette.line, opacity: 1 },
  },
  Link: {
    baseStyle: { _hover: { textDecoration: 'none' } },
  },
};

const styles = {
  global: {
    'html, body': {
      bg: palette.bg,
      color: palette.ink,
      fontFamily: 'body',
      fontWeight: 500,
      fontFeatureSettings: '"clig" off, "liga" off',
      minHeight: '100dvh',
      width: '100%',
      overflowX: 'hidden',
    },
    'h1, h2, h3, h4': { wordSpacing: '0.14em' },
    html: { scrollBehavior: 'smooth', fontSize: 'clamp(14px, 0.35vw + 12.5px, 17px)' },
    body: { position: 'relative', minWidth: '0', backgroundImage: 'none' },
    '#root': { minHeight: '100dvh', width: '100%', display: 'flex', flexDirection: 'column' },
    '*, *::before, *::after': { boxSizing: 'border-box' },
    '::selection': { background: palette.ink, color: palette.bg },
    '*::placeholder': { color: palette.soft },
    'img, picture, video, canvas, svg': { display: 'block', maxWidth: '100%', height: 'auto' },
    '::-webkit-scrollbar': { width: '10px', height: '10px' },
    '::-webkit-scrollbar-track': { background: 'transparent' },
    '::-webkit-scrollbar-thumb': { background: 'rgba(17, 17, 17, 0.22)', borderRadius: '10px', border: '2px solid #f2f2f2' },
    '::-webkit-scrollbar-thumb:hover': { background: 'rgba(17, 17, 17, 0.4)' },
    // Respect users who ask for less motion.
    '@media (prefers-reduced-motion: reduce)': {
      '*, *::before, *::after': { animationDuration: '0.01ms !important', transitionDuration: '0.01ms !important' },
    },
  },
};

const theme = extendTheme({
  config,
  fonts,
  colors,
  radii,
  semanticTokens,
  components,
  styles,
  shadows,
});

export default theme;
