import { SystemStyleObject } from '@chakra-ui/react';

// Library of reusable style snippets described with natural, English-like keys
export const styleCommandLibrary: Record<string, SystemStyleObject> = {
  'rounded glass panel': {
    borderRadius: '24px',
    borderWidth: '1px',
    borderColor: { base: 'rgba(17, 17, 17, 0.08)', _dark: 'rgba(255,255,255,0.12)' },
    bg: { base: 'rgba(255, 248, 240, 0.95)', _dark: 'rgba(8, 15, 37, 0.95)' },
  },
  'soft glow border': {
    boxShadow: {
      base: '0 0 0 1px rgba(17, 17, 17,0.15), 0 15px 45px rgba(148,163,184,0.25)',
      _dark: '0 0 0 1px rgba(17, 17, 17,0.25), 0 15px 45px rgba(15,23,42,0.55)',
    },
  },
  'padded layout': {
    px: { base: 4, md: 6 },
    py: { base: 5, md: 7 },
  },
  'section spacing': {
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  },
  'stacked column layout': {
    display: 'flex',
    flexDirection: { base: 'column', xl: 'row' },
    gap: 6,
    alignItems: 'stretch',
  },
  'sidebar column': {
    flex: { base: 'auto', xl: '0 0 360px' },
  },
  'workspace column': {
    flex: 1,
    minW: 0,
  },
  'toolbar surface': {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 4,
    alignItems: 'center',
  },
  'float shadow': {
    boxShadow: {
      base: '0 25px 65px rgba(148,163,184,0.35)',
      _dark: '0 25px 65px rgba(15,23,42,0.55)',
    },
  },
  'subtle border': {
    borderWidth: '1px',
    borderColor: { base: 'rgba(0,0,0,0.1)', _dark: 'rgba(148,163,184,0.15)' },
    borderRadius: 'xl',
  },
  'scroll container': {
    overflowY: 'auto',
    pr: 2,
  },
  'soft gradient canvas': {
    bgGradient: {
      base: 'linear(180deg, #fff5eb 0%, #ffe8d6 45%, #ffd6ba 100%)',
      _dark: 'linear(180deg, rgba(2,6,23,1) 0%, rgba(4,12,33,0.92) 45%, rgba(7,17,41,0.9) 100%)',
    },
  },
  'contained width': {
    maxW: '1400px',
    mx: 'auto',
    w: '100%',
  },
  'card padding': {
    p: { base: 4, md: 6 },
  },
  'command chip': {
    borderRadius: 'full',
    px: 4,
    py: 2,
    fontWeight: 600,
    bg: { base: 'rgba(255,255,255,0.8)', _dark: 'rgba(15,23,42,0.55)' },
    color: { base: 'gray.700', _dark: 'white' },
  },
  'divider glow': {
    borderBottomWidth: '1px',
    borderColor: { base: 'rgba(0,0,0,0.08)', _dark: 'rgba(255,255,255,0.08)' },
  },
  'sticky sidebar': {
    position: 'sticky',
    top: { base: 2, md: 8 },
  },
  'command center panel': {
    bg: { base: 'rgba(255, 248, 240, 0.95)', _dark: 'rgba(15, 18, 38, 0.92)' },
    borderRadius: '28px',
    borderWidth: '1px',
    borderColor: { base: 'rgba(17, 17, 17,0.2)', _dark: 'rgba(17, 17, 17,0.35)' },
    boxShadow: {
      base: '0 30px 80px rgba(17, 17, 17, 0.08)',
      _dark: '0 30px 80px rgba(7,10,25,0.7)',
    },
  },
  'bleedless': {
    w: '100%',
  },
  'micro radius': {
    borderRadius: 'md',
  },
};

export const interpretCommandText = (commandText: string): string[] => {
  const normalized = commandText.toLowerCase();
  return Object.keys(styleCommandLibrary).filter(key => normalized.includes(key));
};

export const resolveCommandList = (commands: string[] = []): SystemStyleObject => {
  return commands.reduce<SystemStyleObject>((acc, command) => {
    const preset = styleCommandLibrary[command];
    if (!preset) {
      return acc;
    }
    return mergeStyles(acc, preset);
  }, {});
};

type StyleAccumulator = Record<string, any>;

export const mergeStyles = (
  base: SystemStyleObject = {},
  next: SystemStyleObject = {}
): SystemStyleObject => {
  const output: StyleAccumulator = { ...(base ?? {}) };

  Object.entries(next ?? {}).forEach(([key, value]) => {
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      const current = output[key];
      if (current && typeof current === 'object' && !Array.isArray(current)) {
        output[key] = mergeStyles(current as SystemStyleObject, value as SystemStyleObject);
      } else {
        output[key] = mergeStyles({}, value as SystemStyleObject);
      }
    } else {
      output[key] = value;
    }
  });

  return output as SystemStyleObject;
};
