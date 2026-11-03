import React, { useState } from 'react';
import {
  Box,
  FormControl,
  FormErrorMessage,
  FormHelperText,
  FormLabel,
  IconButton,
  Input,
  InputGroup,
  InputProps,
  InputRightElement,
} from '@chakra-ui/react';
import { LuEye, LuEyeOff } from 'react-icons/lu';

const EyeIcon = LuEye as any;
const EyeOffIcon = LuEyeOff as any;

interface FieldProps extends Omit<InputProps, 'onChange'> {
  label: string;
  error?: string;
  helper?: string;
  value: string;
  onChange: (value: string) => void;
}

const labelStyle = { fontSize: '12px', fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase' as const, mb: 2 };

export const TextField: React.FC<FieldProps> = ({ label, error, helper, value, onChange, id, ...rest }) => (
  <FormControl isInvalid={Boolean(error)}>
    <FormLabel htmlFor={id} {...labelStyle}>
      {label}
    </FormLabel>
    <Input id={id} h="52px" value={value} onChange={(event) => onChange(event.target.value)} {...rest} />
    {error ? <FormErrorMessage>{error}</FormErrorMessage> : helper ? <FormHelperText>{helper}</FormHelperText> : null}
  </FormControl>
);

/** Password input with a show/hide button that is reachable by keyboard and labelled for screen readers. */
export const PasswordField: React.FC<FieldProps> = ({ label, error, helper, value, onChange, id, ...rest }) => {
  const [visible, setVisible] = useState(false);
  return (
    <FormControl isInvalid={Boolean(error)}>
      <FormLabel htmlFor={id} {...labelStyle}>
        {label}
      </FormLabel>
      <InputGroup>
        <Input
          id={id}
          h="52px"
          pr="52px"
          type={visible ? 'text' : 'password'}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          {...rest}
        />
        <InputRightElement h="52px" w="52px">
          <IconButton
            aria-label={visible ? 'Hide password' : 'Show password'}
            icon={visible ? <EyeOffIcon size={18} /> : <EyeIcon size={18} />}
            variant="ghost"
            size="sm"
            onClick={() => setVisible((v) => !v)}
          />
        </InputRightElement>
      </InputGroup>
      {error ? <FormErrorMessage>{error}</FormErrorMessage> : helper ? <FormHelperText>{helper}</FormHelperText> : null}
    </FormControl>
  );
};

/** Hairline with a centred word, used between the main action and the alternative. */
export const OrDivider: React.FC<{ label?: string }> = ({ label = 'or' }) => (
  <Box position="relative" my={6} textAlign="center">
    <Box position="absolute" top="50%" left={0} right={0} h="1px" bg="rgba(30,30,30,0.12)" />
    <Box
      as="span"
      position="relative"
      bg="#f2f2f2"
      px={4}
      fontSize="12px"
      fontWeight={700}
      letterSpacing="0.16em"
      textTransform="uppercase"
      color="#838282"
    >
      {label}
    </Box>
  </Box>
);
