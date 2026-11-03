import React, { useState } from 'react';
import {
  Box,
  Button,
  Input,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  Stack,
  Text,
  HStack,
  useColorModeValue,
} from '@chakra-ui/react';
import Iconify from './Iconify';

interface FancySelectOption {
  value: string;
  label: string;
  icon?: React.ReactNode;
  description?: string;
}

interface FancySelectProps {
  options: FancySelectOption[];
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  allowCustom?: boolean;
  customValue?: string;
  onCustomChange?: (value: string) => void;
  label?: string;
  size?: 'sm' | 'md' | 'lg';
}

const FancySelect: React.FC<FancySelectProps> = ({
  options,
  value,
  onChange,
  placeholder = 'Select an option',
  allowCustom = false,
  customValue = '',
  onCustomChange,
  label,
  size = 'md',
}) => {
  const [isCustom, setIsCustom] = useState(false);
  const hoverBg = useColorModeValue('brand.50', 'whiteAlpha.100');
  const selectedBg = useColorModeValue('brand.100', 'rgba(17, 17, 17,0.2)');
  const borderColor = useColorModeValue('brand.200', 'rgba(17, 17, 17,0.3)');
  const menuBg = useColorModeValue('white', 'rgba(12, 16, 35, 0.95)');
  const buttonBg = useColorModeValue('white', 'rgba(20, 24, 45, 0.6)');

  const selectedOption = options.find(opt => opt.value === value);
  const displayValue = isCustom ? customValue : selectedOption?.label || placeholder;

  const handleCustomMode = () => {
    setIsCustom(true);
    onChange('custom');
  };

  return (
    <Box>
      {label && (
        <Text fontSize="sm" fontWeight="600" mb={2}>
          {label}
        </Text>
      )}
      <Menu>
        <MenuButton
          as={Button}
          rightIcon={<Iconify icon="solar:alt-arrow-down-bold" width={16} height={16} />}
          width="full"
          justifyContent="space-between"
          variant="outline"
          borderColor={borderColor}
          bg={buttonBg}
          _hover={{ bg: hoverBg }}
          _active={{ bg: selectedBg }}
          transition="all 0.2s"
          size={size}
        >
          <HStack spacing={2} flex={1} justify="space-between">
            <Text isTruncated>{displayValue}</Text>
          </HStack>
        </MenuButton>
        <MenuList
          bg={menuBg}
          border="1px solid"
          borderColor={borderColor}
          boxShadow="lg"
          zIndex={10}
        >
          {options.map(option => (
            <MenuItem
              key={option.value}
              onClick={() => {
                onChange(option.value);
                setIsCustom(false);
              }}
              bg={value === option.value && !isCustom ? selectedBg : 'transparent'}
              _hover={{ bg: hoverBg }}
              transition="all 0.15s"
            >
              <Stack spacing={1} width="full">
                <HStack spacing={2}>
                  {option.icon && <Box>{option.icon}</Box>}
                  <Text fontWeight={value === option.value && !isCustom ? '600' : '500'}>
                    {option.label}
                  </Text>
                </HStack>
                {option.description && (
                  <Text fontSize="xs" color="gray.500" ml={option.icon ? 6 : 0}>
                    {option.description}
                  </Text>
                )}
              </Stack>
            </MenuItem>
          ))}
          {allowCustom && (
            <>
              <Box height="1px" bg={borderColor} my={2} />
              <MenuItem
                onClick={handleCustomMode}
                bg={isCustom ? selectedBg : 'transparent'}
                _hover={{ bg: hoverBg }}
              >
                <Stack spacing={1} width="full">
                  <Text fontWeight={isCustom ? '600' : '500'}>✏️ Enter Custom Value</Text>
                  {isCustom && onCustomChange && (
                    <Input
                      placeholder="Enter custom value..."
                      value={customValue}
                      onChange={e => {
                        e.stopPropagation();
                        onCustomChange(e.target.value);
                      }}
                      onClick={e => e.stopPropagation()}
                      size="sm"
                      mt={2}
                    />
                  )}
                </Stack>
              </MenuItem>
            </>
          )}
        </MenuList>
      </Menu>
    </Box>
  );
};

export default FancySelect;
