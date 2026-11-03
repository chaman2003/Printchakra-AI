import React from 'react';
import { Box, Button, Flex, HStack, Tooltip } from '@chakra-ui/react';
import { motion } from 'framer-motion';
import { FiLayers, FiMic, FiWifiOff, FiCheckSquare, FiX, FiRepeat, FiArchive, FiTrash2 } from 'react-icons/fi';
import { Iconify } from '../common';
import { DashboardToolbar } from '../layout/DashboardRegions';

const MotionBox = motion.create(Box);

interface DashboardActionPanelProps {
  isChatVisible: boolean;
  selectionMode: boolean;
  selectedFilesCount: number;
  orchestrateMode: 'print' | 'scan' | null;
  showReopenOrchestrate: boolean;
  onTriggerPrint: () => void;
  onToggleChat: () => void;
  onToggleSelectionMode: () => void;
  onOpenConversionModal: () => void;
  onDeleteSelected: () => void;
  onCheckConnectivity: () => void;
  onToggleConvertedDrawer: () => void;
  onSelectAll: () => void;
  onDeselectAll: () => void;
  isConvertedDrawerOpen: boolean;
  onReopenOrchestrate: () => void;
  embedded?: boolean;
}

export const DashboardActionPanel: React.FC<DashboardActionPanelProps> = ({
  isChatVisible,
  selectionMode,
  selectedFilesCount,
  orchestrateMode,
  showReopenOrchestrate,
  onTriggerPrint,
  onToggleChat,
  onToggleSelectionMode,
  onOpenConversionModal,
  onDeleteSelected,
  onCheckConnectivity,
  onToggleConvertedDrawer,
  onSelectAll,
  onDeselectAll,
  isConvertedDrawerOpen,
  onReopenOrchestrate,
  embedded = false,
}) => {
  const Container = embedded ? Box : DashboardToolbar;
  const containerProps = embedded
    ? { w: '100%' }
    : {
        w: '100%',
        direction: { base: 'column', md: 'row' },
        flexWrap: 'wrap',
        gap: { base: 3, md: 4 },
        align: 'stretch',
      };

  const btnSize = embedded ? 'md' : 'md';

  return (
    <Container {...(containerProps as any)}>
      <Flex
        direction={{ base: 'column', md: 'row' }}
        flexWrap="wrap"
        gap={2}
        align="center"
        justify="flex-start"
        w="100%"
      >
        {/* Primary action */}
        <Button
          size={btnSize}
          colorScheme="brand"
          variant="solid"
          onClick={onTriggerPrint}
          leftIcon={<Iconify icon={FiLayers} boxSize={4} />}
          borderRadius="lg"
          fontWeight="700"
          fontSize="13px"
          h="38px"
          px={5}
          _hover={{ transform: 'translateY(-1px)', boxShadow: '0 6px 20px rgba(17, 17, 17,0.5)' }}
          transition="all 0.2s ease"
          w={{ base: '100%', md: 'auto' }}
        >
          Orchestrate
        </Button>

        {showReopenOrchestrate && orchestrateMode && (
          <Button
            size={btnSize}
            colorScheme="brand"
            variant="outline"
            onClick={onReopenOrchestrate}
            leftIcon={<Iconify icon={FiRepeat} boxSize={4} />}
            borderRadius="lg"
            fontWeight="600"
            fontSize="13px"
            h="38px"
            px={4}
            w={{ base: '100%', md: 'auto' }}
          >
            Re-open {orchestrateMode === 'print' ? 'Print' : 'Scan'}
          </Button>
        )}

        {/* AI Chat toggle */}
        <Button
          size={btnSize}
          colorScheme="purple"
          variant={isChatVisible ? 'solid' : 'outline'}
          onClick={onToggleChat}
          leftIcon={<Iconify icon={FiMic} boxSize={4} />}
          borderRadius="lg"
          fontWeight="600"
          fontSize="13px"
          h="38px"
          px={4}
          _hover={{ transform: 'translateY(-1px)' }}
          transition="all 0.2s ease"
          w={{ base: '100%', md: 'auto' }}
        >
          {isChatVisible ? 'Hide' : 'Show'} AI Chat
        </Button>

        {/* Selection mode */}
        <HStack spacing={2}>
          <Button
            size={btnSize}
            variant={selectionMode ? 'solid' : 'ghost'}
            colorScheme={selectionMode ? 'orange' : 'gray'}
            onClick={onToggleSelectionMode}
            leftIcon={<Iconify icon={selectionMode ? FiX : FiCheckSquare} boxSize={4} />}
            borderRadius="lg"
            fontWeight="600"
            fontSize="13px"
            h="38px"
            px={4}
            w={{ base: '100%', md: 'auto' }}
          >
            {selectionMode ? 'Cancel' : 'Select'}
          </Button>

          {selectionMode && (
            <>
              <Button
                size={btnSize}
                variant="outline"
                colorScheme="brand"
                onClick={onSelectAll}
                borderRadius="lg"
                fontWeight="600"
                fontSize="13px"
                h="38px"
                px={4}
              >
                Select All
              </Button>
              <Button
                size={btnSize}
                variant="ghost"
                colorScheme="gray"
                onClick={onDeselectAll}
                borderRadius="lg"
                fontWeight="600"
                fontSize="13px"
                h="38px"
                px={4}
              >
                Deselect
              </Button>
            </>
          )}
        </HStack>

        {selectionMode && selectedFilesCount > 0 && (
          <HStack spacing={2}>
            <Button
              size={btnSize}
              colorScheme="red"
              variant="solid"
              onClick={onDeleteSelected}
              leftIcon={<Iconify icon={FiTrash2} boxSize={4} />}
              borderRadius="lg"
              fontWeight="600"
              fontSize="13px"
              h="38px"
              px={4}
              w={{ base: '100%', md: 'auto' }}
            >
              Delete {selectedFilesCount} Selected
            </Button>
            <Button
              size={btnSize}
              colorScheme="brand"
              variant="solid"
              onClick={onOpenConversionModal}
              borderRadius="lg"
              fontWeight="600"
              fontSize="13px"
              h="38px"
              px={4}
              w={{ base: '100%', md: 'auto' }}
              boxShadow="0 4px 12px rgba(17, 17, 17,0.3)"
            >
              Convert {selectedFilesCount} Selected
            </Button>
          </HStack>
        )}

        <Button
          size={btnSize}
          variant="ghost"
          onClick={onToggleConvertedDrawer}
          leftIcon={<Iconify icon={FiArchive} boxSize={4} />}
          borderRadius="lg"
          fontWeight="600"
          fontSize="13px"
          h="38px"
          px={4}
          w={{ base: '100%', md: 'auto' }}
        >
          {isConvertedDrawerOpen ? 'Hide' : 'Show'} Converted
        </Button>
      </Flex>
    </Container>
  );
};
