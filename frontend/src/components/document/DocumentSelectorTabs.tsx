import React from 'react';
import {
  Badge,
  Box,
  Button,
  Checkbox,
  Flex,
  HStack,
  Input,
  SimpleGrid,
  Spinner,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
  Text,
  VStack,
} from '@chakra-ui/react';
import { FiUpload } from 'react-icons/fi';
import Iconify from '../common/Iconify';
import { Document } from '../../types';

interface DocumentSelectorTabsProps {
  activeTab: number;
  setActiveTab: (index: number) => void;
  currentDocuments: Document[];
  convertedDocuments: Document[];
  uploadedFiles: Document[];
  selectedDocs: Set<string>;
  borderColor: string;
  selectedBg: string;
  isDragging: boolean;
  setIsDragging: (dragging: boolean) => void;
  fileInputRef: React.RefObject<HTMLInputElement | null>;
  allowMultiple: boolean;
  mode: 'scan' | 'print';
  isUploading: boolean;
  uploadProgress: string;
  handleFileUpload: (files: FileList | null) => void;
  handleDocumentClick: (filename: string, index: number, shiftKey: boolean, docs: Document[]) => void;
  onSelectAll: () => void;
  onDeselectAll: () => void;
}

export default function DocumentSelectorTabs({
  activeTab,
  setActiveTab,
  currentDocuments,
  convertedDocuments,
  uploadedFiles,
  selectedDocs,
  borderColor,
  selectedBg,
  isDragging,
  setIsDragging,
  fileInputRef,
  allowMultiple,
  mode,
  isUploading,
  uploadProgress,
  handleFileUpload,
  handleDocumentClick,
  onSelectAll,
  onDeselectAll,
}: DocumentSelectorTabsProps) {
  const renderDocumentCard = (doc: Document, index: number, allDocs: Document[]) => {
    const isSelected = selectedDocs.has(doc.filename);
    const documentIndex = index + 1;
    const pageCount = doc.pages?.length || 1;

    return (
      <Box
        key={doc.filename}
        position="relative"
        borderRadius="xl"
        border="2px solid"
        borderColor={isSelected ? 'brand.400' : borderColor}
        bg={isSelected ? selectedBg : 'white'}
        overflow="hidden"
        cursor="pointer"
        transition="all 0.2s ease"
        _hover={{
          transform: 'translateY(-4px)',
          boxShadow: 'xl',
          borderColor: 'brand.400',
        }}
        onClick={e => handleDocumentClick(doc.filename, index, e.shiftKey, allDocs)}
        _dark={{
          bg: isSelected ? selectedBg : 'gray.800',
        }}
      >
        <Badge
          position="absolute"
          top={3}
          left={3}
          colorScheme="purple"
          borderRadius="full"
          px={3}
          py={1}
          zIndex={2}
        >
          #{documentIndex}
        </Badge>

        {pageCount > 1 && (
          <Badge
            position="absolute"
            top={3}
            right={isSelected ? 10 : 3}
            colorScheme="cyan"
            borderRadius="full"
            px={2}
            py={1}
            fontSize="xs"
            zIndex={2}
          >
            {pageCount} pages
          </Badge>
        )}

        {isSelected && (
          <Box
            position="absolute"
            top={2}
            right={2}
            bg="brand.500"
            color="white"
            borderRadius="full"
            p={1}
            zIndex={2}
          >
            <Iconify icon="solar:check-circle-bold" width={16} height={16} />
          </Box>
        )}

        <Box position="relative" h={{ base: '120px', md: '150px' }} bg="gray.100" overflow="hidden" _dark={{ bg: 'gray.700' }}>
          {doc.thumbnailUrl ? (
            <img
              src={doc.thumbnailUrl}
              alt={doc.filename}
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'contain',
                display: 'block',
                backgroundColor: '#f5f5f5',
              }}
            />
          ) : (
            <Flex align="center" justify="center" h="100%" bg="gray.200" _dark={{ bg: 'gray.700' }}>
              <Iconify
                icon="solar:document-bold"
                boxSize={10}
                color="gray.400"
                _dark={{ color: 'whiteAlpha.400' }}
              />
            </Flex>
          )}
        </Box>

        <VStack align="start" spacing={1} p={3}>
          <Text fontSize="sm" fontWeight="600" noOfLines={1} w="100%">
            {doc.filename}
          </Text>
          <HStack justify="space-between" w="100%">
            <Text fontSize="xs" color="text.muted">
              {doc.size ? (doc.size / 1024).toFixed(1) : '0'} KB
            </Text>
          </HStack>
        </VStack>

        {isSelected && (
          <Box
            position="absolute"
            top={0}
            left={0}
            right={0}
            bottom={0}
            bg="brand.500"
            opacity={0.1}
            pointerEvents="none"
          />
        )}
      </Box>
    );
  };

  return (
    <Tabs
      colorScheme="brand"
      variant="enclosed"
      index={activeTab}
      onChange={index => setActiveTab(index)}
    >
      <TabList>
        <Tab>
          <HStack spacing={2}>
            <Iconify icon="solar:documents-bold" width={18} height={18} />
            <Text>Current Documents</Text>
            <Badge colorScheme="purple">{currentDocuments.length}</Badge>
          </HStack>
        </Tab>
        <Tab>
          <HStack spacing={2}>
            <Iconify icon="solar:file-check-bold" width={18} height={18} />
            <Text>Converted Documents</Text>
            <Badge colorScheme="green">{convertedDocuments.length}</Badge>
          </HStack>
        </Tab>
        <Tab>
          <HStack spacing={2}>
            <Iconify icon={FiUpload} width={18} height={18} />
            <Text>Upload Local Files</Text>
            {uploadedFiles.length > 0 && (
              <Badge colorScheme="blue">{uploadedFiles.length}</Badge>
            )}
          </HStack>
        </Tab>
      </TabList>

      <TabPanels>
        <TabPanel>
          <Flex justify="space-between" align="center" mt={2} mb={4}>
            <Text fontSize="sm" color="text.muted">
              {currentDocuments.length} total documents
            </Text>
            <HStack>
              <Button size="xs" variant="outline" onClick={onSelectAll} leftIcon={<Iconify icon="solar:check-square-bold" />}>
                Select All
              </Button>
              <Button size="xs" variant="ghost" onClick={onDeselectAll} leftIcon={<Iconify icon="solar:close-square-bold" />}>
                Deselect
              </Button>
            </HStack>
          </Flex>
          {currentDocuments.length === 0 ? (
            <Flex direction="column" align="center" justify="center" minH="300px" color="text.muted">
              <Iconify icon="solar:document-bold" width={48} height={48} />
              <Text mt={4} fontSize="lg">
                No documents in current session
              </Text>
            </Flex>
          ) : (
            <SimpleGrid columns={{ base: 2, sm: 3, md: 4 }} spacing={4} mt={4}>
              {currentDocuments.map((doc, index) =>
                renderDocumentCard(doc, index, currentDocuments)
              )}
            </SimpleGrid>
          )}
        </TabPanel>

        <TabPanel>
          <Flex justify="space-between" align="center" mt={2} mb={4}>
            <Text fontSize="sm" color="text.muted">
              {convertedDocuments.length} total converted
            </Text>
            <HStack>
              <Button size="xs" variant="outline" onClick={onSelectAll} leftIcon={<Iconify icon="solar:check-square-bold" />}>
                Select All
              </Button>
              <Button size="xs" variant="ghost" onClick={onDeselectAll} leftIcon={<Iconify icon="solar:close-square-bold" />}>
                Deselect
              </Button>
            </HStack>
          </Flex>
          {convertedDocuments.length === 0 ? (
            <Flex direction="column" align="center" justify="center" minH="300px" color="text.muted">
              <Iconify icon="solar:file-check-bold" width={48} height={48} />
              <Text mt={4} fontSize="lg">
                No converted documents available
              </Text>
            </Flex>
          ) : (
            <SimpleGrid columns={{ base: 2, sm: 3, md: 4 }} spacing={4} mt={4}>
              {convertedDocuments.map((doc, index) =>
                renderDocumentCard(doc, index, convertedDocuments)
              )}
            </SimpleGrid>
          )}
        </TabPanel>

        <TabPanel>
          <VStack spacing={4}>
            <Box
              as="div"
              w="100%"
              minH="200px"
              border="2px dashed"
              borderColor={isDragging ? 'brand.400' : borderColor}
              borderRadius="xl"
              bg={isDragging ? 'rgba(17, 17, 17,0.1)' : 'transparent'}
              transition="all 0.3s"
              cursor="pointer"
              pointerEvents="auto"
              onDrop={(e: any) => {
                e.preventDefault();
                e.stopPropagation();
                setIsDragging(false);
                handleFileUpload(e.dataTransfer.files);
              }}
              onDragOver={(e: any) => {
                e.preventDefault();
                e.stopPropagation();
                setIsDragging(true);
              }}
              onDragEnter={(e: any) => {
                e.preventDefault();
                e.stopPropagation();
                setIsDragging(true);
              }}
              onDragLeave={(e: any) => {
                e.preventDefault();
                e.stopPropagation();
                setIsDragging(false);
              }}
              onClick={() => fileInputRef.current?.click()}
            >
              <Flex direction="column" align="center" justify="center" h="100%" p={6}>
                <Iconify
                  icon={FiUpload}
                  boxSize={16}
                  color={isDragging ? 'brand.400' : 'text.muted'}
                />
                <Text
                  mt={4}
                  fontSize="lg"
                  fontWeight="600"
                  color={isDragging ? 'brand.400' : 'text.primary'}
                >
                  {isDragging ? 'Drop files here' : 'Drag & drop files here'}
                </Text>
                <Text fontSize="sm" color="text.muted" mt={2}>
                  or click to browse
                </Text>
                <Text fontSize="xs" color="text.muted" mt={2}>
                  Supported: {mode === 'print' ? 'PDF, JPG, PNG' : 'JPG, PNG'}
                </Text>
              </Flex>
            </Box>

            <Input
              ref={fileInputRef}
              type="file"
              multiple={allowMultiple}
              accept={mode === 'print' ? '.pdf,.jpg,.jpeg,.png' : '.jpg,.jpeg,.png'}
              display="none"
              onChange={e => handleFileUpload(e.target.files)}
              disabled={isUploading}
            />

            {isUploading && (
              <Flex align="center" justify="center" p={4} bg="rgba(17, 17, 17,0.1)" borderRadius="lg">
                <Spinner size="md" color="brand.400" mr={3} />
                <Text color="brand.300">{uploadProgress || 'Processing files...'}</Text>
              </Flex>
            )}

            {uploadedFiles.length > 0 && (
              <SimpleGrid columns={{ base: 2, sm: 3, md: 4 }} spacing={4} w="100%">
                {uploadedFiles.map((doc, index) =>
                  renderDocumentCard(doc, index, uploadedFiles)
                )}
              </SimpleGrid>
            )}
          </VStack>
        </TabPanel>
      </TabPanels>
    </Tabs>
  );
}
