import React, { useState, useEffect, useCallback } from 'react';
import {
  Box, VStack, HStack, Heading, Text, Button, SimpleGrid, Card, CardBody,
  Select, Input, Checkbox, useToast, Badge, Spinner, Divider, Tabs, TabList,
  TabPanels, Tab, TabPanel, Progress, IconButton, Tooltip, useColorModeValue,
  Modal, ModalOverlay, ModalContent, ModalHeader, ModalBody, ModalCloseButton,
  useDisclosure,
} from '@chakra-ui/react';
import { motion } from 'framer-motion';
import { FiFileText, FiImage, FiRefreshCw, FiDownload, FiTrash2, FiZoomIn, FiLayers, FiCpu } from 'react-icons/fi';
import apiClient from '../apiClient';
import { API_BASE_URL, API_ENDPOINTS, getImageUrl } from '../config';
import { FileInfo, OCRResult } from '../types';
import { runOCR } from '../ocrApi';
import { withToken } from '../auth/session';
import PageShell from '../components/layout/PageShell';
import { Iconify } from '../components/common';

const MotionCard = motion.create(Card);

const Playground: React.FC = () => {
  const toast = useToast();

  // File state
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [loading, setLoading] = useState(false);

  // Selection state
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);

  // Conversion state
  const [targetFormat, setTargetFormat] = useState('pdf');
  const [mergePdf, setMergePdf] = useState(false);
  const [customFilename, setCustomFilename] = useState('');
  const [converting, setConverting] = useState(false);

  // Converted files
  const [convertedFiles, setConvertedFiles] = useState<string[]>([]);

  // OCR state
  const [ocrLoading, setOcrLoading] = useState<Record<string, boolean>>({});
  const [ocrResults, setOcrResults] = useState<Record<string, OCRResult>>(() => {
    try {
      const saved = localStorage.getItem('printchakra_ocr_results');
      return saved ? JSON.parse(saved) : {};
    } catch { return {}; }
  });

  // Preview
  const previewModal = useDisclosure();
  const [previewFile, setPreviewFile] = useState('');

  const bgCard = useColorModeValue('white', 'gray.800');
  const bgSubtle = useColorModeValue('gray.50', 'gray.900');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const selectedBg = useColorModeValue('purple.50', 'purple.900');

  // Load files
  const loadFiles = useCallback(async () => {
    setLoading(true);
    try {
      const res = await apiClient.get(API_ENDPOINTS.files);
      if (res.data?.success) {
        setFiles(res.data.files || []);
      }
    } catch (e) {
      console.error('Load files error:', e);
    } finally {
      setLoading(false);
    }
  }, []);

  const loadConvertedFiles = useCallback(async () => {
    try {
      const res = await apiClient.get(API_ENDPOINTS.getConvertedFiles);
      if (res.data?.success) {
        setConvertedFiles(res.data.files?.map((f: any) => f.filename || f) || []);
      }
    } catch (e) {
      console.error('Load converted error:', e);
    }
  }, []);

  useEffect(() => {
    loadFiles();
    loadConvertedFiles();
  }, [loadFiles, loadConvertedFiles]);

  // Save OCR results to localStorage
  useEffect(() => {
    try {
      localStorage.setItem('printchakra_ocr_results', JSON.stringify(ocrResults));
    } catch { /* ignore */ }
  }, [ocrResults]);

  // Toggle file selection
  const toggleFile = (filename: string) => {
    setSelectedFiles(prev =>
      prev.includes(filename) ? prev.filter(f => f !== filename) : [...prev, filename]
    );
  };

  // Convert
  const handleConvert = async () => {
    if (selectedFiles.length === 0) return;
    setConverting(true);
    try {
      let successCount = 0;
      let failCount = 0;

      if (targetFormat === 'pdf' && mergePdf && selectedFiles.length > 1) {
        const outputName = customFilename.trim() || 'combined.pdf';
        const res = await apiClient.post('/document/convert/images-to-pdf', {
          images: selectedFiles,
          output_name: outputName,
        });
        if (res.data?.success) {
          successCount = selectedFiles.length;
        } else {
          failCount = selectedFiles.length;
        }
      } else {
        for (const file of selectedFiles) {
          try {
            const res = await apiClient.post(API_ENDPOINTS.convert, {
              document_id: file,
              format: targetFormat,
            });
            if (res.data?.success) {
              successCount += 1;
            } else {
              failCount += 1;
            }
          } catch {
            failCount += 1;
          }
        }
      }

      if (successCount > 0) {
        toast({
          title: 'Conversion complete',
          description:
            failCount > 0
              ? `${successCount} converted, ${failCount} failed`
              : `${successCount} converted successfully`,
          status: failCount > 0 ? 'warning' : 'success',
          duration: 3500,
        });
      } else {
        toast({
          title: 'Conversion failed',
          description: 'Could not convert selected files',
          status: 'error',
          duration: 4000,
        });
      }

      loadConvertedFiles();
      setSelectedFiles([]);
    } catch (e: any) {
      toast({ title: 'Conversion error', description: e.message, status: 'error', duration: 4000 });
    } finally {
      setConverting(false);
    }
  };

  // OCR
  const handleOCR = async (filename: string) => {
    setOcrLoading(prev => ({ ...prev, [filename]: true }));
    try {
      const res = await runOCR(filename);
      if (res.success && res.ocr_result) {
        setOcrResults(prev => ({ ...prev, [filename]: res.ocr_result! }));
        toast({ title: 'OCR complete', status: 'success', duration: 2000 });
      }
    } catch (e: any) {
      toast({ title: 'OCR failed', description: e.message, status: 'error', duration: 3000 });
    } finally {
      setOcrLoading(prev => ({ ...prev, [filename]: false }));
    }
  };

  // Batch OCR
  const handleBatchOCR = async () => {
    if (selectedFiles.length === 0) return;
    for (const file of selectedFiles) {
      if (!ocrResults[file]) {
        await handleOCR(file);
      }
    }
  };

  // Delete converted
  const handleDeleteConverted = async (filename: string) => {
    try {
      await apiClient.delete(`${API_ENDPOINTS.deleteConverted}/${filename}`);
      setConvertedFiles(prev => prev.filter(f => f !== filename));
      toast({ title: 'Deleted', status: 'info', duration: 2000 });
    } catch (e) {
      console.error('Delete error:', e);
    }
  };

  // Image files only
  const imageFiles = files.filter(f => {
    const ext = f.filename.split('.').pop()?.toLowerCase();
    return ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp', 'tiff'].includes(ext || '');
  });
  const pdfFiles = files.filter(f => f.filename.toLowerCase().endsWith('.pdf'));
  const conversionCandidates = targetFormat === 'pdf' ? imageFiles : pdfFiles;

  return (
    <PageShell>
      <VStack spacing={6} align="stretch" maxW="1400px" mx="auto" p={4}>
        {/* Header */}
        <HStack justify="space-between" wrap="wrap">
          <VStack align="start" spacing={0}>
            <Heading size="lg">Playground</Heading>
            <Text fontSize="sm" color="gray.500">
              Convert, extract text, and process your documents
            </Text>
          </VStack>
          <HStack>
            <Button size="sm" leftIcon={<Iconify icon={FiRefreshCw} boxSize={4} />} onClick={() => { loadFiles(); loadConvertedFiles(); }} isLoading={loading}>
              Refresh
            </Button>
          </HStack>
        </HStack>

        <Tabs variant="enclosed" colorScheme="brand">
          <TabList>
            <Tab><HStack><Iconify icon={FiLayers} boxSize={4} /><Text>Convert</Text></HStack></Tab>
            <Tab><HStack><Iconify icon={FiFileText} boxSize={4} /><Text>OCR</Text></HStack></Tab>
            <Tab><HStack><Iconify icon={FiCpu} boxSize={4} /><Text>Batch Process</Text></HStack></Tab>
          </TabList>

          <TabPanels>
            {/* ─── CONVERT TAB ─── */}
            <TabPanel px={0}>
              <VStack spacing={4} align="stretch">
                {/* Controls */}
                <Card bg={bgCard} borderWidth="1px" borderColor={borderColor}>
                  <CardBody>
                    <VStack spacing={3} align="stretch">
                      <HStack wrap="wrap" spacing={3}>
                        <Select
                          size="sm" w="140px" value={targetFormat}
                          onChange={e => setTargetFormat(e.target.value)}
                        >
                          <option value="pdf">PDF</option>
                          <option value="png">PNG</option>
                          <option value="jpg">JPG</option>
                        </Select>
                        {targetFormat === 'pdf' && (
                          <Checkbox isChecked={mergePdf} onChange={e => setMergePdf(e.target.checked)} size="sm">
                            Merge into single PDF
                          </Checkbox>
                        )}
                        {mergePdf && (
                          <Input
                            size="sm" w="200px" placeholder="Custom filename..."
                            value={customFilename} onChange={e => setCustomFilename(e.target.value)}
                          />
                        )}
                      </HStack>
                      <HStack>
                        <Button
                          size="sm" colorScheme="brand" onClick={handleConvert}
                          isLoading={converting} isDisabled={selectedFiles.length === 0}
                        >
                          Convert {selectedFiles.length > 0 ? `(${selectedFiles.length})` : ''}
                        </Button>
                        <Button size="sm" variant="outline" onClick={() => setSelectedFiles(conversionCandidates.map(f => f.filename))}>
                          Select All
                        </Button>
                        {selectedFiles.length > 0 && (
                          <Button size="sm" variant="ghost" onClick={() => setSelectedFiles([])}>
                            Deselect
                          </Button>
                        )}
                      </HStack>
                    </VStack>
                  </CardBody>
                </Card>

                {/* File selection grid */}
                <Text fontWeight="600" fontSize="sm">Select files to convert:</Text>
                <SimpleGrid columns={{ base: 2, md: 3, lg: 4, xl: 5 }} spacing={3}>
                  {conversionCandidates.map(file => {
                    const isSelected = selectedFiles.includes(file.filename);
                    return (
                      <MotionCard
                        key={file.filename}
                        size="sm"
                        cursor="pointer"
                        onClick={() => toggleFile(file.filename)}
                        borderWidth="2px"
                        borderColor={isSelected ? 'purple.400' : borderColor}
                        bg={isSelected ? selectedBg : bgCard}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        <CardBody p={2}>
                          <Box
                            h="100px" borderRadius="md" overflow="hidden" mb={2}
                            bg={bgSubtle} display="flex" alignItems="center" justifyContent="center"
                          >
                            <img
                              src={getImageUrl(API_ENDPOINTS.processed, file.filename)}
                              alt={file.filename}
                              style={{ maxHeight: '100%', maxWidth: '100%', objectFit: 'contain' }}
                              loading="lazy"
                            />
                          </Box>
                          <Text fontSize="xs" noOfLines={1} title={file.filename}>
                            {file.filename}
                          </Text>
                          {isSelected && <Badge colorScheme="brand" size="sm" mt={1}>Selected</Badge>}
                        </CardBody>
                      </MotionCard>
                    );
                  })}
                </SimpleGrid>

                {/* Converted files */}
                {convertedFiles.length > 0 && (
                  <>
                    <Divider />
                    <Heading size="sm">Converted Files ({convertedFiles.length})</Heading>
                    <SimpleGrid columns={{ base: 2, md: 3, lg: 4 }} spacing={3}>
                      {convertedFiles.map(filename => (
                        <Card key={filename} size="sm" bg={bgCard} borderWidth="1px" borderColor={borderColor}>
                          <CardBody p={3}>
                            <HStack justify="space-between">
                              <VStack align="start" spacing={0} flex={1} minW={0}>
                                <Text fontSize="xs" noOfLines={1} fontWeight="500">{filename}</Text>
                                <Badge colorScheme="green" size="sm">{filename.split('.').pop()?.toUpperCase()}</Badge>
                              </VStack>
                              <HStack spacing={1}>
                                <Tooltip label="Download">
                                  <IconButton
                                    aria-label="Download" icon={<Iconify icon={FiDownload} boxSize={3} />} size="xs" variant="ghost"
                                    onClick={() => window.open(withToken(`${API_BASE_URL}/document/files/${filename}/download`), '_blank')}
                                  />
                                </Tooltip>
                                <Tooltip label="Delete">
                                  <IconButton
                                    aria-label="Delete" icon={<Iconify icon={FiTrash2} boxSize={3} />} size="xs" variant="ghost" colorScheme="red"
                                    onClick={() => handleDeleteConverted(filename)}
                                  />
                                </Tooltip>
                              </HStack>
                            </HStack>
                          </CardBody>
                        </Card>
                      ))}
                    </SimpleGrid>
                  </>
                )}
              </VStack>
            </TabPanel>

            {/* ─── OCR TAB ─── */}
            <TabPanel px={0}>
              <VStack spacing={4} align="stretch">
                <Card bg={bgCard} borderWidth="1px" borderColor={borderColor}>
                  <CardBody>
                    <HStack>
                      <Button
                        size="sm" colorScheme="brand" onClick={handleBatchOCR}
                        isDisabled={selectedFiles.length === 0}
                      >
                        Run OCR on Selected ({selectedFiles.length})
                      </Button>
                      <Button size="sm" variant="outline" onClick={() => setSelectedFiles(imageFiles.map(f => f.filename))}>
                        Select All
                      </Button>
                      {selectedFiles.length > 0 && (
                        <Button size="sm" variant="ghost" onClick={() => setSelectedFiles([])}>Clear</Button>
                      )}
                    </HStack>
                  </CardBody>
                </Card>

                <SimpleGrid columns={{ base: 1, md: 2, xl: 3 }} spacing={3}>
                  {imageFiles.map(file => {
                    const hasOCR = !!ocrResults[file.filename];
                    const isRunning = !!ocrLoading[file.filename];
                    const isSelected = selectedFiles.includes(file.filename);
                    return (
                      <Card
                        key={file.filename} size="sm" bg={bgCard}
                        borderWidth="2px" borderColor={isSelected ? 'purple.400' : borderColor}
                        cursor="pointer" onClick={() => toggleFile(file.filename)}
                      >
                        <CardBody p={3}>
                          <HStack spacing={3}>
                            <Box
                              w="60px" h="60px" borderRadius="md" overflow="hidden" flexShrink={0}
                              bg={bgSubtle} display="flex" alignItems="center" justifyContent="center"
                            >
                              <img
                              src={getImageUrl(API_ENDPOINTS.processed, file.filename)}
                                alt={file.filename}
                                style={{ maxHeight: '100%', maxWidth: '100%', objectFit: 'contain' }}
                                loading="lazy"
                              />
                            </Box>
                            <VStack align="start" spacing={1} flex={1} minW={0}>
                              <Text fontSize="xs" noOfLines={1} fontWeight="500">{file.filename}</Text>
                              <HStack spacing={2}>
                                {hasOCR && (
                                  <Badge colorScheme="green" size="sm">
                                    OCR ✓ ({ocrResults[file.filename].word_count} words)
                                  </Badge>
                                )}
                                {!hasOCR && !isRunning && (
                                  <Button
                                    size="xs" colorScheme="blue" variant="outline"
                                    onClick={e => { e.stopPropagation(); handleOCR(file.filename); }}
                                  >
                                    Run OCR
                                  </Button>
                                )}
                                {isRunning && <Spinner size="xs" />}
                              </HStack>
                              {hasOCR && (
                                <Text fontSize="xs" color="gray.500" noOfLines={2}>
                                  {ocrResults[file.filename].full_text?.slice(0, 100)}...
                                </Text>
                              )}
                            </VStack>
                          </HStack>
                        </CardBody>
                      </Card>
                    );
                  })}
                </SimpleGrid>
              </VStack>
            </TabPanel>

            {/* ─── BATCH PROCESS TAB ─── */}
            <TabPanel px={0}>
              <VStack spacing={4} align="stretch">
                <Card bg={bgCard} borderWidth="1px" borderColor={borderColor}>
                  <CardBody>
                    <VStack spacing={3} align="start">
                      <Heading size="sm">Batch Document Processing</Heading>
                      <Text fontSize="sm" color="gray.500">
                        Run the full 12-step document pipeline on selected files: detection, crop,
                        shadow removal, enhancement, and OCR text extraction.
                      </Text>
                      <HStack>
                        <Button
                          size="sm" colorScheme="brand"
                          isDisabled={selectedFiles.length === 0}
                          onClick={async () => {
                            toast({ title: `Processing ${selectedFiles.length} files...`, status: 'info', duration: 2000 });
                            for (const file of selectedFiles) {
                              try {
                                await apiClient.post(API_ENDPOINTS.editEnhance, { filename: file, save_as_new: true });
                              } catch { /* continue */ }
                            }
                            toast({ title: 'Batch processing complete', status: 'success', duration: 3000 });
                            loadFiles();
                          }}
                        >
                          Enhance {selectedFiles.length > 0 ? `(${selectedFiles.length})` : ''}
                        </Button>
                        <Button size="sm" variant="outline" onClick={() => setSelectedFiles(imageFiles.map(f => f.filename))}>
                          Select All
                        </Button>
                        {selectedFiles.length > 0 && (
                          <Button size="sm" variant="ghost" onClick={() => setSelectedFiles([])}>Clear</Button>
                        )}
                      </HStack>
                    </VStack>
                  </CardBody>
                </Card>

                <SimpleGrid columns={{ base: 2, md: 3, lg: 4, xl: 5 }} spacing={3}>
                  {imageFiles.map(file => {
                    const isSelected = selectedFiles.includes(file.filename);
                    return (
                      <Card
                        key={file.filename} size="sm" cursor="pointer"
                        onClick={() => toggleFile(file.filename)}
                        borderWidth="2px" borderColor={isSelected ? 'purple.400' : borderColor}
                        bg={bgCard}
                      >
                        <CardBody p={2}>
                          <Box
                            h="80px" borderRadius="md" overflow="hidden" mb={2}
                            bg={bgSubtle} display="flex" alignItems="center" justifyContent="center"
                          >
                            <img
                              src={getImageUrl(API_ENDPOINTS.processed, file.filename)}
                              alt={file.filename}
                              style={{ maxHeight: '100%', maxWidth: '100%', objectFit: 'contain' }}
                              loading="lazy"
                            />
                          </Box>
                          <Text fontSize="xs" noOfLines={1}>{file.filename}</Text>
                          {isSelected && <Badge colorScheme="brand" size="sm" mt={1}>✓</Badge>}
                        </CardBody>
                      </Card>
                    );
                  })}
                </SimpleGrid>
              </VStack>
            </TabPanel>
          </TabPanels>
        </Tabs>
      </VStack>
    </PageShell>
  );
};

export default Playground;
