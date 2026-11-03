import React, { useEffect, useState, useCallback, useMemo } from 'react';
import {
  Box,
  Button,
  Divider,
  Flex,
  HStack,
  IconButton,
  Input,
  Link,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalHeader,
  ModalOverlay,
  NumberDecrementStepper,
  NumberIncrementStepper,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  Progress,
  Radio,
  RadioGroup,
  SimpleGrid,
  Slider,
  SliderFilledTrack,
  SliderThumb,
  SliderTrack,
  Spinner,
  Stack,
  Tag,
  Text,
  Tooltip,
  useColorModeValue,
  useDisclosure,
  useToast,
  VStack,
} from '@chakra-ui/react';
import { FiClock, FiCpu, FiDownload, FiMonitor, FiPlay, FiPrinter, FiRefreshCw, FiSettings, FiTrash2 } from 'react-icons/fi';
import { Iconify } from '../common';
import apiClient from '../../apiClient';
import { API_ENDPOINTS } from '../../config';
import { useCalibration } from '../../context/CalibrationContext';

interface PrinterInfo {
  name: string;
  is_default: boolean;
  driver: string;
  port: string;
  status: string;
}

interface DriverSuggestion {
  printer: string;
  brand: string;
  driver_url: string;
  description: string;
}

interface SystemInfo {
  os: {
    name: string;
    version: string;
    release: string;
    architecture: string;
    processor: string;
  };
  python: {
    version: string;
    implementation: string;
  };
  memory: {
    total_gb: number;
    available_gb: number;
    used_percent: number;
  };
  cpu: {
    cores_physical: number;
    cores_logical: number;
    usage_percent: number;
  };
  gpu: {
    available: boolean;
    name?: string;
    cuda_version?: string;
    device_count?: number;
    total_memory_gb?: number;
    error?: string;
  };
  printers: {
    available: boolean;
    default?: string;
    count?: number;
    list: PrinterInfo[];
    error?: string;
  };
  driver_suggestions: DriverSuggestion[];
}

interface PrintJobInfo {
  id: string;
  document?: string;
  owner?: string;
  status?: string;
  submitted?: string;
  totalPages?: number | null;
  pagesPrinted?: number | null;
  sizeBytes?: number | null;
}

interface PrinterQueueSnapshot {
  name: string;
  status: string;
  isDefault: boolean;
  jobs: PrintJobInfo[];
}

export interface DeviceInfoPanelProps {
  isOpen?: boolean;
  onClose?: () => void;
  showButton?: boolean;
}

export const DeviceInfoPanel: React.FC<DeviceInfoPanelProps> = ({
  isOpen: externalIsOpen,
  onClose: externalOnClose,
  showButton = true,
}) => {
  const { isOpen: internalIsOpen, onOpen: internalOnOpen, onClose: internalOnClose } = useDisclosure();
  
  // Use external state if provided, otherwise use internal state
  const isOpen = externalIsOpen !== undefined ? externalIsOpen : internalIsOpen;
  const onOpen = internalOnOpen;
  const onClose = externalOnClose || internalOnClose;
  
  const printQueuesRef = React.useRef<HTMLDivElement | null>(null);
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [defaultPrinter, setDefaultPrinter] = useState<string>('');
  const [settingDefault, setSettingDefault] = useState(false);
  const [clearingQueue, setClearingQueue] = useState(false);
  const [printerQueues, setPrinterQueues] = useState<PrinterQueueSnapshot[]>([]);
  const [queuesLoading, setQueuesLoading] = useState(false);
  const [queueError, setQueueError] = useState<string | null>(null);
  const [terminatingJobId, setTerminatingJobId] = useState<string | null>(null);
  const toast = useToast();

  // Calibration state
  const {
    initialDelay,
    setInitialDelay,
    isCalibrated,
    startDelayCountdown,
    countdownValue,
    isCountingDown,
    cancelCountdown,
    resetCalibration,
  } = useCalibration();
  const [showCalibrationOptions, setShowCalibrationOptions] = useState(false);
  const [tempDelay, setTempDelay] = useState<number>(initialDelay);
  const [isTestingDelay, setIsTestingDelay] = useState(false);

  // Sync tempDelay with initialDelay when it changes externally
  useEffect(() => {
    setTempDelay(initialDelay);
  }, [initialDelay]);

  // Consistent styling with SurfaceCard
  const bgCard = useColorModeValue('rgba(255, 248, 240, 0.95)', 'rgba(12, 16, 35, 0.92)');
  const borderColor = useColorModeValue('rgba(17, 17, 17, 0.08)', 'rgba(255, 255, 255, 0.08)');
  const shadow = useColorModeValue(
    '0 20px 35px rgba(17, 17, 17, 0.08)',
    '0 25px 50px rgba(0, 0, 0, 0.8)'
  );
  const textMuted = useColorModeValue('gray.600', 'gray.400');
  const tagBg = useColorModeValue('gray.100', 'rgba(17, 17, 17, 0.15)');

  const fetchSystemInfo = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiClient.get('/system/info');
      const raw = response.data || {};

      // Normalize response to prevent runtime crashes when backend omits nested fields.
      const normalized: SystemInfo = {
        os: {
          name: raw.os?.name || 'Unknown',
          version: raw.os?.version || '',
          release: raw.os?.release || '',
          architecture: raw.os?.architecture || '',
          processor: raw.os?.processor || '',
        },
        python: {
          version: raw.python?.version || '',
          implementation: raw.python?.implementation || '',
        },
        memory: {
          total_gb: Number(raw.memory?.total_gb || 0),
          available_gb: Number(raw.memory?.available_gb || 0),
          used_percent: Number(raw.memory?.used_percent || 0),
        },
        cpu: {
          cores_physical: Number(raw.cpu?.cores_physical || 0),
          cores_logical: Number(raw.cpu?.cores_logical || 0),
          usage_percent: Number(raw.cpu?.usage_percent || 0),
        },
        gpu: {
          available: Boolean(raw.gpu?.available),
          name: raw.gpu?.name,
          cuda_version: raw.gpu?.cuda_version,
          device_count: raw.gpu?.device_count,
          total_memory_gb: raw.gpu?.total_memory_gb,
          error: raw.gpu?.error,
        },
        printers: {
          available: Boolean(raw.printers?.available),
          default: raw.printers?.default || '',
          count: Number(raw.printers?.count || 0),
          list: Array.isArray(raw.printers?.list) ? raw.printers.list : [],
          error: raw.printers?.error,
        },
        driver_suggestions: Array.isArray(raw.driver_suggestions) ? raw.driver_suggestions : [],
      };

      setSystemInfo(normalized);
      setDefaultPrinter(normalized.printers.default || '');
    } catch (err: any) {
      setError(err.message || 'Failed to fetch system info');
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchPrinterQueues = useCallback(async () => {
    setQueuesLoading(true);
    setQueueError(null);
    try {
      const { data } = await apiClient.get(API_ENDPOINTS.printerQueues);
      setPrinterQueues(data?.printers || []);
    } catch (err: any) {
      console.error('Failed to fetch printer queues', err);
      setQueueError(err.response?.data?.error || err.message || 'Failed to fetch printer queues');
    } finally {
      setQueuesLoading(false);
    }
  }, []);

  const handleSetDefaultPrinter = useCallback(async (printerName: string) => {
    setSettingDefault(true);
    setError(null);
    try {
      await apiClient.post('/print/printers/default', { printer: printerName });
      setDefaultPrinter(printerName);
      if (systemInfo) {
        setSystemInfo({
          ...systemInfo,
          printers: {
            ...systemInfo.printers,
            default: printerName,
          },
        });
      }
    } catch (err: any) {
      console.error('Failed to set default printer:', err);
      setError(err.response?.data?.error || 'Failed to set default printer');
    } finally {
      setSettingDefault(false);
    }
  }, [systemInfo]);

  const refreshSystemStatus = useCallback(() => {
    fetchSystemInfo();
    fetchPrinterQueues();
  }, [fetchSystemInfo, fetchPrinterQueues]);

  useEffect(() => {
    refreshSystemStatus();
  }, [refreshSystemStatus]);

  const queueByPrinter = useMemo(() => {
    const map = new Map<string, PrinterQueueSnapshot>();
    printerQueues.forEach((printer) => {
      map.set(printer.name, printer);
    });
    return map;
  }, [printerQueues]);

  const handleClearPrintQueue = useCallback(async () => {
    setClearingQueue(true);
    try {
      const { data } = await apiClient.post(API_ENDPOINTS.printerClearQueue);
      toast({
        title: 'Printing queue cleared',
        description: data?.message || 'All pending jobs were removed',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
      fetchPrinterQueues();
    } catch (err: any) {
      const message = err.response?.data?.error || err.message || 'Failed to clear printing queue';
      toast({
        title: 'Unable to clear queue',
        description: message,
        status: 'error',
        duration: 4000,
        isClosable: true,
      });
      setError(message);
    } finally {
      setClearingQueue(false);
    }
  }, [toast, fetchPrinterQueues]);

  const handleTerminateJob = useCallback(
    async (printerName: string, jobId: string) => {
      const key = `${printerName}_${jobId}`;
      setTerminatingJobId(key);
      try {
        await apiClient.post(API_ENDPOINTS.printerCancelJob, {
          printer_name: printerName,
          job_id: jobId,
        });
        toast({
          title: 'Print job terminated',
          status: 'success',
          duration: 2000,
          isClosable: true,
        });
        fetchPrinterQueues();
      } catch (err: any) {
        const description = err.response?.data?.error || err.message || 'Failed to terminate job';
        toast({
          title: 'Unable to terminate job',
          description,
          status: 'error',
          duration: 4000,
          isClosable: true,
        });
      } finally {
        setTerminatingJobId(null);
      }
    },
    [toast, fetchPrinterQueues]
  );

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'ready':
        return 'green';
      case 'offline':
      case 'error':
        return 'red';
      case 'busy':
      case 'printing':
        return 'blue';
      case 'paused':
        return 'yellow';
      default:
        return 'gray';
    }
  };

  const getMemoryColor = (percent: number) => {
    if (percent < 50) return 'green';
    if (percent < 80) return 'yellow';
    return 'red';
  };

  return (
    <>
      {showButton && (
        <Button
          variant="ghost"
          size="sm"
          leftIcon={<Iconify icon={FiMonitor} />}
          color={textMuted}
          _hover={{ color: 'brand.400', bg: 'rgba(17, 17, 17, 0.1)' }}
          fontWeight="medium"
          onClick={onOpen}
        >
          Device Info
        </Button>
      )}

      <Modal isOpen={isOpen} onClose={onClose} size="4xl" isCentered scrollBehavior="inside">
        <ModalOverlay bg="blackAlpha.600" />
        <ModalContent
          bg={bgCard}
          borderRadius="2xl"
          border="1px solid"
          borderColor={borderColor}
          boxShadow={shadow}
          maxH="90vh"
        >
          <Flex
            borderBottom="1px solid"
            borderColor={borderColor}
            align="center"
            justify="space-between"
            py={4}
            px={8}
          >
            <HStack spacing={3}>
              <Iconify icon={FiMonitor} boxSize={6} color="brand.400" />
              <Text fontWeight="700" fontSize="lg">
                System Status & Calibration
              </Text>
              {systemInfo?.printers?.available && (
                <Tag size="lg" colorScheme="green" variant="subtle" borderRadius="full">
                  {systemInfo.printers.count} Printers
                </Tag>
              )}
            </HStack>
            <HStack spacing={2}>
              <Tooltip label="Print queues" placement="top">
                <Button
                  size="sm"
                  variant="ghost"
                  color={textMuted}
                  leftIcon={<Iconify icon={FiPrinter} boxSize={4} />}
                  onClick={() => {
                    // fetch queues and scroll to section
                    fetchPrinterQueues();
                    setTimeout(() => {
                      printQueuesRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }, 150);
                  }}
                >
                  Print Queues
                </Button>
              </Tooltip>
              <Tooltip label="Refresh" placement="top">
                <IconButton
                  aria-label="Refresh"
                  icon={<Iconify icon={FiRefreshCw} boxSize={4} />}
                  size="sm"
                  variant="ghost"
                  color={textMuted}
                  _hover={{ color: 'brand.400', bg: 'rgba(17, 17, 17, 0.1)' }}
                  onClick={refreshSystemStatus}
                  isLoading={loading}
                />
              </Tooltip>
              <ModalCloseButton position="relative" top="0" right="0" />
            </HStack>
          </Flex>

          <ModalBody py={6} px={8}>
            {loading && !systemInfo ? (
              <Flex justify="center" py={16}>
                <Spinner size="lg" color="brand.400" thickness="4px" />
              </Flex>
            ) : error ? (
              <Text color="red.400" fontSize="md" py={4}>
                {error}
              </Text>
            ) : systemInfo ? (
              <VStack align="stretch" spacing={6} w="100%">
                {/* Printers Section */}
                <Box
                  bg={useColorModeValue('gray.50', 'rgba(17, 17, 17, 0.05)')}
                  borderRadius="xl"
                  p={5}
                  border="1px solid"
                  borderColor={borderColor}
                >
                  <Flex mb={4} align="center" gap={3} wrap="wrap">
                    <HStack spacing={3}>
                      <Iconify icon={FiPrinter} boxSize={5} color="brand.400" />
                      <Text fontWeight="700" fontSize="sm" textTransform="uppercase" letterSpacing="wider" color="brand.400">
                        Printers & Default Selection
                      </Text>
                    </HStack>
                    <Box flex="1" />
                    <HStack spacing={2}>
                      <Tooltip label="Refresh queue status" placement="top">
                        <IconButton
                          aria-label="Refresh queues"
                          icon={<Iconify icon={FiRefreshCw} boxSize={4} />}
                          size="sm"
                          variant="ghost"
                          onClick={fetchPrinterQueues}
                          isLoading={queuesLoading}
                        />
                      </Tooltip>
                      <Tooltip label="Clear ALL pending print jobs" placement="top">
                        <Button
                          size="sm"
                          colorScheme="red"
                          variant="solid"
                          leftIcon={<Iconify icon={FiTrash2} boxSize={4} />}
                          onClick={handleClearPrintQueue}
                          isLoading={clearingQueue}
                        >
                          Clear All Queues
                        </Button>
                      </Tooltip>
                    </HStack>
                  </Flex>

                  {systemInfo.printers.available ? (
                    <RadioGroup value={defaultPrinter} onChange={(value) => handleSetDefaultPrinter(value)}>
                      <VStack align="stretch" spacing={3} maxH="300px" overflowY="auto">
                        {systemInfo.printers.list.map((printer, idx) => {
                          const queue = queueByPrinter.get(printer.name);
                          const jobCount = queue?.jobs?.length || 0;
                          return (
                            <Box
                              key={idx}
                              bg={useColorModeValue('white', 'rgba(255,255,255,0.03)')}
                              px={4}
                              py={3}
                              borderRadius="lg"
                              border="1.5px solid"
                              borderColor={defaultPrinter === printer.name ? 'brand.400' : borderColor}
                              cursor="pointer"
                              transition="all 0.2s"
                              _hover={{ borderColor: 'brand.400', boxShadow: `0 0 0 2px rgba(17, 17, 17, 0.1)` }}
                              onClick={() => {
                                if (!settingDefault && defaultPrinter !== printer.name) {
                                  handleSetDefaultPrinter(printer.name);
                                }
                              }}
                              onKeyDown={(e) => {
                                if ((e.key === 'Enter' || e.key === ' ') && !settingDefault && defaultPrinter !== printer.name) {
                                  e.preventDefault();
                                  handleSetDefaultPrinter(printer.name);
                                }
                              }}
                              role="button"
                              tabIndex={0}
                              aria-label={`Select ${printer.name} as default printer`}
                            >
                              <Flex align="center" justify="space-between" gap={3}>
                                <Radio value={printer.name} isDisabled={settingDefault} pointerEvents="none" />
                                <VStack align="start" spacing={0.5} flex={1}>
                                  <Text fontSize="sm" fontWeight="600">
                                    {printer.name}
                                  </Text>
                                  <Text fontSize="xs" color={textMuted}>
                                    {printer.driver}
                                  </Text>
                                </VStack>
                                <HStack spacing={2}>
                                  <Tag size="sm" colorScheme={getStatusColor(queue?.status || printer.status)} variant="subtle">
                                    {queue?.status || printer.status}
                                  </Tag>
                                  <Tag size="sm" colorScheme={jobCount > 0 ? 'purple' : 'gray'} variant="subtle">
                                    {jobCount} in queue
                                  </Tag>
                                  {defaultPrinter === printer.name && (
                                    <Tag size="sm" colorScheme="brand" variant="solid" fontSize="11px">
                                      ACTIVE
                                  </Tag>
                                )}
                              </HStack>
                            </Flex>

                            <Box mt={3} pl={6}>
                              {queuesLoading && !queue ? (
                                <HStack spacing={2} color={textMuted}>
                                  <Spinner size="xs" />
                                  <Text fontSize="xs">Loading queue…</Text>
                                </HStack>
                              ) : jobCount === 0 ? (
                                <Text fontSize="xs" color={textMuted} fontStyle="italic">
                                  No pending jobs.
                                </Text>
                              ) : (
                                <Stack spacing={2}>
                                  {queue?.jobs?.map((job) => {
                                    const terminateKey = `${printer.name}_${job.id}`;
                                    const submittedLabel = job.submitted
                                      ? Number.isNaN(Date.parse(job.submitted))
                                        ? job.submitted
                                        : new Date(job.submitted).toLocaleString()
                                      : null;
                                    return (
                                      <Flex
                                        key={terminateKey}
                                        justify="space-between"
                                        align={{ base: 'flex-start', md: 'center' }}
                                        gap={3}
                                        direction={{ base: 'column', md: 'row' }}
                                        p={2}
                                        border="1px solid"
                                        borderColor={borderColor}
                                        borderRadius="md"
                                        bg={useColorModeValue('gray.50', 'rgba(255,255,255,0.04)')}
                                      >
                                        <Box flex={1}>
                                          <Text fontSize="sm" fontWeight="600">
                                            {job.document || `Job #${job.id}`}
                                          </Text>
                                          <Text fontSize="xs" color={textMuted}>
                                            {job.owner} • {job.status || 'pending'}
                                          </Text>
                                          {submittedLabel && (
                                            <Text fontSize="xs" color={textMuted}>
                                              {submittedLabel}
                                            </Text>
                                          )}
                                        </Box>
                                        <Button
                                          size="xs"
                                          colorScheme="red"
                                          variant="ghost"
                                          onClick={(e) => {
                                            e.stopPropagation();
                                            handleTerminateJob(printer.name, String(job.id));
                                          }}
                                          isLoading={terminatingJobId === terminateKey}
                                        >
                                          Terminate
                                        </Button>
                                      </Flex>
                                    );
                                  })}
                                </Stack>
                              )}
                            </Box>
                          </Box>
                        );
                      })}
                      {queueError && (
                        <Text fontSize="xs" color="red.400">
                          {queueError}
                        </Text>
                      )}
                      </VStack>
                    </RadioGroup>
                  ) : (
                    <Text fontSize="sm" color={textMuted} fontStyle="italic">
                      {systemInfo.printers.error || 'No printers found'}
                    </Text>
                  )}
                </Box>

                <Divider borderColor={borderColor} opacity={0.4} />

                {/* Driver Downloads & Print Queues - Side by Side */}
                <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6} w="100%">
                  {/* Driver Suggestions - Left */}
                  <Box>
                    <HStack mb={4} spacing={3}>
                      <Iconify icon={FiDownload} boxSize={5} color="brand.400" />
                      <Text fontWeight="700" fontSize="sm" textTransform="uppercase" letterSpacing="wider" color="brand.400">
                        Driver Downloads
                      </Text>
                    </HStack>
                    {systemInfo.driver_suggestions.length > 0 ? (
                      <VStack align="stretch" spacing={3}>
                        {systemInfo.driver_suggestions.map((suggestion, idx) => (
                          <Link
                            key={idx}
                            href={suggestion.driver_url}
                            isExternal
                            _hover={{ textDecoration: 'none' }}
                          >
                            <Box
                              bg={useColorModeValue('white', 'rgba(255,255,255,0.03)')}
                              px={4}
                              py={3}
                              borderRadius="lg"
                              border="1px solid"
                              borderColor={borderColor}
                              _hover={{ bg: 'rgba(17, 17, 17, 0.1)', borderColor: 'brand.400', transform: 'translateY(-2px)' }}
                              transition="all 0.2s"
                              boxShadow="sm"
                            >
                              <Flex align="center" justify="space-between" gap={2}>
                                <VStack align="start" spacing={0}>
                                  <Text fontSize="sm" fontWeight="700">
                                    {suggestion.brand}
                                  </Text>
                                  <Text fontSize="xs" color={textMuted}>
                                    {suggestion.description}
                                  </Text>
                                </VStack>
                                <Iconify icon={FiDownload} boxSize={4} color="brand.400" />
                              </Flex>
                            </Box>
                          </Link>
                        ))}
                      </VStack>
                    ) : (
                      <Box
                        bg={useColorModeValue('white', 'rgba(255,255,255,0.03)')}
                        px={4}
                        py={3}
                        borderRadius="lg"
                        border="1px solid"
                        borderColor={borderColor}
                      >
                        <Text fontSize="sm" color={textMuted} fontStyle="italic" textAlign="center">
                          No driver suggestions
                        </Text>
                      </Box>
                    )}
                  </Box>

                  {/* Print Queues - Right */}
                  <Box ref={printQueuesRef as any} id="print-queues-section">
                    <Flex mb={4} align="center" justify="space-between" wrap="wrap" gap={3}>
                      <HStack spacing={3}>
                        <Iconify icon={FiPrinter} boxSize={5} color="brand.400" />
                        <Text fontWeight="700" fontSize="sm" textTransform="uppercase" letterSpacing="wider" color="brand.400">
                          Print Queues
                        </Text>
                      </HStack>
                      <HStack spacing={2}>
                        <Tooltip label="Refresh queues" placement="top">
                          <IconButton
                            aria-label="Refresh queues"
                            icon={<Iconify icon={FiRefreshCw} boxSize={4} />}
                            size="sm"
                            variant="ghost"
                            onClick={fetchPrinterQueues}
                            isLoading={queuesLoading}
                          />
                        </Tooltip>
                        <Tooltip label="Clear ALL pending print jobs" placement="top">
                          <Button
                            size="sm"
                            colorScheme="red"
                            variant="solid"
                            leftIcon={<Iconify icon={FiTrash2} boxSize={4} />}
                            onClick={handleClearPrintQueue}
                            isLoading={clearingQueue}
                          >
                            Clear All Print Jobs
                          </Button>
                        </Tooltip>
                      </HStack>
                    </Flex>
                    <VStack align="stretch" spacing={2} maxH="200px" overflowY="auto">
                      {queuesLoading ? (
                        <HStack spacing={2} color={textMuted} p={3}>
                          <Spinner size="xs" />
                          <Text fontSize="xs">Loading queues...</Text>
                        </HStack>
                      ) : printerQueues.length === 0 || printerQueues.every(q => !q.jobs || q.jobs.length === 0) ? (
                        <Box
                          bg={useColorModeValue('white', 'rgba(255,255,255,0.03)')}
                          px={4}
                          py={3}
                          borderRadius="lg"
                          border="1px solid"
                          borderColor={borderColor}
                        >
                          <Text fontSize="sm" color={textMuted} fontStyle="italic" textAlign="center">
                            No pending print jobs
                          </Text>
                        </Box>
                      ) : (
                        printerQueues.filter(q => q.jobs && q.jobs.length > 0).map((printerQueue) => (
                          <Box
                            key={printerQueue.name}
                            bg={useColorModeValue('white', 'rgba(255,255,255,0.03)')}
                            px={3}
                            py={2}
                            borderRadius="lg"
                            border="1px solid"
                            borderColor={borderColor}
                          >
                            <Text fontSize="xs" fontWeight="700" color="brand.400" mb={1}>
                              {printerQueue.name}
                            </Text>
                            {printerQueue.jobs.map((job) => {
                              const terminateKey = `${printerQueue.name}_${job.id}`;
                              return (
                                <Flex
                                  key={terminateKey}
                                  justify="space-between"
                                  align="center"
                                  py={1}
                                  borderBottom="1px solid"
                                  borderColor={borderColor}
                                  _last={{ borderBottom: 'none' }}
                                >
                                  <Box flex={1}>
                                    <Text fontSize="xs" fontWeight="500" isTruncated maxW="150px">
                                      {job.document || `Job #${job.id}`}
                                    </Text>
                                    <Text fontSize="10px" color={textMuted}>
                                      {job.status || 'pending'}
                                    </Text>
                                  </Box>
                                  <Button
                                    size="xs"
                                    colorScheme="red"
                                    variant="ghost"
                                    fontSize="10px"
                                    h="20px"
                                    onClick={() => handleTerminateJob(printerQueue.name, String(job.id))}
                                    isLoading={terminatingJobId === terminateKey}
                                  >
                                    Cancel
                                  </Button>
                                </Flex>
                              );
                            })}
                          </Box>
                        ))
                      )}
                      {queueError && (
                        <Text fontSize="xs" color="red.400">
                          {queueError}
                        </Text>
                      )}
                    </VStack>
                  </Box>
                </SimpleGrid>

                <Divider borderColor={borderColor} opacity={0.4} />

                {/* System Resources */}
                <Box>
                  <HStack mb={4} spacing={3}>
                    <Iconify icon={FiCpu} boxSize={5} color="brand.400" />
                    <Text fontWeight="700" fontSize="sm" textTransform="uppercase" letterSpacing="wider" color="brand.400">
                      System Resources
                    </Text>
                  </HStack>

                  <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4} w="100%">
                    {/* Left Column */}
                    <VStack align="stretch" spacing={4}>
                      {/* OS */}
                      <Flex justify="space-between" align="center" bg={useColorModeValue('gray.50', 'rgba(17, 17, 17, 0.05)')} p={3} borderRadius="lg">
                        <Text fontSize="sm" color={textMuted} fontWeight="600">
                          OS
                        </Text>
                        <Tag size="sm" variant="outline" colorScheme="gray">
                          {systemInfo.os.name} {systemInfo.os.release}
                        </Tag>
                      </Flex>

                      {/* CPU */}
                      <Flex justify="space-between" align="center" bg={useColorModeValue('gray.50', 'rgba(17, 17, 17, 0.05)')} p={3} borderRadius="lg">
                        <Text fontSize="sm" color={textMuted} fontWeight="600">
                          CPU
                        </Text>
                        <HStack spacing={2}>
                          <Text fontWeight="600" fontSize="sm">
                            {systemInfo.cpu.usage_percent.toFixed(0)}%
                          </Text>
                          <Text color={textMuted} fontSize="xs">
                            ({systemInfo.cpu.cores_logical} cores)
                          </Text>
                        </HStack>
                      </Flex>
                    </VStack>

                    {/* Right Column */}
                    <VStack align="stretch" spacing={4}>
                      {/* Memory */}
                      <Box bg={useColorModeValue('gray.50', 'rgba(17, 17, 17, 0.05)')} p={3} borderRadius="lg">
                        <Flex justify="space-between" align="center" mb={2}>
                          <Text fontSize="sm" color={textMuted} fontWeight="600">
                            Memory
                          </Text>
                          <Text fontSize="sm" fontWeight="600">
                            {systemInfo.memory.available_gb.toFixed(1)} / {systemInfo.memory.total_gb.toFixed(1)} GB
                          </Text>
                        </Flex>
                        <Progress
                          value={systemInfo.memory.used_percent}
                          size="sm"
                          colorScheme={getMemoryColor(systemInfo.memory.used_percent)}
                          borderRadius="full"
                          bg={useColorModeValue('gray.200', 'whiteAlpha.300')}
                        />
                      </Box>

                      {/* GPU */}
                      <Box bg={useColorModeValue('gray.50', 'rgba(17, 17, 17, 0.05)')} p={3} borderRadius="lg">
                        <Flex justify="space-between" align="center">
                          <Text fontSize="sm" color={textMuted} fontWeight="600">
                            GPU
                          </Text>
                          {systemInfo.gpu.available ? (
                            <Tooltip label={`CUDA ${systemInfo.gpu.cuda_version} • ${systemInfo.gpu.total_memory_gb?.toFixed(1)} GB VRAM`}>
                              <Tag size="sm" colorScheme="green" variant="subtle">
                                <Iconify icon={FiCpu} mr={1} boxSize={3} />
                                {systemInfo.gpu.name?.replace('NVIDIA ', '').replace('GeForce ', '')}
                              </Tag>
                            </Tooltip>
                          ) : (
                            <Tag size="sm" colorScheme="gray" variant="subtle">
                              CPU Mode
                            </Tag>
                          )}
                        </Flex>
                      </Box>
                    </VStack>
                  </SimpleGrid>
                </Box>
              </VStack>
            ) : null}
          </ModalBody>
        </ModalContent>
      </Modal>
    </>
  );
};

export default DeviceInfoPanel;
