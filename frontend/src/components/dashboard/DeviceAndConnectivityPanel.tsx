import React, { useEffect, useState, useCallback } from 'react';
import {
  Box,
  Button,
  ButtonGroup,
  Divider,
  Flex,
  HStack,
  IconButton,
  Link,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalHeader,
  ModalOverlay,
  Progress,
  Radio,
  RadioGroup,
  SimpleGrid,
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
import { FiCpu, FiDownload, FiMonitor, FiPrinter, FiRefreshCw, FiSettings, FiTrash2, FiWifiOff } from 'react-icons/fi';
import { Iconify } from '../common';
import apiClient from '../../apiClient';
import { API_ENDPOINTS } from '../../config';
import CalibrationModal from './CalibrationModal';

interface PrintJobInfo {
  id: string | number;
  document: string;
  owner: string;
  status: string;
  submitted: string;
}

interface PrinterQueueSnapshot {
  name: string;
  status: string;
  isDefault: boolean;
  jobs: PrintJobInfo[];
}

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

interface DeviceAndConnectivityPanelProps {
  onCheckConnectivity?: () => void;
}

const createDefaultSystemInfo = (): SystemInfo => ({
  os: {
    name: 'Unknown',
    version: 'Unknown',
    release: 'Unknown',
    architecture: 'Unknown',
    processor: 'Unknown',
  },
  python: {
    version: 'Unknown',
    implementation: 'Python',
  },
  memory: {
    total_gb: 0,
    available_gb: 0,
    used_percent: 0,
  },
  cpu: {
    cores_physical: 0,
    cores_logical: 0,
    usage_percent: 0,
  },
  gpu: {
    available: false,
  },
  printers: {
    available: false,
    default: '',
    count: 0,
    list: [],
  },
  driver_suggestions: [],
});

const normalizeSystemInfo = (payload: any): SystemInfo => {
  const fallback = createDefaultSystemInfo();
  const raw = payload?.system ?? payload ?? {};
  const printerList = Array.isArray(raw.printers)
    ? raw.printers
    : Array.isArray(raw.printers?.list)
      ? raw.printers.list
      : [];
  const defaultPrinter = raw.default_printer || raw.printers?.default || '';

  return {
    os: {
      name: raw.os?.name || raw.platform || fallback.os.name,
      version: raw.os?.version || raw.platform_version || fallback.os.version,
      release: raw.os?.release || raw.platform_release || fallback.os.release,
      architecture: raw.os?.architecture || raw.architecture || fallback.os.architecture,
      processor: raw.os?.processor || raw.processor || fallback.os.processor,
    },
    python: {
      version: raw.python?.version || raw.python_version || fallback.python.version,
      implementation: raw.python?.implementation || fallback.python.implementation,
    },
    memory: {
      total_gb: Number(raw.memory?.total_gb ?? fallback.memory.total_gb),
      available_gb: Number(raw.memory?.available_gb ?? fallback.memory.available_gb),
      used_percent: Number(raw.memory?.used_percent ?? fallback.memory.used_percent),
    },
    cpu: {
      cores_physical: Number(raw.cpu?.cores_physical ?? fallback.cpu.cores_physical),
      cores_logical: Number(raw.cpu?.cores_logical ?? fallback.cpu.cores_logical),
      usage_percent: Number(raw.cpu?.usage_percent ?? fallback.cpu.usage_percent),
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
      available: printerList.length > 0 || Boolean(raw.printers?.available),
      default: defaultPrinter,
      count: Number(raw.printers?.count ?? printerList.length),
      list: printerList.map((printer: any) => ({
        name: printer?.name || 'Unknown printer',
        is_default: Boolean(printer?.is_default) || printer?.name === defaultPrinter,
        driver: printer?.driver || printer?.driver_name || 'Unknown driver',
        port: printer?.port || printer?.port_name || 'Unknown port',
        status: printer?.status || 'unknown',
      })),
      error: raw.printers?.error,
    },
    driver_suggestions: Array.isArray(raw.driver_suggestions) ? raw.driver_suggestions : [],
  };
};

export const DeviceAndConnectivityPanel: React.FC<DeviceAndConnectivityPanelProps> = ({ onCheckConnectivity }) => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const printQueueModal = useDisclosure();
  const calibrationModal = useDisclosure();
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [defaultPrinter, setDefaultPrinter] = useState<string>('');
  const [settingDefault, setSettingDefault] = useState(false);
  
  // Print Queue state
  const [printerQueues, setPrinterQueues] = useState<PrinterQueueSnapshot[]>([]);
  const [queuesLoading, setQueuesLoading] = useState(false);
  const [clearingQueue, setClearingQueue] = useState(false);
  const [terminatingJobId, setTerminatingJobId] = useState<string | null>(null);
  const toast = useToast();

  // Consistent styling with SurfaceCard
  const bgCard = useColorModeValue('rgba(255, 248, 240, 0.95)', 'rgba(12, 16, 35, 0.92)');
  const borderColor = useColorModeValue('rgba(17, 17, 17, 0.08)', 'rgba(255, 255, 255, 0.08)');
  const shadow = useColorModeValue(
    '0 20px 35px rgba(17, 17, 17, 0.08)',
    '0 25px 50px rgba(0, 0, 0, 0.8)'
  );
  const textMuted = useColorModeValue('gray.600', 'gray.400');
  const textColor = useColorModeValue('gray.800', 'white');
  const tagBg = useColorModeValue('gray.100', 'rgba(17, 17, 17, 0.15)');

  const fetchSystemInfo = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiClient.get('/system/info');
      const normalizedInfo = normalizeSystemInfo(response.data);
      setSystemInfo(normalizedInfo);
      setDefaultPrinter(normalizedInfo.printers.default || '');
    } catch (err: any) {
      setError(err.message || 'Failed to fetch system info');
    } finally {
      setLoading(false);
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

  useEffect(() => {
    fetchSystemInfo();
  }, [fetchSystemInfo]);

  // Print Queue handlers
  const fetchPrinterQueues = useCallback(async () => {
    setQueuesLoading(true);
    try {
      const { data } = await apiClient.get(API_ENDPOINTS.printerQueues);
      setPrinterQueues(data?.printers || []);
    } catch (err: any) {
      console.error('Failed to fetch printer queues', err);
      toast({
        title: 'Failed to fetch queues',
        description: err.response?.data?.error || err.message,
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setQueuesLoading(false);
    }
  }, [toast]);

  const handleClearPrintQueue = useCallback(async () => {
    setClearingQueue(true);
    try {
      const { data } = await apiClient.post(API_ENDPOINTS.printerClearQueue);
      toast({
        title: 'Print queue cleared',
        description: data?.message || 'All pending jobs removed',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
      fetchPrinterQueues();
    } catch (err: any) {
      toast({
        title: 'Failed to clear queue',
        description: err.response?.data?.error || err.message,
        status: 'error',
        duration: 4000,
        isClosable: true,
      });
    } finally {
      setClearingQueue(false);
    }
  }, [toast, fetchPrinterQueues]);

  const handleTerminateJob = useCallback(async (printerName: string, jobId: string) => {
    const key = `${printerName}_${jobId}`;
    setTerminatingJobId(key);
    try {
      await apiClient.post(API_ENDPOINTS.printerCancelJob, {
        printer_name: printerName,
        job_id: jobId,
      });
      toast({
        title: 'Print job cancelled',
        status: 'success',
        duration: 2000,
        isClosable: true,
      });
      fetchPrinterQueues();
    } catch (err: any) {
      toast({
        title: 'Failed to cancel job',
        description: err.response?.data?.error || err.message,
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setTerminatingJobId(null);
    }
  }, [toast, fetchPrinterQueues]);

  const handleOpenPrintQueueModal = () => {
    fetchPrinterQueues();
    printQueueModal.onOpen();
  };

  const totalJobs = printerQueues.reduce((sum, p) => sum + (p.jobs?.length || 0), 0);

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

  const handleOpenModal = () => {
    fetchSystemInfo();
    onOpen();
  };

  return (
    <>
      <ButtonGroup variant="outline" size="sm" spacing={2} flexWrap="wrap" rowGap={2}>
        <Button
          leftIcon={<Iconify icon={FiMonitor} />}
          color={textMuted}
          _hover={{ color: 'brand.400', bg: 'rgba(17, 17, 17, 0.1)' }}
          fontWeight="medium"
          onClick={handleOpenModal}
        >
          Device Info
        </Button>

        {onCheckConnectivity && (
          <Button
            colorScheme="cyan"
            variant="solid"
            onClick={onCheckConnectivity}
            leftIcon={<Iconify icon={FiWifiOff} boxSize={3} />}
            transition="all 0.3s"
          >
            Check Connectivity
          </Button>
        )}

        <Button
          colorScheme="pink"
          variant="ghost"
          onClick={handleOpenPrintQueueModal}
          leftIcon={<Iconify icon={FiPrinter} boxSize={3} />}
          _hover={{ bg: 'rgba(17, 17, 17, 0.06)' }}
          transition="all 0.3s"
        >
          Print Queue
        </Button>

        <Button
          colorScheme="purple"
          variant="ghost"
          onClick={calibrationModal.onOpen}
          leftIcon={<Iconify icon={FiSettings} boxSize={3} />}
          _hover={{ bg: 'rgba(17, 17, 17, 0.06)' }}
          transition="all 0.3s"
        >
          Calibrate
        </Button>
      </ButtonGroup>

      <CalibrationModal isOpen={calibrationModal.isOpen} onClose={calibrationModal.onClose} />

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
                System Status
              </Text>
              {systemInfo?.printers?.available && (
                <Tag size="lg" colorScheme="green" variant="subtle" borderRadius="full">
                  {systemInfo.printers.count} Printers
                </Tag>
              )}
            </HStack>
            <HStack spacing={2}>
              <Tooltip label="Refresh" placement="top">
                <IconButton
                  aria-label="Refresh"
                  icon={<Iconify icon={FiRefreshCw} boxSize={4} />}
                  size="sm"
                  variant="ghost"
                  color={textMuted}
                  _hover={{ color: 'brand.400', bg: 'rgba(17, 17, 17, 0.1)' }}
                  onClick={fetchSystemInfo}
                  isLoading={loading}
                />
              </Tooltip>
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
                {/* Printers Section - Top Priority */}
                <Box
                  bg={useColorModeValue('gray.50', 'rgba(17, 17, 17, 0.05)')}
                  borderRadius="xl"
                  p={5}
                  border="1px solid"
                  borderColor={borderColor}
                >
                  <HStack mb={4} spacing={3}>
                    <Iconify icon={FiPrinter} boxSize={5} color="brand.400" />
                    <Text fontWeight="700" fontSize="sm" textTransform="uppercase" letterSpacing="wider" color="brand.400">
                      Printers & Default Selection
                    </Text>
                  </HStack>

                  {systemInfo.printers.available ? (
                    <RadioGroup value={defaultPrinter} onChange={(value) => handleSetDefaultPrinter(value)}>
                      <VStack align="stretch" spacing={3}>
                        {systemInfo.printers.list.map((printer, idx) => (
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
                                <Tag size="sm" colorScheme={getStatusColor(printer.status)} variant="subtle">
                                  {printer.status}
                                </Tag>
                                {defaultPrinter === printer.name && (
                                  <Tag size="sm" colorScheme="brand" variant="solid" fontSize="11px">
                                    ACTIVE
                                  </Tag>
                                )}
                              </HStack>
                            </Flex>
                          </Box>
                        ))}
                      </VStack>
                    </RadioGroup>
                  ) : (
                    <Text fontSize="sm" color={textMuted} fontStyle="italic">
                      {systemInfo.printers.error || 'No printers found'}
                    </Text>
                  )}
                </Box>

                <Divider borderColor={borderColor} opacity={0.4} />

                {/* Driver Suggestions */}
                {systemInfo.driver_suggestions.length > 0 && (
                  <>
                    <Box>
                      <HStack mb={4} spacing={3}>
                        <Iconify icon={FiDownload} boxSize={5} color="brand.400" />
                        <Text fontWeight="700" fontSize="sm" textTransform="uppercase" letterSpacing="wider" color="brand.400">
                          Driver Downloads
                        </Text>
                      </HStack>
                      <SimpleGrid columns={{ base: 1, md: 2 }} spacing={3} w="100%">
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
                      </SimpleGrid>
                    </Box>
                    <Divider borderColor={borderColor} opacity={0.4} />
                  </>
                )}

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

      {/* Print Queue Modal */}
      <Modal isOpen={printQueueModal.isOpen} onClose={printQueueModal.onClose} size="3xl" isCentered scrollBehavior="inside">
        <ModalOverlay bg="blackAlpha.600" />
        <ModalContent
          bg={bgCard}
          borderRadius="2xl"
          border="1px solid"
          borderColor={borderColor}
          boxShadow={shadow}
          maxH="85vh"
        >
          <Flex
            borderBottom="1px solid"
            borderColor={borderColor}
            align="center"
            justify="space-between"
            py={4}
            px={6}
          >
            <Flex align="center" gap={3}>
              <Box
                p={2}
                borderRadius="xl"
                bg="pink.500"
                boxShadow="none"
              >
                <Iconify icon={FiPrinter} color="white" boxSize={5} />
              </Box>
              <Box>
                <Text fontWeight="bold" fontSize="lg" color={textColor}>
                  Print Queues
                </Text>
                <Text fontSize="xs" color={textMuted}>
                  Manage active print jobs
                </Text>
              </Box>
            </Flex>
            <Flex gap={2}>
              <Button
                size="sm"
                colorScheme="red"
                variant="solid"
                leftIcon={<Iconify icon={FiTrash2} boxSize={4} />}
                onClick={handleClearPrintQueue}
                isLoading={clearingQueue}
                loadingText="Clearing..."
              >
                Clear All Queues
              </Button>
            </Flex>
          </Flex>
          <ModalBody py={6} px={6}>
            {queuesLoading ? (
              <Flex justify="center" align="center" py={10}>
                <Spinner size="lg" color="pink.400" />
              </Flex>
            ) : printerQueues.length === 0 ? (
              <Flex direction="column" align="center" justify="center" py={10}>
                <Iconify icon={FiPrinter} boxSize={10} color={textMuted} mb={4} />
                <Text color={textMuted} fontSize="lg">No print queues found</Text>
                <Text color={textMuted} fontSize="sm">All queues are empty</Text>
              </Flex>
            ) : (
              <VStack spacing={4} align="stretch">
                {printerQueues.map((queue) => (
                  <Box
                    key={queue.name}
                    bg={useColorModeValue('gray.50', 'rgba(17, 17, 17, 0.04)')}
                    p={4}
                    borderRadius="xl"
                    border="1px solid"
                    borderColor={useColorModeValue('gray.200', 'whiteAlpha.100')}
                  >
                    <Flex justify="space-between" align="center" mb={3}>
                      <Flex align="center" gap={2}>
                        <Iconify icon={FiPrinter} color="pink.400" boxSize={5} />
                        <Text fontWeight="bold" color={textColor}>{queue.name}</Text>
                        <Tag size="sm" colorScheme={queue.status === 'Ready' ? 'green' : 'yellow'} variant="subtle">
                          {queue.status}
                        </Tag>
                      </Flex>
                      <Tag size="sm" colorScheme="pink" variant="solid">
                        {queue.jobs?.length || 0} jobs
                      </Tag>
                    </Flex>
                    
                    {queue.jobs && queue.jobs.length > 0 ? (
                      <VStack spacing={2} align="stretch">
                        {queue.jobs.map((job) => (
                          <Flex
                            key={job.id}
                            justify="space-between"
                            align="center"
                            bg={useColorModeValue('white', 'whiteAlpha.50')}
                            p={3}
                            borderRadius="lg"
                          >
                            <Box flex={1}>
                              <Text fontSize="sm" fontWeight="medium" color={textColor} isTruncated maxW="300px">
                                {job.document}
                              </Text>
                              <Flex gap={2} mt={1}>
                                <Text fontSize="xs" color={textMuted}>ID: {job.id}</Text>
                                <Text fontSize="xs" color={textMuted}>•</Text>
                                <Text fontSize="xs" color={textMuted}>{job.status}</Text>
                              </Flex>
                            </Box>
                            <Button
                              size="xs"
                              colorScheme="red"
                              variant="ghost"
                              onClick={() => handleTerminateJob(queue.name, String(job.id))}
                              isLoading={terminatingJobId === String(job.id)}
                              leftIcon={<Iconify icon={FiTrash2} boxSize={3} />}
                            >
                              Cancel
                            </Button>
                          </Flex>
                        ))}
                      </VStack>
                    ) : (
                      <Text fontSize="sm" color={textMuted} textAlign="center" py={2}>
                        No active jobs in queue
                      </Text>
                    )}
                  </Box>
                ))}
              </VStack>
            )}
          </ModalBody>
        </ModalContent>
      </Modal>
    </>
  );
};

export default DeviceAndConnectivityPanel;
