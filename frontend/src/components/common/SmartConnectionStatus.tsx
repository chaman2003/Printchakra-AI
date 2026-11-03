import React, { useState, useCallback, useImperativeHandle, forwardRef, useEffect } from 'react';
import {
  Box,
  Button,
  Flex,
  Spinner,
  Stack,
  Text,
  VStack,
  HStack,
  Badge,
  Tooltip,
  useColorModeValue,
  Progress,
} from '@chakra-ui/react';
import { FiWifi, FiCheckCircle, FiAlertCircle, FiRefreshCw, FiCamera, FiPrinter } from 'react-icons/fi';
import Iconify from './Iconify';
import apiClient from '../../apiClient';

// Session storage key for tracking if connectivity check has been done
const SESSION_CHECK_KEY = 'printchakra_connectivity_checked';

interface ConnectionStatus {
  phoneWiFi: 'idle' | 'checking' | 'connected' | 'failed';
  phoneCamera: 'idle' | 'checking' | 'connected' | 'failed';
  laptopPrinter: 'idle' | 'checking' | 'connected' | 'failed';
}

interface ConnectionDetails {
  phoneWiFi: string;
  phoneCamera: string;
  laptopPrinter: string;
}

interface SavedConnectionResult {
  status: ConnectionStatus;
  details: ConnectionDetails;
  overallStatus: 'idle' | 'checking' | 'success' | 'failed';
  timestamp: number;
}

export interface SmartConnectionStatusHandle {
  runCheck: () => Promise<void>;
  forceRecheck: () => Promise<void>;
}

interface SmartConnectionStatusProps {
  onStatusComplete?: (allConnected: boolean) => void;
  isOpen?: boolean;
  onClose?: () => void;
  videoRef?: React.RefObject<HTMLVideoElement | null>;
  isCapturing?: boolean;
  variant?: 'card' | 'minimal';
  autoRun?: boolean;
}

const SmartConnectionStatus = forwardRef<SmartConnectionStatusHandle, SmartConnectionStatusProps>(({ 
  onStatusComplete, 
  isOpen = true, 
  onClose, 
  videoRef,
  isCapturing = false,
  variant = 'card',
  autoRun = false
}, ref) => {
  const [status, setStatus] = useState<ConnectionStatus>({
    phoneWiFi: 'idle',
    phoneCamera: 'idle',
    laptopPrinter: 'idle',
  });
  const [details, setDetails] = useState<ConnectionDetails>({
    phoneWiFi: 'Awaiting check',
    phoneCamera: 'Awaiting check',
    laptopPrinter: 'Awaiting check',
  });
  const [isChecking, setIsChecking] = useState(false);
  const [overallStatus, setOverallStatus] = useState<'idle' | 'checking' | 'success' | 'failed'>(
    'idle'
  );
  const [checkProgress, setCheckProgress] = useState(0);
  const [alreadyCheckedThisSession, setAlreadyCheckedThisSession] = useState(false);

  const bgCard = useColorModeValue(
    'rgba(255, 255, 255, 0.7)',
    'rgba(15, 20, 35, 0.7)'
  );
  const borderColor = useColorModeValue('rgba(17, 17, 17, 0.15)', 'rgba(17, 17, 17, 0.15)');
  const isMinimal = variant === 'minimal';

  // Load saved session result on mount
  useEffect(() => {
    try {
      const saved = sessionStorage.getItem(SESSION_CHECK_KEY);
      if (saved) {
        const result: SavedConnectionResult = JSON.parse(saved);
        // Only use cached result if it's less than 30 minutes old
        const age = Date.now() - result.timestamp;
        if (age < 30 * 60 * 1000) {
          setStatus(result.status);
          setDetails(result.details);
          setOverallStatus(result.overallStatus);
          setAlreadyCheckedThisSession(true);
          if (onStatusComplete) {
            onStatusComplete(result.overallStatus === 'success');
          }
        }
      }
    } catch (e) {
      console.warn('Failed to load session connectivity result:', e);
    }
  }, []);

  // Save result to session storage
  const saveToSession = useCallback((newStatus: ConnectionStatus, newDetails: ConnectionDetails, newOverall: 'success' | 'failed') => {
    try {
      const result: SavedConnectionResult = {
        status: newStatus,
        details: newDetails,
        overallStatus: newOverall,
        timestamp: Date.now(),
      };
      sessionStorage.setItem(SESSION_CHECK_KEY, JSON.stringify(result));
      setAlreadyCheckedThisSession(true);
    } catch (e) {
      console.warn('Failed to save session connectivity result:', e);
    }
  }, []);

  const runSequentialConnectionCheck = useCallback(async () => {
    // Always run - no conditions, no skipping
    console.log('[ConnectionCheck] Starting connection check...');
    setIsChecking(true);
    setOverallStatus('checking');
    setCheckProgress(0);

    let finalStatus: ConnectionStatus = {
      phoneWiFi: 'idle',
      phoneCamera: 'idle',
      laptopPrinter: 'idle',
    };
    let finalDetails: ConnectionDetails = {
      phoneWiFi: 'Awaiting check',
      phoneCamera: 'Awaiting check',
      laptopPrinter: 'Awaiting check',
    };

    try {
      let allConnected = true;

      // STEP 1: Phone ↔ Laptop WiFi Connection
      setCheckProgress(25);
      setStatus(prev => ({ ...prev, phoneWiFi: 'checking' }));
      setDetails(prev => ({ ...prev, phoneWiFi: 'Validating WiFi link...' }));

      try {
        const wifiResponse = await apiClient.post('/connection/validate-wifi', {
          timestamp: Date.now(),
        });
        
        const wifiConnected = wifiResponse?.data?.connected;
        finalStatus.phoneWiFi = wifiConnected ? 'connected' : 'failed';
        finalDetails.phoneWiFi = wifiConnected
          ? `✅ Same network (${wifiResponse?.data?.ip || 'Active'})`
          : wifiResponse?.data?.message || '❌ Not on same WiFi';
        setStatus(prev => ({ ...prev, phoneWiFi: finalStatus.phoneWiFi }));
        setDetails(prev => ({ ...prev, phoneWiFi: finalDetails.phoneWiFi }));
        allConnected = allConnected && wifiConnected;
      } catch (err: any) {
        finalStatus.phoneWiFi = 'failed';
        finalDetails.phoneWiFi = '❌ WiFi check failed';
        setStatus(prev => ({ ...prev, phoneWiFi: 'failed' }));
        setDetails(prev => ({ ...prev, phoneWiFi: '❌ WiFi check failed' }));
        allConnected = false;
      }

      // Small delay between checks for visual effect
      await new Promise(resolve => setTimeout(resolve, 500));

      // STEP 2: Phone Camera Capturing Validation
      setCheckProgress(50);
      setStatus(prev => ({ ...prev, phoneCamera: 'checking' }));
      setDetails(prev => ({ ...prev, phoneCamera: 'Validating camera stream...' }));

      try {
        const cameraResponse = await apiClient.post('/connection/validate-camera', {
          isCapturing: isCapturing || (videoRef?.current?.srcObject !== null),
          timestamp: Date.now(),
        });
        
        const cameraActive = cameraResponse?.data?.capturing;
        finalStatus.phoneCamera = cameraActive ? 'connected' : 'failed';
        finalDetails.phoneCamera = cameraActive
          ? '✅ Camera capturing frames'
          : '❌ Camera not active';
        setStatus(prev => ({ ...prev, phoneCamera: finalStatus.phoneCamera }));
        setDetails(prev => ({ ...prev, phoneCamera: finalDetails.phoneCamera }));
        allConnected = allConnected && cameraActive;
      } catch (err: any) {
        finalStatus.phoneCamera = 'failed';
        finalDetails.phoneCamera = '❌ Camera check failed';
        setStatus(prev => ({ ...prev, phoneCamera: 'failed' }));
        setDetails(prev => ({ ...prev, phoneCamera: '❌ Camera check failed' }));
        allConnected = false;
      }

      // Small delay between checks for visual effect
      await new Promise(resolve => setTimeout(resolve, 500));

      // STEP 3: Laptop ↔ Printer Connection (Auto-print blank page)
      setCheckProgress(75);
      setStatus(prev => ({ ...prev, laptopPrinter: 'checking' }));
      setDetails(prev => ({ ...prev, laptopPrinter: 'Testing printer (printing blank page)...' }));

      try {
        const printerResponse = await apiClient.post('/connection/validate-printer', {
          testPrint: true,  // This will print a blank test page
          timestamp: Date.now(),
        });
        
        const printerReady = printerResponse?.data?.connected;
        finalStatus.laptopPrinter = printerReady ? 'connected' : 'failed';
        finalDetails.laptopPrinter = printerReady
          ? `✅ Printer ready (${printerResponse?.data?.model || 'Connected'}) - Test page sent`
          : printerResponse?.data?.message || '❌ Printer not responding';
        setStatus(prev => ({ ...prev, laptopPrinter: finalStatus.laptopPrinter }));
        setDetails(prev => ({ ...prev, laptopPrinter: finalDetails.laptopPrinter }));
        allConnected = allConnected && printerReady;
      } catch (err: any) {
        finalStatus.laptopPrinter = 'failed';
        finalDetails.laptopPrinter = '❌ Printer check failed';
        setStatus(prev => ({ ...prev, laptopPrinter: 'failed' }));
        setDetails(prev => ({ ...prev, laptopPrinter: '❌ Printer check failed' }));
        allConnected = false;
      }

      // Final status
      setCheckProgress(100);
      const finalOverall = allConnected ? 'success' : 'failed';
      setOverallStatus(finalOverall);

      // Save to session storage so we don't re-run
      saveToSession(finalStatus, finalDetails, finalOverall);

      if (onStatusComplete) {
        onStatusComplete(allConnected);
      }
    } catch (error) {
      console.error('Sequential connection check failed:', error);
      setOverallStatus('failed');
      setStatus({
        phoneWiFi: 'failed',
        phoneCamera: 'failed',
        laptopPrinter: 'failed',
      });
    } finally {
      setIsChecking(false);
    }
  }, [isCapturing, videoRef, onStatusComplete, saveToSession]);

  React.useEffect(() => {
    // Don't auto-run - only run when user explicitly clicks the button
  }, [autoRun, isOpen, runSequentialConnectionCheck]);

  useImperativeHandle(ref, () => ({
    runCheck: () => runSequentialConnectionCheck(),
    forceRecheck: () => runSequentialConnectionCheck(),
  }));

  if (!isOpen && !onClose) {
    return null;
  }

  const content = (
    <Stack spacing={2} position="relative" zIndex={1} w="full">
      {/* Header - Only show in card mode */}
      {!isMinimal && (
        <Flex justify="space-between" align="flex-start" gap={2}>
          <VStack align="flex-start" spacing={0} flex={1}>
            <Text fontSize="sm" fontWeight="600">
              🔗 System Connection
            </Text>
            <Text fontSize="xs" color="text.muted">
              {overallStatus === 'idle' && 'Ready to validate'}
              {overallStatus === 'checking' && 'Validating connections...'}
              {overallStatus === 'success' && '✅ All systems connected'}
              {overallStatus === 'failed' && '⚠️ Issues detected'}
            </Text>
          </VStack>
          <HStack gap={1}>
            {isChecking && <Spinner size="xs" color="brand.400" />}
            {onClose && (
              <Button size="xs" variant="ghost" onClick={onClose} p={0} minW="auto">
                ✕
              </Button>
            )}
          </HStack>
        </Flex>
      )}

      {/* Progress Bar */}
      {isChecking && (
        <Progress value={checkProgress} size="sm" borderRadius="full" colorScheme="brand" />
      )}

      {/* Connection Items */}
      <VStack spacing={2} align="stretch">
        <ConnectionCheckItem
          label="Phone ↔ Laptop"
          icon={FiWifi}
          status={status.phoneWiFi}
          details={details.phoneWiFi}
        />
        <ConnectionCheckItem
          label="Phone Camera"
          icon={FiCamera}
          status={status.phoneCamera}
          details={details.phoneCamera}
        />
        <ConnectionCheckItem
          label="Laptop ↔ Printer"
          icon={FiRefreshCw}
          status={status.laptopPrinter}
          details={details.laptopPrinter}
        />
      </VStack>

      {/* Validate Button - Only show in card mode */}
      {!isMinimal && (
        <Flex gap={2} justify="flex-end" pt={1}>
          <Button
            size="sm"
            colorScheme="brand"
            isLoading={isChecking}
            loadingText="Validating..."
            onClick={() => runSequentialConnectionCheck()}
            leftIcon={<Iconify icon={FiRefreshCw} boxSize={3} />}
            fontSize="xs"
            isDisabled={isChecking}
          >
            {alreadyCheckedThisSession ? 'Re-check' : 'Validate'}
          </Button>
          {overallStatus === 'success' && (
            <Badge
              colorScheme="green"
              p={1}
              borderRadius="md"
              fontSize="xs"
              display="flex"
              alignItems="center"
              gap={1}
            >
              <Iconify icon={FiCheckCircle} boxSize={3} />
              Ready
            </Badge>
          )}
        </Flex>
      )}
    </Stack>
  );

  if (isMinimal) {
    return (
      <Stack spacing={2} w="full">
        {content}
        {overallStatus === 'failed' && (
          <Box
            bg="rgba(245, 101, 101, 0.05)"
            border="1px solid rgba(245, 101, 101, 0.15)"
            borderRadius="md"
            p={2}
            fontSize="xs"
          >
            <VStack align="flex-start" spacing={1}>
              <Text fontWeight="600" color="red.300" fontSize="xs">
                ⚠️ Connection Failed
              </Text>
              <VStack align="flex-start" spacing={0.5} pl={3} fontSize="xs">
                {status.phoneWiFi === 'failed' && (
                  <Text>• Check if phone is on same WiFi network</Text>
                )}
                {status.phoneCamera === 'failed' && (
                  <Text>• Ensure camera is actively capturing</Text>
                )}
                {status.laptopPrinter === 'failed' && (
                  <Text>• Verify printer is powered and online</Text>
                )}
              </VStack>
            </VStack>
          </Box>
        )}
      </Stack>
    );
  }

  return (
    <Stack spacing={2} w="full">
      <Box
        bg={bgCard}
        border={`1px solid ${borderColor}`}
        borderRadius="2xl"
        p={3}
        position="relative"
        overflow="hidden"
        boxShadow="0 4px 20px rgba(0, 0, 0, 0.08)"
        _before={{
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background:
            overallStatus === 'success'
              ? 'linear-gradient(135deg, rgba(72, 187, 120, 0.05) 0%, rgba(72, 187, 120, 0) 100%)'
              : overallStatus === 'failed'
                ? 'linear-gradient(135deg, rgba(245, 101, 101, 0.05) 0%, rgba(245, 101, 101, 0) 100%)'
                : 'transparent',
          pointerEvents: 'none',
        }}
      >
        {content}
      </Box>

      {overallStatus === 'failed' && (
        <Box
          bg="rgba(245, 101, 101, 0.05)"
          border="1px solid rgba(245, 101, 101, 0.15)"
          borderRadius="md"
          p={2}
          fontSize="xs"
        >
          <VStack align="flex-start" spacing={1}>
            <Text fontWeight="600" color="red.300" fontSize="xs">
              ⚠️ Connection Failed
            </Text>
            <VStack align="flex-start" spacing={0.5} pl={3} fontSize="xs">
              {status.phoneWiFi === 'failed' && (
                <Text>• Check if phone is on same WiFi network</Text>
              )}
              {status.phoneCamera === 'failed' && (
                <Text>• Ensure camera is actively capturing</Text>
              )}
              {status.laptopPrinter === 'failed' && (
                <Text>• Verify printer is powered and online</Text>
              )}
            </VStack>
          </VStack>
        </Box>
      )}
    </Stack>
  );
});

// New simplified connection check item component
const ConnectionCheckItem: React.FC<{
  label: string;
  icon: any;
  status: 'idle' | 'checking' | 'connected' | 'failed';
  details: string;
}> = ({ label, icon, status, details }) => {
  const statusColor = status === 'connected' ? 'green.400' : status === 'failed' ? 'red.400' : 'yellow.400';
  const statusBg = status === 'connected' ? 'green' : status === 'failed' ? 'red' : status === 'checking' ? 'yellow' : 'gray';
  
  const itemBg = useColorModeValue('rgba(17, 17, 17, 0.04)', 'rgba(255, 255, 255, 0.03)');
  const itemBorder = useColorModeValue('rgba(17, 17, 17, 0.08)', 'rgba(255, 255, 255, 0.05)');

  return (
    <HStack 
      spacing={3} 
      p={2} 
      borderRadius="xl" 
      bg={itemBg} 
      border="1px solid"
      borderColor={itemBorder}
      w="full"
      transition="all 0.2s ease"
      _hover={{ transform: 'translateX(2px)', bg: useColorModeValue('rgba(17, 17, 17, 0.06)', 'rgba(255, 255, 255, 0.05)') }}
    >
      <Box 
        position="relative" 
        w="8" 
        h="8" 
        display="flex" 
        alignItems="center" 
        justifyContent="center"
        bg={useColorModeValue('white', 'whiteAlpha.100')}
        borderRadius="lg"
        boxShadow="sm"
      >
        {status === 'checking' ? (
          <Spinner size="xs" color="brand.400" />
        ) : status === 'connected' ? (
          <Iconify icon={FiCheckCircle} boxSize={5} color="green.400" />
        ) : status === 'failed' ? (
          <Iconify icon={FiAlertCircle} boxSize={5} color="red.400" />
        ) : (
          <Iconify icon={icon} boxSize={5} color="gray.400" />
        )}
      </Box>
      <VStack align="flex-start" spacing={0} flex={1} minW={0}>
        <Text fontSize="xs" fontWeight="700" letterSpacing="-0.01em">
          {label}
        </Text>
        <Text fontSize="10px" color="text.muted" noOfLines={1} fontWeight="500">
          {details}
        </Text>
      </VStack>
      <Badge 
        colorScheme={statusBg} 
        variant="subtle" 
        fontSize="9px" 
        fontWeight="800"
        py={0.5} 
        px={2}
        borderRadius="full"
        textTransform="uppercase"
      >
        {status === 'checking' ? 'Validating' : status === 'connected' ? 'Ready' : status === 'failed' ? 'Failed' : 'Idle'}
      </Badge>
    </HStack>
  );
};

export default SmartConnectionStatus;
