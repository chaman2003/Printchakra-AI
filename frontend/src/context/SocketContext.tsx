import React, { createContext, useContext, useEffect, useRef, useState, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import { API_BASE_URL, SOCKET_CONFIG, SOCKET_IO_ENABLED, ENVIRONMENT } from '../config';

interface SocketContextType {
  socket: Socket | null;
  connected: boolean;
  reconnect: () => void;
}

const SocketContext = createContext<SocketContextType>({
  socket: null,
  connected: false,
  reconnect: () => {},
});

export const useSocket = () => {
  const context = useContext(SocketContext);
  if (!context) {
    throw new Error('useSocket must be used within SocketProvider');
  }
  return context;
};

interface SocketProviderProps {
  children: React.ReactNode;
}

export const SocketProvider: React.FC<SocketProviderProps> = ({ children }) => {
  const [connected, setConnected] = useState(false);
  const socketRef = useRef<Socket | null>(null);
  const connectionAttemptRef = useRef(0);
  const maxRetries = 10;
  const initializingRef = useRef(false);

  const initSocket = useCallback(() => {
    if (!SOCKET_IO_ENABLED) {
      console.log('[Socket] Socket.IO disabled - using HTTP polling only');
      setConnected(true);
      return;
    }

    // Check if socket already exists globally (from previous mount)
    const existingSocket = (globalThis as any).__PRINTCHAKRA_SOCKET as Socket | null;

    // If socket exists and is still connected, reuse it
    if (existingSocket && existingSocket.connected) {
      console.log('[Socket] Reusing existing Socket.IO connection');
      socketRef.current = existingSocket;
      setConnected(true);
      return;
    }

    // If a socket instance exists but is not connected, try to reuse it instead of creating a new one
    if (existingSocket && !existingSocket.connected) {
      console.log('[Socket] Reusing existing Socket.IO instance and attempting to connect');
      socketRef.current = existingSocket;
      initializingRef.current = true;
      try {
        existingSocket.connect();
      } catch (e) {
        console.warn('[Socket] Failed to connect existing socket:', e);
        // fall through to create a fresh socket
      }
      return;
    }

    // Prevent multiple simultaneous initialization attempts (React StrictMode safety)
    if (initializingRef.current) {
      console.log('[Socket] Connection attempt already in progress');
      return;
    }

    initializingRef.current = true;
    console.log('[Socket] Initializing Socket.IO connection');
    console.log(`[Socket] Environment: ${ENVIRONMENT}`);
    console.log(`[Socket] Backend URL: ${API_BASE_URL}`);

    try {
      // Support both websocket and polling to handle all network conditions
      // (local, HTTPS with self-signed certs, ngrok, deployed)
      const newSocket = io(API_BASE_URL, {
        ...SOCKET_CONFIG,
        transports: ['websocket', 'polling'],
        upgrade: true,
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        reconnectionAttempts: 5,
        timeout: 20000,
      });

      newSocket.on('connect', () => {
        console.log('[Socket] Connected to server');
        socketRef.current = newSocket;
        setConnected(true);
        connectionAttemptRef.current = 0;
        initializingRef.current = false;
      });

      newSocket.on('disconnect', (reason: string) => {
        console.log('[Socket] Disconnected:', reason);
        setConnected(false);
        
        // Auto-reconnect for certain disconnect reasons
        if (reason === 'io server disconnect' || reason === 'transport close') {
          console.log('[Socket] Attempting to reconnect...');
          setTimeout(() => {
            if (!socketRef.current?.connected) {
              newSocket.connect();
            }
          }, 2000);
        }
      });

      newSocket.on('reconnect', (attemptNumber: number) => {
        console.log('[Socket] Reconnected after', attemptNumber, 'attempts');
        setConnected(true);
      });

      newSocket.on('reconnect_attempt', (attemptNumber: number) => {
        console.log('[Socket] Reconnection attempt', attemptNumber);
      });

      newSocket.on('connect_error', (error: any) => {
        connectionAttemptRef.current++;
        console.warn('[Socket] Connection error:', error.message || error);
        setConnected(false);
        if (error.data) {
          console.warn('[Socket] Error details:', error.data);
        }
        if (connectionAttemptRef.current >= maxRetries) {
          console.error('[Socket] Max connection attempts reached');
          initializingRef.current = false;
        }
      });

      newSocket.on('error', (error: any) => {
        console.error('[Socket] Socket error:', error);
      });

      socketRef.current = newSocket;
      
      // Store globally so remounted providers can reuse the same socket
      try {
        (globalThis as any).__PRINTCHAKRA_SOCKET = newSocket;
      } catch (e) {
        console.warn('[Socket] Unable to store socket globally:', e);
      }
    } catch (err) {
      console.error('[Socket] Failed to initialize:', err);
      setConnected(false);
      initializingRef.current = false;
    }
  }, []);

  const reconnect = useCallback(() => {
    console.log('[Socket] Manual reconnect requested');
    initializingRef.current = false;
    connectionAttemptRef.current = 0;
    
    if (socketRef.current) {
      socketRef.current.disconnect();
    }
    
    setTimeout(() => {
      initSocket();
    }, 500);
  }, [initSocket]);

  useEffect(() => {
    initSocket();

    // Cleanup: Only disconnect on actual unmount, not on React StrictMode remount
    return () => {
      // Don't disconnect - keep the connection alive for reuse
    };
  }, [initSocket]);

  // Periodic connection check - if disconnected, try to reconnect
  useEffect(() => {
    const checkConnection = setInterval(() => {
      if (!connected && socketRef.current && !socketRef.current.connected) {
        console.log('[Socket] Connection lost, attempting reconnect...');
        socketRef.current.connect();
      }
    }, 5000);

    return () => clearInterval(checkConnection);
  }, [connected]);

  return (
    <SocketContext.Provider value={{ socket: socketRef.current, connected, reconnect }}>
      {children}
    </SocketContext.Provider>
  );
};

export default SocketContext;
