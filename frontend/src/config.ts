import { withToken } from './auth/session';
// Backend API Configuration
// Automatically detects and routes to the correct backend:
// - Local development: localhost:5000
// - Deployed/ngrok: Same host or environment variable
// - Fallback: Environment variable REACT_APP_API_URL

const getApiBaseUrl = () => {
  // Priority 1: Explicit environment variable (highest priority)
  if (process.env.REACT_APP_API_URL) {
    console.log('[Config] Using REACT_APP_API_URL:', process.env.REACT_APP_API_URL);
    return process.env.REACT_APP_API_URL;
  }

  // Priority 2: Check if running locally (development)
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    console.log('[Config] Detected local development');
    return 'http://localhost:5000';
  }

  // Priority 3: Local network IP (e.g., 192.168.x.x) — backend runs on port 5000
  const ipPattern = /^(\d{1,3}\.){3}\d{1,3}$/;
  if (ipPattern.test(window.location.hostname)) {
    const protocol = window.location.protocol;
    const localUrl = `${protocol}//${window.location.hostname}:5000`;
    console.log('[Config] Detected local network IP, using backend port 5000:', localUrl);
    return localUrl;
  }

  // Priority 4: Same host as frontend (deployed on same server/ngrok)
  const protocol = window.location.protocol;
  const hostname = window.location.hostname;
  const port = window.location.port ? `:${window.location.port}` : '';
  const sameHostUrl = `${protocol}//${hostname}${port}`;

  console.log('[Config] Detected deployed environment, using same host:', sameHostUrl);
  return sameHostUrl;
};

export const API_BASE_URL = getApiBaseUrl();

// 'poll' = realtime events over HTTP polling (serverless hosting such as Vercel).
// Anything else = Socket.IO (local development against the Flask server).
export const REALTIME_MODE: 'poll' | 'socketio' =
  process.env.REACT_APP_REALTIME === 'poll' ? 'poll' : 'socketio';

// Determine environment type for logging/debugging
export const ENVIRONMENT = (() => {
  if (process.env.REACT_APP_API_URL) {
    return 'custom-url';
  }
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    return 'development';
  }
  return 'deployed';
})();

// Check if Socket.IO is available
const isSocketIOEnabled = () => {
  // Enable Socket.IO everywhere - let transport fallback handle issues
  // This allows real-time updates on both local and deployed environments
  return true;
};

export const SOCKET_IO_ENABLED = isSocketIOEnabled();

// Check if using ngrok or similar tunnel service (needs bypass header)
const isUsingTunnel = () => {
  return (
    API_BASE_URL.includes('ngrok') ||
    API_BASE_URL.includes('loca.lt') ||
    API_BASE_URL.includes('.ngrok') ||
    API_BASE_URL.includes('vercel') ||
    API_BASE_URL.includes('heroku')
  );
};

// Get default headers for axios requests
export const getDefaultHeaders = () => {
  const headers: Record<string, string> = {};

  // Add tunnel bypass header if using ngrok/similar
  if (isUsingTunnel()) {
    headers['ngrok-skip-browser-warning'] = 'true';
  }

  return headers;
};

// Separate image base URL for better reliability
export const getImageUrl = (endpoint: string, filename: string) => {
  const baseUrl = API_BASE_URL;
  const fullUrl = `${baseUrl}${endpoint}/${filename}`;

  // Add timestamp to bypass cache
  const separator = fullUrl.includes('?') ? '&' : '?';
  return withToken(`${fullUrl}${separator}_t=${Date.now()}`);
};

// Socket.IO specific configuration - handles both local and deployed
// NOTE: For self-signed SSL certificates (local HTTPS), we need rejectUnauthorized: false
const isSecure = API_BASE_URL.startsWith('https');
const isLocalNetwork = (() => {
  const hostname = window.location.hostname;
  // Check if IP address (local network) vs domain name
  const ipPattern = /^(\d{1,3}\.){3}\d{1,3}$/;
  return ipPattern.test(hostname) && hostname !== '127.0.0.1';
})();

export const SOCKET_CONFIG = {
  reconnection: true,
  reconnectionDelay: 1000,
  reconnectionDelayMax: 5000,
  reconnectionAttempts: 10,
  timeout: 15000,
  transports: ['polling', 'websocket'] as ('polling' | 'websocket')[],
  upgrade: true,
  forceNew: false,
  path: '/socket.io/',
  withCredentials: true, // Needed for deployed environments
  secure: isSecure,
  // CRITICAL: For self-signed SSL certs on local network, don't reject unauthorized
  rejectUnauthorized: isLocalNetwork && isSecure ? false : true,
  autoConnect: true,
  ...(isUsingTunnel()
    ? {
      extraHeaders: {
        'ngrok-skip-browser-warning': 'true',
      },
    }
    : {}),
};

export const API_ENDPOINTS = {
  // Core endpoints - Match backend exactly
  health: '/health',
  upload: '/phone/upload',
  files: '/files',
  processed: '/document/processed',
  uploads: '/public/uploads',
  delete: '/document/files',
  ocr: '/ocr',
  print: '/print',
  printerClearQueue: '/printer/clear-queue',
  printerQueues: '/printer/queues',
  printerCancelJob: '/printer/cancel-job',
  processingStatus: '/processing-status',

  // Advanced processing endpoints
  processAdvanced: '/process/advanced',
  validateQuality: '/validate/quality',
  detectDocument: '/detect/document',
  exportPdf: '/export/pdf',
  pdf: '/pdf',
  pipelineInfo: '/pipeline/info',
  classifyDocument: '/classify/document',
  batchProcess: '/batch/process',

  // File conversion endpoints (prefix with document blueprint)
  convert: '/document/convert',
  converted: '/document/converted',
  getConvertedFiles: '/document/converted',
  deleteConverted: '/document/converted',

  // Folder management endpoints
  folders: '/document/folders',
  folderFiles: '/document/folders', // append /<name>/files
  folderMove: '/document/folders',  // append /<name>/move

  // Image editing endpoints
  editCrop: '/document/edit/crop',
  editRotate: '/document/edit/rotate',
  editAdjust: '/document/edit/adjust',
  editEnhance: '/document/edit/enhance',

  // AI Orchestration endpoints
  orchestrateCommand: '/orchestrate/command',
  orchestrateConfirm: '/orchestrate/confirm',
  orchestrateCancel: '/orchestrate/cancel',
  orchestrateStatus: '/orchestrate/status',
  orchestrateDocuments: '/orchestrate/documents',
  orchestrateSelect: '/orchestrate/select',
  orchestrateConfigure: '/orchestrate/configure',
  orchestrateReset: '/orchestrate/reset',
  orchestrateHistory: '/orchestrate/history',
};

console.log('API Configuration:', {
  API_BASE_URL,
  hostname: window.location.hostname,
  SOCKET_IO_ENABLED,
  endpoints: Object.keys(API_ENDPOINTS).length + ' endpoints configured',
});
