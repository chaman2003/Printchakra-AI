import axios from 'axios';
import { API_BASE_URL, getDefaultHeaders } from './config';
import { authHeader, notifyExpired } from './auth/session';

// Create axios instance with default configuration
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // Increased to 120 seconds for long-running OCR operations
  headers: getDefaultHeaders(),
});

// Request interceptor to ensure headers are always present
apiClient.interceptors.request.use(
  config => {
    // Merge with default headers
    const defaultHeaders = { ...getDefaultHeaders(), ...authHeader() };

    // Safely merge headers
    Object.keys(defaultHeaders).forEach(key => {
      config.headers.set(key, defaultHeaders[key]);
    });

    return config;
  },
  error => {
    console.error('Request interceptor error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  response => {
    console.log(
      `API Response: ${response.config.method?.toUpperCase()} ${response.config.url}`,
      {
        status: response.status,
      }
    );
    return response;
  },
  error => {
    // A rejected token (expired, or the account changed) ends the session. Login attempts are excluded
    // so a wrong password does not log anyone out.
    if (error.response?.status === 401 && !String(error.config?.url || '').startsWith('/auth/')) {
      notifyExpired();
    }
    if (error.response) {
      console.error(`API Error Response: ${error.response.status}`, {
        url: error.config?.url,
        data: error.response.data,
      });
    } else if (error.request) {
      console.error('API No Response:', {
        url: error.config?.url,
        message: error.message,
      });
    } else {
      console.error('API Request Setup Error:', error.message);
    }
    return Promise.reject(error);
  }
);

export default apiClient;
