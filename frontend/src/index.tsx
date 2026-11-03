import React from 'react';
import ReactDOM from 'react-dom/client';
import { ChakraProvider, ColorModeScript } from '@chakra-ui/react';
import App from './App';
import reportWebVitals from './reportWebVitals';
import theme from './theme';
import '@fontsource/instrument-serif/latin-400-italic.css';
import { API_BASE_URL } from './config';
import { installAuthFetch, setApiBase } from './auth/session';
import './index.css';

// Every request to the API carries the login token.
setApiBase(API_BASE_URL);
installAuthFetch();

// The design system is light only: ignore any color mode saved by earlier versions.
const colorModeManager = { type: 'localStorage' as const, get: () => 'light' as const, set: () => undefined };

// Suppress webpack dev server deprecation warnings (non-critical)
if (process.env.NODE_ENV === 'development') {
  const originalWarn = console.warn;
  console.warn = function (...args: any[]) {
    const message = args[0]?.toString?.() || '';
    if (
      message.includes('DEP_WEBPACK_DEV_SERVER') ||
      message.includes('onAfterSetupMiddleware') ||
      message.includes('onBeforeSetupMiddleware') ||
      message.includes('setupMiddlewares')
    ) {
      return;
    }
    return originalWarn.apply(console, args);
  };
}

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);

// Note: StrictMode is disabled in development to prevent double-mounting
// which causes multiple Socket.IO connections. In production, StrictMode
// would help detect side effects but is not needed since we don't use it
// in the production build anyway.
root.render(
  <ChakraProvider theme={theme} colorModeManager={colorModeManager}>
    <ColorModeScript initialColorMode={theme.config.initialColorMode} />
    <App />
  </ChakraProvider>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();

