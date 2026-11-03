import React from 'react';
import { BrowserRouter as Router, Navigate, NavLink, Outlet, Route, Routes, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Box, Flex, IconButton, Text, Tooltip } from '@chakra-ui/react';
import { LuX } from 'react-icons/lu';
import Dashboard from './pages/Dashboard';
import Phone from './pages/Phone';
import Playground from './pages/Playground';
import LandingPage from './pages/LandingPage';
import LoginPage from './pages/auth/LoginPage';
import SignupPage from './pages/auth/SignupPage';
import { SocketProvider } from './context/SocketContext';
import { CalibrationProvider } from './context/CalibrationContext';
import { AIWorkspaceProvider, useAIWorkspace } from './context/AIWorkspaceContext';
import { AuthProvider } from './auth/AuthContext';
import { RedirectIfAuthed, RequireAuth } from './auth/RequireAuth';
import { SiteHeader, HEADER_HEIGHT } from './components/layout/SiteHeader';
import { GeminiLiveVoice } from './components';
import VoiceCommandBridge from './aiassist/VoiceCommandBridge';

const CloseIcon = LuX as any;

const NAV_ITEMS = [
  { path: '/dashboard', label: 'Dashboard' },
  { path: '/playground', label: 'Playground' },
  { path: '/phone', label: 'Phone' },
];

// Mobile bottom navigation for the signed-in app.
function MobileBottomNav() {
  return (
    <Box
      as="nav"
      aria-label="App"
      display={{ base: 'block', md: 'none' }}
      position="fixed"
      bottom={0}
      left={0}
      right={0}
      zIndex={999}
      bg="rgba(242, 242, 242, 0.94)"
      backdropFilter="blur(12px)"
      borderTop="1px solid rgba(30, 30, 30, 0.1)"
      pb="env(safe-area-inset-bottom)"
    >
      <Flex>
        {NAV_ITEMS.map((item) => (
          <NavLink key={item.path} to={item.path} style={{ flex: 1 }}>
            {({ isActive }) => (
              <Flex
                h="56px"
                align="center"
                justify="center"
                fontSize="12px"
                fontWeight={700}
                letterSpacing="0.12em"
                textTransform="uppercase"
                color={isActive ? '#111111' : '#838282'}
                borderTop="2px solid"
                borderColor={isActive ? '#111111' : 'transparent'}
              >
                {item.label}
              </Flex>
            )}
          </NavLink>
        ))}
      </Flex>
    </Box>
  );
}

const pageVariants = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.3, ease: [0.4, 0, 0.2, 1] } },
};

function PageTransition({ children }: { children: React.ReactNode }) {
  return (
    <motion.div variants={pageVariants} initial="initial" animate="animate" style={{ width: '100%' }}>
      {children}
    </motion.div>
  );
}

/** Signed-in layout: header, page, docked voice assistant. */
function AppLayout() {
  const location = useLocation();
  const { isPanelOpen, panelWidth, isResizing, startResize, setPanelOpen } = useAIWorkspace();

  return (
    <Box minH="100vh" bg="#f2f2f2" position="relative">
      <SiteHeader mode="app" />
      <VoiceCommandBridge />

      <Box
        as="main"
        py={{ base: 3, md: 5 }}
        pb={{ base: '80px', md: 8 }}
        px={{ base: 2, md: 4 }}
        mr={isPanelOpen ? { base: 0, lg: `${panelWidth + 16}px` } : 0}
        transition={isResizing ? 'none' : 'margin-right 0.3s ease-out'}
      >
        <PageTransition key={location.pathname}>
          <Outlet />
        </PageTransition>
      </Box>

      <MobileBottomNav />

      {/* Docked voice assistant, kept across pages */}
      {isPanelOpen && (
        <Box
          position="fixed"
          top={{ base: 0, md: `${HEADER_HEIGHT}px` }}
          bottom="0"
          right="0"
          w={{ base: '100%', lg: `${panelWidth}px` }}
          bg="rgba(255, 255, 255, 0.97)"
          backdropFilter="blur(12px)"
          borderLeft="1px solid rgba(30, 30, 30, 0.1)"
          display="flex"
          flexDirection="column"
          overflowY="auto"
          zIndex={2000}
          transition={isResizing ? 'none' : 'transform 0.3s ease-out'}
        >
          <Flex align="center" justify="space-between" px={4} py={3} borderBottom="1px solid rgba(30, 30, 30, 0.1)">
            <Text fontFamily="heading" fontSize="18px" fontWeight={700} letterSpacing="-0.03em" sx={{ wordSpacing: '0.14em' }}>
              Voice assistant
            </Text>
            <Tooltip label="Close panel" hasArrow>
              <IconButton
                aria-label="Close voice panel"
                size="sm"
                variant="ghost"
                icon={<CloseIcon size={18} />}
                onClick={() => setPanelOpen(false)}
              />
            </Tooltip>
          </Flex>
          <Box
            position="absolute"
            left="0"
            top="0"
            bottom="0"
            w="6px"
            cursor="ew-resize"
            bg="transparent"
            _hover={{ bg: 'rgba(17,17,17,0.2)' }}
            onMouseDown={startResize}
            zIndex={2004}
            display={{ base: 'none', lg: 'block' }}
          />
          <GeminiLiveVoice />
        </Box>
      )}
    </Box>
  );
}

function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/login" element={<RedirectIfAuthed><LoginPage /></RedirectIfAuthed>} />
      <Route path="/signup" element={<RedirectIfAuthed><SignupPage /></RedirectIfAuthed>} />
      {/* Signed-in pages share one layout (header, voice panel) through a nested route. */}
      <Route element={<RequireAuth><AppLayout /></RequireAuth>}>
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/playground" element={<Playground />} />
        <Route path="/phone" element={<Phone />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

function App() {
  return (
    <AuthProvider>
      <SocketProvider>
        <CalibrationProvider>
          <AIWorkspaceProvider>
            <Router>
              <AppRoutes />
            </Router>
          </AIWorkspaceProvider>
        </CalibrationProvider>
      </SocketProvider>
    </AuthProvider>
  );
}

export default App;
