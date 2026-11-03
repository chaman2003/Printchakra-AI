import React from 'react';
import { Link as RouterLink, useLocation } from 'react-router-dom';
import {
  Avatar,
  Box,
  Button,
  Drawer,
  DrawerBody,
  DrawerContent,
  DrawerOverlay,
  Flex,
  HStack,
  IconButton,
  Menu,
  MenuButton,
  MenuDivider,
  MenuItem,
  MenuList,
  Text,
  Tooltip,
  VStack,
  useDisclosure,
} from '@chakra-ui/react';
import { LuArrowRight, LuLogOut, LuMenu, LuMic, LuX } from 'react-icons/lu';
import { Logo } from '../brand/Logo';
import { useAuth } from '../../auth/AuthContext';
import { useAIWorkspace } from '../../context/AIWorkspaceContext';

export const HEADER_HEIGHT = 80;

const MARKETING_LINKS = [
  { label: 'Overview', hash: '#overview' },
  { label: 'Engine', hash: '#engine' },
  { label: 'Vision', hash: '#vision' },
];

const APP_LINKS = [
  { label: 'Dashboard', path: '/dashboard' },
  { label: 'Playground', path: '/playground' },
  { label: 'Phone capture', path: '/phone' },
];

// design.md: 14px uppercase links with a 120ms colour transition to #b6b5b5.
const linkStyle = {
  fontSize: '14px',
  fontWeight: 700,
  letterSpacing: '0.06em',
  textTransform: 'uppercase' as const,
  color: '#111111',
  transition: 'color 120ms ease',
  _hover: { color: '#b6b5b5', textDecoration: 'none' },
};

const IconArrow = LuArrowRight as any;
const IconMenu = LuMenu as any;
const IconClose = LuX as any;
const IconMic = LuMic as any;
const IconLogOut = LuLogOut as any;

/** Pill button with a 1px #1e1e1e border that inverts on hover (design.md). */
export const PillButton: React.FC<React.ComponentProps<typeof Button>> = ({ children, ...props }) => (
  <Button
    size="md"
    h="44px"
    px={6}
    fontSize="13px"
    letterSpacing="0.08em"
    textTransform="uppercase"
    {...props}
  >
    {children}
  </Button>
);

const UserMenu: React.FC = () => {
  const { user, logout } = useAuth();
  if (!user) return null;
  return (
    <Menu placement="bottom-end">
      <MenuButton
        as={IconButton}
        aria-label="Account menu"
        variant="unstyled"
        h="44px"
        w="44px"
        minW="44px"
        borderRadius="full"
        border="1px solid #1e1e1e"
        display="flex"
        alignItems="center"
        justifyContent="center"
        _hover={{ bg: '#111111', color: '#f2f2f2' }}
        icon={<Avatar name={user.name} size="sm" bg="transparent" color="inherit" fontWeight={700} getInitials={(n) => n.split(' ').map((p) => p[0]).slice(0, 2).join('').toUpperCase()} />}
      />
      <MenuList>
        <Box px={4} py={2}>
          <Text fontWeight={700} noOfLines={1}>{user.name}</Text>
          <Text fontSize="13px" color="#838282" noOfLines={1}>{user.username ? `@${user.username}` : user.email}</Text>
        </Box>
        <MenuDivider />
        {user.role === 'admin' && (
          <MenuItem as={RouterLink} to="/admin">
            Accounts
          </MenuItem>
        )}
        <MenuItem
          icon={<IconLogOut size={16} />}
          onClick={() => {
            logout();
            // Full page load: lands on the home page and drops every bit of the previous user's in-memory state.
            window.location.assign('/');
          }}
        >
          Log out
        </MenuItem>
      </MenuList>
    </Menu>
  );
};

interface SiteHeaderProps {
  /** 'marketing' for the landing and auth pages, 'app' once signed in. */
  mode?: 'marketing' | 'app';
}

const AppVoiceToggle: React.FC = () => {
  const { isPanelOpen, togglePanel } = useAIWorkspace();
  return (
    <Tooltip label={isPanelOpen ? 'Close voice assistant' : 'Open voice assistant'} hasArrow>
      <IconButton
        aria-label="Toggle voice assistant"
        icon={<IconMic size={18} />}
        onClick={togglePanel}
        variant={isPanelOpen ? 'solid' : 'outline'}
        h="44px"
        w="44px"
        minW="44px"
      />
    </Tooltip>
  );
};

export const SiteHeader: React.FC<SiteHeaderProps> = ({ mode = 'marketing' }) => {
  const { user } = useAuth();
  const location = useLocation();
  const drawer = useDisclosure();

  return (
    <Box
      as="header"
      position="sticky"
      top={0}
      zIndex={1000}
      h={`${HEADER_HEIGHT}px`}
      bg="rgba(242, 242, 242, 0.9)"
      backdropFilter="blur(12px)"
      borderBottom="1px solid rgba(30, 30, 30, 0.1)"
      px={{ base: 5, md: 10 }}
    >
      <Flex h="100%" align="center" justify="space-between" maxW="1600px" mx="auto">
        <Logo to={mode === 'app' ? '/dashboard' : '/'} />

        {/* Desktop navigation */}
        <HStack as="nav" aria-label="Main" spacing={9} display={{ base: 'none', lg: 'flex' }}>
          {mode === 'marketing'
            ? MARKETING_LINKS.map((link) => (
                <Box as={RouterLink} key={link.hash} to={{ pathname: '/', hash: link.hash }} {...linkStyle}>
                  {link.label}
                </Box>
              ))
            : APP_LINKS.map((link) => {
                const active = location.pathname === link.path;
                return (
                  <Box
                    as={RouterLink}
                    key={link.path}
                    to={link.path}
                    position="relative"
                    aria-current={active ? 'page' : undefined}
                    {...linkStyle}
                    color={active ? '#111111' : '#838282'}
                    _after={{
                      content: '""',
                      position: 'absolute',
                      left: 0,
                      right: 0,
                      bottom: '-10px',
                      height: '2px',
                      bg: '#111111',
                      transform: active ? 'scaleX(1)' : 'scaleX(0)',
                      transition: 'transform 200ms ease',
                    }}
                  >
                    {link.label}
                  </Box>
                );
              })}
        </HStack>

        <HStack spacing={3}>
          {mode === 'app' && <AppVoiceToggle />}
          {user ? (
            mode === 'marketing' ? (
              <PillButton as={RouterLink} to="/dashboard" rightIcon={<IconArrow size={16} />} display={{ base: 'none', sm: 'inline-flex' }}>
                Open app
              </PillButton>
            ) : null
          ) : (
            <>
              <Button
                as={RouterLink}
                to="/login"
                variant="ghost"
                h="44px"
                px={5}
                fontSize="13px"
                letterSpacing="0.08em"
                textTransform="uppercase"
                display={{ base: 'none', sm: 'inline-flex' }}
              >
                Log in
              </Button>
              <PillButton as={RouterLink} to="/signup" variant="outline" display={{ base: 'none', sm: 'inline-flex' }}>
                Get started
              </PillButton>
            </>
          )}
          {user && <UserMenu />}
          {mode === 'marketing' && (
            <IconButton
              aria-label="Open menu"
              icon={<IconMenu size={20} />}
              variant="outline"
              h="44px"
              w="44px"
              minW="44px"
              display={{ base: 'inline-flex', lg: 'none' }}
              onClick={drawer.onOpen}
            />
          )}
        </HStack>
      </Flex>

      <Drawer isOpen={drawer.isOpen} placement="right" onClose={drawer.onClose} size="xs">
        <DrawerOverlay />
        <DrawerContent bg="#f2f2f2">
          <Flex justify="flex-end" p={5}>
            <IconButton aria-label="Close menu" icon={<IconClose size={20} />} variant="outline" onClick={drawer.onClose} />
          </Flex>
          <DrawerBody>
            <VStack align="stretch" spacing={0} onClick={drawer.onClose}>
              {MARKETING_LINKS.map((link) => (
                <Box
                  as={RouterLink}
                  key={link.hash}
                  to={{ pathname: '/', hash: link.hash }}
                  py={5}
                  borderBottom="1px solid rgba(30,30,30,0.1)"
                  fontFamily="heading"
                  fontSize="32px"
                  fontWeight={700}
                  sx={{ wordSpacing: '0.14em' }}
                  letterSpacing="-0.03em"
                >
                  {link.label}
                </Box>
              ))}
              <VStack align="stretch" pt={8} spacing={3}>
                {user ? (
                  <PillButton as={RouterLink} to="/dashboard">Open app</PillButton>
                ) : (
                  <>
                    <PillButton as={RouterLink} to="/signup">Get started</PillButton>
                    <PillButton as={RouterLink} to="/login" variant="outline">Log in</PillButton>
                  </>
                )}
              </VStack>
            </VStack>
          </DrawerBody>
        </DrawerContent>
      </Drawer>
    </Box>
  );
};

export default SiteHeader;
