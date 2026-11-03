import React, { useState } from 'react';
import { Link as RouterLink, useLocation, useNavigate } from 'react-router-dom';
import { Alert, AlertIcon, Box, Link, Text, VStack } from '@chakra-ui/react';
import { AuthShell } from './AuthShell';
import { OrDivider, PasswordField, TextField } from './AuthFields';
import { PillButton } from '../../components/layout/SiteHeader';
import { authErrorMessage, useAuth } from '../../auth/AuthContext';

// Demo account for trying the app. It is created on the server by default.
const DEMO = { username: 'test', password: 'test123' };

const LoginPage: React.FC = () => {
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const redirectTo = (location.state as { from?: string } | null)?.from || '/dashboard';

  const [identifier, setIdentifier] = useState('');
  const [password, setPassword] = useState('');
  const [errors, setErrors] = useState<{ identifier?: string; password?: string }>({});
  const [formError, setFormError] = useState('');
  const [busy, setBusy] = useState(false);

  const submit = async (nextIdentifier = identifier, nextPassword = password) => {
    if (busy) return;
    const nextErrors: typeof errors = {};
    if (!nextIdentifier.trim()) nextErrors.identifier = 'Enter your username or email';
    if (!nextPassword) nextErrors.password = 'Enter your password';
    setErrors(nextErrors);
    setFormError('');
    if (Object.keys(nextErrors).length > 0) return;

    setBusy(true);
    try {
      await login(nextIdentifier.trim().toLowerCase(), nextPassword);
      navigate(redirectTo, { replace: true });
    } catch (error) {
      setFormError(authErrorMessage(error));
    } finally {
      setBusy(false);
    }
  };

  return (
    <AuthShell title="Log in" subtitle="Welcome back. Pick up where you left off.">
      <Box
        as="form"
        noValidate
        onSubmit={(event: React.FormEvent) => {
          event.preventDefault();
          void submit();
        }}
      >
        <VStack spacing={5} align="stretch">
          {formError && (
            <Alert status="error" variant="left-accent" role="alert" fontSize="15px">
              <AlertIcon />
              {formError}
            </Alert>
          )}
          <TextField
            id="identifier"
            label="Username or email"
            autoComplete="username"
            autoCapitalize="none"
            spellCheck={false}
            placeholder="username or you@example.com"
            value={identifier}
            onChange={setIdentifier}
            error={errors.identifier}
            autoFocus
          />
          <PasswordField
            id="password"
            label="Password"
            autoComplete="current-password"
            placeholder="Your password"
            value={password}
            onChange={setPassword}
            error={errors.password}
          />
          <PillButton type="submit" w="100%" h="52px" isLoading={busy} loadingText="Logging in">
            Log in
          </PillButton>
        </VStack>
      </Box>

      <OrDivider />
      <PillButton
        variant="outline"
        w="100%"
        h="52px"
        isDisabled={busy}
        onClick={() => {
          setIdentifier(DEMO.username);
          setPassword(DEMO.password);
          void submit(DEMO.username, DEMO.password);
        }}
      >
        Try the demo account
      </PillButton>

      <Text mt={8} color="#838282" fontSize="16px">
        New here?{' '}
        <Link as={RouterLink} to="/signup" color="#111111" fontWeight={700} textDecoration="underline">
          Create an account
        </Link>
      </Text>
    </AuthShell>
  );
};

export default LoginPage;
