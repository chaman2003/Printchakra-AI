import React, { useState } from 'react';
import { Link as RouterLink, useLocation, useNavigate } from 'react-router-dom';
import { Alert, AlertIcon, Box, Link, Text, VStack } from '@chakra-ui/react';
import { AuthShell } from './AuthShell';
import { OrDivider, PasswordField, TextField } from './AuthFields';
import { PillButton } from '../../components/layout/SiteHeader';
import { authErrorMessage, useAuth } from '../../auth/AuthContext';

const EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;

// Demo account for trying the app. It is created on the server by default.
const DEMO = { email: 'test@gmail.com', password: 'test123' };

const LoginPage: React.FC = () => {
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const redirectTo = (location.state as { from?: string } | null)?.from || '/dashboard';

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [errors, setErrors] = useState<{ email?: string; password?: string }>({});
  const [formError, setFormError] = useState('');
  const [busy, setBusy] = useState(false);

  const submit = async (nextEmail = email, nextPassword = password) => {
    if (busy) return;
    const nextErrors: typeof errors = {};
    if (!EMAIL_RE.test(nextEmail.trim())) nextErrors.email = 'Enter a valid email address';
    if (!nextPassword) nextErrors.password = 'Enter your password';
    setErrors(nextErrors);
    setFormError('');
    if (Object.keys(nextErrors).length > 0) return;

    setBusy(true);
    try {
      await login(nextEmail.trim().toLowerCase(), nextPassword);
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
            id="email"
            label="Email"
            type="email"
            autoComplete="email"
            placeholder="you@example.com"
            value={email}
            onChange={setEmail}
            error={errors.email}
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
          setEmail(DEMO.email);
          setPassword(DEMO.password);
          void submit(DEMO.email, DEMO.password);
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
