import React, { useState } from 'react';
import { Link as RouterLink, useNavigate } from 'react-router-dom';
import { Alert, AlertIcon, Box, Link, Text, VStack } from '@chakra-ui/react';
import { AuthShell } from './AuthShell';
import { PasswordField, TextField } from './AuthFields';
import { PillButton } from '../../components/layout/SiteHeader';
import { authErrorMessage, useAuth } from '../../auth/AuthContext';

const EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;
const USERNAME_RE = /^[a-z0-9][a-z0-9_.-]{2,23}$/;
const MIN_PASSWORD = 6;

interface Errors {
  name?: string;
  username?: string;
  email?: string;
  password?: string;
  confirm?: string;
}

const SignupPage: React.FC = () => {
  const { signup } = useAuth();
  const navigate = useNavigate();

  const [name, setName] = useState('');
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [errors, setErrors] = useState<Errors>({});
  const [formError, setFormError] = useState('');
  const [busy, setBusy] = useState(false);

  const validate = (): Errors => {
    const next: Errors = {};
    if (!name.trim()) next.name = 'Enter your name';
    if (!USERNAME_RE.test(username.trim().toLowerCase())) next.username = 'Use 3-24 letters, numbers, dots, dashes or underscores';
    if (!EMAIL_RE.test(email.trim())) next.email = 'Enter a valid email address';
    if (password.length < MIN_PASSWORD) next.password = `Use at least ${MIN_PASSWORD} characters`;
    if (confirm !== password) next.confirm = 'The passwords do not match';
    return next;
  };

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (busy) return;
    const next = validate();
    setErrors(next);
    setFormError('');
    if (Object.keys(next).length > 0) return;

    setBusy(true);
    try {
      await signup(name.trim(), username.trim().toLowerCase(), email.trim().toLowerCase(), password);
      navigate('/dashboard', { replace: true });
    } catch (error) {
      setFormError(authErrorMessage(error));
    } finally {
      setBusy(false);
    }
  };

  return (
    <AuthShell title="Create your account" subtitle="It takes a few seconds and it is free while we are in beta.">
      <Box as="form" noValidate onSubmit={submit}>
        <VStack spacing={5} align="stretch">
          {formError && (
            <Alert status="error" variant="left-accent" role="alert" fontSize="15px">
              <AlertIcon />
              {formError}
            </Alert>
          )}
          <TextField id="name" label="Name" autoComplete="name" placeholder="Your name" value={name} onChange={setName} error={errors.name} autoFocus />
          <TextField
            id="username"
            label="Username"
            autoComplete="username"
            autoCapitalize="none"
            spellCheck={false}
            placeholder="Pick a username"
            helper="You will log in with this."
            value={username}
            onChange={setUsername}
            error={errors.username}
          />
          <TextField
            id="email"
            label="Email"
            type="email"
            autoComplete="email"
            placeholder="you@example.com"
            value={email}
            onChange={setEmail}
            error={errors.email}
          />
          <PasswordField
            id="password"
            label="Password"
            autoComplete="new-password"
            placeholder={`At least ${MIN_PASSWORD} characters`}
            value={password}
            onChange={setPassword}
            error={errors.password}
          />
          <PasswordField
            id="confirm"
            label="Confirm password"
            autoComplete="new-password"
            placeholder="Type it again"
            value={confirm}
            onChange={setConfirm}
            error={errors.confirm}
          />
          <PillButton type="submit" w="100%" h="52px" isLoading={busy} loadingText="Creating account">
            Create account
          </PillButton>
        </VStack>
      </Box>

      <Text mt={8} color="#838282" fontSize="16px">
        Already have an account?{' '}
        <Link as={RouterLink} to="/login" color="#111111" fontWeight={700} textDecoration="underline">
          Log in
        </Link>
      </Text>
    </AuthShell>
  );
};

export default SignupPage;
