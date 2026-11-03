import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import apiClient from '../apiClient';
import {
  AUTH_EXPIRED_EVENT,
  SessionUser,
  clearSession,
  getStoredUser,
  getToken,
  saveSession,
} from './session';

interface AuthContextValue {
  user: SessionUser | null;
  /** True while the stored session is being checked with the server on first load. */
  checking: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (name: string, email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used inside AuthProvider');
  return context;
};

/** Turns an axios failure into a message that is safe to show to the user. */
export const authErrorMessage = (error: any): string => {
  const serverMessage = error?.response?.data?.error;
  if (serverMessage) return serverMessage;
  if (error?.code === 'ERR_NETWORK' || !error?.response) {
    return 'Could not reach the server. Check your connection and try again.';
  }
  return 'Something went wrong. Please try again.';
};

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<SessionUser | null>(() => (getToken() ? getStoredUser() : null));
  const [checking, setChecking] = useState<boolean>(() => Boolean(getToken()));

  // Confirm the stored token is still valid.
  useEffect(() => {
    if (!getToken()) {
      setChecking(false);
      return;
    }
    let cancelled = false;
    apiClient
      .get('/auth/me')
      .then((response) => {
        if (!cancelled) setUser(response.data.user);
      })
      .catch((error) => {
        // Only a rejected token ends the session. A network error keeps the stored login.
        if (!cancelled && error?.response?.status === 401) {
          clearSession();
          setUser(null);
        }
      })
      .finally(() => {
        if (!cancelled) setChecking(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Any API call answered with 401 ends the session.
  useEffect(() => {
    const onExpired = () => {
      clearSession();
      setUser(null);
    };
    window.addEventListener(AUTH_EXPIRED_EVENT, onExpired);
    return () => window.removeEventListener(AUTH_EXPIRED_EVENT, onExpired);
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const response = await apiClient.post('/auth/login', { email, password });
    saveSession(response.data.token, response.data.user);
    setUser(response.data.user);
  }, []);

  const signup = useCallback(async (name: string, email: string, password: string) => {
    const response = await apiClient.post('/auth/signup', { name, email, password });
    saveSession(response.data.token, response.data.user);
    setUser(response.data.user);
  }, []);

  const logout = useCallback(() => {
    clearSession();
    setUser(null);
  }, []);

  const value = useMemo(() => ({ user, checking, login, signup, logout }), [user, checking, login, signup, logout]);
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
