import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { Center, Spinner } from '@chakra-ui/react';
import { useAuth } from './AuthContext';

/** Renders its children only for a logged-in user; everyone else is sent to /login and back afterwards. */
export const RequireAuth: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user, checking } = useAuth();
  const location = useLocation();

  if (checking) {
    return (
      <Center minH="60vh">
        <Spinner size="lg" thickness="2px" color="#111111" />
      </Center>
    );
  }
  if (!user) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }
  return <>{children}</>;
};

/** For /login and /signup: logged-in users go straight to the app. */
export const RedirectIfAuthed: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user, checking } = useAuth();
  if (checking) return null;
  if (user) return <Navigate to="/dashboard" replace />;
  return <>{children}</>;
};
