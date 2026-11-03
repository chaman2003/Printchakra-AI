import React, { useEffect, useState } from 'react';
import { Navigate } from 'react-router-dom';
import { Alert, AlertIcon, Box, Heading, Spinner, Table, TableContainer, Tbody, Td, Text, Th, Thead, Tr } from '@chakra-ui/react';
import apiClient from '../apiClient';
import { authErrorMessage, useAuth } from '../auth/AuthContext';

interface AccountRow {
  id: number;
  username: string | null;
  email: string;
  name: string;
  role: 'user' | 'admin';
  created: string | null;
  file_count: number | null;
  bytes: number | null;
}

const formatBytes = (bytes: number | null) => {
  if (bytes === null) return '-';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

/** Admin only: every account with its file count and storage use. File contents are never shown. */
const AdminPage: React.FC = () => {
  const { user } = useAuth();
  const [rows, setRows] = useState<AccountRow[] | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    if (user?.role !== 'admin') return;
    apiClient
      .get('/admin/users')
      .then((response) => setRows(response.data.users))
      .catch((err) => setError(authErrorMessage(err)));
  }, [user?.role]);

  if (user?.role !== 'admin') return <Navigate to="/dashboard" replace />;

  return (
    <Box maxW="1100px" mx="auto" px={{ base: 3, md: 6 }} py={{ base: 6, md: 10 }}>
      <Heading as="h1" fontSize={{ base: '32px', md: '48px' }} letterSpacing="-0.04em" sx={{ wordSpacing: '0.14em' }}>
        Accounts
      </Heading>
      <Text color="#838282" mt={2} mb={8}>
        Each account has its own private workspace. You can see who is registered and how much they store, not what they store.
      </Text>

      {error && (
        <Alert status="error" variant="left-accent" role="alert">
          <AlertIcon />
          {error}
        </Alert>
      )}
      {!rows && !error && <Spinner thickness="2px" color="#111111" />}
      {rows && (
        <TableContainer bg="white" border="1px solid rgba(30,30,30,0.12)">
          <Table size="md">
            <Thead>
              <Tr>
                {['Username', 'Name', 'Email', 'Role', 'Files', 'Storage'].map((label) => (
                  <Th key={label}>{label}</Th>
                ))}
              </Tr>
            </Thead>
            <Tbody>
              {rows.map((row) => (
                <Tr key={row.id}>
                  <Td fontWeight={700}>{row.username || '-'}</Td>
                  <Td>{row.name}</Td>
                  <Td>{row.email}</Td>
                  <Td textTransform="uppercase" fontSize="12px" letterSpacing="0.12em" fontWeight={700}>
                    {row.role}
                  </Td>
                  <Td isNumeric>{row.file_count ?? '-'}</Td>
                  <Td isNumeric>{formatBytes(row.bytes)}</Td>
                </Tr>
              ))}
            </Tbody>
          </Table>
        </TableContainer>
      )}
    </Box>
  );
};

export default AdminPage;
