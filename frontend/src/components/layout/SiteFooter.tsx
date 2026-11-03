import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import { Box, Divider, Flex, Grid, HStack, Link, Text, VStack } from '@chakra-ui/react';
import { LuExternalLink, LuGithub, LuMessageSquare, LuServer } from 'react-icons/lu';
import { Logo } from '../brand/Logo';

const GithubIcon = LuGithub as any;
const IssueIcon = LuMessageSquare as any;
const StatusIcon = LuServer as any;
const ExternalIcon = LuExternalLink as any;

const REPO_URL = 'https://github.com/chaman2003/Printchakra-AI';

const column = (title: string, children: React.ReactNode) => (
  <VStack align="flex-start" spacing={4}>
    <Text fontSize="12px" fontWeight={700} letterSpacing="0.14em" textTransform="uppercase" color="#f6f6f6">
      {title}
    </Text>
    <VStack align="flex-start" spacing={3}>
      {children}
    </VStack>
  </VStack>
);

const linkProps = {
  color: 'rgba(246, 246, 246, 0.6)',
  fontSize: '15px',
  transition: 'color 120ms ease',
  _hover: { color: '#f6f6f6', textDecoration: 'none' },
};

/** Dark footer: 60% off-white text, 5% white top border, line icons (design.md). */
export const SiteFooter: React.FC = () => (
  <Box as="footer" bg="#1e1e1e" borderTop="1px solid rgba(255, 255, 255, 0.05)" color="rgba(246, 246, 246, 0.6)" px={{ base: 5, md: 10 }} pt={20} pb={10}>
    <Box maxW="1600px" mx="auto">
      <Grid templateColumns={{ base: '1fr', md: 'repeat(2, 1fr)', lg: '1.6fr 1fr 1fr 1.2fr' }} gap={{ base: 12, lg: 16 }}>
        <VStack align="flex-start" spacing={5}>
          <Logo light />
          <Text maxW="340px" lineHeight={1.7}>
            Precision Hardware Logic. Scan with your phone, clean the page up, read the text and print it, all from one workspace.
          </Text>
        </VStack>

        {column(
          'Page',
          <>
            <Link as={RouterLink} to={{ pathname: '/', hash: '#overview' }} {...linkProps}>Overview</Link>
            <Link as={RouterLink} to={{ pathname: '/', hash: '#engine' }} {...linkProps}>Inside the Engine</Link>
            <Link as={RouterLink} to={{ pathname: '/', hash: '#vision' }} {...linkProps}>Project Vision</Link>
          </>
        )}

        {column(
          'App',
          <>
            <Link as={RouterLink} to="/dashboard" {...linkProps}>Dashboard</Link>
            <Link as={RouterLink} to="/phone" {...linkProps}>Phone</Link>
            <Link as={RouterLink} to="/playground" {...linkProps}>Playground</Link>
          </>
        )}
        {column(
          'Project',
          <>
            <Link href={REPO_URL} isExternal {...linkProps}>
              <HStack spacing={3}><GithubIcon size={16} /><span>Source on GitHub</span><ExternalIcon size={13} /></HStack>
            </Link>
            <Link href={`${REPO_URL}/issues`} isExternal {...linkProps}>
              <HStack spacing={3}><IssueIcon size={16} /><span>Report an issue</span><ExternalIcon size={13} /></HStack>
            </Link>
            <HStack spacing={3} fontSize="15px"><StatusIcon size={16} /><span>Hosted on Vercel</span></HStack>
          </>
        )}
      </Grid>

      <Divider borderColor="rgba(255,255,255,0.05)" my={12} />
      <Flex justify="space-between" direction={{ base: 'column', md: 'row' }} gap={3} fontSize="13px">
        <Text>{'\u00a9'} {new Date().getFullYear()} PrintChakra. Built with {'\u2764'} for document efficiency.</Text>
        <Text>Built by Chaman S.</Text>
      </Flex>
    </Box>
  </Box>
);

export default SiteFooter;
