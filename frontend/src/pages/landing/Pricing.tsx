import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import { Badge, Box, Flex, Grid, Heading, List, ListItem, Text } from '@chakra-ui/react';
import { LuCheck } from 'react-icons/lu';
import { Reveal } from '../../components/brand/Reveal';
import { PillButton } from '../../components/layout/SiteHeader';

const Check = LuCheck as any;

interface Plan {
  name: string;
  price: string;
  note: string;
  features: string[];
  available: boolean;
}

// Only Beta is live. The paid tiers are placeholders so the structure is visible; nothing is billed.
const PLANS: Plan[] = [
  {
    name: 'Beta',
    price: 'Free',
    note: 'Everything is unlocked while we are in beta.',
    features: ['Unlimited phone captures', 'Text recognition on every page', 'Voice control', 'Convert, merge and print'],
    available: true,
  },
  {
    name: 'Pro',
    price: 'Soon',
    note: 'For people who scan every day.',
    features: ['Private workspace', 'Higher upload limits', 'Priority text recognition', 'Export history'],
    available: false,
  },
  {
    name: 'Team',
    price: 'Soon',
    note: 'For offices that share a scanner.',
    features: ['Shared folders', 'Roles and permissions', 'Audit log', 'Single sign-on'],
    available: false,
  },
];

export const Pricing: React.FC = () => (
  <Box as="section" id="pricing" scrollMarginTop="80px" px={{ base: 5, md: 10 }} pb={{ base: 20, md: 32 }}>
    <Box maxW="1600px" mx="auto">
      <Reveal>
        <Heading as="h2" fontSize="clamp(40px, 5.6vw, 84px)" letterSpacing="-0.05em" lineHeight={0.92} mb={4}>
          Free while we build it
        </Heading>
        <Text color="#838282" fontWeight={400} fontSize="17px" maxW="560px" mb={{ base: 10, md: 16 }}>
          Create an account and use everything. Paid plans will come later, and nothing is charged today.
        </Text>
      </Reveal>

      <Grid templateColumns={{ base: '1fr', md: 'repeat(3, 1fr)' }} gap="32px">
        {PLANS.map((plan, index) => (
          <Reveal key={plan.name} delay={index * 100}>
            <Flex
              direction="column"
              h="100%"
              p={{ base: 7, md: 9 }}
              borderRadius="sm"
              border="1px solid"
              borderColor={plan.available ? '#111111' : 'rgba(30, 30, 30, 0.1)'}
              bg={plan.available ? '#ffffff' : 'transparent'}
              opacity={plan.available ? 1 : 0.85}
            >
              <Flex justify="space-between" align="center" mb={8}>
                <Text fontSize="12px" fontWeight={700} letterSpacing="0.18em" textTransform="uppercase">
                  {plan.name}
                </Text>
                {plan.available ? <Badge bg="#111111" color="#f2f2f2">Available</Badge> : <Badge variant="subtle">Coming soon</Badge>}
              </Flex>
              <Text fontFamily="heading" fontWeight={700} fontSize="72px" letterSpacing="-0.06em" lineHeight={0.9}>
                {plan.price}
              </Text>
              <Text color="#838282" fontWeight={400} mt={3} mb={8}>
                {plan.note}
              </Text>
              <List spacing={3} mb={10} flex="1">
                {plan.features.map((feature) => (
                  <ListItem key={feature} display="flex" gap={3} alignItems="center" fontSize="16px">
                    <Check size={16} />
                    {feature}
                  </ListItem>
                ))}
              </List>
              {plan.available ? (
                <PillButton as={RouterLink} to="/signup" w="100%">
                  Start free
                </PillButton>
              ) : (
                <PillButton variant="outline" w="100%" isDisabled>
                  Not available yet
                </PillButton>
              )}
            </Flex>
          </Reveal>
        ))}
      </Grid>
    </Box>
  </Box>
);

export default Pricing;
