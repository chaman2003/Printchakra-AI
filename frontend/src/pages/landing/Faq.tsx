import React from 'react';
import {
  Accordion,
  AccordionButton,
  AccordionIcon,
  AccordionItem,
  AccordionPanel,
  Box,
  Grid,
  Heading,
  Text,
} from '@chakra-ui/react';
import { Reveal } from '../../components/brand/Reveal';

const QUESTIONS = [
  {
    q: 'Do I need to install anything?',
    a: 'No. PrintChakra runs in the browser on your phone and your computer. You only need to allow camera access on the phone.',
  },
  {
    q: 'Are my documents private?',
    a: 'Documents are stored in the app and sent to Google Gemini so the text can be read. During the beta every account shares one workspace, so please do not upload anything sensitive yet. Private workspaces are the first thing on the paid roadmap.',
  },
  {
    q: 'Which languages can it read?',
    a: 'Text recognition is done by Gemini, which reads many languages. English is the language we test most, so check results on other languages before relying on them.',
  },
  {
    q: 'How does printing work?',
    a: 'PrintChakra opens your browser\'s print dialog with the cleaned-up page. You pick the printer, copies and paper there, the same as printing any web page.',
  },
  {
    q: 'Does voice control work in every browser?',
    a: 'It needs microphone access and works best in Chrome and Edge. Open the microphone button in the top bar and start the conversation.',
  },
  {
    q: 'What does it cost?',
    a: 'Nothing right now. The beta is free and every feature is on. Paid plans will be announced before anything is charged.',
  },
];

export const Faq: React.FC = () => (
  <Box as="section" id="faq" scrollMarginTop="80px" px={{ base: 5, md: 10 }} pb={{ base: 20, md: 32 }}>
    <Box maxW="1600px" mx="auto">
      <Grid templateColumns={{ base: '1fr', lg: '1fr 1.6fr' }} gap={{ base: 10, lg: 20 }}>
        <Reveal>
          <Heading as="h2" fontSize="clamp(40px, 5.6vw, 84px)" letterSpacing="-0.05em" lineHeight={0.92}>
            Questions, answered
          </Heading>
        </Reveal>
        <Reveal delay={120}>
          <Accordion allowToggle borderTop="1px solid #1e1e1e">
            {QUESTIONS.map((item) => (
              <AccordionItem key={item.q} border="none" borderBottom="1px solid rgba(30,30,30,0.15)">
                <h3>
                  <AccordionButton py={6} px={0} _hover={{ bg: 'transparent' }}>
                    <Text flex="1" textAlign="left" fontFamily="heading" fontWeight={600} fontSize={{ base: '20px', md: '24px' }} letterSpacing="-0.03em" sx={{ wordSpacing: '0.14em' }}>
                      {item.q}
                    </Text>
                    <AccordionIcon boxSize={6} />
                  </AccordionButton>
                </h3>
                <AccordionPanel px={0} pb={6} pt={0} color="#838282" fontWeight={400} fontSize="17px" lineHeight={1.7} maxW="640px">
                  {item.a}
                </AccordionPanel>
              </AccordionItem>
            ))}
          </Accordion>
        </Reveal>
      </Grid>
    </Box>
  </Box>
);

export default Faq;
