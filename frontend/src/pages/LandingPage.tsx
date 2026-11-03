import React, { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { Box } from '@chakra-ui/react';
import { SiteHeader } from '../components/layout/SiteHeader';
import { SiteFooter } from '../components/layout/SiteFooter';
import { Hero } from './landing/Hero';
import { Stats } from './landing/Stats';
import { Engine } from './landing/Engine';
import { Vision } from './landing/Vision';
import { FinalCta } from './landing/FinalCta';

/** Landing page: hero, key figures, "Inside the Engine", project vision, closing call to action, footer. */
const LandingPage: React.FC = () => {
  const { hash } = useLocation();

  // React Router does not scroll to #anchors on its own.
  useEffect(() => {
    if (!hash) {
      window.scrollTo({ top: 0 });
      return;
    }
    const target = document.getElementById(hash.slice(1));
    if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [hash]);

  return (
    <Box bg="#f2f2f2" minH="100vh">
      <SiteHeader mode="marketing" />
      <main>
        <Hero />
        <Stats />
        <Engine />
        <Vision />
        <FinalCta />
      </main>
      <SiteFooter />
    </Box>
  );
};

export default LandingPage;
