import React, { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { Box } from '@chakra-ui/react';
import { SiteHeader } from '../components/layout/SiteHeader';
import { SiteFooter } from '../components/layout/SiteFooter';
import { Hero } from './landing/Hero';
import { Philosophy } from './landing/Philosophy';
import { Showcase } from './landing/Showcase';
import { Features } from './landing/Features';
import { HowItWorks } from './landing/HowItWorks';
import { Pricing } from './landing/Pricing';
import { Faq } from './landing/Faq';
import { FinalCta } from './landing/FinalCta';

/** Public marketing page. Section order follows design.md: hero, philosophy, showcase, services, then the SaaS basics. */
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
        <Philosophy />
        <Showcase />
        <Features />
        <HowItWorks />
        <Pricing />
        <Faq />
        <FinalCta />
      </main>
      <SiteFooter />
    </Box>
  );
};

export default LandingPage;
