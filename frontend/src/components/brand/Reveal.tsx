import React, { useEffect, useRef, useState } from 'react';
import { Box, BoxProps } from '@chakra-ui/react';

// design.md: 700ms cubic-bezier(0.77, 0, 0.175, 1) for image reveals.
export const REVEAL_EASE = 'cubic-bezier(0.77, 0, 0.175, 1)';

/** True once the element has scrolled into view (and stays true). */
export function useInView<T extends HTMLElement>(threshold = 0.15) {
  const ref = useRef<T | null>(null);
  const [seen, setSeen] = useState(false);

  useEffect(() => {
    const node = ref.current;
    if (!node || seen) return;
    if (typeof IntersectionObserver === 'undefined') {
      setSeen(true);
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setSeen(true);
          observer.disconnect();
        }
      },
      { threshold }
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [seen, threshold]);

  return { ref, seen };
}

interface RevealProps extends BoxProps {
  /** 'clip' wipes the content in from the top (images). 'rise' fades it up (text). */
  kind?: 'clip' | 'rise';
  delay?: number;
}

export const Reveal: React.FC<RevealProps> = ({ kind = 'rise', delay = 0, children, ...rest }) => {
  const { ref, seen } = useInView<HTMLDivElement>();
  const style =
    kind === 'clip'
      ? {
          clipPath: seen ? 'inset(0% 0% 0% 0%)' : 'inset(0% 0% 100% 0%)',
          transition: `clip-path 700ms ${REVEAL_EASE} ${delay}ms`,
        }
      : {
          opacity: seen ? 1 : 0,
          transform: seen ? 'translateY(0)' : 'translateY(24px)',
          transition: `opacity 700ms ${REVEAL_EASE} ${delay}ms, transform 700ms ${REVEAL_EASE} ${delay}ms`,
        };
  // The observed element must not be clipped itself: a clip-path of zero height makes the browser report
  // it as never visible. So the outer box is watched and the inner box is animated.
  return (
    <Box ref={ref} h="100%" {...rest}>
      <Box h="100%" {...style}>
        {children}
      </Box>
    </Box>
  );
};

export default Reveal;
