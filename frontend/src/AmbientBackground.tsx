import { useEffect, useRef, useState } from 'react';

/** Same CloudFront ambient clip used by companion/hero.py */
export const VIDEO_SRC =
  'https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260314_131748_f2ca2a28-fed7-44c8-b9a9-bd9acdd5ec31.mp4';

function prefersReducedMotion(): boolean {
  if (typeof window === 'undefined' || !window.matchMedia) return false;
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/**
 * Fixed full-viewport muted loop video + soft dark-blue scrim.
 * Behind the app (z-index: -1). Falls back to a static dark gradient when
 * the user prefers reduced motion, or if the video fails to load.
 */
export function AmbientBackground() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [reduceMotion, setReduceMotion] = useState(prefersReducedMotion);
  const [videoFailed, setVideoFailed] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    const onChange = () => setReduceMotion(mq.matches);
    onChange();
    mq.addEventListener?.('change', onChange);
    return () => mq.removeEventListener?.('change', onChange);
  }, []);

  useEffect(() => {
    if (reduceMotion) return;
    const v = videoRef.current;
    if (!v) return;
    const tryPlay = () => {
      void v.play().catch(() => undefined);
    };
    tryPlay();
    const onVis = () => {
      if (!document.hidden) tryPlay();
    };
    document.addEventListener('visibilitychange', onVis);
    return () => document.removeEventListener('visibilitychange', onVis);
  }, [reduceMotion]);

  const showVideo = !reduceMotion && !videoFailed;

  return (
    <div className="ambient-bg" aria-hidden="true">
      {showVideo ? (
        <video
          ref={videoRef}
          className="ambient-video"
          src={VIDEO_SRC}
          autoPlay
          loop
          muted
          playsInline
          preload="auto"
          onError={() => setVideoFailed(true)}
        />
      ) : (
        <div className="ambient-fallback" />
      )}
      <div className="ambient-scrim" />
    </div>
  );
}
