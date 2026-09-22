import { useCallback, useEffect, useRef, useState } from 'react';
import { pollSegments, type SegmentInfo } from './api';

type LoadedSeg = {
  index: number;
  duration: number;
  buffer: AudioBuffer;
  start: number; // timeline offset
};

function fmt(secs: number) {
  const s = Math.max(0, Math.floor(secs));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${r.toString().padStart(2, '0')}`;
}

export function ProgressivePlayer({
  jobId,
  onDone,
}: {
  jobId: string | null;
  onDone?: () => void;
}) {
  const [status, setStatus] = useState('');
  const [loadedSec, setLoadedSec] = useState(0);
  const [total, setTotal] = useState(0);
  const [readyCount, setReadyCount] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);

  const ctxRef = useRef<AudioContext | null>(null);
  const segsRef = useRef<LoadedSeg[]>([]);
  const fetchedRef = useRef<Set<number>>(new Set());
  const sourceRef = useRef<AudioBufferSourceNode | null>(null);
  const playOriginRef = useRef(0); // ctx.currentTime when play started
  const offsetRef = useRef(0); // timeline offset at play start
  const rafRef = useRef(0);
  const doneRef = useRef(false);

  const ensureCtx = () => {
    if (!ctxRef.current) {
      ctxRef.current = new AudioContext();
    }
    return ctxRef.current;
  };

  const stopSource = useCallback(() => {
    if (sourceRef.current) {
      try {
        sourceRef.current.stop();
      } catch {
        /* already stopped */
      }
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
  }, []);

  const rebuildTimeline = useCallback(() => {
    let t = 0;
    for (const s of segsRef.current) {
      s.start = t;
      t += s.duration;
    }
    setLoadedSec(t);
  }, []);

  const concatLoaded = useCallback((): AudioBuffer | null => {
    const ctx = ensureCtx();
    const segs = segsRef.current;
    if (!segs.length) return null;
    const sr = segs[0].buffer.sampleRate;
    const channels = segs[0].buffer.numberOfChannels;
    const totalLen = segs.reduce((n, s) => n + s.buffer.length, 0);
    const out = ctx.createBuffer(channels, totalLen, sr);
    for (let c = 0; c < channels; c++) {
      const dest = out.getChannelData(c);
      let offset = 0;
      for (const s of segs) {
        dest.set(s.buffer.getChannelData(c), offset);
        offset += s.buffer.length;
      }
    }
    return out;
  }, []);

  const tick = useCallback(() => {
    const ctx = ctxRef.current;
    if (!ctx || !sourceRef.current) return;
    const t = offsetRef.current + (ctx.currentTime - playOriginRef.current);
    setCurrentTime(Math.min(t, loadedSec || t));
    if (t >= loadedSec - 0.05 && readyCount >= total && total > 0) {
      setPlaying(false);
      stopSource();
      return;
    }
    rafRef.current = requestAnimationFrame(tick);
  }, [loadedSec, readyCount, total, stopSource]);

  const playFrom = useCallback(
    async (timelineSec: number) => {
      const ctx = ensureCtx();
      await ctx.resume();
      stopSource();
      const buf = concatLoaded();
      if (!buf) return;
      const max = buf.duration;
      const startAt = Math.max(0, Math.min(timelineSec, Math.max(0, max - 0.01)));
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(ctx.destination);
      src.start(0, startAt);
      sourceRef.current = src;
      playOriginRef.current = ctx.currentTime;
      offsetRef.current = startAt;
      setPlaying(true);
      setCurrentTime(startAt);
      rafRef.current = requestAnimationFrame(tick);
      src.onended = () => {
        // Only mark stopped if this source is still current
        if (sourceRef.current === src) {
          setPlaying(false);
          sourceRef.current = null;
        }
      };
    },
    [concatLoaded, stopSource, tick],
  );

  // Poll segments
  useEffect(() => {
    if (!jobId) return;
    segsRef.current = [];
    fetchedRef.current = new Set();
    doneRef.current = false;
    setStatus('Preparing…');
    setLoadedSec(0);
    setCurrentTime(0);
    setReadyCount(0);
    setTotal(0);
    stopSource();

    let cancelled = false;
    const fetchAudio = async (seg: SegmentInfo) => {
      if (!seg.url || fetchedRef.current.has(seg.index) || seg.error) return;
      fetchedRef.current.add(seg.index);
      const res = await fetch(seg.url);
      if (!res.ok) return;
      const ab = await res.arrayBuffer();
      const ctx = ensureCtx();
      const buffer = await ctx.decodeAudioData(ab.slice(0));
      if (cancelled) return;
      segsRef.current.push({
        index: seg.index,
        duration: buffer.duration,
        buffer,
        start: 0,
      });
      segsRef.current.sort((a, b) => a.index - b.index);
      rebuildTimeline();
      setStatus(`~${fmt(segsRef.current.reduce((n, s) => n + s.duration, 0))} loaded`);
    };

    const loop = async () => {
      while (!cancelled) {
        try {
          const data = await pollSegments(jobId);
          if (cancelled) break;
          setTotal(data.total);
          setReadyCount(data.ready_count);
          setStatus(
            data.done
              ? `Ready ${data.ready_count}/${data.total}`
              : `Loading ${data.ready_count}/${data.total}…`,
          );
          await Promise.all(data.segments.map(fetchAudio));
          // Auto-start when first segment ready
          if (!playing && segsRef.current.length > 0 && sourceRef.current === null && !doneRef.current) {
            doneRef.current = true; // gate autoplay once
            await playFrom(0);
          }
          if (data.done) {
            onDone?.();
            break;
          }
        } catch (e) {
          setStatus(e instanceof Error ? e.message : String(e));
          break;
        }
        await new Promise((r) => setTimeout(r, 600));
      }
    };
    loop();
    return () => {
      cancelled = true;
      stopSource();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  if (!jobId) return null;

  return (
    <div className="player">
      <div className="player-row">
        <button
          type="button"
          className="btn"
          onClick={() => (playing ? (stopSource(), setPlaying(false)) : playFrom(currentTime))}
        >
          {playing ? 'Pause' : 'Play'}
        </button>
        <input
          type="range"
          min={0}
          max={Math.max(loadedSec, 0.01)}
          step={0.05}
          value={Math.min(currentTime, loadedSec)}
          onChange={(e) => {
            const v = Number(e.target.value);
            setCurrentTime(v);
          }}
          onMouseUp={(e) => playFrom(Number((e.target as HTMLInputElement).value))}
          onTouchEnd={(e) => playFrom(Number((e.target as HTMLInputElement).value))}
          className="scrubber"
        />
        <span className="time">
          {fmt(currentTime)} / {fmt(loadedSec)}
        </span>
      </div>
      <div className="muted">{status} ({readyCount}/{total})</div>
    </div>
  );
}
