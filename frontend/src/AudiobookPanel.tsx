import { useEffect, useState } from 'react';
import { pollAudiobook, startAudiobook, stopAudiobook, type SectionMeta, type TtsOpts } from './api';

type Props = {
  bookId: string | null;
  sections: SectionMeta[];
  tts: TtsOpts;
};

export function AudiobookPanel({ bookId, sections, tts }: Props) {
  const indices = sections.map((s) => s.index);
  const minIdx = indices.length ? Math.min(...indices) : 0;
  const maxIdx = indices.length ? Math.max(...indices) : 0;
  const [startIdx, setStartIdx] = useState(minIdx);
  const [endIdx, setEndIdx] = useState(maxIdx);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState('');
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [filename, setFilename] = useState('');
  const [bytes, setBytes] = useState(0);
  const [error, setError] = useState('');
  const [running, setRunning] = useState(false);

  useEffect(() => {
    if (sections.length) {
      setStartIdx(minIdx);
      setEndIdx(maxIdx);
    }
  }, [bookId, minIdx, maxIdx, sections.length]);

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    const loop = async () => {
      while (!cancelled) {
        try {
          const s = await pollAudiobook(jobId);
          if (cancelled) break;
          setStatus(
            s.done
              ? s.error
                ? `Failed: ${s.error}`
                : `Ready (${s.done_sections}/${s.total_sections})`
              : `Generating ${s.done_sections}/${s.total_sections}…`,
          );
          if (s.ready && s.download_url) {
            setDownloadUrl(s.download_url);
            setFilename(s.filename || 'audiobook');
            setBytes(s.bytes);
            setRunning(false);
            break;
          }
          if (s.done) {
            if (s.error) setError(s.error);
            setRunning(false);
            break;
          }
        } catch (e) {
          setError(e instanceof Error ? e.message : String(e));
          setRunning(false);
          break;
        }
        await new Promise((r) => setTimeout(r, 800));
      }
    };
    void loop();
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  const generate = async () => {
    if (!bookId) return;
    if (startIdx > endIdx) {
      setError('Start section must be ≤ end section');
      return;
    }
    setError('');
    setDownloadUrl(null);
    setStatus('Starting…');
    setRunning(true);
    try {
      const r = await startAudiobook({
        book_id: bookId,
        start_idx: startIdx,
        end_idx: endIdx,
        ...tts,
      });
      setJobId(r.job_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setRunning(false);
    }
  };

  const stop = async () => {
    if (!jobId) return;
    try {
      await stopAudiobook(jobId);
      setRunning(false);
      setStatus('Stopped');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const selectedCount = sections.filter((s) => s.index >= startIdx && s.index <= endIdx).length;

  return (
    <div className="card">
      <h3>Audiobook export</h3>
      <p className="muted" style={{ marginTop: 0 }}>
        Generate a single audio file for a range of sections.
      </p>
      <div className="ab-range">
        <label className="field">
          Start
          <input
            type="number"
            value={startIdx}
            min={minIdx}
            max={maxIdx}
            disabled={!bookId || running}
            onChange={(e) => setStartIdx(Number(e.target.value))}
          />
        </label>
        <label className="field">
          End
          <input
            type="number"
            value={endIdx}
            min={minIdx}
            max={maxIdx}
            disabled={!bookId || running}
            onChange={(e) => setEndIdx(Number(e.target.value))}
          />
        </label>
      </div>
      <div className="muted" style={{ marginBottom: '0.5rem' }}>
        {bookId ? `${selectedCount} sections selected` : 'Upload a book first'}
      </div>
      <div className="btn-row">
        <button type="button" className="btn primary" disabled={!bookId || running} onClick={generate}>
          {running ? 'Generating…' : 'Generate audiobook'}
        </button>
        {running && (
          <button type="button" className="btn" onClick={stop}>
            Stop
          </button>
        )}
      </div>
      {status && <div className="muted">{status}</div>}
      {error && <div className="error">{error}</div>}
      {downloadUrl && (
        <a className="btn primary download-link" href={downloadUrl} download={filename}>
          Download {filename}
          {bytes ? ` (${(bytes / 1024 / 1024).toFixed(1)} MB)` : ''}
        </a>
      )}
    </div>
  );
}
