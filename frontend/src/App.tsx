import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  fetchSummary,
  fetchVocab,
  fetchWordAudio,
  getSection,
  listModels,
  listVoices,
  startSectionTts,
  uploadBook,
  uploadSpeaker,
  type SectionMeta,
  type VocabEntry,
} from './api';
import { ProgressivePlayer } from './ProgressivePlayer';
import './index.css';

const LANGS = [
  'Italian',
  'German',
  'French',
  'Spanish',
  'Portuguese',
  'Dutch',
  'Polish',
  'Russian',
  'Finnish',
  'Estonian',
  'English',
];

const ENGINES = ['Edge TTS', 'Kokoro', 'XTTS'] as const;

export default function App() {
  const [models, setModels] = useState<string[]>([]);
  const [modelLive, setModelLive] = useState(false);
  const [model, setModel] = useState('llama3.1:8b');
  const [language, setLanguage] = useState('Italian');
  const [engine, setEngine] = useState<(typeof ENGINES)[number]>('Edge TTS');
  const [voices, setVoices] = useState<{
    edge: Record<string, string>;
    kokoro: Record<string, string>;
    xtts_languages: Record<string, string>;
  } | null>(null);
  const [voice, setVoice] = useState('Aria (US, Female)');
  const [speakerKey, setSpeakerKey] = useState<string | null>(null);

  const [bookId, setBookId] = useState<string | null>(null);
  const [filename, setFilename] = useState('');
  const [sections, setSections] = useState<SectionMeta[]>([]);
  const [sectionIdx, setSectionIdx] = useState(0);
  const [sectionTitle, setSectionTitle] = useState('');
  const [sectionText, setSectionText] = useState('');

  const [summary, setSummary] = useState('');
  const [vocab, setVocab] = useState<VocabEntry[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [jobId, setJobId] = useState<string | null>(null);
  const [wordBusy, setWordBusy] = useState<string | null>(null);
  const wordCache = useMemo(() => new Map<string, string>(), []);

  useEffect(() => {
    listModels()
      .then((r) => {
        setModels(r.models);
        setModelLive(r.live);
        if (r.models.length) setModel(r.models[0]);
      })
      .catch((e) => setError(String(e.message || e)));
    listVoices()
      .then(setVoices)
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!voices) return;
    if (engine === 'Edge TTS') {
      const keys = Object.keys(voices.edge);
      if (keys.length) setVoice(keys[0]);
    } else if (engine === 'Kokoro') {
      const keys = Object.keys(voices.kokoro);
      if (keys.length) setVoice(keys[0]);
    } else {
      setVoice(language in voices.xtts_languages ? language : 'Italian');
    }
  }, [engine, voices]); // eslint-disable-line react-hooks/exhaustive-deps

  const loadSection = useCallback(async (bid: string, idx: number) => {
    setBusy('section');
    setError('');
    try {
      const s = await getSection(bid, idx);
      setSectionIdx(s.index);
      setSectionTitle(s.title);
      setSectionText(s.text);
      setSummary('');
      setVocab([]);
      setJobId(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }, []);

  const onUpload = async (file: File | null) => {
    if (!file) return;
    setBusy('upload');
    setError('');
    try {
      const res = await uploadBook(file);
      setBookId(res.book_id);
      setFilename(res.filename);
      setSections(res.sections);
      if (res.sections.length) {
        await loadSection(res.book_id, res.sections[0].index);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const onSpeaker = async (file: File | null) => {
    if (!file) return;
    try {
      const res = await uploadSpeaker(file);
      setSpeakerKey(res.speaker_key);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const runSummary = async () => {
    if (!bookId) return;
    setBusy('summary');
    setError('');
    try {
      const r = await fetchSummary({
        book_id: bookId,
        section_idx: sectionIdx,
        language,
        model,
      });
      setSummary(r.summary);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const runVocab = async () => {
    if (!bookId) return;
    setBusy('vocab');
    setError('');
    try {
      const r = await fetchVocab({
        book_id: bookId,
        section_idx: sectionIdx,
        language,
        model,
      });
      setVocab(r.vocab);
      if (!r.parse_ok) {
        setError(r.raw_preview ? `Vocab parse failed. Preview: ${r.raw_preview}` : 'Vocab parse failed');
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const runPrepare = async () => {
    if (!bookId) return;
    setBusy('prepare');
    setError('');
    try {
      const [s, v] = await Promise.all([
        fetchSummary({ book_id: bookId, section_idx: sectionIdx, language, model }),
        fetchVocab({ book_id: bookId, section_idx: sectionIdx, language, model }),
      ]);
      setSummary(s.summary);
      setVocab(v.vocab);
      if (!v.parse_ok) {
        setError(v.raw_preview ? `Vocab parse failed. Preview: ${v.raw_preview}` : 'Vocab parse failed');
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const runReadAloud = async () => {
    if (!bookId) return;
    setBusy('tts');
    setError('');
    try {
      const r = await startSectionTts({
        book_id: bookId,
        section_idx: sectionIdx,
        engine,
        voice,
        rate: 1.0,
        speaker_key: speakerKey,
      });
      setJobId(r.job_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const playWord = async (word: string) => {
    setWordBusy(word);
    try {
      let url = wordCache.get(`${language}|${word}`);
      if (!url) {
        const blob = await fetchWordAudio(word, language, speakerKey);
        url = URL.createObjectURL(blob);
        wordCache.set(`${language}|${word}`, url);
      }
      const audio = new Audio(url);
      await audio.play();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setWordBusy(null);
    }
  };

  const voiceOptions = useMemo(() => {
    if (!voices) return [] as string[];
    if (engine === 'Edge TTS') return Object.keys(voices.edge);
    if (engine === 'Kokoro') return Object.keys(voices.kokoro);
    return Object.keys(voices.xtts_languages);
  }, [engine, voices]);

  return (
    <div className="app">
      <aside className="sidebar">
        <h1 style={{ fontSize: '1.1rem', marginTop: 0 }}>Reading Companion</h1>
        <div className="muted" style={{ marginBottom: '0.75rem' }}>
          Language Learning MVP (React)
        </div>

        <label className="field">
          Upload PDF / EPUB
          <input
            type="file"
            accept=".pdf,.epub"
            onChange={(e) => onUpload(e.target.files?.[0] ?? null)}
          />
        </label>
        {filename && <div className="muted">{filename}</div>}

        <label className="field">
          Ollama model {modelLive ? <span className="ok">● live</span> : <span>● fallback</span>}
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          Book language
          <select value={language} onChange={(e) => setLanguage(e.target.value)}>
            {LANGS.map((l) => (
              <option key={l} value={l}>
                {l}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          TTS engine
          <select
            value={engine}
            onChange={(e) => setEngine(e.target.value as (typeof ENGINES)[number])}
          >
            {ENGINES.map((en) => (
              <option key={en} value={en}>
                {en}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          Voice / language
          <select value={voice} onChange={(e) => setVoice(e.target.value)}>
            {voiceOptions.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>

        {engine === 'XTTS' && (
          <label className="field">
            XTTS speaker WAV
            <input type="file" accept=".wav,audio/*" onChange={(e) => onSpeaker(e.target.files?.[0] ?? null)} />
            {speakerKey ? <span className="ok">Speaker ready</span> : <span className="muted">Required for XTTS</span>}
          </label>
        )}

        <h3 style={{ fontSize: '0.85rem', color: 'var(--muted)' }}>Sections</h3>
        <ul className="section-list">
          {sections.map((s) => (
            <li key={s.index}>
              <button
                type="button"
                className={s.index === sectionIdx ? 'active' : ''}
                onClick={() => bookId && loadSection(bookId, s.index)}
              >
                <div>{s.title}</div>
                <div className="muted">{s.char_count} chars</div>
              </button>
            </li>
          ))}
        </ul>
      </aside>

      <main className="main">
        <div className="header">
          <h1>{sectionTitle || 'No section selected'}</h1>
          {bookId && <span className="badge">section {sectionIdx}</span>}
          {busy && <span className="muted">Working: {busy}…</span>}
        </div>
        {error && <div className="error" style={{ padding: '0.5rem 1.25rem' }}>{error}</div>}
        <div className="section-text">{sectionText || 'Upload a book to begin.'}</div>
        <div className="player-bar">
          <div className="btn-row">
            <button type="button" className="btn primary" disabled={!bookId || !!busy} onClick={runReadAloud}>
              Read aloud
            </button>
            <button type="button" className="btn" disabled={!bookId || !!busy} onClick={runPrepare}>
              Prepare
            </button>
            <button type="button" className="btn" disabled={!bookId || !!busy} onClick={runSummary}>
              English summary
            </button>
            <button type="button" className="btn" disabled={!bookId || !!busy} onClick={runVocab}>
              Vocabulary
            </button>
          </div>
          <ProgressivePlayer jobId={jobId} />
        </div>
      </main>

      <aside className="rail">
        <div className="card">
          <h3>English summary</h3>
          <div style={{ whiteSpace: 'pre-wrap', fontSize: '0.92rem' }}>
            {summary || <span className="muted">Not generated yet.</span>}
          </div>
        </div>

        <div className="card">
          <h3>Vocabulary</h3>
          {vocab.length === 0 ? (
            <span className="muted">Not generated yet.</span>
          ) : (
            <table className="vocab-table">
              <thead>
                <tr>
                  <th></th>
                  <th>Word</th>
                  <th>Translation</th>
                </tr>
              </thead>
              <tbody>
                {vocab.map((v) => (
                  <tr key={v.word}>
                    <td>
                      <button
                        type="button"
                        className="play-word"
                        title={`Pronounce ${v.word}`}
                        disabled={wordBusy === v.word}
                        onClick={() => playWord(v.word)}
                      >
                        {wordBusy === v.word ? '…' : '▶'}
                      </button>
                    </td>
                    <td>{v.word}</td>
                    <td>{v.translation}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        <div className="card stub">
          <h3>Chat</h3>
          Coming soon — prioritize read + LL AI + audio in this MVP.
        </div>
      </aside>
    </div>
  );
}
