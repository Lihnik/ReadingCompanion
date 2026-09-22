import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  deleteBook,
  fetchSummary,
  fetchVocab,
  fetchWordAudio,
  getBook,
  getSection,
  listBooks,
  listModels,
  listVoices,
  patchBook,
  startSectionTts,
  startTextTts,
  uploadBook,
  uploadSpeaker,
  type BookListItem,
  type SectionMeta,
  type VocabEntry,
} from './api';
import {
  pickDefaultsForLanguage,
  pickEdgeEnglish,
  pickEnglishSummaryDefaults,
  type EngineName,
  type VoicesPayload,
} from './ttsDefaults';
import { AudiobookPanel } from './AudiobookPanel';
import { ChatPanel } from './ChatPanel';
import { ProgressivePlayer } from './ProgressivePlayer';
import { ReadingPanel } from './ReadingPanel';
import { AmbientBackground } from './AmbientBackground';
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
  'Czech',
  'Turkish',
  'Arabic',
  'Chinese',
  'Japanese',
  'Korean',
  'Hungarian',
];

const ENGINES = ['Edge TTS', 'Kokoro', 'XTTS', 'Piper Italian'] as const;
type AppMode = 'language_learning' | 'reading';

export default function App() {
  const [models, setModels] = useState<string[]>([]);
  const [modelLive, setModelLive] = useState(false);
  const [model, setModel] = useState('llama3.1:8b');
  const [language, setLanguage] = useState('Italian');
  const [engine, setEngine] = useState<EngineName>('Edge TTS');
  const [rate, setRate] = useState(1.0);
  const [appMode, setAppMode] = useState<AppMode>('language_learning');
  const [voices, setVoices] = useState<VoicesPayload | null>(null);
  const [voice, setVoice] = useState('Aria (US, Female)');
  const [speakerKey, setSpeakerKey] = useState<string | null>(null);
  /** When true, sidebar engine/voice changes are respected until language changes. */
  const ttsUserOverride = useRef(false);
  const lastAutoLang = useRef<string | null>(null);

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
  const [library, setLibrary] = useState<BookListItem[]>([]);

  const ttsOpts = useMemo(
    () => ({
      engine,
      voice,
      rate,
      speaker_key: speakerKey,
    }),
    [engine, voice, rate, speakerKey],
  );

  /** English gist / summary always uses Kokoro (preferred) or Edge — never Piper Italian. */
  const summaryTtsOpts = useMemo(() => {
    const picked = pickEnglishSummaryDefaults(voices);
    return { ...picked, rate, speaker_key: null as string | null };
  }, [voices, rate]);

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

  // Auto-apply engine/voice defaults on first voices load and when language changes.
  // Manual sidebar changes set ttsUserOverride and are left alone until language changes.
  useEffect(() => {
    if (!voices) return;
    const langChanged = lastAutoLang.current !== null && lastAutoLang.current !== language;
    if (langChanged) {
      ttsUserOverride.current = false;
    }
    if (ttsUserOverride.current && lastAutoLang.current === language) return;
    const picked = pickDefaultsForLanguage(language, voices);
    if (picked) {
      setEngine(picked.engine);
      setVoice(picked.voice);
    }
    lastAutoLang.current = language;
  }, [language, voices]);

  const refreshLibrary = useCallback(async () => {
    try {
      const r = await listBooks();
      setLibrary(r.books);
    } catch {
      /* ignore listing errors on refresh */
    }
  }, []);

  const loadSection = useCallback(async (bid: string, idx: number) => {
    setBusy('section');
    setError('');
    setJobId(null);
    try {
      const s = await getSection(bid, idx);
      setSectionIdx(s.index);
      setSectionTitle(s.title);
      setSectionText(s.text);
      setSummary('');
      setVocab([]);
      void patchBook(bid, { last_section_idx: s.index }).then(() => refreshLibrary());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }, [refreshLibrary]);

  const openBook = useCallback(
    async (bid: string, preferIdx?: number) => {
      setBusy('book');
      setError('');
      try {
        const meta = await getBook(bid);
        setBookId(meta.book_id);
        setFilename(meta.filename);
        setSections(meta.sections);
        if (meta.language) setLanguage(meta.language);
        if (meta.app_mode === 'reading' || meta.app_mode === 'language_learning') {
          setAppMode(meta.app_mode);
        }
        const idx =
          preferIdx ??
          meta.last_section_idx ??
          (meta.sections.length ? meta.sections[0].index : 0);
        if (meta.sections.length) {
          await loadSection(meta.book_id, idx);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(null);
      }
    },
    [loadSection],
  );

  useEffect(() => {
    listBooks()
      .then((r) => {
        setLibrary(r.books);
        if (r.books.length) {
          const recent = r.books[0];
          void openBook(recent.book_id, recent.last_section_idx);
        }
      })
      .catch(() => undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- restore once on mount
  }, []);

  const sectionPos = useMemo(() => {
    const i = sections.findIndex((s) => s.index === sectionIdx);
    return i < 0 ? 0 : i;
  }, [sections, sectionIdx]);

  const goPrev = () => {
    if (!bookId || sectionPos <= 0) return;
    void loadSection(bookId, sections[sectionPos - 1].index);
  };
  const goNext = () => {
    if (!bookId || sectionPos >= sections.length - 1) return;
    void loadSection(bookId, sections[sectionPos + 1].index);
  };

  const onUpload = async (file: File | null) => {
    if (!file) return;
    setBusy('upload');
    setError('');
    try {
      const res = await uploadBook(file);
      setBookId(res.book_id);
      setFilename(res.filename);
      setSections(res.sections);
      await refreshLibrary();
      if (res.sections.length) {
        const idx = res.last_section_idx ?? res.sections[0].index;
        await loadSection(res.book_id, idx);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const onDeleteBook = async (bid: string) => {
    if (!window.confirm('Delete this book and its AI/chat cache?')) return;
    try {
      await deleteBook(bid);
      await refreshLibrary();
      if (bookId === bid) {
        setBookId(null);
        setFilename('');
        setSections([]);
        setSectionText('');
        setSectionTitle('');
        setSummary('');
        setVocab([]);
        const remaining = (await listBooks()).books;
        setLibrary(remaining);
        if (remaining.length) {
          void openBook(remaining[0].book_id, remaining[0].last_section_idx);
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
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
        ...ttsOpts,
      });
      setJobId(r.job_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const runReadSummary = async () => {
    if (!summary) return;
    setBusy('summary-tts');
    setError('');
    try {
      try {
        const r = await startTextTts({ text: summary, ...summaryTtsOpts });
        setJobId(r.job_id);
      } catch (e) {
        if (summaryTtsOpts.engine === 'Kokoro') {
          const edge = pickEdgeEnglish(voices);
          const r = await startTextTts({
            text: summary,
            engine: edge.engine,
            voice: edge.voice,
            rate,
            speaker_key: null,
          });
          setJobId(r.job_id);
        } else {
          throw e;
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const onEngineChange = (en: EngineName) => {
    ttsUserOverride.current = true;
    setEngine(en);
    if (!voices) return;
    if (en === 'Edge TTS') {
      const keys = Object.keys(voices.edge);
      if (keys.length) setVoice(keys[0]);
    } else if (en === 'Kokoro') {
      const keys = Object.keys(voices.kokoro);
      if (keys.length) setVoice(keys[0]);
    } else if (en === 'Piper Italian') {
      const keys = Object.keys(voices.piper_italian || { 'Paola (it_IT medium)': 'it_IT-paola-medium' });
      if (keys.length) setVoice(keys[0]);
    } else {
      setVoice(language in voices.xtts_languages ? language : 'Italian');
    }
  };

  const onVoiceChange = (v: string) => {
    ttsUserOverride.current = true;
    setVoice(v);
  };

  const playWord = async (word: string) => {
    setWordBusy(word);
    try {
      let url = wordCache.get(`${language}|${word}|${rate}|${engine}`);
      if (!url) {
        const blob = await fetchWordAudio(word, language, speakerKey, rate, engine);
        url = URL.createObjectURL(blob);
        wordCache.set(`${language}|${word}|${rate}|${engine}`, url);
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
    if (engine === 'Edge TTS') {
      const keys = Object.keys(voices.edge);
      // Include raw Edge Neural id when auto-defaulted to a book-language voice.
      if (voice && !keys.includes(voice)) {
        return [...keys, voice];
      }
      return keys;
    }
    if (engine === 'Kokoro') return Object.keys(voices.kokoro);
    if (engine === 'Piper Italian') {
      return Object.keys(voices.piper_italian || { 'Paola (it_IT medium)': 'it_IT-paola-medium' });
    }
    return Object.keys(voices.xtts_languages);
  }, [engine, voices, voice]);

  const hasBook = Boolean(bookId);

  return (
    <>
      <AmbientBackground />
      <div className="app">
      <aside className="sidebar">
        <h1 className="brand-title">Reading Companion</h1>
        <div className="muted brand-sub">
          React + FastAPI
        </div>

        <div className="mode-toggle">
          <button
            type="button"
            className={appMode === 'language_learning' ? 'active' : ''}
            onClick={() => {
              setAppMode('language_learning');
              if (bookId) void patchBook(bookId, { app_mode: 'language_learning' });
            }}
          >
            Language Learning
          </button>
          <button
            type="button"
            className={appMode === 'reading' ? 'active' : ''}
            onClick={() => {
              setAppMode('reading');
              if (bookId) void patchBook(bookId, { app_mode: 'reading' });
            }}
          >
            Reading
          </button>
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

        <div className="library-block">
          <h3 className="library-heading">Library</h3>
          {library.length === 0 ? (
            <div className="muted">No saved books yet</div>
          ) : (
            <ul className="library-list">
              {library.map((b) => (
                <li key={b.book_id} className={b.book_id === bookId ? 'active' : ''}>
                  <button
                    type="button"
                    className="library-item"
                    onClick={() => void openBook(b.book_id, b.last_section_idx)}
                    title={b.filename}
                  >
                    <div className="library-name">{b.filename}</div>
                    <div className="muted">
                      {b.section_count} sections · idx {b.last_section_idx}
                    </div>
                  </button>
                  <button
                    type="button"
                    className="btn btn-sm library-delete"
                    title="Delete book"
                    onClick={(e) => {
                      e.stopPropagation();
                      void onDeleteBook(b.book_id);
                    }}
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

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

        {appMode === 'language_learning' && (
          <label className="field">
            Book language
            <select
              value={language}
              onChange={(e) => {
                const lang = e.target.value;
                ttsUserOverride.current = false;
                setLanguage(lang);
                if (bookId) void patchBook(bookId, { language: lang }).then(() => refreshLibrary());
              }}
            >
              {LANGS.map((l) => (
                <option key={l} value={l}>
                  {l}
                </option>
              ))}
            </select>
          </label>
        )}

        <label className="field">
          TTS engine
          <select
            value={engine}
            onChange={(e) => onEngineChange(e.target.value as EngineName)}
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
          <select value={voice} onChange={(e) => onVoiceChange(e.target.value)}>
            {voiceOptions.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          Rate ({rate.toFixed(2)}×)
          <input
            type="range"
            min={0.5}
            max={2}
            step={0.05}
            value={rate}
            onChange={(e) => setRate(Number(e.target.value))}
          />
        </label>

        {engine === 'XTTS' && (
          <label className="field">
            XTTS speaker WAV
            <input type="file" accept=".wav,audio/*" onChange={(e) => onSpeaker(e.target.files?.[0] ?? null)} />
            {speakerKey ? <span className="ok">Speaker ready</span> : <span className="muted">Required for XTTS</span>}
          </label>
        )}

        {engine === 'Piper Italian' && (
          <div className="muted" style={{ marginTop: '-0.35rem', marginBottom: '0.75rem' }}>
            Dedicated Italian Piper voice: Paola (it_IT medium)
            {voices && voices.piper_italian_available === false && (
              <div style={{ color: '#fbbf24', marginTop: '0.35rem' }}>
                Piper deps missing — install <code>pip install &quot;.[piper]&quot;</code> (or{' '}
                <code>piper-tts onnxruntime</code>) then restart uvicorn. Word ▶ falls back to Edge
                until then.
              </div>
            )}
          </div>
        )}

        {language === 'Italian' && voices?.piper_italian_available === false && engine !== 'Piper Italian' && (
          <div className="muted" style={{ marginTop: '-0.35rem', marginBottom: '0.75rem', color: '#fbbf24' }}>
            Piper deps missing — Italian defaults use Edge. Install{' '}
            <code>pip install &quot;.[piper]&quot;</code> then restart uvicorn.
          </div>
        )}

        {engine === 'XTTS' && language === 'Estonian' && !speakerKey && (
          <div className="muted" style={{ marginTop: '-0.35rem', marginBottom: '0.75rem' }}>
            Estonian defaults to XTTS — upload a speaker WAV above to start.
          </div>
        )}

        <div className="nav-row">
          <button type="button" className="btn btn-sm" disabled={!bookId || sectionPos <= 0} onClick={goPrev}>
            ← Prev
          </button>
          <button
            type="button"
            className="btn btn-sm"
            disabled={!bookId || sectionPos >= sections.length - 1}
            onClick={goNext}
          >
            Next →
          </button>
        </div>

        <label className="field">
          Jump to section
          <select
            value={sectionIdx}
            disabled={!bookId}
            onChange={(e) => bookId && loadSection(bookId, Number(e.target.value))}
          >
            {sections.map((s) => (
              <option key={s.index} value={s.index}>
                {s.index}: {s.title}
              </option>
            ))}
          </select>
        </label>

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

        <AudiobookPanel bookId={bookId} sections={sections} tts={ttsOpts} />
      </aside>

      <main className="main">
        <div className="header">
          <h1>{hasBook ? (sectionTitle || 'Loading…') : 'Welcome'}</h1>
          {bookId && (
            <span className="badge">
              {sectionPos + 1}/{sections.length} · idx {sectionIdx}
            </span>
          )}
          <span className="badge mode-badge">{appMode === 'reading' ? 'Reading' : 'LL'}</span>
          {busy && <span className="muted">Working: {busy}…</span>}
        </div>
        {error && (
          <div className="error" style={{ padding: '0.5rem 1.25rem' }}>
            {error}
          </div>
        )}
        {hasBook ? (
          <div className="section-text">{sectionText || 'Loading section…'}</div>
        ) : (
          <div className="landing">
            <h2 className="landing-title">
              Where <em>dreams</em> rise <em>through the silence.</em>
            </h2>
            <p className="landing-sub">
              A companion for deep readers and patient thinkers. Upload a PDF or
              EPUB to hear it narrated, ask questions, and learn vocabulary —
              all with a quiet cinematic workspace.
            </p>
            <p className="landing-hint">Upload a book in the sidebar to begin</p>
          </div>
        )}

        {hasBook && appMode === 'language_learning' ? (
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
            <ProgressivePlayer jobId={jobId} onStopped={() => setJobId(null)} />
          </div>
        ) : hasBook ? (
          <div className="player-bar reading-bar">
            <ReadingPanel
              bookId={bookId}
              sectionIdx={sectionIdx}
              sectionText={sectionText}
              model={model}
              tts={ttsOpts}
              onReadSection={runReadAloud}
              onJobId={setJobId}
              busy={busy}
            />
            <ProgressivePlayer jobId={jobId} onStopped={() => setJobId(null)} />
          </div>
        ) : null}
      </main>

      <aside className="rail">
        {appMode === 'language_learning' && (
          <>
            <div className="card">
              <h3>English summary</h3>
              <div style={{ whiteSpace: 'pre-wrap', fontSize: '0.92rem' }}>
                {summary || <span className="muted">Not generated yet.</span>}
              </div>
              {summary && (
                <div className="btn-row" style={{ marginTop: '0.65rem' }}>
                  <button
                    type="button"
                    className="btn btn-sm"
                    disabled={!!busy}
                    onClick={() => void runReadSummary()}
                  >
                    Read summary
                  </button>
                  <span className="muted" style={{ fontSize: '0.8rem' }}>
                    Uses {summaryTtsOpts.engine} (English)
                  </span>
                </div>
              )}
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
          </>
        )}

        <ChatPanel
          bookId={bookId}
          sectionIdx={sectionIdx}
          model={model}
          tts={ttsOpts}
          onJobId={setJobId}
        />
      </aside>
    </div>
    </>
  );
}
