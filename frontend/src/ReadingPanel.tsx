import { useEffect, useRef, useState } from 'react';
import {
  fetchCommentary,
  fetchFeedback,
  fetchQuestion,
  fetchSectionSummary,
  startTextTts,
  type TtsOpts,
} from './api';

type Props = {
  bookId: string | null;
  sectionIdx: number;
  sectionText: string;
  model: string;
  tts: TtsOpts;
  onReadSection: () => void;
  onJobId: (jobId: string | null) => void;
  busy: string | null;
};

export function ReadingPanel({
  bookId,
  sectionIdx,
  sectionText,
  model,
  tts,
  onReadSection,
  onJobId,
  busy,
}: Props) {
  const [commentary, setCommentary] = useState('');
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [feedback, setFeedback] = useState('');
  const [answered, setAnswered] = useState(false);
  const [summary, setSummary] = useState('');
  const [localBusy, setLocalBusy] = useState<string | null>(null);
  const [error, setError] = useState('');

  const commentaryCache = useRef(new Map<string, string>());
  const questionCache = useRef(new Map<string, string>());
  const summaryCache = useRef(new Map<string, string>());

  const cacheKey = `${sectionIdx}|${model}`;

  useEffect(() => {
    setCommentary(commentaryCache.current.get(cacheKey) || '');
    setQuestion(questionCache.current.get(cacheKey) || '');
    setSummary(summaryCache.current.get(cacheKey) || '');
    setAnswer('');
    setFeedback('');
    setAnswered(false);
    setError('');
  }, [cacheKey]);

  const run = async (label: string, fn: () => Promise<void>) => {
    setLocalBusy(label);
    setError('');
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLocalBusy(null);
    }
  };

  const genInsight = () =>
    run('insight', async () => {
      if (!bookId) return;
      const cached = commentaryCache.current.get(cacheKey);
      if (cached) {
        setCommentary(cached);
        return;
      }
      const r = await fetchCommentary({ book_id: bookId, section_idx: sectionIdx, model });
      commentaryCache.current.set(cacheKey, r.commentary);
      setCommentary(r.commentary);
    });

  const genQuestion = () =>
    run('question', async () => {
      if (!bookId) return;
      const cached = questionCache.current.get(cacheKey);
      if (cached) {
        setQuestion(cached);
        setAnswered(false);
        setFeedback('');
        return;
      }
      const r = await fetchQuestion({ book_id: bookId, section_idx: sectionIdx, model });
      questionCache.current.set(cacheKey, r.question);
      setQuestion(r.question);
      setAnswered(false);
      setFeedback('');
    });

  const submitAnswer = () =>
    run('feedback', async () => {
      if (!bookId || !question || !answer.trim()) return;
      const r = await fetchFeedback({
        book_id: bookId,
        section_idx: sectionIdx,
        question,
        answer: answer.trim(),
        model,
      });
      setFeedback(r.feedback);
      setAnswered(true);
    });

  const genSummary = () =>
    run('summary', async () => {
      if (!bookId) return;
      const cached = summaryCache.current.get(cacheKey);
      if (cached) {
        setSummary(cached);
        return;
      }
      const r = await fetchSectionSummary({ book_id: bookId, section_idx: sectionIdx, model });
      summaryCache.current.set(cacheKey, r.summary);
      setSummary(r.summary);
    });

  const speak = async (text: string) => {
    if (!text) return;
    try {
      const r = await startTextTts({ text, ...tts });
      onJobId(r.job_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const disabled = !bookId || !!busy || !!localBusy;

  return (
    <div className="reading-panel">
      <div className="btn-row">
        <button type="button" className="btn primary" disabled={disabled || !sectionText} onClick={onReadSection}>
          Read aloud
        </button>
        <button type="button" className="btn" disabled={disabled} onClick={genInsight}>
          {localBusy === 'insight' ? 'Generating…' : 'Generate insight'}
        </button>
        <button type="button" className="btn" disabled={disabled} onClick={genQuestion}>
          {localBusy === 'question' ? 'Generating…' : 'Get a question'}
        </button>
        <button type="button" className="btn" disabled={disabled} onClick={genSummary}>
          {localBusy === 'summary' ? 'Summarizing…' : 'Summarize section'}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      <div className="card">
        <h3>AI Commentary</h3>
        {commentary ? (
          <>
            <div style={{ whiteSpace: 'pre-wrap', fontSize: '0.92rem' }}>{commentary}</div>
            <div className="btn-row">
              <button type="button" className="btn btn-sm" onClick={() => speak(commentary)}>
                Read commentary
              </button>
            </div>
          </>
        ) : (
          <span className="muted">Read the section, then generate an insight.</span>
        )}
      </div>

      <div className="card">
        <h3>Comprehension Check</h3>
        {!question ? (
          <span className="muted">Get a question when you&apos;re ready.</span>
        ) : (
          <>
            <blockquote className="question-quote">{question}</blockquote>
            <div className="btn-row">
              <button type="button" className="btn btn-sm" onClick={() => speak(question)}>
                Read question
              </button>
            </div>
            {!answered ? (
              <div className="answer-box">
                <input
                  type="text"
                  value={answer}
                  onChange={(e) => setAnswer(e.target.value)}
                  placeholder="Your answer…"
                  disabled={!!localBusy}
                />
                <div className="btn-row">
                  <button
                    type="button"
                    className="btn primary"
                    disabled={!answer.trim() || !!localBusy}
                    onClick={submitAnswer}
                  >
                    {localBusy === 'feedback' ? 'Evaluating…' : 'Submit answer'}
                  </button>
                  <button
                    type="button"
                    className="btn"
                    disabled={!!localBusy}
                    onClick={() => setAnswered(true)}
                  >
                    Skip
                  </button>
                </div>
              </div>
            ) : (
              feedback && (
                <>
                  <div className="feedback ok-box">{feedback}</div>
                  <div className="btn-row">
                    <button type="button" className="btn btn-sm" onClick={() => speak(feedback)}>
                      Read feedback
                    </button>
                  </div>
                </>
              )
            )}
          </>
        )}
      </div>

      {summary && (
        <div className="card">
          <h3>Section Summary</h3>
          <div style={{ whiteSpace: 'pre-wrap', fontSize: '0.92rem' }}>{summary}</div>
          <div className="btn-row">
            <button type="button" className="btn btn-sm" onClick={() => speak(summary)}>
              Read summary
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
