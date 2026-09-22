import { useEffect, useRef, useState } from 'react';
import { getChat, putChat, streamChat, startTextTts, type ChatMsg, type TtsOpts } from './api';

type Props = {
  bookId: string | null;
  sectionIdx: number;
  model: string;
  tts: TtsOpts;
  onJobId: (jobId: string | null) => void;
  clearOnSectionChange?: boolean;
};

export function ChatPanel({
  bookId,
  sectionIdx,
  model,
  tts,
  onJobId,
  clearOnSectionChange = true,
}: Props) {
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState('');
  const listRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const prevSection = useRef(sectionIdx);
  const loadGen = useRef(0);

  useEffect(() => {
    if (clearOnSectionChange && prevSection.current !== sectionIdx) {
      setError('');
      abortRef.current?.abort();
      setStreaming(false);
    }
    prevSection.current = sectionIdx;
  }, [sectionIdx, clearOnSectionChange]);

  useEffect(() => {
    if (!bookId) {
      setMessages([]);
      return;
    }
    const gen = ++loadGen.current;
    setMessages([]);
    getChat(bookId, sectionIdx)
      .then((r) => {
        if (loadGen.current === gen) setMessages(r.messages || []);
      })
      .catch(() => {
        if (loadGen.current === gen) setMessages([]);
      });
  }, [bookId, sectionIdx]);

  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, streaming]);

  const send = async () => {
    const text = input.trim();
    if (!bookId || !text || streaming) return;
    setInput('');
    setError('');
    const history = messages.slice(-6);
    const userMsg: ChatMsg = { role: 'user', content: text };
    setMessages((m) => [...m, userMsg, { role: 'assistant', content: '' }]);
    setStreaming(true);
    const ac = new AbortController();
    abortRef.current = ac;
    let assistantText = '';
    try {
      assistantText = await streamChat(
        {
          book_id: bookId,
          section_idx: sectionIdx,
          message: text,
          history,
          model,
        },
        (token) => {
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last?.role === 'assistant') {
              next[next.length - 1] = { ...last, content: last.content + token };
            }
            return next;
          });
        },
        ac.signal,
      );
      const finalMessages: ChatMsg[] = [
        ...messages,
        userMsg,
        { role: 'assistant', content: assistantText },
      ];
      setMessages(finalMessages);
      void putChat(bookId, sectionIdx, finalMessages).catch(() => undefined);
    } catch (e) {
      if ((e as Error).name !== 'AbortError') {
        setError(e instanceof Error ? e.message : String(e));
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last?.role === 'assistant' && !last.content) next.pop();
          return next;
        });
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  };

  const lastAssistant = [...messages].reverse().find((m) => m.role === 'assistant' && m.content);

  const readLast = async () => {
    if (!lastAssistant) return;
    try {
      const r = await startTextTts({ text: lastAssistant.content, ...tts });
      onJobId(r.job_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="card chat-panel">
      <div className="chat-header">
        <h3>Chat</h3>
        {lastAssistant && (
          <button type="button" className="btn btn-sm" onClick={readLast} disabled={streaming}>
            Read last response
          </button>
        )}
      </div>
      <div className="chat-list" ref={listRef}>
        {messages.length === 0 ? (
          <div className="muted">Ask anything about this section.</div>
        ) : (
          messages.map((m, i) => (
            <div key={i} className={`chat-msg ${m.role}`}>
              <div className="chat-role">{m.role === 'user' ? 'You' : 'Companion'}</div>
              <div className="chat-content">
                {m.content || (streaming && i === messages.length - 1 ? '…' : '')}
              </div>
            </div>
          ))
        )}
      </div>
      {error && <div className="error">{error}</div>}
      <form
        className="chat-input-row"
        onSubmit={(e) => {
          e.preventDefault();
          void send();
        }}
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={bookId ? 'Ask about this book…' : 'Upload a book first'}
          disabled={!bookId || streaming}
        />
        <button type="submit" className="btn primary" disabled={!bookId || streaming || !input.trim()}>
          {streaming ? '…' : 'Send'}
        </button>
      </form>
    </div>
  );
}
