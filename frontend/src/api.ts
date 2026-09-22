const BASE = '';

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      try {
        detail = await res.text();
      } catch {
        /* ignore */
      }
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return res.json() as Promise<T>;
}

export type SectionMeta = { index: number; title: string; char_count: number };
export type VocabEntry = { word: string; translation: string };
export type ChatMsg = { role: 'user' | 'assistant'; content: string };

export type TtsOpts = {
  engine: string;
  voice: string;
  rate: number;
  speaker_key?: string | null;
};

export async function health() {
  return json<{ status: string }>(await fetch(`${BASE}/api/health`));
}

export async function uploadBook(file: File) {
  const fd = new FormData();
  fd.append('file', file);
  return json<{
    book_id: string;
    filename: string;
    sections: SectionMeta[];
  }>(await fetch(`${BASE}/api/books/upload`, { method: 'POST', body: fd }));
}

export async function getSection(bookId: string, idx: number) {
  return json<{
    book_id: string;
    index: number;
    title: string;
    text: string;
    char_count: number;
  }>(await fetch(`${BASE}/api/books/${bookId}/sections/${idx}`));
}

export async function listModels() {
  return json<{ models: string[]; live: boolean; error: string | null }>(
    await fetch(`${BASE}/api/ollama/models`),
  );
}

export async function fetchSummary(opts: {
  book_id: string;
  section_idx: number;
  language: string;
  model: string;
}) {
  return json<{ summary: string }>(
    await fetch(`${BASE}/api/ai/summary`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(opts),
    }),
  );
}

export async function fetchVocab(opts: {
  book_id: string;
  section_idx: number;
  language: string;
  model: string;
}) {
  return json<{ vocab: VocabEntry[]; parse_ok: boolean; raw_preview: string | null }>(
    await fetch(`${BASE}/api/ai/vocab`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(opts),
    }),
  );
}

export async function fetchCommentary(opts: {
  book_id: string;
  section_idx: number;
  model: string;
}) {
  return json<{ commentary: string }>(
    await fetch(`${BASE}/api/ai/commentary`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(opts),
    }),
  );
}

export async function fetchQuestion(opts: {
  book_id: string;
  section_idx: number;
  model: string;
}) {
  return json<{ question: string }>(
    await fetch(`${BASE}/api/ai/question`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(opts),
    }),
  );
}

export async function fetchFeedback(opts: {
  book_id: string;
  section_idx: number;
  question: string;
  answer: string;
  model: string;
}) {
  return json<{ feedback: string }>(
    await fetch(`${BASE}/api/ai/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(opts),
    }),
  );
}

export async function fetchSectionSummary(opts: {
  book_id: string;
  section_idx: number;
  model: string;
}) {
  return json<{ summary: string }>(
    await fetch(`${BASE}/api/ai/section-summary`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(opts),
    }),
  );
}

/** Stream chat tokens via SSE. Calls onToken for each piece; returns full assistant text. */
export async function streamChat(
  opts: {
    book_id: string;
    section_idx: number;
    message: string;
    history: ChatMsg[];
    model: string;
  },
  onToken: (token: string) => void,
  signal?: AbortSignal,
): Promise<string> {
  const res = await fetch(`${BASE}/api/ai/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(opts),
    signal,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      detail = await res.text();
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  if (!res.body) throw new Error('No response body for chat stream');

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let full = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split('\n\n');
    buffer = chunks.pop() || '';
    for (const chunk of chunks) {
      const lines = chunk.split('\n');
      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trimStart();
        if (payload === '[DONE]') continue;
        try {
          const token = JSON.parse(payload) as string;
          if (token) {
            full += token;
            onToken(token);
          }
        } catch {
          // Non-JSON fallback (legacy plain tokens)
          if (payload) {
            full += payload;
            onToken(payload);
          }
        }
      }
    }
  }
  return full;
}

export async function uploadSpeaker(file: File) {
  const fd = new FormData();
  fd.append('file', file);
  return json<{ speaker_key: string; bytes: number }>(
    await fetch(`${BASE}/api/tts/speaker`, { method: 'POST', body: fd }),
  );
}

export async function startSectionTts(
  opts: {
    book_id: string;
    section_idx: number;
  } & TtsOpts,
) {
  return json<{ job_id: string; total_segments: number; engine: string; voice: string }>(
    await fetch(`${BASE}/api/tts/section/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(opts),
    }),
  );
}

export async function startTextTts(opts: { text: string } & TtsOpts) {
  return json<{ job_id: string; total_segments: number; engine: string; voice: string }>(
    await fetch(`${BASE}/api/tts/text/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(opts),
    }),
  );
}

export async function oneshotTts(opts: { text: string } & TtsOpts) {
  const res = await fetch(`${BASE}/api/tts/oneshot`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(opts),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(t || res.statusText);
  }
  return res.blob();
}

export type SegmentInfo = {
  index: number;
  duration_sec: number;
  mime?: string;
  url?: string;
  error?: string;
};

export async function pollSegments(jobId: string) {
  return json<{
    job_id: string;
    total: number;
    ready_count: number;
    done: boolean;
    cancelled?: boolean;
    error: string | null;
    loaded_sec: number;
    segments: SegmentInfo[];
  }>(await fetch(`${BASE}/api/tts/jobs/${jobId}/segments`));
}

export async function stopTtsJob(jobId: string) {
  return json<{ ok: boolean }>(
    await fetch(`${BASE}/api/tts/jobs/${jobId}/stop`, { method: 'POST' }),
  );
}

export async function fetchWordAudio(
  word: string,
  language: string,
  speaker_key?: string | null,
  rate = 1.0,
  engine?: string | null,
) {
  const res = await fetch(`${BASE}/api/tts/word`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      word,
      language,
      speaker_key: speaker_key || undefined,
      rate,
      engine: engine || undefined,
    }),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail || body);
    } catch {
      try {
        detail = await res.text();
      } catch {
        /* ignore */
      }
    }
    throw new Error(detail || res.statusText);
  }
  return res.blob();
}

export async function listVoices() {
  return json<{
    edge: Record<string, string>;
    kokoro: Record<string, string>;
    xtts_languages: Record<string, string>;
    piper_italian?: Record<string, string>;
    edge_lang_voices?: Record<string, string>;
    piper_italian_available?: boolean;
  }>(await fetch(`${BASE}/api/tts/voices`));
}

export async function startAudiobook(
  opts: {
    book_id: string;
    start_idx: number;
    end_idx: number;
  } & TtsOpts,
) {
  return json<{ job_id: string; total_sections: number; engine: string; voice: string }>(
    await fetch(`${BASE}/api/tts/audiobook/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(opts),
    }),
  );
}

export async function pollAudiobook(jobId: string) {
  return json<{
    job_id: string;
    done: boolean;
    cancelled?: boolean;
    error: string | null;
    done_sections: number;
    total_sections: number;
    ready: boolean;
    mime: string | null;
    filename: string | null;
    bytes: number;
    download_url: string | null;
  }>(await fetch(`${BASE}/api/tts/audiobook/${jobId}`));
}

export async function stopAudiobook(jobId: string) {
  return json<{ ok: boolean }>(
    await fetch(`${BASE}/api/tts/audiobook/${jobId}/stop`, { method: 'POST' }),
  );
}
