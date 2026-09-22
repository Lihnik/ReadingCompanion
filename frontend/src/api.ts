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

export async function uploadSpeaker(file: File) {
  const fd = new FormData();
  fd.append('file', file);
  return json<{ speaker_key: string; bytes: number }>(
    await fetch(`${BASE}/api/tts/speaker`, { method: 'POST', body: fd }),
  );
}

export async function startSectionTts(opts: {
  book_id: string;
  section_idx: number;
  engine: string;
  voice: string;
  rate: number;
  speaker_key?: string | null;
}) {
  return json<{ job_id: string; total_segments: number; engine: string; voice: string }>(
    await fetch(`${BASE}/api/tts/section/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(opts),
    }),
  );
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
    error: string | null;
    loaded_sec: number;
    segments: SegmentInfo[];
  }>(await fetch(`${BASE}/api/tts/jobs/${jobId}/segments`));
}

export async function fetchWordAudio(word: string, language: string, speaker_key?: string | null) {
  const res = await fetch(`${BASE}/api/tts/word`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ word, language, speaker_key: speaker_key || undefined }),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(t || res.statusText);
  }
  return res.blob();
}

export async function listVoices() {
  return json<{
    edge: Record<string, string>;
    kokoro: Record<string, string>;
    xtts_languages: Record<string, string>;
  }>(await fetch(`${BASE}/api/tts/voices`));
}
