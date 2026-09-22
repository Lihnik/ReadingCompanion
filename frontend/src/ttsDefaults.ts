/** Language-aware TTS engine/voice defaults for Reading Companion. */

export type VoicesPayload = {
  edge: Record<string, string>;
  kokoro: Record<string, string>;
  xtts_languages: Record<string, string>;
  piper_italian?: Record<string, string>;
  piper_italian_available?: boolean;
  edge_lang_voices?: Record<string, string>;
};

export type EngineName = 'Edge TTS' | 'Kokoro' | 'XTTS' | 'Piper Italian';

export type TtsPick = { engine: EngineName; voice: string };

const PIPER_PAOLA = 'Paola (it_IT medium)';
const KOKORO_HEART = 'Heart (US, Female) A';

function firstKey(map: Record<string, string> | undefined, fallback: string): string {
  if (!map) return fallback;
  const keys = Object.keys(map);
  return keys[0] || fallback;
}

/**
 * Default engine/voice for section read-aloud / vocab (book / content language).
 * Returns null when we should leave the current selection alone (unknown language).
 */
export function pickDefaultsForLanguage(
  language: string,
  voices: VoicesPayload,
): TtsPick | null {
  const lang = language || 'Italian';

  if (lang === 'Italian') {
    if (voices.piper_italian_available !== false) {
      const voice =
        Object.keys(voices.piper_italian || {})[0] || PIPER_PAOLA;
      return { engine: 'Piper Italian', voice };
    }
    // Piper missing → Edge Italian voice id if known, else first Edge English
    const itId = voices.edge_lang_voices?.Italian;
    if (itId) {
      const labeled = Object.entries(voices.edge).find(([, id]) => id === itId)?.[0];
      return { engine: 'Edge TTS', voice: labeled || itId };
    }
    return { engine: 'Edge TTS', voice: firstKey(voices.edge, 'Aria (US, Female)') };
  }

  if (lang === 'Estonian') {
    const voice =
      lang in (voices.xtts_languages || {})
        ? lang
        : firstKey(voices.xtts_languages, 'Estonian');
    return { engine: 'XTTS', voice };
  }

  if (lang === 'English') {
    const kokoroKeys = Object.keys(voices.kokoro || {});
    if (kokoroKeys.length) {
      const voice = kokoroKeys.includes(KOKORO_HEART) ? KOKORO_HEART : kokoroKeys[0];
      return { engine: 'Kokoro', voice };
    }
    return { engine: 'Edge TTS', voice: firstKey(voices.edge, 'Aria (US, Female)') };
  }

  // Other XTTS-supported languages → XTTS (avoids staying on Piper after leaving Italian)
  if (lang in (voices.xtts_languages || {})) {
    return { engine: 'XTTS', voice: lang };
  }

  return null;
}

/** Always English-capable: Kokoro preferred, Edge fallback. */
export function pickEnglishSummaryDefaults(voices: VoicesPayload | null): TtsPick {
  if (voices) {
    const kokoroKeys = Object.keys(voices.kokoro || {});
    if (kokoroKeys.length) {
      const voice = kokoroKeys.includes(KOKORO_HEART) ? KOKORO_HEART : kokoroKeys[0];
      return { engine: 'Kokoro', voice };
    }
    return { engine: 'Edge TTS', voice: firstKey(voices.edge, 'Aria (US, Female)') };
  }
  return { engine: 'Edge TTS', voice: 'Aria (US, Female)' };
}

/** First Edge voice for retry after Kokoro failure. */
export function pickEdgeEnglish(voices: VoicesPayload | null): TtsPick {
  return {
    engine: 'Edge TTS',
    voice: voices ? firstKey(voices.edge, 'Aria (US, Female)') : 'Aria (US, Female)',
  };
}
