"""Cinematic landing hero and ambient background utilities.

The hero is rendered via st.components.v1.html (Streamlit iframe). React,
ReactDOM, and Babel are loaded from unpkg inside the iframe; no new Python
dependencies. Headline content and scrim settings are passed via
window.RC_CONFIG so the JSX template never needs Python-level string
interpolation.

render_ambient_bg_html() returns a raw HTML snippet for st.markdown() that
injects the same background video as a fixed full-viewport layer behind
Streamlit's own content (used for Paths 2 and 3 in ui.py).
"""
from __future__ import annotations

import json

import streamlit.components.v1 as components

VIDEO_SRC = (
    "https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/"
    "hf_20260314_131748_f2ca2a28-fed7-44c8-b9a9-bd9acdd5ec31.mp4"
)

# Single HTML document — not an f-string, so JSX {expressions} are literal.
# The only placeholder replaced at call time is __CONFIG_JSON__.
_DOC = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --background: 201 100% 13%;
    --foreground: 0 0% 100%;
    --muted-foreground: 240 4% 66%;
    --font-display: 'Instrument Serif', serif;
    --font-body: 'Inter', sans-serif;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    font-family: var(--font-body);
    color: hsl(var(--foreground));
    background: hsl(var(--background));
    min-height: 100vh;
    overflow-x: hidden;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }
  .bg-video {
    position: absolute; inset: 0;
    width: 100%; height: 100%;
    object-fit: cover; z-index: 0;
  }
  .bg-scrim {
    position: absolute; inset: 0; z-index: 1; pointer-events: none;
    background: linear-gradient(180deg,
      rgba(0,16,33,0.35) 0%, rgba(0,16,33,0.15) 35%, rgba(0,16,33,0.45) 100%);
  }
  .page {
    position: relative;
    min-height: 100vh;
    display: flex; flex-direction: column;
    overflow: hidden;
  }
  main.hero {
    position: relative; z-index: 10; flex: 1;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center; padding: 90px 24px;
  }
  h1.hero-title {
    font-family: var(--font-display);
    font-weight: 400; font-style: normal;
    line-height: 0.95;
    letter-spacing: -2.46px;
    max-width: 1280px; margin: 0;
    text-wrap: balance;
    font-size: clamp(48px, 9vw, 128px);
  }
  h1.hero-title em {
    font-style: normal;
    color: hsl(var(--muted-foreground));
  }
  .hero-sub {
    color: hsl(var(--muted-foreground));
    max-width: 640px; margin: 32px auto 0;
    line-height: 1.65;
    font-size: clamp(16px, 1.4vw, 18px);
    text-wrap: pretty;
  }
  .hero-hint {
    color: hsl(var(--muted-foreground));
    font-size: 13px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 40px;
    opacity: 0.65;
  }
  @keyframes fade-rise {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .animate-fade-rise         { animation: fade-rise 0.8s ease-out both; }
  .animate-fade-rise-delay   { animation: fade-rise 0.8s ease-out 0.2s both; }
  .animate-fade-rise-delay-2 { animation: fade-rise 0.8s ease-out 0.4s both; }
  .hero-meta {
    position: relative; z-index: 10;
    max-width: 1280px; margin: 0 auto;
    padding: 0 32px 40px;
    display: flex; justify-content: space-between;
    color: hsl(var(--muted-foreground));
    font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase;
  }
  .hero-meta .dot {
    display: inline-block; width: 6px; height: 6px;
    border-radius: 9999px;
    background: hsl(var(--foreground));
    margin-right: 8px; vertical-align: middle;
    box-shadow: 0 0 12px rgba(255,255,255,0.55);
    animation: pulse 2.4s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.35; }
  }
  @media (max-width: 640px) {
    main.hero { padding: 56px 20px 72px; }
    .hero-meta { padding: 0 20px 24px; }
  }
</style>
</head>
<body>
  <div id="root"></div>

  <script>window.RC_CONFIG = __CONFIG_JSON__;</script>

  <script src="https://unpkg.com/react@18.3.1/umd/react.development.js"
          integrity="sha384-hD6/rw4ppMLGNu3tX5cjIb+uRZ7UkRJ6BPkLpg4hAu/6onKUg4lLsHAs9EBPT82L"
          crossorigin="anonymous"></script>
  <script src="https://unpkg.com/react-dom@18.3.1/umd/react-dom.development.js"
          integrity="sha384-u6aeetuaXnQ38mYT8rp6sbXaQe3NL9t+IBXmnYxwkUI2Hw4bsp2Wvmx4yRQF1uAm"
          crossorigin="anonymous"></script>
  <script src="https://unpkg.com/@babel/standalone@7.29.0/babel.min.js"
          integrity="sha384-m08KidiNqLdpJqLq95G/LEi8Qvjl/xUYll3QILypMoQ65QorJ9Lvtp2RXYGBFj1y"
          crossorigin="anonymous"></script>

  <script type="text/babel">
    const { useEffect, useRef } = React;
    const cfg = window.RC_CONFIG;

    const HEADLINES = {
      silence: {
        html: 'Where <em>dreams</em> rise <em>through the silence.</em>',
        sub: "A companion for deep readers and patient thinkers. Reading Companion turns any PDF or EPUB into a narrated chapter — with thoughtful commentary, comprehension questions, and a voice of your own."
      },
      quiet: {
        html: 'Read like <em>the world</em> is <em>still listening.</em>',
        sub: "Upload a book. Hear it in your voice, or any voice you choose. Ask questions aloud. Pause to think. Reading Companion is the quiet room a good book deserves."
      },
      pages: {
        html: 'Every page, a <em>conversation</em> <em>worth keeping.</em>',
        sub: "Turn dense chapters into dialogue. Local AI reads alongside you — offering commentary, testing comprehension, and answering questions without a single page leaving your device."
      },
      voice: {
        html: 'Give every book <em>a voice</em> <em>you return to.</em>',
        sub: "Choose from natural neural voices, or clone a voice you love from a six-second clip. Generate a full audiobook from any chapter range — offline, private, yours."
      }
    };

    const scrimStyles = {
      none:  {},
      soft:  { background: "linear-gradient(180deg, rgba(0,16,33,0.35) 0%, rgba(0,16,33,0.15) 35%, rgba(0,16,33,0.45) 100%)" },
      heavy: { background: "linear-gradient(180deg, rgba(0,16,33,0.55) 0%, rgba(0,16,33,0.35) 35%, rgba(0,16,33,0.7) 100%)" }
    };

    function BackgroundVideo() {
      const vref = useRef(null);
      useEffect(() => {
        const v = vref.current;
        if (!v) return;
        const tryPlay = () => v.play().catch(() => {});
        tryPlay();
        const onVis = () => { if (!document.hidden) tryPlay(); };
        document.addEventListener("visibilitychange", onVis);
        return () => document.removeEventListener("visibilitychange", onVis);
      }, []);
      return (
        <>
          <video ref={vref} className="bg-video" src={cfg.videoUrl}
                 autoPlay loop muted playsInline preload="auto" aria-hidden="true" />
          <div className="bg-scrim" style={scrimStyles[cfg.scrim] || scrimStyles.soft} />
        </>
      );
    }

    function HeroMeta() {
      if (!cfg.showMeta) return null;
      return (
        <div className="hero-meta">
          <span><span className="dot"></span>Chapter 01 · Ready to read</span>
          <span>A Reading Companion Film</span>
        </div>
      );
    }

    function App() {
      const h = HEADLINES[cfg.headlineVariant] || HEADLINES.silence;
      return (
        <div className="page">
          <BackgroundVideo />
          <main className="hero">
            <h1 className="hero-title animate-fade-rise"
                dangerouslySetInnerHTML={{__html: h.html}} />
            <p className="hero-sub animate-fade-rise-delay">{h.sub}</p>
            {cfg.showHint && (
              <p className="hero-hint animate-fade-rise-delay-2">
                Upload a book in the sidebar to begin.
              </p>
            )}
          </main>
          <HeroMeta />
        </div>
      );
    }

    ReactDOM.createRoot(document.getElementById("root")).render(<App />);
  </script>
</body>
</html>
"""


def render_landing_hero(
    *,
    headline_variant: str = "silence",
    scrim: str = "soft",
    show_meta: bool = True,
    show_hint: bool = True,
    height: int = 820,
) -> None:
    """Render the cinematic hero inside a Streamlit iframe.

    All parameters are keyword-only. Users interact via the sidebar uploader,
    which the iframe sandbox cannot programmatically trigger.
    """
    config = {
        "headlineVariant": headline_variant,
        "scrim": scrim,
        "showMeta": show_meta,
        "showHint": show_hint,
        "videoUrl": VIDEO_SRC,
    }
    doc = _DOC.replace("__CONFIG_JSON__", json.dumps(config, ensure_ascii=False))
    components.html(doc, height=height, scrolling=False)


def render_ambient_bg_html(
    opacity: float = 0.85,
    scrim_opacity: float = 0.15,
    glass_ui: bool = False,
) -> str:
    """Return an HTML snippet for st.markdown() that injects the ambient video
    as a fixed full-viewport background behind Streamlit's content.

    opacity: video visibility (0.0–1.0). Default 0.85 keeps the video vivid.
    scrim_opacity: dark base tint behind the video (0.0 = none, lower = brighter).
    glass_ui: when True, applies frosted-glass CSS to Streamlit's main containers
              so they blur the video showing through them (Apple-style panels).
    """
    glass_css = ""
    if glass_ui:
        glass_css = """
  /* Transparent root chain — video reaches each panel directly.
     Columns must also be transparent so backdrop-filter inside them
     can see through to the fixed video at z-index: -1. */
  [data-testid="stMain"] .block-container,
  [data-testid="stHorizontalBlock"],
  [data-testid="stColumn"],
  [data-testid="stColumn"] > div:first-child {
    background: transparent !important;
  }

  /* Streamlit's markdown wrapper uses align-items: flex-start in some versions,
     which causes block children to shrink-wrap. Force full-width stretch. */
  [data-testid="stMarkdownContainer"] {
    width: 100% !important;
  }

  /* Unified glass surface applied to every panel type.
     Also override Streamlit's theme CSS custom properties so any inner div
     that inherits --secondary-background-color becomes transparent. */
  [data-testid="stVerticalBlockBorderWrapper"],
  [data-testid="stAlert"],
  [data-testid="stForm"],
  [data-testid="stExpander"] {
    --background-color: transparent;
    --secondary-background-color: transparent;
    background: rgba(6, 14, 34, 0.50) !important;
    backdrop-filter: blur(22px) saturate(1.5) !important;
    -webkit-backdrop-filter: blur(22px) saturate(1.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.13) !important;
    border-radius: 14px !important;
  }

  /* Aggressively clear child backgrounds — go 3 levels deep to catch
     Streamlit's scroll-wrapper div regardless of nesting depth. */
  [data-testid="stVerticalBlockBorderWrapper"] > div,
  [data-testid="stVerticalBlockBorderWrapper"] > div > div,
  [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"],
  [data-testid="stAlert"] > div,
  [data-testid="stAlert"] > div > div,
  [data-testid="stAlert"] > div > div > div {
    background: transparent !important;
    background-color: transparent !important;
  }

  [data-testid="stExpanderDetails"] {
    background: transparent !important;
    border-top: 1px solid rgba(255, 255, 255, 0.08) !important;
  }
  [data-testid="stChatInputContainer"] {
    background: rgba(6, 14, 34, 0.52) !important;
    backdrop-filter: blur(22px) saturate(1.5) !important;
    -webkit-backdrop-filter: blur(22px) saturate(1.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.13) !important;
    border-radius: 28px !important;
  }

  /* Headings and labels on transparent background — text-shadow for contrast */
  [data-testid="stMain"] h1,
  [data-testid="stMain"] h2,
  [data-testid="stMain"] h3,
  [data-testid="stMain"] h4 {
    text-shadow: 0 1px 12px rgba(0,0,0,0.9), 0 2px 24px rgba(0,0,0,0.6) !important;
    color: rgba(240, 245, 255, 0.98) !important;
  }
  [data-testid="stMain"] p,
  [data-testid="stMain"] span,
  [data-testid="stMain"] label {
    text-shadow: 0 1px 6px rgba(0, 0, 0, 0.7) !important;
  }

  /* Sidebar glass */
  [data-testid="stSidebar"] {
    background: rgba(4, 12, 28, 0.56) !important;
    backdrop-filter: blur(24px) saturate(1.3) !important;
    -webkit-backdrop-filter: blur(24px) saturate(1.3) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
  }
  [data-testid="stSidebar"] > div:first-child {
    background: transparent !important;
  }

  /* Header glass */
  header[data-testid="stHeader"] {
    background: rgba(4, 12, 28, 0.38) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.07) !important;
  }
"""
    return f"""
<style>
  [data-testid="stApp"] {{ background: transparent !important; }}
  [data-testid="stMain"] {{ background: transparent !important; }}
  {glass_css}
</style>
<div id="rc-ambient-bg" style="position:fixed;inset:0;z-index:-1;pointer-events:none;overflow:hidden;">
  <video id="rc-ambient-video" src="{VIDEO_SRC}" autoplay loop muted playsinline preload="auto"
    style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:{opacity};"></video>
  <div style="position:absolute;inset:0;background:rgba(0,10,20,{scrim_opacity});"></div>
</div>
"""
