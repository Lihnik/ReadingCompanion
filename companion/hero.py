import html as _html

import streamlit.components.v1 as components

HEADLINES = {
    "silence": ("In the silence between words,", "you find your own."),
    "quiet":   ("Quiet pages.",                  "Loud thoughts."),
    "pages":   ("Every page",                    "a new conversation."),
    "voice":   ("Give every book",               "your full attention."),
}

_SCRIM_COLORS = {
    "none":  "",
    "soft":  "rgba(0,0,0,0.35)",
    "heavy": "rgba(0,0,0,0.65)",
}

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  html, body {{
    width: 100%; height: 100%;
    background: transparent;
    overflow: hidden;
  }}

  .hero {{
    position: relative;
    width: 100%; height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    padding: 3rem 5rem;
    background: radial-gradient(ellipse 120% 100% at 30% 40%, #1a1520 0%, #0e0e12 60%, #070709 100%);
    animation: drift 45s ease-in-out infinite alternate;
    background-size: 200% 200%;
  }}

  @keyframes drift {{
    0%   {{ background-position: 0% 40%; }}
    50%  {{ background-position: 60% 60%; }}
    100% {{ background-position: 30% 20%; }}
  }}

  .scrim {{
    position: absolute;
    inset: 0;
    background: {scrim_color};
    pointer-events: none;
  }}

  .brand {{
    position: absolute;
    top: 2rem; left: 2.5rem;
    font-family: sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #b8995e;
    opacity: 0.85;
  }}

  .headline {{
    position: relative;
    z-index: 1;
    font-family: 'Instrument Serif', Georgia, serif;
    font-weight: 400;
    font-size: clamp(2.8rem, 5.5vw, 4.2rem);
    line-height: 1.15;
    color: #e8e3d8;
    letter-spacing: -0.01em;
  }}

  .headline em {{
    font-style: italic;
    color: #d4c9b4;
  }}

  .cta {{
    position: relative;
    z-index: 1;
    margin-top: 2.5rem;
    display: inline-block;
    padding: 0.65rem 1.75rem;
    border: 1px solid rgba(184, 153, 94, 0.55);
    border-radius: 100px;
    font-family: sans-serif;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #b8995e;
    background: transparent;
    cursor: default;
    transition: background 0.25s, color 0.25s;
    user-select: none;
  }}

  .meta {{
    position: absolute;
    bottom: 2rem; left: 2.5rem;
    z-index: 1;
    font-family: sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    color: #4a4855;
    text-transform: uppercase;
  }}
</style>
</head>
<body>
<div class="hero">
  {scrim_div}
  <span class="brand">{brand}</span>
  <div class="headline">
    {line1}<br><em>{line2}</em>
  </div>
  <div class="cta">{cta_label}</div>
  {meta_div}
</div>
</body>
</html>
"""


def render_landing_hero(
    headline_variant: str = "silence",
    brand: str = "Reading Companion",
    cta_label: str = "Open the sidebar to begin",
    scrim: str = "soft",
    show_meta: bool = True,
    height: int = 820,
) -> None:
    line1, line2 = HEADLINES.get(headline_variant, HEADLINES["silence"])
    scrim_color = _SCRIM_COLORS.get(scrim, "")
    scrim_div = f'<div class="scrim"></div>' if scrim_color else ""
    meta_div = '<div class="meta">PDF &middot; EPUB &middot; AI commentary &middot; TTS</div>' if show_meta else ""

    html_content = _HTML_TEMPLATE.format(
        scrim_color=scrim_color,
        scrim_div=scrim_div,
        brand=_html.escape(brand),
        line1=_html.escape(line1),
        line2=_html.escape(line2),
        cta_label=_html.escape(cta_label),
        meta_div=meta_div,
    )
    components.html(html_content, height=height, scrolling=False)
