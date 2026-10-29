"""
Roast My Startup & Pitch Optimizer AI Agent
===========================================
Nebius x NVIDIA Global AI Hackathon submission (single file).

Two chained calls to NVIDIA's open-source Nemotron model, served through Nebius'
OpenAI-compatible API:

  Stage 1  Chad Sterling, a merciless VC, roasts the startup and scores it.
  Stage 2  Maya, a startup mentor, rebuilds the pitch, fixes the strategy, writes a
           30-second elevator pitch and scores the result.

Quick start:
    pip install -U streamlit openai
    export NEBIUS_API_KEY="your-key"
    streamlit run app.py --theme.base dark
"""
from __future__ import annotations

import html
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse

import streamlit as st

try:
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
        NotFoundError,
        OpenAI,
        OpenAIError,
        PermissionDeniedError,
        RateLimitError,
    )
except ImportError:
    st.error("A required package is missing. Run `pip install -U openai streamlit` in your terminal, then restart the app.")
    st.stop()


# ======================================================================================
# 1. Configuration
# ======================================================================================
NEMOTRON_MODEL = "nvidia/Nemotron-3_5-Lightning"  # approved NVIDIA open-source model
DEFAULT_BASE_URL = "https://api.studio.nebius.ai/v1/"  # Nebius OpenAI-compatible endpoint
WEBSITE_HOSTS = {"nebius.ai", "www.nebius.ai", "nebius.com", "www.nebius.com"}  # company sites, not APIs

INTENSITY = {
    "🥩 Medium rare": "Firm but friendly. Poke fun, land the hits, keep a twinkle in your eye.",
    "🔥 Well done": "Savage. Heavy sarcasm, vivid metaphors, zero mercy for buzzwords.",
    "☄️ Scorched earth": "Absolutely merciless and theatrical, a hall-of-fame roast. Still PG-13 and aimed only at the idea.",
}
DEFAULT_INTENSITY = "🔥 Well done"

SCORE_KEYS = ("viability", "originality", "market_clarity", "tagline_punch")
SCORE_LABELS = {
    "viability": "Viability",
    "originality": "Originality",
    "market_clarity": "Market clarity",
    "tagline_punch": "Tagline punch",
}
SCORE_LINE = (
    '[[SCORES]] {"viability": <0-100>, "originality": <0-100>, '
    '"market_clarity": <0-100>, "tagline_punch": <0-100>}'
)

EXAMPLES = [
    {
        "idea": "A smart water bottle with a built-in screen that nags you to drink water. It syncs to an app, "
        "tracks hydration streaks, and costs $14.99 a month for premium hydration insights.",
        "audience": "Busy office workers who forget to drink water",
        "tagline": "Hydrate smarter. Live better.",
    },
    {
        "idea": "A monthly subscription box that mails you one single, hand-carved artisanal ice cube in "
        "insulated packaging, with a collectible card about its terroir.",
        "audience": "Cocktail enthusiasts with disposable income",
        "tagline": "Chill, but make it luxury.",
    },
    {
        "idea": "Tinder for co-founders, where matches are made purely on zodiac compatibility, with a premium "
        "tier that predicts your cap table using astrology.",
        "audience": "First-time founders looking for a technical co-founder",
        "tagline": "Find your co-founder written in the stars.",
    },
    {
        "idea": "An AI assistant that attends all your meetings for you and emails a summary of what it agreed "
        "to on your behalf. Right now it can only say: let us circle back on that.",
        "audience": "Middle managers drowning in calendar invites",
        "tagline": "Meetings, delegated.",
    },
]


@dataclass
class Pitch:
    idea: str
    audience: str
    tagline: str
    intensity: str


@dataclass
class Slots:
    """Places on the page that are filled in while the agent works."""

    status: Any
    body: Any
    metrics: Any
    roast: Any
    rebuild: Any
    pitch: Any


# ======================================================================================
# 2. Errors: every failure becomes a plain-English message, never a stack trace
# ======================================================================================
class PipelineError(Exception):
    """A problem we can explain to the user in plain English."""

    def __init__(self, title: str, hint: str = "") -> None:
        super().__init__(title)
        self.title = title
        self.hint = hint


def _transport_kind(exc: BaseException) -> str | None:
    """Spot raw HTTP-library errors (httpx or its successors) without importing the library,
    so the app works with any openai SDK version."""
    mro = type(exc).__mro__
    if not any(k.__module__.split(".")[0].startswith("httpx") for k in mro):
        return None
    return "timeout" if any("Timeout" in k.__name__ for k in mro) else "connection"


def explain_error(exc: BaseException, cfg: dict[str, str]) -> tuple[str, str]:
    """Translate any exception into (what happened, how to fix it)."""
    model = cfg.get("model") or NEMOTRON_MODEL
    base = cfg.get("base_url") or DEFAULT_BASE_URL

    if isinstance(exc, PipelineError):
        return exc.title, exc.hint
    if isinstance(exc, AuthenticationError):
        return (
            "Nebius rejected the API key (401)",
            "Copy the key again from your Nebius console and paste it into the sidebar. "
            "Check for extra spaces and make sure the key has not been revoked.",
        )
    if isinstance(exc, PermissionDeniedError):
        return (
            "Access denied (403)",
            f"The key is valid but cannot use `{model}`. Check that the model is enabled for your Nebius "
            "project and that your credits or billing are active.",
        )
    if isinstance(exc, NotFoundError):
        return (
            "Model or endpoint not found (404)",
            f"Either the base URL (`{base}`) is wrong or `{model}` is not served there. Open the sidebar and "
            "press **Test connection** to list the exact Nemotron model IDs your account can use.",
        )
    if isinstance(exc, RateLimitError):
        return (
            "Rate limit or quota reached (429)",
            "Wait 20 to 30 seconds and run it again. If it keeps happening, check your Nebius credits and limits.",
        )
    if isinstance(exc, BadRequestError):
        return (
            "Nebius rejected the request (400)",
            f"This usually means a wrong model ID or an unsupported setting. Nebius said: {getattr(exc, 'message', exc)}",
        )
    if isinstance(exc, APITimeoutError) or _transport_kind(exc) == "timeout":
        return (
            "Nebius took too long to answer",
            "The model may be cold-starting or busy. Wait a few seconds and run it again.",
        )
    if isinstance(exc, APIConnectionError) or _transport_kind(exc) == "connection":
        return (
            "Could not reach the Nebius endpoint",
            f"Check your internet connection, VPN or proxy, and the base URL (`{base}`).",
        )
    if isinstance(exc, APIStatusError):
        return (
            f"Nebius returned HTTP {exc.status_code}",
            "This is usually temporary. Wait a moment and run it again.",
        )
    if isinstance(exc, OpenAIError):
        return "The OpenAI client reported a problem", str(exc)
    return "Something unexpected happened", "Run it again. The technical details below show what went wrong."


def show_error(exc: BaseException, cfg: dict[str, str]) -> None:
    """Render an error card. Technical details never include the API key."""
    title, hint = explain_error(exc, cfg)
    st.error(f"**{title}**\n\n{hint}" if hint else f"**{title}**", icon="🚨")
    if not isinstance(exc, PipelineError):
        detail = f"{type(exc).__name__}: {exc}"
        if cfg.get("api_key"):
            detail = detail.replace(cfg["api_key"], "***")
        with st.expander("Technical details"):
            st.code(detail[:1200], language=None)


# ======================================================================================
# 3. Nebius connection helpers
# ======================================================================================
def default_api_key() -> str:
    """Key from the environment (or Streamlit secrets). It is never sent to the browser."""
    key = os.getenv("NEBIUS_API_KEY", "")
    if not key:
        try:
            key = str(st.secrets.get("NEBIUS_API_KEY", ""))
        except Exception:  
            key = ""
    
    # Ultimate fallback if everything else fails to fetch the value
    if not key or key == "":
        key = ""
        
    return key.strip()


def clean_key(raw: str) -> str:
    return (raw or "").strip().strip("\"'").strip()


def validate_base_url(raw: str) -> str:
    """Normalise the base URL and catch the classic mistake of using the website URL."""
    url = (raw or "").strip()
    if not url:
        raise PipelineError("The API base URL is empty", f"Use the Nebius OpenAI-compatible endpoint, for example {DEFAULT_BASE_URL}")
    if "://" not in url:
        url = "https://" + url
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or not host:
        raise PipelineError("The API base URL looks invalid", f"It should look like {DEFAULT_BASE_URL}")
    if host in WEBSITE_HOSTS:
        raise PipelineError(
            f"`{host}` is the Nebius company website, not an API endpoint",
            f"Use the OpenAI-compatible endpoint instead: {DEFAULT_BASE_URL} "
            "(if your Nebius console shows a different base URL, paste that one).",
        )
    url = url.rstrip("/")
    if not re.search(r"/v\d+$", url):
        url += "/v1"
    return url + "/"


def prepare_client(cfg: dict[str, str]) -> OpenAI:
    """Build the OpenAI-compatible client that talks to Nebius."""
    if not cfg["api_key"]:
        raise PipelineError(
            "No Nebius API key found",
            "Paste your key into the sidebar, or set the NEBIUS_API_KEY environment variable and restart the app.",
        )
    if not cfg["model"]:
        raise PipelineError("No model ID set", f"Open Advanced settings in the sidebar and enter {NEMOTRON_MODEL}.")
    return OpenAI(
        api_key=cfg["api_key"],
        base_url=validate_base_url(cfg["base_url"]),
        timeout=60.0,
        max_retries=2,
    )


def validate_inputs(p: Pitch) -> None:
    missing = [
        label
        for label, value in (("business idea", p.idea), ("target audience", p.audience), ("current tagline", p.tagline))
        if not value
    ]
    if missing:
        raise PipelineError(
            "Your pitch is incomplete",
            "Fill in: " + ", ".join(missing) + ". Press **Load a sample startup** to try the app instantly.",
        )
    if len(p.idea) < 15:
        raise PipelineError("Give the VC something to work with", "Describe the idea in at least one full sentence.")


# ======================================================================================
# 4. Prompts and text helpers
# ======================================================================================
def roast_messages(p: Pitch) -> list[dict[str, str]]:
    """Stage 1: the merciless VC."""
    system = f"""You are "Chad Sterling", a legendary, merciless Silicon Valley venture capitalist with a podcast, a fleece vest and strong opinions. You are filming the "Roast My Startup" segment of your show.

RULES
- Roast the IDEA, the MARKET, the BUSINESS MODEL and the TAGLINE. Never insult the founder personally. No slurs, no profanity: keep it PG-13.
- Be genuinely funny: sharp analogies, VC cliches, deadpan sarcasm. But hide a REAL, specific, valid criticism inside every joke (unit economics, distribution, moat, retention, competition, regulation, timing).
- Be specific to THIS startup. Reference their actual idea, audience and tagline. Generic roasts are boring.
- Roast intensity: {INTENSITY.get(p.intensity, INTENSITY[DEFAULT_INTENSITY])}

FORMAT (Markdown, 200 to 300 words, exactly these sections in this order)
### 💀 The verdict
One savage sentence.
### 🔥 The roast
Two short, punchy paragraphs.
### ☠️ Top 3 reasons this dies
A numbered list. Each item: a bold headline, then one sentence.
### 🏷️ Tagline autopsy
One or two sentences dissecting the current tagline.

Finish with ONE last line and nothing after it: an honest score of the ORIGINAL pitch (integers 0-100):
{SCORE_LINE}"""
    user = "\n".join(
        [
            f"STARTUP IDEA: {p.idea}",
            f"TARGET AUDIENCE: {p.audience}",
            f'CURRENT TAGLINE: "{p.tagline}"',
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def mentor_messages(p: Pitch, roast: str, before: dict[str, int] | None) -> list[dict[str, str]]:
    """Stage 2: the mentor who turns the roast into a winning pitch."""
    system = f"""You are "Maya", a world-class startup mentor and pitch coach: a two-time founder and the advisor every Demo Day team begs for. The founder just survived a brutal VC roast. Turn it into rocket fuel.

RULES
- Be warm, direct and razor-sharp. Take the VC's VALID criticisms seriously and FIX every one of them. Ignore the jokes.
- Be concrete and specific to this startup: a sharper wedge or niche, a pricing model, a distribution channel, a moat, the first 100 customers. No generic startup platitudes.
- Never invent traction, customers or statistics. If you use a number, label it as a target or an assumption.
- The elevator pitch must be written to be SPOKEN ALOUD: plain words, 70 to 85 words (about 30 seconds). Flow: hook, problem, solution, why now or why us, ask or traction.

FORMAT (Markdown, exactly these sections in this order)
### 🧭 Sharper positioning
Two or three sentences: the new wedge, exactly who it is for, why they will pay.
### ✨ Tagline options
Three numbered taglines (8 words max each): one bold, one benefit-led, one curiosity-driven.
### 🛠️ Strategy fixes
3 to 5 bullets, each formatted as: **The problem:** the concrete fix.
### 🗓️ 90-day game plan
Three bullets: Days 1-30, Days 31-60, Days 61-90.
### 🎤 30-second elevator pitch
A single plain-text paragraph. No bullets, no bold, no quotation marks.

Finish with ONE last line and nothing after it: an honest score of the IMPROVED pitch on the same scale the VC used (integers 0-100, do not inflate):
{SCORE_LINE}"""
    lines = [
        "ORIGINAL PITCH",
        f"- Idea: {p.idea}",
        f"- Audience: {p.audience}",
        f'- Tagline: "{p.tagline}"',
        "",
        "THE VC'S ROAST (valid criticisms are hidden inside the jokes):",
        roast,
    ]
    if before:
        lines += ["", "THE VC'S SCORES FOR THE ORIGINAL PITCH: " + json.dumps(before)]
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(lines)}]


_TAG = "[[SCORES"
_TAG_RE = re.compile(r"\[\[\s*SCORES", re.I)
_PITCH_HEADING_RE = re.compile(r"^(?:#{2,4}\s*|\*\*)[^\n]*elevator[^\n]*\n", re.I | re.M)
_NEXT_HEADING_RE = re.compile(r"^#{2,4}\s", re.M)


def strip_scores(text: str) -> str:
    """Hide the machine-readable score line, even while it is still being typed."""
    m = _TAG_RE.search(text)
    if m:
        text = text[: m.start()]
    else:
        tail = text[-len(_TAG):].upper()
        for n in range(len(_TAG) - 1, 0, -1):  # half-typed tag at the very end, e.g. "[[SC"
            if tail.endswith(_TAG[:n]):
                text = text[:-n]
                break
    return re.sub(r"\s*`{3}\w*\s*$", "", text).rstrip()


def parse_scores(text: str) -> dict[str, int] | None:
    """Read the trailing score line. Returns None when the model did not follow the format."""
    m = _TAG_RE.search(text)
    if not m:
        return None
    blob = re.search(r"\{.*?\}", text[m.end():], flags=re.S)
    if not blob:
        return None
    try:
        raw = json.loads(blob.group(0))
        return {k: max(0, min(100, int(round(float(raw[k]))))) for k in SCORE_KEYS}
    except (ValueError, KeyError, TypeError):
        return None


def split_pitch(text: str) -> tuple[str, str]:
    """Split the mentor's answer into (everything else, the elevator pitch)."""
    m = _PITCH_HEADING_RE.search(text)
    if not m:
        return text.strip(), ""
    body, pitch = text[: m.start()].rstrip(), text[m.end():]
    nxt = _NEXT_HEADING_RE.search(pitch)
    if nxt:
        pitch = pitch[: nxt.start()]
    pitch = pitch.replace("**", "").strip().strip("\"“”").strip()
    return body, pitch


# ======================================================================================
# 5. Streaming call to NVIDIA Nemotron on Nebius
# ======================================================================================
def stream_completion(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    slot: Any,
    *,
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict[str, float]]:
    """Stream one chat completion into `slot` with a typewriter effect.

    Returns the raw text (including the trailing score line) and simple speed stats.
    """
    started = time.perf_counter()
    stream = client.chat.completions.create(
        model=model,  # explicit call to the NVIDIA Nemotron model
        messages=messages,
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
        stream=True,
    )
    text, pieces, reported, last_paint = "", 0, 0, 0.0
    try:
        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None and getattr(usage, "completion_tokens", None):
                reported = usage.completion_tokens
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content
            if not piece:
                continue
            text += piece
            pieces += 1
            now = time.perf_counter()
            if now - last_paint > 0.05:  # throttle repaints
                slot.markdown(strip_scores(text) + " ▌")
                last_paint = now
    finally:
        slot.markdown(strip_scores(text))  # always remove the cursor, even after an error
    if not text.strip():
        raise PipelineError(
            "The model returned an empty answer",
            "Check the model ID and base URL (sidebar, then Test connection) and run it again.",
        )
    elapsed = max(time.perf_counter() - started, 1e-6)
    return text, {"seconds": elapsed, "tokens": reported or pieces}


# ======================================================================================
# 6. Page assets
# ======================================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500..800&display=swap');
h1, h2, h3, h4, [data-testid="stMetricValue"], .hero-title, .stage-name, .pitch-quote {
  font-family: 'Bricolage Grotesque', 'Segoe UI', system-ui, sans-serif !important;
}
.block-container { padding-top: 2.2rem; max-width: 1240px; }
footer, .stDeployButton, [data-testid="stAppDeployButton"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
.hero { padding: 2.3rem 2.6rem 2rem; margin-bottom: 1.4rem; border-radius: 14px; border-left: 8px solid #FF5A2C; background: #17120F; color: #F4EFE6; }
.hero-title { margin: 0 0 .6rem; font-size: clamp(2.6rem, 5.4vw, 4.2rem); font-weight: 800; line-height: 1; letter-spacing: -.035em; color: #F4EFE6; }
.hero-sub { max-width: 40rem; font-size: 1.15rem; line-height: 1.55; color: #BDB3A7; }
.hero-meta { margin-top: 1.3rem; font-size: .92rem; color: #948B80; }
.hero-meta b { color: #76B900; font-weight: 700; }
.st-key-input_card { padding: 1.4rem 1.6rem 1.2rem; border-radius: 14px; border: 1px solid rgba(128,128,128,.3); background: rgba(128,128,128,.06); }
.st-key-roast_card, .st-key-rebuild_card { padding: 1.2rem 1.5rem 1rem; border-radius: 8px; border: 1px solid rgba(128,128,128,.25); background: rgba(128,128,128,.05); }
.st-key-roast_card { border-left: 6px solid #FF5A2C; }
.st-key-rebuild_card { border-left: 6px solid #76B900; }
.st-key-roast_card h3, .st-key-rebuild_card h3 { margin: 1.1rem 0 .25rem; padding: 0; font-size: 1.02rem; font-weight: 700; }
.st-key-roast_card [data-testid="stMarkdownContainer"] > h3:first-child, .st-key-rebuild_card [data-testid="stMarkdownContainer"] > h3:first-child { margin-top: 0; }
.stage-name { font-size: 1.4rem; font-weight: 700; letter-spacing: -.015em; line-height: 1.2; }
.stage-who { margin-bottom: .7rem; font-size: .9rem; opacity: .65; }
.st-key-pitch_card { margin-top: .6rem; padding: 1.8rem 2.2rem 1.5rem; border-radius: 22px; background: #17120F; }
.st-key-pitch_card .stage-name { color: #F4EFE6; }
.st-key-pitch_card .stage-who { color: #948B80; opacity: 1; }
.pitch-quote { margin: .4rem 0 1rem; max-width: 52rem; font-size: 1.55rem; font-weight: 500; line-height: 1.5; color: #F4EFE6; white-space: pre-wrap; }
.pitch-quote::before { content: "\\201C"; display: block; height: 2.2rem; font-size: 5rem; font-weight: 800; line-height: 1; color: #76B900; }
.pitch-meta { font-size: .9rem; color: #948B80; }
[data-testid="stMetric"] { padding: .9rem 1.1rem; border-radius: 10px; border: 1px solid rgba(128,128,128,.3); background: rgba(128,128,128,.05); }
[data-testid="stMetricValue"] { font-weight: 700; }
.stButton > button, .stDownloadButton > button { width: 100%; border-radius: 10px; font-weight: 600; padding: .6rem 1rem; transition: transform .12s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] { color: #fff; border: 0; background: #FF5A2C; }
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover { background: #FF4313; color: #fff; }
[data-testid="stExpander"] details { border-radius: 10px; }
.empty-hint { margin-top: 1rem; padding: 1.2rem 1.5rem; border-radius: 10px; border: 1px dashed rgba(128,128,128,.5); text-align: center; }
@media (prefers-reduced-motion: reduce) { .stButton > button, .stDownloadButton > button { transition: none; } }
</style>
"""

HERO = """
<div class="hero">
<div class="hero-title">Roast my startup</div>
<div class="hero-sub">You pitch it. Chad, a merciless VC, tears it apart. Then Maya, a startup mentor, rebuilds it into a pitch that can raise.</div>
<div class="hero-meta">Runs on <b>NVIDIA Nemotron</b> through <b>Nebius AI</b></div>
</div>
"""


# ======================================================================================
# 7. UI building blocks
# ======================================================================================
def card(key: str):
    """A container the CSS can style. `key=` needs Streamlit 1.39+, older versions get a plain container."""
    try:
        return st.container(key=key)
    except TypeError:
        return st.container()


def init_state() -> None:
    for name, value in {"idea": "", "audience": "", "tagline": "", "intensity": DEFAULT_INTENSITY}.items():
        st.session_state.setdefault(name, value)


def load_example() -> None:
    """Button callback: fill the form with a sample startup (different from the current one)."""
    pool = [e for e in EXAMPLES if e["idea"] != st.session_state.get("idea")] or EXAMPLES
    st.session_state.update(random.choice(pool))


def test_connection(cfg: dict[str, str]) -> None:
    """Sidebar health check: reach Nebius and confirm the Nemotron model is available."""
    try:
        client = prepare_client(cfg)
        with st.spinner("Contacting Nebius…"):
            ids = sorted(m.id for m in client.models.list())
    except Exception as exc:
        show_error(exc, cfg)
        return
    if cfg["model"] in ids:
        st.success(f"Connected. `{cfg['model']}` is available.")
        return
    nemotron = [i for i in ids if "nemotron" in i.lower()]
    if nemotron:
        st.warning(f"Connected, but `{cfg['model']}` is not in the catalog. Nemotron models on this endpoint:")
        st.code("\n".join(nemotron), language=None)
    elif ids:
        st.warning("Connected, but no Nemotron model is listed for this account.")
    else:
        st.info("Connected, but the endpoint returned no model list. Run a roast to test the model itself.")


def render_sidebar() -> dict[str, str]:
    with st.sidebar:
        st.markdown("## Connection")
        env_key = default_api_key()
        typed = st.text_input(
            "Nebius API key",
            type="password",
            placeholder="Using NEBIUS_API_KEY from the environment" if env_key else "Paste your Nebius API key",
            help="Kept in memory for this session only. Set NEBIUS_API_KEY in your terminal to avoid typing it on camera.",
        )
        api_key = clean_key(typed) or env_key
        if api_key:
            st.success("API key ready" + ("" if typed else " (from environment)"), icon="🔑")
        else:
            st.warning("Add your Nebius API key to start", icon="🔒")

        with st.expander("Advanced settings"):
            model = st.text_input(
                "Model ID",
                value=os.getenv("NEBIUS_MODEL", NEMOTRON_MODEL),
                help="Must be a NVIDIA Nemotron model on Nebius. Use Test connection to see the exact IDs.",
            )
            base_url = st.text_input(
                "API base URL",
                value=os.getenv("NEBIUS_BASE_URL", DEFAULT_BASE_URL),
                help="The OpenAI-compatible Nebius endpoint.",
            )
        cfg = {"api_key": api_key, "model": model.strip(), "base_url": base_url.strip()}

        if st.button("Test connection", key="test_btn"):
            test_connection(cfg)

        st.divider()
        st.markdown("### How it works")
        st.markdown(
            "1. **Chad**, a VC persona, roasts your idea and grades it.\n"
            "2. **Maya**, a mentor persona, fixes every flaw he found.\n"
            "3. You get a new scorecard, a strategy and a 30-second pitch.\n\n"
            "Both stages are chained calls to NVIDIA Nemotron on Nebius."
        )
        st.caption(f"Model: {cfg['model'] or 'not set'}")
    return cfg


def render_inputs() -> tuple[Pitch, bool]:
    with card("input_card"):
        st.markdown("#### Pitch your startup")
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.text_area(
                "Business idea",
                key="idea",
                height=168,
                max_chars=1200,
                placeholder="What does it do, how does it make money, and why will people care?",
            )
        with right:
            st.text_input("Target audience", key="audience", max_chars=200, placeholder="Who is it for?")
            st.text_input("Current tagline", key="tagline", max_chars=200, placeholder="Your best one-liner today")
        c1, c2, c3 = st.columns([3, 1.3, 1.5], gap="medium", vertical_alignment="bottom")
        with c1:
            st.select_slider("Roast intensity", options=list(INTENSITY), key="intensity")
        with c2:
            st.button("Load a sample startup", key="sample_btn", on_click=load_example)
        with c3:
            run = st.button("Roast my startup", key="run_btn", type="primary")
    pitch = Pitch(
        idea=st.session_state["idea"].strip(),
        audience=st.session_state["audience"].strip(),
        tagline=st.session_state["tagline"].strip(),
        intensity=st.session_state["intensity"],
    )
    return pitch, run


def stage_header(name: str, who: str) -> None:
    st.markdown(
        f'<div class="stage-name">{html.escape(name)}</div><div class="stage-who">{html.escape(who)}</div>',
        unsafe_allow_html=True,
    )


def build_layout() -> Slots:
    """Lay out the results area once; the slots are filled while streaming and at the end."""
    status = st.container()
    body = st.empty()
    with body.container():
        metrics = st.empty()
        render_scorecard(metrics, None, None)
        left, right = st.columns(2, gap="large")
        with left, card("roast_card"):
            stage_header("Stage 1: The roast", "Chad Sterling, venture capitalist")
            roast = st.empty()
        with right, card("rebuild_card"):
            stage_header("Stage 2: The rebuild", "Maya, startup mentor")
            rebuild = st.empty()
        pitch = st.container()
    return Slots(status=status, body=body, metrics=metrics, roast=roast, rebuild=rebuild, pitch=pitch)


def _avg(scores: dict[str, int]) -> int:
    return round(sum(scores.values()) / len(scores))


def _metric(col: Any, label: str, now: int, was: int | None) -> None:
    if was is None:
        col.metric(label, f"{now}/100")
    else:
        diff = now - was
        col.metric(label, f"{now}/100", f"{diff:+d} (was {was})", delta_color="normal" if diff else "off")


def render_scorecard(slot: Any, before: dict[str, int] | None, after: dict[str, int] | None) -> None:
    """Scoreboard: empty at first, then the VC's verdict, then the rebuilt scores with deltas."""
    shown = after or before
    with slot.container():
        cols = st.columns(len(SCORE_KEYS) + 1)
        if not shown:
            for col, label in zip(cols, ["Pitch score", *SCORE_LABELS.values()]):
                col.metric(label, "—")
            st.caption("The scoreboard fills in as the agent works.")
            return
        was = before if after else None
        _metric(cols[0], "Pitch score", _avg(shown), _avg(was) if was else None)
        for col, key in zip(cols[1:], SCORE_KEYS):
            _metric(col, SCORE_LABELS[key], shown[key], was[key] if was else None)
        st.caption(
            "The mentor's grade for the rebuilt pitch, compared with the VC's grade for your original. "
            "AI-generated opinions: a fun signal, not market validation."
            if after
            else "The VC's grade for your original pitch. Let's see if the mentor can fix it."
        )


def build_report(result: dict[str, Any]) -> str:
    p = result["pitch"]
    parts = [
        "# Roast my startup: report",
        f"Generated with `{result['model']}` on Nebius AI.",
        "## The original pitch",
        f"- **Idea:** {p['idea']}",
        f"- **Audience:** {p['audience']}",
        f"- **Tagline:** {p['tagline']}",
        "## Stage 1: The roast",
        result["roast"],
        "## Stage 2: The rebuild",
        result["rebuild"],
    ]
    if result["elevator"]:
        parts += ["### 🎤 30-second elevator pitch", result["elevator"]]
    return "\n\n".join(parts) + "\n"


def render_pitch(slot: Any, result: dict[str, Any]) -> None:
    """The elevator pitch, speed stats and the download button."""
    pitch = result["elevator"]
    with slot:
        if pitch:
            words = len(pitch.split())
            with card("pitch_card"):
                stage_header("Your 30-second pitch", "Written to be said out loud")
                st.markdown(f'<div class="pitch-quote">{html.escape(pitch)}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="pitch-meta">{words} words, about {max(1, round(words / 2.5))} seconds spoken aloud</div>',
                    unsafe_allow_html=True,
                )
        s1, s2 = result["stats"]
        total = s1["seconds"] + s2["seconds"]
        tokens = int(s1["tokens"] + s2["tokens"])
        left, right = st.columns([3, 1], vertical_alignment="center")
        left.caption(
            f"Nemotron generated about {tokens} tokens in {total:.1f} seconds "
            f"(roughly {tokens / total:.0f} tokens per second) across two chained calls on Nebius."
        )
        right.download_button(
            "Download report",
            data=build_report(result),
            file_name="roast-my-startup-report.md",
            mime="text/markdown",
            key="download_btn",
        )


def present(result: dict[str, Any], slots: Slots) -> None:
    """Final, stable rendering of a finished result (also used after reruns)."""
    if result["before"] or result["after"]:
        render_scorecard(slots.metrics, result["before"], result["after"])
    else:
        slots.metrics.empty()
    slots.roast.markdown(result["roast"])
    slots.rebuild.markdown(result["rebuild"])
    render_pitch(slots.pitch, result)


# ======================================================================================
# 8. The two-stage agent run
# ======================================================================================
def handle_run(cfg: dict[str, str], pitch: Pitch) -> None:
    """Validate, run both stages with live streaming, then store and present the result."""
    try:
        validate_inputs(pitch)
        client = prepare_client(cfg)
    except Exception as exc:  # configuration problems are explained, never raised at the user
        show_error(exc, cfg)
        return

    st.session_state.pop("result", None)
    slots = build_layout()
    model = cfg["model"]
    roast_raw = rebuild_raw = roast = ""
    before: dict[str, int] | None = None
    s1 = s2 = {"seconds": 0.0, "tokens": 0}
    failure: BaseException | None = None

    with slots.status:
        with st.status("Stage 1: The VC is sharpening his knives…", expanded=True) as status:
            try:
                st.write(f"🔥 Chad Sterling is roasting your pitch with `{model}`")
                roast_raw, s1 = stream_completion(
                    client, model, roast_messages(pitch), slots.roast, temperature=0.95, max_tokens=1300
                )
                roast, before = strip_scores(roast_raw), parse_scores(roast_raw)
                render_scorecard(slots.metrics, before, None)
                st.write(f"✅ Roast delivered in {s1['seconds']:.1f}s")

                status.update(label="Stage 2: The mentor is rebuilding your pitch…")
                st.write("🧠 Maya is turning the wreckage into a pitch that can raise")
                rebuild_raw, s2 = stream_completion(
                    client, model, mentor_messages(pitch, roast, before), slots.rebuild, temperature=0.6, max_tokens=2500
                )
                st.write(f"✅ Rebuild delivered in {s2['seconds']:.1f}s")
                status.update(label=f"Done in {s1['seconds'] + s2['seconds']:.1f} seconds", state="complete", expanded=False)
            except Exception as exc:
                failure = exc
                status.update(label="The agent stopped before finishing", state="error", expanded=False)
        if failure is not None:
            show_error(failure, cfg)

    if failure is not None:
        if not roast_raw.strip():
            slots.body.empty()  # nothing useful to show: leave only the error message
        return

    rebuild_body, elevator = split_pitch(strip_scores(rebuild_raw))
    result = {
        "pitch": asdict(pitch),
        "model": model,
        "roast": roast,
        "rebuild": rebuild_body,
        "elevator": elevator,
        "before": before,
        "after": parse_scores(rebuild_raw),
        "stats": [s1, s2],
    }
    st.session_state["result"] = result
    present(result, slots)


# ======================================================================================
# 9. App entry point
# ======================================================================================
def main() -> None:
    st.set_page_config(
        page_title="Roast my startup | NVIDIA Nemotron on Nebius",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    cfg = render_sidebar()
    st.markdown(HERO, unsafe_allow_html=True)
    pitch, run_clicked = render_inputs()

    if run_clicked:
        handle_run(cfg, pitch)
    elif st.session_state.get("result"):
        present(st.session_state["result"], build_layout())
    else:
        st.markdown(
            '<div class="empty-hint">Describe your startup above and press <b>Roast my startup</b>. '
            "No idea yet? Press <b>Load a sample startup</b>.</div>",
            unsafe_allow_html=True,
        )
    st.caption("Built for the Nebius x NVIDIA Global AI Hackathon with Streamlit and NVIDIA Nemotron on Nebius AI.")


if __name__ == "__main__":
    main()
