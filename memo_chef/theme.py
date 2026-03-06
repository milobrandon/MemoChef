"""Shared design tokens and small UI helpers.

Brand palette sourced from the Subtext Brand Guide (October 2023):
  Slate Gray     #2b2825   — primary dark background
  Everest Green  #16352e   — secondary dark, accent panels
  Birch          #a95818   — warm highlight / warning
  Brown          #512213   — deep accent
  Beige          #f7f1e3   — primary text on dark backgrounds
  Lime Green     #c1d100   — call-to-action, primary accent
"""

APP_TITLE = "Memo Automator"
APP_SUBTITLE = "Institutional-grade memo updates with reviewable automation."

STAGES = [
    "Upload",
    "Validate",
    "Extract",
    "Map",
    "Review",
    "Export",
]


def app_css() -> str:
    return """
<style>
:root {
    /* ── Subtext Brand Palette (Oct 2023) ── */
    --slate-gray: #2b2825;
    --everest-green: #16352e;
    --birch: #a95818;
    --brown: #512213;
    --beige: #f7f1e3;
    --lime-green: #c1d100;

    /* ── Semantic tokens derived from palette ── */
    --bg: var(--slate-gray);
    --bg-elevated: #322f2b;
    --panel: rgba(43, 40, 37, 0.88);
    --panel-soft: rgba(247, 241, 227, 0.06);
    --text: var(--beige);
    --muted: #bfb8a8;
    --line: rgba(247, 241, 227, 0.14);
    --accent: var(--lime-green);
    --accent-strong: #d4e82a;
    --success: var(--lime-green);
    --warning: var(--birch);
    --danger: #d44;
    --radius: 18px;
    --shadow: 0 20px 50px rgba(22, 53, 46, 0.30);
}

/* ── Global background & text ── */
.stApp {
    background:
        radial-gradient(ellipse at 20% -10%, rgba(22, 53, 46, 0.45), transparent 50%),
        radial-gradient(ellipse at 80% 105%, rgba(193, 209, 0, 0.06), transparent 35%),
        var(--bg) !important;
    color: var(--text) !important;
}
[data-testid="stHeader"] {
    background: transparent !important;
}
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ── Hero banner ── */
.memo-hero {
    background: linear-gradient(135deg,
        rgba(43, 40, 37, 0.96),
        rgba(22, 53, 46, 0.88));
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 1.75rem 1.75rem 1.5rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.25rem;
}
.memo-kicker {
    color: var(--accent);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.memo-title {
    color: var(--text);
    font-size: 2.2rem;
    line-height: 1.1;
    font-weight: 700;
    margin: 0.5rem 0 0.65rem;
}
.memo-subtitle {
    color: var(--muted);
    font-size: 1rem;
    max-width: 58rem;
}

/* ── Stage indicator pills ── */
.memo-stage-row {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 0.75rem;
    margin-top: 1.4rem;
}
.memo-stage {
    border: 1px solid var(--line);
    background: var(--panel-soft);
    border-radius: 14px;
    padding: 0.8rem 0.9rem;
    transition: border-color 0.2s, background 0.2s;
}
.memo-stage:hover {
    border-color: rgba(193, 209, 0, 0.3);
    background: rgba(193, 209, 0, 0.06);
}
.memo-stage-num {
    color: var(--accent);
    font-size: 0.75rem;
    font-weight: 700;
}
.memo-stage-label {
    color: var(--text);
    font-size: 0.95rem;
    font-weight: 600;
    margin-top: 0.2rem;
}

/* ── Info cards ── */
.memo-card {
    border: 1px solid var(--line);
    background: var(--panel);
    border-radius: var(--radius);
    padding: 1rem 1.1rem;
    box-shadow: var(--shadow);
}
.memo-card-title {
    color: var(--text);
    font-size: 1rem;
    font-weight: 650;
    margin-bottom: 0.25rem;
}
.memo-card-copy {
    color: var(--muted);
    font-size: 0.92rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(34, 31, 28, 0.95) !important;
    border-right: 1px solid var(--line) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

/* ── Primary buttons (Lime Green CTA) ── */
.stButton > button[kind="primary"], button[kind="primary"],
.stForm button[kind="secondaryFormSubmit"] {
    background: linear-gradient(135deg, var(--accent), var(--accent-strong)) !important;
    color: var(--slate-gray) !important;
    border: none !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
    min-height: 2.75rem;
    box-shadow: 0 12px 24px rgba(193, 209, 0, 0.18);
    transition: filter 0.15s, box-shadow 0.15s;
}
.stButton > button[kind="primary"]:hover, button[kind="primary"]:hover {
    filter: brightness(1.08);
    box-shadow: 0 14px 32px rgba(193, 209, 0, 0.28);
}

/* ── Secondary / download buttons ── */
.stButton > button:not([kind="primary"]),
.stDownloadButton > button {
    border: 1px solid var(--line) !important;
    border-radius: 999px !important;
    color: var(--text) !important;
    background: rgba(247, 241, 227, 0.03) !important;
    transition: border-color 0.15s, background 0.15s;
}
.stButton > button:not([kind="primary"]):hover,
.stDownloadButton > button:hover {
    border-color: rgba(193, 209, 0, 0.35) !important;
    background: rgba(193, 209, 0, 0.06) !important;
}

/* ── Inputs, alerts, uploaders, expanders ── */
[data-testid="stFileUploader"],
[data-testid="stStatusWidget"],
[data-testid="stAlert"],
details {
    border-color: var(--line) !important;
    background: var(--panel) !important;
}

/* ── Progress bar (branded Lime Green with pulse) ── */
.stProgress > div > div > div,
section[data-testid="stSidebar"] .stProgress > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent-strong)) !important;
    transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}
.stProgress > div > div > div {
    animation: progress-glow 2s ease-in-out infinite;
}
@keyframes progress-glow {
    0%, 100% { box-shadow: 0 0 8px rgba(193, 209, 0, 0.25); }
    50% { box-shadow: 0 0 20px rgba(193, 209, 0, 0.50); }
}

/* ── Progress bar track (darker brand bg) ── */
.stProgress > div > div {
    background: rgba(22, 53, 46, 0.35) !important;
}

/* ── Metrics ── */
[data-testid="stMetricValue"] { color: var(--text) !important; }
[data-testid="stMetricLabel"], .stCaption { color: var(--muted) !important; }

/* ── Tabs (pill-style) ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    background: rgba(247, 241, 227, 0.03);
    border: 1px solid var(--line);
    padding-inline: 1rem;
    transition: background 0.15s, border-color 0.15s;
}
.stTabs [aria-selected="true"] {
    background: rgba(193, 209, 0, 0.12);
    border-color: rgba(193, 209, 0, 0.35);
}

/* ── Links ── */
a { color: var(--accent) !important; }

/* ── Select / inputs (Streamlit overrides) ── */
[data-testid="stSelectbox"],
[data-testid="stTextInput"],
[data-testid="stTextArea"] {
    color: var(--text);
}

/* ── Table / dataframe ── */
[data-testid="stDataFrame"] th {
    background: var(--panel) !important;
    color: var(--accent) !important;
}

/* ── Responsive ── */
@media (max-width: 980px) {
    .memo-stage-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
"""


def hero_html() -> str:
    stage_markup = "".join(
        f"""
        <div class="memo-stage">
            <div class="memo-stage-num">0{i}</div>
            <div class="memo-stage-label">{label}</div>
        </div>
        """
        for i, label in enumerate(STAGES, start=1)
    )
    return f"""
<div class="memo-hero">
  <div class="memo-kicker">Memo Operations Platform</div>
  <div class="memo-title">{APP_TITLE}</div>
  <div class="memo-subtitle">{APP_SUBTITLE}</div>
  <div class="memo-stage-row">{stage_markup}</div>
</div>
"""


def info_card(title: str, copy: str) -> str:
    return (
        '<div class="memo-card">'
        f'<div class="memo-card-title">{title}</div>'
        f'<div class="memo-card-copy">{copy}</div>'
        "</div>"
    )
