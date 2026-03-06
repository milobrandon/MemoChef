"""Shared design tokens and small UI helpers."""

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
    --bg: #111827;
    --bg-elevated: #172033;
    --panel: rgba(23, 32, 51, 0.82);
    --panel-soft: rgba(148, 163, 184, 0.08);
    --text: #f8fafc;
    --muted: #94a3b8;
    --line: rgba(148, 163, 184, 0.18);
    --accent: #c1d100;
    --accent-strong: #d9eb39;
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
    --radius: 18px;
    --shadow: 0 20px 50px rgba(15, 23, 42, 0.28);
}
.stApp {
    background:
        radial-gradient(circle at top right, rgba(193, 209, 0, 0.09), transparent 24%),
        linear-gradient(180deg, #0b1220 0%, #111827 100%) !important;
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
.memo-hero {
    background: linear-gradient(135deg, rgba(23,32,51,0.96), rgba(30,52,46,0.92));
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
section[data-testid="stSidebar"] {
    background: rgba(9, 13, 23, 0.92) !important;
    border-right: 1px solid var(--line) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text) !important;
}
.stButton > button[kind="primary"], button[kind="primary"],
.stForm button[kind="secondaryFormSubmit"] {
    background: linear-gradient(135deg, var(--accent), var(--accent-strong)) !important;
    color: #111827 !important;
    border: none !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
    min-height: 2.75rem;
    box-shadow: 0 12px 24px rgba(193, 209, 0, 0.22);
}
.stButton > button:not([kind="primary"]),
.stDownloadButton > button {
    border: 1px solid var(--line) !important;
    border-radius: 999px !important;
    color: var(--text) !important;
    background: rgba(255,255,255,0.03) !important;
}
[data-testid="stFileUploader"],
[data-testid="stStatusWidget"],
[data-testid="stAlert"],
details {
    border-color: var(--line) !important;
    background: var(--panel) !important;
}
.stProgress > div > div > div,
section[data-testid="stSidebar"] .stProgress > div > div {
    background: linear-gradient(135deg, var(--accent), var(--accent-strong)) !important;
}
[data-testid="stMetricValue"] { color: var(--text) !important; }
[data-testid="stMetricLabel"], .stCaption { color: var(--muted) !important; }
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--line);
    padding-inline: 1rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(193, 209, 0, 0.12);
    border-color: rgba(193, 209, 0, 0.35);
}
a { color: var(--accent) !important; }
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
