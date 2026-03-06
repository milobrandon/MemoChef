"""Shared design tokens and small UI helpers.

Brand palette sourced from the Subtext Brand Guide (October 2023):
  Slate Gray     #2b2825   — primary dark background
  Everest Green  #16352e   — secondary dark, accent panels
  Birch          #a95818   — warm highlight / warning
  Brown          #512213   — deep accent
  Beige          #f7f1e3   — primary text on dark backgrounds
  Lime Green     #c1d100   — call-to-action, primary accent
"""

import streamlit as st

APP_TITLE = "Memo Automator"
APP_SUBTITLE = "Institutional-grade memo updates with reviewable automation."
APP_SLOGANS = [
    "From raw ingredients to a Michelin-star memo",
    "Precision-plated drafts for investment committee review",
    "Queue it. Review it. Serve it with traceable artifacts.",
]

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
.memo-hero-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.35fr) minmax(300px, 0.95fr);
    gap: 1.5rem;
    align-items: center;
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
.memo-slogan-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-top: 1rem;
}
.memo-slogan-pill {
    border: 1px solid rgba(193, 209, 0, 0.2);
    background: rgba(247, 241, 227, 0.05);
    color: var(--text);
    border-radius: 999px;
    padding: 0.42rem 0.8rem;
    font-size: 0.83rem;
    line-height: 1.2;
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

/* ── Shrimp chef scene ── */
.memo-chef-scene {
    position: relative;
    min-height: 240px;
    border: 1px solid var(--line);
    border-radius: 22px;
    background:
        radial-gradient(circle at top, rgba(193, 209, 0, 0.12), transparent 45%),
        linear-gradient(180deg, rgba(247, 241, 227, 0.04), rgba(247, 241, 227, 0.02));
    overflow: hidden;
    padding: 1rem 1rem 0.9rem;
}
.memo-chef-stage {
    position: absolute;
    left: 1rem;
    right: 1rem;
    bottom: 0.85rem;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(247, 241, 227, 0.22), transparent);
}
.memo-chef-row {
    position: relative;
    z-index: 2;
    display: flex;
    justify-content: center;
    align-items: flex-end;
    gap: 0.75rem;
    height: 100%;
}
.memo-chef {
    position: relative;
    width: 108px;
    text-align: center;
    animation: chef-bob 1.4s ease-in-out infinite;
}
.memo-chef:nth-child(2) {
    animation-delay: 0.2s;
}
.memo-chef:nth-child(3) {
    animation-delay: 0.4s;
}
.memo-chef-hat {
    position: absolute;
    top: 6px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 1.5rem;
    z-index: 3;
}
.memo-chef-body {
    display: inline-block;
    font-size: 4.7rem;
    line-height: 1;
    filter: drop-shadow(0 10px 18px rgba(0, 0, 0, 0.22));
}
.memo-chef-tool {
    position: absolute;
    top: 58px;
    right: 3px;
    font-size: 1.35rem;
    animation: chef-stir 0.9s ease-in-out infinite;
    transform-origin: bottom left;
}
.memo-chef-badge {
    position: absolute;
    left: 50%;
    bottom: 22px;
    transform: translateX(-50%);
    background: linear-gradient(135deg, var(--accent), var(--accent-strong));
    color: var(--slate-gray);
    border-radius: 999px;
    padding: 0.2rem 0.5rem;
    font-size: 0.62rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    box-shadow: 0 10px 18px rgba(193, 209, 0, 0.18);
}
.memo-chef-label {
    margin-top: 0.35rem;
    color: var(--muted);
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.memo-chef-flame {
    position: absolute;
    bottom: 52px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 0.95rem;
    animation: flame-flicker 0.7s ease-in-out infinite;
}
.memo-chef-steam {
    position: absolute;
    top: 32px;
    left: 50%;
    margin-left: 18px;
    font-size: 0.9rem;
    opacity: 0.7;
    animation: steam-rise 1.6s ease-out infinite;
}
.memo-chef-scene::before,
.memo-chef-scene::after {
    content: "";
    position: absolute;
    border-radius: 999px;
    background: rgba(193, 209, 0, 0.07);
    filter: blur(2px);
}
.memo-chef-scene::before {
    width: 120px;
    height: 120px;
    top: -30px;
    right: -10px;
}
.memo-chef-scene::after {
    width: 90px;
    height: 90px;
    bottom: 20px;
    left: -10px;
}
@keyframes chef-bob {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
}
@keyframes chef-stir {
    0%, 100% { transform: rotate(-14deg); }
    50% { transform: rotate(14deg); }
}
@keyframes steam-rise {
    0% { opacity: 0; transform: translate(-50%, 6px) scale(0.9); }
    25% { opacity: 0.7; }
    100% { opacity: 0; transform: translate(-50%, -18px) scale(1.3); }
}
@keyframes flame-flicker {
    0%, 100% { transform: translateX(-50%) scale(0.9); opacity: 0.8; }
    50% { transform: translateX(-50%) scale(1.15); opacity: 1; }
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

/* Suppress raw JSON widgets in the user-facing app */
[data-testid="stJson"] {
    display: none !important;
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
    .memo-hero-grid { grid-template-columns: 1fr; }
    .memo-stage-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .memo-chef-scene { min-height: 210px; }
    .memo-chef { width: 96px; }
    .memo-chef-body { font-size: 4rem; }
}
</style>
"""


def hero_html() -> str:
    stage_markup = "".join(
        f'<div class="memo-stage"><div class="memo-stage-num">0{i}</div><div class="memo-stage-label">{label}</div></div>'
        for i, label in enumerate(STAGES, start=1)
    )
    slogan_markup = "".join(
        f'<div class="memo-slogan-pill">{slogan}</div>'
        for slogan in APP_SLOGANS
    )
    chef_markup = "".join(
        (
            '<div class="memo-chef">'
            '<div class="memo-chef-hat">&#x1F9D1;&#x200D;&#x1F373;</div>'
            '<div class="memo-chef-flame">&#x1F525;</div>'
            '<div class="memo-chef-steam">&#x2668;&#xFE0F;</div>'
            f'<div class="memo-chef-tool">{tool}</div>'
            '<div class="memo-chef-body">&#x1F990;</div>'
            f'<div class="memo-chef-badge">{badge}</div>'
            f'<div class="memo-chef-label">{label}</div>'
            "</div>"
        )
        for tool, badge, label in [
            ("&#x1F944;", "SUBTEXT", "Lead Chef"),
            ("&#x1F9C2;", "BRANDED", "Sauce Station"),
            ("&#x1F37D;&#xFE0F;", "REVIEWED", "Pass Expeditor"),
        ]
    )
    return (
        '<div class="memo-hero">'
        '<div class="memo-hero-grid">'
        "<div>"
        '<div class="memo-kicker">Memo Operations Platform</div>'
        f'<div class="memo-title">{APP_TITLE}</div>'
        f'<div class="memo-subtitle">{APP_SUBTITLE}</div>'
        f'<div class="memo-slogan-row">{slogan_markup}</div>'
        "</div>"
        f'<div class="memo-chef-scene"><div class="memo-chef-row">{chef_markup}</div><div class="memo-chef-stage"></div></div>'
        "</div>"
        f'<div class="memo-stage-row">{stage_markup}</div>'
        "</div>"
    )


def render_hero() -> None:
    """Render the top hero with native Streamlit primitives."""
    with st.container():
        top_cols = st.columns([1.7, 1.0], vertical_alignment="center")
        with top_cols[0]:
            st.caption("MEMO OPERATIONS PLATFORM")
            st.title(APP_TITLE)
            st.caption(APP_SUBTITLE)
            slogan_cols = st.columns(len(APP_SLOGANS))
            for col, slogan in zip(slogan_cols, APP_SLOGANS):
                with col:
                    st.caption(slogan)
        with top_cols[1]:
            chef_cols = st.columns(3)
            for col, badge, label, tool in zip(
                chef_cols,
                ["SUBTEXT", "BRANDED", "REVIEWED"],
                ["Lead Chef", "Sauce Station", "Pass Expeditor"],
                ["🥄", "🧈", "🍽️"],
            ):
                with col:
                    st.markdown("### 🦐")
                    st.caption(f"🧑‍🍳 {tool}")
                    st.caption(badge)
                    st.caption(label)

        stage_cols = st.columns(len(STAGES))
        for index, (col, label) in enumerate(zip(stage_cols, STAGES), start=1):
            with col:
                st.caption(f"{index:02d}")
                st.markdown(f"**{label}**")


def info_card(title: str, copy: str) -> str:
    return (
        '<div class="memo-card">'
        f'<div class="memo-card-title">{title}</div>'
        f'<div class="memo-card-copy">{copy}</div>'
        "</div>"
    )
