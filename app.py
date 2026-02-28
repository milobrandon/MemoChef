#!/usr/bin/env python3
"""Memo Chef — Streamlit dashboard wrapping memo_automator.py."""

import logging
import os
import re
import tempfile
import time


import anthropic
import streamlit as st

from memo_automator import (
    apply_branding,
    apply_updates,
    chunk_memo_by_pages,
    create_backup,
    extract_memo_content,
    extract_proforma_data,
    extract_schedule_data,
    get_metric_mappings,
    load_config,
    normalize_layout,
    pre_validate_mappings,
    validate_mappings,
    write_change_log,
)

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="Memo Chef", page_icon="\U0001f9d1\u200d\U0001f373", layout="centered")


# ============================================================================
# PASSWORD GATE
# ============================================================================
def check_password() -> bool:
    """Show a login form and return True only when the correct password is entered."""
    if st.session_state.get("authenticated"):
        return True

    try:
        app_password = st.secrets["APP_PASSWORD"]
    except (KeyError, FileNotFoundError):
        st.error("APP_PASSWORD not configured in Streamlit secrets.")
        st.stop()

    st.title("\U0001f512 Memo Chef — Login")
    with st.form("login_form"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
    if submitted:
        if password == app_password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


if not check_password():
    st.stop()

st.title("\U0001f9d1\u200d\U0001f373 Memo Chef")
st.caption("From raw ingredients to a Michelin-star memo")

# ============================================================================
# ANIMATED SHRIMP HTML
# ============================================================================
CHEF_SHRIMP_HTML = """
<style>
@keyframes chef-bounce {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-8px); }
}
@keyframes stir {
  0%   { transform: rotate(-15deg); }
  50%  { transform: rotate(15deg); }
  100% { transform: rotate(-15deg); }
}
@keyframes flame-dance {
  0%, 100% { transform: scaleY(1) scaleX(1); opacity: 0.85; }
  25% { transform: scaleY(1.5) scaleX(0.8); opacity: 1; }
  50% { transform: scaleY(0.8) scaleX(1.2); opacity: 0.7; }
  75% { transform: scaleY(1.3) scaleX(0.9); opacity: 1; }
}
@keyframes steam-rise {
  0% { opacity: 0.7; transform: translateY(0) scale(1); }
  100% { opacity: 0; transform: translateY(-55px) scale(2); }
}
@keyframes pan-tilt {
  0%, 100% { transform: rotate(0deg); }
  30% { transform: rotate(-2deg); }
  60% { transform: rotate(2deg); }
}
@keyframes sparkle {
  0%, 100% { opacity: 0; transform: scale(0.5) translateY(0); }
  50% { opacity: 1; transform: scale(1.2) translateY(-18px); }
}
.chef-scene {
  display: flex; justify-content: center; align-items: flex-end;
  height: 240px; margin: 10px 0 15px 0; user-select: none;
  position: relative;
}
.chef-group {
  position: relative;
  display: flex; align-items: flex-end; gap: 0;
}
/* --- the shrimp chef character --- */
.shrimp-chef {
  position: relative; z-index: 5;
  animation: chef-bounce 1.1s ease-in-out infinite;
}
.chef-toque {
  font-size: 32px; position: absolute; z-index: 7;
  top: -30px; left: 50%; transform: translateX(-50%);
}
.shrimp-body { font-size: 80px; display: block; }
.chef-coat {
  font-size: 11px; position: absolute; bottom: 8px;
  left: 50%; transform: translateX(-50%);
  background: #fff; border: 2px solid #ddd; border-radius: 4px;
  padding: 1px 8px; white-space: nowrap; font-weight: bold;
  color: #333; letter-spacing: 0.5px; z-index: 6;
  box-shadow: 0 1px 3px rgba(0,0,0,0.12);
}
/* spatula arm */
.spatula {
  font-size: 38px; position: absolute; z-index: 8;
  top: 12px; right: -30px;
  animation: stir 0.7s ease-in-out infinite;
  transform-origin: bottom center;
}
/* --- pan + fire station --- */
.station {
  position: relative; margin-left: -10px; margin-bottom: -5px;
}
.pan-area {
  animation: pan-tilt 1.2s ease-in-out infinite;
  transform-origin: center bottom;
  position: relative; z-index: 3;
}
.pan-emoji { font-size: 70px; }
.flame-bit {
  position: absolute; font-size: 24px; z-index: 2;
  animation: flame-dance 0.35s ease-in-out infinite alternate;
}
.f1 { bottom: -12px; left: 8px; animation-delay: 0s; }
.f2 { bottom: -9px;  left: 28px; animation-delay: 0.12s; }
.f3 { bottom: -12px; left: 48px; animation-delay: 0.25s; }
.f4 { bottom: -9px;  left: 68px; animation-delay: 0.08s; }
.steam-puff {
  position: absolute; font-size: 18px; z-index: 7;
  animation: steam-rise 1.6s ease-out infinite;
}
.st1 { top: -15px; left: 10px; animation-delay: 0s; }
.st2 { top: -20px; left: 35px; animation-delay: 0.5s; }
.st3 { top: -12px; left: 60px; animation-delay: 1s; }
.sparkle {
  position: absolute; font-size: 14px; z-index: 9;
  animation: sparkle 1.3s ease-in-out infinite;
}
.sp1 { top: -25px; left: 5px;  animation-delay: 0.2s; }
.sp2 { top: -35px; left: 40px; animation-delay: 0.8s; }
.sp3 { top: -20px; left: 72px; animation-delay: 1.2s; }
</style>
<div class="chef-scene">
  <div class="chef-group">
    <div class="shrimp-chef">
      <div class="chef-toque">\U0001f9d1\u200d\U0001f373</div>
      <span class="shrimp-body">\U0001f990</span>
      <div class="chef-coat">CHEF</div>
      <div class="spatula">\U0001f944</div>
    </div>
    <div class="station">
      <div class="steam-puff st1">\u2668\ufe0f</div>
      <div class="steam-puff st2">\u2668\ufe0f</div>
      <div class="steam-puff st3">\u2668\ufe0f</div>
      <div class="sparkle sp1">\u2728</div>
      <div class="sparkle sp2">\u2728</div>
      <div class="sparkle sp3">\u2728</div>
      <div class="pan-area">
        <div class="pan-emoji">\U0001f373</div>
      </div>
      <div class="flame-bit f1">\U0001f525</div>
      <div class="flame-bit f2">\U0001f525</div>
      <div class="flame-bit f3">\U0001f525</div>
      <div class="flame-bit f4">\U0001f525</div>
    </div>
  </div>
</div>
"""


# ============================================================================
# API KEY RESOLUTION
# ============================================================================
def _get_api_key() -> str | None:
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None


# ============================================================================
# LOG CAPTURE HANDLER
# ============================================================================
class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(self.format(record))


# ============================================================================
# MISE EN PLACE  (ingredient uploaders)
# ============================================================================
st.subheader("\U0001f52a Mise en Place")
col1, col2, col3 = st.columns(3)
memo_file = col1.file_uploader("The Memo (.pptx)", type=["pptx"])
proforma_file = col2.file_uploader("The Proforma (.xlsx / .xlsm)", type=["xlsx", "xlsm"])
schedule_file = col3.file_uploader("The Schedule (.mpp)", type=["mpp"])

property_name = st.text_input(
    "Property Name (as shown in proforma)",
    placeholder="e.g. EVER at Reston",
    help="Optional. Helps the Chef match metrics when a property has been rebranded.",
)

# ============================================================================
# CHEF'S PREFERENCES
# ============================================================================
with st.expander("\U0001f9c2 Chef's Preferences"):
    dry_run = st.checkbox("Tasting portion only (dry run -- preview without saving)")
    skip_validation = st.checkbox("Skip the sous-chef QA pass (express ticket)")

# ============================================================================
# FIRE BUTTON
# ============================================================================
if st.button(
    "\U0001f525  Fire!",
    type="primary",
    disabled=not (memo_file and proforma_file),
):
    # Clear previous results
    for key in ["memo_bytes", "log_bytes", "filename", "n_changes",
                "n_rejected", "n_missed", "log_lines"]:
        st.session_state.pop(key, None)

    api_key = _get_api_key()
    if not api_key:
        st.error(
            "86'd! ANTHROPIC_API_KEY not found. "
            "Add it to Streamlit secrets (.streamlit/secrets.toml or Cloud dashboard)."
        )
        st.stop()

    logger = logging.getLogger("memo_automator")
    log_handler = _LogCapture()
    log_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logger.addHandler(log_handler)

    with tempfile.TemporaryDirectory() as tmpdir:
      try:
        # Save uploaded files to tmpdir
        memo_path = os.path.join(tmpdir, memo_file.name)
        proforma_path = os.path.join(tmpdir, proforma_file.name)
        with open(memo_path, "wb") as f:
            f.write(memo_file.getvalue())
        with open(proforma_path, "wb") as f:
            f.write(proforma_file.getvalue())

        # Load config from script directory
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        cfg = load_config(config_path)

        # Initialize Anthropic client
        client = anthropic.Anthropic(
            api_key=api_key,
            max_retries=5,
            timeout=900.0,
        )

        # Progress bar (outside status so it's always visible)
        progress_bar = st.progress(0, text="\U0001f525 Firing up the pass...")

        with st.status("\U0001f525 Firing up the pass...", expanded=True) as status:
            # Show animated chef shrimp while processing
            anim_placeholder = st.empty()
            anim_placeholder.markdown(CHEF_SHRIMP_HTML, unsafe_allow_html=True)

            # Step a: Backup
            progress_bar.progress(3, text="\U0001f9ca Icing the original...")
            st.write("\U0001f9ca Icing the original -- backup on the rail...")
            backup_path = create_backup(memo_path, tmpdir)

            # Step b: Extract proforma
            progress_bar.progress(8, text="\U0001f52a Breaking down the proforma...")
            st.write("\U0001f52a Breaking down the proforma...")
            try:
                proforma_data = extract_proforma_data(proforma_path, cfg)
            except Exception as e:
                st.error(f"Could not read the proforma file. Is it a valid .xlsx/.xlsm? ({e})")
                st.stop()

            # Step b2: Extract schedule (optional)
            if schedule_file:
                progress_bar.progress(12, text="\U0001f4c5 Reading the schedule...")
                st.write("\U0001f4c5 Reading the schedule...")
                schedule_path = os.path.join(tmpdir, schedule_file.name)
                with open(schedule_path, "wb") as f:
                    f.write(schedule_file.getvalue())
                try:
                    schedule_data = extract_schedule_data(schedule_path, cfg)
                    proforma_data += "\n\n" + schedule_data
                except Exception as e:
                    st.error(f"Could not read the schedule. Is it a valid .mpp? ({e})")
                    st.stop()

            # Step c: Extract memo
            progress_bar.progress(15, text="\U0001f4dc Reading the old ticket...")
            st.write("\U0001f4dc Reading the old ticket...")
            try:
                memo_content = extract_memo_content(memo_path, cfg)
            except Exception as e:
                st.error(f"Could not read the memo file. Is it a valid .pptx? ({e})")
                st.stop()

            # Step d: Map metrics (mirrors main() batching logic exactly)
            # Progress: 15% -> 70% for mapping
            SEAR_START, SEAR_END = 15, 70
            progress_bar.progress(SEAR_START, text="\U0001f373 Searing the metrics...")
            st.write("\U0001f373 Searing the metrics (~1-2 min on high heat)...")
            BATCH_THRESHOLD = 80_000
            RATE_LIMIT_INTERVAL = 65
            prompt_size = len(proforma_data) + len(memo_content)

            if prompt_size > BATCH_THRESHOLD:
                memo_chunks = chunk_memo_by_pages(memo_content, pages_per_chunk=3)
                n_chunks = len(memo_chunks)
                mappings = {"table_updates": [], "text_updates": [], "row_inserts": []}
                last_api_call = 0

                for i, chunk in enumerate(memo_chunks, 1):
                    pct = SEAR_START + int((SEAR_END - SEAR_START) * i / n_chunks)
                    progress_bar.progress(
                        pct,
                        text=f"\U0001f373 Searing batch {i}/{n_chunks}...",
                    )

                    if i > 1 and last_api_call > 0:
                        elapsed = time.time() - last_api_call
                        wait = RATE_LIMIT_INTERVAL - elapsed
                        if wait > 0:
                            time.sleep(wait)

                    last_api_call = time.time()
                    try:
                        batch = get_metric_mappings(client, proforma_data, chunk, cfg,
                                                   property_name=property_name)
                    except (anthropic.APIError, anthropic.APIConnectionError, Exception) as e:
                        st.warning(f"Batch {i}/{n_chunks} failed: {e}")
                        batch = {"table_updates": [], "text_updates": [],
                                 "row_inserts": [], "_truncated": True}

                    if batch.pop("_truncated", False):
                        covered_pages = set()
                        for e in batch.get("table_updates", []):
                            covered_pages.add(e.get("page"))
                        for e in batch.get("text_updates", []):
                            covered_pages.add(e.get("page"))
                        for e in batch.get("row_inserts", []):
                            covered_pages.add(e.get("page"))
                        mappings["table_updates"].extend(batch.get("table_updates", []))
                        mappings["text_updates"].extend(batch.get("text_updates", []))
                        mappings["row_inserts"].extend(batch.get("row_inserts", []))

                        sub_chunks = chunk_memo_by_pages(chunk, pages_per_chunk=1)
                        for j, sub_chunk in enumerate(sub_chunks, 1):
                            sub_pages = set(
                                int(m) for m in re.findall(r"PAGE (\d+)", sub_chunk)
                            )
                            if sub_pages and sub_pages.issubset(covered_pages):
                                continue

                            elapsed = time.time() - last_api_call
                            wait = RATE_LIMIT_INTERVAL - elapsed
                            if wait > 0:
                                time.sleep(wait)

                            last_api_call = time.time()
                            try:
                                sub_batch = get_metric_mappings(
                                    client, proforma_data, sub_chunk, cfg,
                                    property_name=property_name,
                                )
                            except (anthropic.APIError, anthropic.APIConnectionError, Exception) as e:
                                st.warning(f"Sub-batch {j} retry failed: {e}")
                                continue
                            sub_batch.pop("_truncated", False)
                            mappings["table_updates"].extend(
                                sub_batch.get("table_updates", [])
                            )
                            mappings["text_updates"].extend(
                                sub_batch.get("text_updates", [])
                            )
                            mappings["row_inserts"].extend(
                                sub_batch.get("row_inserts", [])
                            )
                    else:
                        mappings["table_updates"].extend(batch.get("table_updates", []))
                        mappings["text_updates"].extend(batch.get("text_updates", []))
                        mappings["row_inserts"].extend(batch.get("row_inserts", []))
            else:
                mappings = get_metric_mappings(client, proforma_data, memo_content, cfg,
                                               property_name=property_name)
                mappings.pop("_truncated", None)

            progress_bar.progress(70, text="\U0001f373 Reducing the sauce...")

            # Strip no-op entries (old == new)
            mappings["table_updates"] = [
                e for e in mappings["table_updates"]
                if e.get("old_value") != e.get("new_value")
            ]
            mappings["text_updates"] = [
                e for e in mappings["text_updates"]
                if e.get("old_text") != e.get("new_text")
            ]

            # Pre-validate (local Python check)
            mappings = pre_validate_mappings(mappings, memo_content)

            # Step e: Validate (Claude QA pass) — 72% -> 88%
            if skip_validation:
                progress_bar.progress(88, text="\U0001f9af Sous-chef on break...")
                st.write("\U0001f9af Sous-chef on break -- skipping QA...")
                validated = mappings
                validated.setdefault("rejected", [])
                validated.setdefault("missed", [])
            else:
                progress_bar.progress(72, text="\U0001f9d1\u200d\U0001f373 Sous-chef tasting...")
                st.write("\U0001f9d1\u200d\U0001f373 Sous-chef tasting for quality...")
                validated = validate_mappings(
                    client, mappings, proforma_data, memo_content, cfg,
                    property_name=property_name,
                )
                progress_bar.progress(88, text="\U0001f9d1\u200d\U0001f373 QA complete...")

            # Step f: Apply updates — 90%
            progress_bar.progress(90, text="\U0001f37d\ufe0f Plating the dish...")
            st.write("\U0001f37d\ufe0f Plating the dish...")
            changes = apply_updates(memo_path, validated, dry_run=dry_run)

            # Step g: Apply Subtext branding — 93%
            progress_bar.progress(93, text="\U0001f3a8 Applying Subtext branding...")
            st.write("\U0001f3a8 Applying Subtext branding...")
            theme_path = cfg.get("branding", {}).get("theme_path", "")
            if not theme_path:
                # Default: look for theme beside this script
                theme_path = os.path.join(os.path.dirname(__file__), "Subtext Brand Theme.thmx")
            if os.path.exists(theme_path) and not dry_run:
                n_branded = apply_branding(memo_path, theme_path, cfg)
                st.write(f"\U0001f3a8 Branded {n_branded} text runs")
            elif not os.path.exists(theme_path):
                st.warning("Subtext Brand Theme not found — skipping branding")

            # Step g2: Normalize layout — 95%
            if not dry_run:
                progress_bar.progress(95, text="Aligning layout...")
                st.write("\U0001f4d0 Aligning layout...")
                layout_summary = normalize_layout(memo_path, cfg)
                log.info("Layout normalized: %s", layout_summary)
                st.write(f"\U0001f4d0 Layout healed: {layout_summary['titles_snapped']} titles, "
                         f"{layout_summary['page_numbers_snapped']} page numbers snapped")

            # Step h: Write change log — 96%
            progress_bar.progress(96, text="\U0001f4cb Printing the ticket...")
            st.write("\U0001f4cb Printing the ticket...")
            log_path = write_change_log(
                tmpdir, changes, validated, memo_path, proforma_path, backup_path
            )

            n_changes = len(changes)
            progress_bar.progress(100, text=f"\U0001f31f Order up! {n_changes} updates plated.")
            status.update(
                label=f"\U0001f31f Order up! {n_changes} updates plated.",
                state="complete",
            )

        # Read output bytes into memory before tmpdir is cleaned up
        with open(memo_path, "rb") as f:
            memo_bytes = f.read()
        with open(log_path, "rb") as f:
            log_bytes = f.read()

        n_rejected = len(validated.get("rejected", []))
        n_missed = len(validated.get("missed", []))

        # Persist results in session state so download buttons survive reruns
        st.session_state["memo_bytes"] = memo_bytes
        st.session_state["log_bytes"] = log_bytes
        st.session_state["filename"] = memo_file.name
        st.session_state["n_changes"] = n_changes
        st.session_state["n_rejected"] = n_rejected
        st.session_state["n_missed"] = n_missed
        st.session_state["log_lines"] = log_handler.lines[:]

      except Exception as e:
        st.error(f"In the weeds! {e}")
        raise

      finally:
        # Always clear animation and clean up logging handler
        try:
            anim_placeholder.empty()
        except Exception:
            pass
        logger.removeHandler(log_handler)

# ============================================================================
# SERVICE WINDOW  (results -- persists from session_state across reruns)
# ============================================================================
if "memo_bytes" in st.session_state:
    n_changes = st.session_state["n_changes"]
    n_rejected = st.session_state["n_rejected"]
    n_missed = st.session_state["n_missed"]
    memo_bytes = st.session_state["memo_bytes"]
    log_bytes = st.session_state["log_bytes"]
    filename = st.session_state["filename"]
    log_lines = st.session_state["log_lines"]

    st.success(f"\U0001f37d\ufe0f Chef's kiss! Your memo is ready for service. {n_changes} updates plated.")

    dl_col1, dl_col2 = st.columns(2)
    dl_col1.download_button(
        "\u2b07\ufe0f Pick Up -- Updated Memo",
        memo_bytes,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )
    dl_col2.download_button(
        "\u2b07\ufe0f Grab the Ticket -- Change Log",
        log_bytes,
        file_name="CHANGE_LOG.md",
        mime="text/markdown",
    )

    stat_col1, stat_col2, stat_col3 = st.columns(3)
    stat_col1.metric("Plated", n_changes)
    stat_col2.metric("Sent back", n_rejected)
    stat_col3.metric("Needs garnish", n_missed)

    with st.expander("\U0001f4cb Full kitchen ticket"):
        st.code("\n".join(log_lines), language=None)
