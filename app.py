#!/usr/bin/env python3
"""Memo Chef - Streamlit dashboard wrapping memo_automator.py."""

import logging
import os
import random
import re
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta


import anthropic
import psycopg2
import streamlit as st

from app_helpers import should_disable_fire_button, verify_password
from memo_automator import (
    apply_branding,
    apply_updates,
    chunk_memo_by_pages,
    create_backup,
    extract_market_data,
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
# CREDIT SYSTEM  (persistent Neon Postgres)
# ============================================================================
@st.cache_resource
def _get_db_conn():
    """Return a psycopg2 connection to the credits database (cached)."""
    conn = psycopg2.connect(st.secrets["CREDITS_DATABASE_URL"])
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS credit_usage ("
            "  username TEXT PRIMARY KEY,"
            "  week TEXT NOT NULL,"
            "  used INTEGER NOT NULL DEFAULT 0"
            ")"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS credit_charge_events ("
            "  username TEXT NOT NULL,"
            "  week TEXT NOT NULL,"
            "  run_id TEXT NOT NULL,"
            "  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
            "  PRIMARY KEY (username, week, run_id)"
            ")"
        )
    return conn


@contextmanager
def _db_cursor():
    """
    Yield a live DB cursor, retrying once with a fresh connection if the cached
    connection went stale (common on Streamlit Cloud cold/warm cycles).
    """
    conn = _get_db_conn()
    try:
        with conn.cursor() as cur:
            yield cur
        return
    except (psycopg2.InterfaceError, psycopg2.OperationalError):
        _get_db_conn.clear()
        conn = _get_db_conn()
        with conn.cursor() as cur:
            yield cur


def _current_week_start() -> str:
    """ISO Monday date for the current week."""
    today = datetime.now()
    monday = today - timedelta(days=today.weekday())
    return monday.strftime("%Y-%m-%d")


def _get_user_credits(username: str, credits_per_week: int) -> tuple[int, int]:
    """Return (used, remaining). Auto-resets on week rollover."""
    week = _current_week_start()
    with _db_cursor() as cur:
        cur.execute(
            "SELECT week, used FROM credit_usage WHERE username = %s",
            (username,),
        )
        row = cur.fetchone()
        if row is None or row[0] != week:
            # New user or new week — upsert with reset
            cur.execute(
                "INSERT INTO credit_usage (username, week, used) VALUES (%s, %s, 0) "
                "ON CONFLICT (username) DO UPDATE SET week = %s, used = 0",
                (username, week, week),
            )
            return 0, credits_per_week
        used = row[1]
    return used, max(0, credits_per_week - used)


def _consume_credit(username: str, credits_per_week: int, run_id: str | None = None) -> bool:
    """
    Consume one credit atomically for the current week.
    Returns False if the weekly limit is already reached.
    """
    week = _current_week_start()
    with _db_cursor() as cur:
        if run_id:
            cur.execute(
                "INSERT INTO credit_charge_events (username, week, run_id) "
                "VALUES (%s, %s, %s) "
                "ON CONFLICT DO NOTHING "
                "RETURNING run_id",
                (username, week, run_id),
            )
            inserted = cur.fetchone()
            if inserted is None:
                return True

        # Ensure row exists and is reset if week rolled over.
        cur.execute(
            "INSERT INTO credit_usage (username, week, used) VALUES (%s, %s, 0) "
            "ON CONFLICT (username) DO UPDATE SET "
            "  used = CASE WHEN credit_usage.week = %s THEN credit_usage.used ELSE 0 END, "
            "  week = %s",
            (username, week, week, week),
        )
        # Enforce weekly cap in the database to avoid race-condition overuse.
        cur.execute(
            "UPDATE credit_usage "
            "SET used = used + 1 "
            "WHERE username = %s AND week = %s AND used < %s "
            "RETURNING used",
            (username, week, credits_per_week),
        )
        charged = cur.fetchone() is not None
        if not charged and run_id:
            cur.execute(
                "DELETE FROM credit_charge_events WHERE username = %s AND week = %s AND run_id = %s",
                (username, week, run_id),
            )
        return charged


def _reset_user_credits(username: str) -> None:
    week = _current_week_start()
    with _db_cursor() as cur:
        cur.execute(
            "INSERT INTO credit_usage (username, week, used) VALUES (%s, %s, 0) "
            "ON CONFLICT (username) DO UPDATE SET week = %s, used = 0",
            (username, week, week),
        )


# ============================================================================
# AUTH GATE
# ============================================================================
def _get_users() -> dict:
    """Load user definitions from st.secrets['users']."""
    try:
        return dict(st.secrets["users"])
    except (KeyError, FileNotFoundError):
        return {}


def check_password() -> bool:
    """Per-user login: username + password.  Sets session_state on success."""
    if st.session_state.get("authenticated"):
        return True

    users = _get_users()
    if not users:
        st.error("No users configured in Streamlit secrets.")
        st.stop()

    _LOGIN_SLOGANS = [
        "The only James Beard Award Winning Chef in Memo City!",
        "Serving memos fresh out the kitchen.",
        "No reservations needed — just credentials.",
        "Where every memo is a chef's kiss.",
    ]
    st.title("\U0001f512 Memo Chef — Login")
    st.caption(random.choice(_LOGIN_SLOGANS))
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
    if submitted:
        user_cfg = users.get(username)
        if user_cfg and verify_password(password, user_cfg["password_hash"]):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["role"] = user_cfg.get("role", "user")
            st.session_state["credits_per_week"] = int(
                user_cfg.get("credits_per_week", 5)
            )
            st.rerun()
        else:
            st.error("Invalid username or password.")
    st.stop()


if not check_password():
    st.stop()

# ============================================================================
# SIDEBAR — user info, credits, admin panel
# ============================================================================
_username = st.session_state["username"]
_role = st.session_state["role"]
_credits_per_week = st.session_state["credits_per_week"]
_credits_error = None
try:
    _used, _remaining = _get_user_credits(_username, _credits_per_week)
except Exception as e:
    _used, _remaining = 0, 0
    _credits_error = str(e)

with st.sidebar:
    st.markdown(f"**{_username}** &nbsp; `{_role}`")
    if _credits_error:
        st.warning("Credits service unavailable. Runs are temporarily disabled.")
        st.caption("Credits: unavailable")
        if st.button("Retry credits service"):
            _get_db_conn.clear()
            st.rerun()
    else:
        st.caption(f"Credits: {_remaining} / {_credits_per_week} remaining this week")
    st.progress(
        min(_used / _credits_per_week, 1.0) if _credits_per_week > 0 else 1.0,
        text=f"{_used} used",
    )
    if st.button("Clock out"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    # Admin panel
    if _role == "admin":
        st.divider()
        st.subheader("Admin Panel")
        users = _get_users()
        rows = []
        for uname, ucfg in users.items():
            cpw = int(ucfg.get("credits_per_week", 5))
            try:
                u, r = _get_user_credits(uname, cpw)
            except Exception:
                u, r = 0, 0
            rows.append(
                {"User": uname, "Role": ucfg.get("role", "user"),
                 "Used": u, "Limit": cpw, "Remaining": r}
            )
        st.table(rows)
        reset_user = st.selectbox(
            "Reset credits for", [r["User"] for r in rows], index=None,
            placeholder="Select a user...",
        )
        if reset_user and st.button(f"Reset {reset_user}"):
            try:
                _reset_user_credits(reset_user)
            except Exception as e:
                st.error(f"Failed to reset credits for {reset_user}: {e}")
            else:
                st.rerun()

st.title("\U0001f9d1\u200d\U0001f373 Memo Chef")
st.caption("From raw ingredients to a Michelin-star memo")

# ============================================================================
# ANIMATED SHRIMP GIF — locally generated cooking shrimp chef
# ============================================================================
_CHEF_CAPTIONS = [
    "The shrimp is vibing while your memo cooks...",
    "Chef Shrimp is plating your metrics...",
    "Shrimp doing the happy dance for your data...",
    "Shrimply vibing while the numbers simmer...",
    "Party shrimp is prepping the garnish...",
]

_COOKING_SHRIMP_B64: str | None = None


def _load_cooking_shrimp_b64() -> str:
    """Load the cooking shrimp GIF as a base64 data URI (cached)."""
    global _COOKING_SHRIMP_B64
    if _COOKING_SHRIMP_B64 is None:
        import base64, pathlib
        gif_path = pathlib.Path(__file__).parent / "assets" / "cooking_shrimp.gif"
        _COOKING_SHRIMP_B64 = base64.b64encode(gif_path.read_bytes()).decode()
    return _COOKING_SHRIMP_B64


def _chef_gif_html() -> str:
    """Return HTML for the animated cooking shrimp chef GIF."""
    import random
    caption = random.choice(_CHEF_CAPTIONS)
    b64 = _load_cooking_shrimp_b64()
    return f"""
<div style="text-align:center; margin: 15px 0;">
  <img src="data:image/gif;base64,{b64}" alt="Chef Shrimp"
       style="max-height:220px; border-radius:12px;
              box-shadow: 0 4px 15px rgba(0,0,0,0.15);" />
  <p style="margin-top:8px; font-size:14px; color:#888;
            font-style:italic;">{caption}</p>
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
col1, col2, col3, col4 = st.columns(4)
memo_file = col1.file_uploader("The Memo (.pptx)", type=["pptx"])
proforma_file = col2.file_uploader("The Proforma (.xlsx / .xlsm)", type=["xlsx", "xlsm"])
schedule_file = col3.file_uploader("The Schedule (.mpp)", type=["mpp"])
market_data_file = col4.file_uploader("Market Data (.xlsx)", type=["xlsx", "xlsm"])

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
# FIRE BUTTON (credit-gated)
# ============================================================================
_fire_disabled = should_disable_fire_button(
    memo_file, proforma_file, _remaining, _credits_error
)
_fire_label = (
    f"\U0001f525  Fire!  ({_remaining} credits left)"
    if _remaining > 0
    else "\U0001f6ab  No credits remaining"
)

if st.button(_fire_label, type="primary", disabled=_fire_disabled):
    run_id = uuid.uuid4().hex
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
        _PROGRESS_SLOGANS = {
            "fire": [
                "\U0001f525 Firing up the pass...",
                "\U0001f525 Preheating the kitchen...",
                "\U0001f525 Lighting the burners...",
            ],
            "sear": [
                "\U0001f373 Searing the metrics...",
                "\U0001f373 Heating the yield up above a 6.50%...",
                "\U0001f373 Getting a nice crust on these numbers...",
                "\U0001f373 Caramelizing the cap rates...",
            ],
            "reduce": [
                "\U0001f373 Reducing the sauce...",
                "\U0001f373 Letting the flavors meld...",
                "\U0001f373 Deglazing the pan...",
            ],
            "plate": [
                "\U0001f37d\ufe0f Plating the dish...",
                "\U0001f37d\ufe0f Garnishing the entrée...",
                "\U0001f37d\ufe0f Wiping the rim of the plate...",
            ],
            "ticket": [
                "\U0001f4cb Printing the ticket...",
                "\U0001f4cb Hanging the ticket on the rail...",
                "\U0001f4cb Calling the check...",
            ],
            "market": [
                "\U0001f4ca Reading the market data...",
                "\U0001f4ca Checking the comps...",
                "\U0001f4ca Scanning the rent rolls...",
                "\U0001f4ca Pulling the market pulse...",
            ],
        }

        def _slogan(key: str) -> str:
            return random.choice(_PROGRESS_SLOGANS[key])

        progress_bar = st.progress(0, text=_slogan("fire"))

        with st.status("\U0001f525 Firing up the pass...", expanded=True) as status:
            # Show a random quirky shrimp chef GIF while processing
            anim_placeholder = st.empty()
            anim_placeholder.markdown(_chef_gif_html(), unsafe_allow_html=True)

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

            # Step b3: Extract market data (optional)
            if market_data_file:
                progress_bar.progress(13, text=_slogan("market"))
                st.write("\U0001f4ca Reading the market data...")
                market_data_path = os.path.join(tmpdir, market_data_file.name)
                with open(market_data_path, "wb") as f:
                    f.write(market_data_file.getvalue())
                try:
                    market_text = extract_market_data(market_data_path, cfg)
                    if market_text:
                        proforma_data += "\n\n" + market_text
                        st.write(f"\U0001f4ca Market data extracted ({len(market_text):,} chars)")
                    else:
                        st.warning("No dashboard tabs found in market data file — continuing without it.")
                except Exception as e:
                    st.warning(f"Could not read market data: {e}. Continuing without it.")

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
            progress_bar.progress(SEAR_START, text=_slogan("sear"))
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

            progress_bar.progress(70, text=_slogan("reduce"))

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
            progress_bar.progress(90, text=_slogan("plate"))
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
                st.write(f"\U0001f4d0 Layout healed: {layout_summary['titles_snapped']} titles, "
                         f"{layout_summary['page_numbers_snapped']} page numbers snapped")

            # Step h: Write change log — 96%
            progress_bar.progress(96, text=_slogan("ticket"))
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

        # Charge one credit only after a successful end-to-end run.
        try:
            charged = _consume_credit(_username, _credits_per_week, run_id=run_id)
            if not charged:
                st.warning("Run completed, but no credits remained to charge this run.")
        except Exception as credit_err:
            st.warning(f"Run completed, but credit usage could not be updated: {credit_err}")

      except Exception as e:
        st.error(f"In the weeds! {e}")

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
