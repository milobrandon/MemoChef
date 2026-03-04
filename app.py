#!/usr/bin/env python3
"""Memo Chef — Streamlit dashboard wrapping memo_automator.py."""

import hashlib
import logging
import os
import re
import secrets as _secrets
import tempfile
import time
from datetime import datetime, timedelta


import anthropic
import psycopg2
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
import user_management as um

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="Memo Chef", page_icon="\U0001f9d1\u200d\U0001f373", layout="centered")


# ============================================================================
# PASSWORD HELPERS
# ============================================================================
def _hash_password(password: str, salt: bytes | None = None) -> str:
    """Return ``"salt_hex:hash_hex"`` using PBKDF2-SHA256."""
    if salt is None:
        salt = _secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000)
    return f"{salt.hex()}:{dk.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    salt_hex, hash_hex = stored_hash.split(":", 1)
    salt = bytes.fromhex(salt_hex)
    candidate = _hash_password(password, salt)
    return candidate == stored_hash


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
    # Create user management tables (idempotent)
    um.create_tables(conn)
    return conn


def _current_week_start() -> str:
    """ISO Monday date for the current week."""
    today = datetime.now()
    monday = today - timedelta(days=today.weekday())
    return monday.strftime("%Y-%m-%d")


def _get_user_credits(username: str, credits_per_week: int) -> tuple[int, int]:
    """Return (used, remaining). Auto-resets on week rollover."""
    conn = _get_db_conn()
    week = _current_week_start()
    with conn.cursor() as cur:
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


def _consume_credit(username: str) -> None:
    conn = _get_db_conn()
    week = _current_week_start()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO credit_usage (username, week, used) VALUES (%s, %s, 1) "
            "ON CONFLICT (username) DO UPDATE SET "
            "  used = CASE WHEN credit_usage.week = %s THEN credit_usage.used + 1 ELSE 1 END, "
            "  week = %s",
            (username, week, week, week),
        )


def _reset_user_credits(username: str) -> None:
    conn = _get_db_conn()
    week = _current_week_start()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO credit_usage (username, week, used) VALUES (%s, %s, 0) "
            "ON CONFLICT (username) DO UPDATE SET week = %s, used = 0",
            (username, week, week),
        )


# ============================================================================
# AUTH GATE  (DB-first, secrets.toml fallback)
# ============================================================================
def _get_secrets_users() -> dict:
    """Load user definitions from st.secrets['users'] (legacy fallback)."""
    try:
        return dict(st.secrets["users"])
    except (KeyError, FileNotFoundError):
        return {}


def _get_smtp_config() -> dict | None:
    """Load SMTP settings from secrets, or return None if not configured."""
    try:
        return {
            "host": st.secrets["SMTP_HOST"],
            "port": st.secrets["SMTP_PORT"],
            "user": st.secrets["SMTP_USER"],
            "password": st.secrets["SMTP_PASSWORD"],
            "from_name": st.secrets.get("SMTP_FROM_NAME", "Memo Chef"),
        }
    except (KeyError, FileNotFoundError):
        return None


def _get_app_url() -> str:
    """Return the configured APP_URL or a sensible default."""
    try:
        return st.secrets["APP_URL"]
    except (KeyError, FileNotFoundError):
        return "http://localhost:8501"


def _handle_invite_signup() -> bool:
    """If ``?invite=TOKEN`` is in the URL, show the signup form.

    Returns ``True`` if the invite flow consumed the page (caller should stop).
    """
    params = st.query_params
    token = params.get("invite")
    if not token:
        return False

    conn = _get_db_conn()
    invite = um.get_invite(conn, token)

    st.title("\U0001f4e8 Memo Chef — Accept Invite")

    if invite is None:
        st.error("Invalid invite link.")
        return True
    if invite["is_used"]:
        st.info("This invite has already been used. Please log in.")
        return True
    from datetime import timezone as _tz
    if invite["expires_at"].replace(tzinfo=_tz.utc) < datetime.now(_tz.utc):
        st.error("This invite has expired. Ask your admin for a new one.")
        return True

    st.success(f"You've been invited as **{invite['role']}** by **{invite['invited_by']}**.")
    with st.form("signup_form"):
        new_username = st.text_input("Choose a username")
        new_display = st.text_input("Display name")
        new_password = st.text_input("Choose a password", type="password")
        new_password2 = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create Account")

    if submitted:
        if not new_username or not new_password:
            st.error("Username and password are required.")
        elif new_password != new_password2:
            st.error("Passwords do not match.")
        elif um.get_user(conn, new_username):
            st.error("Username already taken.")
        else:
            user = um.accept_invite(conn, token, new_username, new_password)
            if user:
                if new_display:
                    um.update_user(conn, new_username, display_name=new_display)
                st.success("Account created! You can now log in.")
                st.query_params.clear()
                st.rerun()
            else:
                st.error("Could not accept invite. It may have expired.")
    return True


def check_password() -> bool:
    """Per-user login: DB-first, secrets.toml fallback.  Sets session_state."""
    if st.session_state.get("authenticated"):
        return True

    # Handle invite signup flow first
    if _handle_invite_signup():
        st.stop()

    conn = _get_db_conn()

    st.title("\U0001f512 Memo Chef — Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        # Try database first
        db_user = um.verify_user(conn, username, password)
        if db_user:
            st.session_state["authenticated"] = True
            st.session_state["username"] = db_user["username"]
            st.session_state["role"] = db_user["role"]
            st.session_state["credits_per_week"] = db_user["credits_per_week"]
            um.log_audit(conn, username, "login", username)
            st.rerun()
            return True

        # Fallback to secrets.toml
        secrets_users = _get_secrets_users()
        user_cfg = secrets_users.get(username)
        if user_cfg and _verify_password(password, user_cfg["password_hash"]):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["role"] = user_cfg.get("role", "user")
            st.session_state["credits_per_week"] = int(
                user_cfg.get("credits_per_week", 5)
            )
            st.rerun()
            return True

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
_used, _remaining = _get_user_credits(_username, _credits_per_week)

with st.sidebar:
    st.markdown(f"**{_username}** &nbsp; `{_role}`")
    st.caption(f"Credits: {_remaining} / {_credits_per_week} remaining this week")
    st.progress(
        min(_used / _credits_per_week, 1.0) if _credits_per_week > 0 else 1.0,
        text=f"{_used} used",
    )
    if st.button("Clock out"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    # ================================================================
    # ADMIN PANEL (tabbed: Users / Invites / Audit)
    # ================================================================
    if _role == "admin":
        st.divider()
        st.subheader("Admin Panel")
        _conn = _get_db_conn()
        admin_tab = st.radio(
            "Section", ["Users", "Invites", "Audit"], horizontal=True, label_visibility="collapsed",
        )

        # ------ USERS TAB ------
        if admin_tab == "Users":
            # Show DB users + secrets.toml users merged
            db_users = um.list_users(_conn)
            secrets_users = _get_secrets_users()
            db_usernames = {u["username"] for u in db_users}

            rows = []
            for u in db_users:
                used, rem = _get_user_credits(u["username"], u["credits_per_week"])
                rows.append({
                    "User": u["username"],
                    "Role": u["role"],
                    "Credits": f"{used}/{u['credits_per_week']}",
                    "Active": "Yes" if u["is_active"] else "No",
                    "Source": "DB",
                })
            for uname, ucfg in secrets_users.items():
                if uname not in db_usernames:
                    cpw = int(ucfg.get("credits_per_week", 5))
                    used, rem = _get_user_credits(uname, cpw)
                    rows.append({
                        "User": uname,
                        "Role": ucfg.get("role", "user"),
                        "Credits": f"{used}/{cpw}",
                        "Active": "Yes",
                        "Source": "Config",
                    })
            if rows:
                st.table(rows)

            # Reset credits
            all_usernames = [r["User"] for r in rows]
            reset_user = st.selectbox(
                "Reset credits for", all_usernames, index=None,
                placeholder="Select a user...", key="admin_reset_user",
            )
            if reset_user and st.button(f"Reset {reset_user}"):
                _reset_user_credits(reset_user)
                um.log_audit(_conn, _username, "credits_reset", reset_user)
                st.rerun()

            # Add new user
            st.markdown("---")
            st.markdown("**Add User**")
            with st.form("add_user_form"):
                new_uname = st.text_input("Username", key="new_user_name")
                new_display = st.text_input("Display name", key="new_user_display")
                new_email = st.text_input("Email (optional)", key="new_user_email")
                new_pw = st.text_input("Password", type="password", key="new_user_pw")
                new_role = st.selectbox("Role", ["user", "admin"], key="new_user_role")
                new_cpw = st.number_input("Credits/week", min_value=1, max_value=100, value=5, key="new_user_cpw")
                add_submitted = st.form_submit_button("Create User")
            if add_submitted:
                if not new_uname or not new_pw:
                    st.error("Username and password required.")
                elif um.get_user(_conn, new_uname):
                    st.error(f"User '{new_uname}' already exists.")
                else:
                    um.create_user(
                        _conn, new_uname, new_pw,
                        email=new_email or None,
                        display_name=new_display,
                        role=new_role,
                        credits_per_week=int(new_cpw),
                    )
                    um.log_audit(_conn, _username, "created", new_uname, {"role": new_role})
                    st.success(f"User '{new_uname}' created.")
                    st.rerun()

            # Edit existing DB user
            db_usernames_list = [u["username"] for u in db_users]
            if db_usernames_list:
                st.markdown("---")
                st.markdown("**Edit User**")
                edit_target = st.selectbox(
                    "Select user", db_usernames_list, index=None,
                    placeholder="Choose...", key="edit_target_user",
                )
                if edit_target:
                    eu = um.get_user(_conn, edit_target)
                    with st.form("edit_user_form"):
                        ed_display = st.text_input("Display name", value=eu["display_name"], key="ed_display")
                        ed_email = st.text_input("Email", value=eu.get("email") or "", key="ed_email")
                        ed_role = st.selectbox(
                            "Role", ["user", "admin"],
                            index=0 if eu["role"] == "user" else 1, key="ed_role",
                        )
                        ed_cpw = st.number_input(
                            "Credits/week", min_value=1, max_value=100,
                            value=eu["credits_per_week"], key="ed_cpw",
                        )
                        ed_active = st.checkbox("Active", value=eu["is_active"], key="ed_active")
                        save_submitted = st.form_submit_button("Save Changes")
                    if save_submitted:
                        um.update_user(
                            _conn, edit_target,
                            display_name=ed_display,
                            email=ed_email or None,
                            role=ed_role,
                            credits_per_week=int(ed_cpw),
                            is_active=ed_active,
                        )
                        um.log_audit(_conn, _username, "updated", edit_target, {
                            "role": ed_role, "active": ed_active,
                        })
                        st.success(f"User '{edit_target}' updated.")
                        st.rerun()

        # ------ INVITES TAB ------
        elif admin_tab == "Invites":
            st.markdown("**Send Invite**")
            with st.form("invite_form"):
                inv_email = st.text_input("Email address", key="inv_email")
                inv_role = st.selectbox("Role", ["user", "admin"], key="inv_role")
                inv_cpw = st.number_input("Credits/week", min_value=1, max_value=100, value=5, key="inv_cpw")
                inv_submitted = st.form_submit_button("Send Invite")
            if inv_submitted:
                if not inv_email:
                    st.error("Email is required.")
                else:
                    token = um.create_invite(
                        _conn, inv_email, _username,
                        role=inv_role, credits_per_week=int(inv_cpw),
                    )
                    app_url = _get_app_url()
                    invite_link = f"{app_url.rstrip('/')}/?invite={token}"

                    # Try sending email
                    smtp_cfg = _get_smtp_config()
                    if smtp_cfg:
                        sent = um.send_invite_email(inv_email, token, _username, app_url, smtp_cfg)
                        if sent:
                            st.success(f"Invite sent to {inv_email}")
                        else:
                            st.warning("Email delivery failed. Share the link manually:")
                            st.code(invite_link)
                    else:
                        st.info("SMTP not configured. Share this link manually:")
                        st.code(invite_link)

            # Pending invites table
            invites = um.list_invites(_conn)
            if invites:
                st.markdown("---")
                st.markdown("**Recent Invites**")
                from datetime import timezone as _tz
                inv_rows = []
                for inv in invites[:20]:
                    if inv["is_used"]:
                        status = "Accepted"
                    elif inv["expires_at"].replace(tzinfo=_tz.utc) < datetime.now(_tz.utc):
                        status = "Expired"
                    else:
                        status = "Pending"
                    inv_rows.append({
                        "Email": inv["email"],
                        "Role": inv["role"],
                        "Status": status,
                        "Sent": inv["created_at"].strftime("%m/%d"),
                        "By": inv["invited_by"],
                    })
                st.table(inv_rows)

        # ------ AUDIT TAB ------
        elif admin_tab == "Audit":
            st.markdown("**Recent Activity**")
            audit = um.get_audit_log(_conn, limit=30)
            if audit:
                for entry in audit:
                    ts = entry["created_at"].strftime("%m/%d %H:%M")
                    target = f' "{entry["target_user"]}"' if entry["target_user"] else ""
                    st.caption(f"{ts} — **{entry['actor']}** {entry['action']}{target}")
            else:
                st.caption("No activity yet.")

st.title("\U0001f9d1\u200d\U0001f373 Memo Chef")
st.caption("From raw ingredients to a Michelin-star memo")

# ============================================================================
# ANIMATED SHRIMP GIFS — rotates between 5 quirky chef/shrimp GIFs
# ============================================================================
_CHEF_GIFS = [
    # Dancing shrimp (nemomi) — cute bouncy shrimp dance
    ("https://media.giphy.com/media/H4rDezQiLmkdbAzHcI/giphy.gif",
     "The shrimp is vibing while your memo cooks..."),
    # Chef shrimp by Walk-On's — shrimp with chef energy
    ("https://media.giphy.com/media/37RfRKBoosceYgis5e/giphy.gif",
     "Chef Shrimp is plating your metrics..."),
    # Happy dancing shrimp (chopt) — joyful bouncing
    ("https://media.giphy.com/media/3o7buikVSfLTuXjbC8/giphy.gif",
     "Shrimp doing the happy dance for your data..."),
    # Orange shrimp vibing — chill vibes
    ("https://media.giphy.com/media/kFxSwhFVe3414uK8F5/giphy.gif",
     "Shrimply vibing while the numbers simmer..."),
    # Party shrimp — celebration mode
    ("https://media.giphy.com/media/yGJdHAm1Vu0MlLq6CL/giphy.gif",
     "Party shrimp is prepping the garnish..."),
]


def _chef_gif_html() -> str:
    """Return HTML for a randomly chosen shrimp chef GIF."""
    import random
    url, caption = random.choice(_CHEF_GIFS)
    return f"""
<div style="text-align:center; margin: 15px 0;">
  <img src="{url}" alt="Chef Shrimp"
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
# FIRE BUTTON (credit-gated)
# ============================================================================
_fire_disabled = not (memo_file and proforma_file) or _remaining <= 0
_fire_label = (
    f"\U0001f525  Fire!  ({_remaining} credits left)"
    if _remaining > 0
    else "\U0001f6ab  No credits remaining"
)

if st.button(_fire_label, type="primary", disabled=_fire_disabled):
    # Consume a credit up front
    _consume_credit(_username)

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
