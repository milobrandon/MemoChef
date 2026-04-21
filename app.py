#!/usr/bin/env python3
"""Modern Streamlit dashboard for Memo Automator."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import time
import uuid

import streamlit as st

from app_helpers import (
    count_changes_from_log,
    fire_button_disabled_reason,
    should_disable_fire_button,
    verify_password,
)
from app_services import (
    accept_invitation,
    add_user,
    consume_credit,
    create_auth_session,
    create_invitation,
    delete_job,
    delete_user,
    enqueue_job,
    ensure_users_seeded,
    get_auth_session,
    get_db_conn,
    get_invitation,
    get_invitations,
    get_job,
    get_job_queue,
    get_job_staging_dir,
    get_platform_health,
    get_profiles,
    get_recent_runs,
    get_run_artifact_paths,
    get_run_storage_dir,
    cleanup_old_artifacts,
    get_run_details,
    get_user_credits,
    get_users,
    record_run,
    revoke_auth_session,
    reset_user_credits,
    save_profile,
    start_background_worker,
    update_job_status,
    update_run_approval,
    update_user,
)
from managed_agents.run_session import (
    create_session as ma_create_session,
    download_file_to as ma_download_file_to,
    get_output_files as ma_get_output_files,
    send_message as ma_send_message,
    stream_events as ma_stream_events,
    upload_example_memos as ma_upload_example_memos,
    upload_file as ma_upload_file,
    upload_fireflies_config as ma_upload_fireflies_config,
    build_user_message as ma_build_user_message,
)
from memo_chef.theme import APP_SUBTITLE, APP_TITLE, app_css, info_card, render_hero

# ---------------------------------------------------------------------------
# Animated shrimp chef GIF shown during pipeline runs
# ---------------------------------------------------------------------------
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
        import base64
        gif_path = Path(__file__).parent / "assets" / "cooking_shrimp.gif"
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

st.set_page_config(page_title=APP_TITLE, page_icon="✨", layout="wide")
st.markdown(app_css(), unsafe_allow_html=True)

try:
    ensure_users_seeded()
except Exception as e:
    import logging as _logging
    _logging.getLogger(__name__).warning("Failed to seed users from database: %s", e)

# Start background worker for auto-processing queued jobs
start_background_worker()

# --- Invitation sign-up page (no auth required) ---
invite_token = st.query_params.get("invite")
if invite_token:
    render_hero()
    invite = get_invitation(invite_token)
    if invite is None:
        st.error("Invalid invitation link.")
        st.stop()
    if invite["status"] == "accepted":
        st.info("This invitation has already been used. Please sign in.")
        st.stop()
    if invite["status"] == "expired" or invite["expires_at"] < datetime.now(invite["expires_at"].tzinfo):
        st.warning("This invitation has expired. Please contact your administrator for a new one.")
        st.stop()

    st.markdown(
        info_card("Create your account", f"You've been invited as a **{invite['role']}** with **{invite['credits_per_week']}** weekly runs."),
        unsafe_allow_html=True,
    )
    with st.form("invite_signup_form"):
        signup_username = st.text_input("Choose a username")
        signup_password = st.text_input("Password", type="password")
        signup_confirm = st.text_input("Confirm password", type="password")
        signup_submitted = st.form_submit_button("Create account", type="primary")
    if signup_submitted:
        if not signup_username.strip():
            st.error("Username is required.")
        elif len(signup_password) < 6:
            st.error("Password must be at least 6 characters.")
        elif signup_password != signup_confirm:
            st.error("Passwords do not match.")
        else:
            if accept_invitation(invite_token, signup_username.strip(), signup_password):
                st.success("Account created! You can now sign in.")
                st.query_params.clear()
                time.sleep(2)
                st.rerun()
            else:
                st.error("Could not create account. The username may already be taken or the invitation has expired.")
    st.stop()


_AUTH_QUERY_PARAM = "session"


def _clear_run_state() -> None:
    for key in [
        "memo_bytes",
        "log_bytes",
        "manifest_bytes",
        "filename",
        "n_changes",
        "n_rejected",
        "n_missed",
        "log_lines",
        "warnings",
        "manifest",
        "changes",
    ]:
        st.session_state.pop(key, None)


def _missing_required_run_inputs(
    memo_file, proforma_file, meeting_lookback_days: int = 0
) -> str | None:
    if not memo_file:
        return "Upload a memo deck before starting a run."
    if not proforma_file and meeting_lookback_days <= 0:
        return "Upload a proforma, or set meeting lookback > 0 for a narrative-only run."
    return None


def _set_authenticated_user(
    *,
    username: str,
    role: str,
    credits_per_week: int,
    session_token: str | None = None,
) -> None:
    st.session_state["authenticated"] = True
    st.session_state["username"] = username
    st.session_state["role"] = role
    st.session_state["credits_per_week"] = int(credits_per_week)
    if session_token:
        st.session_state["auth_session_token"] = session_token
        st.query_params[_AUTH_QUERY_PARAM] = session_token


def _clear_authenticated_user() -> None:
    for key in (
        "authenticated",
        "username",
        "role",
        "credits_per_week",
        "auth_session_token",
    ):
        st.session_state.pop(key, None)
    if _AUTH_QUERY_PARAM in st.query_params:
        del st.query_params[_AUTH_QUERY_PARAM]


def _restore_authenticated_user() -> bool:
    token = st.session_state.get("auth_session_token") or st.query_params.get(_AUTH_QUERY_PARAM)
    if not token:
        return False
    session_user = get_auth_session(token)
    if session_user is None:
        _clear_authenticated_user()
        return False
    _set_authenticated_user(
        username=session_user["username"],
        role=session_user.get("role", "user"),
        credits_per_week=int(session_user.get("credits_per_week", 5)),
        session_token=token,
    )
    return True


def _queue_item_from_inputs(
    *,
    memo_file,
    proforma_file,
    supplemental_file=None,
    property_name: str,
    profile_name: str | None,
    project_name: str = "",
    meeting_lookback_days: int = 0,
    fireflies_api_key: str = "",
    instructions: str = "",
) -> dict:
    job_id = uuid.uuid4().hex
    staging = get_job_staging_dir(job_id)

    memo_path = str(staging / memo_file.name)
    with open(memo_path, "wb") as f:
        f.write(memo_file.getvalue())

    proforma_path: str | None = None
    if proforma_file is not None:
        proforma_path = str(staging / proforma_file.name)
        with open(proforma_path, "wb") as f:
            f.write(proforma_file.getvalue())

    supp_path = None
    if supplemental_file:
        supp_path = str(staging / supplemental_file.name)
        with open(supp_path, "wb") as f:
            f.write(supplemental_file.getvalue())

    return {
        "job_id": job_id,
        "memo_name": memo_file.name,
        "memo_path": memo_path,
        "proforma_name": proforma_file.name if proforma_file is not None else None,
        "proforma_path": proforma_path,
        "supplemental_name": supplemental_file.name if supplemental_file else None,
        "supplemental_path": supp_path,
        "property_name": property_name or None,
        "profile_name": profile_name or "",
        "project_name": project_name or "",
        "meeting_lookback_days": meeting_lookback_days,
        "fireflies_api_key": fireflies_api_key or "",
        "instructions": instructions or "",
    }


def _execute_job(
    *,
    job: dict,
    username: str,
    credits_per_week: int,
    queue_position: int | None = None,
    queue_total: int | None = None,
) -> bool:
    _clear_run_state()
    cleanup_old_artifacts(max_age_seconds=3600)

    run_id = uuid.uuid4().hex
    if job.get("job_id"):
        update_job_status(job["job_id"], "running", run_id=run_id)
    started = time.time()

    prefix = ""
    if queue_position is not None and queue_total is not None:
        prefix = f"Queue item {queue_position}/{queue_total} · "

    progress_bar = st.progress(0, text=f"{prefix}Uploading files...")
    shrimp_placeholder = st.empty()
    shrimp_placeholder.markdown(_chef_gif_html(), unsafe_allow_html=True)
    status_box = st.empty()
    stage_log = st.empty()
    stage_lines: list[str] = []

    run_dir = get_run_storage_dir(run_id)
    memo_path = Path(job["memo_path"])
    proforma_path = Path(job["proforma_path"]) if job.get("proforma_path") else None

    try:
        # ── Upload files ──────────────────────────────────────────────────────
        resources = []

        progress_bar.progress(5, text=f"{prefix}Uploading memo...")
        memo_file_id = ma_upload_file(memo_path)
        resources.append({
            "type": "file",
            "file_id": memo_file_id,
            "mount_path": f"/mnt/session/uploads/{memo_path.name}",
        })

        if proforma_path is not None:
            progress_bar.progress(10, text=f"{prefix}Uploading proforma...")
            proforma_file_id = ma_upload_file(proforma_path)
            resources.append({
                "type": "file",
                "file_id": proforma_file_id,
                "mount_path": f"/mnt/session/uploads/{proforma_path.name}",
            })

        supplemental_names: list[str] = []
        if job.get("supplemental_path") and os.path.isfile(job["supplemental_path"]):
            sup_path = Path(job["supplemental_path"])
            progress_bar.progress(13, text=f"{prefix}Uploading supplemental...")
            sup_id = ma_upload_file(sup_path)
            resources.append({
                "type": "file",
                "file_id": sup_id,
                "mount_path": f"/mnt/session/uploads/{sup_path.name}",
            })
            supplemental_names.append(sup_path.name)

        # ── Fireflies config ──────────────────────────────────────────────────
        meeting_lookback_days = job.get("meeting_lookback_days", 0)
        if meeting_lookback_days > 0:
            project_name = job.get("project_name", "")
            if project_name:
                search_terms = [t for t in project_name.split() if len(t) > 2]
            else:
                search_terms = [t for t in memo_path.stem.replace("_", " ").split() if len(t) > 3]
            ff_resource = ma_upload_fireflies_config(
                lookback_days=meeting_lookback_days,
                search_terms=search_terms,
                api_key_override=job.get("fireflies_api_key") or None,
            )
            if ff_resource:
                resources.append(ff_resource)

        # ── Example memos ─────────────────────────────────────────────────────
        progress_bar.progress(15, text=f"{prefix}Loading style references...")
        resources.extend(ma_upload_example_memos())

        # ── Create session ────────────────────────────────────────────────────
        progress_bar.progress(20, text=f"{prefix}Creating agent session...")
        session_id = ma_create_session(
            uploaded_resources=resources,
            title=f"Memo Chef: {memo_path.name}",
        )

        message = ma_build_user_message(
            proforma_filename=proforma_path.name if proforma_path else None,
            memo_filename=memo_path.name,
            supplemental_filenames=supplemental_names or None,
            instructions=job.get("instructions", ""),
            meeting_lookback_days=meeting_lookback_days if meeting_lookback_days > 0 else None,
        )
        ma_send_message(session_id, message)

        # ── Stream events ─────────────────────────────────────────────────────
        progress_bar.progress(25, text=f"{prefix}Agent is working...")
        status_box.caption(f"{prefix}Agent is working...")

        for event in ma_stream_events(session_id):
            t = event.get("type", "")
            if t == "agent.tool_use":
                tool_name = event.get("name", "")
                status_box.caption(f"{prefix}Using: {tool_name}")
                stage_lines.append(f"[Tool] {tool_name}")
                stage_log.code("\n".join(stage_lines[-15:]), language=None)
            elif t == "agent.message" and event.get("text"):
                stage_lines.append(f"[Agent] {event['text'][:300]}")
                stage_log.code("\n".join(stage_lines[-15:]), language=None)
            elif t == "span.model_request_start":
                status_box.caption(f"{prefix}Thinking...")
            elif t == "session.error":
                raise RuntimeError(f"Agent session error: {event.get('error')}")

        # ── Retrieve output files ─────────────────────────────────────────────
        progress_bar.progress(95, text=f"{prefix}Retrieving output files...")
        output_files = ma_get_output_files(session_id)

        memo_bytes: bytes | None = None
        log_bytes: bytes | None = None

        for f in output_files:
            if f["filename"] == "output.pptx" and f.get("downloadable", True):
                dest = run_dir / f"memo{memo_path.suffix}"
                ma_download_file_to(f["id"], dest)
                memo_bytes = dest.read_bytes()
            elif f["filename"] == "changelog.md" and f.get("downloadable", True):
                dest = run_dir / "change_log.md"
                ma_download_file_to(f["id"], dest)
                log_bytes = dest.read_bytes()

        if not memo_bytes:
            raise RuntimeError("Agent did not produce output.pptx — check agent logs.")

        duration = round(time.time() - started, 2)
        progress_bar.progress(100, text=f"{prefix}Run complete")
        shrimp_placeholder.empty()
        status_box.success(f"{prefix}Draft generated successfully.")

        # Store in session state for download buttons
        st.session_state["memo_bytes"] = memo_bytes
        st.session_state["memo_name"] = job["memo_name"]
        st.session_state["log_bytes"] = log_bytes or b""
        st.session_state["warnings"] = []

        # Build minimal manifest for history display
        log_text = log_bytes.decode("utf-8", errors="replace") if log_bytes else ""
        change_count = count_changes_from_log(log_text)
        manifest = {
            "run_id": run_id,
            "session_id": session_id,
            "status": "completed",
            "memo_name": job["memo_name"],
            "proforma_name": job.get("proforma_name") or "(narrative-only)",
            "property_name": job.get("property_name"),
            "duration_seconds": duration,
            "counts": {"change_count": change_count},
        }
        manifest_bytes = json.dumps(manifest, indent=2).encode()
        st.session_state["manifest"] = manifest
        st.session_state["manifest_bytes"] = manifest_bytes
        (run_dir / "run_manifest.json").write_bytes(manifest_bytes)

        record_run(
            run_id=run_id,
            username=username,
            status="completed",
            memo_name=job["memo_name"],
            proforma_name=job.get("proforma_name") or "(narrative-only)",
            property_name=job.get("property_name"),
            dry_run=False,
            skip_validation=False,
            change_count=change_count,
            rejected_count=0,
            missed_count=0,
            duration_seconds=duration,
            warnings=[],
            run_manifest_json=manifest_bytes.decode("utf-8"),
            change_log_html=log_text or None,
        )

        if job.get("job_id"):
            update_job_status(job["job_id"], "completed", run_id=run_id)

        try:
            charged = consume_credit(username, credits_per_week, run_id=run_id)
            if not charged:
                st.warning("The run completed, but weekly credits were already exhausted.")
        except Exception as err:
            st.warning(f"Run completed, but credits could not be updated: {err}")

        return True

    except Exception as err:
        duration = round(time.time() - started, 2)
        progress_bar.progress(100, text=f"{prefix}Run failed")
        shrimp_placeholder.empty()
        status_box.error(f"{prefix}Run failed")
        stage_log.code("\n".join(stage_lines[-10:]), language=None)
        try:
            record_run(
                run_id=run_id,
                username=username,
                status="failed",
                memo_name=job["memo_name"],
                proforma_name=job.get("proforma_name") or "(narrative-only)",
                property_name=job.get("property_name"),
                dry_run=False,
                skip_validation=False,
                change_count=0,
                rejected_count=0,
                missed_count=0,
                duration_seconds=duration,
                warnings=[{"stage": "agent", "message": str(err)}],
            )
        except Exception:
            pass
        if job.get("job_id"):
            update_job_status(job["job_id"], "failed", run_id=run_id, error_message=str(err))
        st.error(f"{prefix}Run failed: {err}")
        return False


def check_password() -> bool:
    if st.session_state.get("authenticated"):
        if not st.session_state.get("auth_session_token") and st.session_state.get("username"):
            session_token = create_auth_session(st.session_state["username"])
            _set_authenticated_user(
                username=st.session_state["username"],
                role=st.session_state.get("role", "user"),
                credits_per_week=int(st.session_state.get("credits_per_week", 5)),
                session_token=session_token,
            )
        return True
    if _restore_authenticated_user():
        return True

    users = get_users()
    if not users:
        st.error("No users configured. Add users to Streamlit secrets or the database.")
        st.stop()

    render_hero()
    st.markdown(
        info_card(
            "Secure workspace",
            "Sign in to access governed memo runs, queue execution, approvals, and operational controls.",
        ),
        unsafe_allow_html=True,
    )

    with st.form("login_form"):
        cols = st.columns([1, 1, 1])
        username = cols[0].text_input("Username")
        password = cols[1].text_input("Password", type="password")
        cols[2].markdown("<div style='height: 1.8rem'></div>", unsafe_allow_html=True)
        submitted = cols[2].form_submit_button("Sign in", type="primary", width="stretch")

    if submitted:
        user_cfg = users.get(username)
        if user_cfg and verify_password(password, user_cfg["password_hash"]):
            session_token = create_auth_session(username)
            _set_authenticated_user(
                username=username,
                role=user_cfg.get("role", "user"),
                credits_per_week=int(user_cfg.get("credits_per_week", 5)),
                session_token=session_token,
            )
            st.rerun()
        st.error("Invalid username or password.")
    st.stop()


if not check_password():
    st.stop()

username = st.session_state["username"]
role = st.session_state["role"]
credits_per_week = st.session_state["credits_per_week"]
credits_error = None

try:
    used, remaining = get_user_credits(username, credits_per_week)
except Exception as err:
    used, remaining = 0, 0
    credits_error = str(err)

with st.sidebar:
    st.markdown(f"### {username}")
    st.caption(f"Role: `{role}`")
    if credits_error:
        st.warning("Credits service unavailable.")
        if st.button("Reconnect services"):
            get_db_conn.clear()
            st.rerun()
    else:
        st.caption(f"{remaining} of {credits_per_week} weekly runs remaining")
        st.progress(
            min(used / credits_per_week, 1.0) if credits_per_week > 0 else 1.0,
            text=f"{used} used this week",
        )
    try:
        queue_count = len(
            [job for job in get_job_queue(None if role == "admin" else username) if job["status"] == "queued"]
        )
    except Exception:
        queue_count = 0
    st.caption(f"Batch queue: {queue_count} item(s)")
    st.divider()
    st.caption("Platform")
    st.write("Reviewable automation, typed configuration, queueing, and traceable outputs.")
    if st.button("Sign out", width="stretch"):
        revoke_auth_session(
            st.session_state.get("auth_session_token") or st.query_params.get(_AUTH_QUERY_PARAM)
        )
        _clear_authenticated_user()
        st.rerun()

render_hero()

card_cols = st.columns(4)
card_cols[0].markdown(
    info_card("Guardrails", "Two-pass mapping and validation with checkpointed artifacts."),
    unsafe_allow_html=True,
)
card_cols[1].markdown(
    info_card("Operations", "Queue multiple runs and review outcomes from a single console."),
    unsafe_allow_html=True,
)
card_cols[2].markdown(
    info_card("Governance", "Track approval status, reviewer, and warnings per run."),
    unsafe_allow_html=True,
)
card_cols[3].markdown(
    info_card("Brand system", "Refreshed dark UI, cleaner actions, and a premium visual hierarchy."),
    unsafe_allow_html=True,
)

tab_labels = ["New Run", "Run History", "Operations", "How To"] + (["Admin"] if role == "admin" else [])
tabs = st.tabs(tab_labels)


def render_new_run_tab() -> None:
    st.subheader("New run")
    st.caption(APP_SUBTITLE)

    profiles = get_profiles(None if role == "admin" else username)
    profile_lookup = {row["Profile"]: row for row in profiles}
    selected_profile = st.selectbox(
        "Saved profile",
        options=[""] + sorted(profile_lookup.keys()),
        format_func=lambda value: "None" if value == "" else value,
        help="Load saved preferences for property naming and QA behavior.",
    )
    profile = profile_lookup.get(selected_profile, {})

    # ── Memo City News ────────────────────────────────────────────────────────────
    _news_path = Path(__file__).parent / "memo_city_news.json"
    if _news_path.exists():
        _news = json.loads(_news_path.read_text(encoding="utf-8"))
        with st.expander("📰 Memo City News — What's new in the app", expanded=False):
            for item in _news:
                st.markdown(f"**v{item['version']} · {item['date']} — {item['title']}**")
                for bullet in item["bullets"]:
                    st.markdown(f"- {bullet}")
                st.markdown("---")

    upload_cols = st.columns(3)
    memo_file = upload_cols[0].file_uploader("Memo deck", type=["pptx"], key="memo_upload")
    proforma_file = upload_cols[1].file_uploader("Proforma", type=["xlsx", "xlsm"], key="proforma_upload")
    supplemental_file = upload_cols[2].file_uploader(
        "Supplemental data",
        type=["pdf", "xlsx", "xlsm", "csv"],
        key="supplemental_upload",
        help="Optional additional file for the agent to reference",
    )

    name_cols = st.columns(2)
    property_name = name_cols[0].text_input(
        "Property name (as it appears in memo)",
        value=profile.get("Property", ""),
        placeholder="e.g. VERVE Lexington",
        help="The property name currently used in the memo deck.",
    )
    project_name = name_cols[1].text_input(
        "Project / deal name",
        placeholder="e.g. Limestone, Knoxville, VERVE Pittsburgh",
        help="Used to search Fireflies for relevant meeting transcripts.",
    )

    meeting_lookback_days = st.number_input(
        "Meeting transcript lookback (days)",
        min_value=0,
        max_value=365,
        value=0,
        step=5,
        help="Include Fireflies meeting transcripts from the last N days. Set to 0 to disable.",
    )

    with st.expander("Fireflies API key (optional)"):
        st.caption(
            "Override the platform-level Fireflies key with your own. "
            "Leave blank to use the platform default."
        )
        fireflies_api_key = st.text_input(
            "Fireflies API key",
            type="password",
            key="fireflies_api_key_input",
            placeholder="Leave blank to use platform default",
        )

    instructions = st.text_area(
        "Additional instructions for the agent",
        placeholder="e.g. Focus on cash flow section. Do not update the cover slide.",
        height=80,
        key="instructions_input",
    )

    save_profile_name = st.text_input(
        "Save current preferences as profile",
        placeholder="e.g. Standard IC review",
    )
    profile_notes = st.text_area(
        "Profile notes",
        placeholder="Optional guidance for this profile",
        height=80,
    )
    profile_cols = st.columns([1, 3])
    if profile_cols[0].button("Save profile", width="stretch"):
        if not save_profile_name.strip():
            st.error("Enter a profile name before saving.")
        else:
            save_profile(
                save_profile_name.strip(),
                username,
                property_name or None,
                False,
                False,
                profile_notes or None,
            )
            st.success(f"Saved profile `{save_profile_name.strip()}`.")
            st.rerun()

    action_disabled = should_disable_fire_button(memo_file, proforma_file, remaining, credits_error)
    disabled_reason = fire_button_disabled_reason(
        memo_file,
        proforma_file,
        remaining,
        credits_error,
        meeting_lookback_days=int(meeting_lookback_days),
    )
    action_cols = st.columns(2)

    if action_cols[0].button(
        f"Generate draft ({remaining} credits left)" if remaining > 0 else "No credits remaining",
        type="primary",
        disabled=action_disabled,
        width="stretch",
    ):
        missing_inputs = _missing_required_run_inputs(
            memo_file, proforma_file, int(meeting_lookback_days)
        )
        if missing_inputs:
            st.warning(missing_inputs)
        else:
            job = _queue_item_from_inputs(
                memo_file=memo_file,
                proforma_file=proforma_file,
                supplemental_file=supplemental_file,
                property_name=property_name,
                profile_name=selected_profile or save_profile_name.strip() or None,
                project_name=project_name,
                meeting_lookback_days=int(meeting_lookback_days),
                fireflies_api_key=fireflies_api_key,
                instructions=instructions,
            )
            _execute_job(job=job, username=username, credits_per_week=credits_per_week)

    if action_cols[1].button(
        "Add to queue",
        disabled=action_disabled,
        width="stretch",
    ):
        missing_inputs = _missing_required_run_inputs(
            memo_file, proforma_file, int(meeting_lookback_days)
        )
        if proforma_file is None:
            st.warning(
                "Narrative-only runs (no proforma) aren't supported in the queue yet — "
                "use Generate draft instead."
            )
        elif missing_inputs:
            st.warning(missing_inputs)
        else:
            job = _queue_item_from_inputs(
                memo_file=memo_file,
                proforma_file=proforma_file,
                supplemental_file=supplemental_file,
                property_name=property_name,
                profile_name=selected_profile or save_profile_name.strip() or None,
                project_name=project_name,
                meeting_lookback_days=int(meeting_lookback_days),
                fireflies_api_key=fireflies_api_key,
                instructions=instructions,
            )
            enqueue_job(username, job)
            st.success(f"Queued `{job['memo_name']}`.")

    if disabled_reason:
        st.caption(disabled_reason)

    if "memo_bytes" in st.session_state:
        st.divider()
        st.success("Artifacts are ready for review and download.")

        download_cols = st.columns(3)
        download_cols[0].download_button(
            "Download updated memo",
            st.session_state["memo_bytes"],
            file_name=st.session_state.get("memo_name", "output.pptx"),
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            width="stretch",
        )
        download_cols[1].download_button(
            "Download change log",
            st.session_state.get("log_bytes", b""),
            file_name="CHANGE_LOG.md",
            mime="text/markdown",
            width="stretch",
        )
        download_cols[2].download_button(
            "Download run manifest",
            st.session_state.get("manifest_bytes", b""),
            file_name="run_manifest.json",
            mime="application/json",
            width="stretch",
        )

        warnings = st.session_state.get("warnings", [])
        if warnings:
            with st.expander("Warnings"):
                for warning in warnings:
                    st.warning(f"{warning['stage']}: {warning['message']}")

        _log_bytes = st.session_state.get("log_bytes", b"")
        if _log_bytes:
            with st.expander("Changelog"):
                st.code(_log_bytes.decode("utf-8", errors="replace"), language=None)


def render_history_tab() -> None:
    st.subheader("Run history")
    st.caption("Recent runs, outcomes, approvals, and warning counts for auditing and reruns.")
    try:
        rows = get_recent_runs(None if role == "admin" else username, limit=30)
    except Exception as err:
        st.warning(f"Run history is unavailable: {err}")
        return
    if not rows:
        st.info("No completed or recorded runs yet.")
        return
    st.dataframe(rows, width="stretch", hide_index=True)

    st.divider()
    st.markdown("#### Approval workflow")
    run_options = [row["Run ID"] for row in rows]
    selected_run = st.selectbox("Select run", run_options, index=0)
    details = get_run_details(selected_run)
    if details:
        detail_cols = st.columns(6)
        detail_cols[0].metric("Status", details["status"])
        detail_cols[1].metric("Approval", details["approval_status"])
        detail_cols[2].metric("Changes", details["change_count"])
        detail_cols[3].metric("Warnings", len(details["warnings"]))
        conf = details.get("confidence_score")
        detail_cols[4].metric("Confidence", f"{conf:.0f}/100" if conf is not None else "—")
        detail_cols[5].metric("Slides inserted", details.get("slides_inserted", 0))
        if details["warnings"]:
            with st.expander("Run warnings"):
                for warning in details["warnings"]:
                    st.warning(f"{warning['stage']}: {warning['message']}")
        if conf is not None:
            with st.expander("Accuracy breakdown"):
                acc_cols = st.columns(4)
                acc_cols[0].metric("Confidence", f"{conf:.1f}/100")
                cov = details.get("coverage_pct")
                acc_cols[1].metric("Coverage", f"{cov:.1f}%" if cov is not None else "—")
                corr = details.get("correction_rate_pct")
                acc_cols[2].metric("Correction rate", f"{corr:.1f}%" if corr is not None else "—")
                acc_cols[3].metric("Slides inserted", details.get("slides_inserted", 0))
        with st.form("approval_form"):
            approval_status = st.selectbox(
                "Approval decision",
                ["pending", "approved", "needs_revision", "rejected"],
                index=["pending", "approved", "needs_revision", "rejected"].index(
                    details["approval_status"] if details["approval_status"] in {"pending", "approved", "needs_revision", "rejected"} else "pending"
                ),
            )
            approval_notes = st.text_area("Reviewer notes", value=details["approval_notes"], height=100)
            submitted = st.form_submit_button("Save approval", type="primary")
        if submitted:
            update_run_approval(selected_run, approval_status, username, approval_notes or None)
            st.success(f"Updated approval for `{selected_run}`.")
            st.rerun()
        artifact_paths = get_run_artifact_paths(selected_run)
        if artifact_paths:
            action_cols = st.columns(4)
            memo_path = artifact_paths.get("memo")
            log_path = artifact_paths.get("change_log")
            manifest_path = artifact_paths.get("run_manifest")
            if memo_path and os.path.exists(memo_path):
                action_cols[0].download_button(
                    "Download memo",
                    open(memo_path, "rb").read(),
                    file_name=os.path.basename(memo_path),
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    width="stretch",
                )
            if log_path and os.path.exists(log_path):
                action_cols[1].download_button(
                    "Download log",
                    open(log_path, "rb").read(),
                    file_name=os.path.basename(log_path),
                    mime="text/markdown",
                    width="stretch",
                )
            if manifest_path and os.path.exists(manifest_path):
                action_cols[2].download_button(
                    "Download manifest",
                    open(manifest_path, "rb").read(),
                    file_name=os.path.basename(manifest_path),
                    mime="application/json",
                    width="stretch",
                )
            if action_cols[3].button("Requeue from history", width="stretch"):
                input_memo_path = artifact_paths.get("input_memo")
                input_proforma_path = artifact_paths.get("input_proforma")
                if input_memo_path and input_proforma_path and os.path.exists(input_memo_path) and os.path.exists(input_proforma_path):
                    payload = {
                        "job_id": uuid.uuid4().hex,
                        "memo_name": details["memo_name"],
                        "memo_bytes": open(input_memo_path, "rb").read(),
                        "proforma_name": details["proforma_name"],
                        "proforma_bytes": open(input_proforma_path, "rb").read(),
                        "schedule_name": os.path.basename(artifact_paths["input_schedule"]) if artifact_paths.get("input_schedule") else None,
                        "schedule_bytes": open(artifact_paths["input_schedule"], "rb").read() if artifact_paths.get("input_schedule") and os.path.exists(artifact_paths["input_schedule"]) else None,
                        "market_data_name": os.path.basename(artifact_paths["input_market_data"]) if artifact_paths.get("input_market_data") else None,
                        "market_data_bytes": open(artifact_paths["input_market_data"], "rb").read() if artifact_paths.get("input_market_data") and os.path.exists(artifact_paths["input_market_data"]) else None,
                        "property_name": details["property_name"] or None,
                        "dry_run": details["dry_run"],
                        "skip_validation": details["skip_validation"],
                        "profile_name": "",
                    }
                    enqueue_job(username, payload)
                    st.success(f"Requeued `{details['memo_name']}`.")
                    st.rerun()
                else:
                    st.error("Stored input artifacts were not found for this run.")


def render_operations_tab() -> None:
    st.subheader("Operations")
    st.caption("Batch execution, platform health, and profile inventory.")
    ops_tabs = st.tabs(["Queue", "Health", "Profiles"])

    with ops_tabs[0]:
        queue = [job for job in get_job_queue(None if role == "admin" else username) if job["status"] in {"queued", "running", "failed"}]
        if queue:
            queue_rows = [
                {
                    "Job ID": item["job_id"],
                    "Status": item["status"],
                    "Memo": item["payload"]["memo_name"],
                    "Proforma": item["payload"]["proforma_name"],
                    "Property": item["payload"].get("property_name") or "",
                    "Preview": "Yes" if item["payload"].get("dry_run") else "No",
                    "Skip QA": "Yes" if item["payload"].get("skip_validation") else "No",
                    "Profile": item["payload"].get("profile_name", ""),
                    "Run ID": item["run_id"],
                    "Error": item["error_message"],
                }
                for item in queue
            ]
            st.dataframe(queue_rows, width="stretch", hide_index=True)
            n_queued = sum(1 for item in queue if item["status"] == "queued")
            n_running = sum(1 for item in queue if item["status"] == "running")
            if n_running:
                st.info(f"Background worker is processing a job. {n_queued} remaining in queue.")
            elif n_queued:
                st.caption(f"{n_queued} queued job(s) will auto-start within ~10 seconds.")
            queue_cols = st.columns(4)
            if queue_cols[0].button("Refresh status", type="primary", width="stretch"):
                st.rerun()
            selected_job_id = queue_cols[1].selectbox(
                "Job",
                [item["job_id"] for item in queue],
                key="ops_job_select",
                label_visibility="collapsed",
            )
            if queue_cols[2].button("Delete selected job", width="stretch"):
                delete_job(selected_job_id)
                st.info(f"Deleted job `{selected_job_id}`.")
                st.rerun()
            if queue_cols[3].button("Retry failed job", width="stretch"):
                failed_job = get_job(selected_job_id)
                if failed_job and failed_job["status"] == "failed":
                    update_job_status(selected_job_id, "queued", error_message=None)
                    st.success(f"Job `{selected_job_id}` moved back to queued.")
                    st.rerun()
                st.warning("Select a failed job to retry.")
        else:
            st.info("No queued jobs yet.")

    with ops_tabs[1]:
        health_rows = get_platform_health()
        st.dataframe(health_rows, width="stretch", hide_index=True)

    with ops_tabs[2]:
        profiles = get_profiles(None if role == "admin" else username)
        if profiles:
            st.dataframe(profiles, width="stretch", hide_index=True)
        else:
            st.info("No saved profiles yet.")


def render_admin_tab() -> None:
    st.subheader("Admin")
    st.caption("Manage users, credits, and review system activity.")
    users = get_users()
    rows = []
    for user_name, user_cfg in users.items():
        credit_limit = int(user_cfg.get("credits_per_week", 5))
        try:
            used_count, remaining_count = get_user_credits(user_name, credit_limit)
        except Exception:
            used_count, remaining_count = 0, 0
        rows.append(
            {
                "User": user_name,
                "Role": user_cfg.get("role", "user"),
                "Used": used_count,
                "Limit": credit_limit,
                "Remaining": remaining_count,
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)

    admin_tabs = st.tabs(["Invite user", "Add user", "Edit user", "Delete user", "Reset credits", "Invitations", "Recent activity"])

    with admin_tabs[0]:
        st.caption("Generate an invite link to share with a new user.")
        with st.form("invite_user_form"):
            invite_email = st.text_input("Email address")
            invite_role = st.selectbox("Role", ["user", "admin"], index=0)
            invite_credits = st.number_input("Credits per week", min_value=1, value=5)
            invite_submitted = st.form_submit_button("Generate invite link", type="primary")
        if invite_submitted:
            if not invite_email.strip() or "@" not in invite_email:
                st.error("Please enter a valid email address.")
            else:
                token = create_invitation(
                    email=invite_email.strip(),
                    role=invite_role,
                    credits_per_week=int(invite_credits),
                    invited_by=username,
                )
                invite_url = f"https://memochef.streamlit.app?invite={token}"
                st.success(f"Invite created for **{invite_email.strip()}**")
                st.code(invite_url, language=None)
                st.caption("Copy this link and send it to the user.")

    with admin_tabs[1]:
        with st.form("add_user_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            new_role = st.selectbox("Role", ["user", "admin"], index=0)
            new_credits = st.number_input("Credits per week", min_value=1, value=5)
            submitted = st.form_submit_button("Create user", type="primary")
        if submitted:
            if not new_username.strip():
                st.error("Username is required.")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                if add_user(new_username.strip(), new_password, new_role, int(new_credits)):
                    st.success(f"User `{new_username}` created.")
                    st.rerun()
                st.error(f"User `{new_username}` already exists.")

    with admin_tabs[2]:
        usernames = [row["User"] for row in rows]
        selected = st.selectbox("User", usernames, index=None, placeholder="Select a user")
        if selected:
            current_cfg = users[selected]
            with st.form("edit_user_form"):
                edit_role = st.selectbox(
                    "Role",
                    ["user", "admin"],
                    index=0 if current_cfg.get("role", "user") == "user" else 1,
                )
                edit_credits = st.number_input(
                    "Credits per week",
                    min_value=1,
                    value=int(current_cfg.get("credits_per_week", 5)),
                )
                edit_password = st.text_input("New password", type="password")
                submitted = st.form_submit_button("Save changes", type="primary")
            if submitted:
                if edit_password and len(edit_password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    update_user(
                        selected,
                        role=edit_role,
                        credits_per_week=int(edit_credits),
                        new_password=edit_password or None,
                    )
                    if selected == username:
                        st.session_state["role"] = edit_role
                        st.session_state["credits_per_week"] = int(edit_credits)
                    st.success(f"Updated `{selected}`.")
                    st.rerun()

    with admin_tabs[3]:
        deletable = [row["User"] for row in rows if row["User"] != username]
        selected = st.selectbox("User to delete", deletable, index=None, placeholder="Select a user")
        if selected and st.button(f"Delete {selected}"):
            delete_user(selected)
            st.success(f"Deleted `{selected}`.")
            st.rerun()

    with admin_tabs[4]:
        selected = st.selectbox(
            "User to reset",
            [row["User"] for row in rows],
            index=None,
            placeholder="Select a user",
        )
        if selected and st.button(f"Reset credits for {selected}"):
            reset_user_credits(selected)
            st.success(f"Credits reset for `{selected}`.")
            st.rerun()

    with admin_tabs[5]:
        invitations = get_invitations()
        if not invitations:
            st.info("No invitations sent yet.")
        else:
            inv_rows = []
            for inv in invitations:
                status = inv["status"]
                if status == "pending" and inv["expires_at"] < datetime.now(inv["expires_at"].tzinfo):
                    status = "expired"
                inv_rows.append({
                    "Email": inv["email"],
                    "Role": inv["role"],
                    "Credits": inv["credits_per_week"],
                    "Status": status,
                    "Invited by": inv["invited_by"],
                    "Sent": inv["created_at"].strftime("%Y-%m-%d %H:%M") if inv["created_at"] else "",
                })
            st.dataframe(inv_rows, width="stretch", hide_index=True)

    with admin_tabs[6]:
        try:
            runs = get_recent_runs(None, limit=200)
        except Exception as err:
            st.warning(f"Recent activity is unavailable: {err}")
            runs = []
        if not runs:
            st.info("No recorded runs yet.")
        else:
            st.dataframe(runs, width="stretch", hide_index=True)
            st.divider()
            st.markdown("#### Download run logs")
            run_labels = [
                f"{r['Run ID'][:8]}… — {r['User']} — {r['Property'] or r['Memo']} ({r['Created']})"
                for r in runs
            ]
            selected_idx = st.selectbox(
                "Select a run",
                range(len(run_labels)),
                format_func=lambda i: run_labels[i],
                key="admin_run_select",
            )
            selected_run_id = runs[selected_idx]["Run ID"]
            dl_cols = st.columns(3)

            # Try file-based artifacts first, fall back to DB
            artifact_paths = get_run_artifact_paths(selected_run_id)
            log_path = artifact_paths.get("change_log")
            manifest_path = artifact_paths.get("run_manifest")
            details = None

            if log_path and os.path.exists(log_path):
                dl_cols[0].download_button(
                    "Download change log",
                    open(log_path, "rb").read(),
                    file_name=f"{selected_run_id}_change_log{os.path.splitext(log_path)[1]}",
                    mime="text/markdown",
                    key="admin_dl_log",
                )
            else:
                details = get_run_details(selected_run_id)
                if details and details.get("change_log_html"):
                    dl_cols[0].download_button(
                        "Download change log",
                        details["change_log_html"].encode("utf-8"),
                        file_name=f"{selected_run_id}_change_log.md",
                        mime="text/markdown",
                        key="admin_dl_log",
                    )
                else:
                    dl_cols[0].caption("No change log available")

            if manifest_path and os.path.exists(manifest_path):
                dl_cols[1].download_button(
                    "Download manifest",
                    open(manifest_path, "rb").read(),
                    file_name=f"{selected_run_id}_manifest.json",
                    mime="application/json",
                    key="admin_dl_manifest",
                )
            else:
                if details is None:
                    details = get_run_details(selected_run_id)
                if details and details.get("run_manifest_json"):
                    dl_cols[1].download_button(
                        "Download manifest",
                        details["run_manifest_json"].encode("utf-8"),
                        file_name=f"{selected_run_id}_manifest.json",
                        mime="application/json",
                        key="admin_dl_manifest",
                    )
                else:
                    dl_cols[1].caption("No manifest available")


def render_how_to_tab() -> None:
    st.subheader("How To")
    st.caption("Quick-start guide and tips for your team.")
    with st.expander("Getting Started", expanded=True):
        st.info("Content coming soon")
    with st.expander("Input Requirements"):
        st.info("Content coming soon")
    with st.expander("Understanding Results"):
        st.info("Content coming soon")


with tabs[0]:
    render_new_run_tab()

with tabs[1]:
    render_history_tab()

with tabs[2]:
    render_operations_tab()

with tabs[3]:
    render_how_to_tab()

if role == "admin":
    with tabs[4]:
        render_admin_tab()
