#!/usr/bin/env python3
"""Modern Streamlit dashboard for Memo Automator."""

from __future__ import annotations

import glob
import os
import time
import uuid

import streamlit as st

from app_helpers import should_disable_fire_button, verify_password
from app_services import (
    add_user,
    consume_credit,
    delete_job,
    delete_user,
    enqueue_job,
    ensure_users_seeded,
    get_db_conn,
    get_job,
    get_job_queue,
    get_platform_health,
    get_profiles,
    get_recent_runs,
    get_run_artifact_paths,
    get_run_storage_dir,
    get_run_details,
    get_user_credits,
    get_users,
    record_run,
    reset_user_credits,
    save_profile,
    update_job_status,
    update_run_approval,
    update_user,
)
from memo_chef.models import RunRequest, StageUpdate
from memo_chef.pipeline import run_memo_pipeline
from memo_chef.theme import APP_SUBTITLE, APP_TITLE, app_css, hero_html, info_card

st.set_page_config(page_title=APP_TITLE, page_icon="✨", layout="wide")
st.markdown(app_css(), unsafe_allow_html=True)

try:
    ensure_users_seeded()
except Exception:
    pass


_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")


def _list_config_profiles() -> list[str]:
    """Return stem names of YAML files in configs/, sorted."""
    if not os.path.isdir(_CONFIGS_DIR):
        return []
    return sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(_CONFIGS_DIR, "*.yaml"))
    )


def _config_override_path(profile_name: str | None) -> str | None:
    if not profile_name:
        return None
    path = os.path.join(_CONFIGS_DIR, f"{profile_name}.yaml")
    return path if os.path.exists(path) else None


def _get_api_key() -> str | None:
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None


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
    ]:
        st.session_state.pop(key, None)


def _queue_item_from_inputs(
    *,
    memo_file,
    proforma_file,
    schedule_file,
    market_data_file,
    property_name: str,
    dry_run: bool,
    skip_validation: bool,
    profile_name: str | None,
    config_profile_name: str | None = None,
) -> dict:
    return {
        "job_id": uuid.uuid4().hex,
        "memo_name": memo_file.name,
        "memo_bytes": memo_file.getvalue(),
        "proforma_name": proforma_file.name,
        "proforma_bytes": proforma_file.getvalue(),
        "schedule_name": schedule_file.name if schedule_file else None,
        "schedule_bytes": schedule_file.getvalue() if schedule_file else None,
        "market_data_name": market_data_file.name if market_data_file else None,
        "market_data_bytes": market_data_file.getvalue() if market_data_file else None,
        "property_name": property_name or None,
        "dry_run": dry_run,
        "skip_validation": skip_validation,
        "profile_name": profile_name or "",
        "config_profile_name": config_profile_name or "",
    }


def _persist_result(result, filename: str) -> None:
    st.session_state["memo_bytes"] = result.memo_bytes
    st.session_state["log_bytes"] = result.log_bytes
    st.session_state["manifest_bytes"] = result.manifest_bytes
    st.session_state["filename"] = filename
    st.session_state["n_changes"] = len(result.changes)
    st.session_state["n_rejected"] = len(result.rejected)
    st.session_state["n_missed"] = len(result.missed)
    st.session_state["log_lines"] = result.log_lines
    st.session_state["warnings"] = [warning.model_dump() for warning in result.manifest.warnings]
    st.session_state["manifest"] = result.manifest.model_dump()


def _execute_job(
    *,
    job: dict,
    username: str,
    credits_per_week: int,
    queue_position: int | None = None,
    queue_total: int | None = None,
) -> bool:
    _clear_run_state()
    api_key = _get_api_key()
    if not api_key:
        st.error("`ANTHROPIC_API_KEY` is not configured in Streamlit secrets.")
        return False

    run_id = uuid.uuid4().hex
    if job.get("job_id"):
        update_job_status(job["job_id"], "running", run_id=run_id)
    started = time.time()
    prefix = ""
    if queue_position is not None and queue_total is not None:
        prefix = f"Queue item {queue_position}/{queue_total} · "
    progress_bar = st.progress(0, text=f"{prefix}Initializing run")
    status_box = st.empty()
    stage_log = st.empty()
    stage_lines: list[str] = []

    def on_stage(update: StageUpdate) -> None:
        progress_bar.progress(update.percent, text=f"{prefix}{update.label}")
        message = update.detail or update.label
        stage_lines.append(f"{update.percent:>3}%  {message}")
        status_box.caption(f"{prefix}{update.label}")
        stage_log.code("\n".join(stage_lines[-10:]), language=None)

    run_dir = get_run_storage_dir(run_id)
    memo_path = str(run_dir / f"input_memo{os.path.splitext(job['memo_name'])[1]}")
    proforma_path = str(run_dir / f"input_proforma{os.path.splitext(job['proforma_name'])[1]}")
    with open(memo_path, "wb") as handle:
        handle.write(job["memo_bytes"])
    with open(proforma_path, "wb") as handle:
        handle.write(job["proforma_bytes"])

    schedule_path = None
    if job.get("schedule_bytes"):
        schedule_path = str(run_dir / f"input_schedule{os.path.splitext(job['schedule_name'])[1]}")
        with open(schedule_path, "wb") as handle:
            handle.write(job["schedule_bytes"])

    market_data_path = None
    if job.get("market_data_bytes"):
        market_data_path = str(run_dir / f"input_market_data{os.path.splitext(job['market_data_name'])[1]}")
        with open(market_data_path, "wb") as handle:
            handle.write(job["market_data_bytes"])

    request = RunRequest(
        memo_path=memo_path,
        proforma_path=proforma_path,
        schedule_path=schedule_path,
        market_data_path=market_data_path,
        output_dir=str(run_dir),
        api_key=api_key,
        config_path=os.path.join(os.path.dirname(__file__), "config.yaml"),
        config_override_path=_config_override_path(job.get("config_profile_name")),
        run_id=run_id,
        property_name=job.get("property_name"),
        dry_run=job.get("dry_run", False),
        skip_validation=job.get("skip_validation", False),
    )

    try:
        result = run_memo_pipeline(request, callback=on_stage)
        duration = round(time.time() - started, 2)
        progress_bar.progress(100, text=f"{prefix}Run complete")
        status_box.success(f"{prefix}Draft generated successfully.")
        _persist_result(result, job["memo_name"])
        (run_dir / f"memo{os.path.splitext(job['memo_name'])[1]}").write_bytes(result.memo_bytes)
        (run_dir / "change_log.md").write_bytes(result.log_bytes)
        (run_dir / "run_manifest.json").write_bytes(result.manifest_bytes)
        record_run(
            run_id=run_id,
            username=username,
            status=result.manifest.status,
            memo_name=result.manifest.memo_name,
            proforma_name=result.manifest.proforma_name,
            property_name=result.manifest.property_name,
            dry_run=job.get("dry_run", False),
            skip_validation=job.get("skip_validation", False),
            change_count=len(result.changes),
            rejected_count=len(result.rejected),
            missed_count=len(result.missed),
            duration_seconds=duration,
            warnings=st.session_state["warnings"],
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
        status_box.error(f"{prefix}Run failed")
        stage_log.code("\n".join(stage_lines[-10:]), language=None)
        try:
            record_run(
                run_id=run_id,
                username=username,
                status="failed",
                memo_name=job["memo_name"],
                proforma_name=job["proforma_name"],
                property_name=job.get("property_name"),
                dry_run=job.get("dry_run", False),
                skip_validation=job.get("skip_validation", False),
                change_count=0,
                rejected_count=0,
                missed_count=0,
                duration_seconds=duration,
                warnings=[{"stage": "pipeline", "message": str(err)}],
            )
        except Exception:
            pass
        if job.get("job_id"):
            update_job_status(job["job_id"], "failed", run_id=run_id, error_message=str(err))
        st.error(f"{prefix}Run failed: {err}")
        return False


def check_password() -> bool:
    if st.session_state.get("authenticated"):
        return True

    users = get_users()
    if not users:
        st.error("No users configured. Add users to Streamlit secrets or the database.")
        st.stop()

    st.markdown(hero_html(), unsafe_allow_html=True)
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
        submitted = cols[2].form_submit_button("Sign in", type="primary", use_container_width=True)

    if submitted:
        user_cfg = users.get(username)
        if user_cfg and verify_password(password, user_cfg["password_hash"]):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["role"] = user_cfg.get("role", "user")
            st.session_state["credits_per_week"] = int(user_cfg.get("credits_per_week", 5))
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
    if st.button("Sign out", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

st.markdown(hero_html(), unsafe_allow_html=True)

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

tab_labels = ["New Run", "Run History", "Operations"] + (["Admin"] if role == "admin" else [])
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

    upload_cols = st.columns(4)
    memo_file = upload_cols[0].file_uploader("Memo deck", type=["pptx"], key="memo_upload")
    proforma_file = upload_cols[1].file_uploader("Proforma", type=["xlsx", "xlsm"], key="proforma_upload")
    schedule_file = upload_cols[2].file_uploader("Schedule", type=["mpp"], key="schedule_upload")
    market_data_file = upload_cols[3].file_uploader("Market data", type=["xlsx", "xlsm"], key="market_upload")

    property_name = st.text_input(
        "Property name override",
        value=profile.get("Property", ""),
        placeholder="Optional alias used inside the proforma or memo",
        help="Use this when the property has been renamed or appears differently across sources.",
    )

    config_profiles = _list_config_profiles()
    config_profile_name = st.selectbox(
        "Config profile",
        options=[""] + config_profiles,
        index=(config_profiles.index(profile["Config Profile"]) + 1)
        if profile.get("Config Profile") and profile["Config Profile"] in config_profiles
        else 0,
        format_func=lambda v: "Default (config.yaml)" if v == "" else v.replace("_", " ").title(),
        help="Override proforma tabs, model, or other settings for this property type.",
    )

    pref_cols = st.columns(2)
    with pref_cols[0]:
        dry_run = st.checkbox(
            "Preview only",
            value=bool(profile.get("Preview Only", False)),
            help="Runs the pipeline without saving final deck changes.",
        )
    with pref_cols[1]:
        skip_validation = st.checkbox(
            "Skip AI validation",
            value=bool(profile.get("Skip QA", False)),
            help="Faster, but less safe. Recommended only for trusted dry runs.",
        )

    review_cols = st.columns(4)
    review_cols[0].info("Required inputs: memo + proforma")
    review_cols[1].info("Optional enrichments: schedule + market data")
    review_cols[2].info("Artifacts include a machine-readable run manifest")
    review_cols[3].info("Queue jobs to run sequentially with shared settings")

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
    if profile_cols[0].button("Save profile", use_container_width=True):
        if not save_profile_name.strip():
            st.error("Enter a profile name before saving.")
        else:
            save_profile(
                save_profile_name.strip(),
                username,
                property_name or None,
                dry_run,
                skip_validation,
                profile_notes or None,
                config_profile=config_profile_name or None,
            )
            st.success(f"Saved profile `{save_profile_name.strip()}`.")
            st.rerun()

    action_disabled = should_disable_fire_button(memo_file, proforma_file, remaining, credits_error)
    action_cols = st.columns(2)
    if action_cols[0].button(
        f"Generate draft ({remaining} credits left)" if remaining > 0 else "No credits remaining",
        type="primary",
        disabled=action_disabled,
        use_container_width=True,
    ):
        job = _queue_item_from_inputs(
            memo_file=memo_file,
            proforma_file=proforma_file,
            schedule_file=schedule_file,
            market_data_file=market_data_file,
            property_name=property_name,
            dry_run=dry_run,
            skip_validation=skip_validation,
            profile_name=selected_profile or save_profile_name.strip() or None,
            config_profile_name=config_profile_name or None,
        )
        _execute_job(job=job, username=username, credits_per_week=credits_per_week)

    if action_cols[1].button(
        "Add to queue",
        disabled=action_disabled,
        use_container_width=True,
    ):
        job = _queue_item_from_inputs(
            memo_file=memo_file,
            proforma_file=proforma_file,
            schedule_file=schedule_file,
            market_data_file=market_data_file,
            property_name=property_name,
            dry_run=dry_run,
            skip_validation=skip_validation,
            profile_name=selected_profile or save_profile_name.strip() or None,
            config_profile_name=config_profile_name or None,
        )
        enqueue_job(username, job)
        st.success(f"Queued `{job['memo_name']}`.")

    if "memo_bytes" in st.session_state:
        st.divider()
        st.success("Artifacts are ready for review and download.")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Applied changes", st.session_state["n_changes"])
        metric_cols[1].metric("Rejected", st.session_state["n_rejected"])
        metric_cols[2].metric("Needs review", st.session_state["n_missed"])
        metric_cols[3].metric("Warnings", len(st.session_state.get("warnings", [])))

        download_cols = st.columns(3)
        download_cols[0].download_button(
            "Download updated memo",
            st.session_state["memo_bytes"],
            file_name=st.session_state["filename"],
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            use_container_width=True,
        )
        download_cols[1].download_button(
            "Download change log",
            st.session_state["log_bytes"],
            file_name="CHANGE_LOG.md",
            mime="text/markdown",
            use_container_width=True,
        )
        download_cols[2].download_button(
            "Download run manifest",
            st.session_state["manifest_bytes"],
            file_name="run_manifest.json",
            mime="application/json",
            use_container_width=True,
        )

        warnings = st.session_state.get("warnings", [])
        if warnings:
            with st.expander("Warnings"):
                for warning in warnings:
                    st.warning(f"{warning['stage']}: {warning['message']}")

        with st.expander("Execution log"):
            st.code("\n".join(st.session_state["log_lines"]), language=None)

        with st.expander("Run manifest"):
            st.json(st.session_state["manifest"])


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
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Approval workflow")
    run_options = [row["Run ID"] for row in rows]
    selected_run = st.selectbox("Select run", run_options, index=0)
    details = get_run_details(selected_run)
    if details:
        detail_cols = st.columns(4)
        detail_cols[0].metric("Status", details["status"])
        detail_cols[1].metric("Approval", details["approval_status"])
        detail_cols[2].metric("Changes", details["change_count"])
        detail_cols[3].metric("Warnings", len(details["warnings"]))
        if details["warnings"]:
            with st.expander("Run warnings"):
                for warning in details["warnings"]:
                    st.warning(f"{warning['stage']}: {warning['message']}")
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
                    use_container_width=True,
                )
            if log_path and os.path.exists(log_path):
                action_cols[1].download_button(
                    "Download log",
                    open(log_path, "rb").read(),
                    file_name=os.path.basename(log_path),
                    mime="text/markdown",
                    use_container_width=True,
                )
            if manifest_path and os.path.exists(manifest_path):
                action_cols[2].download_button(
                    "Download manifest",
                    open(manifest_path, "rb").read(),
                    file_name=os.path.basename(manifest_path),
                    mime="application/json",
                    use_container_width=True,
                )
            if action_cols[3].button("Requeue from history", use_container_width=True):
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
            st.dataframe(queue_rows, use_container_width=True, hide_index=True)
            queue_cols = st.columns(4)
            if queue_cols[0].button("Run queued jobs", type="primary", use_container_width=True):
                queued_items = [item for item in queue if item["status"] == "queued"]
                for index, item in enumerate(queued_items, start=1):
                    ok = _execute_job(
                        job={**item["payload"], "job_id": item["job_id"]},
                        username=username,
                        credits_per_week=credits_per_week,
                        queue_position=index,
                        queue_total=len(queued_items),
                    )
                    if not ok:
                        break
                st.success("Queue execution finished.")
                st.rerun()
            selected_job_id = queue_cols[1].selectbox(
                "Job",
                [item["job_id"] for item in queue],
                key="ops_job_select",
                label_visibility="collapsed",
            )
            if queue_cols[2].button("Delete selected job", use_container_width=True):
                delete_job(selected_job_id)
                st.info(f"Deleted job `{selected_job_id}`.")
                st.rerun()
            if queue_cols[3].button("Retry failed job", use_container_width=True):
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
        st.dataframe(health_rows, use_container_width=True, hide_index=True)

    with ops_tabs[2]:
        profiles = get_profiles(None if role == "admin" else username)
        if profiles:
            st.dataframe(profiles, use_container_width=True, hide_index=True)
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
    st.dataframe(rows, use_container_width=True, hide_index=True)

    admin_tabs = st.tabs(["Add user", "Edit user", "Delete user", "Reset credits", "Recent activity"])

    with admin_tabs[0]:
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

    with admin_tabs[1]:
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

    with admin_tabs[2]:
        deletable = [row["User"] for row in rows if row["User"] != username]
        selected = st.selectbox("User to delete", deletable, index=None, placeholder="Select a user")
        if selected and st.button(f"Delete {selected}"):
            delete_user(selected)
            st.success(f"Deleted `{selected}`.")
            st.rerun()

    with admin_tabs[3]:
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

    with admin_tabs[4]:
        try:
            runs = get_recent_runs(None, limit=30)
        except Exception as err:
            st.warning(f"Recent activity is unavailable: {err}")
        else:
            st.dataframe(runs, use_container_width=True, hide_index=True)


with tabs[0]:
    render_new_run_tab()

with tabs[1]:
    render_history_tab()

with tabs[2]:
    render_operations_tab()

if role == "admin":
    with tabs[3]:
        render_admin_tab()
