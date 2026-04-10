#!/usr/bin/env python3
"""
FastAPI backend for Memo Chef (Managed Agents edition).

Endpoints:
  POST /api/run       -- upload files, start a session, return session_id
  GET  /api/stream/{session_id} -- SSE stream of agent events
  GET  /api/files/{session_id}  -- list output files
  GET  /api/download/{file_id}  -- download a single output file
  GET  /                        -- serve the frontend

Usage:
    uvicorn managed_agents.server:app --reload --port 8501
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from managed_agents.config import AGENT_ID, ENVIRONMENT_ID
from managed_agents.run_session import (
    build_user_message,
    create_session,
    download_file_to,
    get_output_files,
    send_message,
    stream_events,
    upload_example_memos,
    upload_file,
)

app = FastAPI(title="Memo Chef — Managed Agents", version="0.1.0")

# Serve static frontend files
_frontend_dir = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")

# Temp directory for downloads
_tmp = Path(tempfile.mkdtemp(prefix="memochef_"))


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main frontend page."""
    return (_frontend_dir / "index.html").read_text(encoding="utf-8")


@app.post("/api/run")
async def start_run(
    proforma: UploadFile = File(...),
    memo: UploadFile = File(...),
    supplemental: list[UploadFile] = File(default=[]),
    instructions: str = Form(default=""),
):
    """Upload files, create a session, send the initial message, return session ID."""
    if not AGENT_ID or not ENVIRONMENT_ID:
        return {"error": "Agent or environment not provisioned. Run setup scripts first."}

    resources = []

    # Upload proforma
    proforma_path = _tmp / proforma.filename
    proforma_path.write_bytes(await proforma.read())
    proforma_file_id = upload_file(proforma_path)
    resources.append({
        "type": "file",
        "file_id": proforma_file_id,
        "mount_path": f"/mnt/session/uploads/{proforma.filename}",
    })

    # Upload memo template
    memo_path = _tmp / memo.filename
    memo_path.write_bytes(await memo.read())
    memo_file_id = upload_file(memo_path)
    resources.append({
        "type": "file",
        "file_id": memo_file_id,
        "mount_path": f"/mnt/session/uploads/{memo.filename}",
    })

    # Upload supplemental files
    supplemental_names = []
    for sup_file in supplemental:
        if sup_file.filename:
            sup_path = _tmp / sup_file.filename
            sup_path.write_bytes(await sup_file.read())
            sup_id = upload_file(sup_path)
            resources.append({
                "type": "file",
                "file_id": sup_id,
                "mount_path": f"/mnt/session/uploads/{sup_file.filename}",
            })
            supplemental_names.append(sup_file.filename)

    # Upload example memos for house-style reference
    example_resources = upload_example_memos()
    resources.extend(example_resources)

    # Create session
    session_id = create_session(
        uploaded_resources=resources,
        title=f"Memo Chef: {proforma.filename}",
    )

    # Build and send initial message in background thread
    message = build_user_message(
        proforma_filename=proforma.filename,
        memo_filename=memo.filename,
        supplemental_filenames=supplemental_names or None,
        instructions=instructions,
    )

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, send_message, session_id, message)

    return {"session_id": session_id}


@app.get("/api/stream/{session_id}")
async def stream_session(session_id: str):
    """SSE endpoint that streams agent events."""

    def _generate():
        for event in stream_events(session_id):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/files/{session_id}")
async def list_output_files(session_id: str):
    """List files the agent wrote during the session."""
    files = get_output_files(session_id)
    return {"files": files}


@app.get("/api/download/{file_id}")
async def download_output_file(file_id: str, filename: str = "output.pptx"):
    """Download a specific output file."""
    dest = _tmp / f"{file_id}_{filename}"
    download_file_to(file_id, dest)
    return FileResponse(
        path=str(dest),
        filename=filename,
        media_type="application/octet-stream",
    )
