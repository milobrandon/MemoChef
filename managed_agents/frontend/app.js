/**
 * Memo Chef — Managed Agents Frontend
 *
 * Handles file upload, SSE streaming of agent events, and output download.
 */

const form = document.getElementById("run-form");
const runBtn = document.getElementById("run-btn");
const btnText = runBtn.querySelector(".btn-text");
const btnLoading = runBtn.querySelector(".btn-loading");
const progressSection = document.getElementById("progress-section");
const statusLabel = document.getElementById("status-label");
const eventLog = document.getElementById("event-log");
const resultsSection = document.getElementById("results-section");
const outputFiles = document.getElementById("output-files");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  // Validate required files
  const proforma = document.getElementById("proforma").files[0];
  const memo = document.getElementById("memo").files[0];
  if (!proforma || !memo) {
    alert("Please select both a proforma and a memo template.");
    return;
  }

  // Disable form and show progress
  setRunning(true);
  progressSection.hidden = false;
  resultsSection.hidden = true;
  eventLog.innerHTML = "";
  outputFiles.innerHTML = "";
  addLog("Uploading files and starting session...", "status");

  try {
    // Build form data
    const fd = new FormData();
    fd.append("proforma", proforma);
    fd.append("memo", memo);

    const supplemental = document.getElementById("supplemental").files;
    for (const f of supplemental) {
      fd.append("supplemental", f);
    }

    const instructions = document.getElementById("instructions").value.trim();
    if (instructions) {
      fd.append("instructions", instructions);
    }

    const projectName = document.getElementById("project-name").value.trim();
    if (projectName) {
      fd.append("project_name", projectName);
    }

    const meetingLookback = parseInt(
      document.getElementById("meeting-lookback").value,
      10,
    );
    if (Number.isFinite(meetingLookback) && meetingLookback > 0) {
      fd.append("meeting_lookback_days", String(meetingLookback));
    }

    // Start the run
    const resp = await fetch("/api/run", { method: "POST", body: fd });
    const data = await resp.json();

    if (data.error) {
      addLog(`Error: ${data.error}`, "error");
      setRunning(false);
      return;
    }

    const sessionId = data.session_id;
    addLog(`Session created: ${sessionId}`, "status");
    statusLabel.textContent = "Agent is working...";

    // Stream events via SSE
    await streamEvents(sessionId);

    // Fetch output files
    statusLabel.textContent = "Retrieving output files...";
    await fetchOutputFiles(sessionId);

    statusLabel.textContent = "Complete!";
  } catch (err) {
    addLog(`Error: ${err.message}`, "error");
    statusLabel.textContent = "Failed";
  } finally {
    setRunning(false);
  }
});

async function streamEvents(sessionId) {
  const resp = await fetch(`/api/stream/${sessionId}`);
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop(); // Keep incomplete line in buffer

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const json = line.slice(6);
      if (!json) continue;

      try {
        const event = JSON.parse(json);
        handleEvent(event);

        if (
          event.type === "session.status_idle" ||
          event.type === "session.status_terminated"
        ) {
          return;
        }
      } catch {
        // Skip malformed JSON
      }
    }
  }
}

function handleEvent(event) {
  switch (event.type) {
    case "agent.message":
      if (event.text) {
        // Show a truncated preview of agent messages
        const preview =
          event.text.length > 200
            ? event.text.slice(0, 200) + "..."
            : event.text;
        addLog(preview, "message");
      }
      break;

    case "agent.tool_use":
      addLog(`[Tool] ${event.name}`, "tool");
      statusLabel.textContent = `Using: ${event.name}`;
      break;

    case "agent.tool_result":
      // Optionally show brief tool results
      break;

    case "session.status_idle":
      addLog("Agent finished.", "status");
      break;

    case "session.status_terminated":
      addLog("Session terminated.", "error");
      break;

    case "session.error":
      addLog(`Error: ${JSON.stringify(event.error)}`, "error");
      break;

    case "span.model_request_start":
      statusLabel.textContent = "Thinking...";
      break;

    case "span.model_request_end":
      statusLabel.textContent = "Agent is working...";
      break;

    default:
      // Other event types (thinking, etc.) — skip
      break;
  }
}

async function fetchOutputFiles(sessionId) {
  const resp = await fetch(`/api/files/${sessionId}`);
  const data = await resp.json();

  if (!data.files || data.files.length === 0) {
    addLog("No output files found.", "error");
    return;
  }

  resultsSection.hidden = false;
  for (const file of data.files) {
    const card = document.createElement("div");
    card.className = "file-card";

    const sizeKB = Math.round(file.size_bytes / 1024);
    card.innerHTML = `
      <div>
        <span class="file-name">${escapeHtml(file.filename)}</span>
        <span class="file-size">(${sizeKB} KB)</span>
      </div>
      <a href="/api/download/${file.id}?filename=${encodeURIComponent(file.filename)}"
         download="${escapeHtml(file.filename)}">
        Download
      </a>
    `;
    outputFiles.appendChild(card);
  }
}

function addLog(text, className = "") {
  const entry = document.createElement("div");
  entry.className = `log-entry ${className}`;
  entry.textContent = text;
  eventLog.appendChild(entry);
  eventLog.scrollTop = eventLog.scrollHeight;
}

function setRunning(running) {
  runBtn.disabled = running;
  btnText.hidden = running;
  btnLoading.hidden = !running;

  // Disable file inputs during run
  for (const input of form.querySelectorAll("input, textarea")) {
    input.disabled = running;
  }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
