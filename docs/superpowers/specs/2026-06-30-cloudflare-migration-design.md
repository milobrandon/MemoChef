# Cloudflare Migration Design — Memo Chef (DEFERRED plan)

> **Status: DEFERRED, not executed.** The user explicitly chose to stay on
> Streamlit for now ("keep it on streamlit for now and save migration for another
> time"). This document is the ready-to-execute plan for when that deferral is
> lifted. **Nothing in production changes until someone explicitly starts the
> cutover steps below.** Date: 2026-06-30.

## 1. Goal

Replace the Streamlit deployment of Memo Chef with a Cloudflare-hosted front end,
without losing functionality. "Migrate to Cloudflare instead of Streamlit."

## 2. Why this is feasible (and what actually moves)

The repo already contains the architecture that makes Cloudflare viable: the
**managed-agent** path (`managed_agents/`). In that design, all heavy Office/Python
work (python-pptx, openpyxl, pdfplumber) runs **inside Anthropic's agent sandbox**,
not on our server. The only thing we host is a **thin proxy + a static frontend**:

- `managed_agents/server.py` — a 180-line FastAPI proxy with six endpoints:
  `POST /api/run` (multipart upload → Anthropic Files API → create session → send
  message), `GET /api/stream/{id}` (SSE relay), `GET /api/files/{id}`,
  `GET /api/download/{id}`, and `GET /` (serve the page).
- `managed_agents/frontend/` — static `index.html` + `app.js` + `style.css`
  (~650 lines, no build step).

Because the proxy does **no heavy computation**, it ports to Cloudflare cleanly.
The Streamlit monolith (`app.py`, `app_services.py`, `memo_automator.py`) and its
local Python pipeline are **not** migrated — the managed agent supersedes them as
the cloud product. The local pipeline stays available via the `/memo-chef` Claude
Code skill for local runs.

## 3. Recommended architecture

**Cloudflare Pages (static frontend) + a Cloudflare Worker (TypeScript) for the API.**

```
Browser ──▶ Cloudflare Pages         (managed_agents/frontend/*, static)
        └─▶ Worker (TypeScript)      rewrite of the 6 endpoints
              ├─ multipart upload  ─▶ Anthropic Files API
              ├─ create session / send message ─▶ Anthropic Agents/Sessions API
              ├─ SSE relay (ReadableStream) ◀─ Anthropic stream
              └─ file list / download ◀─▶ Anthropic Files API
            secrets: ANTHROPIC_API_KEY (Worker secret), MANAGED_AGENT_ID,
                     MANAGED_ENVIRONMENT_ID
            R2 bucket: only if uploads exceed Worker request limits (see §6)
```

The Worker reimplements `run_session.py` + `api_client.py`'s HTTP calls in TS
(`fetch` + `ReadableStream` for SSE). No Python at the edge, scales to zero,
cheapest option.

### Alternatives considered (and why not)
- **Python Workers (Pyodide).** Keeps `server.py` mostly as-is, but SSE
  streaming + `httpx` + multipart on Pyodide are beta and not all proven; risky
  for a streaming proxy. Reconsider only if the TS rewrite proves expensive.
- **Cloudflare Container running the FastAPI proxy unchanged.** Minimal code
  change, but it's a long-running container (not scale-to-zero), costlier, and
  reintroduces a Python runtime at the edge — defeats the point.

## 4. Auth & credits

The Streamlit app has email/password auth + a Postgres credits system + an admin
console. Recommended for the Cloudflare app:

- **Cloudflare Access (Zero Trust)** in front of Pages + Worker — SSO / email
  allowlist, no database to run. Drop the Postgres credits + admin system.
- If usage metering is still required later, port credits as a Worker + Postgres
  (via Hyperdrive) — but treat as a separate follow-up, not part of the cutover.

## 5. College House data

The agent sandbox cannot reach the Azure SQL DB (`subtextresearch...`). Keep the
current model: the user uploads a pre-built `college_house_extract.xlsx` with the
proforma. No live SQL at the edge (a Worker can't run pyodbc, and the Azure
firewall won't allow-list edge IPs). This matches the managed-agent design and the
Cowork skill's approach.

## 6. File handling & limits

- Uploads (proforma `.xlsm` + memo `.pptx`) can be tens of MB. Stream multipart
  straight to the Anthropic Files API where possible. If a file exceeds the
  Worker request-body limit, buffer through an **R2** bucket (presigned PUT from
  the browser, Worker reads from R2 → Files API).
- Downloads stream from the Files API back through the Worker.

## 7. Migration steps (execute only when un-deferred)

1. **Provision** a Cloudflare account/project; create the Worker + Pages project;
   set `ANTHROPIC_API_KEY`, `MANAGED_AGENT_ID`, `MANAGED_ENVIRONMENT_ID` as Worker
   secrets; (optional) create an R2 bucket.
2. **Port the proxy** — rewrite the 6 endpoints in TypeScript (`wrangler` project).
   Reuse the request/response shapes in `managed_agents/run_session.py` and
   `api_client.py` as the spec.
3. **Deploy the frontend** — publish `managed_agents/frontend/` to Pages; point its
   API calls at the Worker route.
4. **Wire auth** — put Cloudflare Access in front; configure the email allowlist.
5. **Parallel run** — keep Streamlit live; test the Cloudflare app end-to-end
   (upload → stream → download) against real proforma+memo pairs.
6. **Cutover** — move the DNS/custom domain to the Cloudflare app once verified.
7. **Decommission** — retire the Render/Streamlit service after a soak period.

## 8. Risks & rollback

- **SSE longevity / Worker CPU limits** for long agent runs — validate during the
  parallel run; mitigate with the SSE-resume logic already present server-side.
- **Large-file uploads** — the R2 fallback (§6) covers this.
- **Rollback** is trivial during parallel run: Streamlit stays up until cutover;
  revert DNS to roll back.

## 9. Out of scope

- Re-platforming the local Python pipeline (`memo_automator.py`) — stays as the
  local `/memo-chef` skill.
- Porting the credits/admin system (separate follow-up if needed).

## 10. Open decisions to confirm before executing

1. Hosting shape: TS Worker + Pages (recommended) vs Python Worker vs Container.
2. Auth: Cloudflare Access + drop credits (recommended) vs port credits.
3. Whether the Streamlit app is decommissioned or kept as a fallback.

(These were the three forks raised at the start of the engagement; they remain the
first questions to settle when the migration is un-deferred.)
