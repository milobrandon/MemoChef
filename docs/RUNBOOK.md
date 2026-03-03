# Memo Automator — Operations Runbook

> Procedures for deploying, operating, and troubleshooting the Memo Automator.

---

## Deployment

### Streamlit Cloud

1. Sign in to [share.streamlit.io](https://share.streamlit.io) with the GitHub account that owns the repo.
2. Click **New app**.
3. Select repo `milobrandon/MemoChef`, branch `main`, main file `app.py`.
4. Under **Advanced settings**, paste the contents of `.streamlit/secrets.toml` into the secrets editor.
5. Click **Deploy**.
6. The app will be available at `https://<app-name>.streamlit.app`.

### Updating a Deployed App

Push to `main` — Streamlit Cloud auto-deploys on every push. Monitor the deploy log in the Streamlit dashboard for errors.

### Local Development

```bash
git clone https://github.com/milobrandon/MemoChef.git
cd MemoChef
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env         # add your ANTHROPIC_API_KEY
streamlit run app.py         # web UI on localhost:8501
```

---

## Key Rotation

### Anthropic API Key

1. Go to [console.anthropic.com](https://console.anthropic.com/) → API Keys.
2. Create a new key.
3. Update the key in:
   - `.env` (local development)
   - Streamlit Cloud secrets (Settings → Secrets)
   - Any team members' local `.env` files
4. Revoke the old key.
5. Verify the app works by running a test memo.

### Neon Database Password

1. Go to the [Neon Console](https://console.neon.tech/) → Connection Details.
2. Reset the role password.
3. Update `DATABASE_URL` in `.env` and Streamlit Cloud secrets.
4. Verify by logging into the app and confirming credits load correctly.

---

## Adding Users

The app uses per-user authentication stored in a Neon Postgres database.

1. Connect to the database:
   ```bash
   psql "$DATABASE_URL"
   ```
2. Insert a new user:
   ```sql
   INSERT INTO users (username, password_hash, display_name)
   VALUES ('newuser', crypt('their-password', gen_salt('bf')), 'Display Name');
   ```
3. The user gets 10 credits per week (reset on Monday at midnight UTC).

---

## Troubleshooting

### "ANTHROPIC_API_KEY not set"

- **Local:** Check that `.env` exists and contains `ANTHROPIC_API_KEY=sk-ant-...`.
- **Streamlit Cloud:** Check Settings → Secrets for the key.

### "Tab 'X' not found in proforma — skipping"

The proforma doesn't have the expected tab name. Either:
- Rename the tab in Excel to match `config.yaml` → `proforma.tabs`.
- Or update `config.yaml` to match the actual tab names.

### Claude API timeout / rate limit

- The tool waits 65 seconds between batched API calls to stay under rate limits.
- For very large decks (30+ slides), increase `timeout` in the Anthropic client (currently 900s).
- If rate-limited, wait 60 seconds and retry.

### openpyxl returns None for formula cells

Open the proforma in Excel, let it calculate, **Save**, close, then re-run. `openpyxl` with `data_only=True` reads cached formula values, which only exist after a save.

### "No changes applied" but memo has stale values

- Check `mappings_raw.json` — did Claude identify any mappings?
- Check `mappings_validated.json` — were all mappings rejected?
- Verify the proforma tabs contain updated values (not stale from an older version).
- Try `--skip-validation` to bypass QA and see if mappings apply without the validation filter.

### PPTX formatting looks wrong after update

- The tool preserves run-level formatting. If formatting breaks, check that the original memo didn't have mixed formatting within a single text box.
- Branding is applied after updates — verify the theme file (`Subtext Brand Theme.thmx`) is present.

### Java / JPype errors (schedule extraction)

- Schedule `.mpp` extraction requires Java. Install a JRE/JDK and ensure `java` is on PATH.
- On Streamlit Cloud, add `packages.txt` with `default-jre` for apt-based Java installation.

---

## Monitoring

### API Costs

- Check the [Anthropic dashboard](https://console.anthropic.com/) for usage and spend.
- Each memo run uses 2 API calls (mapping + validation). Large decks may use more due to batching.
- Typical cost: $0.10–$0.50 per memo run with Sonnet.

### Database

- Neon dashboard shows connection count, storage, and compute usage.
- Free tier: 0.5 GB storage, 190 compute hours/month.

### Application Logs

- **Streamlit Cloud:** View logs in the dashboard → Manage app → Logs.
- **Local:** Logs print to stdout with timestamps and severity levels.
