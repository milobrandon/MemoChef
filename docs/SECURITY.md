# Security Policy

## Data Sensitivity

Memo Automator processes **confidential financial data** including:

- Projected cash flows, returns (IRR, yield-on-cost), and cap rates
- Unit mix, rent rolls, and operating expense budgets
- Development budgets and construction timelines
- Property acquisition pricing and deal terms

This data is **material non-public information** in many contexts. Handle accordingly.

---

## API Key Management

### Anthropic API Key

| Rule | Detail |
|------|--------|
| Storage | `.env` file (local) or Streamlit secrets (deployed) |
| Never commit | `.env` is in `.gitignore` — verify before every push |
| Rotation | Rotate keys quarterly or immediately if exposed |
| Scope | Use a dedicated key for this project, not a personal key |
| Monitoring | Check [Anthropic dashboard](https://console.anthropic.com/) for unexpected usage |

### If a Key Is Compromised

1. **Immediately** revoke the key at [console.anthropic.com](https://console.anthropic.com/).
2. Generate a new key.
3. Update `.env` / Streamlit secrets.
4. Review API usage logs for unauthorized calls.
5. Notify the team.

---

## Data Handling

### What Gets Sent to the Claude API

Every run sends the following to Anthropic's API:

1. **Proforma text extraction** — Raw text from specified Excel tabs (financial metrics, not formulas).
2. **Memo text extraction** — Raw text from all PowerPoint slides (metrics, labels, narrative).
3. **Schedule data** (optional) — Task names, dates, milestones from .mpp files.

### What Does NOT Get Sent

- File paths or system information
- API keys or credentials
- Images, charts, or embedded objects
- Raw binary file content

### Anthropic's Data Policy

Per Anthropic's [API terms](https://www.anthropic.com/policies/terms):
- API inputs are **not used to train models**.
- Data may be retained for **up to 30 days** for trust and safety purposes.
- Review current terms periodically for changes.

### Recommendations

- **Do not send PII** (tenant names, SSNs, etc.) — proformas typically don't contain this, but verify.
- **Consider data classification** — if your firm classifies proforma data as "Restricted" or "Confidential," confirm that sending it to a cloud API is permitted under your data governance policy.
- **On-prem alternative** — if cloud API usage is prohibited, evaluate running a local model (reduced accuracy trade-off).

---

## Access Control

### Current State

- **CLI**: Anyone with repo access + API key can run.
- **Streamlit UI**: Password-protected (single shared password in secrets).

### Recommended Improvements (v1.0+)

| Control | Implementation |
|---------|---------------|
| Individual user accounts | Streamlit auth with SSO or per-user credentials |
| Role-based access | Analyst (run), Reviewer (read-only), Admin (config) |
| Audit logging | Log who ran what, when, with which inputs |
| API key per user | Separate Anthropic keys per analyst for cost tracking |

---

## File Security

### Files to Never Commit

Verify `.gitignore` includes:

```gitignore
.env
*.pptx          # Real memo files
*.xlsx          # Real proforma files
*.xlsm
*.mpp           # Real schedule files
secrets.toml
```

### Test Fixtures

- Use **anonymized/synthetic** data for test fixtures.
- Replace real property names, addresses, and financial figures.
- Store test fixtures in `tests/fixtures/` with a README explaining they are synthetic.

---

## Dependency Security

### Current Dependencies

All dependencies are from PyPI. Review periodically:

```bash
# Check for known vulnerabilities
pip audit

# Check for outdated packages
pip list --outdated
```

### Recommendations

- Pin exact versions in `requirements.txt` for reproducibility.
- Run `pip audit` in CI pipeline.
- Subscribe to security advisories for key packages (`anthropic`, `openpyxl`, `python-pptx`).

---

## Incident Response

If you discover a security issue:

1. **Do not post publicly** (GitHub issues, Slack channels).
2. Contact `@brandon` directly.
3. Describe the issue, affected scope, and any evidence.
4. We will triage, patch, rotate credentials as needed, and disclose once resolved.
