---
name: fireflies-transcripts
description: Pull meeting transcripts from the Fireflies GraphQL API and apply qualitative insights to entitlement, due diligence, and program-team narratives in the IC memo. Use whenever /mnt/session/uploads/fireflies_config.json is present in the session uploads. Covers the GraphQL query shapes, what transcript content is in-scope vs out-of-scope (numerical data is always out), how open action items get labeled, and how to cite sources in the changelog.
---

# Fireflies Transcripts

**When to use:** `/mnt/session/uploads/fireflies_config.json` is present in the session uploads.

When a Fireflies API key is provided (mounted at `/mnt/session/uploads/fireflies_config.json`), you have access to meeting transcripts that contain due diligence updates, entitlement status, design decisions, and schedule discussions that the proforma alone cannot capture.

## How to use Fireflies

1. **Read the config file** to get the API key and lookback window:
   ```python
   import json
   config = json.loads(open("/mnt/session/uploads/fireflies_config.json").read())
   api_key = config["api_key"]
   lookback_days = config["lookback_days"]
   search_terms = config["search_terms"]  # e.g. ["Limestone", "Lexington", "VERVE"]
   ```

2. **Search for relevant meetings** using the Fireflies GraphQL API:
   ```python
   import httpx, time
   cutoff_ms = int((time.time() - lookback_days * 86400) * 1000)
   query = '{ transcripts(limit: 50) { id title date duration organizer_email summary { overview action_items } } }'
   resp = httpx.post(
       "https://api.fireflies.ai/graphql",
       headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
       json={"query": query},
       timeout=30,
   )
   transcripts = resp.json()["data"]["transcripts"]
   # Filter by date and search terms
   relevant = [t for t in transcripts
                if t["date"] >= cutoff_ms
                and any(term.lower() in t["title"].lower() for term in search_terms)]
   ```

3. **Fetch full transcripts** for the most relevant meetings:
   ```python
   query = 'query($id: String!) { transcript(id: $id) { title date sentences { text speaker_name start_time end_time } summary { overview action_items keywords } } }'
   resp = httpx.post(
       "https://api.fireflies.ai/graphql",
       headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
       json={"query": query, "variables": {"id": transcript_id}},
       timeout=30,
   )
   ```

4. **Extract actionable context** from transcripts:
   - **Entitlement status**: zoning approvals, variance requests, planning board dates.
   - **Due diligence findings**: environmental, survey, title, geotechnical updates.
   - **Design decisions**: unit mix changes, amenity decisions, material selections.
   - **Schedule milestones**: closing dates, construction start, CO, move-in.
   - **Open action items**: open items (pending approvals, outstanding outreach, unresolved design questions) ARE in scope for the Due Diligence narrative. Include them and label them as open/pending so readers know they are not yet resolved (e.g. "HOA outreach to the adjacent condo building is pending" or "A nine-story height allowance is under evaluation").

5. **Apply transcript insights to the memo**:
   - Update narrative sections about entitlement progress and schedule.
   - Update due diligence status paragraphs.
   - Cross-reference transcript discussions with proforma numbers.
   - Add context that explains changes (e.g. "unit count increased from 250 to 270 per design team decision to convert amenity space to additional 4BR units").
   - Log all transcript-sourced updates in the changelog with meeting date + title.

## Important rules for transcript data

- Transcript data supplements but does NOT override proforma numbers. If a meeting discussion mentions "$160M total budget" but the proforma says $157.7M, use the proforma number.
- Only use transcript data for narrative/qualitative updates, not financial metrics.
- Always cite the meeting title and date when using transcript information.
- If no relevant meetings are found within the lookback window, skip this step and note it in the changelog.
- **Transcript data may be used to update three sections of the memo only:**
  1. Entitlements status narratives,
  2. Due diligence status narratives, and
  3. Program / Underwriting narrative bullets — but for Program, ONLY for team and consultant selection updates (e.g. GC selection, architect selection, civil/survey/geotech firm selection, design team changes). In the Program section, ADD a new bullet if one doesn't already cover that topic rather than modifying existing program bullets.

  Do NOT use transcript data to update contracts, deposits, PSA terms, purchase price, schedule Gantt tables, unit counts, bed counts, budget numbers, returns, market data, or any other numeric/financial content. If a transcript mentions contract terms or deposit amounts, ignore that information — those sections are governed by the PSA and are updated manually by the deal team, not by this pipeline.
