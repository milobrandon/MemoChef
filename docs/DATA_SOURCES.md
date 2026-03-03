# Market Data Sources

> Reference for all external data sources used (or planned) by the Memo Automator.

---

## Current Sources (v1.0)

None — the tool currently relies entirely on user-provided proforma and schedule files. Market data integration is planned for v1.5.

---

## Planned Free Sources (v1.5)

### FRED API (Federal Reserve Economic Data)

| Field | Detail |
|-------|--------|
| **URL** | https://fred.stlouisfed.org/docs/api/fred/ |
| **Data** | SOFR, 10-year Treasury, CPI, unemployment, GDP |
| **Access** | Free REST API with API key |
| **Rate Limit** | 120 requests/minute |
| **License** | Public domain — free to use and cache |
| **Freshness** | Daily for rates, monthly for CPI/employment |

**Key Series IDs:**
- `SOFR` — Secured Overnight Financing Rate
- `DGS10` — 10-Year Treasury Constant Maturity
- `CPIAUCSL` — Consumer Price Index (All Urban)
- `UNRATE` — Unemployment Rate

### Census / ACS API

| Field | Detail |
|-------|--------|
| **URL** | https://api.census.gov/data.html |
| **Data** | Population, median income, housing units, demographics by MSA |
| **Access** | Free REST API with API key |
| **Rate Limit** | 500 requests/day |
| **License** | Public domain |
| **Freshness** | Annual (ACS 1-year and 5-year estimates) |

---

## Planned Commercial Sources (Post v1.5)

### CoStar

| Field | Detail |
|-------|--------|
| **Data** | Rent comps, sale comps, vacancy, absorption, pipeline |
| **Access** | API or manual CSV export |
| **Cost** | Enterprise license (per-seat) |
| **License** | No redistribution; per-seat usage only |
| **Consideration** | Verify that sending comp data to Claude API is permitted under license |

### Yardi Matrix

| Field | Detail |
|-------|--------|
| **Data** | Rent comps, occupancy, pipeline, market trends |
| **Access** | API or CSV export |
| **Cost** | Subscription (per-seat) |
| **License** | Restricted use; no redistribution |

### RCA / MSCI Real Capital Analytics

| Field | Detail |
|-------|--------|
| **Data** | Transaction comps, cap rates, price indices |
| **Access** | API |
| **Cost** | Enterprise license |
| **License** | Institutional license |

---

## Data Schema

All market data follows this conceptual schema (see ROADMAP.md Section 6 for full YAML):

```
market_data/
├── property         # Name, address, submarket, MSA, type
├── rent_comps[]     # Comp name, distance, unit type, rent, occupancy
├── sale_comps[]     # Comp name, sale date, price, cap rate, units
├── submarket_stats  # Avg rent, vacancy, absorption, pipeline, rent growth
└── macro            # SOFR, 10yr Treasury, CPI
```

---

## Validation Rules

| Rule | Threshold |
|------|-----------|
| Monetary values | Must be positive |
| Dates | Within last 12 months (configurable staleness) |
| Occupancy + vacancy | Must sum to ~100% |
| Cap rates | Between 2% and 15% |
| Rent values | $0 < rent < $10,000/unit/month |
| Source attribution | Required for every data point |

---

## Staleness Thresholds

| Data Type | Update Cadence | Max Age Before Warning |
|-----------|---------------|----------------------|
| Rent comps | Weekly | 30 days |
| Sale comps | Monthly | 90 days |
| Submarket stats | Monthly | 60 days |
| Macro indicators | Daily | 7 days |
| Demographics | Annually | 365 days |

---

## Licensing Notes

- **Free public data (FRED, Census):** Cache freely. Attribute source in outputs.
- **Commercial data (CoStar, Yardi, RCA):** Do NOT cache beyond current session unless license permits. Confirm with legal before integrating.
- **Internal deal data:** No restrictions. Version and audit all changes.
- **Recommendation:** Start with free sources only. Add commercial after license review.
