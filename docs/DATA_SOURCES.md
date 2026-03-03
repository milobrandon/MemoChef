# Market Data Sources

Last reviewed: 2026-03-03

Reference for external data sources planned for Memo Automator enrichment.

## Current state (v1.0)

No external market data ingestion in production pipeline yet.
Primary data source remains user-provided proforma and optional schedule files.

## Planned free sources (v1.5)

## FRED API

- URL: https://fred.stlouisfed.org/docs/api/fred/
- Example series: `SOFR`, `DGS10`, `CPIAUCSL`, `UNRATE`
- License: public domain

## Census / ACS API

- URL: https://api.census.gov/data.html
- Data: demographics, population, income, housing stats
- License: public domain

## Commercial sources (post-license review)

- CoStar
- Yardi Matrix
- RCA / MSCI

These require legal/license validation before integration, especially for any data sent to third-party AI APIs.

## Validation rules (target)

- Monetary values positive and reasonable
- Dates within freshness thresholds
- Occupancy + vacancy near 100%
- Cap rates in expected band (2%-15%)
- Source attribution required per datapoint

## Staleness thresholds (target)

- Rent comps: 30 days
- Sale comps: 90 days
- Submarket stats: 60 days
- Macro indicators: 7 days
- Demographics: 365 days
