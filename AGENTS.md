# AGENTS.md

Polymarket Daily Weather Quant Engine & Pro Terminal (`backend/` Python 3.11 + FastAPI + LightGBM/CatBoost, `frontend/` Next.js + React).

## Domain & Architecture Pointers

- Read `GLOSSARY.md` before naming types, modules, API fields, or tests so code speaks the exact domain vocabulary.
- Read `docs/adr/` before changing station eligibility, weather/ML data sources, order execution rules, or the backend/frontend boundary.

## Core Engineering Standards

- **Temporal integrity (zero data leakage):** Build ML features strictly from historical model forecasts (`historical-forecast-api.open-meteo.com`) and observations timestamped strictly prior to decision time. Split train/validation/backtest datasets chronologically (`train_end < val_start < test_start`).
- **Exact station truth:** Query weather forecasts at exact ICAO airport runway coordinates and elevation. Parse routine hourly `METAR` reports only (including tenths-of-degree `RMK` groups where present) and discard `SPECI` special reports.
- **Reactive hot path:** Perform all network I/O inside `asyncio` loops using async clients (`httpx`, `websockets`, `aiosqlite`). Offload CPU-heavy ML training and Parquet batch jobs outside the live orderbook loop.
- **Deep modules & high seams:** Hide external API schemas, fee math, and METAR parsing quirks behind narrow typed Pydantic/TypeScript interfaces. Test observable behavior at module boundaries using deterministic fixtures (`tdd`).
- **Realistic paper execution:** Fill simulated `Post-Only` maker orders only when the live Polymarket `CLOB` opposing side touches or trades through our limit price, and enforce bankroll risk caps (`0.2x` Fractional Kelly, $5/bucket, $8/city, 40% intraday cash reserve).
- **Sanctioned API usage only:** Use documented public endpoints (`gamma-api`, `clob`, `open-meteo`, `aviationweather.gov`, `mesonet.agron.iastate.edu`) without geo-restriction circumvention or proxy spoofing.

## Agent skills

### Issue tracker

Issues and specs live as local markdown files under `.scratch/<feature-slug>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the five default canonical triage labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout (`GLOSSARY.md` and `docs/adr/` at the repo root). See `docs/agents/domain.md`.
