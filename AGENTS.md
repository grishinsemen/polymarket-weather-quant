# AGENTS.md

Polymarket Daily Weather Quant Engine & Pro Terminal (`backend/` Python 3.11 + FastAPI + LightGBM, `frontend/` Next.js + React).

## Navigation & Standards Pointers

- **Domain vocabulary:** Read `GLOSSARY.md` before naming types, modules, API fields, or tests.
- **Architectural trade-offs:** Read `docs/adr/` before altering station eligibility, data sources, execution rules, or backend/frontend boundaries.
- **Runtime wiring & `app.state`:** Read `docs/architecture/runtime-wiring.md` before adding routers, background workers, or `app.state` services.
- **Code review standards:** Read `CODING_STANDARDS.md` during `/code-review` (mechanical import/SQL/env checks are enforced automatically by `backend/tests/test_09_architecture_guardrails.py`).
- **Live diagnostics:** Check `logs/backend.log` and `logs/frontend.log` when investigating local server errors.

## Verification & Environment Rules

- **Single verification command:** Run `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\check.ps1` (runs backend `pytest` + architecture guardrails, `tsc --noEmit`, and frontend `vitest`).
- **Windows PowerShell 5.1:** Chain commands with `;` under `$ErrorActionPreference = 'Stop'` and check `$LASTEXITCODE` (never use `&&`); pipe multi-line Python snippets via `@' ... '@ | .\.venv\Scripts\python.exe -`.
- **Subagent quota discipline:** Launch at most one heavy `self` implementer subagent at a time and commit green TDD slices frequently inside worktrees.

## Non-Negotiable Domain Guardrails

- **Temporal integrity:** Build ML features strictly from archived forecasts (`historical-forecast-api.open-meteo.com`) and observations with `timestamp < decision_time`, using chronological splits (`train_end < val_start < test_start`).
- **Exact station truth:** Query ensembles at ICAO runway coordinates/elevation; accept routine `METAR` observations (including `RMK` tenths) and discard `SPECI` reports.
- **Realistic paper execution:** Fill `Post-Only` maker orders only on opposing `CLOB` touch or trade-through; enforce `0.2x` Fractional Kelly, `$5` per `Bucket`, `$8` `City Exposure`, and `40%` `Intraday Cash Reserve`.
- **Sanctioned APIs only:** Use documented public endpoints (`gamma-api`, `clob`, `open-meteo`, `aviationweather.gov`, `mesonet.agron.iastate.edu`) with zero geo-restriction circumvention.

## Agent skills

### Issue tracker

Issues and specs live as local markdown files under `.scratch/<feature-slug>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the five default canonical triage labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout (`GLOSSARY.md` and `docs/adr/` at the repo root). See `docs/agents/domain.md`.
