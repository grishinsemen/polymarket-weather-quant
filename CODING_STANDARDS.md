# CODING_STANDARDS.md

Judgement-call standards for `/code-review` across `backend/` and `frontend/` (mechanical syntax/import/env checks live in `backend/tests/test_09_architecture_guardrails.py`).

## 1. End-to-End Runtime Wiring
- Wire every new feed, probability provider, strategy stage, or alert sink into `AutonomousRuntime` (`backend/src/weather_quant/api/runtime.py`) and `app.py` lifespan so 24/7 operation exercises the same code path as manual REST routes.

## 2. Unit & Version Invariance at Seams
- Convert every temperature leaving Celsius domain storage into UI/chart DTOs via `to_event_unit(..., event.unit)`.
- Scope every portfolio exposure, cash, PnL, and `Closing Line Value (CLV)` query strictly by `(model_version, strategy_version)`; never divide multi-city total exposure by the single-city `$8` `City Exposure` cap.

## 3. Deep Modules & Thin Route Files
- Keep `backend/src/weather_quant/api/routes_*.py` files thin HTTP/WebSocket transport adapters (target `<350` lines for new modules).
- Place probability math, feature extraction, replay orchestration, and fixture builders inside domain/service modules behind narrow typed interfaces.

## 4. Reactive Non-Blocking Hot Path
- Perform all external HTTP and WebSocket I/O with async clients (`httpx`, `websockets`, `aiosqlite`).
- Offload disk writes, `Parquet` batch reads/writes, `LightGBM` training, and `Experiment Replay` simulations outside the live `asyncio` event loop via `asyncio.to_thread`.
