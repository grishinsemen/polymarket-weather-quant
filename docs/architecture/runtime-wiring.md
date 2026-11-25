# Runtime Wiring & `app.state` Map

Navigation reference showing where shared runtime singletons live and how the 24/7 loop connects modules cleanly.

## `app.state` Registry (`backend/src/weather_quant/api/app.py`)
- `db`: `Database` (`aiosqlite` repository; path from `WEATHER_QUANT_DB_PATH`)
- `clock`: `Clock` (`SystemClock` in live runs, `VirtualClock` in tests/replays)
- `archive_dir` / `parquet_archive`: `ParquetArchiveStore` (`WEATHER_QUANT_ARCHIVE_DIR`, partitioned `orderbook/`, `metar/`, `ensemble/`)
- `radar_service`: `StationRadarService` (Gamma discovery + `Routine METAR` polling + `/ws/stream` fan-out)
- `clob_stream`: `ClobOrderBookStream` (`wss://ws-subscriptions-clob.polymarket.com/ws/market` + 10s `PING` + depth/stale tracking)
- `ensemble_fetcher`: `EnsembleFetcherPort` (`OpenMeteoEnsembleFetcher` fetching temp/wind/cloud/humidity)
- `probability_engine`: `EnsembleProbabilityEngine` (3-layer baseline `ensemble-3layer-v1`)
- `mos_provider`: `MosProbabilityProvider` (persisted `LightGBM` quantile MOS loaded via `load_mos_provider_for_version`)
- `strategy_store` / `paper_executor` / `strategy_engine`: `StrategyStore`, `PaperExecutor`, `StrategyEngine` (honors active DB `strategy_version`)
- `telegram_bot`: `TelegramRadarBot` (graceful no-op when `TELEGRAM_BOT_TOKEN` is unset)
- `autonomous_runtime`: `AutonomousRuntime` (`backend/src/weather_quant/api/runtime.py`)

## Router Auto-Discovery (`OPTIONAL_ROUTER_MODULES` in `api/app.py`)
`routes_stations`, `routes_clob`, `routes_probability`, `routes_ml`, `routes_strategy`, `routes_replay`, `routes_telegram`.

## `AutonomousRuntime.run_cycle()` Flow
1. `radar_service.sync_once()` → discovers eligible ICAO `METAR` `WeatherEvent`s, rejects `SPECI`, writes `Routine METAR`s to SQLite + `ParquetArchiveStore`, alerts `TelegramRadarBot` on sharp shifts.
2. `clob_stream.register_events(events)` → snapshots live `CLOB` Bid/Ask books into `ParquetArchiveStore`.
3. `load_mos_provider_for_version` / `EnsembleProbabilityEngine` → fetches & archives `EnsembleSnapshot`s, persists normalized `EventProbabilitySnapshot`s tagged with active `model_version`.
4. `strategy_engine.run_cycle(...)` → evaluates `Premarket Entry` & `Intraday Repricing`, simulates passive `Post-Only` fills via `PaperExecutor`, dispatches deduplicated `TelegramRadarBot` cards, broadcasts updated state over `/ws/stream`.

## Runtime Logs
`scripts/start-all.ps1` tees console output into `logs/backend.log` and `logs/frontend.log` (gitignored).
