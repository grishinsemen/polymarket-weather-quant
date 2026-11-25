# 05: Two-Phase Maker Strategy, $100 Risk Guardrails & 24/7 Realistic PaperExecutor

**What to build:** Autonomous 24/7 two-phase trading decision engine (`Premarket Entry` + `Intraday Repricing`), versioned `Strategy Version` risk manager calibrated for a `$100` bankroll, realistic passive `Post-Only` `PaperExecutor` behind a clean executor seam, and automatic `Closing Line Value (CLV)` tracking.

**Blocked by:** 02: Live CLOB Bucket Orderbook Stream & Partitioned Parquet Archive, 03: Three-Layer Ensemble Probability Engine & Intraday METAR Repricing

**Status:** ready-for-agent

- [ ] Strategy configurations register as immutable `Strategy Version` manifests encoding reference bankroll (`$100`), `0.2x` Fractional Kelly multiplier, `$5` max allocation per `Bucket`, `$8` max aggregate `City Exposure` per `Weather Event`, `40%` protected `Intraday Cash Reserve` during `Premarket Entry`, `Edge >= 5¢` entry threshold, minimum liquidity depth, stale-feed pause, and pre-settlement cutoff window.
- [ ] Phase 1 (`Premarket Entry`) places passive `Post-Only` `Maker Order` quotes at 0% fee whenever net `Edge >= 5¢` and risk/reserve caps permit; Phase 2 (`Intraday Repricing`) places early `Maker Order` profit-taking exits when the `CLOB` book converges toward `Fair Probability` or exits exposure early when fresh `Routine METAR` updates erode edge, freeing capital across Asian, European, and American timezones.
- [ ] `PaperExecutor` evaluates resting virtual `Maker Order` quotes against live `CLOB` orderbook streams (and controllable virtual clock ticks) and fills a passive quote only when the opposing market side touches or trades through our limit price (never assuming instant fills without opposing touch), accounting for `weather_fees` if a taker exit is ever triggered.
- [ ] Every generated signal, order, fill, open/closed position, realized/unrealized PnL update, structured decision reason, and post-entry `Closing Line Value (CLV)` measurement (at fixed post-fill horizons and pre-settlement close) is persisted with both `Model Version` and `Strategy Version` foreign keys, with operator pause/resume endpoints exposed via the backend API.
- [ ] Automated behavioral tests across the engine seam verify `$5` `Bucket` cap, `$8` correlated `City Exposure` cap across multiple `Bucket` slices of one city, `40%` `Intraday Cash Reserve` protection, strict touch/trade-through fill physics, early `Intraday Repricing` exit on `Routine METAR` shift, `CLV` calculation, and dual `(Model Version, Strategy Version)` attribution.
