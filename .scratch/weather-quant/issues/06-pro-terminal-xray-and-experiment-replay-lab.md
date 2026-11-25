# 06: Pro Next.js Trading Terminal (Edge Screener, 11-Bucket X-Ray & Parquet Replay Lab)

**What to build:** Complete TigerTrade-inspired dark `Next.js` pro trading terminal and deterministic `Experiment Replay` runner over accumulated `Parquet` and historical archives, enabling real-time visual inspection of every city's 11-`Bucket` orderbook vs model distribution alongside side-by-side `(Model Version, Strategy Version)` replay comparisons and one-click promotion.

**Blocked by:** 04: Historical Calibration Bench & Leak-Free Quantile ML MOS for Top Stations, 05: Two-Phase Maker Strategy, $100 Risk Guardrails & 24/7 Realistic PaperExecutor

**Status:** resolved

- [x] Deterministic `Experiment Replay` runner replays any candidate `(Model Version, Strategy Version)` pair over archived `Parquet` orderbook, `Ensemble Distribution`, and `Routine METAR` streams using a virtual clock (strictly enforcing ingestion time prior to decision time), running outside the live hot path and persisting reproducible experiment summaries (Brier Score, average `CLV`, fill rate, trade count, max drawdown, net PnL).
- [x] Terminal top portfolio bar streams live `$100` bankroll equity, realized/unrealized PnL, average `CLV`, active `City Exposure`, `Intraday Cash Reserve` gauge, pause/resume toggle, and global selectors to filter metrics by `Model Version` and `Strategy Version`.
- [x] Live multi-city `Edge` Screener ranks all active `Weather Event` markets by net `Edge`, spread, book depth, active `Model Version` (baseline vs Quantile ML MOS badge), time to daily extreme, and fresh `Routine METAR` indicator.
- [x] Clicking any city opens the 11-`Bucket` City X-Ray displaying side-by-side `Fair Probability` vs `CLOB` Bid/Ask depth bars, an intraday `Ensemble Distribution` fan chart overlaid with realized `Routine METAR` sensor readings, resting `Maker Order` / position state, and the structured decision audit log explaining why each order was placed, adjusted, or skipped.
- [x] Experiment Lab view lets the operator trigger a new `Experiment Replay`, inspect side-by-side comparison tables across `(Model Version, Strategy Version)` combinations, and promote a winning pair to active 24/7 paper execution while preserving all historical attribution.
- [x] Automated backend replay tests prove bit-for-bit determinism across repeated `Experiment Replay` runs, and frontend component/integration tests verify Screener sorting, 11-`Bucket` X-Ray rendering, version filtering, and Experiment Lab comparison workflows against deterministic state fixtures.
