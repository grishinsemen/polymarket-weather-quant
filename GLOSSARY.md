# Polymarket Weather Quant

Domain vocabulary for autonomous probability modeling, intraday sensor tracking, and maker-first trading on Polymarket daily city temperature markets.

## Language

### Weather & Observations

**Station**:
The specific ICAO airport meteorological facility (such as `LFPB` Paris-Le Bourget or `RKSI` Seoul Incheon) whose official reports settle a city's temperature market.
_Avoid_: City weather, downtown sensor, generic location

**Routine METAR**:
A scheduled hourly aviation weather observation beginning with `METAR` that qualifies for Polymarket temperature settlement.
_Avoid_: Weather update, raw report, SPECI

**SPECI**:
An unscheduled special aviation weather report issued between routine hours; excluded from qualifying settlement temperatures under Polymarket METAR rules.
_Avoid_: Extra METAR, interim observation

**Remark Tenths**:
The optional high-precision temperature group in a Routine METAR `RMK` section reporting temperature to `0.1°C` rather than whole degrees.
_Avoid_: Rounded temperature, raw decimal

**Ensemble Distribution**:
The combined set of ~120 physical weather model trajectories (`ECMWF`, `GFS`, `ICON`) at a Station's coordinates representing forecast uncertainty across hours of the day.
_Avoid_: Single point forecast, weather app prediction

**Station Bias**:
The systematic historical offset and spread error between Ensemble Distribution forecasts and observed Routine METAR temperatures at a specific Station (Model Output Statistics / MOS).
_Avoid_: Random noise, global error

### Prediction Market & Orderbook

**Weather Event**:
A daily Polymarket negative-risk (`negRisk`) market group for a single Station and local date ("Highest" or "Lowest" temperature), partitioned into mutually exclusive Bucket contracts.
_Avoid_: Bet, game, match

**Bucket**:
One mutually exclusive temperature outcome slice within a Weather Event (for example `6°C` or `2°C or below`) paying `$1.00` if the Station's qualifying daily extreme lands inside it and `$0.00` otherwise.
_Avoid_: Strike, line, option leg

**Fair Probability**:
Our calibrated model estimate (`0.00` to `1.00`) that a Bucket will resolve `Yes`, combining the Ensemble Distribution, Station Bias / ML MOS quantiles, and live intraday Routine METAR trajectory.
_Avoid_: Market odds, implied probability

**Edge**:
The net expected advantage per share (`Fair Probability` minus execution price and applicable fees) required before placing an order (`>= 5¢` minimum).
_Avoid_: Profit, spread

**Maker Order**:
A `Post-Only` limit order resting passively in the Polymarket `CLOB` book at `0%` maker fee rather than crossing the spread as a fee-paying taker.
_Avoid_: Market order, instant buy

**Closing Line Value (CLV)**:
The favorable price movement of a Bucket's midpoint from our entry price to subsequent market convergence or pre-settlement close, measuring true signal quality independent of single-day variance.
_Avoid_: Win rate, luck

### Trading Lifecycle & Risk

**Premarket Entry**:
Phase 1 of the trading cycle: placing passive Maker Orders hours ahead of a Station's daily temperature peak/trough when Edge exceeds threshold.
_Avoid_: Blind hold, pre-bet

**Intraday Repricing**:
Phase 2 of the trading cycle: updating Fair Probability as each new Routine METAR arrives, taking profit early via Maker Orders on convergence or cutting exposure when actual sensor readings diverge from forecast.
_Avoid_: Hold-to-expiry, set-and-forget

**City Exposure**:
The aggregate capital at risk across all open Buckets and resting Maker Orders within a single Weather Event, capped (`$8` on a `$100` bankroll) to prevent correlated weather-front losses.
_Avoid_: Per-bet stake, isolated position

**Intraday Cash Reserve**:
The protected share of bankroll (`40%`) kept unallocated from Premarket Entry so capital remains available for high-conviction Intraday Repricing signals across timezones.
_Avoid_: Idle balance
