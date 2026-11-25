# 02: Live CLOB Bucket Orderbook Stream & Partitioned Parquet Archive

**What to build:** Persistent asynchronous Polymarket `CLOB` WebSocket orderbook streaming across all `Bucket` outcomes of eligible `Weather Event` markets, fee-aware spread/depth tracking, and continuous timestamped columnar `Parquet` archiving of orderbook and `Routine METAR` streams.

**Blocked by:** 01: Live METAR Station Radar, Versioned Storage & Terminal Skeleton

**Status:** resolved

- [x] Asynchronous `CLOB` WebSocket client subscribes to all token IDs for active `Weather Event` `Bucket` contracts, sends 10-second `PING` heartbeats, reconnects automatically with snapshot resynchronization on disconnect, and maintains live best Bid, best Ask, midpoint, spread, and top-5 depth levels in memory.
- [x] Fee calculator accurately computes Polymarket `weather_fees` for taker execution vs 0% fee for passive `Post-Only` `Maker Order` quotes across any `Bucket` price level, and flags stale feeds whenever orderbook updates exceed freshness thresholds.
- [x] Columnar archive writer continuously appends live `CLOB` orderbook snapshots and ingested `Routine METAR` observations into partitioned `Parquet` datasets stamped with exact ingestion UTC timestamps for future leak-free replay.
- [x] Backend REST/WebSocket contract exposes live `Bucket` Bid/Ask prices, top-of-book liquidity, implied probability sums across each negative-risk `Weather Event`, and archive health metrics.
- [x] Automated behavioral tests verify WebSocket book reconstruction, 10-second `PING` keepalive/reconnect handling, stale-feed detection, `weather_fees` vs 0% `Maker Order` math, and `Parquet` round-trip persistence with strict ingestion timestamps.
