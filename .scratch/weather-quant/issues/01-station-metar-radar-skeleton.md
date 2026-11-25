# 01: Live METAR Station Radar, Versioned Storage & Terminal Skeleton

**What to build:** End-to-end runnable walking skeleton linking the asynchronous backend, versioned operational relational database, live Polymarket `Weather Event` discovery, strict ICAO `Station` and `Routine METAR` observation ingestion, and dark pro web terminal shell so the operator immediately sees live eligible temperature markets and verified airport temperatures on screen.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Backend service and standalone dark `Next.js` web terminal start cleanly and connect over REST and WebSocket health/state feeds.
- [x] Operational database initializes immutable `Model Version` and `Strategy Version` registry records so every downstream prediction, signal, order, fill, and experiment references explicit version identifiers from day one.
- [x] Discovery worker fetches active Polymarket daily temperature `Weather Event` listings, accepts only contracts settling strictly on ICAO airport `Routine METAR` reports, extracts the 4-letter `Station` code, runway coordinates and elevation, local settlement date/timezone, "Highest" vs "Lowest" polarity, `Bucket` interval bounds, `Remark Tenths` precision rule, and `weather_fees` rate, and quarantines non-`METAR` observatories or ambiguous rule descriptions with explicit reason codes.
- [x] Aviation sensor worker fetches live airport reports, accepts scheduled `Routine METAR` lines, discards every unscheduled `SPECI` report, and accurately decodes both whole-degree temperatures and `RMK` `Remark Tenths` groups (including negative tenths).
- [x] Web terminal renders the live table of eligible `Station` markets, local settlement clock countdowns, `Bucket` lists, quarantined market count, and latest verified `Routine METAR` reading and timestamp per airport.
- [x] Automated behavioral tests across the high engine/API seam verify `METAR` vs `SPECI` filtering, positive/negative `Remark Tenths` decoding, non-`METAR` quarantine (such as local observatory markets), `Bucket` boundary parsing, and version registry initialization using deterministic fixtures.
