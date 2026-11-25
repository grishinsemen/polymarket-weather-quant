# 03: Three-Layer Ensemble Probability Engine & Intraday METAR Repricing

**What to build:** Baseline 3-layer `Fair Probability` engine for every eligible `Station` combining ~120 physical `Ensemble Distribution` trajectories at airport runway coordinates, rolling 30–45 day `Station Bias` calibration, and real-time intraday `Routine METAR` Bayesian truncation/trend updating, persisted under an explicit `Model Version`.

**Blocked by:** 01: Live METAR Station Radar, Versioned Storage & Terminal Skeleton

**Status:** ready-for-agent

- [ ] Ensemble worker fetches ~120 hourly temperature trajectories (`ECMWF`, `GFS`, `ICON`) queried at each `Station`'s exact runway latitude, longitude, and elevation, archiving raw `Ensemble Distribution` snapshots with exact ingestion UTC timestamps.
- [ ] Layer 2 computes rolling 30–45 day `Station Bias` (mean offset and dispersion scaling) by comparing recent forecasts against verified `Routine METAR` daily extremes at each `Station`.
- [ ] Layer 3 updates `Fair Probability` immediately upon each new intraday `Routine METAR` observation: zeroes out mathematically impossible `Bucket` slices (below realized daily high for "Highest" markets, or above realized daily low for "Lowest" markets) and conditions remaining-day trajectories on observed morning/afternoon temperature drift.
- [ ] Computed `Bucket` `Fair Probability` distributions normalize cleanly (~1.00 across mutually exclusive outcomes of a `Weather Event`) and persist in operational storage tagged with an explicit baseline `Model Version`.
- [ ] Automated behavioral tests verify exact `Bucket` probability normalization, `Station Bias` shift correction, immediate zeroing of impossible `Bucket` outcomes after an intraday `Routine METAR` jump, non-reaction to disqualified `SPECI` reports, and `Model Version` tagging.
