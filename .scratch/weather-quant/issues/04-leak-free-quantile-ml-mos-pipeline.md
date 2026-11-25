# 04: Historical Calibration Bench & Leak-Free Quantile ML MOS for Top Stations

**What to build:** Out-of-band historical ingestion, chronological calibration benchmark, and Quantile GBDT (`LightGBM`/`CatBoost`) MOS training pipeline for the top 6–8 highest-volume `Station` markets using archived historical forecasts and historical `Routine METAR` observations with strict zero-leakage enforcement, registering versioned `Model Version` artifacts that plug seamlessly into live `Fair Probability` estimation.

**Blocked by:** 03: Three-Layer Ensemble Probability Engine & Intraday METAR Repricing

**Status:** ready-for-agent

- [ ] Historical dataset builder pulls archived model forecasts (`ECMWF`, `GFS`, `ICON` temperature, wind speed/direction, cloud cover, relative humidity, solar/diurnal/seasonal indicators) from the historical forecast endpoint and verified `Routine METAR` archives (excluding `SPECI`) for the top 6–8 `Station` markets, enforcing forecast availability strictly prior to decision time with zero post-factum reanalysis leakage.
- [ ] Chronological evaluation harness splits datasets strictly in time (training before validation before backtest) and computes Brier Score, quantile pinball loss, and reliability calibration curves comparing the 3-layer baseline against candidate Quantile GBDT MOS models.
- [ ] Trained Quantile GBDT MOS models execute outside the live event loop, register immutable `Model Version` records (storing training window, feature schema hash, and validation/test calibration diagnostics), and seamlessly supply calibrated `Bucket` `Fair Probability` distributions for covered `Station` markets while uncovered stations automatically use the 3-layer baseline `Model Version`.
- [ ] Automated tests assert that attempting to pass any feature or observation timestamped at or after decision time raises a temporal-integrity error, verify chronological split ordering, and verify end-to-end `Model Version` registration and live inference integration.
