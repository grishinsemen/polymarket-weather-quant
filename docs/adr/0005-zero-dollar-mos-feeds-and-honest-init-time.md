# ADR-0005: $0 Operational MOS Feeds & Honest `init_time + p95 pub_time` Training

- **Status:** Accepted
- **Date:** 2026-11-27

## Context

Our v1 ML MOS training pipeline (`ml/dataset_builder.py`) relied on `historical-forecast-api.open-meteo.com`, which stitches the earliest hours (`f00–f06`) of consecutive NWP cycles rather than preserving true D-1/D-2 forecast lead errors. Additionally, live calibration lacked direct zero-cost operational MOS anchors (`DWD MOSMIX`, `NOAA NBM v5.0 / LAMP`, `NWS api.weather.gov`).

## Decision

1. Introduce a dedicated `weather_quant.feeds` package with sanctioned `$0` lightweight clients/parsers:
   - `DwdMosmixClient` (`opendata.dwd.de`, KMZ/KML XML, `TTT` + `E_TTT`)
   - `NoaaTextMosClient` (`nomads.ncep.noaa.gov` ASCII `NBS` / `LAV` bulletins)
   - `NwsPointForecastClient` (`api.weather.gov` GeoJSON point forecasts for US stations)
   - `PreviousRunsArchiveFetcher` (`previous-runs-api.open-meteo.com` D-1/D-2 runs shifted by empirical `p95` publication latency per NWP family)
2. Support assembling leak-free `MosDatasetSample` rows directly from recorded point-in-time `ParquetArchiveStore` ensembles (`build_mos_samples_from_parquet_ensembles`).
3. Extend `FEATURE_COLUMNS` with 24h atmospheric dynamics (`delta_obs_temp_24h`, `prior_diurnal_range_c`, `pressure_msl_tendency_24h`) and optional MOS anchors (`mosmix_extreme_c`, `nbm_extreme_c`, `nws_point_extreme_c`).
4. Defer heavy GRIB2 byte-range (`eccodes`/S3) ingestion to Phase E behind `EnsembleFetcherPort` after validating V2 calibration and execution improvements.

## Consequences

- Zero new third-party dependencies (`zipfile`, `xml.etree.ElementTree`, `httpx`).
- Eliminates stitched-analysis lookahead in D-1 MOS training and updates `FEATURE_SCHEMA_HASH` cleanly for new `ModelVersion` artifacts.
