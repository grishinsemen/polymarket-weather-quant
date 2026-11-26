"""Behavioral tests for leak-free Quantile ML MOS pipeline and calibration bench."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from weather_quant.api.app import create_app
from weather_quant.discovery.stations import get_station
from weather_quant.domain.clock import TemporalIntegrityError, VirtualClock
from weather_quant.domain.models import (
    Bucket,
    EnsembleSnapshot,
    MetarObservation,
    Polarity,
    WeatherEvent,
)
from weather_quant.ml.dataset_builder import (
    FEATURE_SCHEMA_HASH,
    HISTORICAL_FORECAST_URL,
    IEM_ASOS_METAR_URL,
    TOP_LIQUIDITY_STATIONS,
    HistoricalArchiveFetcher,
    HistoricalForecastPoint,
    MosDatasetSample,
    build_mos_sample,
    build_mos_samples_from_history,
)
from weather_quant.ml.quantile_mos import (
    DEFAULT_QUANTILES,
    MosProbabilityProvider,
    QuantileMosModel,
    split_chronologically,
    train_and_register_mos,
)
from weather_quant.probability.engine import DEFAULT_MODEL_VERSION
from weather_quant.storage.database import Database


def _build_historical_open_meteo_payload(
    start_utc: datetime,
    hours: int,
    models: tuple[str, ...] = ("ecmwf_ifs025", "gfs_seamless", "icon_seamless"),
    base_temp_c: float = 12.0,
) -> list[dict[str, Any]]:
    timestamps = [(start_utc + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(hours)]
    blocks: list[dict[str, Any]] = []
    for idx, model_name in enumerate(models):
        temps = [
            round(base_temp_c + idx * 0.4 + 4.0 * max(0.0, 1.0 - abs((h % 24) - 14) / 8.0), 2)
            for h in range(hours)
        ]
        blocks.append(
            {
                "model": model_name,
                "hourly": {
                    "time": timestamps,
                    "temperature_2m": temps,
                    "wind_speed_10m": [10.0 + idx for _ in range(hours)],
                    "wind_direction_10m": [210.0 for _ in range(hours)],
                    "cloud_cover": [35.0 for _ in range(hours)],
                    "relative_humidity_2m": [65.0 for _ in range(hours)],
                },
            }
        )
    return blocks


@pytest.mark.asyncio
async def test_historical_fetcher_queries_sanctioned_endpoints_and_filters_speci() -> None:
    expected = ("LFPB", "RKSI", "EDDM", "EGLL", "KMIA", "KATL", "RKPK", "VILK")
    assert set(TOP_LIQUIDITY_STATIONS) == set(expected)
    assert len(TOP_LIQUIDITY_STATIONS) == 8

    station = get_station("LFPB")
    assert station is not None
    clock = VirtualClock(datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc))
    seen_urls: list[httpx.Request] = []

    forecast_payload = _build_historical_open_meteo_payload(
        start_utc=datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc),
        hours=48,
    )
    iem_csv = "\n".join(
        [
            "station,valid,metar",
            "LFPB,2026-11-01 12:00,LFPB 011200Z 24010KT 9999 SCT030 14/08 Q1016 RMK T01430080",
            "LFPB,2026-11-01 12:30,SPECI LFPB 011230Z 24018KT 4000 BKN015 24/08 Q1015 RMK T02400080",
            "LFPB,2026-11-01 13:00,METAR LFPB 011300Z 24012KT 9999 FEW040 15/08 Q1016 RMK T01540080",
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(request)
        url_str = str(request.url)
        if url_str.startswith(HISTORICAL_FORECAST_URL):
            return httpx.Response(200, json=forecast_payload)
        if url_str.startswith(IEM_ASOS_METAR_URL):
            return httpx.Response(200, text=iem_csv)
        raise AssertionError(f"Unexpected URL: {url_str}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        fetcher = HistoricalArchiveFetcher(client=client, clock=clock)
        forecasts = await fetcher.fetch_historical_forecasts(
            station=station,
            start_date=date(2026, 11, 1),
            end_date=date(2026, 11, 2),
        )
        metars = await fetcher.fetch_historical_routine_metars(
            station=station,
            start_date=date(2026, 11, 1),
            end_date=date(2026, 11, 2),
        )

    assert len(seen_urls) == 2
    fc_req, metar_req = seen_urls
    assert str(fc_req.url).startswith(HISTORICAL_FORECAST_URL)
    assert float(fc_req.url.params["latitude"]) == pytest.approx(station.lat)
    assert float(fc_req.url.params["longitude"]) == pytest.approx(station.lon)
    assert float(fc_req.url.params["elevation"]) == pytest.approx(station.elevation_m)
    for var in ("temperature_2m", "wind_speed_10m", "wind_direction_10m", "cloud_cover", "relative_humidity_2m"):
        assert var in fc_req.url.params["hourly"]

    assert str(metar_req.url).startswith(IEM_ASOS_METAR_URL)
    assert metar_req.url.params["station"] == "LFPB"
    assert metar_req.url.params["report_type"] == "3"

    assert len(forecasts) == 48 * 3
    # SPECI must be discarded, bare LFPB and METAR LFPB lines must both be accepted with RMK tenths
    assert len(metars) == 2
    assert all(m.is_routine and not m.raw_report.upper().startswith("SPECI") for m in metars)
    assert [m.temp_c for m in metars] == pytest.approx([14.3, 15.4])


def test_dataset_builder_raises_temporal_integrity_error_on_future_forecast_or_obs() -> None:
    station = get_station("LFPB")
    assert station is not None
    target_date = date(2026, 11, 10)
    decision_time = datetime(2026, 11, 9, 18, 0, tzinfo=timezone.utc)

    valid_forecast = HistoricalForecastPoint(
        station_icao="LFPB",
        model="ecmwf_ifs025",
        valid_at_utc=datetime(2026, 11, 10, 14, 0, tzinfo=timezone.utc),
        available_at_utc=datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc),
        temperature_2m_c=14.0,
        wind_speed_10m=10.0,
        wind_direction_10m=220.0,
        cloud_cover=30.0,
        relative_humidity_2m=60.0,
    )
    leaked_forecast = HistoricalForecastPoint(
        station_icao="LFPB",
        model="gfs_seamless",
        valid_at_utc=datetime(2026, 11, 10, 14, 0, tzinfo=timezone.utc),
        available_at_utc=decision_time,  # Equal to decision_time -> must raise!
        temperature_2m_c=15.0,
    )

    with pytest.raises(TemporalIntegrityError):
        build_mos_sample(
            station=station,
            local_date=target_date,
            polarity=Polarity.HIGHEST,
            decision_time_utc=decision_time,
            forecasts=[valid_forecast, leaked_forecast],
            prior_observations=[],
        )

    leaked_obs = MetarObservation(
        station_icao="LFPB",
        raw_report="METAR LFPB 091900Z 24010KT CAVOK 12/06 Q1015",
        observed_at_utc=decision_time + timedelta(hours=1),  # After decision_time -> must raise!
        ingested_at_utc=decision_time + timedelta(hours=1),
        temp_c=12.0,
        whole_temp_c=12,
        is_routine=True,
    )
    with pytest.raises(TemporalIntegrityError):
        build_mos_sample(
            station=station,
            local_date=target_date,
            polarity=Polarity.HIGHEST,
            decision_time_utc=decision_time,
            forecasts=[valid_forecast],
            prior_observations=[leaked_obs],
        )



def _make_synthetic_mos_history(
    station_icao: str = "LFPB",
    n_days: int = 60,
    start_date: date = date(2026, 9, 1),
) -> list[MosDatasetSample]:
    """Generate deterministic non-linear Station Bias history where ML MOS beats naive baseline."""
    station = get_station(station_icao)
    assert station is not None
    forecasts: list[HistoricalForecastPoint] = []
    metars: list[MetarObservation] = []

    for d in range(n_days):
        day = start_date + timedelta(days=d)
        # Systematic non-linear microclimate bias: sunny & light wind -> +2.2C warmer than NWP
        wind = 4.0 if d % 2 == 0 else 16.0
        cloud = 15.0 if d % 2 == 0 else 85.0
        true_bias = 2.2 if (wind < 10.0 and cloud < 50.0) else -0.6
        base_temp = 12.0 + (d % 7) * 0.5

        for h in range(24):
            valid_utc = datetime(day.year, day.month, day.day, h, 0, tzinfo=timezone.utc)
            avail_utc = datetime(day.year, day.month, day.day, 0, 0, tzinfo=timezone.utc) - timedelta(hours=12)
            diurnal = 3.0 * max(0.0, 1.0 - abs(h - 14) / 7.0)
            for m_idx, m_name in enumerate(("ecmwf_ifs025", "gfs_seamless", "icon_seamless")):
                forecasts.append(
                    HistoricalForecastPoint(
                        station_icao=station_icao,
                        model=m_name,
                        valid_at_utc=valid_utc,
                        available_at_utc=avail_utc,
                        temperature_2m_c=round(base_temp + diurnal + (m_idx - 1) * 0.2, 2),
                        wind_speed_10m=wind,
                        wind_direction_10m=200.0,
                        cloud_cover=cloud,
                        relative_humidity_2m=60.0,
                    )
                )

        # Realized daily peak at 14:00 UTC
        peak_obs_utc = datetime(day.year, day.month, day.day, 14, 0, tzinfo=timezone.utc)
        realized_peak = round(base_temp + 3.0 + true_bias, 1)
        metars.append(
            MetarObservation(
                station_icao=station_icao,
                raw_report=f"METAR {station_icao} {peak_obs_utc:%d%H%M}Z 20006KT CAVOK {int(round(realized_peak)):02d}/06 Q1016 RMK T0{int(round(realized_peak * 10)):03d}0060",
                observed_at_utc=peak_obs_utc,
                ingested_at_utc=peak_obs_utc,
                temp_c=realized_peak,
                whole_temp_c=int(round(realized_peak)),
                remark_tenths_c=realized_peak,
                has_remark_tenths=True,
                is_routine=True,
            )
        )

    return build_mos_samples_from_history(
        station=station,
        forecasts=forecasts,
        routine_metars=metars,
        polarity=Polarity.HIGHEST,
    )


def test_chronological_split_enforces_strict_ordering_and_no_overlap() -> None:
    samples = _make_synthetic_mos_history(n_days=30)
    t_train_end = samples[15].decision_time_utc
    t_val_start = samples[16].decision_time_utc
    t_test_start = samples[23].decision_time_utc

    train, val, test = split_chronologically(
        samples,
        train_end=t_train_end,
        val_start=t_val_start,
        test_start=t_test_start,
    )
    assert len(train) == 15
    assert len(val) == 7
    assert len(test) == 7
    assert max(s.decision_time_utc for s in train) < t_val_start
    assert min(s.decision_time_utc for s in val) >= t_val_start
    assert max(s.decision_time_utc for s in val) < t_test_start
    assert min(s.decision_time_utc for s in test) >= t_test_start

    # Overlap (train_end >= val_start) must raise TemporalIntegrityError
    with pytest.raises(TemporalIntegrityError):
        split_chronologically(
            samples,
            train_end=t_val_start,
            val_start=t_val_start,
            test_start=t_test_start,
        )

    # Overlap (val_start >= test_start) must raise TemporalIntegrityError
    with pytest.raises(TemporalIntegrityError):
        split_chronologically(
            samples,
            train_end=t_train_end,
            val_start=t_test_start,
            test_start=t_test_start,
        )

    # Unsorted samples must raise TemporalIntegrityError
    unsorted = [samples[5], samples[2], *samples[6:]]
    with pytest.raises(TemporalIntegrityError):
        split_chronologically(
            unsorted,
            train_end=t_train_end,
            val_start=t_val_start,
            test_start=t_test_start,
        )


@pytest.mark.asyncio
async def test_quantile_mos_training_benchmarks_registration_and_unified_inference() -> None:
    samples = _make_synthetic_mos_history(station_icao="LFPB", n_days=60)
    t_train_end = samples[35].decision_time_utc
    t_val_start = samples[36].decision_time_utc
    t_test_start = samples[48].decision_time_utc

    db = Database(":memory:")
    await db.initialize()
    try:
        mos_model, version_record, benchmark = await train_and_register_mos(
            db=db,
            samples=samples,
            model_version="mos-lgbm-v1-test",
            train_end_utc=t_train_end,
            val_start_utc=t_val_start,
            test_start_utc=t_test_start,
            quantiles=DEFAULT_QUANTILES,
            lgbm_params={"n_estimators": 20, "max_depth": 3, "num_leaves": 7, "learning_rate": 0.15, "min_child_samples": 3, "verbosity": -1},
            activate=True,
        )

        assert version_record.model_version == "mos-lgbm-v1-test"
        assert version_record.feature_schema_hash == FEATURE_SCHEMA_HASH
        assert version_record.train_start_utc is not None
        assert version_record.train_end_utc is not None
        assert version_record.train_end_utc < t_val_start
        assert version_record.brier_score is not None
        assert version_record.pinball_loss is not None
        assert "reliability_curve" in version_record.config_json
        assert version_record.is_active is True

        # Non-linear synthetic bias should let MOS outperform naive baseline on test Brier & pinball
        assert benchmark.test_mos.brier_score < benchmark.test_baseline.brier_score
        assert benchmark.test_mos.pinball_loss < benchmark.test_baseline.pinball_loss
        assert len(benchmark.test_mos.reliability_curve) > 0

        # Duplicate version registration must fail (immutable Model Version)
        with pytest.raises(ValueError):
            await train_and_register_mos(
                db=db,
                samples=samples,
                model_version="mos-lgbm-v1-test",
                train_end_utc=t_train_end,
                val_start_utc=t_val_start,
                test_start_utc=t_test_start,
                lgbm_params={"n_estimators": 10, "min_child_samples": 3, "verbosity": -1},
            )

        # Unified provider: covered top station LFPB uses MOS; uncovered station NZWN falls back to ensemble-3layer-v1
        provider = MosProbabilityProvider(mos_model=mos_model)
        decision_time = datetime(2026, 11, 20, 10, 0, tzinfo=timezone.utc)
        clock = VirtualClock(decision_time)
        lfpb = get_station("LFPB")
        nzwn = get_station("NZWN")
        assert lfpb is not None and nzwn is not None

        def _make_event_and_ensemble(station_obj: Any, eid: str) -> tuple[WeatherEvent, EnsembleSnapshot]:
            d = date(2026, 11, 20)
            buckets = [
                Bucket(bucket_id="b14", market_id="m14", token_id="t14", label="14°C or below", lower_bound=None, upper_bound=14.0, unit="C", outcome_index=0),
                Bucket(bucket_id="b15", market_id="m15", token_id="t15", label="15°C", lower_bound=15.0, upper_bound=15.0, unit="C", outcome_index=1),
                Bucket(bucket_id="b16", market_id="m16", token_id="t16", label="16°C", lower_bound=16.0, upper_bound=16.0, unit="C", outcome_index=2),
                Bucket(bucket_id="b17", market_id="m17", token_id="t17", label="17°C or higher", lower_bound=17.0, upper_bound=None, unit="C", outcome_index=3),
            ]
            ev = WeatherEvent(
                event_id=eid,
                slug=eid,
                title=f"Highest temp {station_obj.city}",
                url=f"https://polymarket.com/event/{eid}",
                station=station_obj,
                local_date=d,
                polarity=Polarity.HIGHEST,
                remark_tenths_enabled=True,
                unit="C",
                buckets=buckets,
                discovered_at_utc=decision_time - timedelta(hours=12),
            )
            ts = [datetime(2026, 11, 19, 10, 0, tzinfo=timezone.utc) + timedelta(hours=h) for h in range(48)]
            ens = EnsembleSnapshot(
                event_id=eid,
                station_icao=station_obj.icao,
                fetched_at_utc=decision_time - timedelta(hours=2),
                ingested_at_utc=decision_time - timedelta(hours=2),
                models=["ecmwf_ifs025", "gfs_seamless", "icon_seamless"],
                member_count=60,
                hourly_timestamps_utc=ts,
                member_temps_c=[[14.8 + (m - 30) * 0.02 for _ in ts] for m in range(60)],
                metadata={"wind_speed_10m": [4.0], "cloud_cover": [15.0], "relative_humidity_2m": [60.0]},
            )
            return ev, ens

        ev_lfpb, ens_lfpb = _make_event_and_ensemble(lfpb, "evt-lfpb-live")
        intraday_metar = MetarObservation(
            station_icao="LFPB",
            raw_report="METAR LFPB 200900Z 20005KT CAVOK 15/06 Q1016 RMK T01520060",
            observed_at_utc=datetime(2026, 11, 20, 9, 0, tzinfo=timezone.utc),
            ingested_at_utc=datetime(2026, 11, 20, 9, 5, tzinfo=timezone.utc),
            temp_c=15.2,
            whole_temp_c=15,
            remark_tenths_c=15.2,
            has_remark_tenths=True,
            is_routine=True,
        )
        snap_lfpb = provider.compute_probabilities(
            station=lfpb,
            event=ev_lfpb,
            ensemble=ens_lfpb,
            intraday_metars=[intraday_metar],
            clock=clock,
        )
        assert snap_lfpb.model_version == "mos-lgbm-v1-test"
        assert snap_lfpb.probability_sum == pytest.approx(1.0, abs=1e-9)
        by_lfpb = {b.bucket_id: b for b in snap_lfpb.buckets}
        # 14°C or below is impossible after realized 15.2°C Routine METAR
        assert by_lfpb["b14"].is_impossible is True
        assert by_lfpb["b14"].fair_probability == 0.0

        ev_nzwn, ens_nzwn = _make_event_and_ensemble(nzwn, "evt-nzwn-live")
        snap_nzwn = provider.compute_probabilities(
            station=nzwn,
            event=ev_nzwn,
            ensemble=ens_nzwn,
            clock=clock,
        )
        assert snap_nzwn.model_version == DEFAULT_MODEL_VERSION
        assert snap_nzwn.probability_sum == pytest.approx(1.0, abs=1e-9)
    finally:
        await db.close()



def test_fastapi_ml_routes_train_versions_covered_stations_and_benchmarks() -> None:
    decision_time = datetime(2026, 11, 25, 18, 0, tzinfo=timezone.utc)
    clock = VirtualClock(decision_time)
    samples = _make_synthetic_mos_history(station_icao="LFPB", n_days=50, start_date=date(2026, 9, 15))
    t_train_end = samples[29].decision_time_utc
    t_val_start = samples[30].decision_time_utc
    t_test_start = samples[40].decision_time_utc

    app = create_app(db_path=":memory:", clock=clock, enable_background_poll=False)
    with TestClient(app) as client:
        covered_resp = client.get("/api/ml/covered-stations")
        assert covered_resp.status_code == 200
        covered_data = covered_resp.json()
        assert covered_data["icaos"] == list(TOP_LIQUIDITY_STATIONS)
        assert len(covered_data["stations"]) == 8
        assert covered_data["mos_active"] is False

        versions_before = client.get("/api/ml/versions")
        assert versions_before.status_code == 200
        assert versions_before.json()["active_model_version"] == DEFAULT_MODEL_VERSION

        train_resp = client.post(
            "/api/ml/train",
            json={
                "model_version": "mos-lgbm-api-v1",
                "samples": [s.model_dump(mode="json") for s in samples],
                "train_end_utc": t_train_end.isoformat(),
                "val_start_utc": t_val_start.isoformat(),
                "test_start_utc": t_test_start.isoformat(),
                "lgbm_params": {"n_estimators": 15, "max_depth": 3, "num_leaves": 7, "min_child_samples": 3, "verbose": -1},
                "activate": True,
            },
        )
        assert train_resp.status_code == 200
        train_data = train_resp.json()
        assert train_data["model_version"]["model_version"] == "mos-lgbm-api-v1"
        assert train_data["model_version"]["is_active"] is True
        assert train_data["benchmark"]["test_mos"]["brier_score"] >= 0.0
        assert train_data["benchmark"]["test_mos"]["pinball_loss"] >= 0.0

        # Duplicate version should be rejected
        dup_resp = client.post(
            "/api/ml/train",
            json={
                "model_version": "mos-lgbm-api-v1",
                "samples": [s.model_dump(mode="json") for s in samples],
                "train_end_utc": t_train_end.isoformat(),
                "val_start_utc": t_val_start.isoformat(),
                "test_start_utc": t_test_start.isoformat(),
            },
        )
        assert dup_resp.status_code == 409

        # Invalid chronological split should be rejected
        bad_split_resp = client.post(
            "/api/ml/train",
            json={
                "model_version": "mos-lgbm-bad-split",
                "samples": [s.model_dump(mode="json") for s in samples],
                "train_end_utc": t_val_start.isoformat(),
                "val_start_utc": t_val_start.isoformat(),
                "test_start_utc": t_test_start.isoformat(),
            },
        )
        assert bad_split_resp.status_code == 400

        versions_after = client.get("/api/ml/versions")
        assert versions_after.status_code == 200
        v_json = versions_after.json()
        assert v_json["active_model_version"] == "mos-lgbm-api-v1"
        assert any(v["model_version"] == "mos-lgbm-api-v1" for v in v_json["versions"])

        benchmarks_resp = client.get("/api/ml/benchmarks")
        assert benchmarks_resp.status_code == 200
        b_json = benchmarks_resp.json()
        assert b_json["count"] == 1
        assert b_json["benchmarks"][0]["model_version"] == "mos-lgbm-api-v1"

        covered_after = client.get("/api/ml/covered-stations")
        assert covered_after.status_code == 200
        assert covered_after.json()["mos_active"] is True
        assert covered_after.json()["active_model_version"] == "mos-lgbm-api-v1"


def test_fastapi_ml_train_from_historical_fetcher_seam() -> None:
    decision_time = datetime(2026, 11, 25, 18, 0, tzinfo=timezone.utc)
    clock = VirtualClock(decision_time)
    station = get_station("LFPB")
    assert station is not None

    forecasts: list[HistoricalForecastPoint] = []
    metars: list[MetarObservation] = []
    start_d = date(2026, 10, 1)
    for d in range(30):
        day = start_d + timedelta(days=d)
        for h in range(24):
            valid_utc = datetime(day.year, day.month, day.day, h, 0, tzinfo=timezone.utc)
            avail_utc = datetime(day.year, day.month, day.day, 0, 0, tzinfo=timezone.utc) - timedelta(hours=12)
            forecasts.append(
                HistoricalForecastPoint(
                    station_icao="LFPB",
                    model="ecmwf_ifs025",
                    valid_at_utc=valid_utc,
                    available_at_utc=avail_utc,
                    temperature_2m_c=14.0 + (2.0 if h == 14 else 0.0),
                    wind_speed_10m=6.0,
                    cloud_cover=25.0,
                    relative_humidity_2m=60.0,
                )
            )
        obs_utc = datetime(day.year, day.month, day.day, 14, 0, tzinfo=timezone.utc)
        metars.append(
            MetarObservation(
                station_icao="LFPB",
                raw_report=f"METAR LFPB {obs_utc:%d%H%M}Z 20006KT CAVOK 17/06 Q1016 RMK T01700060",
                observed_at_utc=obs_utc,
                ingested_at_utc=obs_utc,
                temp_c=17.0,
                whole_temp_c=17,
                remark_tenths_c=17.0,
                has_remark_tenths=True,
                is_routine=True,
            )
        )

    class _FakeHistoricalFetcher:
        async def fetch_historical_forecasts(
            self, station: Any, start_date: date, end_date: date
        ) -> list[HistoricalForecastPoint]:
            return forecasts

        async def fetch_historical_routine_metars(
            self, station: Any, start_date: date, end_date: date
        ) -> list[MetarObservation]:
            return metars

    app = create_app(db_path=":memory:", clock=clock, enable_background_poll=False)
    app.state.historical_fetcher = _FakeHistoricalFetcher()

    with TestClient(app) as client:
        resp = client.post(
            "/api/ml/train",
            json={
                "model_version": "mos-lgbm-fetcher-v1",
                "station_icaos": ["LFPB"],
                "start_date": "2026-10-01",
                "end_date": "2026-10-30",
                "lgbm_params": {"n_estimators": 10, "max_depth": 3, "num_leaves": 7, "min_child_samples": 2, "verbose": -1},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_version"]["model_version"] == "mos-lgbm-fetcher-v1"
        assert body["benchmark"]["train_count"] > 0
        assert body["benchmark"]["val_count"] > 0
        assert body["benchmark"]["test_count"] > 0


def test_mos_intraday_layer3_and_premarket_prior_obs_isolation() -> None:
    """Intraday METARs must not corrupt premarket prior_obs_temp_c, and post-peak cooling
    must collapse MOS probabilities onto the realized daily extreme via Layer 3 conditioning."""
    lfpb = get_station("LFPB")
    assert lfpb is not None
    samples = _make_synthetic_mos_history(station_icao="LFPB", n_days=40, start_date=date(2026, 9, 15))
    mos_model = QuantileMosModel.fit(
        samples,
        model_version="mos-layer3-test",
        covered_stations=["LFPB"],
        lgbm_params={"n_estimators": 15, "max_depth": 3, "num_leaves": 7, "min_child_samples": 3, "verbose": -1},
    )

    # 18:00 UTC on Nov 20 (19:00 Paris): afternoon peak (15°C) already passed, evening cooled to 11°C
    decision_time = datetime(2026, 11, 20, 18, 0, tzinfo=timezone.utc)
    clock = VirtualClock(decision_time)
    buckets = [
        Bucket(bucket_id="b14", market_id="m14", token_id="t14", label="14°C or below", lower_bound=None, upper_bound=14.0, unit="C", outcome_index=0),
        Bucket(bucket_id="b15", market_id="m15", token_id="t15", label="15°C", lower_bound=15.0, upper_bound=15.0, unit="C", outcome_index=1),
        Bucket(bucket_id="b16", market_id="m16", token_id="t16", label="16°C", lower_bound=16.0, upper_bound=16.0, unit="C", outcome_index=2),
        Bucket(bucket_id="b17", market_id="m17", token_id="t17", label="17°C", lower_bound=17.0, upper_bound=17.0, unit="C", outcome_index=3),
        Bucket(bucket_id="b18", market_id="m18", token_id="t18", label="18°C or higher", lower_bound=18.0, upper_bound=None, unit="C", outcome_index=4),
    ]
    event = WeatherEvent(
        event_id="evt-lfpb-evening",
        slug="highest-temp-paris-2026-11-20",
        title="Highest temp Paris?",
        url="https://polymarket.com/event/lfpb-evening",
        station=lfpb,
        local_date=date(2026, 11, 20),
        polarity=Polarity.HIGHEST,
        remark_tenths_enabled=True,
        unit="C",
        buckets=buckets,
        discovered_at_utc=decision_time - timedelta(hours=20),
    )
    # Morning ensemble predicted 17.5°C peak at 14:00 UTC, cooling to 12°C by 18:00..22:00 UTC
    ts = [datetime(2026, 11, 19, 23, 0, tzinfo=timezone.utc) + timedelta(hours=h) for h in range(24)]
    trajs = [
        [round(11.0 + 6.5 * max(0.0, 1.0 - abs(h - 14) / 7.0) + (m - 30) * 0.03, 2) for h in range(24)]
        for m in range(60)
    ]
    ens = EnsembleSnapshot(
        event_id=event.event_id,
        station_icao="LFPB",
        fetched_at_utc=decision_time - timedelta(hours=18),
        ingested_at_utc=decision_time - timedelta(hours=18),
        models=["ecmwf_ifs025", "gfs_seamless", "icon_seamless"],
        member_count=60,
        hourly_timestamps_utc=ts,
        member_temps_c=trajs,
    )
    yesterday_metar = MetarObservation(
        station_icao="LFPB",
        raw_report="METAR LFPB 191800Z 20005KT CAVOK 10/06 Q1016 RMK T01000060",
        observed_at_utc=datetime(2026, 11, 19, 18, 0, tzinfo=timezone.utc),
        ingested_at_utc=datetime(2026, 11, 19, 18, 5, tzinfo=timezone.utc),
        temp_c=10.0,
        whole_temp_c=10,
        remark_tenths_c=10.0,
        has_remark_tenths=True,
        is_routine=True,
    )
    peak_metar = MetarObservation(
        station_icao="LFPB",
        raw_report="METAR LFPB 201300Z 20005KT CAVOK 15/06 Q1016 RMK T01510060",
        observed_at_utc=datetime(2026, 11, 20, 13, 0, tzinfo=timezone.utc),
        ingested_at_utc=datetime(2026, 11, 20, 13, 5, tzinfo=timezone.utc),
        temp_c=15.1,
        whole_temp_c=15,
        remark_tenths_c=15.1,
        has_remark_tenths=True,
        is_routine=True,
    )
    evening_metar = MetarObservation(
        station_icao="LFPB",
        raw_report="METAR LFPB 201730Z 20005KT CAVOK 11/06 Q1016 RMK T01100060",
        observed_at_utc=datetime(2026, 11, 20, 17, 30, tzinfo=timezone.utc),
        ingested_at_utc=datetime(2026, 11, 20, 17, 35, tzinfo=timezone.utc),
        temp_c=11.0,
        whole_temp_c=11,
        remark_tenths_c=11.0,
        has_remark_tenths=True,
        is_routine=True,
    )

    from weather_quant.ml.dataset_builder import extract_features_from_ensemble

    feats = extract_features_from_ensemble(
        station=lfpb,
        local_date=date(2026, 11, 20),
        polarity=Polarity.HIGHEST,
        ensemble=ens,
        decision_time_utc=decision_time,
        prior_observations=[yesterday_metar, peak_metar, evening_metar],
    )
    # Must use strictly prior-day 10.0C, NOT today's 15.1C or 11.0C!
    assert feats["prior_obs_temp_c"] == pytest.approx(10.0)

    snap = mos_model.predict_event_probabilities(
        station=lfpb,
        event=event,
        ensemble=ens,
        intraday_metars=[yesterday_metar, peak_metar, evening_metar],
        clock=clock,
    )
    by_id = {b.bucket_id: b for b in snap.buckets}
    assert by_id["b14"].is_impossible is True
    assert by_id["b14"].fair_probability == 0.0
    # After peak passed and evening cooled to 11°C, realized 15°C must strongly dominate 17°C+
    assert by_id["b15"].fair_probability > 0.85
    assert by_id["b17"].fair_probability < 0.05


@pytest.mark.asyncio
async def test_ensemble_fetcher_preserves_per_model_hourly_means_and_spread() -> None:
    """Open-Meteo multi-model suffixed ensemble payloads must preserve ECMWF/GFS/ICON hourly means
    so extract_features_from_ensemble computes real per-day model_spread_c for HIGHEST & LOWEST."""
    from weather_quant.ml.dataset_builder import extract_features_from_ensemble
    from weather_quant.probability.ensemble_fetcher import OpenMeteoEnsembleFetcher

    lfpb = get_station("LFPB")
    assert lfpb is not None
    ts = [f"2026-11-26T{h:02d}:00" for h in range(24)]
    payload = {
        "hourly": {
            "time": ts,
            "temperature_2m_member01_ecmwf_ifs025": [10.0 + (2.0 if h == 14 else 0.0) for h in range(24)],
            "temperature_2m_member01_gfs_seamless": [13.0 + (2.0 if h == 14 else 0.0) for h in range(24)],
            "temperature_2m_member01_icon_seamless": [11.0 + (2.0 if h == 14 else 0.0) for h in range(24)],
        }
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    clock = VirtualClock(datetime(2026, 11, 25, 18, 0, tzinfo=timezone.utc))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        fetcher = OpenMeteoEnsembleFetcher(client=client, clock=clock)
        snap = await fetcher.fetch_ensemble(lfpb, event_id="evt-spread-test", local_date=date(2026, 11, 26))

    assert "model_hourly_means_c" in snap.metadata
    assert set(snap.metadata["model_hourly_means_c"].keys()) == {
        "ecmwf_ifs025",
        "gfs_seamless",
        "icon_seamless",
    }

    feats_hi = extract_features_from_ensemble(
        station=lfpb,
        local_date=date(2026, 11, 26),
        polarity=Polarity.HIGHEST,
        ensemble=snap,
        decision_time_utc=datetime(2026, 11, 25, 22, 0, tzinfo=timezone.utc),
    )
    assert feats_hi["ecmwf_extreme_c"] == pytest.approx(12.0)
    assert feats_hi["gfs_extreme_c"] == pytest.approx(15.0)
    assert feats_hi["icon_extreme_c"] == pytest.approx(13.0)
    assert feats_hi["model_spread_c"] == pytest.approx(3.0)

    feats_lo = extract_features_from_ensemble(
        station=lfpb,
        local_date=date(2026, 11, 26),
        polarity=Polarity.LOWEST,
        ensemble=snap,
        decision_time_utc=datetime(2026, 11, 25, 22, 0, tzinfo=timezone.utc),
    )
    assert feats_lo["ecmwf_extreme_c"] == pytest.approx(10.0)
    assert feats_lo["gfs_extreme_c"] == pytest.approx(13.0)
    assert feats_lo["model_spread_c"] == pytest.approx(3.0)


