"""Behavioral tests for 3-layer ensemble probability engine & intraday METAR repricing."""

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
from weather_quant.probability.engine import (
    DEFAULT_MODEL_VERSION,
    DailyBiasObservation,
    EnsembleProbabilityEngine,
    get_latest_event_probability_snapshot,
    save_event_probability_snapshot,
)
from weather_quant.probability.ensemble_fetcher import (
    OPEN_METEO_ENSEMBLE_URL,
    OpenMeteoEnsembleFetcher,
)
from weather_quant.storage.database import Database


def _build_open_meteo_payload(
    start_utc: datetime,
    hours: int,
    model_members: dict[str, int],
    base_temp_c: float = 15.0,
) -> list[dict[str, Any]]:
    """Build deterministic multi-model Open-Meteo Ensemble API response fixture."""
    timestamps = [
        (start_utc + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M")
        for h in range(hours)
    ]
    responses: list[dict[str, Any]] = []
    for model_idx, (model_name, count) in enumerate(model_members.items()):
        hourly: dict[str, Any] = {"time": timestamps}
        # Control run + perturbed members totaling `count` trajectories per model
        for m in range(count):
            key = "temperature_2m" if m == 0 else f"temperature_2m_member{m:02d}"
            offset = (m - count / 2.0) * 0.05 + model_idx * 0.1
            hourly[key] = [
                round(base_temp_c + offset + 4.0 * (1.0 - abs(((h % 24) - 14) / 14.0)), 2)
                for h in range(hours)
            ]
        responses.append({"model": model_name, "hourly": hourly})
    return responses


@pytest.mark.asyncio
async def test_open_meteo_ensemble_fetcher_queries_runway_coords_and_extracts_120_members() -> None:
    station = get_station("LFPB")
    assert station is not None
    decision_time = datetime(2026, 11, 25, 18, 0, tzinfo=timezone.utc)
    clock = VirtualClock(decision_time)
    seen_requests: list[httpx.Request] = []

    payload = _build_open_meteo_payload(
        start_utc=datetime(2026, 11, 25, 0, 0, tzinfo=timezone.utc),
        hours=48,
        model_members={
            "ecmwf_ifs025": 51,
            "gfs_seamless": 31,
            "icon_seamless": 40,
        },
    )

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        fetcher = OpenMeteoEnsembleFetcher(client=client, clock=clock)
        snapshot = await fetcher.fetch_ensemble(
            station=station,
            event_id="evt-lfpb-nov26",
            local_date=date(2026, 11, 26),
        )

    assert len(seen_requests) == 1
    req = seen_requests[0]
    assert str(req.url).startswith(OPEN_METEO_ENSEMBLE_URL)
    params = req.url.params
    assert float(params["latitude"]) == pytest.approx(station.lat)
    assert float(params["longitude"]) == pytest.approx(station.lon)
    assert float(params["elevation"]) == pytest.approx(station.elevation_m)
    assert params["hourly"] == "temperature_2m"
    assert "ecmwf_ifs025" in params["models"]
    assert "gfs_seamless" in params["models"]
    assert "icon_seamless" in params["models"]

    assert snapshot.event_id == "evt-lfpb-nov26"
    assert snapshot.station_icao == "LFPB"
    assert snapshot.member_count == 122
    assert len(snapshot.member_temps_c) == 122
    assert len(snapshot.hourly_timestamps_utc) == 48
    assert all(len(traj) == 48 for traj in snapshot.member_temps_c)
    assert snapshot.ingested_at_utc == decision_time
    assert snapshot.fetched_at_utc == decision_time


def _make_rksi_event(
    local_date: date,
    polarity: Polarity = Polarity.HIGHEST,
    remark_tenths: bool = True,
) -> WeatherEvent:
    station = get_station("RKSI")
    assert station is not None
    buckets = [
        Bucket(bucket_id="b-10", market_id="m-10", token_id="t-10", label="10°C or below", lower_bound=None, upper_bound=10.0, unit="C", outcome_index=0),
        Bucket(bucket_id="b-11", market_id="m-11", token_id="t-11", label="11°C", lower_bound=11.0, upper_bound=11.0, unit="C", outcome_index=1),
        Bucket(bucket_id="b-12", market_id="m-12", token_id="t-12", label="12°C", lower_bound=12.0, upper_bound=12.0, unit="C", outcome_index=2),
        Bucket(bucket_id="b-13", market_id="m-13", token_id="t-13", label="13°C", lower_bound=13.0, upper_bound=13.0, unit="C", outcome_index=3),
        Bucket(bucket_id="b-14", market_id="m-14", token_id="t-14", label="14°C or higher", lower_bound=14.0, upper_bound=None, unit="C", outcome_index=4),
    ]
    return WeatherEvent(
        event_id="evt-rksi-2026-11-26",
        slug="highest-temp-seoul-nov-26",
        title="Highest temperature in Seoul on November 26?",
        url="https://polymarket.com/event/highest-temp-seoul-nov-26",
        station=station,
        local_date=local_date,
        polarity=polarity,
        remark_tenths_enabled=remark_tenths,
        unit="C",
        buckets=buckets,
        discovered_at_utc=datetime(2026, 11, 25, 0, 0, tzinfo=timezone.utc),
    )


def test_layer1_local_timezone_extremes_normalization_and_polarity() -> None:
    """RKSI (UTC+9): local Nov 26 runs 2026-11-25 15:00 UTC through 2026-11-26 14:00 UTC.
    Spikes outside local Nov 26 (e.g. Nov 26 18:00 UTC = Nov 27 03:00 KST) must be ignored.
    """
    event_high = _make_rksi_event(local_date=date(2026, 11, 26), polarity=Polarity.HIGHEST)
    event_low = _make_rksi_event(local_date=date(2026, 11, 26), polarity=Polarity.LOWEST)
    clock = VirtualClock(datetime(2026, 11, 25, 14, 0, tzinfo=timezone.utc))

    # Build 48h UTC grid starting 2026-11-25 12:00 UTC
    t0 = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
    timestamps = [t0 + timedelta(hours=h) for h in range(48)]

    # 120 members: inside Nov 26 KST, max is ~12.0C (for 90 members) or ~13.0C (for 30 members),
    # min is ~10.0C (for 100 members) or ~11.0C (for 20 members).
    # Outside Nov 26 KST (Nov 27 03:00 KST -> h=30, i.e. Nov 26 18:00 UTC), put a decoy 25.0C / 2.0C!
    trajectories: list[list[float]] = []
    for idx in range(120):
        day_peak = 12.1 if idx < 90 else 13.1
        day_trough = 10.0 if idx < 100 else 11.0
        row: list[float] = []
        for ts in timestamps:
            # Nov 26 KST is [2026-11-25 15:00 UTC, 2026-11-26 15:00 UTC)
            if datetime(2026, 11, 25, 15, 0, tzinfo=timezone.utc) <= ts < datetime(2026, 11, 26, 15, 0, tzinfo=timezone.utc):
                if ts.hour == 5:  # 14:00 KST afternoon peak
                    row.append(day_peak)
                elif ts.hour == 21:  # 06:00 KST morning low
                    row.append(day_trough)
                else:
                    row.append(11.4)
            else:
                # Decoy extremes outside local Nov 26!
                row.append(25.0 if ts.hour == 18 else 2.0)
        trajectories.append(row)

    ensemble = EnsembleSnapshot(
        event_id=event_high.event_id,
        station_icao="RKSI",
        fetched_at_utc=clock.now(),
        ingested_at_utc=clock.now(),
        models=["ecmwf_ifs025", "gfs_seamless", "icon_seamless"],
        member_count=120,
        hourly_timestamps_utc=timestamps,
        member_temps_c=trajectories,
    )

    engine = EnsembleProbabilityEngine()
    snap_high = engine.compute_probabilities(
        station=event_high.station,
        event=event_high,
        ensemble=ensemble,
        recent_bias_obs=[],
        intraday_metars=[],
        clock=clock,
        model_version=DEFAULT_MODEL_VERSION,
    )

    assert snap_high.model_version == "ensemble-3layer-v1"
    assert snap_high.probability_sum == pytest.approx(1.0, abs=1e-9)
    assert sum(b.fair_probability for b in snap_high.buckets) == pytest.approx(1.0, abs=1e-9)
    by_id_high = {b.bucket_id: b for b in snap_high.buckets}
    # 12°C has 90/120 = 0.75, 13°C has 30/120 = 0.25 -> 12°C must clearly dominate 13°C and 14°C+
    assert by_id_high["b-12"].fair_probability > 0.70
    assert by_id_high["b-13"].fair_probability > 0.20
    assert by_id_high["b-14"].fair_probability < 0.02

    # LOWEST polarity picks morning trough (~10C vs ~11C) and ignores outside 2.0C
    snap_low = engine.compute_probabilities(
        station=event_low.station,
        event=event_low,
        ensemble=ensemble,
        recent_bias_obs=[],
        intraday_metars=[],
        clock=clock,
    )
    assert snap_low.probability_sum == pytest.approx(1.0, abs=1e-9)
    by_id_low = {b.bucket_id: b for b in snap_low.buckets}
    assert by_id_low["b-10"].fair_probability > 0.75
    assert by_id_low["b-11"].fair_probability > 0.14
    assert by_id_low["b-12"].fair_probability < 0.02


def test_layer2_station_bias_shifts_mean_and_scales_dispersion() -> None:
    """30-day rolling Station Bias (+2.0C offset, wider dispersion) shifts Fair Probability
    mass from 11C into 13C/14C+ while ignoring SPECI and future bias observations."""
    event = _make_rksi_event(local_date=date(2026, 11, 26), polarity=Polarity.HIGHEST)
    decision_time = datetime(2026, 11, 25, 14, 0, tzinfo=timezone.utc)
    clock = VirtualClock(decision_time)

    timestamps = [
        datetime(2026, 11, 25, 15, 0, tzinfo=timezone.utc) + timedelta(hours=h)
        for h in range(24)
    ]
    # Raw ensemble centered around 11.0C (range [10.6, 11.4] -> all inside 11°C bucket)
    trajectories = [
        [11.0 + ((m - 59.5) / 60.0) * 0.40 for _ in timestamps]
        for m in range(120)
    ]
    ensemble = EnsembleSnapshot(
        event_id=event.event_id,
        station_icao="RKSI",
        fetched_at_utc=decision_time,
        ingested_at_utc=decision_time,
        models=["ecmwf_ifs025", "gfs_seamless", "icon_seamless"],
        member_count=120,
        hourly_timestamps_utc=timestamps,
        member_temps_c=trajectories,
    )

    # 30 recent routine days where station consistently ran +2.0C warmer than forecast
    # with alternating +-0.8C day-to-day residual spread (std ~ 0.8C vs forecast_std 0.4C -> scale=2.0)
    bias_obs: list[DailyBiasObservation] = []
    for d in range(1, 31):
        obs_date = date(2026, 11, 25) - timedelta(days=d)
        residual = 2.0 + (0.8 if d % 2 == 0 else -0.8)
        bias_obs.append(
            DailyBiasObservation(
                local_date=obs_date,
                forecast_extreme_c=10.0,
                observed_extreme_c=10.0 + residual,
                forecast_std_c=0.4,
                observed_at_utc=datetime(obs_date.year, obs_date.month, obs_date.day, 14, 0, tzinfo=timezone.utc),
                is_routine=True,
            )
        )
    # Disqualified SPECI outlier (-15C) & future leaked record (+15C) must be ignored
    bias_obs.append(
        DailyBiasObservation(
            local_date=date(2026, 11, 24),
            forecast_extreme_c=10.0,
            observed_extreme_c=-5.0,
            is_routine=False,
        )
    )
    bias_obs.append(
        DailyBiasObservation(
            local_date=date(2026, 11, 26),
            forecast_extreme_c=10.0,
            observed_extreme_c=25.0,
            observed_at_utc=decision_time + timedelta(hours=2),
            is_routine=True,
        )
    )

    engine = EnsembleProbabilityEngine()
    snap = engine.compute_probabilities(
        station=event.station,
        event=event,
        ensemble=ensemble,
        recent_bias_obs=bias_obs,
        intraday_metars=[],
        clock=clock,
    )

    assert snap.probability_sum == pytest.approx(1.0, abs=1e-9)
    by_id = {b.bucket_id: b for b in snap.buckets}
    # Raw ensemble was centered at 11°C
    assert (by_id["b-11"].raw_ensemble_probability or 0.0) > 0.90
    # After +2.0C Station Bias, peak shifts to 13°C (and scaled dispersion spreads some mass to 12°C/14°C)
    assert (by_id["b-13"].bias_adjusted_probability or 0.0) > 0.60
    assert by_id["b-13"].fair_probability > by_id["b-11"].fair_probability
    assert (by_id["b-14"].fair_probability > 0.01) or (by_id["b-12"].fair_probability > 0.01)


def _make_metar(
    icao: str,
    observed_at_utc: datetime,
    temp_c: float,
    whole_temp_c: int,
    is_routine: bool = True,
    has_remark_tenths: bool = True,
) -> MetarObservation:
    prefix = "METAR" if is_routine else "SPECI"
    return MetarObservation(
        station_icao=icao,
        raw_report=f"{prefix} {icao} {observed_at_utc:%d%H%M}Z 24010KT CAVOK {whole_temp_c:02d}/05 Q1015",
        observed_at_utc=observed_at_utc,
        ingested_at_utc=observed_at_utc,
        temp_c=temp_c,
        whole_temp_c=whole_temp_c,
        remark_tenths_c=temp_c if has_remark_tenths else None,
        has_remark_tenths=has_remark_tenths,
        is_routine=is_routine,
    )


def test_layer3_intraday_metar_zeroes_impossible_buckets_and_ignores_speci_and_future() -> None:
    """Intraday Routine METAR at 12.1C (HIGHEST) zeroes 10C/11C buckets, drifts remaining
    afternoon trajectories warmer, and completely ignores SPECI and future METARs."""
    event = _make_rksi_event(local_date=date(2026, 11, 26), polarity=Polarity.HIGHEST, remark_tenths=True)
    # 2026-11-26 02:30 UTC == 11:30 KST
    decision_time = datetime(2026, 11, 26, 2, 30, tzinfo=timezone.utc)
    clock = VirtualClock(decision_time)

    # Ensemble diurnal curve on Nov 26 KST: 8.0C overnight -> 11.0C peak at 14:00 KST (05:00 UTC)
    # At 11:00 KST (02:00 UTC), ensemble predicts ~10.0C (+- 0.6C across members)
    timestamps = [
        datetime(2026, 11, 25, 15, 0, tzinfo=timezone.utc) + timedelta(hours=h)
        for h in range(24)
    ]
    trajectories: list[list[float]] = []
    for m in range(120):
        member_offset = ((m - 59.5) / 60.0) * 0.6
        row: list[float] = []
        for ts in timestamps:
            kst_hour = (ts.hour + 9) % 24
            diurnal = 8.0 + 3.0 * max(0.0, 1.0 - abs(kst_hour - 14) / 6.0)
            row.append(round(diurnal + member_offset, 3))
        trajectories.append(row)

    ensemble = EnsembleSnapshot(
        event_id=event.event_id,
        station_icao="RKSI",
        fetched_at_utc=decision_time - timedelta(hours=4),
        ingested_at_utc=decision_time - timedelta(hours=4),
        models=["ecmwf_ifs025", "gfs_seamless", "icon_seamless"],
        member_count=120,
        hourly_timestamps_utc=timestamps,
        member_temps_c=trajectories,
    )

    engine = EnsembleProbabilityEngine()

    # 1) Disqualified SPECI at 14.4C and future Routine METAR at 14.8C must NOT affect probabilities
    disqualified_only = [
        _make_metar("RKSI", datetime(2026, 11, 26, 2, 15, tzinfo=timezone.utc), 14.4, 14, is_routine=False),
        _make_metar("RKSI", decision_time + timedelta(minutes=30), 14.8, 15, is_routine=True),
    ]
    snap_disqualified = engine.compute_probabilities(
        station=event.station,
        event=event,
        ensemble=ensemble,
        recent_bias_obs=[],
        intraday_metars=disqualified_only,
        clock=clock,
    )
    by_id_disq = {b.bucket_id: b for b in snap_disqualified.buckets}
    assert by_id_disq["b-11"].is_impossible is False
    assert by_id_disq["b-11"].fair_probability > 0.50
    assert by_id_disq["b-14"].fair_probability < 0.02

    # strict_temporal=True must raise TemporalIntegrityError on future METAR
    with pytest.raises(TemporalIntegrityError):
        engine.compute_probabilities(
            station=event.station,
            event=event,
            ensemble=ensemble,
            recent_bias_obs=[],
            intraday_metars=disqualified_only,
            clock=clock,
            strict_temporal=True,
        )

    # 2) Verified Routine METAR at 11:00 KST (02:00 UTC) reads 12.1C (vs ~10.0C ensemble -> +2.1C warm drift!)
    valid_intraday = [
        _make_metar("RKSI", datetime(2026, 11, 26, 2, 0, tzinfo=timezone.utc), 12.1, 12, is_routine=True),
        *disqualified_only,
    ]
    snap_intraday = engine.compute_probabilities(
        station=event.station,
        event=event,
        ensemble=ensemble,
        recent_bias_obs=[],
        intraday_metars=valid_intraday,
        clock=clock,
    )
    assert snap_intraday.probability_sum == pytest.approx(1.0, abs=1e-9)
    by_id_intra = {b.bucket_id: b for b in snap_intraday.buckets}
    # 10C or below and 11C are strictly below realized 12.1C -> mathematically impossible (0.0)
    assert by_id_intra["b-10"].is_impossible is True
    assert by_id_intra["b-10"].fair_probability == 0.0
    assert by_id_intra["b-11"].is_impossible is True
    assert by_id_intra["b-11"].fair_probability == 0.0
    # Warm drift lifts remaining afternoon peak from 11C into 12C/13C
    assert (by_id_intra["b-12"].fair_probability + by_id_intra["b-13"].fair_probability) > 0.90
    assert by_id_intra["b-13"].fair_probability > by_id_intra["b-13"].raw_ensemble_probability  # type: ignore[operator]


def test_layer3_lowest_polarity_and_remark_tenths_vs_whole_degree() -> None:
    """LOWEST market zeroes buckets strictly above realized daily min Routine METAR;
    whole-degree mode uses whole_temp_c while remark_tenths mode uses remark_tenths_c."""
    decision_time = datetime(2026, 11, 26, 6, 0, tzinfo=timezone.utc)
    clock = VirtualClock(decision_time)
    timestamps = [
        datetime(2026, 11, 25, 15, 0, tzinfo=timezone.utc) + timedelta(hours=h)
        for h in range(24)
    ]
    trajectories = [[12.5 + ((m - 59.5) / 60.0) * 1.2 for _ in timestamps] for m in range(120)]

    event_low = _make_rksi_event(local_date=date(2026, 11, 26), polarity=Polarity.LOWEST, remark_tenths=True)
    ensemble = EnsembleSnapshot(
        event_id=event_low.event_id,
        station_icao="RKSI",
        fetched_at_utc=decision_time - timedelta(hours=2),
        ingested_at_utc=decision_time - timedelta(hours=2),
        models=["ecmwf_ifs025"],
        member_count=120,
        hourly_timestamps_utc=timestamps,
        member_temps_c=trajectories,
    )

    # Routine METAR observed 11.2C -> buckets 12C, 13C, 14C+ are impossible in LOWEST
    obs = _make_metar("RKSI", datetime(2026, 11, 26, 1, 0, tzinfo=timezone.utc), 11.2, 11, is_routine=True)
    engine = EnsembleProbabilityEngine()
    snap_low = engine.compute_probabilities(
        station=event_low.station,
        event=event_low,
        ensemble=ensemble,
        intraday_metars=[obs],
        clock=clock,
    )
    by_low = {b.bucket_id: b for b in snap_low.buckets}
    assert by_low["b-12"].is_impossible is True and by_low["b-12"].fair_probability == 0.0
    assert by_low["b-13"].is_impossible is True and by_low["b-13"].fair_probability == 0.0
    assert by_low["b-14"].is_impossible is True and by_low["b-14"].fair_probability == 0.0
    assert by_low["b-11"].is_impossible is False
    assert snap_low.probability_sum == pytest.approx(1.0, abs=1e-9)

    # Remark Tenths enabled vs disabled when METAR has whole_temp=12 and remark_tenths=12.6
    # Under RMK tenths (12.6 rounds to 13), in HIGHEST market 12°C is impossible;
    # under whole-degree (whole_temp_c=12), 12°C remains possible.
    obs_borderline = _make_metar("RKSI", datetime(2026, 11, 26, 2, 0, tzinfo=timezone.utc), 12.6, 12, is_routine=True, has_remark_tenths=True)
    event_rmk = _make_rksi_event(local_date=date(2026, 11, 26), polarity=Polarity.HIGHEST, remark_tenths=True)
    event_whole = _make_rksi_event(local_date=date(2026, 11, 26), polarity=Polarity.HIGHEST, remark_tenths=False)

    snap_rmk = engine.compute_probabilities(event_rmk.station, event_rmk, ensemble, intraday_metars=[obs_borderline], clock=clock)
    snap_whole = engine.compute_probabilities(event_whole.station, event_whole, ensemble, intraday_metars=[obs_borderline], clock=clock)

    assert {b.bucket_id: b for b in snap_rmk.buckets}["b-12"].is_impossible is True
    assert {b.bucket_id: b for b in snap_whole.buckets}["b-12"].is_impossible is False


class _ProbabilityFixtureWorldFeed:
    def __init__(self, events: list[dict[str, Any]], metar_lines: list[str]) -> None:
        self._events = events
        self._metar_lines = metar_lines

    async def fetch_weather_events(self) -> list[dict[str, Any]]:
        return list(self._events)

    async def fetch_metar_reports(self, station_icaos: list[str]) -> list[str]:
        return [line for line in self._metar_lines if any(icao in line for icao in station_icaos)]


@pytest.mark.asyncio
async def test_database_persists_and_loads_tagged_predictions() -> None:
    db = Database(":memory:")
    await db.initialize()
    try:
        event = _make_rksi_event(local_date=date(2026, 11, 26), polarity=Polarity.HIGHEST)
        await db.upsert_weather_event(event)

        clock = VirtualClock(datetime(2026, 11, 25, 18, 0, tzinfo=timezone.utc))
        timestamps = [
            datetime(2026, 11, 25, 15, 0, tzinfo=timezone.utc) + timedelta(hours=h)
            for h in range(24)
        ]
        ensemble = EnsembleSnapshot(
            event_id=event.event_id,
            station_icao="RKSI",
            fetched_at_utc=clock.now(),
            ingested_at_utc=clock.now(),
            models=["ecmwf_ifs025"],
            member_count=120,
            hourly_timestamps_utc=timestamps,
            member_temps_c=[[12.0 for _ in timestamps] for _ in range(120)],
        )
        engine = EnsembleProbabilityEngine()
        snap = engine.compute_probabilities(
            station=event.station,
            event=event,
            ensemble=ensemble,
            clock=clock,
            model_version="ensemble-3layer-v1",
        )
        await save_event_probability_snapshot(db, snap)

        loaded = await get_latest_event_probability_snapshot(db, event.event_id, model_version="ensemble-3layer-v1")
        assert loaded is not None
        assert loaded.event_id == event.event_id
        assert loaded.model_version == "ensemble-3layer-v1"
        assert loaded.probability_sum == pytest.approx(1.0, abs=1e-9)
        assert len(loaded.buckets) == 5
        assert {b.bucket_id: b.label for b in loaded.buckets}["b-12"] == "12°C"
    finally:
        await db.close()


def test_fastapi_probability_routes_recompute_and_serve_persisted_snapshots() -> None:
    # 2026-11-26 11:30 UTC (12:30 CET in Paris)
    decision_time = datetime(2026, 11, 26, 11, 30, tzinfo=timezone.utc)
    clock = VirtualClock(decision_time)

    world_feed = _ProbabilityFixtureWorldFeed(
        events=[
            {
                "id": "evt-lfpb-nov26",
                "slug": "highest-temp-paris-nov-26",
                "title": "Highest temperature in Paris on November 26?",
                "description": (
                    "Resolves on routine hourly METAR at Paris-Le Bourget (ICAO: LFPB) "
                    "on November 26, 2026. Tenths of a degree via RMK if available."
                ),
                "markets": [
                    {"id": "b1", "groupItemTitle": "17°C or below", "clobTokenIds": '["t1","n1"]'},
                    {"id": "b2", "groupItemTitle": "18°C", "clobTokenIds": '["t2","n2"]'},
                    {"id": "b3", "groupItemTitle": "19°C or higher", "clobTokenIds": '["t3","n3"]'},
                ],
            }
        ],
        metar_lines=[
            # Routine METAR at 11:00 UTC observed 18.2C -> 17°C or below is impossible
            "METAR LFPB 261100Z 24012KT 9999 BKN040 18/11 Q1015 RMK T01820110",
            # Disqualified SPECI at 11:15 UTC jumping to 22.0C must be ignored
            "SPECI LFPB 261115Z 24015KT 9999 SCT030 22/10 Q1015 RMK T02200100",
        ],
    )

    payload = _build_open_meteo_payload(
        start_utc=datetime(2026, 11, 25, 0, 0, tzinfo=timezone.utc),
        hours=72,
        model_members={"ecmwf_ifs025": 51, "gfs_seamless": 31, "icon_seamless": 40},
        base_temp_c=14.0,  # Peak around 18.0C
    )
    mock_http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _req: httpx.Response(200, json=payload))
    )

    app = create_app(
        db_path=":memory:",
        world_feed=world_feed,
        clock=clock,
        enable_background_poll=False,
    )
    app.state.ensemble_fetcher = OpenMeteoEnsembleFetcher(client=mock_http, clock=clock)

    with TestClient(app) as client:
        assert client.post("/api/stations/sync").status_code == 200

        recompute_resp = client.post("/api/probability/recompute", json={})
        assert recompute_resp.status_code == 200
        recompute_data = recompute_resp.json()
        assert recompute_data["count"] == 1
        assert recompute_data["model_version"] == "ensemble-3layer-v1"

        list_resp = client.get("/api/probability/events")
        assert list_resp.status_code == 200
        list_data = list_resp.json()
        assert len(list_data["events"]) == 1

        detail_resp = client.get("/api/probability/events/evt-lfpb-nov26")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["event_id"] == "evt-lfpb-nov26"
        assert detail["model_version"] == "ensemble-3layer-v1"
        assert detail["probability_sum"] == pytest.approx(1.0, abs=1e-9)

        by_id = {b["bucket_id"]: b for b in detail["buckets"]}
        # 17°C or below is strictly below realized 18.2°C Routine METAR -> impossible (0.0)
        assert by_id["b1"]["is_impossible"] is True
        assert by_id["b1"]["fair_probability"] == 0.0
        # 19°C+ is still possible, and 18°C+19°C sum to 1.0
        assert by_id["b2"]["fair_probability"] + by_id["b3"]["fair_probability"] == pytest.approx(1.0, abs=1e-9)

        missing_resp = client.get("/api/probability/events/nonexistent-evt")
        assert missing_resp.status_code == 404




