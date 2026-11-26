"""TDD tests for Phase A: exact 0.1C->F quantization bounds and D0 Censored-Max Kalman-Cholesky."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from weather_quant.discovery.stations import get_station
from weather_quant.domain.clock import VirtualClock
from weather_quant.domain.models import (
    Bucket,
    EnsembleSnapshot,
    MetarObservation,
    Polarity,
    WeatherEvent,
)
from weather_quant.probability.censored_max import simulate_censored_extreme
from weather_quant.probability.engine import (
    EnsembleProbabilityEngine,
    bucket_effective_bounds_c,
    map_extremes_c_to_probabilities,
)


def test_fahrenheit_quantization_captures_18f_cycle_wide_vs_narrow_buckets() -> None:
    """Every 9th 2°F bucket spans 12 tenths (1.20°C) instead of 11 tenths (1.10°C)."""
    b_narrow = Bucket(
        bucket_id="b-44-45",
        market_id="m-44-45",
        token_id="t-44-45",
        label="44-45°F",
        lower_bound=44.0,
        upper_bound=45.0,
        unit="F",
        outcome_index=0,
    )
    b_wide = Bucket(
        bucket_id="b-46-47",
        market_id="m-46-47",
        token_id="t-46-47",
        label="46-47°F",
        lower_bound=46.0,
        upper_bound=47.0,
        unit="F",
        outcome_index=1,
    )

    low_n, high_n, is_int_n = bucket_effective_bounds_c(b_narrow, remark_tenths_enabled=True)
    low_w, high_w, is_int_w = bucket_effective_bounds_c(b_wide, remark_tenths_enabled=True)

    assert is_int_n is True and is_int_w is True
    assert low_n == pytest.approx(6.35, abs=1e-6)
    assert high_n == pytest.approx(7.45, abs=1e-6)
    assert high_n - low_n == pytest.approx(1.10, abs=1e-6)

    assert low_w == pytest.approx(7.45, abs=1e-6)
    assert high_w == pytest.approx(8.65, abs=1e-6)
    assert high_w - low_w == pytest.approx(1.20, abs=1e-6)

    # Uniform temperature grid over [6.35, 8.65) must give ~12/23 vs ~11/23 (~9.09% more mass to 46-47°F)
    grid_c = [6.35 + (i + 0.5) * (2.30 / 2300.0) for i in range(2300)]
    probs = map_extremes_c_to_probabilities(
        grid_c,
        [b_narrow, b_wide],
        [False, False],
        remark_tenths_enabled=True,
        smoothing_alpha=0.0,
    )
    assert probs[0] == pytest.approx(11.0 / 23.0, abs=1e-3)
    assert probs[1] == pytest.approx(12.0 / 23.0, abs=1e-3)
    assert probs[1] / probs[0] == pytest.approx(12.0 / 11.0, abs=1e-2)


def test_censored_max_peak_lock_probability_approaches_one_after_afternoon_peak() -> None:
    """After afternoon peak has passed and evening is cooling, Censored-Max locks mass at M_t."""
    t0 = datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc)
    day_ts = [t0 + timedelta(hours=h) for h in range(24)]
    # Diurnal curve peaking at 14:00 at 20.0C and cooling to 12.0C by 23:00
    prior_mean = [
        12.0 + 8.0 * max(0.0, 1.0 - abs(h - 14) / 8.0)
        for h in range(24)
    ]
    prior_std = [1.0 for _ in range(24)]

    # Observations up to 18:00: peak 21.0C at 14:00, already cooled to 16.5C at 18:00
    obs_points = [
        (t0 + timedelta(hours=10), 17.0),
        (t0 + timedelta(hours=14), 21.0),
        (t0 + timedelta(hours=16), 19.0),
        (t0 + timedelta(hours=18), 16.5),
    ]

    res = simulate_censored_extreme(
        day_timestamps=day_ts,
        prior_mean_curve_c=prior_mean,
        prior_std_curve_c=prior_std,
        obs_points=obs_points,
        polarity=Polarity.HIGHEST,
        seed_key="evt-test:2026-11-26T18:00:00Z",
        n_simulations=500,
    )

    assert res.realized_extreme_c == pytest.approx(21.0)
    assert res.lock_probability > 0.95
    assert all(x >= 21.0 - 1e-9 for x in res.simulated_extremes_c)

    # Determinism check with identical seed_key
    res_repeat = simulate_censored_extreme(
        day_timestamps=day_ts,
        prior_mean_curve_c=prior_mean,
        prior_std_curve_c=prior_std,
        obs_points=obs_points,
        polarity=Polarity.HIGHEST,
        seed_key="evt-test:2026-11-26T18:00:00Z",
        n_simulations=500,
    )
    assert res.simulated_extremes_c == res_repeat.simulated_extremes_c


def test_censored_max_supports_lowest_polarity_and_morning_upside() -> None:
    """Morning observation still below afternoon peak leaves non-trivial upside (lock_probability < 0.5)."""
    t0 = datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc)
    day_ts = [t0 + timedelta(hours=h) for h in range(24)]
    prior_mean = [10.0 + 10.0 * max(0.0, 1.0 - abs(h - 14) / 8.0) for h in range(24)]
    prior_std = [1.2 for _ in range(24)]

    # Morning 09:00 observation at 13.8C — afternoon peak ~20C is still ahead
    res_high = simulate_censored_extreme(
        day_timestamps=day_ts,
        prior_mean_curve_c=prior_mean,
        prior_std_curve_c=prior_std,
        obs_points=[(t0 + timedelta(hours=9), 13.8)],
        polarity=Polarity.HIGHEST,
        seed_key="evt-morning:09Z",
    )
    assert res_high.lock_probability < 0.1
    assert max(res_high.simulated_extremes_c) > 18.0

    # LOWEST polarity: morning trough at 06:00 already reached 9.0C, afternoon warming
    res_low = simulate_censored_extreme(
        day_timestamps=day_ts,
        prior_mean_curve_c=prior_mean,
        prior_std_curve_c=prior_std,
        obs_points=[(t0 + timedelta(hours=6), 9.0), (t0 + timedelta(hours=11), 16.0)],
        polarity=Polarity.LOWEST,
        seed_key="evt-low:11Z",
    )
    assert res_low.realized_extreme_c == pytest.approx(9.0)
    assert all(x <= 9.0 + 1e-9 for x in res_low.simulated_extremes_c)


def test_ensemble_engine_intraday_censored_max_concentrates_on_realized_peak_bucket() -> None:
    """End-to-end check: EnsembleProbabilityEngine concentrates mass on 14°C once 14.1°C peak has passed."""
    station = get_station("EGLC")
    assert station is not None
    t0 = datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc)
    timestamps = [t0 + timedelta(hours=h) for h in range(24)]
    trajectories = [
        [8.0 + 6.0 * max(0.0, 1.0 - abs(h - 14) / 7.0) + (m - 30) * 0.03 for h in range(24)]
        for m in range(60)
    ]
    buckets = [
        Bucket(bucket_id="b-12", market_id="m-12", token_id="t-12", label="12°C or below", lower_bound=None, upper_bound=12.0, unit="C", outcome_index=0),
        Bucket(bucket_id="b-13", market_id="m-13", token_id="t-13", label="13°C", lower_bound=13.0, upper_bound=13.0, unit="C", outcome_index=1),
        Bucket(bucket_id="b-14", market_id="m-14", token_id="t-14", label="14°C", lower_bound=14.0, upper_bound=14.0, unit="C", outcome_index=2),
        Bucket(bucket_id="b-15", market_id="m-15", token_id="t-15", label="15°C or higher", lower_bound=15.0, upper_bound=None, unit="C", outcome_index=3),
    ]
    event = WeatherEvent(
        event_id="evt-eglc-nov26",
        slug="eglc-nov26",
        title="Highest temp in London on Nov 26?",
        url="https://polymarket.com/event/eglc-nov26",
        station=station,
        local_date=date(2026, 11, 26),
        polarity=Polarity.HIGHEST,
        remark_tenths_enabled=True,
        unit="C",
        buckets=buckets,
        discovered_at_utc=t0 - timedelta(hours=12),
    )
    ensemble = EnsembleSnapshot(
        event_id=event.event_id,
        station_icao="EGLC",
        fetched_at_utc=t0,
        ingested_at_utc=t0,
        models=["ecmwf_ifs025"],
        member_count=60,
        hourly_timestamps_utc=timestamps,
        member_temps_c=trajectories,
    )
    metars = [
        MetarObservation(
            station_icao="EGLC",
            raw_report="METAR EGLC 261420Z 14/10 RMK T01410100",
            observed_at_utc=t0 + timedelta(hours=14, minutes=20),
            ingested_at_utc=t0 + timedelta(hours=14, minutes=25),
            temp_c=14.1,
            whole_temp_c=14,
            remark_tenths_c=14.1,
            has_remark_tenths=True,
            is_routine=True,
        ),
        MetarObservation(
            station_icao="EGLC",
            raw_report="METAR EGLC 261820Z 10/08 RMK T01020080",
            observed_at_utc=t0 + timedelta(hours=18, minutes=20),
            ingested_at_utc=t0 + timedelta(hours=18, minutes=25),
            temp_c=10.2,
            whole_temp_c=10,
            remark_tenths_c=10.2,
            has_remark_tenths=True,
            is_routine=True,
        ),
    ]

    engine = EnsembleProbabilityEngine()
    snap = engine.compute_probabilities(
        station=station,
        event=event,
        ensemble=ensemble,
        intraday_metars=metars,
        clock=VirtualClock(t0 + timedelta(hours=18, minutes=30)),
    )
    by_id = {b.bucket_id: b for b in snap.buckets}
    assert snap.probability_sum == pytest.approx(1.0, abs=1e-9)
    assert by_id["b-12"].is_impossible is True and by_id["b-12"].fair_probability == 0.0
    assert by_id["b-13"].is_impossible is True and by_id["b-13"].fair_probability == 0.0
    assert by_id["b-14"].fair_probability > 0.95
    assert by_id["b-15"].fair_probability < 0.05
