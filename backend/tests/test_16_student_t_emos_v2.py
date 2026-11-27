"""Unit tests for analytical Student-t EMOS v2, EWMA/t-MLE calibration, and warm-start merge."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

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
from weather_quant.probability.calibration import build_synthetic_degree_buckets
from weather_quant.probability.engine import (
    DailyBiasObservation,
    EnsembleProbabilityEngine,
    bucket_effective_bounds_c,
)
from weather_quant.probability.student_t_emos import (
    DEFAULT_REGIONAL_DEBIASED_HET_VAR,
    DEFAULT_REGIONAL_NWP_WEIGHTS,
    map_student_t_cdf_to_probabilities,
    solve_weighted_t_mle_sigma_mult,
)


def test_student_t_cdf_normalizes_zeros_impossible_and_has_heavier_tails_than_gaussian() -> None:
    buckets = build_synthetic_degree_buckets(12, event_id="evt-t-cdf", half_width=5)
    bounds_c = [bucket_effective_bounds_c(b, remark_tenths_enabled=True) for b in buckets]
    impossible = [i < 2 for i in range(len(buckets))]

    probs_t5 = map_student_t_cdf_to_probabilities(
        mu_c=12.0, sigma_c=1.2, nu=5.0, bounds_c=bounds_c, impossible_mask=impossible
    )
    probs_gauss_like = map_student_t_cdf_to_probabilities(
        mu_c=12.0, sigma_c=1.2, nu=200.0, bounds_c=bounds_c, impossible_mask=impossible
    )

    assert sum(probs_t5) == pytest.approx(1.0, abs=1e-9)
    assert probs_t5[0] == 0.0 and probs_t5[1] == 0.0
    # Extreme upper tail bucket (17°C or higher, >4 sigma away) gets strictly more mass under Student-t(5)
    assert probs_t5[-1] > probs_gauss_like[-1]


def test_weighted_t_mle_sigma_mult_is_outlier_resistant() -> None:
    # 19 clean unit-variance residuals + 1 massive 10-sigma storm outlier
    clean_z = [(-1.0 if i % 2 == 0 else 1.0) * 0.9 for i in range(19)]
    with_outlier = [*clean_z, 10.0]
    weights = [1.0] * len(with_outlier)

    t_mult = solve_weighted_t_mle_sigma_mult(with_outlier, weights, nu=7.0)
    raw_rms = (sum(z * z for z in with_outlier) / len(with_outlier)) ** 0.5
    # Student-t MLE downweights the 10-sigma outlier dramatically compared to raw RMS (~2.40)
    assert t_mult < 1.35
    assert raw_rms > 2.2


def test_engine_from_model_config_preserves_v1_and_enables_student_t_v2_with_warmstart(
    tmp_path: Path,
) -> None:
    # 1) Legacy v1 config stays histogram + use_censored_max=False
    v1 = EnsembleProbabilityEngine.from_model_config({"bias_days": 30})
    assert v1.distribution_mode == "histogram"
    assert v1.use_censored_max is False
    assert v1.bias_window_days == 30

    # 2) Write a strictly pre-OOS warm-start file with +2.0C warm bias for LFPB
    warm_path = tmp_path / "warmstart.json"
    warm_entries = [
        {
            "local_date": (date(2026, 11, 24) - timedelta(days=i)).isoformat(),
            "forecast_extreme_c": 12.0,
            "observed_extreme_c": 14.0,
            "forecast_std_c": 1.0,
            "observed_at_utc": (datetime(2026, 11, 24, 20, 0, tzinfo=timezone.utc) - timedelta(days=i)).isoformat(),
            "is_routine": True,
        }
        for i in range(15)
    ]
    warm_path.write_text(json.dumps({"LFPB_HIGHEST": warm_entries}), encoding="utf-8")

    v2 = EnsembleProbabilityEngine.from_model_config(
        {
            "distribution_mode": "student_t_cdf",
            "student_t_df": 7.0,
            "bias_window_days": 60,
            "bias_half_life_days": 14.0,
            "scale_half_life_days": 21.0,
            "use_regional_nwp_weights": True,
            "use_censored_max": True,
            "warmstart_bias_path": str(warm_path),
            "regional_nwp_weights": {k: list(v) for k, v in DEFAULT_REGIONAL_NWP_WEIGHTS.items()},
            "regional_het_var": {k: list(v) for k, v in DEFAULT_REGIONAL_DEBIASED_HET_VAR.items()},
        }
    )
    assert v2.distribution_mode == "student_t_cdf"
    assert v2.use_censored_max is True

    station = get_station("LFPB")
    assert station is not None
    local_d = date(2026, 11, 26)
    decision_t = datetime(2026, 11, 25, 18, 0, tzinfo=timezone.utc)
    buckets = build_synthetic_degree_buckets(13, event_id="evt-lfpb-v2", half_width=5)
    event = WeatherEvent(
        event_id="evt-lfpb-v2",
        slug="highest-temp-paris-2026-11-26",
        title="Highest temp Paris",
        url="https://polymarket.com/event/evt-lfpb-v2",
        station=station,
        local_date=local_d,
        polarity=Polarity.HIGHEST,
        remark_tenths_enabled=True,
        unit="C",
        buckets=buckets,
        discovered_at_utc=decision_t - timedelta(hours=2),
    )

    base_utc = datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc)
    timestamps = [base_utc + timedelta(hours=h) for h in range(24)]
    diurnal = [8.0 + 4.0 * max(0.0, 1.0 - abs(h - 14) / 6.0) for h in range(24)]
    ensemble = EnsembleSnapshot(
        event_id=event.event_id,
        station_icao="LFPB",
        fetched_at_utc=decision_t - timedelta(minutes=10),
        ingested_at_utc=decision_t - timedelta(minutes=5),
        models=["ecmwf_ifs025", "gfs_seamless", "icon_seamless"],
        member_count=10,
        hourly_timestamps_utc=timestamps,
        member_temps_c=[[t + (m - 4.5) * 0.15 for t in diurnal] for m in range(10)],
        metadata={
            "model_hourly_means_c": {
                "ecmwf_ifs025": diurnal,
                "gfs_seamless": [t - 0.5 for t in diurnal],
                "icon_seamless": [t + 0.2 for t in diurnal],
            }
        },
    )

    snap_pre = v2.compute_probabilities(
        station=station,
        event=event,
        ensemble=ensemble,
        recent_bias_obs=[],
        intraday_metars=[],
        clock=VirtualClock(decision_t),
        model_version="emos-student-t-v2",
        strict_temporal=True,
    )
    assert snap_pre.probability_sum == pytest.approx(1.0, abs=1e-6)
    # Warm-start +2.0C bias shifts mode from ~12°C up to ~13-14°C even with empty live recent_bias_obs
    prob_by_label = {b.label: b.fair_probability for b in snap_pre.buckets}
    assert prob_by_label["14°C"] > prob_by_label["11°C"]

    # Intraday METAR at 14.2°C zeroes <=13°C buckets via Censored-Max Layer 3
    obs_t = datetime(2026, 11, 26, 13, 0, tzinfo=timezone.utc)
    metar = MetarObservation(
        station_icao="LFPB",
        raw_report="METAR LFPB 261300Z 24010KT 9999 SCT030 14/06 Q1016 RMK T01420060",
        observed_at_utc=obs_t,
        ingested_at_utc=obs_t + timedelta(minutes=2),
        temp_c=14.2,
        whole_temp_c=14,
        remark_tenths_c=14.2,
        has_remark_tenths=True,
        is_routine=True,
    )
    snap_intra = v2.compute_probabilities(
        station=station,
        event=event,
        ensemble=ensemble,
        recent_bias_obs=[],
        intraday_metars=[metar],
        clock=VirtualClock(obs_t + timedelta(minutes=5)),
        model_version="emos-student-t-v2",
        strict_temporal=True,
    )
    assert snap_intra.probability_sum == pytest.approx(1.0, abs=1e-6)
    intra_by_label = {b.label: b for b in snap_intra.buckets}
    assert intra_by_label["13°C"].is_impossible is True
    assert intra_by_label["13°C"].fair_probability == 0.0
    assert intra_by_label["14°C"].fair_probability > 0.25


def test_rkpk_has_dwd_mosmix_id_and_ettt_scales_anchor_weight() -> None:
    from weather_quant.feeds.mosmix import ICAO_TO_DWD_STATION_ID

    assert ICAO_TO_DWD_STATION_ID.get("RKPK") == "47153"

    engine = EnsembleProbabilityEngine(mos_anchor_weight=0.30)
    t0 = datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc)
    ens_low_unc = EnsembleSnapshot(
        event_id="evt-ettt",
        station_icao="EDDM",
        fetched_at_utc=t0 - timedelta(hours=6),
        ingested_at_utc=t0 - timedelta(hours=6),
        models=["ecmwf_ifs025"],
        member_count=1,
        hourly_timestamps_utc=[t0],
        member_temps_c=[[10.0]],
        metadata={"mosmix_high_c": 12.0, "mosmix_uncertainty_c": 0.50},
    )
    ens_high_unc = ens_low_unc.model_copy(
        update={"metadata": {"mosmix_high_c": 12.0, "mosmix_uncertainty_c": 1.60}}
    )

    shift_low_unc = engine._compute_external_mos_anchor_shift(
        ensemble=ens_low_unc, polarity=Polarity.HIGHEST, emos_mean_c=10.0, emos_std_c=0.90
    )
    shift_high_unc = engine._compute_external_mos_anchor_shift(
        ensemble=ens_high_unc, polarity=Polarity.HIGHEST, emos_mean_c=10.0, emos_std_c=0.90
    )
    # Low E_TTT (0.50C) pulls ~0.45 * 2C = +0.90C; high E_TTT (1.60C) pulls ~0.14 * 2C = +0.28C
    assert shift_low_unc > 0.80
    assert shift_high_unc < 0.35
    assert shift_low_unc > shift_high_unc * 2.0


def test_calibration_walk_forward_uses_student_t_engine_and_warmstart(tmp_path: Path) -> None:
    from weather_quant.probability.calibration import (
        CalibrationDaySample,
        evaluate_calibration_walk_forward,
    )

    warm_path = tmp_path / "warmstart_audit.json"
    warm_entries = [
        {
            "local_date": (date(2026, 11, 24) - timedelta(days=i)).isoformat(),
            "forecast_extreme_c": 10.0,
            "observed_extreme_c": 12.2,
            "forecast_std_c": 1.0,
            "observed_at_utc": (
                datetime(2026, 11, 24, 20, 0, tzinfo=timezone.utc) - timedelta(days=i)
            ).isoformat(),
            "is_routine": True,
        }
        for i in range(15)
    ]
    warm_path.write_text(json.dumps({"LFPB_HIGHEST": warm_entries}), encoding="utf-8")

    station = get_station("LFPB")
    assert station is not None
    t_day = date(2026, 11, 26)
    ts = [datetime(t_day.year, t_day.month, t_day.day, h, 0, tzinfo=timezone.utc) for h in range(24)]
    # 10 narrow members around 10.0C; realized temperature matches warm-start bias at 12.2C
    ens = EnsembleSnapshot(
        event_id="cal-lfpb-t",
        station_icao="LFPB",
        fetched_at_utc=datetime(2026, 11, 25, 18, 0, tzinfo=timezone.utc),
        ingested_at_utc=datetime(2026, 11, 25, 18, 0, tzinfo=timezone.utc),
        models=["ecmwf_ifs025", "gfs_seamless", "icon_seamless"],
        member_count=10,
        hourly_timestamps_utc=ts,
        member_temps_c=[[10.0 + (m - 4.5) * 0.1 for _ in range(24)] for m in range(10)],
    )
    samples = [
        CalibrationDaySample(
            station=station,
            local_date=t_day,
            polarity=Polarity.HIGHEST,
            remark_tenths_enabled=True,
            ensemble=ens,
            actual_extreme_c=12.2,
            observed_at_utc=datetime(2026, 11, 26, 15, 0, tzinfo=timezone.utc),
        )
    ]

    v1_report = evaluate_calibration_walk_forward(samples, model_version="ensemble-3layer-v1")
    v2_engine = EnsembleProbabilityEngine.from_model_config(
        {
            "distribution_mode": "student_t_cdf",
            "student_t_df": 7.0,
            "bias_window_days": 60,
            "bias_half_life_days": 14.0,
            "scale_half_life_days": 21.0,
            "use_regional_nwp_weights": False,
            "warmstart_bias_path": str(warm_path),
            "regional_het_var": {k: list(v) for k, v in DEFAULT_REGIONAL_DEBIASED_HET_VAR.items()},
        }
    )
    v2_report = evaluate_calibration_walk_forward(
        samples,
        engine=v2_engine,
        model_version="emos-student-t-v2",
    )

    assert v1_report.model_version == "ensemble-3layer-v1"
    assert v2_report.model_version == "emos-student-t-v2"
    # Cold-start v1 misses 12.2C (~+2.2C error); warm-started Student-t v2 shifts close to 12.2C and captures P10-P90
    assert v2_report.overall_calibrated.mae_c < v1_report.overall_calibrated.mae_c
    assert v2_report.overall_calibrated.log_loss < v1_report.overall_calibrated.log_loss
    assert v2_report.overall_calibrated.coverage_p10_p90 == 1.0
    assert v2_report.stations[0].learned_bias_c > 1.0
