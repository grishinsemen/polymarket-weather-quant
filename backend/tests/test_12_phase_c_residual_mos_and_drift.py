"""TDD tests for Phase C: ResidualMosModel extrapolation, Temperature Scaling, regime diagnostics & KS/CUSUM drift."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from weather_quant.discovery.stations import get_station
from weather_quant.domain.models import EnsembleSnapshot, Polarity
from weather_quant.ml.dataset_builder import FEATURE_COLUMNS, MosDatasetSample
from weather_quant.ml.residual_mos import (
    ResidualMosModel,
    apply_temperature_scaling,
    evaluate_station_overfit_gate,
    fit_temperature_scaling,
)
from weather_quant.probability.calibration import CalibrationDaySample, evaluate_calibration_walk_forward
from weather_quant.probability.drift import evaluate_residual_drift


def _synthetic_sample(day_idx: int, ens_mean: float, target: float) -> MosDatasetSample:
    d = date(2026, 6, 1) + timedelta(days=day_idx)
    dt = datetime(d.year, d.month, d.day, 0, 0, tzinfo=timezone.utc)
    feats = {col: 0.0 for col in FEATURE_COLUMNS}
    feats.update(
        {
            "ecmwf_extreme_c": ens_mean,
            "gfs_extreme_c": ens_mean,
            "icon_extreme_c": ens_mean,
            "ensemble_mean_c": ens_mean,
            "ensemble_std_c": 1.0,
            "ensemble_min_c": ens_mean - 1.5,
            "ensemble_max_c": ens_mean + 1.5,
            "rolling_bias_c": 0.0,
            "prior_obs_temp_c": ens_mean,
            "mosmix_extreme_c": ens_mean,
            "nbm_extreme_c": ens_mean,
            "nws_point_extreme_c": ens_mean,
        }
    )
    return MosDatasetSample(
        station_icao="LFPB",
        local_date=d,
        polarity=Polarity.HIGHEST,
        decision_time_utc=dt - timedelta(hours=6),
        forecast_available_at_utc=dt - timedelta(hours=12),
        features=feats,
        baseline_mean_c=ens_mean,
        baseline_std_c=1.0,
        baseline_extremes_c=[ens_mean],
        target_temp_c=target,
        target_observed_at_utc=dt + timedelta(hours=14),
    )


def test_residual_mos_extrapolates_into_unseen_record_heatwave() -> None:
    """Trained on 15-25C days with +1.0C station bias, ResidualMosModel extrapolates cleanly to 42C."""
    train = [
        _synthetic_sample(i, ens_mean=15.0 + (i % 11), target=15.0 + (i % 11) + 1.0)
        for i in range(60)
    ]
    model = ResidualMosModel.fit(train, model_version="mos-residual-v2-test", max_residual_delta_c=3.0)

    heatwave_feat = dict(train[0].features)
    heatwave_feat["ensemble_mean_c"] = 42.0
    heatwave_feat["ecmwf_extreme_c"] = 42.0
    heatwave_feat["gfs_extreme_c"] = 42.0
    heatwave_feat["icon_extreme_c"] = 42.0

    q_pred = model.predict_quantiles([heatwave_feat])[0]
    # Pure tree MOS would plateau <= 26C; ResidualMosModel extrapolates via EMOS base to ~43C!
    assert 41.5 <= q_pred[0.5] <= 45.0
    assert q_pred[0.1] <= q_pred[0.5] <= q_pred[0.9]


def test_temperature_scaling_preserves_simplex_and_improves_overconfident_probs() -> None:
    overconfident = [[0.92, 0.05, 0.03] for _ in range(10)]
    outcomes = [0, 0, 0, 0, 0, 0, 1, 1, 2, 0]  # 70% hit rate, not 92%
    tau = fit_temperature_scaling(overconfident, outcomes)
    assert tau > 1.0

    calibrated = apply_temperature_scaling([0.92, 0.05, 0.03], tau, impossible_mask=[False, False, True])
    assert sum(calibrated) == pytest.approx(1.0, abs=1e-9)
    assert calibrated[2] == 0.0
    assert calibrated[0] < 0.92


def test_overfit_gate_rejects_degraded_or_high_gap_mos() -> None:
    assert evaluate_station_overfit_gate(
        mos_val_brier=0.08, base_val_brier=0.10, mos_test_brier=0.085, base_test_brier=0.10,
        mos_train_pinball=0.25, mos_val_pinball=0.35,
    ) is True
    assert evaluate_station_overfit_gate(
        mos_val_brier=0.12, base_val_brier=0.10, mos_test_brier=0.09, base_test_brier=0.10,
    ) is False
    assert evaluate_station_overfit_gate(
        mos_val_brier=0.08, base_val_brier=0.10, mos_test_brier=0.085, base_test_brier=0.10,
        mos_train_pinball=0.10, mos_val_pinball=0.30, max_train_val_ratio=2.0,
    ) is False


def test_ks_and_cusum_drift_detector_flags_sensor_shift_and_ignores_stable_residuals() -> None:
    stable = [0.1, -0.2, 0.0, 0.15, -0.1, 0.05] * 8
    ok_report = evaluate_residual_drift("EGLC", stable, recent_window=14)
    assert ok_report.alert is False
    assert ok_report.recommended_sigma_multiplier == pytest.approx(1.0)

    shifted = [0.0, 0.1, -0.1, 0.05] * 8 + [1.6, 1.8, 1.5, 1.9, 1.7, 1.6, 2.0, 1.5, 1.8, 1.7, 1.6, 1.9, 1.7, 1.8]
    alert_report = evaluate_residual_drift("EGLC", shifted, recent_window=14)
    assert alert_report.alert is True
    assert alert_report.ks_triggered is True
    assert alert_report.cusum_triggered is True
    assert alert_report.mean_shift_c > 1.0
    assert alert_report.recommended_sigma_multiplier == pytest.approx(1.20)


def test_calibration_report_includes_regime_monthly_worst_days_and_pit() -> None:
    station = get_station("LFPB")
    assert station is not None
    samples: list[CalibrationDaySample] = []
    for idx, actual_c in enumerate([5.0, 6.0, 16.0, 18.0, 28.0, 30.0]):
        d = date(2026, 1 + idx, 15)
        t0 = datetime(d.year, d.month, d.day, 0, 0, tzinfo=timezone.utc)
        ts = [t0 + timedelta(hours=h) for h in range(24)]
        # Make day idx=4 have a large 4.0C error so it ranks #1 in worst_days
        ens_base = actual_c - (4.0 if idx == 4 else 0.2)
        ens = EnsembleSnapshot(
            event_id=f"cal-{idx}",
            station_icao="LFPB",
            fetched_at_utc=t0 - timedelta(hours=12),
            ingested_at_utc=t0 - timedelta(hours=12),
            models=["ecmwf_ifs025"],
            member_count=10,
            hourly_timestamps_utc=ts,
            member_temps_c=[[ens_base + (m - 5) * 0.2 for _ in range(24)] for m in range(10)],
        )
        samples.append(
            CalibrationDaySample(
                station=station,
                local_date=d,
                ensemble=ens,
                actual_extreme_c=actual_c,
                observed_at_utc=t0 + timedelta(hours=14),
            )
        )

    report = evaluate_calibration_walk_forward(samples)
    regime_names = {r.regime for r in report.regime_breakdown}
    assert regime_names == {"COLD", "MILD", "HOT"}
    assert len(report.monthly_breakdown) == 6
    assert len(report.worst_days) == 5
    assert report.worst_days[0].local_date == date(2026, 5, 15)
    assert sum(report.pit_histogram) == 6


def test_legacy_23_feature_mos_model_predicts_against_29_feature_dict() -> None:
    """Saved v1 23-feature LightGBM models must score 29-feature live dicts without shape errors."""
    import lightgbm as lgb
    import numpy as np
    from weather_quant.ml.quantile_mos import QuantileMosModel

    legacy_cols = FEATURE_COLUMNS[:23]
    x_legacy = np.zeros((10, 23), dtype=np.float64)
    y_legacy = np.linspace(10.0, 20.0, 10)
    ds = lgb.Dataset(x_legacy, label=y_legacy, feature_name=list(legacy_cols), free_raw_data=False)
    booster = lgb.train({"objective": "quantile", "alpha": 0.5, "verbose": -1, "min_data_in_leaf": 2}, ds, num_boost_round=5)

    model = QuantileMosModel(
        model_version="mos-legacy-23f",
        quantiles=(0.5,),
        boosters={0.5: booster},
        feature_columns=legacy_cols,
    )
    full_29_feat = {col: 1.0 for col in FEATURE_COLUMNS}
    preds = model.predict_quantiles([full_29_feat])
    assert len(preds) == 1
    assert 0.5 in preds[0]
