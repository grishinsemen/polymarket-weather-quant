"""Calibration evaluation bench for 3-Layer Ensemble & EMOS spread/bias calibration.

Computes strictly leak-free walk-forward Brier score, multi-class Log-Loss,
P10-P90 empirical interval coverage, tail Brier score (< 10c), bias, and MAE
per ICAO Station and overall, comparing raw Layer 1 vs calibrated EMOS Layer 2.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import numpy as np
from pydantic import BaseModel, Field

from weather_quant.archive.parquet_store import ParquetArchiveStore
from weather_quant.domain.clock import VirtualClock
from weather_quant.domain.models import (
    Bucket,
    EnsembleSnapshot,
    MetarObservation,
    Polarity,
    Station,
    WeatherEvent,
)
from weather_quant.probability.engine import (
    DEFAULT_MODEL_VERSION,
    DailyBiasObservation,
    EnsembleProbabilityEngine,
    bucket_distance,
    bucket_effective_bounds,
    round_half_away,
    to_event_unit,
)
from weather_quant.storage.database import Database

_LOG_EPS = 1e-6


class CalibrationDaySample(BaseModel):
    """Single verified station-day forecast vs Routine METAR daily extreme pair."""

    station: Station
    local_date: date
    polarity: Polarity = Polarity.HIGHEST
    remark_tenths_enabled: bool = True
    ensemble: EnsembleSnapshot
    actual_extreme_c: float
    observed_at_utc: datetime


class CalibrationMetricsSummary(BaseModel):
    """Probabilistic & interval calibration metrics over a set of verified station-days."""

    sample_count: int = 0
    brier_score: float = 0.0
    log_loss: float = 0.0
    coverage_p10_p90: float = 0.0
    tail_brier_score: float | None = None
    tail_bucket_count: int = 0
    mean_bias_c: float = 0.0
    mae_c: float = 0.0
    rmse_c: float = 0.0


class StationCalibrationReport(BaseModel):
    """Side-by-side raw vs EMOS-calibrated metrics for a single ICAO Station."""

    station_icao: str
    city: str
    sample_count: int
    raw: CalibrationMetricsSummary
    calibrated: CalibrationMetricsSummary
    learned_bias_c: float = 0.0
    learned_spread_scale: float = 1.0


class RegimeErrorSummary(BaseModel):
    """Error stratification for a temperature regime (COLD / MILD / HOT)."""

    regime: str
    sample_count: int
    mean_bias_c: float
    mae_c: float
    rmse_c: float
    brier_score: float


class MonthlyErrorSummary(BaseModel):
    """Calendar-month error breakdown for seasonal calibration diagnostics."""

    month: int
    sample_count: int
    mean_bias_c: float
    mae_c: float
    brier_score: float


class WorstDayDiagnostic(BaseModel):
    """Single worst-error day record for offline tail-failure inspection."""

    station_icao: str
    local_date: date
    actual_c: float
    predicted_p50_c: float
    residual_c: float
    abs_error_c: float
    brier_score: float


class CalibrationEvaluationReport(BaseModel):
    """Full per-station and aggregate calibration report."""

    generated_at_utc: datetime
    model_version: str = DEFAULT_MODEL_VERSION
    tail_price_threshold: float = 0.10
    overall_raw: CalibrationMetricsSummary
    overall_calibrated: CalibrationMetricsSummary
    stations: list[StationCalibrationReport] = Field(default_factory=list)
    regime_breakdown: list[RegimeErrorSummary] = Field(default_factory=list)
    monthly_breakdown: list[MonthlyErrorSummary] = Field(default_factory=list)
    worst_days: list[WorstDayDiagnostic] = Field(default_factory=list)
    pit_histogram: list[int] = Field(default_factory=list)


def build_synthetic_degree_buckets(
    center_c: int,
    *,
    event_id: str = "cal-evt",
    half_width: int = 5,
) -> list[Bucket]:
    """Build standard 11-Bucket 1°C temperature partition centered around *center_c*."""
    low_edge = center_c - half_width
    high_edge = center_c + half_width
    buckets: list[Bucket] = [
        Bucket(
            bucket_id=f"{event_id}-b0",
            market_id=f"{event_id}-m0",
            token_id=f"{event_id}-t0",
            label=f"{low_edge}°C or below",
            lower_bound=None,
            upper_bound=float(low_edge),
            unit="C",
            outcome_index=0,
        )
    ]
    for idx, deg in enumerate(range(low_edge + 1, high_edge), start=1):
        buckets.append(
            Bucket(
                bucket_id=f"{event_id}-b{idx}",
                market_id=f"{event_id}-m{idx}",
                token_id=f"{event_id}-t{idx}",
                label=f"{deg}°C",
                lower_bound=float(deg),
                upper_bound=float(deg),
                unit="C",
                outcome_index=idx,
            )
        )
    buckets.append(
        Bucket(
            bucket_id=f"{event_id}-b{2 * half_width}",
            market_id=f"{event_id}-m{2 * half_width}",
            token_id=f"{event_id}-t{2 * half_width}",
            label=f"{high_edge}°C or higher",
            lower_bound=float(high_edge),
            upper_bound=None,
            unit="C",
            outcome_index=2 * half_width,
        )
    )
    return buckets


def resolve_bucket_index(actual_unit: float, buckets: Sequence[Bucket]) -> int:
    """Resolve the winning Bucket index for a realized temperature in event units."""
    bounds = [bucket_effective_bounds(b) for b in buckets]
    is_integer = any(b[2] for b in bounds)
    test_val = float(round_half_away(actual_unit)) if is_integer else actual_unit
    for idx, (low, high, _) in enumerate(bounds):
        if low <= test_val < high:
            return idx
    return min(range(len(bounds)), key=lambda i: bucket_distance(test_val, bounds[i][0], bounds[i][1]))


def compute_multi_class_brier(
    probs_list: Sequence[Sequence[float]],
    true_indices: Sequence[int],
) -> float:
    """Compute mean multi-class Brier score ``mean_i sum_k (p_{i,k} - y_{i,k})^2``."""
    if not probs_list or len(probs_list) != len(true_indices):
        return 0.0
    total = 0.0
    for probs, y_idx in zip(probs_list, true_indices):
        for k, p in enumerate(probs):
            y = 1.0 if k == y_idx else 0.0
            total += (p - y) ** 2
    return total / len(probs_list)


def compute_multi_class_log_loss(
    probs_list: Sequence[Sequence[float]],
    true_indices: Sequence[int],
) -> float:
    """Compute mean multi-class Log-Loss ``-mean_i log(clip(p_{i, y_i}, eps, 1))``."""
    if not probs_list or len(probs_list) != len(true_indices):
        return 0.0
    total = 0.0
    for probs, y_idx in zip(probs_list, true_indices):
        p_true = probs[y_idx] if 0 <= y_idx < len(probs) else 0.0
        total -= math.log(max(_LOG_EPS, min(1.0, p_true)))
    return total / len(probs_list)


def compute_interval_coverage(
    actuals: Sequence[float],
    lowers: Sequence[float],
    uppers: Sequence[float],
) -> float:
    """Compute empirical coverage fraction ``P(lower <= actual <= upper)`` (target ~0.80 for P10-P90)."""
    if not actuals or not (len(actuals) == len(lowers) == len(uppers)):
        return 0.0
    hits = sum(1 for a, lo, hi in zip(actuals, lowers, uppers) if lo - 1e-9 <= a <= hi + 1e-9)
    return hits / len(actuals)


def _extract_day_window(
    ensemble: EnsembleSnapshot,
    station: Station,
    local_date: date,
    polarity: Polarity,
) -> tuple[list[int], list[list[float]], list[float]]:
    tz = ZoneInfo(station.timezone)
    day_indices = [
        idx
        for idx, ts in enumerate(ensemble.hourly_timestamps_utc)
        if (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)).astimezone(tz).date() == local_date
    ]
    if not day_indices:
        return [], [], []
    max_idx = max(day_indices)
    day_trajectories = [
        [float(traj[idx]) for idx in day_indices]
        for traj in ensemble.member_temps_c
        if len(traj) > max_idx
    ]
    raw_extremes_c = [
        max(traj) if polarity == Polarity.HIGHEST else min(traj)
        for traj in day_trajectories
    ]
    return day_indices, day_trajectories, raw_extremes_c


def _summarize_records(records: list[dict[str, float | int | list[float]]], tail_threshold: float) -> CalibrationMetricsSummary:
    if not records:
        return CalibrationMetricsSummary()
    probs_list = [r["probs"] for r in records]  # type: ignore[misc]
    true_indices = [int(r["true_idx"]) for r in records]
    actuals = [float(r["actual_c"]) for r in records]
    p10s = [float(r["p10_c"]) for r in records]
    p50s = [float(r["p50_c"]) for r in records]
    p90s = [float(r["p90_c"]) for r in records]

    errors = [a - m for a, m in zip(actuals, p50s)]
    tail_sq_errors: list[float] = []
    for probs, y_idx in zip(probs_list, true_indices):
        for k, p in enumerate(probs):  # type: ignore[arg-type]
            if 0.0 < float(p) < tail_threshold:
                y = 1.0 if k == y_idx else 0.0
                tail_sq_errors.append((float(p) - y) ** 2)

    return CalibrationMetricsSummary(
        sample_count=len(records),
        brier_score=round(compute_multi_class_brier(probs_list, true_indices), 6),  # type: ignore[arg-type]
        log_loss=round(compute_multi_class_log_loss(probs_list, true_indices), 6),  # type: ignore[arg-type]
        coverage_p10_p90=round(compute_interval_coverage(actuals, p10s, p90s), 6),
        tail_brier_score=round(sum(tail_sq_errors) / len(tail_sq_errors), 6) if tail_sq_errors else None,
        tail_bucket_count=len(tail_sq_errors),
        mean_bias_c=round(sum(errors) / len(errors), 4),
        mae_c=round(sum(abs(e) for e in errors) / len(errors), 4),
        rmse_c=round(math.sqrt(sum(e * e for e in errors) / len(errors)), 4),
    )


def evaluate_calibration_walk_forward(
    samples: Sequence[CalibrationDaySample],
    *,
    engine: EnsembleProbabilityEngine | None = None,
    mos_provider: Any | None = None,
    model_version: str = DEFAULT_MODEL_VERSION,
    tail_price_threshold: float = 0.10,
) -> CalibrationEvaluationReport:
    """Run strictly chronological walk-forward calibration evaluation per station and overall.

    For each day ``t``, EMOS bias & dispersion calibration uses strictly prior days ``< t``
    (zero future leakage). Compares uncalibrated Layer 1 (``scale=1.0, bias=0.0``) against
    the calibrated model (configured EMOS / Student-t / optional MOS provider).
    """
    cal_engine = engine or EnsembleProbabilityEngine()
    raw_engine = EnsembleProbabilityEngine(
        smoothing_alpha=cal_engine.smoothing_alpha,
        bias_window_days=cal_engine.bias_window_days,
        intraday_decay_hours=cal_engine.intraday_decay_hours,
        obs_noise_c=cal_engine.obs_noise_c,
        default_spread_inflation=1.0,
    )

    by_station: dict[str, list[CalibrationDaySample]] = {}
    for s in samples:
        by_station.setdefault(s.station.icao.strip().upper(), []).append(s)

    # Strict chronological day-by-day walk-forward across all stations so hierarchical
    # global_bias_obs only ever sees days strictly < item.local_date (zero leakage).
    station_history: dict[str, list[DailyBiasObservation]] = {icao: [] for icao in by_station}
    station_raw_records: dict[str, list[dict[str, float | int | list[float]]]] = {icao: [] for icao in by_station}
    station_cal_records: dict[str, list[dict[str, float | int | list[float]]]] = {icao: [] for icao in by_station}
    station_last_params: dict[str, tuple[float, float]] = {
        icao: (0.0, cal_engine.default_spread_inflation) for icao in by_station
    }
    global_history: list[DailyBiasObservation] = []
    all_raw_records: list[dict[str, float | int | list[float]]] = []
    all_cal_records: list[dict[str, float | int | list[float]]] = []

    by_date: dict[date, list[CalibrationDaySample]] = {}
    for s in samples:
        by_date.setdefault(s.local_date, []).append(s)

    for current_date in sorted(by_date):
        day_items = sorted(by_date[current_date], key=lambda x: x.station.icao.strip().upper())
        day_new_obs: list[tuple[str, DailyBiasObservation]] = []

        for item in day_items:
            icao = item.station.icao.strip().upper()
            day_indices, day_trajectories, raw_extremes_c = _extract_day_window(
                item.ensemble, item.station, item.local_date, item.polarity
            )
            if not raw_extremes_c:
                continue

            raw_arr = np.asarray(raw_extremes_c, dtype=np.float64)
            raw_mean = float(np.mean(raw_arr))
            center = round_half_away(raw_mean)
            buckets = build_synthetic_degree_buckets(center, event_id=f"cal-{icao}-{item.local_date.isoformat()}")
            station_tz = ZoneInfo(item.station.timezone)
            decision_time = datetime(
                item.local_date.year,
                item.local_date.month,
                item.local_date.day,
                0,
                0,
                tzinfo=station_tz,
            ).astimezone(timezone.utc)
            clock = VirtualClock(decision_time)
            event = WeatherEvent(
                event_id=f"cal-{icao}-{item.local_date.isoformat()}",
                slug=f"cal-{icao.lower()}-{item.local_date.isoformat()}",
                title=f"Calibration {icao} {item.local_date.isoformat()}",
                url="https://polymarket.com",
                station=item.station,
                local_date=item.local_date,
                polarity=item.polarity,
                remark_tenths_enabled=item.remark_tenths_enabled,
                unit="C",
                buckets=buckets,
                discovered_at_utc=decision_time - timedelta(hours=6),
            )

            true_idx = resolve_bucket_index(to_event_unit(item.actual_extreme_c, "C"), buckets)

            # Raw Layer 1 evaluation (zero bias, scale=1.0)
            raw_snap = raw_engine.compute_probabilities(
                station=item.station,
                event=event,
                ensemble=item.ensemble,
                recent_bias_obs=[],
                intraday_metars=[],
                clock=clock,
            )
            raw_rec = {
                "probs": [b.fair_probability for b in raw_snap.buckets],
                "true_idx": true_idx,
                "actual_c": item.actual_extreme_c,
                "p10_c": float(np.percentile(raw_arr, 10)),
                "p50_c": float(np.percentile(raw_arr, 50)),
                "p90_c": float(np.percentile(raw_arr, 90)),
            }
            station_raw_records[icao].append(raw_rec)

            # Calibrated evaluation (hierarchical walk-forward: strictly prior days only + configured model)
            history_bias = station_history[icao]
            layer2 = cal_engine.evaluate_premarket_layer2(
                station=item.station,
                event=event,
                ensemble=item.ensemble,
                day_indices=day_indices,
                day_trajectories=day_trajectories,
                raw_extremes_c=raw_extremes_c,
                decision_time=decision_time,
                recent_bias_obs=history_bias,
                global_bias_obs=global_history,
                strict_temporal=False,
            )
            station_last_params[icao] = (layer2.bias_c, layer2.scale)

            cal_probs = layer2.bias_probs
            p10_c = layer2.p10_c
            p50_c = layer2.p50_c
            p90_c = layer2.p90_c
            pit_val = layer2.pit(item.actual_extreme_c)

            if mos_provider is not None and mos_provider.is_station_covered(icao) and mos_provider.mos_model is not None:
                from weather_quant.ml.dataset_builder import extract_features_from_ensemble

                cal_snap = mos_provider.compute_probabilities(
                    station=item.station,
                    event=event,
                    ensemble=item.ensemble,
                    recent_bias_obs=history_bias,
                    intraday_metars=[],
                    clock=clock,
                    strict_temporal=False,
                )
                cal_probs = [b.fair_probability for b in cal_snap.buckets]
                features = extract_features_from_ensemble(
                    station=item.station,
                    local_date=event.local_date,
                    polarity=event.polarity,
                    ensemble=item.ensemble,
                    decision_time_utc=decision_time,
                    recent_bias_obs=history_bias,
                    prior_observations=(),
                )
                q_pred = mos_provider.mos_model.predict_quantiles([features])[0]
                mos_arr = np.asarray(mos_provider.mos_model.quantiles_to_extremes(q_pred), dtype=np.float64)
                p10_c, p50_c, p90_c = (float(v) for v in np.percentile(mos_arr, (10, 50, 90)))
                pit_val = float(np.mean(mos_arr <= item.actual_extreme_c))

            day_brier = sum(
                (p - (1.0 if k == true_idx else 0.0)) ** 2 for k, p in enumerate(cal_probs)
            )
            cal_rec = {
                "probs": cal_probs,
                "true_idx": true_idx,
                "actual_c": item.actual_extreme_c,
                "p10_c": p10_c,
                "p50_c": p50_c,
                "p90_c": p90_c,
                "station_icao": icao,
                "local_date": item.local_date,
                "pit": pit_val,
                "brier": day_brier,
            }
            station_cal_records[icao].append(cal_rec)

            day_new_obs.append(
                (
                    icao,
                    DailyBiasObservation(
                        local_date=item.local_date,
                        forecast_extreme_c=layer2.forecast_ref_c,
                        raw_forecast_extreme_c=layer2.raw_mean_c,
                        observed_extreme_c=item.actual_extreme_c,
                        forecast_std_c=(
                            layer2.forecast_ref_std_c if layer2.forecast_ref_std_c > 1e-6 else None
                        ),
                        observed_at_utc=item.observed_at_utc,
                        is_routine=True,
                    ),
                )
            )

        # Append today's verified observations ONLY AFTER scoring all stations for current_date
        for icao, bias_obs in day_new_obs:
            station_history[icao].append(bias_obs)
            global_history.append(bias_obs)

    station_reports: list[StationCalibrationReport] = []
    for icao in sorted(by_station):
        raw_records = station_raw_records[icao]
        cal_records = station_cal_records[icao]
        if raw_records:
            last_bias, last_scale = station_last_params[icao]
            station_reports.append(
                StationCalibrationReport(
                    station_icao=icao,
                    city=by_station[icao][0].station.city,
                    sample_count=len(raw_records),
                    raw=_summarize_records(raw_records, tail_price_threshold),
                    calibrated=_summarize_records(cal_records, tail_price_threshold),
                    learned_bias_c=round(last_bias, 4),
                    learned_spread_scale=round(last_scale, 4),
                )
            )
            all_raw_records.extend(raw_records)
            all_cal_records.extend(cal_records)

    regime_breakdown, monthly_breakdown, worst_days, pit_hist = _build_extended_diagnostics(all_cal_records)

    return CalibrationEvaluationReport(
        generated_at_utc=datetime.now(timezone.utc),
        model_version=model_version,
        tail_price_threshold=tail_price_threshold,
        overall_raw=_summarize_records(all_raw_records, tail_price_threshold),
        overall_calibrated=_summarize_records(all_cal_records, tail_price_threshold),
        stations=station_reports,
        regime_breakdown=regime_breakdown,
        monthly_breakdown=monthly_breakdown,
        worst_days=worst_days,
        pit_histogram=pit_hist,
    )


def _build_extended_diagnostics(
    records: list[dict[str, object]],
    *,
    cold_threshold_c: float = 10.0,
    hot_threshold_c: float = 25.0,
    top_n: int = 5,
    pit_bins: int = 10,
) -> tuple[list[RegimeErrorSummary], list[MonthlyErrorSummary], list[WorstDayDiagnostic], list[int]]:
    if not records:
        return [], [], [], []

    by_regime: dict[str, list[dict[str, object]]] = {"COLD": [], "MILD": [], "HOT": []}
    by_month: dict[int, list[dict[str, object]]] = {}
    pit_counts = [0] * pit_bins

    for rec in records:
        actual = float(rec["actual_c"])  # type: ignore[arg-type]
        if actual < cold_threshold_c:
            by_regime["COLD"].append(rec)
        elif actual > hot_threshold_c:
            by_regime["HOT"].append(rec)
        else:
            by_regime["MILD"].append(rec)

        d = rec.get("local_date")
        if isinstance(d, date):
            by_month.setdefault(d.month, []).append(rec)

        pit = float(rec.get("pit", 0.5))  # type: ignore[arg-type]
        b_idx = min(pit_bins - 1, max(0, int(pit * pit_bins)))
        pit_counts[b_idx] += 1

    regimes: list[RegimeErrorSummary] = []
    for name in ("COLD", "MILD", "HOT"):
        grp = by_regime[name]
        if not grp:
            continue
        errs = [float(r["actual_c"]) - float(r["p50_c"]) for r in grp]  # type: ignore[arg-type]
        briers = [float(r.get("brier", 0.0)) for r in grp]  # type: ignore[arg-type]
        regimes.append(
            RegimeErrorSummary(
                regime=name,
                sample_count=len(grp),
                mean_bias_c=round(sum(errs) / len(errs), 4),
                mae_c=round(sum(abs(e) for e in errs) / len(errs), 4),
                rmse_c=round(math.sqrt(sum(e * e for e in errs) / len(errs)), 4),
                brier_score=round(sum(briers) / len(briers), 6),
            )
        )

    months: list[MonthlyErrorSummary] = []
    for m in sorted(by_month):
        grp = by_month[m]
        errs = [float(r["actual_c"]) - float(r["p50_c"]) for r in grp]  # type: ignore[arg-type]
        briers = [float(r.get("brier", 0.0)) for r in grp]  # type: ignore[arg-type]
        months.append(
            MonthlyErrorSummary(
                month=m,
                sample_count=len(grp),
                mean_bias_c=round(sum(errs) / len(errs), 4),
                mae_c=round(sum(abs(e) for e in errs) / len(errs), 4),
                brier_score=round(sum(briers) / len(briers), 6),
            )
        )

    worst_sorted = sorted(
        records,
        key=lambda r: abs(float(r["actual_c"]) - float(r["p50_c"])),  # type: ignore[arg-type]
        reverse=True,
    )[:top_n]
    worst_days = [
        WorstDayDiagnostic(
            station_icao=str(r.get("station_icao", "")),
            local_date=r["local_date"],  # type: ignore[arg-type]
            actual_c=round(float(r["actual_c"]), 3),  # type: ignore[arg-type]
            predicted_p50_c=round(float(r["p50_c"]), 3),  # type: ignore[arg-type]
            residual_c=round(float(r["actual_c"]) - float(r["p50_c"]), 3),  # type: ignore[arg-type]
            abs_error_c=round(abs(float(r["actual_c"]) - float(r["p50_c"])), 3),  # type: ignore[arg-type]
            brier_score=round(float(r.get("brier", 0.0)), 6),  # type: ignore[arg-type]
        )
        for r in worst_sorted
        if isinstance(r.get("local_date"), date)
    ]
    return regimes, months, worst_days, pit_counts


async def load_local_calibration_samples(
    db: Database,
    archive_store: ParquetArchiveStore,
    *,
    lookback_days: int = 45,
    stations_filter: set[str] | None = None,
    min_metars_per_day: int = 6,
) -> list[CalibrationDaySample]:
    """Load verified station-day samples from local SQLite Routine METARs and Parquet ensembles."""
    from weather_quant.discovery.stations import get_station
    from weather_quant.probability.engine import list_station_routine_metars

    now_utc = datetime.now(timezone.utc)
    ensembles = await archive_store.aread_latest_ensemble_snapshots(
        start_utc=now_utc - timedelta(days=lookback_days),
        dedupe_by="station_day",
        keep="earliest",
    )
    by_icao: dict[str, list[EnsembleSnapshot]] = {}
    for snap in ensembles:
        icao = snap.station_icao.strip().upper()
        if stations_filter and icao not in stations_filter:
            continue
        by_icao.setdefault(icao, []).append(snap)

    samples: list[CalibrationDaySample] = []
    for icao, snaps in sorted(by_icao.items()):
        station = get_station(icao)
        if station is None:
            continue
        tz = ZoneInfo(station.timezone)
        metars = await list_station_routine_metars(db, icao)
        metars_by_day: dict[date, list[MetarObservation]] = {}
        for obs in metars:
            if not obs.is_routine or obs.raw_report.strip().upper().startswith("SPECI"):
                continue
            obs_utc = (
                obs.observed_at_utc
                if obs.observed_at_utc.tzinfo
                else obs.observed_at_utc.replace(tzinfo=timezone.utc)
            ).astimezone(timezone.utc)
            d = obs_utc.astimezone(tz).date()
            metars_by_day.setdefault(d, []).append(obs)

        snaps_sorted = sorted(
            snaps,
            key=lambda e: (
                e.ingested_at_utc
                if e.ingested_at_utc.tzinfo
                else e.ingested_at_utc.replace(tzinfo=timezone.utc)
            ),
        )
        today_local = now_utc.astimezone(tz).date()
        for d, day_obs in sorted(metars_by_day.items()):
            # Strictly exclude incomplete in-progress/future local days
            if d >= today_local:
                continue
            if len(day_obs) < min_metars_per_day:
                continue
            matched_snap = next(
                (
                    s
                    for s in snaps_sorted
                    if any(
                        (
                            ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
                        ).astimezone(tz).date()
                        == d
                        for ts in s.hourly_timestamps_utc
                    )
                ),
                None,
            )
            if matched_snap is None:
                continue
            temps = [EnsembleProbabilityEngine.obs_settlement_temp_c(m, True) for m in day_obs]
            last_utc = max(
                (
                    m.observed_at_utc
                    if m.observed_at_utc.tzinfo
                    else m.observed_at_utc.replace(tzinfo=timezone.utc)
                ).astimezone(timezone.utc)
                for m in day_obs
            )
            samples.append(
                CalibrationDaySample(
                    station=station,
                    local_date=d,
                    polarity=Polarity.HIGHEST,
                    remark_tenths_enabled=True,
                    ensemble=matched_snap,
                    actual_extreme_c=max(temps),
                    observed_at_utc=last_utc,
                )
            )
    return samples


async def load_calibration_target(
    db: Database,
    model_version: str | None = None,
) -> tuple[str, EnsembleProbabilityEngine, Any | None]:
    """Resolve active (or requested) ModelVersionRecord into a probability engine and optional MOS provider."""
    if model_version:
        versions = await db.list_model_versions()
        model_rec = next((m for m in versions if m.model_version == model_version), None)
        if model_rec is None:
            raise ValueError(f"Unknown model_version: {model_version!r}")
    else:
        try:
            model_rec = await db.get_active_model_version()
        except RuntimeError:
            return DEFAULT_MODEL_VERSION, EnsembleProbabilityEngine(), None

    base_engine = EnsembleProbabilityEngine.from_model_config(model_rec.config_json)
    from weather_quant.ml.quantile_mos import is_ml_mos_model, load_mos_provider_for_version

    mos_provider: Any | None = None
    if is_ml_mos_model(model_rec):
        mos_provider = await load_mos_provider_for_version(db, model_rec.model_version)
        if mos_provider is not None:
            mos_provider.baseline_engine = base_engine

    return model_rec.model_version, base_engine, mos_provider
