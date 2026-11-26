"""Calibration evaluation bench for 3-Layer Ensemble & EMOS spread/bias calibration.

Computes strictly leak-free walk-forward Brier score, multi-class Log-Loss,
P10-P90 empirical interval coverage, tail Brier score (< 10c), bias, and MAE
per ICAO Station and overall, comparing raw Layer 1 vs calibrated EMOS Layer 2.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from typing import Sequence
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


class CalibrationEvaluationReport(BaseModel):
    """Full per-station and aggregate calibration report."""

    generated_at_utc: datetime
    tail_price_threshold: float = 0.10
    overall_raw: CalibrationMetricsSummary
    overall_calibrated: CalibrationMetricsSummary
    stations: list[StationCalibrationReport] = Field(default_factory=list)


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


def _extract_day_extremes_c(
    ensemble: EnsembleSnapshot,
    station: Station,
    local_date: date,
    polarity: Polarity,
) -> list[float]:
    tz = ZoneInfo(station.timezone)
    day_indices = [
        idx
        for idx, ts in enumerate(ensemble.hourly_timestamps_utc)
        if (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)).astimezone(tz).date() == local_date
    ]
    if not day_indices:
        return []
    return [
        (
            max(float(traj[i]) for i in day_indices)
            if polarity == Polarity.HIGHEST
            else min(float(traj[i]) for i in day_indices)
        )
        for traj in ensemble.member_temps_c
        if len(traj) > max(day_indices)
    ]


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
    tail_price_threshold: float = 0.10,
) -> CalibrationEvaluationReport:
    """Run strictly chronological walk-forward calibration evaluation per station and overall.

    For each day ``t``, EMOS bias & dispersion calibration uses strictly prior days ``< t``
    (zero future leakage). Compares uncalibrated Layer 1 (``scale=1.0, bias=0.0``) against
    calibrated Layer 2 (EMOS station bias + spread calibration).
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

    station_reports: list[StationCalibrationReport] = []
    all_raw_records: list[dict[str, float | int | list[float]]] = []
    all_cal_records: list[dict[str, float | int | list[float]]] = []

    for icao in sorted(by_station):
        ordered = sorted(by_station[icao], key=lambda x: x.local_date)
        history_bias: list[DailyBiasObservation] = []
        raw_records: list[dict[str, float | int | list[float]]] = []
        cal_records: list[dict[str, float | int | list[float]]] = []
        last_bias = 0.0
        last_scale = cal_engine.default_spread_inflation

        for item in ordered:
            raw_extremes_c = _extract_day_extremes_c(
                item.ensemble, item.station, item.local_date, item.polarity
            )
            if not raw_extremes_c:
                continue

            raw_arr = np.asarray(raw_extremes_c, dtype=np.float64)
            raw_mean = float(np.mean(raw_arr))
            raw_std = float(np.std(raw_arr))
            center = round_half_away(raw_mean)
            buckets = build_synthetic_degree_buckets(center, event_id=f"cal-{icao}-{item.local_date.isoformat()}")
            decision_time = datetime(
                item.local_date.year,
                item.local_date.month,
                item.local_date.day,
                0,
                0,
                tzinfo=timezone.utc,
            )
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
            raw_records.append(
                {
                    "probs": [b.fair_probability for b in raw_snap.buckets],
                    "true_idx": true_idx,
                    "actual_c": item.actual_extreme_c,
                    "p10_c": float(np.percentile(raw_arr, 10)),
                    "p50_c": float(np.percentile(raw_arr, 50)),
                    "p90_c": float(np.percentile(raw_arr, 90)),
                }
            )

            # Calibrated Layer 2 evaluation (walk-forward strictly prior days only)
            bias_c, scale = cal_engine.compute_station_bias(
                recent_bias_obs=history_bias,
                event_local_date=item.local_date,
                decision_time=decision_time,
                raw_extremes_c=raw_extremes_c,
                strict_temporal=False,
            )
            last_bias, last_scale = bias_c, scale
            cal_snap = cal_engine.compute_probabilities(
                station=item.station,
                event=event,
                ensemble=item.ensemble,
                recent_bias_obs=history_bias,
                intraday_metars=[],
                clock=clock,
            )
            cal_arr = raw_mean + bias_c + scale * (raw_arr - raw_mean)
            cal_records.append(
                {
                    "probs": [b.fair_probability for b in cal_snap.buckets],
                    "true_idx": true_idx,
                    "actual_c": item.actual_extreme_c,
                    "p10_c": float(np.percentile(cal_arr, 10)),
                    "p50_c": float(np.percentile(cal_arr, 50)),
                    "p90_c": float(np.percentile(cal_arr, 90)),
                }
            )

            # Append today's verified observation ONLY AFTER scoring today (leak-free walk-forward)
            history_bias.append(
                DailyBiasObservation(
                    local_date=item.local_date,
                    forecast_extreme_c=raw_mean,
                    observed_extreme_c=item.actual_extreme_c,
                    forecast_std_c=raw_std if raw_std > 1e-6 else None,
                    observed_at_utc=item.observed_at_utc,
                    is_routine=True,
                )
            )

        if raw_records:
            station_reports.append(
                StationCalibrationReport(
                    station_icao=icao,
                    city=ordered[0].station.city,
                    sample_count=len(raw_records),
                    raw=_summarize_records(raw_records, tail_price_threshold),
                    calibrated=_summarize_records(cal_records, tail_price_threshold),
                    learned_bias_c=round(last_bias, 4),
                    learned_spread_scale=round(last_scale, 4),
                )
            )
            all_raw_records.extend(raw_records)
            all_cal_records.extend(cal_records)

    return CalibrationEvaluationReport(
        generated_at_utc=datetime.now(timezone.utc),
        tail_price_threshold=tail_price_threshold,
        overall_raw=_summarize_records(all_raw_records, tail_price_threshold),
        overall_calibrated=_summarize_records(all_cal_records, tail_price_threshold),
        stations=station_reports,
    )


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
